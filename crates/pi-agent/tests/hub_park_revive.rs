//! Behavioral park/revive parity: keep-alive adoption, idle TTL parking, hub
//! roster visibility, JSONL-replayed revival, restart restoration, `history://`,
//! and tombstone refusal.

use std::{
	path::PathBuf,
	sync::{
		Arc, Mutex,
		atomic::{AtomicUsize, Ordering},
	},
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use pi_agent::{
	CancellationToken,
	provider::{ChunkStream, ModelProvider, ProviderError, ToolChoice},
	state::{AgentContext, Message},
	subagent_lifecycle::{
		DEFAULT_AGENT_IDLE_TTL_MS, DeliveryOutcome, LifecycleStatus, attach,
	},
	supervisor::{ChildSpawn, DelegationLimits, Supervisor},
	task_persisted_revive::{
		DefaultPersistedReviverFactory, PersistedChildSession, load_transcript,
	},
	tools::{
		Tool,
		hub::{HubTool, HubToolDetails},
		hub_park,
		internal_urls::{InternalUrlRouter, UrlScope},
		registry::ToolRegistry,
	},
};
use serde_json::json;
use tokio::sync::{Notify, Semaphore};

struct GatedProvider {
	reply:    String,
	captured: Arc<Mutex<Vec<Message>>>,
	entered:  Arc<Notify>,
	gate:     Arc<Semaphore>,
	calls:    AtomicUsize,
}

#[async_trait]
impl ModelProvider for GatedProvider {
	async fn complete(
		&self,
		messages: &[Message],
		_tools: &[serde_json::Value],
		_tool_choice: &ToolChoice,
	) -> Result<Message, ProviderError> {
		self.captured
			.lock()
			.expect("captured")
			.extend_from_slice(messages);
		self.calls.fetch_add(1, Ordering::SeqCst);
		self.entered.notify_one();
		let _permit = self
			.gate
			.acquire()
			.await
			.map_err(|_| ProviderError::Protocol("gate closed".to_string()))?;
		Ok(Message::assistant(self.reply.clone()))
	}

	async fn stream(
		&self,
		_messages: &[Message],
		_tools: &[serde_json::Value],
		_tool_choice: &ToolChoice,
	) -> Result<ChunkStream, ProviderError> {
		Err(ProviderError::Protocol("revived runs use complete".to_string()))
	}
}

fn gated_provider(reply: &str) -> Arc<GatedProvider> {
	Arc::new(GatedProvider {
		reply:    reply.to_string(),
		captured: Arc::new(Mutex::new(Vec::new())),
		entered:  Arc::new(Notify::new()),
		gate:     Arc::new(Semaphore::new(0)),
		calls:    AtomicUsize::new(0),
	})
}

fn session_root(label: &str) -> PathBuf {
	let nanos = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map(|duration| duration.as_nanos())
		.unwrap_or_default();
	let dir = std::env::temp_dir().join(format!(
		"omp-hub-park-{label}-{}-{nanos}",
		std::process::id()
	));
	std::fs::create_dir_all(&dir).expect("session root");
	dir
}

fn test_supervisor(root: &PathBuf, provider: Arc<GatedProvider>) -> Arc<Supervisor> {
	let supervisor = Supervisor::new(
		ToolRegistry::with_builtins(),
		provider,
		None,
		DelegationLimits::default(),
	);
	supervisor.set_child_session_root(Some(root.clone()));
	supervisor
}

fn adoptable_child(supervisor: &Arc<Supervisor>, root: &PathBuf) -> (String, PersistedChildSession) {
	let (id, _token) = supervisor
		.reserve_child_full(ChildSpawn {
			name:                "kept",
			index:               1,
			agent:               "scout",
			agent_source:        "builtin",
			task:                "investigate the repo",
			parent_id:           None,
			parent_tool_call_id: Some("call-park"),
			detached:            true,
		})
		.expect("reserve child");
	supervisor.append_child_transcript(
		&id,
		&[
			&Message::user("Complete assignment thoroughly:\n\ninvestigate the repo"),
			&Message::assistant("prior-answer"),
		],
	);
	let contract = PersistedChildSession {
		id: id.clone(),
		agent: Some("scout".to_string()),
		task: "investigate the repo".to_string(),
		system_prompt: "child system prompt".to_string(),
		tools: vec!["read".to_string(), "yield".to_string()],
		model_role: Some("smol".to_string()),
		resolved_model: Some("test/kept".to_string()),
		output_schema: None,
		output_schema_mode: None,
		restrict_tool_names: Some(true),
		spawns: Some("*".to_string()),
		read_summarize: None,
		advisor: None,
		read_only: None,
		cwd: root.clone(),
		max_turns: 20,
		context_limit: 100_000,
		display_name: Some("kept".to_string()),
		parent_id: None,
	};
	contract.write(root).expect("write contract");
	supervisor.mark_completed(&id, 7, "prior-answer");
	(id, contract)
}

fn reviver_factory(provider: Arc<GatedProvider>) -> Arc<DefaultPersistedReviverFactory> {
	Arc::new(DefaultPersistedReviverFactory::new(
		provider,
		ToolRegistry::with_builtins(),
	))
}

fn hub_details(result: pi_agent::state::ToolResult) -> HubToolDetails {
	serde_json::from_value(result.details.expect("hub details"))
		.expect("hub details shape")
}

fn captured_has(provider: &GatedProvider, body: &str) -> bool {
	provider
		.captured
		.lock()
		.expect("captured")
		.iter()
		.any(|message| message.content.contains(body))
}

#[tokio::test]
async fn keep_alive_adopts_parks_revives_replays_and_survives_restart() {
	let root = session_root("full");
	let provider = gated_provider("revived-reply");
	let supervisor = test_supervisor(&root, provider.clone());
	let (id, contract) = adoptable_child(&supervisor, &root);

	let manager = attach(&supervisor, 120, reviver_factory(provider.clone()));
	manager.adopt(&id, Some(contract)).expect("adopt");
	assert_eq!(manager.status(&id), Some(LifecycleStatus::Idle));

	// Mail addressed to the idle agent waits in the mailbox; the parked row
	// exposes it as unread until a revive drains it through the peer inbox.
	supervisor
		.mailbox()
		.buffer(pi_agent::mailbox::IrcMessage::new(
			"Main",
			id.clone(),
			"queued note",
		));

	tokio::time::sleep(Duration::from_millis(350)).await;
	assert_eq!(manager.status(&id), Some(LifecycleStatus::Parked));

	let hub = HubTool::root(supervisor.clone());
	let ctx = AgentContext::new(root.clone());
	let listed = hub
		.execute(&json!({ "op": "list", "status": "parked" }), &ctx)
		.await
		.expect("hub list");
	let details = hub_details(listed);
	let peers = details.peers.expect("peers");
	assert_eq!(peers.len(), 1, "exactly the parked row");
	assert_eq!(peers[0].id, id);
	assert_eq!(peers[0].status, "parked");
	assert_eq!(peers[0].unread, 1, "buffered mail is visible as unread");
	let counts = details.counts.expect("counts");
	assert_eq!(counts.parked, 1);
	assert_eq!(counts.running, 0);

	// First send revives from the persisted contract + JSONL replay.
	let sent = hub
		.execute(
			&json!({ "op": "send", "to": id, "message": "wake up" }),
			&ctx,
		)
		.await
		.expect("hub send");
	let receipt = hub_details(sent).receipts.expect("receipts").remove(0);
	assert_eq!(receipt.outcome, "revived");
	assert!(receipt.error.is_none());

	tokio::time::timeout(Duration::from_secs(5), provider.entered.notified())
		.await
		.expect("revived turn entered provider");
	// The replay carries the previous turn, the buffered note, and the new send.
	assert!(captured_has(&provider, "investigate the repo"));
	assert!(captured_has(&provider, "prior-answer"));
	assert!(captured_has(&provider, "queued note"));
	assert!(captured_has(&provider, "wake up"));
	assert!(captured_has(&provider, "child system prompt"));

	// Second send goes to the live revived session, not another revive.
	let again = hub
		.execute(
			&json!({ "op": "send", "to": id, "message": "follow-up" }),
			&ctx,
		)
		.await
		.expect("second hub send");
	let receipt = hub_details(again).receipts.expect("receipts").remove(0);
	assert_eq!(receipt.outcome, "injected");

	// Release the first turn; the queued follow-up runs against the same state.
	provider.gate.add_permits(1);
	tokio::time::timeout(Duration::from_secs(5), provider.entered.notified())
		.await
		.expect("follow-up entered provider");
	assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
	assert!(captured_has(&provider, "follow-up"));
	provider.gate.add_permits(1);

	// The follow-up reply is appended to the real transcript.
	let transcript = hub_park::transcript_path(&supervisor, &id).expect("transcript path");
	let mut journaled = false;
	for _ in 0..200 {
		let messages = load_transcript(&transcript).expect("transcript");
		if messages.iter().any(|m| m.content == "revived-reply") {
			journaled = true;
			break;
		}
		tokio::time::sleep(Duration::from_millis(10)).await;
	}
	assert!(journaled, "revived turn must be appended to the child JSONL");

	let router = InternalUrlRouter::new();
	hub_park::register_history(&router, manager.clone()).expect("history handler");
	let resource = router
		.resolve(
			&format!("history://{id}"),
			&UrlScope::default().resolve_context(&root, &CancellationToken::new()),
		)
		.await
		.expect("history readable");
	assert!(resource.content.contains("prior-answer"));
	assert!(resource.content.contains("revived-reply"));

	// Restart: a fresh supervisor + manager restores the same parked row from
	// disk without any in-memory adoption.
	drop(hub);
	drop(manager);
	drop(supervisor);
	let restarted_provider = gated_provider("unused-reply");
	let restarted = test_supervisor(&root, restarted_provider.clone());
	let restarted_manager = attach(&restarted, 120, reviver_factory(restarted_provider));
	let restarted_hub = HubTool::root(restarted.clone());
	let listed = restarted_hub
		.execute(
			&json!({ "op": "list", "status": "parked" }),
			&AgentContext::new(root.clone()),
		)
		.await
		.expect("restarted list");
	let details = hub_details(listed);
	let peers = details.peers.expect("peers");
	assert_eq!(peers.len(), 1, "restored parked row");
	assert_eq!(peers[0].id, id);
	assert_eq!(peers[0].status, "parked");

	// A hard kill tombstones the id: transcript stays readable, revive refuses.
	assert!(restarted_manager.tombstone(&id, "hard kill"));
	let refused = restarted_hub
		.execute(
			&json!({ "op": "send", "to": id, "message": "please come back" }),
			&AgentContext::new(root.clone()),
		)
		.await
		.expect("tombstoned send returns a receipt");
	let receipt = hub_details(refused).receipts.expect("receipts").remove(0);
	assert_eq!(receipt.outcome, "failed");
	assert!(
		receipt
			.error
			.as_deref()
			.is_some_and(|error| error.contains("cannot be messaged or revived")),
		"tombstoned id must refuse revive: {:?}",
		receipt.error
	);
	let after = restarted_hub
		.execute(
			&json!({ "op": "list", "status": "parked" }),
			&AgentContext::new(root.clone()),
		)
		.await
		.expect("post-tombstone list");
	assert_eq!(
		hub_details(after).peers.expect("peers").len(),
		0,
		"tombstoned row leaves the roster"
	);
	let router = InternalUrlRouter::new();
	hub_park::register_history(&router, restarted_manager).expect("history handler");
	let history = router
		.resolve(
			&format!("history://{id}"),
			&UrlScope::default().resolve_context(&root, &CancellationToken::new()),
		)
		.await
		.expect("tombstoned transcript readable");
	assert!(history.content.contains("prior-answer"));
}

#[tokio::test]
async fn gone_workspace_is_transcript_only_and_refuses_revive() {
	let root = session_root("gone");
	let provider = gated_provider("never");
	let supervisor = test_supervisor(&root, provider.clone());
	let (id, mut contract) = adoptable_child(&supervisor, &root);
	contract.cwd = root.join("workspace-that-was-removed");
	contract.write(&root).expect("write invalid contract");
	assert!(
		PersistedChildSession::load(&root, &id).is_none(),
		"a gone workspace is not a usable contract"
	);

	let manager = attach(&supervisor, DEFAULT_AGENT_IDLE_TTL_MS, reviver_factory(provider));
	manager.adopt(&id, Some(contract)).expect("adopt");
	manager.park(&id);
	let outcome: Result<DeliveryOutcome, String> = manager.send(&id, "hello".to_string());
	let error = outcome.expect_err("revive must refuse a gone workspace");
	assert!(
		error.contains("no longer exists"),
		"exact refusal should name the workspace: {error}"
	);

	let router = InternalUrlRouter::new();
	hub_park::register_history(&router, manager).expect("history handler");
	let resource = router
		.resolve(
			&format!("history://{id}"),
			&UrlScope::default().resolve_context(&root, &CancellationToken::new()),
		)
		.await
		.expect("transcript remains readable");
	assert!(resource.content.contains("prior-answer"));
}

