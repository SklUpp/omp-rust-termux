//! Behavioral park/revive parity: keep-alive adoption, idle TTL parking, hub
//! roster visibility, JSONL-replayed revival, restart restoration, `history://`,
//! and tombstone refusal.

use std::{
	path::{Path, PathBuf},
	sync::{
		Arc, Mutex,
		atomic::{AtomicUsize, Ordering},
	},
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use futures::stream;
use pi_agent::{
	CancellationToken,
	provider::{
		ChunkStream, ModelProvider, ProviderError, StreamChunk, ToolCallDelta, ToolChoice,
	},
	state::{AgentContext, Message, Role, ToolCall, ToolResult, TurnResult},
	subagent_lifecycle::{
		DEFAULT_AGENT_IDLE_TTL_MS, DeliveryOutcome, LifecycleStatus, attach,
	},
	supervisor::{ChildSpawn, DelegationLimits, Supervisor},
	task_persisted_revive::{
		DefaultPersistedReviverFactory, PersistedChildReviverFactory,
		PersistedChildSession, load_transcript, transcript_path,
		turn_journal_messages,
	},
	tools::{
		Tool,
		hub::{HubTool, HubToolDetails},
		hub_park,
		internal_urls::{InternalUrlRouter, UrlScope},
		registry::ToolRegistry,
		task_catalog::{AgentCatalog, AgentDefinition, AgentSource},
	},
};
use serde_json::json;
use tokio::sync::{Notify, Semaphore};

fn stream_text(content: &str) -> StreamChunk {
	StreamChunk {
		content: Some(content.to_string()),
		reasoning: None,
		tool_call_deltas: Vec::new(),
		usage: None,
		is_done: false,
		content_index: None,
		signature: None,
		redacted: None,
		stop_reason: None,
		error_message: None,
		disabled_features: None,
	}
}

fn stream_done() -> StreamChunk {
	StreamChunk {
		content: None,
		reasoning: None,
		tool_call_deltas: Vec::new(),
		usage: None,
		is_done: true,
		content_index: None,
		signature: None,
		redacted: None,
		stop_reason: None,
		error_message: None,
		disabled_features: None,
	}
}

/// Convert a scripted assistant message into stream chunks. Tool calls use a
/// complete `arguments_delta`; the loop assembles the same message the
/// non-streaming path would have returned.
fn message_chunks(message: &Message) -> Vec<Result<StreamChunk, ProviderError>> {
	let mut chunks = Vec::new();
	if !message.content.is_empty() {
		chunks.push(Ok(stream_text(&message.content)));
	}
	for (index, call) in message.tool_calls.iter().enumerate() {
		chunks.push(Ok(StreamChunk {
			content: None,
			reasoning: None,
			tool_call_deltas: vec![ToolCallDelta {
				index,
				id: Some(call.id.clone()),
				name: Some(call.name.clone()),
				arguments_delta: Some(call.arguments.to_string()),
				content_index: None,
			}],
			usage: None,
			is_done: false,
			content_index: None,
			signature: None,
			redacted: None,
			stop_reason: None,
			error_message: None,
			disabled_features: None,
		}));
	}
	chunks.push(Ok(stream_done()));
	chunks
}

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
		messages: &[Message],
		_tools: &[serde_json::Value],
		_tool_choice: &ToolChoice,
	) -> Result<ChunkStream, ProviderError> {
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
		Ok(Box::pin(stream::iter(vec![
			Ok(stream_text(&self.reply)),
			Ok(stream_done()),
		])))
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

/// Caller scope that explicitly authorizes the manager root.
fn history_scope(root: &Path) -> UrlScope {
	let mut scope = UrlScope::default();
	scope
		.local_roots
		.insert("local".to_string(), root.to_path_buf());
	scope
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
			output_prefix:       None,
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
		depth: 1,
		max_turns: 20,
		context_limit: 100_000,
		display_name: Some("kept".to_string()),
		parent_id: None,
		output_id: None,
		output_published: false,
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
			&history_scope(&root).resolve_context(&root, &CancellationToken::new()),
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
			&history_scope(&root).resolve_context(&root, &CancellationToken::new()),
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
			&history_scope(&root).resolve_context(&root, &CancellationToken::new()),
		)
		.await
		.expect("transcript remains readable");
	assert!(resource.content.contains("prior-answer"));
}


fn home_session_root(label: &str) -> PathBuf {
	let nanos = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map(|duration| duration.as_nanos())
		.unwrap_or_default();
	let home = std::env::var_os("HOME")
		.map(PathBuf::from)
		.expect("HOME must be set");
	let dir = home.join(format!(
		".omp-hub-park-{label}-{}-{nanos}",
		std::process::id()
	));
	std::fs::create_dir_all(&dir).expect("session root");
	dir
}

fn scout_definition() -> AgentDefinition {
	AgentDefinition {
		name: "scout".to_string(),
		description: "test scout".to_string(),
		tools: None,
		spawns: None,
		model: vec!["@task".to_string()],
		output: None,
		thinking_level: None,
		blocking: false,
		read_summarize: None,
		prewalk: None,
		advisor: None,
		autoload_skills: Vec::new(),
		system_prompt: String::new(),
		source: AgentSource::Bundled,
		file_path: "embedded:scout.md".to_string(),
	}
}

#[tokio::test]
async fn actor_rewrites_only_yielding_follow_up_artifacts() {
	let root = home_session_root("actor-artifact");
	let provider = ScriptProvider::new(vec![
		Message::assistant_with_tools(
			"",
			vec![ToolCall {
				id:     "call-yield".to_string(),
				name:   "yield".to_string(),
				arguments: json!({ "result": { "data": { "value": "yielded" } } }),
			}],
		),
		Message::assistant("plain reply"),
	]);
	let supervisor = one_turn(provider.clone(), 2);
	supervisor.set_child_session_root(Some(root.clone()));
	let mut scope = UrlScope::default();
	scope
		.local_roots
		.insert("local".to_string(), root.clone());
	supervisor.set_url_scope(Arc::new(scope));
	supervisor.set_agent_catalog(Arc::new(AgentCatalog::new(
		vec![scout_definition()],
		None,
	)));
	let (id, mut contract) = adoptable_child(&supervisor, &root);
	contract.output_id = Some("Saved".to_string());
	contract.output_published = true;
	contract.write(&root).expect("published contract");
	std::fs::write(root.join("Saved.md"), "initial artifact").expect("initial artifact");
	let reviver: Arc<dyn PersistedChildReviverFactory> = Arc::new(
		DefaultPersistedReviverFactory::new(
			provider,
			ToolRegistry::with_builtins(),
		),
	);
	let manager = attach(&supervisor, 120, reviver);
	manager.adopt(&id, Some(contract)).expect("adopt");
	let hub = HubTool::root(supervisor.clone());
	let ctx = AgentContext::new(root.clone());

	let sent = hub
		.execute(&json!({ "op": "send", "to": id, "message": "yield" }), &ctx)
		.await
		.expect("hub send");
	let receipt = hub_details(sent).receipts.expect("receipts").remove(0);
	assert!(receipt.error.is_none(), "{:?}", receipt.error);
	let mut yielded = None;
	for _ in 0..300 {
		let body = std::fs::read_to_string(root.join("Saved.md")).unwrap_or_default();
		if body.contains("yielded") {
			yielded = Some(body);
			break;
		}
		tokio::time::sleep(Duration::from_millis(10)).await;
	}
	let yielded = yielded.expect("yielding follow-up must rewrite Saved.md");

	let sent = hub
		.execute(&json!({ "op": "send", "to": id, "message": "plain" }), &ctx)
		.await
		.expect("second hub send");
	let receipt = hub_details(sent).receipts.expect("receipts").remove(0);
	assert!(receipt.error.is_none(), "{:?}", receipt.error);
	let mut replied = false;
	for _ in 0..300 {
		let messages = load_transcript(&transcript_path(&root, &id)).unwrap_or_default();
		if messages.iter().any(|message| message.content.contains("plain reply")) {
			replied = true;
			break;
		}
		tokio::time::sleep(Duration::from_millis(10)).await;
	}
	assert!(replied, "plain follow-up must run");
	assert_eq!(
		std::fs::read_to_string(root.join("Saved.md")).expect("artifact"),
		yielded,
		"yield-less follow-up must leave the prior artifact intact"
	);
	let _ = std::fs::remove_dir_all(&root);
}

fn contract_for(
	id: &str,
	root: &PathBuf,
	task: &str,
	tools: Vec<String>,
	agent: &str,
	depth: i32,
) -> PersistedChildSession {
	PersistedChildSession {
		id: id.to_string(),
		agent: Some(agent.to_string()),
		task: task.to_string(),
		system_prompt: "child system prompt".to_string(),
		tools,
		model_role: None,
		resolved_model: None,
		output_schema: None,
		output_schema_mode: None,
		restrict_tool_names: Some(true),
		spawns: Some("*".to_string()),
		read_summarize: None,
		advisor: None,
		read_only: None,
		cwd: root.clone(),
		depth,
		max_turns: 20,
		context_limit: 100_000,
		display_name: Some(id.to_string()),
		parent_id: Some("Main".to_string()),
		output_id: None,
		output_published: false,
	}
}

/// Replies from a fixed script; running out of replies fails the turn so a
/// surprise extra request is visible instead of hanging.
struct ScriptProvider {
	replies: Mutex<Vec<Message>>,
}

impl ScriptProvider {
	fn new(mut replies: Vec<Message>) -> Arc<Self> {
		replies.reverse();
		Arc::new(Self {
			replies: Mutex::new(replies),
		})
	}
}

#[async_trait]
impl ModelProvider for ScriptProvider {
	async fn complete(
		&self,
		_messages: &[Message],
		_tools: &[serde_json::Value],
		_tool_choice: &ToolChoice,
	) -> Result<Message, ProviderError> {
		self.replies
			.lock()
			.expect("replies")
			.pop()
			.ok_or_else(|| ProviderError::Protocol("script exhausted".to_string()))
	}

	async fn stream(
		&self,
		_messages: &[Message],
		_tools: &[serde_json::Value],
		_tool_choice: &ToolChoice,
	) -> Result<ChunkStream, ProviderError> {
		let message = self
			.replies
			.lock()
			.expect("replies")
			.pop()
			.ok_or_else(|| ProviderError::Protocol("script exhausted".to_string()))?;
		Ok(Box::pin(stream::iter(message_chunks(&message))))
	}
}

/// Enforces the provider-side rule a broken replay violates: every assistant
/// `tool_calls` id must be answered by the immediately following tool results.
struct PairingProvider {
	seen: Mutex<Vec<Message>>,
}

impl PairingProvider {
	fn new() -> Arc<Self> {
		Arc::new(Self {
			seen: Mutex::new(Vec::new()),
		})
	}

	fn saw(&self, body: &str) -> bool {
		self.seen
			.lock()
			.expect("seen")
			.iter()
			.any(|message| message.content.contains(body))
	}
}

#[async_trait]
impl ModelProvider for PairingProvider {
	async fn complete(
		&self,
		messages: &[Message],
		_tools: &[serde_json::Value],
		_tool_choice: &ToolChoice,
	) -> Result<Message, ProviderError> {
		for (index, message) in messages.iter().enumerate() {
			if message.role != Role::Assistant || message.tool_calls.is_empty() {
				continue;
			}
			for call in &message.tool_calls {
				let answered = messages[index + 1..]
					.iter()
					.take_while(|next| next.role == Role::Tool)
					.any(|next| next.tool_call_id.as_deref() == Some(call.id.as_str()));
				if !answered {
					return Err(ProviderError::Protocol(format!(
						"tool_use {} without tool_result",
						call.id
					)));
				}
			}
		}
		self.seen
			.lock()
			.expect("seen")
			.extend_from_slice(messages);
		Ok(Message::assistant("revived-ok"))
	}

	async fn stream(
		&self,
		_messages: &[Message],
		_tools: &[serde_json::Value],
		_tool_choice: &ToolChoice,
	) -> Result<ChunkStream, ProviderError> {
		Err(ProviderError::Protocol(
			"revived runs use complete".to_string(),
		))
	}
}

fn one_turn(provider: Arc<dyn ModelProvider>, depth_limit: i32) -> Arc<Supervisor> {
	let supervisor = Supervisor::new(
		ToolRegistry::with_builtins(),
		provider,
		None,
		DelegationLimits {
			max_depth: depth_limit,
			max_concurrency: 32,
		},
	);
	supervisor
}

fn reserve_in(supervisor: &Arc<Supervisor>, name: &str) -> String {
	let (id, _token) = supervisor
		.reserve_child_full(ChildSpawn {
			name,
			index: 1,
			agent: "scout",
			agent_source: "builtin",
			task: "investigate",
			parent_id: None,
			parent_tool_call_id: Some("call-depth"),
			detached: false,
			output_prefix: None,
		})
		.expect("reserve child");
	id
}

fn task_call() -> Message {
	Message::assistant_with_tools(
		"",
		vec![ToolCall {
			id:     "call-task".to_string(),
			name:   "task".to_string(),
			arguments: json!({ "agent": "scout", "task": "nested" }),
		}],
	)
}

/// Blocker 1: a revived child must be rebuilt with the child-bound registry at
/// its recorded depth, not from the root registry.
#[tokio::test]
async fn revived_child_is_bound_at_its_recorded_depth() {
	// At max depth the child's registry has no `task` tool at all; a root
	// registry would still register one.
	let root = session_root("depth-max");
	let provider = ScriptProvider::new(vec![task_call(), Message::assistant("done")]);
	let supervisor = one_turn(provider.clone(), 1);
	supervisor.set_child_session_root(Some(root.clone()));
	let id = reserve_in(&supervisor, "depth-max");
	supervisor.append_child_transcript(
		&id,
		&[&Message::user("prior"), &Message::assistant("ok")],
	);
	let contract = contract_for(
		&id,
		&root,
		"investigate",
		vec!["task".to_string(), "yield".to_string()],
		"scout",
		1,
	);
	contract.write(&root).expect("write contract");
	let replayed = load_transcript(&transcript_path(&root, &id)).expect("transcript");
	let mut revived = DefaultPersistedReviverFactory::new(provider, ToolRegistry::with_builtins())
		.revive(&contract, replayed, Some(&supervisor))
		.expect("revive");
	let turns = revived
		.agent_loop
		.run_dialog_message(Message::user("go"), &mut revived.state, None, None)
		.await
		.expect("revived turn");
	assert!(!turns.is_empty());
	let result = &turns[0].tool_results[0];
	assert!(result.is_error, "task must not run at max depth");
	assert!(
		result.content.contains("Tool not found: task"),
		"a revived child at max depth keeps the depth guard: {}",
		result.content
	);

	// Below max depth the nested `task` tool keeps the agent context, so the
	// self-spawn guard fires instead of the root tool accepting the spawn.
	let root = session_root("depth-agent");
	let provider = ScriptProvider::new(vec![task_call(), Message::assistant("done")]);
	let supervisor = one_turn(provider.clone(), 2);
	supervisor.set_child_session_root(Some(root.clone()));
	let id = reserve_in(&supervisor, "depth-agent");
	supervisor.append_child_transcript(
		&id,
		&[&Message::user("prior"), &Message::assistant("ok")],
	);
	let contract = contract_for(
		&id,
		&root,
		"investigate",
		vec!["task".to_string(), "yield".to_string()],
		"scout",
		1,
	);
	let replayed = load_transcript(&transcript_path(&root, &id)).expect("transcript");
	let mut revived = DefaultPersistedReviverFactory::new(provider, ToolRegistry::with_builtins())
		.revive(&contract, replayed, Some(&supervisor))
		.expect("revive");
	let turns = revived
		.agent_loop
		.run_dialog_message(Message::user("go"), &mut revived.state, None, None)
		.await
		.expect("revived turn");
	let result = &turns[0].tool_results[0];
	assert!(
		result.content.contains("from within itself"),
		"the revived task tool must carry the child agent context: {}",
		result.content
	);
}

/// Blocker 1 (return channel): a revived child can actually call `yield`, the
/// tool only a child-bound registry mounts.
#[tokio::test]
async fn revived_child_can_yield() {
	let root = session_root("yield");
	let provider = ScriptProvider::new(vec![
		Message::assistant_with_tools(
			"",
			vec![ToolCall {
				id:     "call-yield".to_string(),
				name:   "yield".to_string(),
				arguments: json!({ "result": { "data": "done" } }),
			}],
		),
		Message::assistant("done"),
	]);
	let supervisor = one_turn(provider.clone(), 2);
	supervisor.set_child_session_root(Some(root.clone()));
	let id = reserve_in(&supervisor, "yield");
	supervisor.append_child_transcript(
		&id,
		&[&Message::user("prior"), &Message::assistant("ok")],
	);
	let contract = contract_for(&id, &root, "investigate", vec!["yield".to_string()], "scout", 1);
	let replayed = load_transcript(&transcript_path(&root, &id)).expect("transcript");
	let mut revived = DefaultPersistedReviverFactory::new(provider, ToolRegistry::with_builtins())
		.revive(&contract, replayed, Some(&supervisor))
		.expect("revive");
	let turns = revived
		.agent_loop
		.run_dialog_message(Message::user("go"), &mut revived.state, None, None)
		.await
		.expect("revived turn");
	let result = &turns[0].tool_results[0];
	assert!(!result.is_error, "yield must exist: {}", result.content);
	assert!(result.terminate, "yield terminates the turn");
}

/// Blocker 2: the journal records tool results with their calls, and a legacy
/// transcript with dangling calls is stripped so the first revived request is
/// not rejected.
#[tokio::test]
async fn revived_replay_pairs_tool_calls_with_results() {
	let root = session_root("replay");
	let provider = PairingProvider::new();
	let supervisor = one_turn(provider.clone(), 2);
	supervisor.set_child_session_root(Some(root.clone()));
	let id = reserve_in(&supervisor, "replay");
	let journaled = turn_journal_messages(&[TurnResult {
		message:       Message::assistant_with_tools(
			"working",
			vec![ToolCall {
				id:     "call-read".to_string(),
				name:   "read".to_string(),
				arguments: json!({ "path": "a" }),
			}],
		),
		tool_results:  vec![ToolResult {
			tool_call_id: "call-read".to_string(),
			content:      "file body".to_string(),
			is_error:     false,
			terminate:    false,
			details:      None,
		}],
		is_completed:  false,
	}]);
	let journaled_refs: Vec<&Message> = journaled.iter().collect();
	supervisor.append_child_transcript(&id, &journaled_refs);
	// Legacy/truncated tail: an assistant call with no result on disk.
	supervisor.append_child_transcript(
		&id,
		&[&Message::assistant_with_tools(
			"legacy",
			vec![ToolCall {
				id:     "stale-call".to_string(),
				name:   "read".to_string(),
				arguments: json!({ "path": "b" }),
			}],
		)],
	);

	let replayed = load_transcript(&transcript_path(&root, &id)).expect("transcript");
	assert!(
		replayed
			.iter()
			.filter(|message| message.role == Role::Assistant)
			.flat_map(|message| message.tool_calls.iter())
			.all(|call| call.id != "stale-call"),
		"an unanswered call must not be replayed"
	);
	assert!(
		replayed
			.iter()
			.any(|message| message.role == Role::Tool && message.content == "file body"),
		"the tool result is part of the journal"
	);

	let contract = contract_for(
		&id,
		&root,
		"prior work",
		vec!["read".to_string(), "yield".to_string()],
		"scout",
		1,
	);
	let mut revived = DefaultPersistedReviverFactory::new(provider.clone(), ToolRegistry::with_builtins())
		.revive(&contract, replayed, Some(&supervisor))
		.expect("revive");
	let turns = revived
		.agent_loop
		.run_dialog_message(Message::user("go"), &mut revived.state, None, None)
		.await
		.expect("a revived tool-using child must not be rejected");
	assert_eq!(turns.len(), 1);
	assert!(
		provider.saw("file body"),
		"the revived child keeps its tool context"
	);
}

/// Major 3: a restored parked `task-N` must never collide with a fresh spawn
/// after the supervisor's sequence restarts.
#[tokio::test]
async fn resumed_parked_child_does_not_collide_with_a_fresh_spawn() {
	let root = session_root("resume");
	let provider = gated_provider("unused");
	let supervisor = test_supervisor(&root, provider.clone());
	let (id, contract) = adoptable_child(&supervisor, &root);
	let manager = attach(
		&supervisor,
		DEFAULT_AGENT_IDLE_TTL_MS,
		reviver_factory(provider),
	);
	// `attach` scanned the spawn-time contract as a restored parked row.
	assert_eq!(manager.status(&id), Some(LifecycleStatus::Parked));

	let mut different = contract.clone();
	different.task = "a different spawn".to_string();
	different.display_name = Some("different".to_string());
	assert!(
		manager.adopt(&id, Some(different)).is_err(),
		"a different contract must not replace a restored parked row"
	);
	manager
		.adopt(&id, Some(contract))
		.expect("the same session may be re-adopted");
	assert_eq!(manager.status(&id), Some(LifecycleStatus::Idle));
	drop(manager);
	drop(supervisor);

	// A fresh supervisor on the same stable session directory restarts ids at 1.
	let restarted_provider = gated_provider("unused-restart");
	let restarted = test_supervisor(&root, restarted_provider.clone());
	let restarted_manager = attach(
		&restarted,
		DEFAULT_AGENT_IDLE_TTL_MS,
		reviver_factory(restarted_provider),
	);
	assert_eq!(
		restarted_manager.status(&id),
		Some(LifecycleStatus::Parked),
		"the parked row is restored from disk"
	);
	let fresh_id = reserve_in(&restarted, "fresh");
	assert_ne!(fresh_id, id, "a resumed parked id must never be reused");
	restarted.append_child_transcript(
		&fresh_id,
		&[&Message::user("fresh work"), &Message::assistant("fresh-answer")],
	);
	let fresh_contract = contract_for(
		&fresh_id,
		&root,
		"fresh work",
		vec!["read".to_string()],
		"scout",
		1,
	);
	fresh_contract.write(&root).expect("fresh contract");
	restarted.mark_completed(&fresh_id, 3, "fresh-answer");
	restarted_manager
		.adopt(&fresh_id, Some(fresh_contract))
		.expect("adopt the fresh child");

	let roster = restarted_manager.roster();
	assert!(roster
		.iter()
		.any(|row| row.id == id && row.status == LifecycleStatus::Parked));
	assert!(roster
		.iter()
		.any(|row| row.id == fresh_id && row.status == LifecycleStatus::Idle));
	let restored = PersistedChildSession::load(&root, &id).expect("restored contract");
	assert_eq!(
		restored.task, "investigate the repo",
		"the parked child's contract is untouched"
	);

	// Minor 4: a tombstone restored after a restart has no roster entry, yet
	// `history://` must still find its transcript on disk.
	assert!(restarted_manager.tombstone(&id, "hard kill"));
	drop(restarted_manager);
	drop(restarted);
	let after_provider = gated_provider("unused-after");
	let after = test_supervisor(&root, after_provider.clone());
	let after_manager = attach(
		&after,
		DEFAULT_AGENT_IDLE_TTL_MS,
		reviver_factory(after_provider),
	);
	assert_eq!(
		after_manager.status(&id),
		None,
		"a tombstoned id has no roster row"
	);
	assert!(
		after_manager.transcript_path(&id).is_some(),
		"history:// falls back to the transcript on disk"
	);
}
