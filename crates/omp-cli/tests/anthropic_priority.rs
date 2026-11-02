//! Anthropic priority (`serviceTier: "priority"`) wire contract for the real
//! runner: the direct provider requests fast mode, a server rejection retries
//! once without it, the terminal message reports the dropped feature, and the
//! following request stays downgraded.
//!
//! Source: TS `anthropic.ts:2764-2783,2846-2848,3478-3480` (request flag,
//! one-shot fast-mode fallback, terminal `disabledFeatures`) and
//! `error/flags.ts` `matchesFastModeUnsupported`.

use std::{
	collections::BTreeMap,
	io::{Read, Write},
	net::{TcpListener, TcpStream},
	path::PathBuf,
	sync::{Arc, mpsc},
	time::Duration,
};

use omp_cli::{
	agent::{AgentRunner, Event, RunRequest},
	approve::{ApprovalMode, Approvals},
	catalog::{Catalog, ModelInfo},
	config::{Config, ProviderConfig},
	output::OutputMode,
	runtime::PiAgentRunner,
	session::{SessionOptions, SessionStore},
};

const FAST_REJECTION: &str = r#"{"type":"error","error":{"type":"invalid_request_error","message":"speed is not supported for this model"}}"#;

const TURN: &str = concat!(
	"event: message_start\n",
	"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
	"event: content_block_start\n",
	"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
	"event: content_block_delta\n",
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"fast ok\"}}\n\n",
	"event: content_block_stop\n",
	"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
	"event: message_delta\n",
	"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
	"event: message_stop\n",
	"data: {\"type\":\"message_stop\"}\n\n"
);

fn catalog(addr: std::net::SocketAddr) -> Catalog {
	let mut models = BTreeMap::new();
	models.insert("claude-priority".to_string(), ModelInfo {
		provider: "anthropic".to_string(),
		api: Some("anthropic-messages".to_string()),
		base_url: Some(format!("http://{addr}/v1")),
		max_tokens: Some(4096),
		..ModelInfo::default()
	});
	let mut providers = BTreeMap::new();
	providers.insert("anthropic".to_string(), models);
	Catalog { providers, connections: BTreeMap::new(), source: PathBuf::from("models.json") }
}

fn request(catalog: Catalog, tiers: &[(&str, &str)]) -> RunRequest {
	let mut config = Config::default();
	config.providers.insert("anthropic".to_string(), ProviderConfig {
		api_key: Some("sk-ant-priority".to_string()),
		..ProviderConfig::default()
	});
	let mut service_tier_by_family = BTreeMap::new();
	for (family, tier) in tiers {
		service_tier_by_family.insert((*family).to_string(), (*tier).to_string());
	}
	RunRequest {
		prompt: "answer quickly".to_string(),
		parts: Vec::new(),
		cwd: std::env::temp_dir(),
		output_mode: OutputMode::Json,
		model: Some("anthropic/claude-priority".to_string()),
		provider: None,
		api_key: None,
		provider_session_id: None,
		prompt_cache_key: None,
		system_prompt: Some("base prompt".to_string()),
		append_system_prompt: None,
		context_files: Vec::new(),
		skills: Vec::new(),
		history: Vec::new(),
		history_messages: None,
		checkpoint_controller: None,
		turn_ack: None,
		todo_phases: Vec::new(),
		session_persisted: false,
		session_file: None,
		session_id: None,
		config,
		catalog: Arc::new(catalog),
		approvals: Approvals::new(ApprovalMode::Yolo, Vec::new()),
		max_time: Some(Duration::from_secs(30)),
		max_turns: 1,
		tools_enabled: false,
		cancellation_token: None,
		extra_tools: Vec::new(),
		acp_bridge: None,
		extra_mcp_servers: Vec::new(),
		internal_urls: None,
		url_scope: None,
		subagent_observer: None,
		child_session_root: None,
		goal: None,
		goal_context: None,
		credential_resolver: None,
		plan_mode: Default::default(),
		continuation: false,
		plan_context: None,
		service_tier_by_family,
	}
}

fn read_request(socket: &mut TcpStream) -> (String, String) {
	let mut buffer = Vec::new();
	let mut chunk = [0u8; 4096];
	let header_end = loop {
		let read = socket.read(&mut chunk).expect("read request");
		assert!(read > 0, "client closed before the request headers");
		buffer.extend_from_slice(&chunk[..read]);
		if let Some(position) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
			break position + 4;
		}
	};
	let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
	let length = head
		.lines()
		.find_map(|line| {
			let (name, value) = line.split_once(':')?;
			name.eq_ignore_ascii_case("content-length")
				.then(|| value.trim().parse::<usize>().ok())?
		})
		.unwrap_or(0);
	while buffer.len() < header_end + length {
		let read = socket.read(&mut chunk).expect("read body");
		assert!(read > 0, "client closed before the request body");
		buffer.extend_from_slice(&chunk[..read]);
	}
	(
		head,
		String::from_utf8(buffer[header_end..header_end + length].to_vec()).expect("utf8 body"),
	)
}

fn respond(socket: &mut TcpStream, status: &str, content_type: &str, body: &str) {
	let response = format!(
		"HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
		 Connection: close\r\n\r\n{body}",
		body.len()
	);
	socket.write_all(response.as_bytes()).expect("write response");
	socket.flush().expect("flush response");
}

fn run(request: RunRequest) -> (Vec<String>, Option<Vec<String>>) {
	let runner = PiAgentRunner::new().expect("runner");
	let mut stops = Vec::new();
	let mut disabled = None;
	for event in runner.run(&request).expect("stream") {
		match event {
			Ok(Event::MessageEnd { message, .. }) => disabled = message.disabled_features,
			Ok(Event::TurnEnd { stop_reason, .. }) => stops.push(stop_reason),
			_ => {},
		}
	}
	(stops, disabled)
}

fn body_field<'a>(body: &'a str, name: &str) -> Option<&'a serde_json::Value> {
	let value: serde_json::Value = serde_json::from_str(body).expect("json body");
	value.get(name)
}

#[test]
fn rejected_priority_retries_without_fast_mode_and_reports_the_drop() {
	let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
	let addr = listener.local_addr().expect("addr");
	let (captured_tx, captured_rx) = mpsc::channel::<(String, String)>();
	let server = std::thread::spawn(move || {
		// Priorität request: the server rejects the `speed` parameter.
		let (mut socket, _) = listener.accept().expect("accept");
		let captured = read_request(&mut socket);
		captured_tx.send(captured).expect("capture reject");
		respond(&mut socket, "400 Bad Request", "application/json", FAST_REJECTION);
		// Downgraded retry: same turn, no fast mode.
		let (mut socket, _) = listener.accept().expect("accept");
		let captured = read_request(&mut socket);
		captured_tx.send(captured).expect("capture retry");
		respond(&mut socket, "200 OK", "text/event-stream", TURN);
		// Following turn: the caller no longer asks for priority.
		let (mut socket, _) = listener.accept().expect("accept");
		let captured = read_request(&mut socket);
		captured_tx.send(captured).expect("capture follow-up");
		respond(&mut socket, "200 OK", "text/event-stream", TURN);
	});

	let first = request(catalog(addr), &[("anthropic", "priority")]);
	let (stops, disabled) = run(first);
	assert_eq!(stops.last().map(String::as_str), Some("stop"));
	assert_eq!(
		disabled.as_deref(),
		Some(["priority".to_string()].as_slice()),
		"the terminal message reports the dropped feature"
	);

	let (reject_head, reject_body) = captured_rx.recv().expect("reject request");
	assert!(reject_head.starts_with("POST /v1/messages HTTP/1.1"), "{reject_head}");
	assert!(
		reject_head
			.to_ascii_lowercase()
			.contains("anthropic-beta:") && reject_head.contains("fast-mode-2026-02-01"),
		"the priority request carries the fast-mode beta: {reject_head}"
	);
	assert_eq!(body_field(&reject_body, "speed"), Some(&serde_json::json!("fast")));

	let (retry_head, retry_body) = captured_rx.recv().expect("retry request");
	let retry_lower = retry_head.to_ascii_lowercase();
	assert!(
		!retry_lower.contains("fast-mode-2026-02-01"),
		"the retry drops the beta: {retry_head}"
	);
	assert_eq!(body_field(&retry_body, "speed"), None, "the retry drops `speed`");
	assert_eq!(body_field(&retry_body, "stream"), Some(&serde_json::json!(true)));

	let follow = request(catalog(addr), &[]);
	run(follow);
	let (_head, follow_body) = captured_rx.recv().expect("follow-up request");
	assert_eq!(body_field(&follow_body, "speed"), None, "no priority, no speed");
	server.join().expect("server");
}

fn tier_options(dir: std::path::PathBuf, id: &str) -> SessionOptions {
	SessionOptions {
		enabled: true,
		cwd: PathBuf::from("/work"),
		model: Some("anthropic/claude-priority".to_string()),
		provider: Some("anthropic".to_string()),
		dir: Some(dir),
		id: Some(id.to_string()),
		continue_latest: false,
		parent_session: None,
		no_title: true,
	}
}

fn tiers(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
	pairs
		.iter()
		.map(|(family, tier)| ((*family).to_string(), (*tier).to_string()))
		.collect()
}

/// TS `session-entries.ts:91-95` + `session-context.ts:263-264`: the entry is
/// the string-valued family map (or `null`), the last entry on the branch wins,
/// and an explicit reset is a recorded state — it must not fall back to config.
#[test]
fn service_tier_change_journals_strings_and_reset_survives_resume() {
	let dir = std::env::temp_dir().join(format!(
		"omp-anthropic-priority-{}-{}",
		std::process::id(),
		std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.expect("clock")
			.as_nanos()
	));
	std::fs::create_dir_all(&dir).expect("temp dir");

	let mut store = SessionStore::open(&tier_options(dir.clone(), "priority-tier"));
	assert!(!store.has_service_tier_entry(), "a fresh session has no selection");
	store
		.append_service_tier_change(&tiers(&[("anthropic", "priority")]))
		.expect("append selection");

	let reopened = SessionStore::open(&tier_options(dir.clone(), "priority-tier"));
	assert!(reopened.has_service_tier_entry());
	assert_eq!(
		reopened.service_tier_by_family(),
		tiers(&[("anthropic", "priority")]),
		"resume restores the recorded selection"
	);

	// Explicit reset: a recorded empty map, not a missing entry.
	let mut reset = SessionStore::open(&tier_options(dir.clone(), "priority-tier"));
	reset
		.append_service_tier_change(&BTreeMap::new())
		.expect("append reset");
	let resumed = SessionStore::open(&tier_options(dir.clone(), "priority-tier"));
	assert!(resumed.has_service_tier_entry(), "the reset is itself recorded");
	assert!(
		resumed.service_tier_by_family().is_empty(),
		"the reset must not re-enable the configured priority"
	);

	// Explicit re-arm after the reset wins as the last branch entry.
	let mut rearm = SessionStore::open(&tier_options(dir.clone(), "priority-tier"));
	rearm
		.append_service_tier_change(&tiers(&[("anthropic", "priority")]))
		.expect("append re-arm");
	assert_eq!(
		SessionStore::open(&tier_options(dir.clone(), "priority-tier")).service_tier_by_family(),
		tiers(&[("anthropic", "priority")]),
		"an explicit re-arm restores priority"
	);

	let log = std::fs::read_dir(&dir)
		.expect("session dir")
		.flatten()
		.map(|entry| entry.path())
		.find(|path| path.extension().and_then(|ext| ext.to_str()) == Some("jsonl"))
		.expect("session log");
	let contents = std::fs::read_to_string(&log).expect("read log");
	let entries: Vec<serde_json::Value> = contents
		.lines()
		.filter(|line| !line.trim().is_empty())
		.map(|line| serde_json::from_str(line).expect("json entry"))
		.collect();
	let tiers_json: Vec<&serde_json::Value> = entries
		.iter()
		.filter(|entry| entry["type"] == "service_tier_change")
		.map(|entry| &entry["serviceTier"])
		.collect();
	assert_eq!(tiers_json.len(), 3, "selection, reset, re-arm were all journaled");
	assert_eq!(tiers_json[0]["anthropic"], "priority", "values are ServiceTier strings");
	assert!(tiers_json[1].is_null(), "the reset is null, never false");
	let _ = std::fs::remove_dir_all(&dir);
}
