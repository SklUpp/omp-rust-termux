//! Wire-dialect dispatch contract for the real runner.
//!
//! A model's per-model `api`/`baseUrl` metadata (`models.json`/`models.yml`)
//! decides which adapter the CLI builds; the connection block is only a
//! fallback. These fixtures drive `PiAgentRunner` against a local HTTP server,
//! so the assertions cover the bytes each adapter puts on the wire instead of
//! constructor wiring: endpoint path, credentials, request body, stream parsing
//! and signed two-turn replay. An `api` this build has no adapter for fails
//! before any socket is opened.

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
};

/// One catalog provider/model pair, optionally with the provider-level
/// connection block a per-model `baseUrl` must override.
fn catalog_with_model(
	provider: &str,
	model: &str,
	info: ModelInfo,
	connection: Option<ProviderConfig>,
) -> Catalog {
	let mut models = BTreeMap::new();
	models.insert(model.to_string(), info);
	let mut providers = BTreeMap::new();
	providers.insert(provider.to_string(), models);
	let mut connections = BTreeMap::new();
	if let Some(connection) = connection {
		connections.insert(provider.to_string(), connection);
	}
	Catalog { providers, connections, source: PathBuf::from("models.json") }
}

/// Root request whose `config.yml` entry carries only the credential: the
/// catalog holds the (dead) provider fallback, so only the per-model `baseUrl`
/// can make the turn succeed.
fn wire_request(catalog: Catalog, provider: &str, model: &str, api_key: Option<&str>) -> RunRequest {
	let mut config = Config::default();
	config.providers.insert(provider.to_string(), ProviderConfig {
		api_key: api_key.map(str::to_string),
		..ProviderConfig::default()
	});
	RunRequest {
		prompt: "fix the build".to_string(),
		parts: Vec::new(),
		cwd: std::env::temp_dir(),
		output_mode: OutputMode::Json,
		model: Some(format!("{provider}/{model}")),
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
		max_turns: 4,
		tools_enabled: true,
		cancellation_token: None,
		extra_tools: Vec::new(),
		acp_bridge: None,
		extra_mcp_servers: Vec::new(),
		goal: None,
		goal_context: None,
		internal_urls: None,
		url_scope: None,
		subagent_observer: None,
		child_session_root: None,
		credential_resolver: None,
		plan_mode: Default::default(),
		continuation: false,
	}
}

/// Capture one blocking HTTP/1.1 request: head text and decoded body.
fn read_http_request(socket: &mut TcpStream) -> (String, String) {
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
	let content_length = head
		.lines()
		.find_map(|line| {
			let (name, value) = line.split_once(':')?;
			name.eq_ignore_ascii_case("content-length")
				.then(|| value.trim().parse::<usize>().ok())?
		})
		.unwrap_or(0);
	while buffer.len() < header_end + content_length {
		let read = socket.read(&mut chunk).expect("read body");
		assert!(read > 0, "client closed before the request body");
		buffer.extend_from_slice(&chunk[..read]);
	}
	(
		head,
		String::from_utf8(buffer[header_end..header_end + content_length].to_vec())
			.expect("utf-8 request body"),
	)
}

fn write_http_response(socket: &mut TcpStream, content_type: &str, body: &str) {
	let response = format!(
		"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
		 Connection: close\r\n\r\n{body}",
		body.len()
	);
	socket.write_all(response.as_bytes()).expect("write response");
	socket.flush().expect("flush response");
}

const ANTHROPIC_TURN_ONE: &str = concat!(
	"event: message_start\n",
	"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n",
	"event: content_block_start\n",
	"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
	"event: content_block_delta\n",
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan\"}}\n\n",
	"event: content_block_delta\n",
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig-root\"}}\n\n",
	"event: content_block_stop\n",
	"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
	"event: content_block_start\n",
	"data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
	"event: content_block_delta\n",
	"data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"checking\"}}\n\n",
	"event: content_block_stop\n",
	"data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
	"event: content_block_start\n",
	"data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"read\",\"input\":{}}}\n\n",
	"event: content_block_delta\n",
	"data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"missing-omp-fixture.txt\\\"}\"}}\n\n",
	"event: content_block_stop\n",
	"data: {\"type\":\"content_block_stop\",\"index\":2}\n\n",
	"event: message_delta\n",
	"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":42}}\n\n",
	"event: message_stop\n",
	"data: {\"type\":\"message_stop\"}\n\n"
);

const ANTHROPIC_TURN_TWO: &str = concat!(
	"event: message_start\n",
	"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_2\"}}\n\n",
	"event: content_block_start\n",
	"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
	"event: content_block_delta\n",
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n",
	"event: content_block_stop\n",
	"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
	"event: message_delta\n",
	"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n",
	"event: message_stop\n",
	"data: {\"type\":\"message_stop\"}\n\n"
);

/// The model's `anthropic-messages` dialect wins over the connection fallback,
/// the `/v1` base is normalized to the adapter's `/v1/messages` endpoint, a
/// non-official endpoint gets a bearer credential (never `x-api-key`), the
/// catalog `maxTokens` caps the request, and the signed
/// thinking block is replayed on the next turn.
#[test]
fn root_anthropic_dispatch_posts_messages_and_replays_signatures() {
	let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
	let addr = listener.local_addr().expect("addr");
	let (captured_tx, captured_rx) = mpsc::channel::<(String, String)>();
	let server = std::thread::spawn(move || {
		for body in [ANTHROPIC_TURN_ONE, ANTHROPIC_TURN_TWO] {
			let (mut socket, _) = listener.accept().expect("accept");
			let (head, request) = read_http_request(&mut socket);
			captured_tx.send((head, request)).expect("capture");
			write_http_response(&mut socket, "text/event-stream", body);
		}
	});

	let catalog = catalog_with_model("anthropic-proxy", "claude-test", ModelInfo {
		provider: "anthropic-proxy".to_string(),
		api: Some("anthropic-messages".to_string()),
		base_url: Some(format!("http://{addr}/v1")),
		max_tokens: Some(4096),
		..ModelInfo::default()
	}, Some(ProviderConfig {
		base_url: Some("http://127.0.0.1:1".to_string()),
		..ProviderConfig::default()
	}));
	let request = wire_request(catalog, "anthropic-proxy", "claude-test", Some("sk-ant-rootkey"));

	let runner = PiAgentRunner::new().expect("runner");
	let mut stops = Vec::new();
	for event in runner.run(&request).expect("stream") {
		if let Ok(Event::TurnEnd { stop_reason, .. }) = event {
			stops.push(stop_reason);
		}
	}
	server.join().expect("server");
	assert_eq!(stops.last().map(String::as_str), Some("stop"));

	let (head, body) = captured_rx.recv().expect("first request");
	assert!(head.starts_with("POST /v1/messages HTTP/1.1"), "{head}");
	let lower = head.to_ascii_lowercase();
	// Non-official endpoints authenticate with a bearer token; `x-api-key` is
	// reserved for the first-party host (adapter policy, see the native
	// non-official bearer fixture).
	assert!(
		lower.contains("authorization: bearer sk-ant-rootkey"),
		"non-official endpoint must send the bearer credential: {head}"
	);
	assert!(!lower.contains("x-api-key"), "non-official endpoint must not send x-api-key: {head}");
	assert!(lower.contains("anthropic-version:"), "{head}");
	let body: serde_json::Value = serde_json::from_str(&body).expect("json body");
	assert_eq!(body["model"], "claude-test");
	assert_eq!(body["stream"], true);
	assert_eq!(body["max_tokens"], 4096, "the catalog maxTokens caps the request");

	let (head, body) = captured_rx.recv().expect("second request");
	assert!(head.starts_with("POST /v1/messages HTTP/1.1"), "{head}");
	let body: serde_json::Value = serde_json::from_str(&body).expect("json body");
	let messages = body["messages"].to_string();
	assert!(messages.contains("\"type\":\"thinking\""), "{messages}");
	assert!(messages.contains("\"signature\":\"sig-root\""), "{messages}");
	assert!(messages.contains("\"type\":\"tool_result\""), "{messages}");
}

/// `ollama-chat` models post to `/api/chat`; the adapter strips the configured
/// `/api` suffix and parses the NDJSON stream back into the turn.
#[test]
fn root_ollama_dispatch_posts_api_chat_and_reads_ndjson() {
	const NDJSON: &str = concat!(
		"{\"model\":\"qwen3\",\"message\":{\"role\":\"assistant\",\"content\":\"hello\"},\"done\":false}\n",
		"{\"model\":\"qwen3\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":3,\"eval_count\":4}\n"
	);

	let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
	let addr = listener.local_addr().expect("addr");
	let (captured_tx, captured_rx) = mpsc::channel::<(String, String)>();
	let server = std::thread::spawn(move || {
		let (mut socket, _) = listener.accept().expect("accept");
		let (head, request) = read_http_request(&mut socket);
		captured_tx.send((head, request)).expect("capture");
		write_http_response(&mut socket, "application/x-ndjson", NDJSON);
	});

	let catalog = catalog_with_model("ollama-proxy", "qwen3", ModelInfo {
		provider: "ollama-proxy".to_string(),
		api: Some("ollama-chat".to_string()),
		base_url: Some(format!("http://{addr}/api")),
		max_tokens: Some(8192),
		..ModelInfo::default()
	}, Some(ProviderConfig {
		base_url: Some("http://127.0.0.1:1".to_string()),
		..ProviderConfig::default()
	}));
	let request = wire_request(catalog, "ollama-proxy", "qwen3", None);

	let runner = PiAgentRunner::new().expect("runner");
	let mut answer = None;
	let mut stops = Vec::new();
	for event in runner.run(&request).expect("stream") {
		match event {
			Ok(Event::MessageEnd { message, .. }) => answer = Some(message.content),
			Ok(Event::TurnEnd { stop_reason, .. }) => stops.push(stop_reason),
			_ => {},
		}
	}
	server.join().expect("server");
	assert_eq!(answer.as_deref(), Some("hello"), "NDJSON deltas reach the turn");
	assert_eq!(stops.last().map(String::as_str), Some("stop"));

	let (head, body) = captured_rx.recv().expect("request");
	assert!(head.starts_with("POST /api/chat HTTP/1.1"), "{head}");
	let body: serde_json::Value = serde_json::from_str(&body).expect("json body");
	assert_eq!(body["model"], "qwen3");
	assert_eq!(body["stream"], true);
	assert_eq!(body["options"]["num_predict"], 8192);
}

/// A dialect this build has no adapter for is a setup error before any contact,
/// never a silent OpenAI-compatible request to the wrong endpoint.
#[test]
fn unknown_provider_api_fails_before_contact() {
	let catalog = catalog_with_model("mystery", "model-x", ModelInfo {
		provider: "mystery".to_string(),
		api: Some("openai-responses".to_string()),
		base_url: Some("http://127.0.0.1:9".to_string()),
		..ModelInfo::default()
	}, None);
	let request = wire_request(catalog, "mystery", "model-x", None);
	let runner = PiAgentRunner::new().expect("runner");
	let error = match runner.run(&request) {
		Ok(_) => panic!("unsupported api must fail before building a provider"),
		Err(error) => error,
	};
	let message = error.to_string();
	assert!(message.contains("unsupported provider api `openai-responses`"), "{message}");
}
