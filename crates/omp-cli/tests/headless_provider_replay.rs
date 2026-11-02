//! Headless provider-context boundaries owned by `main.rs`:
//! - a journaled turn persists its streamed blocks in order (thinking, text,
//!   tool call) without a duplicated `tool_calls` projection, and a resumed run
//!   replays that call exactly once;
//! - a stored conversation that cannot be rebuilt losslessly fails loudly
//!   before any provider request instead of replaying a degraded history.

use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcCommand, Output, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;

static COUNTER: AtomicU32 = AtomicU32::new(0);

const DEFAULT_SSE: &str =
	"data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

fn temp_dir(prefix: &str) -> PathBuf {
	let dir = std::env::temp_dir().join(format!(
		"omp-headless-replay-{prefix}-{}-{}",
		std::process::id(),
		COUNTER.fetch_add(1, Ordering::Relaxed)
	));
	fs::create_dir_all(&dir).expect("create temp dir");
	dir
}

struct MockWireServer {
	port: u16,
	requests: Arc<Mutex<Vec<serde_json::Value>>>,
	stop_tx: mpsc::Sender<()>,
	handle: Option<thread::JoinHandle<()>>,
}

impl MockWireServer {
	fn start(bodies: Vec<String>) -> Self {
		let listener = TcpListener::bind("127.0.0.1:0").expect("bind tcp");
		let port = listener.local_addr().expect("local addr").port();
		listener.set_nonblocking(true).expect("set nonblocking");
		let (stop_tx, stop_rx) = mpsc::channel::<()>();
		let requests = Arc::new(Mutex::new(Vec::new()));
		let reqs_clone = Arc::clone(&requests);
		let bodies = Arc::new(bodies);

		let handle = thread::spawn(move || {
			loop {
				if stop_rx.try_recv().is_ok() {
					break;
				}
				match listener.accept() {
					Ok((mut stream, _)) => {
						let mut buf = Vec::new();
						let mut temp = [0u8; 4096];
						let mut header_pos = None;
						while header_pos.is_none() {
							match stream.read(&mut temp) {
								Ok(0) => break,
								Ok(n) => {
									buf.extend_from_slice(&temp[..n]);
									if let Some(pos) =
										buf.windows(4).position(|w| w == b"\r\n\r\n")
									{
										header_pos = Some(pos);
									}
								},
								Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
									thread::sleep(Duration::from_millis(5));
								},
								Err(_) => break,
							}
						}
						if let Some(pos) = header_pos {
							let len: usize = String::from_utf8_lossy(&buf[..pos])
								.lines()
								.find(|line| {
									line.to_ascii_lowercase().starts_with("content-length:")
								})
								.and_then(|line| line.split(':').nth(1))
								.and_then(|value| value.trim().parse().ok())
								.unwrap_or(0);
							while buf.len() < pos + 4 + len {
								match stream.read(&mut temp) {
									Ok(0) => break,
									Ok(n) => buf.extend_from_slice(&temp[..n]),
									Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
										thread::sleep(Duration::from_millis(5));
									},
									Err(_) => break,
								}
							}
							if buf.len() >= pos + 4 + len
								&& let Ok(json) =
									serde_json::from_slice::<serde_json::Value>(&buf[pos + 4..pos + 4 + len])
							{
								reqs_clone.lock().expect("lock").push(json);
							}
						}
						let index = reqs_clone.lock().expect("lock").len().saturating_sub(1);
						let body = bodies
							.get(index)
							.or_else(|| bodies.last())
							.map(String::as_str)
							.unwrap_or(DEFAULT_SSE);
						let response = format!(
							"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
							body.len(),
							body
						);
						let _ = stream.write_all(response.as_bytes());
						let _ = stream.flush();
					},
					Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
						thread::sleep(Duration::from_millis(10));
					},
					Err(_) => break,
				}
			}
		});

		Self { port, requests, stop_tx, handle: Some(handle) }
	}

	fn request_count(&self) -> usize {
		self.requests.lock().expect("lock").len()
	}

	fn request(&self, index: usize) -> serde_json::Value {
		self.requests
			.lock()
			.expect("lock")
			.get(index)
			.cloned()
			.unwrap_or_else(|| panic!("mock request {index} not recorded"))
	}

	fn last_request(&self) -> serde_json::Value {
		self.requests
			.lock()
			.expect("lock")
			.last()
			.cloned()
			.expect("mock request recorded")
	}

	fn stop(mut self) {
		let _ = self.stop_tx.send(());
		if let Some(handle) = self.handle.take() {
			let _ = handle.join();
		}
	}
}

fn setup(port: u16, base: &Path) -> (PathBuf, PathBuf, PathBuf) {
	let agent_dir = base.join("agent");
	let cwd = base.join("work");
	let sessions = base.join("sessions");
	fs::create_dir_all(&agent_dir).expect("agent dir");
	fs::create_dir_all(&cwd).expect("cwd");
	fs::create_dir_all(&sessions).expect("sessions dir");
	fs::write(
		agent_dir.join("config.yml"),
		format!(
			"defaultModel: mock/mock-model\nproviders:\n  mock:\n    baseUrl: \"http://127.0.0.1:{port}/v1\"\n    api: openai-completions\n    apiKey: test-key\n"
		),
	)
	.expect("config.yml");
	fs::write(
		agent_dir.join("models.yml"),
		format!(
			"providers:\n  mock:\n    baseUrl: \"http://127.0.0.1:{port}/v1\"\n    api: openai-completions\n    apiKey: test-key\n    models:\n      - id: mock-model\n        contextWindow: 128000\n"
		),
	)
	.expect("models.yml");
	(agent_dir, cwd, sessions)
}

fn omp(agent_dir: &Path, cwd: &Path, sessions: &Path) -> ProcCommand {
	let mut cmd = ProcCommand::new(env!("CARGO_BIN_EXE_omp"));
	cmd.env("PI_CODING_AGENT_DIR", agent_dir);
	cmd.args([
		"--cwd",
		cwd.to_str().expect("utf8 cwd"),
		"--session-dir",
		sessions.to_str().expect("utf8 sessions"),
		"--no-title",
	])
	.stdin(Stdio::null());
	cmd
}

fn session_file(sessions: &Path, id: &str) -> PathBuf {
	let mut files: Vec<PathBuf> = fs::read_dir(sessions)
		.expect("session dir")
		.flatten()
		.map(|entry| entry.path())
		.filter(|path| {
			path.extension().and_then(|ext| ext.to_str()) == Some("jsonl")
				&& path
					.file_stem()
					.and_then(|stem| stem.to_str())
					.is_some_and(|stem| stem.contains(id))
		})
		.collect();
	files.sort();
	assert_eq!(files.len(), 1, "expected one log for {id}: {files:?}");
	files.remove(0)
}

fn entries(path: &Path) -> Vec<serde_json::Value> {
	fs::read_to_string(path)
		.expect("session log")
		.lines()
		.filter(|line| !line.trim().is_empty())
		.map(|line| serde_json::from_str(line).expect("valid session jsonl"))
		.collect()
}

fn run(cmd: &mut ProcCommand) -> Output {
	cmd.output().expect("run omp")
}

/// Count literal prompt-token occurrences in a recorded provider request.
fn occurrences(body: &serde_json::Value, needle: &str) -> usize {
	serde_json::to_string(body)
		.expect("serialize request")
		.matches(needle)
		.count()
}

#[test]
fn headless_turn_persists_ordered_blocks_and_resume_replays_call_once() {
	let tool_sse = r#"data: {"choices":[{"delta":{"reasoning_content":"REASON-TOKEN-9f"},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"checking "},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"read","arguments":"{\"path\":\"note.txt\"}"}}]},"finish_reason":null}]}

data: [DONE]

"#;
	let server = MockWireServer::start(vec![
		tool_sse.to_string(),
		DEFAULT_SSE.to_string(),
		DEFAULT_SSE.to_string(),
	]);
	let base = temp_dir("ordered-blocks");
	let (agent_dir, cwd, sessions) = setup(server.port, &base);
	fs::write(cwd.join("note.txt"), "note body").expect("note");

	let first = run(
		omp(&agent_dir, &cwd, &sessions)
			.args(["--session-id", "replay-1", "-p", "PROMPT-TOKEN-3c read note.txt"]),
	);
	assert!(
		first.status.success(),
		"first headless turn failed: {}",
		String::from_utf8_lossy(&first.stderr)
	);
	assert_eq!(server.request_count(), 2, "one turn plus its tool follow-up");
	// The typed rebuild must describe the journal *before* this turn's user
	// entry is appended, otherwise the loop prompt lands a second time.
	assert_eq!(
		occurrences(&server.request(0), "PROMPT-TOKEN-3c"),
		1,
		"first request duplicated the current prompt"
	);
	assert_eq!(
		occurrences(&server.request(1), "PROMPT-TOKEN-3c"),
		1,
		"tool follow-up duplicated the current prompt"
	);

	let log = session_file(&sessions, "replay-1");
	let stored = entries(&log);
	let assistant = stored
		.iter()
		.find(|entry| entry["type"] == "message" && entry["message"]["role"] == "assistant")
		.expect("assistant entry");
	let parts = assistant["message"]["content"]
		.as_array()
		.expect("content must be ordered parts");
	let kinds: Vec<&str> = parts
		.iter()
		.map(|part| part["type"].as_str().unwrap_or_default())
		.collect();
	assert_eq!(
		kinds,
		["thinking", "text", "toolCall"],
		"streamed block order must persist as-is: {assistant}"
	);
	assert_eq!(
		parts
			.iter()
			.filter(|part| part["type"] == "toolCall")
			.count(),
		1,
		"the tool call must not be duplicated beside its ordered block: {assistant}"
	);
	assert_eq!(parts[2]["id"], "call-1");
	assert_eq!(parts[2]["name"], "read");
	assert!(parts[0]["thinking"].as_str().unwrap_or_default().contains("REASON-TOKEN-9f"));
	assert_eq!(assistant["message"]["api"], "openai-completions");
	assert_eq!(assistant["message"]["provider"], "mock");
	assert_eq!(assistant["message"]["model"], "mock-model");
	// The OpenAI adapter reports no terminal reason, so this is the event's
	// tool-use status, not a normalized `stop`.
	assert_eq!(assistant["message"]["stopReason"], "toolUse");
	assert!(
		assistant["message"].get("errorMessage").is_none(),
		"a clean stop must not carry an error message: {assistant}"
	);
	assert!(
		stored.iter().any(|entry| {
			entry["type"] == "message"
				&& entry["message"]["role"] == "toolResult"
				&& entry["message"]["toolCallId"] == "call-1"
		}),
		"tool result must follow its call: {stored:?}"
	);

	let second = run(
		omp(&agent_dir, &cwd, &sessions).args(["--resume", "replay-1", "-p", "RESUME-PROMPT-8a continue"]),
	);
	assert!(
		second.status.success(),
		"resumed headless turn failed: {}",
		String::from_utf8_lossy(&second.stderr)
	);

	let body = server.last_request();
	assert_eq!(
		occurrences(&body, "RESUME-PROMPT-8a"),
		1,
		"resume duplicated the new prompt: {body}"
	);
	assert_eq!(
		occurrences(&body, "PROMPT-TOKEN-3c"),
		1,
		"resume duplicated the replayed prompt: {body}"
	);
	let wire = serde_json::to_string(&body).expect("serialize request");
	assert_eq!(
		wire.matches("call-1").count(),
		2,
		"call id appears once as the call and once on its result: {body}"
	);
	let messages = body["messages"].as_array().expect("messages");
	let assistant = messages
		.iter()
		.find(|message| message["role"] == "assistant")
		.expect("replayed assistant");
	let calls = assistant["tool_calls"].as_array().expect("tool_calls");
	assert_eq!(calls.len(), 1, "resumed call must replay once: {assistant}");
	assert_eq!(calls[0]["id"], "call-1");
	assert!(
		messages
			.iter()
			.any(|message| message["role"] == "tool" && message["tool_call_id"] == "call-1"),
		"tool result must replay after the call: {body}"
	);

	// Resume must not downgrade the stored state to a text-only turn.
	let after = entries(&session_file(&sessions, "replay-1"));
	let assistant = after
		.iter()
		.find(|entry| entry["type"] == "message" && entry["message"]["role"] == "assistant")
		.expect("assistant entry after resume");
	assert_eq!(assistant["message"]["content"][0]["type"], "thinking");
	assert!(assistant.to_string().contains("REASON-TOKEN-9f"));

	server.stop();
	fs::remove_dir_all(base).ok();
}

#[test]
fn corrupt_stored_context_fails_before_any_provider_request() {
	let server = MockWireServer::start(vec![DEFAULT_SSE.to_string()]);
	let base = temp_dir("corrupt-context");
	let (agent_dir, cwd, sessions) = setup(server.port, &base);

	// A `toolCall` block without its name parses as a valid entry but cannot be
	// rebuilt losslessly, so the CLI must stop instead of replaying the turn.
	let log = sessions.join("2026-11-02T00-00-00-000Z_broken-context.jsonl");
	let lines = [
		serde_json::json!({
			"type": "session",
			"version": 3,
			"id": "broken-context",
			"timestamp": "2026-11-02T00:00:00.000Z",
			"cwd": cwd.to_str().expect("cwd"),
		})
		.to_string(),
		serde_json::json!({
			"type": "message",
			"id": "msg-u",
			"parentId": null,
			"timestamp": "2026-11-02T00:00:01.000Z",
			"message": {
				"role": "user",
				"content": [{ "type": "text", "text": "prior turn" }],
				"timestamp": 1000,
			},
		})
		.to_string(),
		serde_json::json!({
			"type": "message",
			"id": "msg-a",
			"parentId": "msg-u",
			"timestamp": "2026-11-02T00:00:02.000Z",
			"message": {
				"role": "assistant",
				"content": [{ "type": "toolCall", "id": "call-1", "arguments": { "path": "note.txt" } }],
				"api": "openai-completions",
				"provider": "mock",
				"model": "mock-model",
				"stopReason": "toolUse",
				"timestamp": 2000,
			},
		})
		.to_string(),
	];
	fs::write(&log, format!("{}\n", lines.join("\n"))).expect("write corrupt log");

	let output = run(
		omp(&agent_dir, &cwd, &sessions).args(["--resume", "broken-context", "-p", "CORRUPT-PROMPT-5e"]),
	);
	let stderr = String::from_utf8_lossy(&output.stderr);
	assert!(
		!output.status.success(),
		"a corrupt stored conversation must fail the turn: {stderr}"
	);
	assert!(
		stderr.contains("toolCall part missing name"),
		"failure must name the unreplayable block: {stderr}"
	);
	assert!(
		!stderr.contains("cannot rebuild branch context"),
		"the turn must not continue after a downgraded rebuild: {stderr}"
	);
	assert_eq!(
		server.request_count(),
		0,
		"the provider must never be called with degraded context"
	);
	assert!(
		!entries(&log)
			.iter()
			.any(|entry| entry.to_string().contains("CORRUPT-PROMPT-5e")),
		"the request must be built before the journal gains this turn's prompt"
	);

	server.stop();
	fs::remove_dir_all(base).ok();
}

#[tokio::test]
async fn rewind_rebuild_failure_aborts_before_another_provider_request() {
	// One real checkpoint+rewind tool pair in the first turn: the turn-end
	// handler applies the pending rewind, then cannot rebuild the branch.
	let tool_sse = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-cp","type":"function","function":{"name":"checkpoint","arguments":"{\"goal\":\"probe\"}"}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call-rw","type":"function","function":{"name":"rewind","arguments":"{\"report\":\"probe report\"}"}}]},"finish_reason":null}]}

data: [DONE]

"#;
	let server = MockWireServer::start(vec![tool_sse.to_string(), DEFAULT_SSE.to_string()]);
	let base = temp_dir("rewind-failure");
	let (agent_dir, cwd, sessions) = setup(server.port, &base);
	let mut config = fs::read_to_string(agent_dir.join("config.yml")).expect("read config");
	config.push_str("checkpoint:\n  enabled: true\n");
	fs::write(agent_dir.join("config.yml"), config).expect("enable checkpoint");

	let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_omp"));
	command
		.env("PI_CODING_AGENT_DIR", &agent_dir)
		.env("OMP_TEST_FAIL_REWIND_REBUILD", "1")
		.args([
			"--cwd",
			cwd.to_str().expect("utf8 cwd"),
			"--session-dir",
			sessions.to_str().expect("utf8 sessions"),
			"--no-title",
			"--yolo",
			"--session-id",
			"rewind-1",
			"-p",
			"checkpoint then rewind",
		])
		.stdin(std::process::Stdio::null());
	let output = tokio::time::timeout(Duration::from_secs(30), command.output())
		.await
		.expect("a failed rewind rebuild must not park the turn-end ack")
		.expect("run omp");
	let stderr = String::from_utf8_lossy(&output.stderr);
	assert!(
		!output.status.success(),
		"a failed rewind rebuild must fail the turn: {stderr}"
	);
	assert!(
		stderr.contains("cannot rebuild branch context after rewind"),
		"the failure must be reported: {stderr}"
	);
	assert!(
		stderr.contains("injected rewind rebuild failure"),
		"the debug injection must be the cause: {stderr}"
	);
	assert_eq!(
		server.request_count(),
		1,
		"no provider request may start after the failed rewind rebuild"
	);

	server.stop();
	fs::remove_dir_all(base).ok();
}
