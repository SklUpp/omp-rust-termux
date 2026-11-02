//! Consumer boundaries for the Ollama `/api/chat` provider:
//! - request body shape: Ollama history conversion, vision guard, schema
//!   sanitization, cloud history stripping, no-user demotion, options and
//!   forced tool selection;
//! - NDJSON framing across transport chunks, UTF-8 sequences and line
//!   boundaries, EOF tails, malformed/truncated records and done-reason errors;
//! - bounded HTTP error bodies, credential-free diagnostics and drop
//!   cancellation closing the transport.
//!
//! The fixtures are local TCP servers: no Ollama daemon, model or account is
//! required.

use std::time::Duration;

use futures::StreamExt;
use pi_agent::provider_ollama::{OllamaProvider, NON_VISION_IMAGE_PLACEHOLDER};
use pi_agent::state::{Attribution, StopReason};
use pi_agent::{
	ChunkStream, ContentPart, Message, ModelProvider, ProviderError, StreamChunk, ToolCall,
	ToolChoice,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn read_request(socket: &mut TcpStream) -> Option<(String, Value)> {
	let mut buf = Vec::new();
	let mut header_end = None;
	let mut tmp = [0u8; 1024];
	let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
	while header_end.is_none() && tokio::time::Instant::now() < deadline {
		let n = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut tmp))
			.await
			.ok()?
			.ok()?;
		if n == 0 {
			return None;
		}
		buf.extend_from_slice(&tmp[..n]);
		header_end = buf.windows(4).position(|w| w == b"\r\n\r\n");
	}
	let header_end = header_end?;
	let headers = String::from_utf8_lossy(&buf[..header_end]).into_owned();
	let content_length = headers
		.lines()
		.find_map(|line| {
			line.to_ascii_lowercase()
				.strip_prefix("content-length:")
				.map(|value| value.trim().parse::<usize>().unwrap_or(0))
		})
		.unwrap_or(0);
	let body_start = header_end + 4;
	let mut body = buf[body_start..].to_vec();
	while body.len() < content_length {
		let n = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut tmp))
			.await
			.ok()?
			.ok()?;
		if n == 0 {
			break;
		}
		body.extend_from_slice(&tmp[..n]);
	}
	Some((headers, serde_json::from_slice(&body).unwrap_or(Value::Null)))
}

fn response(status: &str, content_type: &str, body: &str) -> Vec<u8> {
	format!(
		"HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
		body.len()
	)
	.into_bytes()
}

async fn serve_once(response: Vec<u8>) -> (String, tokio::sync::oneshot::Receiver<(String, Value)>) {
	let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
	let addr = listener.local_addr().expect("addr");
	let (request_tx, request_rx) = tokio::sync::oneshot::channel();
	tokio::spawn(async move {
		let (mut socket, _) = listener.accept().await.expect("accept");
		if let Some(captured) = read_request(&mut socket).await {
			let _ = request_tx.send(captured);
		}
		let _ = socket.write_all(&response).await;
		let _ = socket.flush().await;
	});
	(format!("http://{addr}"), request_rx)
}

async fn serve_chunked(pieces: Vec<Vec<u8>>) -> String {
	let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
	let addr = listener.local_addr().expect("addr");
	tokio::spawn(async move {
		let (mut socket, _) = listener.accept().await.expect("accept");
		let _ = read_request(&mut socket).await;
		let headers = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
		let _ = socket.write_all(headers.as_bytes()).await;
		for piece in pieces {
			let mut framed = format!("{:x}\r\n", piece.len()).into_bytes();
			framed.extend_from_slice(&piece);
			framed.extend_from_slice(b"\r\n");
			if socket.write_all(&framed).await.is_err() {
				return;
			}
			let _ = socket.flush().await;
			tokio::time::sleep(Duration::from_millis(2)).await;
		}
		let _ = socket.write_all(b"0\r\n\r\n").await;
		let _ = socket.flush().await;
	});
	format!("http://{addr}")
}

fn make_provider(base: &str, model: &str) -> OllamaProvider {
	OllamaProvider::new(base, "test-key", model).expect("provider")
}

async fn drain(mut stream: ChunkStream) -> Result<Vec<StreamChunk>, ProviderError> {
	let mut chunks = Vec::new();
	while let Some(item) = stream.next().await {
		chunks.push(item?);
	}
	Ok(chunks)
}

async fn drain_body(records: &str) -> Vec<StreamChunk> {
	let base = serve_chunked(vec![records.as_bytes().to_vec()]).await;
	let provider = make_provider(&base, "m");
	drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect("chunks")
}

fn tool(name: &str, description: &str) -> Value {
	json!({
		"type": "function",
		"function": {
			"name": name,
			"description": description,
			"parameters": {
				"type": "object",
				"properties": {"path": {"type": "string"}},
				"additionalProperties": false
			}
		}
	})
}

#[tokio::test]
async fn request_body_matches_ollama_history_and_forced_tool_selection() {
	let (base, request_rx) = serve_once(response(
		"200 OK",
		"application/x-ndjson",
		"{\"message\":{\"content\":\"ok\"},\"done\":true,\"prompt_eval_count\":2,\"eval_count\":1}\n",
	))
	.await;

	let mut assistant = Message::assistant("");
	assistant.parts = vec![
		ContentPart::thinking("plan"),
		ContentPart::Text {
			text: "answer".into(),
		},
	];
	assistant.tool_calls = vec![ToolCall {
		id: "call_1".into(),
		name: "read".into(),
		arguments: json!({"path": "a"}),
	}];

	let mut image_user = Message::user_with_parts(vec![
		ContentPart::text("look"),
		ContentPart::image("aGVsbG8=", "image/png").expect("image"),
	]);
	image_user.content = "look".into();

	let messages = vec![
		Message::system("sys"),
		Message::user("hello"),
		assistant,
		Message::tool_with_status("call_1", "read", "file body", false),
		image_user,
	];
	let tools = vec![tool("read", "Read a file"), tool("write", "Write a file")];
	let choice = ToolChoice::Tool {
		name: "read".into(),
	};

	let provider = make_provider(&base, "llama3.2")
		.with_model_reasoning(true)
		.with_reasoning(Some("minimal".into()))
		.with_max_tokens(Some(70_000))
		.with_temperature(Some(0.2))
		.with_top_p(Some(0.9));
	let message = provider
		.complete(&messages, &tools, &choice)
		.await
		.expect("complete");
	assert_eq!(message.content, "ok");
	let usage = message.usage.expect("usage");
	assert_eq!((usage.input, usage.output, usage.total_tokens), (2, 1, 3));

	let (headers, body) = request_rx.await.expect("request");
	assert!(
		headers
			.to_ascii_lowercase()
			.contains("authorization: bearer test-key"),
		"{headers}"
	);
	assert_eq!(body["model"], "llama3.2");
	assert_eq!(body["stream"], true);
	assert_eq!(body["think"], "low");
	assert_eq!(body["tool_choice"], "required");
	assert_eq!(body["options"]["num_predict"], 70_000);
	assert_eq!(body["options"]["temperature"], 0.2);
	assert_eq!(body["options"]["top_p"], 0.9);
	assert!(!body.to_string().contains("test-key"));

	let wire_tools = body["tools"].as_array().expect("tools");
	assert_eq!(wire_tools.len(), 1, "forced choice narrows the offer");
	assert_eq!(wire_tools[0]["function"]["name"], "read");
	assert!(
		wire_tools[0]["function"]["parameters"]
			.get("additionalProperties")
			.is_none()
	);

	let wire_messages = body["messages"].as_array().expect("messages");
	assert_eq!(wire_messages.len(), 5);
	assert_eq!(wire_messages[0]["role"], "system");
	assert_eq!(wire_messages[0]["content"], "sys");
	assert_eq!(wire_messages[1]["role"], "user");
	assert_eq!(wire_messages[2]["role"], "assistant");
	assert_eq!(wire_messages[2]["content"], "answer");
	assert_eq!(wire_messages[2]["thinking"], "plan");
	assert_eq!(wire_messages[2]["tool_calls"][0]["function"]["name"], "read");
	assert_eq!(wire_messages[3]["role"], "tool");
	assert_eq!(wire_messages[3]["tool_name"], "read");
	assert_eq!(wire_messages[3]["content"], "file body");
	assert_eq!(wire_messages[4]["role"], "user");
	assert_eq!(
		wire_messages[4]["content"],
		format!("look\n{NON_VISION_IMAGE_PLACEHOLDER}")
	);
	assert!(wire_messages[4].get("images").is_none());
}

#[tokio::test]
async fn cloud_dialect_caps_tokens_strips_history_thinking_and_demotes_system() {
	let (base, request_rx) = serve_once(response(
		"200 OK",
		"application/x-ndjson",
		"{\"message\":{\"content\":\"ok\"},\"done\":true}\n",
	))
	.await;

	let mut assistant = Message::assistant("");
	assistant.parts = vec![ContentPart::thinking("private plan")];
	let messages = vec![
		Message::system("prefix"),
		assistant,
		Message::developer("reminder"),
	];

	let provider = make_provider(&base, "deepseek-v4-pro")
		.with_cloud(true)
		.with_model_reasoning(true)
		.with_reasoning(Some("minimal".into()))
		.with_max_tokens(Some(70_000));
	provider
		.complete(&messages, &[], &ToolChoice::Auto)
		.await
		.expect("complete");

	let (_, body) = request_rx.await.expect("request");
	assert_eq!(body["options"]["num_predict"], 65_536);
	assert_eq!(body["think"], "low");
	let wire = body["messages"].as_array().expect("messages");
	assert_eq!(wire[1]["role"], "assistant");
	assert!(wire[1].get("thinking").is_none(), "cloud drops history thinking");
	assert_eq!(wire[2]["role"], "user", "no-user request demotes the last system turn");
}

#[tokio::test]
async fn omit_max_output_tokens_and_disable_reasoning_omit_wire_fields() {
	let (base, request_rx) = serve_once(response(
		"200 OK",
		"application/x-ndjson",
		"{\"done\":true}\n",
	))
	.await;
	let provider = make_provider(&base, "m")
		.with_model_reasoning(true)
		.with_disable_reasoning(true)
		.with_max_tokens(Some(100))
		.with_omit_max_output_tokens(true);
	provider
		.complete(&[Message::user("hi")], &[], &ToolChoice::Auto)
		.await
		.expect("complete");
	let (_, body) = request_rx.await.expect("request");
	assert_eq!(body["think"], false);
	assert!(body.get("options").is_none(), "{body}");
}

#[tokio::test]
async fn forced_tool_choice_is_validated_before_the_socket_opens() {
	let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
	let addr = listener.local_addr().expect("addr");
	let provider = make_provider(&format!("http://{addr}"), "m");
	let err = provider
		.stream(
			&[Message::user("hi")],
			&[tool("read", "Read")],
			&ToolChoice::Tool {
				name: "missing".into(),
			},
		)
		.await
		.err()
		.expect("unoffered tool must fail before the request");
	match err {
		ProviderError::InvalidToolChoice(message) => {
			assert!(message.contains("not offered"), "{message}");
		},
		other => panic!("expected InvalidToolChoice, got {other:?}"),
	}
	assert!(
		tokio::time::timeout(Duration::from_millis(300), listener.accept())
			.await
			.is_err(),
		"validation must not open a connection"
	);
}

#[tokio::test]
async fn stream_handles_fragmented_utf8_text_thinking_tools_and_usage() {
	let records = concat!(
		"{\"message\":{\"thinking\":\"plan \"},\"done\":false}\n",
		"{\"message\":{\"thinking\":\"\u{2713}\"},\"done\":false}\n",
		"{\"message\":{\"content\":\"Hello\"},\"done\":false}\n",
		"{\"message\":{\"content\":\", world\"},\"done\":false}\n",
		"{\"message\":{\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}}]},\"done\":false}\n",
		"{\"message\":{\"content\":\"after\"},\"done\":false}\n",
		"{\"done\":true,\"done_reason\":\"tool_calls\",\"prompt_eval_count\":12,\"eval_count\":7}\n",
	);
	let pieces: Vec<Vec<u8>> = records
		.as_bytes()
		.chunks(7)
		.map(|piece| piece.to_vec())
		.collect();
	let base = serve_chunked(pieces).await;

	let provider = make_provider(&base, "llama3.2");
	let chunks = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect("chunks");

	let done = chunks.last().expect("done");
	assert!(done.is_done);
	assert_eq!(done.stop_reason, Some(StopReason::ToolUse));
	assert_eq!(done.error_message, None);
	let usage = done.usage.expect("usage");
	assert_eq!((usage.input, usage.output, usage.total_tokens), (12, 7, 19));

	let indices: Vec<Option<usize>> = chunks
		.iter()
		.filter(|chunk| !chunk.is_done)
		.map(|chunk| chunk.content_index)
		.collect();
	assert_eq!(
		indices,
		vec![Some(0), Some(0), Some(1), Some(1), Some(2), Some(3)],
		"thinking 0, text 1, tool 2, text after the tool 3"
	);

	let mut reasoning = String::new();
	let mut content = String::new();
	let mut deltas = Vec::new();
	for chunk in &chunks {
		if let Some(delta) = &chunk.reasoning {
			reasoning.push_str(delta);
		}
		if let Some(delta) = &chunk.content {
			content.push_str(delta);
		}
		deltas.extend(chunk.tool_call_deltas.iter().cloned());
	}
	assert_eq!(reasoning, "plan \u{2713}");
	assert_eq!(content, "Hello, worldafter");
	assert_eq!(deltas.len(), 1);
	assert_eq!(deltas[0].id.as_deref(), Some("ollama:2:read"));
	assert_eq!(deltas[0].name.as_deref(), Some("read"));
	assert_eq!(deltas[0].arguments_delta.as_deref(), Some("{\"path\":\"a\"}"));
	assert_eq!(deltas[0].content_index, Some(2));

	let mut assembler = pi_agent::StreamAssembler::new();
	for chunk in &chunks {
		assembler.push(chunk).expect("indexed chunk");
	}
	let message = assembler.finish().expect("assembled");
	assert_eq!(message.stop_reason, Some(StopReason::ToolUse));
	assert_eq!(message.content, "Hello, worldafter");
	assert_eq!(
		message.parts,
		vec![
			ContentPart::thinking("plan \u{2713}"),
			ContentPart::text("Hello, world"),
			ContentPart::ToolCall {
				id: "ollama:2:read".to_string(),
				name: "read".to_string(),
				arguments: json!({"path": "a"}),
			},
			ContentPart::text("after"),
		]
	);
	assert_eq!(message.tool_calls.len(), 1);
	assert_eq!(message.tool_calls[0].name, "read");
	assert_eq!(message.tool_calls[0].arguments, json!({"path": "a"}));
}

#[tokio::test]
async fn length_with_visible_content_keeps_text_and_reports_length() {
	let body = "{\"message\":{\"content\":\"partial\"},\"done\":true,\"done_reason\":\"length\",\"prompt_eval_count\":4,\"eval_count\":2}\n";
	let base = serve_chunked(vec![body.as_bytes().to_vec()]).await;
	let provider = make_provider(&base, "m");
	let chunks = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect("chunks");
	assert_eq!(chunks[0].content.as_deref(), Some("partial"));
	assert_eq!(chunks[0].content_index, Some(0));
	let done = chunks.last().expect("done");
	assert!(done.is_done);
	assert_eq!(done.stop_reason, Some(StopReason::Length));
	assert_eq!(done.error_message, None, "visible output is not an error");
	assert_eq!(done.usage.expect("usage").total_tokens, 6);

	let mut assembler = pi_agent::StreamAssembler::new();
	for chunk in &chunks {
		assembler.push(chunk).expect("indexed chunk");
	}
	let message = assembler.finish().expect("assembled");
	assert_eq!(message.content, "partial");
	assert_eq!(message.stop_reason, Some(StopReason::Length));
	assert_eq!(message.error_message, None);
}

#[tokio::test]
async fn stream_flushes_an_eof_record_without_a_trailing_newline() {
	let body = "{\"message\":{\"content\":\"tail\"},\"done\":false}";
	let base = serve_chunked(vec![body.as_bytes().to_vec()]).await;
	let provider = make_provider(&base, "m");
	let chunks = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect("chunks");
	assert_eq!(chunks.len(), 2);
	assert_eq!(chunks[0].content.as_deref(), Some("tail"));
	assert!(chunks[1].is_done);
	assert_eq!(chunks[1].stop_reason, Some(StopReason::Stop));
	assert_eq!(chunks[1].error_message, None);
	assert!(chunks[1].usage.is_none());
}

#[tokio::test]
async fn malformed_and_truncated_records_end_in_a_terminal_error_event() {
	let base = serve_chunked(vec![b"not-json\n".to_vec()]).await;
	let provider = make_provider(&base, "m");
	let chunks = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect("stream ends with an error event");
	let terminal = chunks.last().expect("terminal");
	assert!(terminal.is_done);
	assert_eq!(terminal.stop_reason, Some(StopReason::Error));

	let base = serve_chunked(vec![
		b"{\"message\":{\"content\":\"x\"},\"done\":false}\n".to_vec(),
		b"{\"message\":".to_vec(),
	])
	.await;
	let provider = make_provider(&base, "m");
	let chunks = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect("stream ends with an error event");
	assert_eq!(chunks.len(), 2, "partial output precedes the error event");
	assert_eq!(chunks[0].content.as_deref(), Some("x"));
	let terminal = chunks.last().expect("terminal");
	assert!(terminal.is_done);
	assert_eq!(terminal.stop_reason, Some(StopReason::Error));
}

#[tokio::test]
async fn load_done_reason_and_empty_length_emit_partial_terminal_errors() {
	const LOAD: &str = "Ollama loaded the model but generated nothing (done_reason: load): the request contained no user-role message.";
	const EMPTY_LENGTH: &str = "Model returned no content: prompt filled the context window; raise Ollama num_ctx or shorten the prompt.";

	let base = serve_chunked(vec![
		b"{\"message\":{\"content\":\"partial\"},\"done\":false}\n".to_vec(),
		b"{\"done\":true,\"done_reason\":\"load\",\"eval_count\":0}\n".to_vec(),
	])
	.await;
	let provider = make_provider(&base, "m");
	let chunks = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect("load is a terminal error event, not a transport failure");
	assert_eq!(chunks.len(), 2, "the partial content event survives");
	assert_eq!(chunks[0].content.as_deref(), Some("partial"));
	let terminal = chunks.last().expect("terminal");
	assert!(terminal.is_done);
	assert_eq!(terminal.stop_reason, Some(StopReason::Error));
	assert_eq!(terminal.error_message.as_deref(), Some(LOAD));
	assert_eq!(terminal.usage.expect("usage").total_tokens, 0);

	let mut assembler = pi_agent::StreamAssembler::new();
	for chunk in &chunks {
		assembler.push(chunk).expect("indexed chunk");
	}
	let message = assembler.finish().expect("assembled");
	assert_eq!(message.content, "partial");
	assert_eq!(message.stop_reason, Some(StopReason::Error));
	assert_eq!(message.error_message.as_deref(), Some(LOAD));

	let base = serve_chunked(vec![
		b"{\"done\":true,\"done_reason\":\"length\",\"eval_count\":0}\n".to_vec(),
	])
	.await;
	let provider = make_provider(&base, "m");
	let chunks = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect("empty length is a terminal error event");
	let terminal = chunks.last().expect("terminal");
	assert!(terminal.is_done);
	assert_eq!(terminal.stop_reason, Some(StopReason::Error));
	assert_eq!(terminal.error_message.as_deref(), Some(EMPTY_LENGTH));
}

#[tokio::test]
async fn complete_returns_the_partial_message_with_a_terminal_error() {
	let body = concat!(
		"{\"message\":{\"content\":\"partial\"},\"done\":false}\n",
		"{\"done\":true,\"done_reason\":\"load\",\"eval_count\":0}\n",
	);
	let (base, _request_rx) = serve_once(response("200 OK", "application/x-ndjson", body)).await;
	let provider = make_provider(&base, "m");
	let message = provider
		.complete(&[Message::user("hi")], &[], &ToolChoice::Auto)
		.await
		.expect("complete returns the partial message, like TS");
	assert_eq!(message.content, "partial");
	assert_eq!(message.stop_reason, Some(StopReason::Error));
	assert_eq!(
		message.error_message.as_deref(),
		Some(
			"Ollama loaded the model but generated nothing (done_reason: load): the request contained no user-role message."
		)
	);
	assert_eq!(message.api.as_deref(), Some("ollama-chat"));
	assert_eq!(message.provider.as_deref(), Some("ollama"));
	assert_eq!(message.model.as_deref(), Some("m"));
}

#[tokio::test]
async fn none_tool_choice_sends_the_wire_mode_and_keeps_offered_tools() {
	let (base, request_rx) = serve_once(response(
		"200 OK",
		"application/x-ndjson",
		"{\"message\":{\"content\":\"ok\"},\"done\":true}\n",
	))
	.await;
	let provider = make_provider(&base, "m");
	provider
		.complete(
			&[Message::user("hi")],
			&[tool("read", "Read")],
			&ToolChoice::None,
		)
		.await
		.expect("complete");
	let (_, body) = request_rx.await.expect("request");
	assert_eq!(body["tool_choice"], "none");
	assert_eq!(body["tools"].as_array().expect("tools").len(), 1);
}

#[tokio::test]
async fn developer_attribution_selects_system_or_user_role() {
	let (base, request_rx) = serve_once(response(
		"200 OK",
		"application/x-ndjson",
		"{\"message\":{\"content\":\"ok\"},\"done\":true}\n",
	))
	.await;
	let mut user_turn = Message::developer("captured user note");
	user_turn.attribution = Some(Attribution::User);
	let mut agent_turn = Message::developer("agent reminder");
	agent_turn.attribution = Some(Attribution::Agent);
	let plain_developer = Message::developer("no attribution");
	let messages = vec![
		Message::system("prefix"),
		user_turn,
		agent_turn,
		plain_developer,
		Message::user("hello"),
	];
	make_provider(&base, "m")
		.complete(&messages, &[], &ToolChoice::Auto)
		.await
		.expect("complete");
	let (_, body) = request_rx.await.expect("request");
	let roles: Vec<&str> = body["messages"]
		.as_array()
		.expect("messages")
		.iter()
		.map(|message| message["role"].as_str().expect("role"))
		.collect();
	assert_eq!(roles, vec!["system", "user", "system", "system", "user"]);
}

#[tokio::test]
async fn completed_message_carries_the_true_source_metadata() {
	let (base, _request_rx) = serve_once(response(
		"200 OK",
		"application/x-ndjson",
		"{\"message\":{\"content\":\"ok\"},\"done\":true}\n",
	))
	.await;
	let provider = make_provider(&base, "llama3.2");
	let message = provider
		.complete(&[Message::user("hi")], &[], &ToolChoice::Auto)
		.await
		.expect("complete");
	assert_eq!(message.api.as_deref(), Some("ollama-chat"));
	assert_eq!(message.provider.as_deref(), Some("ollama"));
	assert_eq!(message.model.as_deref(), Some("llama3.2"));
	assert_eq!(provider.api(), Some("ollama-chat"));
	assert_eq!(provider.provider(), Some("ollama"));
	assert_eq!(provider.model(), Some("llama3.2"));

	let cloud = make_provider(&base, "llama3.2").with_cloud(true);
	assert_eq!(cloud.provider(), Some("ollama-cloud"));
}

#[tokio::test]
async fn explicit_stop_with_tool_calls_promotes_to_tool_use() {
	let without_reason = drain_body(
		"{\"message\":{\"tool_calls\":[{\"function\":{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}}]},\"done\":false}\n{\"done\":true}\n",
	)
	.await;
	assert_eq!(
		without_reason.last().expect("terminal").stop_reason,
		Some(StopReason::ToolUse)
	);

	let explicit_stop = drain_body(
		"{\"message\":{\"tool_calls\":[{\"function\":{\"name\":\"read\",\"arguments\":{}}}]},\"done\":false}\n{\"done\":true,\"done_reason\":\"stop\"}\n",
	)
	.await;
	assert_eq!(
		explicit_stop.last().expect("terminal").stop_reason,
		Some(StopReason::ToolUse)
	);

	let length_with_calls = drain_body(
		"{\"message\":{\"tool_calls\":[{\"function\":{\"name\":\"read\",\"arguments\":{}}}]},\"done\":false}\n{\"done\":true,\"done_reason\":\"length\"}\n",
	)
	.await;
	assert_eq!(
		length_with_calls.last().expect("terminal").stop_reason,
		Some(StopReason::Length)
	);
}

#[tokio::test]
async fn http_error_body_is_bounded_and_redacts_the_effective_bearer_token() {
	let key = "sk-ollama-secret-token";
	let mut oversized = format!("gateway echoed {key} ");
	// The read ceiling lands inside this run; the trailing credential must not
	// survive as a prefix of the cut token.
	oversized.push_str(&"a".repeat(70 * 1024));
	oversized.push_str(key);
	let (base, request_rx) = serve_once(response("500 Internal Server Error", "text/plain", &oversized)).await;
	let provider = OllamaProvider::new(base.as_str(), key, "m").expect("provider");
	let err = provider
		.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
		.await
		.err()
		.expect("HTTP failure");
	let (_, body) = request_rx.await.expect("request");
	assert_eq!(body["model"], "m");
	match &err {
		ProviderError::HttpStatus(status, body) => {
			assert_eq!(*status, 500);
			assert!(body.len() < 70 * 1024, "error body must be bounded: {}", body.len());
			assert!(body.contains("[redacted]"), "{body}");
			assert!(body.contains("[truncated]"), "{body}");
			assert!(!body.contains(key), "{body}");
			assert!(!body.contains("aaaaaaaaaa"), "cut token prefix leaked: {body}");
		},
		other => panic!("expected HttpStatus, got {other:?}"),
	}
	assert!(!err.to_string().contains(key));
}

#[tokio::test]
async fn malformed_record_diagnostics_redact_the_bearer_token() {
	let key = "sk-ollama-secret-token";
	let record = format!("not-json echoed {key}\n");
	let base = serve_chunked(vec![record.into_bytes()]).await;
	let provider = OllamaProvider::new(base.as_str(), key, "m").expect("provider");
	let chunks = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect("chunks");
	let terminal = chunks.last().expect("terminal");
	assert_eq!(terminal.stop_reason, Some(StopReason::Error));
	let message = terminal.error_message.clone().expect("error message");
	assert!(message.contains("[redacted]"), "{message}");
	assert!(!message.contains(key), "{message}");
}

#[tokio::test]
async fn oversized_ndjson_record_fails_instead_of_buffering() {
	let huge = vec![b'a'; 8 * 1024 * 1024 + 16];
	let base = serve_chunked(vec![huge]).await;
	let provider = make_provider(&base, "m");
	let chunks = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect("chunks");
	let terminal = chunks.last().expect("terminal");
	assert_eq!(terminal.stop_reason, Some(StopReason::Error));
	assert!(
		terminal
			.error_message
			.as_deref()
			.unwrap_or_default()
			.contains("exceeded"),
		"{terminal:?}"
	);
}

#[tokio::test]
async fn large_but_legitimate_record_is_not_truncated() {
	let text = "z".repeat(1024 * 1024);
	let records = format!(
		"{{\"message\":{{\"content\":\"{text}\"}},\"done\":false}}\n{{\"done\":true}}\n"
	);
	let chunks = drain_body(&records).await;
	let streamed: String = chunks
		.iter()
		.filter_map(|chunk| chunk.content.as_deref())
		.collect();
	assert_eq!(streamed.len(), text.len(), "large record must not be truncated");
	assert_eq!(streamed, text);
}

#[tokio::test]
async fn dropping_a_live_stream_closes_the_transport() {
	let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
	let addr = listener.local_addr().expect("addr");
	let (closed_tx, closed_rx) = tokio::sync::oneshot::channel();
	tokio::spawn(async move {
		let (mut socket, _) = listener.accept().await.expect("accept");
		let _ = read_request(&mut socket).await;
		let headers = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\n\r\n";
		let _ = socket.write_all(headers.as_bytes()).await;
		let record = b"{\"message\":{\"content\":\"first\"},\"done\":false}\n";
		let mut framed = format!("{:x}\r\n", record.len()).into_bytes();
		framed.extend_from_slice(record);
		framed.extend_from_slice(b"\r\n");
		let _ = socket.write_all(&framed).await;
		let _ = socket.flush().await;
		let mut tmp = [0u8; 64];
		let _ = socket.read(&mut tmp).await;
		let _ = closed_tx.send(());
	});

	let provider = make_provider(&format!("http://{addr}"), "m");
	let mut stream = provider
		.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
		.await
		.expect("stream");
	let first = stream.next().await.expect("chunk").expect("ok");
	assert_eq!(first.content.as_deref(), Some("first"));
	drop(stream);

	tokio::time::timeout(Duration::from_secs(2), closed_rx)
		.await
		.expect("server must observe the closed stream")
		.expect("closed signal");
}

#[tokio::test]
async fn completed_record_cap_rejects_a_newline_bearing_chunk() {
	let mut record = vec![b'a'; 8 * 1024 * 1024 + 16];
	record.push(b'\n');
	let base = serve_chunked(vec![record]).await;
	let provider = make_provider(&base, "m");
	let chunks = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect("chunks");
	let terminal = chunks.last().expect("terminal");
	assert_eq!(terminal.stop_reason, Some(StopReason::Error));
	assert!(
		terminal
			.error_message
			.as_deref()
			.unwrap_or_default()
			.contains("exceeded"),
		"{terminal:?}"
	);
}

#[tokio::test]
async fn custom_provider_id_is_stamped_and_default_is_unchanged() {
	let records = "{\"message\":{\"content\":\"hi\"},\"done\":false}\n{\"done\":true}\n";
	let base = serve_chunked(vec![records.as_bytes().to_vec()]).await;
	let provider = make_provider(&base, "m").with_provider_id(Some("custom-gateway".to_string()));
	let message = provider
		.complete(&[Message::user("go")], &[], &ToolChoice::Auto)
		.await
		.expect("complete");
	assert_eq!(message.provider.as_deref(), Some("custom-gateway"));
	assert_eq!(message.api.as_deref(), Some("ollama-chat"));
	assert_eq!(message.model.as_deref(), Some("m"));

	let local = OllamaProvider::new("http://127.0.0.1:9", "", "m").expect("provider");
	assert_eq!(local.provider(), Some("ollama"));
	assert_eq!(
		local.with_provider_id(Some("   ".to_string())).provider(),
		Some("ollama"),
		"an empty override keeps the dialect default"
	);
	let cloud = OllamaProvider::new("http://127.0.0.1:9", "", "m").expect("provider");
	assert_eq!(cloud.with_cloud(true).provider(), Some("ollama-cloud"));
	let custom = OllamaProvider::new("http://127.0.0.1:9", "", "m").expect("provider");
	assert_eq!(
		custom
			.with_cloud(true)
			.with_provider_id(Some("gateway-7".to_string()))
			.provider(),
		Some("gateway-7")
	);
}
