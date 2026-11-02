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
		.expect_err("unoffered tool must fail before the request");
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
	let usage = done.usage.expect("usage");
	assert_eq!((usage.input, usage.output, usage.total_tokens), (12, 7, 19));

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

	let mut assembler = pi_agent::StreamAssembler::new();
	for chunk in &chunks {
		assembler.push(chunk);
	}
	let message = assembler.finish().expect("assembled");
	assert_eq!(message.content, "Hello, worldafter");
	assert_eq!(message.parts.len(), 3);
	assert_eq!(message.parts[0], ContentPart::thinking("plan \u{2713}"));
	assert_eq!(message.tool_calls.len(), 1);
	assert_eq!(message.tool_calls[0].name, "read");
	assert_eq!(message.tool_calls[0].arguments, json!({"path": "a"}));
}

#[tokio::test]
async fn length_with_visible_content_keeps_text_and_reaches_the_shared_boundary() {
	// TS sets `stopReason: "length"` here. `StreamChunk` has no stop field yet,
	// so this fixture pins the adapter-visible behavior only: text survives and
	// the terminal chunk is still marked done. The exact reason is computed in
	// `map_done_reason` and reported to the parent as a seam gap.
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
	let done = chunks.last().expect("done");
	assert!(done.is_done);
	assert_eq!(done.usage.expect("usage").total_tokens, 6);
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
	assert!(chunks[1].usage.is_none());
}

#[tokio::test]
async fn malformed_and_truncated_records_are_protocol_errors() {
	let base = serve_chunked(vec![b"not-json\n".to_vec()]).await;
	let provider = make_provider(&base, "m");
	let err = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect_err("malformed record");
	assert!(matches!(&err, ProviderError::Protocol(_)), "{err:?}");

	let base = serve_chunked(vec![
		b"{\"message\":{\"content\":\"x\"},\"done\":false}\n".to_vec(),
		b"{\"message\":".to_vec(),
	])
	.await;
	let provider = make_provider(&base, "m");
	let err = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect_err("truncated tail");
	assert!(matches!(&err, ProviderError::Protocol(_)), "{err:?}");
}

#[tokio::test]
async fn load_done_reason_and_empty_length_surface_as_errors() {
	let base = serve_chunked(vec![
		b"{\"done\":true,\"done_reason\":\"load\",\"eval_count\":0}\n".to_vec(),
	])
	.await;
	let provider = make_provider(&base, "m");
	let err = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect_err("load is an error");
	assert!(matches!(&err, ProviderError::Protocol(_)), "{err:?}");
	assert!(err.to_string().contains("no user-role message"), "{err}");

	let base = serve_chunked(vec![
		b"{\"done\":true,\"done_reason\":\"length\",\"eval_count\":0}\n".to_vec(),
	])
	.await;
	let provider = make_provider(&base, "m");
	let err = drain(
		provider
			.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
			.await
			.expect("stream"),
	)
	.await
	.expect_err("empty length is an error");
	assert!(err.to_string().contains("no content"), "{err}");
}

#[tokio::test]
async fn http_error_body_is_bounded_and_does_not_echo_the_api_key() {
	let oversized = "x".repeat(3000);
	let (base, request_rx) = serve_once(response("500 Internal Server Error", "text/plain", &oversized)).await;
	let provider = make_provider(&base, "m");
	let err = provider
		.stream(&[Message::user("hi")], &[], &ToolChoice::Auto)
		.await
		.expect_err("HTTP failure");
	let (_, body) = request_rx.await.expect("request");
	assert_eq!(body["model"], "m");
	match &err {
		ProviderError::HttpStatus(status, body) => {
			assert_eq!(*status, 500);
			assert_eq!(body.chars().count(), 2048);
		},
		other => panic!("expected HttpStatus, got {other:?}"),
	}
	assert!(!err.to_string().contains("test-key"));
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
