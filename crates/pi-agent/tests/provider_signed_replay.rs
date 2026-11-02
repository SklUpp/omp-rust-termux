//! Consumer-visible signed/redacted reasoning and terminal-reason parity.
//!
//! These tests drive the public assembler and serde surface the way the CLI
//! does across two turns: stream (with explicit block indices), finalize,
//! serialize into a session, deserialize on resume, then build the next
//! provider request.

use pi_agent::{
	ContentPart, Message, ProviderError, StopReason, StreamAssembler, StreamChunk, ToolCallDelta,
	ToolChoice, to_openai_messages,
};

fn chunk(
	content_index: Option<usize>,
	reasoning: Option<&str>,
	content: Option<&str>,
	signature: Option<&str>,
	redacted: Option<&str>,
	tool_call_deltas: Vec<ToolCallDelta>,
) -> StreamChunk {
	StreamChunk {
		content: content.map(str::to_string),
		reasoning: reasoning.map(str::to_string),
		tool_call_deltas,
		usage: None,
		is_done: false,
		content_index,
		signature: signature.map(str::to_string),
		redacted: redacted.map(str::to_string),
		stop_reason: None,
		error_message: None,
	}
}

fn done(stop_reason: StopReason, error_message: Option<&str>) -> StreamChunk {
	StreamChunk {
		content: None,
		reasoning: None,
		tool_call_deltas: Vec::new(),
		usage: None,
		is_done: true,
		content_index: None,
		signature: None,
		redacted: None,
		stop_reason: Some(stop_reason),
		error_message: error_message.map(str::to_string),
	}
}

#[test]
fn signed_thinking_tool_boundary_and_length_survive_two_turn_replay() {
	let mut assembler = StreamAssembler::new();
	assembler
		.push(&chunk(Some(0), Some("plan "), None, None, None, Vec::new()))
		.expect("thinking delta");
	assembler
		.push(&chunk(Some(0), None, None, Some("sig-turn-1"), None, Vec::new()))
		.expect("signature delta");
	assembler
		.push(&chunk(Some(1), None, Some("answer "), None, None, Vec::new()))
		.expect("text delta");
	assembler
		.push(&chunk(
			Some(2),
			None,
			None,
			None,
			None,
			vec![ToolCallDelta {
				index: 0,
				id: Some("call_1".to_string()),
				name: Some("read".to_string()),
				arguments_delta: Some("{\"path\":\"a\"}".to_string()),
				content_index: Some(2),
			}],
		))
		.expect("tool delta");
	assembler
		.push(&done(StopReason::Length, None))
		.expect("done");

	let message = assembler.finish().expect("assembled turn");
	assert_eq!(message.stop_reason, Some(StopReason::Length));
	assert_eq!(message.tool_calls.len(), 1);
	assert_eq!(message.tool_calls[0].name, "read");
	assert_eq!(
		message
			.parts
			.iter()
			.map(|part| match part {
				ContentPart::Thinking { .. } => "thinking",
				ContentPart::Text { .. } => "text",
				ContentPart::ToolCall { .. } => "toolCall",
				ContentPart::RedactedThinking { .. } => "redacted",
				ContentPart::Image { .. } => "image",
			})
			.collect::<Vec<_>>(),
		["thinking", "text", "toolCall"],
		"tool-call block keeps its streamed position"
	);
	assert_eq!(message.parts[0].thinking_signature(), Some("sig-turn-1"));

	// Persistence: session serialization must round-trip the signature and the
	// truncation reason, not fold it into a clean stop.
	let json = serde_json::to_string(&message).expect("serialize message");
	assert!(json.contains("\"thinkingSignature\":\"sig-turn-1\""));
	assert!(json.contains("\"stopReason\":\"length\""));
	let resumed: Message = serde_json::from_str(&json).expect("deserialize message");
	assert_eq!(resumed, message);

	// Turn 2: resuming history and building an OpenAI-compatible request strips
	// signed reasoning (it is only valid where it was issued) but keeps the call.
	let request = to_openai_messages(&[Message::user("continue"), resumed]).expect("wire");
	let wire = serde_json::to_string(&request).expect("serialize wire");
	assert!(!wire.contains("sig-turn-1"));
	assert!(!wire.contains("thinkingSignature"));
	assert!(wire.contains("\"tool_calls\""));
}

#[test]
fn repeated_thinking_blocks_keep_their_own_signatures() {
	let mut assembler = StreamAssembler::new();
	assembler
		.push(&chunk(Some(0), Some("first"), None, Some("sig-a"), None, Vec::new()))
		.expect("first thinking");
	assembler
		.push(&chunk(Some(1), None, Some("between"), None, None, Vec::new()))
		.expect("text");
	assembler
		.push(&chunk(Some(2), Some("second"), None, Some("sig-b"), None, Vec::new()))
		.expect("second thinking");

	let message = assembler.finish().expect("assembled turn");
	assert_eq!(message.parts.len(), 3);
	assert_eq!(message.parts[0].as_thinking(), Some("first"));
	assert_eq!(message.parts[0].thinking_signature(), Some("sig-a"));
	assert_eq!(message.parts[2].as_thinking(), Some("second"));
	assert_eq!(message.parts[2].thinking_signature(), Some("sig-b"));
}

#[test]
fn redacted_reasoning_and_error_terminal_survive_roundtrip() {
	let mut assembler = StreamAssembler::new();
	assembler
		.push(&chunk(Some(0), None, None, None, Some("opaque-blob"), Vec::new()))
		.expect("redacted block");
	assembler
		.push(&done(StopReason::Error, Some("refused by safety")))
		.expect("error done");

	let message = assembler.finish().expect("assembled turn");
	assert_eq!(message.stop_reason, Some(StopReason::Error));
	assert_eq!(message.error_message.as_deref(), Some("refused by safety"));
	assert_eq!(message.parts[0].as_redacted_thinking(), Some("opaque-blob"));
	assert_eq!(message.stop_reason.map(StopReason::as_str), Some("error"));

	let json = serde_json::to_string(&message).expect("serialize message");
	assert!(json.contains("\"type\":\"redactedThinking\""));
	assert!(json.contains("\"errorMessage\":\"refused by safety\""));
	let resumed: Message = serde_json::from_str(&json).expect("deserialize message");
	assert_eq!(resumed, message);
}

#[test]
fn signature_for_an_unknown_block_is_a_protocol_error() {
	let mut assembler = StreamAssembler::new();
	let err = assembler
		.push(&chunk(Some(3), None, None, Some("orphan"), None, Vec::new()))
		.expect_err("signature without its thinking block must fail");
	assert!(matches!(err, ProviderError::Protocol(_)));
}

#[test]
fn tool_choice_none_stays_distinct_from_auto() {
	assert_ne!(ToolChoice::None, ToolChoice::Auto);
}
