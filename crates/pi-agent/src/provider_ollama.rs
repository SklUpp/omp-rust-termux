//! Ollama `/api/chat` provider (`packages/ai/src/providers/ollama.ts`).
//!
//! Native NDJSON streaming, Ollama-shaped history conversion and the
//! vision/cloud quirks of the TypeScript source. The caller owns cancellation:
//! dropping the returned stream drops the HTTP response body and closes the
//! connection, exactly like the other provider adapters in this crate.

use std::{
	collections::{BTreeMap, HashMap, VecDeque},
	fmt::Display,
	time::Duration,
};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
	provider::{
		ChunkStream, ModelProvider, ProviderError, StreamAssembler, StreamChunk, ToolCallDelta,
		ToolChoice,
	},
	state::{
		Attribution, ContentPart, Message, Role, StopReason, TokenUsage, normalize_image_payload,
	},
};

/// TCP/TLS connect budget; a live NDJSON completion owns no total timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Maximum characters of an error body or malformed record kept for diagnostics.
const ERROR_BODY_CHARS: usize = 2048;
/// Hard ceiling on bytes read from an HTTP error body before diagnostics.
const ERROR_BODY_BYTES: usize = 64 * 1024;
/// Hard ceiling on one NDJSON record; overflow is a terminal error, never a
/// silently dropped or truncated success.
const MAX_RECORD_BYTES: usize = 8 * 1024 * 1024;
/// Ollama Cloud rejects `num_predict` above this value with HTTP 400.
const OLLAMA_CLOUD_NUM_PREDICT_CAP: u64 = 65_536;
/// `vision-guard.ts` placeholder for an image sent to a text-only model.
pub const NON_VISION_IMAGE_PLACEHOLDER: &str = "[image omitted: model does not support vision]";
const EMPTY_OLLAMA_LENGTH_COMPLETION_MESSAGE: &str =
	"Model returned no content: prompt filled the context window; raise Ollama num_ctx or shorten the prompt.";
const EMPTY_OLLAMA_LOAD_COMPLETION_MESSAGE: &str =
	"Ollama loaded the model but generated nothing (done_reason: load): the request contained no user-role message.";

/// `normalizeBaseUrl`: blank means Ollama Cloud; one trailing slash and a
/// trailing `/api` are both stripped so [`Self::endpoint`] can append once.
fn normalize_base_url(base_url: &str) -> String {
	let value = base_url.trim();
	if value.is_empty() {
		return "https://ollama.com".to_string();
	}
	let trimmed = value.strip_suffix('/').unwrap_or(value);
	trimmed.strip_suffix("/api").unwrap_or(trimmed).to_string()
}

fn validate_base_url(base_url: &str) -> Result<(), ProviderError> {
	let parsed = reqwest::Url::parse(base_url)
		.map_err(|err| ProviderError::InvalidBaseUrl(err.to_string()))?;
	if parsed.scheme() != "http" && parsed.scheme() != "https" {
		return Err(ProviderError::InvalidBaseUrl(format!(
			"unsupported scheme `{}`",
			parsed.scheme()
		)));
	}
	Ok(())
}

fn network_error(err: reqwest::Error) -> ProviderError {
	ProviderError::Network(err.to_string())
}

/// Bearer token(s) this provider can put on the wire.
fn credential_values(api_key: Option<&str>) -> Vec<String> {
	api_key
		.map(str::trim)
		.filter(|value| !value.is_empty())
		.map(|value| vec![value.to_string()])
		.unwrap_or_default()
}

fn redact_with(text: &str, secrets: &[String]) -> String {
	let mut redacted = text.to_string();
	for secret in secrets {
		redacted = redacted.replace(secret.as_str(), "[redacted]");
	}
	redacted
}

/// Redact before bounding so a credential cut by the character limit cannot
/// survive as a prefix; `truncated` marks a byte-capped read.
fn bounded_redaction(text: &str, secrets: &[String], truncated: bool) -> String {
	let mut bounded: String = redact_with(text, secrets)
		.chars()
		.take(ERROR_BODY_CHARS)
		.collect();
	if truncated {
		bounded.push_str("\n[truncated]");
	}
	bounded
}

/// Drop the incomplete trailing token of a byte-capped read: a credential cut
/// by the ceiling would otherwise survive as its prefix.
fn trim_partial_token(text: &mut String) {
	let boundary = text
		.char_indices()
		.rev()
		.find(|(_, ch)| {
			ch.is_whitespace() || matches!(ch, '"' | '\'' | ',' | ':' | '{' | '}' | '[' | ']')
		})
		.map(|(index, ch)| index + ch.len_utf8());
	if let Some(boundary) = boundary
		&& boundary < text.len()
	{
		text.truncate(boundary);
	}
}

struct BoundedBody {
	text: String,
	truncated: bool,
}

/// Read at most `limit` body bytes; never grows past the ceiling.
async fn read_body_bounded(
	response: reqwest::Response,
	limit: usize,
) -> Result<BoundedBody, ProviderError> {
	let mut stream = response.bytes_stream();
	let mut bytes: Vec<u8> = Vec::new();
	let mut truncated = false;
	while let Some(item) = stream.next().await {
		let chunk = item.map_err(network_error)?;
		if bytes.len() + chunk.len() > limit {
			let room = limit.saturating_sub(bytes.len());
			bytes.extend_from_slice(&chunk[..room]);
			truncated = true;
			break;
		}
		bytes.extend_from_slice(&chunk);
	}
	let mut text = String::from_utf8_lossy(&bytes).into_owned();
	if truncated {
		trim_partial_token(&mut text);
	}
	Ok(BoundedBody { text, truncated })
}

/// Ollama `think` value: `false` disables reasoning, an effort string selects it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
enum OllamaThink {
	Bool(bool),
	Effort(String),
}

#[derive(Debug, Clone, Serialize)]
struct OllamaFunctionTool {
	#[serde(rename = "type")]
	kind: &'static str,
	function: OllamaFunction,
}

#[derive(Debug, Clone, Serialize)]
struct OllamaFunction {
	name: String,
	description: String,
	parameters: Value,
}

#[derive(Debug, Clone, Serialize)]
struct OllamaWireToolCall {
	#[serde(rename = "type")]
	kind: &'static str,
	function: OllamaWireFunction,
}

#[derive(Debug, Clone, Serialize)]
struct OllamaWireFunction {
	name: String,
	arguments: Value,
}

#[derive(Debug, Clone, Serialize)]
struct OllamaWireMessage {
	role: &'static str,
	content: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	images: Option<Vec<String>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	thinking: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_calls: Option<Vec<OllamaWireToolCall>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_name: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct OllamaOptions {
	#[serde(skip_serializing_if = "Option::is_none")]
	num_predict: Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	temperature: Option<f64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	top_p: Option<f64>,
}

/// `createChatBody` wire shape, field order included.
#[derive(Debug, Clone, Serialize)]
struct OllamaChatRequest {
	model: String,
	messages: Vec<OllamaWireMessage>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tools: Option<Vec<OllamaFunctionTool>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	think: Option<OllamaThink>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_choice: Option<&'static str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	options: Option<OllamaOptions>,
	stream: bool,
}

/// One NDJSON record from `/api/chat`.
#[derive(Debug, Clone, Default, Deserialize)]
struct OllamaChatChunk {
	#[serde(default)]
	message: Option<OllamaChunkMessage>,
	#[serde(default)]
	done: Option<bool>,
	#[serde(default)]
	done_reason: Option<String>,
	#[serde(default)]
	prompt_eval_count: Option<u64>,
	#[serde(default)]
	eval_count: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct OllamaChunkMessage {
	#[serde(default)]
	content: Option<String>,
	#[serde(default)]
	thinking: Option<String>,
	#[serde(default)]
	tool_calls: Option<Vec<OllamaChunkToolCall>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct OllamaChunkToolCall {
	#[serde(default)]
	function: Option<OllamaChunkFunction>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct OllamaChunkFunction {
	#[serde(default)]
	name: Option<String>,
	#[serde(default)]
	arguments: Option<Value>,
}

/// Ollama `/api/chat` model provider.
///
/// Base URL, key and model are caller-supplied. The remaining knobs mirror the
/// TS `Model<"ollama-chat">`/`OllamaChatOptions` fields that influence the wire
/// body; the Rust [`ModelProvider`] trait has no per-request options.
pub struct OllamaProvider {
	http: reqwest::Client,
	base_url: String,
	api_key: Option<String>,
	model: String,
	cloud: bool,
	supports_images: bool,
	/// TS `model.reasoning`.
	model_reasoning: bool,
	/// TS `options.reasoning`.
	reasoning: Option<String>,
	/// TS `options.disableReasoning`.
	disable_reasoning: bool,
	/// TS `model.thinking.effortMap` (cloud only).
	effort_map: BTreeMap<String, String>,
	/// TS `model.omitMaxOutputTokens`.
	omit_max_output_tokens: bool,
	/// TS `options.maxTokens`.
	max_tokens: Option<u64>,
	/// TS `options.temperature`.
	temperature: Option<f64>,
	/// TS `options.topP`.
	top_p: Option<f64>,
}

impl OllamaProvider {
	/// Build a provider with a rustls HTTP client.
	pub fn new(
		base_url: impl Into<String>,
		api_key: impl Into<String>,
		model: impl Into<String>,
	) -> Result<Self, ProviderError> {
		let raw_base_url: String = base_url.into();
		let base_url = normalize_base_url(&raw_base_url);
		validate_base_url(&base_url)?;
		let http = reqwest::Client::builder()
			.connect_timeout(CONNECT_TIMEOUT)
			.build()
			.map_err(network_error)?;
		Ok(Self::with_client(http, base_url, api_key, model))
	}

	/// Build a provider around a caller-configured HTTP client.
	///
	/// An empty API key omits `Authorization`, which keeps a local Ollama
	/// daemon usable; the TS adapter always sends a bearer token because its
	/// credential resolver rejects a missing key up front.
	pub fn with_client(
		http: reqwest::Client,
		base_url: impl Into<String>,
		api_key: impl Into<String>,
		model: impl Into<String>,
	) -> Self {
		let api_key: String = api_key.into();
		let base_url: String = base_url.into();
		Self {
			http,
			base_url: normalize_base_url(&base_url),
			api_key: (!api_key.trim().is_empty()).then_some(api_key),
			model: model.into(),
			cloud: false,
			supports_images: false,
			model_reasoning: false,
			reasoning: None,
			disable_reasoning: false,
			effort_map: BTreeMap::new(),
			omit_max_output_tokens: false,
			max_tokens: None,
			temperature: None,
			top_p: None,
		}
	}

	/// Enable or disable multimodal image inputs (`model.input` includes image).
	pub fn with_images(mut self, enabled: bool) -> Self {
		self.supports_images = enabled;
		self
	}

	/// Select Ollama Cloud dialect behavior (`model.provider === "ollama-cloud"`).
	pub fn with_cloud(mut self, enabled: bool) -> Self {
		self.cloud = enabled;
		self
	}

	/// Whether the model supports reasoning (`model.reasoning`).
	pub fn with_model_reasoning(mut self, enabled: bool) -> Self {
		self.model_reasoning = enabled;
		self
	}

	/// Requested reasoning effort (TS `options.reasoning`).
	pub fn with_reasoning(mut self, effort: Option<String>) -> Self {
		self.reasoning = effort.filter(|value| !value.trim().is_empty());
		self
	}

	/// Force native reasoning off (TS `options.disableReasoning`).
	pub fn with_disable_reasoning(mut self, enabled: bool) -> Self {
		self.disable_reasoning = enabled;
		self
	}

	/// Cloud effort remapping (`model.thinking.effortMap`).
	pub fn with_effort_map(
		mut self,
		map: impl IntoIterator<Item = (String, String)>,
	) -> Self {
		self.effort_map = map.into_iter().collect();
		self
	}

	/// Whether the model omits `num_predict` (`model.omitMaxOutputTokens`).
	pub fn with_omit_max_output_tokens(mut self, enabled: bool) -> Self {
		self.omit_max_output_tokens = enabled;
		self
	}

	/// Requested output token budget (TS `options.maxTokens`).
	pub fn with_max_tokens(mut self, max_tokens: Option<u64>) -> Self {
		self.max_tokens = max_tokens;
		self
	}

	/// Requested sampling temperature (TS `options.temperature`).
	pub fn with_temperature(mut self, temperature: Option<f64>) -> Self {
		self.temperature = temperature;
		self
	}

	/// Requested nucleus sampling (TS `options.topP`).
	pub fn with_top_p(mut self, top_p: Option<f64>) -> Self {
		self.top_p = top_p;
		self
	}

	/// `{normalized_base_url}/api/chat`.
	pub fn endpoint(&self) -> String {
		format!("{}/api/chat", self.base_url)
	}

	/// `mapReasoning` after the cloud `effortMap` remap.
	fn think(&self) -> Option<OllamaThink> {
		if self.disable_reasoning && self.model_reasoning {
			return Some(OllamaThink::Bool(false));
		}
		let mut effort = self.reasoning.as_deref();
		if self.cloud {
			if let Some(requested) = effort {
				if let Some(mapped) = self.effort_map.get(requested) {
					effort = Some(mapped.as_str());
				}
			}
		}
		match effort {
			Some("minimal" | "low") => Some(OllamaThink::Effort("low".to_string())),
			Some("medium") => Some(OllamaThink::Effort("medium".to_string())),
			Some("high") => Some(OllamaThink::Effort("high".to_string())),
			Some("max") => Some(OllamaThink::Effort("max".to_string())),
			Some("xhigh") => Some(OllamaThink::Effort("high".to_string())),
			_ => None,
		}
	}

	/// `resolveNumPredict`: cloud caps the requested budget at 65536.
	fn num_predict(&self) -> Option<u64> {
		let requested = self.max_tokens?;
		if self.omit_max_output_tokens {
			return None;
		}
		Some(if self.cloud {
			requested.min(OLLAMA_CLOUD_NUM_PREDICT_CAP)
		} else {
			requested
		})
	}

	/// Validate a forced tool before any socket is opened and return the tool
	/// selection plus the required wire mode.
	fn select_tools<'a>(
		&self,
		tools: &'a [Value],
		tool_choice: &ToolChoice,
	) -> Result<(&'a [Value], Option<&'static str>), ProviderError> {
		match tool_choice {
			ToolChoice::Auto => Ok((tools, None)),
			// `mapToolChoice`: `none` still sends the offered tools; the wire
			// field alone forbids calling them.
			ToolChoice::None => Ok((tools, Some("none"))),
			ToolChoice::Required => Ok((tools, Some("required"))),
			ToolChoice::Tool { name } => {
				if name.trim().is_empty() {
					return Err(ProviderError::InvalidToolChoice(
						"forced tool name must not be empty".to_string(),
					));
				}
				let Some(tool) = tools.iter().find(|tool| tool_name(tool) == Some(name.as_str()))
				else {
					return Err(ProviderError::InvalidToolChoice(format!(
						"forced tool `{name}` is not offered in this request"
					)));
				};
				// `selectToolsForToolChoice` narrows the request to the forced
				// tool; the wire only knows `tool_choice: "required"`.
				Ok((std::slice::from_ref(tool), Some("required")))
			},
		}
	}

	fn request(
		&self,
		messages: &[Message],
		tools: &[Value],
		tool_choice: &ToolChoice,
		stream: bool,
	) -> Result<OllamaChatRequest, ProviderError> {
		let (selected_tools, wire_tool_choice) = self.select_tools(tools, tool_choice)?;
		let options = OllamaOptions {
			num_predict: self.num_predict(),
			temperature: self.temperature,
			top_p: self.top_p,
		};
		let has_options =
			options.num_predict.is_some() || options.temperature.is_some() || options.top_p.is_some();
		Ok(OllamaChatRequest {
			model: self.model.clone(),
			messages: self.convert_messages(messages),
			tools: convert_tools(selected_tools),
			think: self.think(),
			tool_choice: wire_tool_choice,
			options: has_options.then_some(options),
			stream,
		})
	}

	fn convert_messages(&self, messages: &[Message]) -> Vec<OllamaWireMessage> {
		let call_names = tool_call_names(messages);
		// The TS adapter receives system prompts separately and emits them first.
		// Rust messages carry them inline, so a leading system-prompt run is the
		// closest analogue of `systemPrompts.length`.
		let prefix = messages
			.iter()
			.take_while(|message| is_system_prompt(message))
			.count();
		let mut converted: Vec<OllamaWireMessage> = messages
			.iter()
			.enumerate()
			.map(|(index, message)| {
				// `convertMessages`: a user-attributed developer turn after the
				// static prefix drops to `user` so the in-conversation byte prefix
				// stays stable; agent-attributed control turns stay `system`.
				let developer_role =
					if index < prefix || !is_user_attributed(message) {
						"system"
					} else {
						"user"
					};
				convert_message(
					message,
					self.supports_images,
					&call_names,
					developer_role,
				)
			})
			.collect();
		if self.cloud {
			// Ollama Cloud rejects `thinking` in assistant history.
			for message in &mut converted {
				if message.role == "assistant" && message.thinking.is_some() {
					message.thinking = None;
				}
			}
		}
		// `done_reason: "load"` is returned when no `user` turn reaches the model;
		// demote the last non-prefix system turn so the request can produce output.
		if !converted.iter().any(|message| message.role == "user") {
			let prefix = prefix.min(converted.len());
			for message in converted[prefix..].iter_mut().rev() {
				if message.role == "system" {
					message.role = "user";
					break;
				}
			}
		}
		converted
	}
}

/// TS `systemPrompts`: the static prefix emitted before conversation turns.
/// Rust carries those prompts inline as leading `Role::System` turns; developer
/// turns are conversation-level and never part of the immutable prefix.
fn is_system_prompt(message: &Message) -> bool {
	message.role == Role::System
}

fn is_user_attributed(message: &Message) -> bool {
	matches!(message.attribution.as_ref(), Some(Attribution::User))
}

fn tool_name(tool: &Value) -> Option<&str> {
	tool
		.get("function")
		.and_then(|function| function.get("name"))
		.and_then(Value::as_str)
}

fn tool_call_names(messages: &[Message]) -> HashMap<String, String> {
	let mut names = HashMap::new();
	for message in messages {
		for call in &message.tool_calls {
			names.insert(call.id.clone(), call.name.clone());
		}
	}
	names
}

/// `toPlainContent` + the vision guard: text blocks joined with `\n`, omitted
/// images replaced by the shared placeholder, included images as raw base64.
fn plain_content(message: &Message, supports_images: bool) -> (String, Option<Vec<String>>) {
	if message.parts.is_empty() {
		return (message.content.clone(), None);
	}
	let mut text_blocks = Vec::new();
	let mut images = Vec::new();
	let mut omitted = false;
	for part in &message.parts {
		match part {
			ContentPart::Text { text } => text_blocks.push(text.as_str()),
			ContentPart::Image { data, mime_type } => {
				if supports_images {
					images.push(
						normalize_image_payload(data, mime_type)
							.map(|(clean, _)| clean)
							.unwrap_or_else(|_| data.clone()),
					);
				} else {
					omitted = true;
				}
			},
			// Ollama's wire has no reasoning field for user/system/tool turns;
			// `convertMessage` drops reasoning and never moves a tool call into
			// `content` for a non-assistant turn.
			ContentPart::Thinking { .. }
			| ContentPart::RedactedThinking { .. }
			| ContentPart::ToolCall { .. } => {},
		}
	}
	let mut parts = Vec::with_capacity(2);
	let joined = text_blocks.join("\n");
	// Parts are authoritative upstream, but Rust messages may also carry the
	// plain string when only reasoning/image parts were appended.
	let text = if joined.is_empty() && !message.content.is_empty() {
		message.content.clone()
	} else {
		joined
	};
	if !text.is_empty() {
		parts.push(text);
	}
	if omitted {
		parts.push(NON_VISION_IMAGE_PLACEHOLDER.to_string());
	}
	let content = parts.join("\n");
	(content, (!images.is_empty()).then_some(images))
}

fn convert_message(
	message: &Message,
	supports_images: bool,
	call_names: &HashMap<String, String>,
	developer_role: &'static str,
) -> OllamaWireMessage {
	match message.role {
		Role::User => {
			let (content, images) = plain_content(message, supports_images);
			OllamaWireMessage {
				role: "user",
				content,
				images,
				thinking: None,
				tool_calls: None,
				tool_name: None,
			}
		},
		Role::System | Role::Developer => {
			let (content, images) = plain_content(message, supports_images);
			OllamaWireMessage {
				role: if message.role == Role::System {
					"system"
				} else {
					developer_role
				},
				content,
				images,
				thinking: None,
				tool_calls: None,
				tool_name: None,
			}
		},
		Role::Tool => {
			let (content, images) = plain_content(message, supports_images);
			let tool_name = message.name.clone().or_else(|| {
				message
					.tool_call_id
					.as_ref()
					.and_then(|id| call_names.get(id).cloned())
			});
			OllamaWireMessage {
				role: "tool",
				content,
				images,
				thinking: None,
				tool_calls: None,
				tool_name,
			}
		},
		Role::Assistant => {
			let mut text_blocks = Vec::new();
			let mut thinking_blocks = Vec::new();
			let mut part_tool_calls = Vec::new();
			if message.parts.is_empty() {
				text_blocks.push(message.content.as_str());
			} else {
				for part in &message.parts {
					match part {
						ContentPart::Text { text } => text_blocks.push(text.as_str()),
						ContentPart::Thinking { thinking, .. } => {
							thinking_blocks.push(thinking.as_str())
						},
						ContentPart::ToolCall {
							name, arguments, ..
						} => part_tool_calls.push(OllamaWireToolCall {
							kind: "function",
							function: OllamaWireFunction {
								name: name.clone(),
								arguments: arguments.clone(),
							},
						}),
						// `convertMessage` has no Ollama wire slot for redacted
						// reasoning; the source block is intentionally skipped, as
						// in TS.
						ContentPart::RedactedThinking { .. } | ContentPart::Image { .. } => {},
					}
				}
			}
			// TS reads calls from the ordered content blocks. Messages built
			// before the content-part contract may only carry the execution
			// projection, so fall back to it when no block holds a call.
			let wire_calls = if part_tool_calls.is_empty() {
				message
					.tool_calls
					.iter()
					.map(|call| OllamaWireToolCall {
						kind: "function",
						function: OllamaWireFunction {
							name: call.name.clone(),
							arguments: call.arguments.clone(),
						},
					})
					.collect()
			} else {
				part_tool_calls
			};
			let tool_calls = (!wire_calls.is_empty()).then_some(wire_calls);
			let thinking = (!thinking_blocks.is_empty()).then(|| thinking_blocks.join("\n"));
			let joined = text_blocks.join("\n");
			let content = if joined.is_empty() && !message.content.is_empty() {
				message.content.clone()
			} else {
				joined
			};
			OllamaWireMessage {
				role: "assistant",
				content,
				images: None,
				thinking,
				tool_calls,
				tool_name: None,
			}
		},
	}
}

/// `convertTools` + `sanitizeSchemaForOllama`.
///
/// ponytail: Rust callers pass an already-built tool schema, so the
/// `toolWireSchema` upgrade/post-process stage is skipped; add it here if
/// callers can hand over raw Ark/draft schemas.
fn convert_tools(tools: &[Value]) -> Option<Vec<OllamaFunctionTool>> {
	if tools.is_empty() {
		return None;
	}
	Some(
		tools
			.iter()
			.map(|tool| {
				let function = tool.get("function");
				let name = function
					.and_then(|function| function.get("name"))
					.and_then(Value::as_str)
					.unwrap_or_default()
					.to_string();
				let description = function
					.and_then(|function| function.get("description"))
					.and_then(Value::as_str)
					.unwrap_or_default()
					.to_string();
				let parameters = function
					.and_then(|function| function.get("parameters"))
					.cloned()
					.unwrap_or_else(|| json!({}));
				OllamaFunctionTool {
					kind: "function",
					function: OllamaFunction {
						name,
						description,
						parameters: sanitize_schema(&parameters),
					},
				}
			})
			.collect(),
	)
}

/// Widened stand-in for a bare `true` subschema (Ollama's Go parser cannot
/// unmarshal boolean schemas).
fn open_subschema_widening() -> Value {
	json!({
		"anyOf": [
			{"type": "string"},
			{"type": "number"},
			{"type": "boolean"},
			{"type": "object"},
			{"type": "array"},
			{"type": "null"},
		]
	})
}

const SUBSCHEMA_MAP_KEYS: &[&str] = &[
	"properties",
	"patternProperties",
	"dependencies",
	"dependentSchemas",
	"$defs",
	"definitions",
];
const SUBSCHEMA_ARRAY_KEYS: &[&str] = &["anyOf", "oneOf", "allOf", "prefixItems"];
const OLLAMA_SCHEMA_VALUE_KEYS: &[&str] = &[
	"items",
	"additionalItems",
	"contains",
	"contentSchema",
	"propertyNames",
	"if",
	"then",
	"else",
	"not",
	"additionalProperties",
	"unevaluatedItems",
	"unevaluatedProperties",
];

/// Port of `sanitizeSchemaForOllama` (`utils/schema/normalize.ts`).
fn sanitize_schema(schema: &Value) -> Value {
	sanitize_node(schema, &open_subschema_widening())
}

fn sanitize_node(value: &Value, widening: &Value) -> Value {
	match value {
		Value::Bool(true) => widening.clone(),
		Value::Bool(false) => json!({ "not": widening.clone() }),
		Value::Array(items) => Value::Array(
			items
				.iter()
				.map(|item| sanitize_node(item, widening))
				.collect(),
		),
		Value::Object(map) => sanitize_object(map, widening),
		other => other.clone(),
	}
}

fn sanitize_object(map: &Map<String, Value>, widening: &Value) -> Value {
	let mut output = Map::new();
	let mut type_alternatives: Option<Vec<Value>> = None;
	for (key, child) in map {
		if (key == "additionalProperties" || key == "unevaluatedProperties")
			&& child.is_boolean()
		{
			continue;
		}
		if key == "type" {
			if let Some(entries) = child.as_array() {
				let mut variants: Vec<String> = Vec::new();
				for entry in entries {
					if let Some(text) = entry.as_str() {
						if !variants.iter().any(|existing| existing == text) {
							variants.push(text.to_string());
						}
					}
				}
				let non_null: Vec<&String> =
					variants.iter().filter(|entry| entry.as_str() != "null").collect();
				if non_null.len() <= 1 {
					let selected = non_null
						.first()
						.map(|entry| Value::String((*entry).clone()))
						.or_else(|| variants.first().map(|entry| Value::String(entry.clone())))
						.or_else(|| entries.first().cloned())
						.unwrap_or(Value::Null);
					output.insert("type".to_string(), selected);
				} else {
					type_alternatives = Some(
						variants
							.iter()
							.map(|entry| json!({ "type": entry }))
							.collect(),
					);
				}
				continue;
			}
		}

		let next = if SUBSCHEMA_MAP_KEYS.contains(&key.as_str()) {
			match child.as_object() {
				Some(children) => {
					let mut mapped = Map::new();
					for (child_key, child_value) in children {
						mapped.insert(
							child_key.clone(),
							sanitize_node(child_value, widening),
						);
					}
					Value::Object(mapped)
				},
				None => child.clone(),
			}
		} else if SUBSCHEMA_ARRAY_KEYS.contains(&key.as_str()) {
			match child.as_array() {
				Some(items) => Value::Array(
					items
						.iter()
						.map(|item| sanitize_node(item, widening))
						.collect(),
				),
				None => child.clone(),
			}
		} else if OLLAMA_SCHEMA_VALUE_KEYS.contains(&key.as_str()) {
			sanitize_node(child, widening)
		} else {
			child.clone()
		};
		output.insert(key.clone(), next);
	}

	if let Some(alternatives) = type_alternatives {
		let type_union = json!({ "anyOf": alternatives });
		let all_of = match output.get("allOf").and_then(Value::as_array) {
			Some(existing) => {
				let mut combined = Vec::with_capacity(existing.len() + 1);
				combined.push(type_union);
				combined.extend(existing.iter().cloned());
				combined
			},
			None => vec![type_union],
		};
		output.insert("allOf".to_string(), Value::Array(all_of));
	}
	Value::Object(output)
}

#[async_trait]
impl ModelProvider for OllamaProvider {
	fn supports_images(&self) -> bool {
		self.supports_images
	}

	/// TS `createEmptyOutput`: `api: "ollama-chat"`.
	fn api(&self) -> Option<&str> {
		Some("ollama-chat")
	}

	/// TS `model.provider`: local `ollama` or `ollama-cloud`.
	fn provider(&self) -> Option<&str> {
		Some(if self.cloud { "ollama-cloud" } else { "ollama" })
	}

	fn model(&self) -> Option<&str> {
		Some(self.model.as_str())
	}

	/// Text-only models omit images through the vision guard instead of failing
	/// the request; a supported image still has to be a valid local payload.
	fn check_support(&self, messages: &[Message]) -> Result<(), ProviderError> {
		if !self.supports_images {
			return Ok(());
		}
		for message in messages {
			for image in message.image_parts() {
				if let Err(err) = image.validate() {
					return Err(ProviderError::UnsupportedImage(format!(
						"invalid image payload: {err}"
					)));
				}
			}
		}
		Ok(())
	}

	async fn complete(
		&self,
		messages: &[Message],
		tools: &[Value],
		tool_choice: &ToolChoice,
	) -> Result<Message, ProviderError> {
		let mut stream = self.stream(messages, tools, tool_choice).await?;
		let mut assembler = StreamAssembler::new();
		while let Some(item) = stream.next().await {
			assembler.push(&item?)?;
		}
		let mut message = assembler.finish()?;
		// TS `createEmptyOutput` stamps the source on the assistant message so a
		// later `transformMessages` pass can tell same-model replays from
		// cross-model ones instead of guessing.
		message.api = self.api().map(str::to_string);
		message.provider = self.provider().map(str::to_string);
		message.model = self.model().map(str::to_string);
		Ok(message)
	}

	async fn stream(
		&self,
		messages: &[Message],
		tools: &[Value],
		tool_choice: &ToolChoice,
	) -> Result<ChunkStream, ProviderError> {
		self.check_support(messages)?;
		let body = self.request(messages, tools, tool_choice, true)?;
		let mut request = self.http.post(self.endpoint()).json(&body);
		if let Some(api_key) = self.api_key.as_deref() {
			request = request.bearer_auth(api_key);
		}
		let response = request.send().await.map_err(network_error)?;
		let status = response.status();
		let secrets = credential_values(self.api_key.as_deref());
		if !status.is_success() {
			let body = match read_body_bounded(response, ERROR_BODY_BYTES).await {
				Ok(body) => bounded_redaction(&body.text, &secrets, body.truncated),
				Err(err) => format!("<failed to read error body: {err}>"),
			};
			return Err(ProviderError::HttpStatus(status.as_u16(), body));
		}
		let body = response
			.bytes_stream()
			.map(|item| item.map_err(|err| err.to_string()));
		// ponytail: TS additionally wraps this request in `fetchWithRetry`,
		// `withReplaySafeStreamRetry`, `armedPreResponseTimeout` and
		// `StreamMarkupHealing`. The NDJSON parser now carries terminal stop
		// reasons/errors so the first three have a surface to hook; the markup
		// healer needs the dialect scanner subsystem, which this crate does not
		// have yet.
		Ok(ndjson_stream(body, secrets))
	}
}

/// Accumulated parser state for one NDJSON response body.
#[derive(Debug, Default)]
struct ParseState {
	/// Number of content blocks TS would have pushed (`output.content.length`).
	blocks: usize,
	/// Kind of the block still receiving deltas, or `None` after a tool call.
	active: Option<BlockKind>,
	tool_index: usize,
	saw_tool_calls: bool,
	/// Cumulative `hasVisibleAssistantContent` (text/thinking/tool call).
	visible: bool,
	/// Last `done` record's usage and raw done-reason. The terminal event is
	/// only emitted at EOF so a later record can still add content, matching
	/// TS's post-loop finalization.
	terminal: Option<(Option<TokenUsage>, StopReason)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
	Thinking,
	Text,
}

impl ParseState {
	/// Open (or continue) a text/thinking block and return its content index.
	fn open(&mut self, kind: BlockKind) -> usize {
		if self.active == Some(kind) {
			return self.blocks.saturating_sub(1);
		}
		self.active = Some(kind);
		let index = self.blocks;
		self.blocks += 1;
		index
	}

	/// Each tool call is its own content block; the next delta opens a fresh
	/// text/thinking block after it.
	fn open_tool(&mut self) -> usize {
		self.active = None;
		let index = self.blocks;
		self.blocks += 1;
		index
	}
}

/// TS post-stream finalization: tool calls always mean "execute and continue",
/// then the empty-`length` and `load` records become terminal errors that keep
/// the partial message.
fn finalize_stop(state: &ParseState, stop: StopReason) -> (StopReason, Option<String>) {
	let mut stop = stop;
	if matches!(stop, StopReason::Stop) && state.saw_tool_calls {
		stop = StopReason::ToolUse;
	}
	if matches!(stop, StopReason::Length) && !state.visible {
		return (
			StopReason::Error,
			Some(EMPTY_OLLAMA_LENGTH_COMPLETION_MESSAGE.to_string()),
		);
	}
	if matches!(stop, StopReason::Error) {
		return (
			StopReason::Error,
			Some(EMPTY_OLLAMA_LOAD_COMPLETION_MESSAGE.to_string()),
		);
	}
	(stop, None)
}

struct NdjsonState<S> {
	source: std::pin::Pin<Box<S>>,
	buffer: Vec<u8>,
	queue: VecDeque<Result<StreamChunk, ProviderError>>,
	parse: ParseState,
	finished: bool,
	/// Effective bearer credential(s) scrubbed from parser diagnostics.
	secrets: Vec<String>,
}

fn take_lines(buffer: &mut Vec<u8>) -> Vec<Vec<u8>> {
	let mut lines = Vec::new();
	while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
		let mut line: Vec<u8> = buffer.drain(..=position).collect();
		line.pop();
		if line.last() == Some(&b'\r') {
			line.pop();
		}
		lines.push(line);
	}
	lines
}

fn delta_chunk(
	content: Option<String>,
	reasoning: Option<String>,
	tool_call_deltas: Vec<ToolCallDelta>,
	content_index: usize,
) -> StreamChunk {
	StreamChunk {
		content,
		reasoning,
		tool_call_deltas,
		usage: None,
		is_done: false,
		content_index: Some(content_index),
		signature: None,
		redacted: None,
		stop_reason: None,
		error_message: None,
	}
}

fn done_chunk(
	usage: Option<TokenUsage>,
	stop_reason: StopReason,
	error_message: Option<String>,
) -> StreamChunk {
	StreamChunk {
		content: None,
		reasoning: None,
		tool_call_deltas: Vec::new(),
		usage,
		is_done: true,
		content_index: None,
		signature: None,
		redacted: None,
		stop_reason: Some(stop_reason),
		error_message,
	}
}

/// TS catch block: a failed record or transport ends the turn as an `error`
/// event carrying the partial output, not as a thrown provider error.
fn error_chunk(message: String) -> StreamChunk {
	done_chunk(None, StopReason::Error, Some(message))
}

fn map_done_reason(reason: Option<&str>, saw_tool_calls: bool) -> StopReason {
	match reason {
		Some("length") => StopReason::Length,
		Some("tool_calls") => StopReason::ToolUse,
		Some("load") => StopReason::Error,
		None if saw_tool_calls => StopReason::ToolUse,
		_ => StopReason::Stop,
	}
}

fn tool_arguments(call: &OllamaChunkToolCall) -> String {
	let arguments = call
		.function
		.as_ref()
		.and_then(|function| function.arguments.as_ref());
	match arguments {
		Some(Value::String(text)) => text.clone(),
		None | Some(Value::Null) => "{}".to_string(),
		Some(value) => value.to_string(),
	}
}

/// Parse one NDJSON line into zero or more stream chunks.
fn process_line(
	line: &[u8],
	state: &mut ParseState,
	secrets: &[String],
) -> Result<Vec<StreamChunk>, ProviderError> {
	let text = std::str::from_utf8(line).map_err(|err| {
		ProviderError::Protocol(format!("invalid UTF-8 in Ollama stream record: {err}"))
	})?;
	let trimmed = text.trim();
	if trimmed.is_empty() {
		return Ok(Vec::new());
	}
	let chunk: OllamaChatChunk = serde_json::from_str(trimmed).map_err(|err| {
		ProviderError::Protocol(format!(
			"invalid Ollama stream record: {err}; record: {}",
			bounded_redaction(trimmed, secrets, false)
		))
	})?;

	let mut output = Vec::new();

	if let Some(message) = chunk.message {
		// TS pushes one event per block delta, thinking before text and tool
		// calls after both, so each chunk carries exactly one content index.
		if let Some(thinking) = message.thinking.filter(|value| !value.is_empty()) {
			if !thinking.trim().is_empty() {
				state.visible = true;
			}
			let index = state.open(BlockKind::Thinking);
			output.push(delta_chunk(None, Some(thinking), Vec::new(), index));
		}
		if let Some(delta) = message.content.filter(|value| !value.is_empty()) {
			if !delta.trim().is_empty() {
				state.visible = true;
			}
			let index = state.open(BlockKind::Text);
			output.push(delta_chunk(Some(delta), None, Vec::new(), index));
		}
		if let Some(calls) = message.tool_calls.filter(|calls| !calls.is_empty()) {
			for call in calls {
				let name = call
					.function
					.as_ref()
					.and_then(|function| function.name.clone())
					.unwrap_or_else(|| "unknown_tool".to_string());
				let partial_json = tool_arguments(&call);
				let index = state.open_tool();
				let id = format!("ollama:{index}:{name}");
				let delta = ToolCallDelta {
					index: state.tool_index,
					id: Some(id),
					name: Some(name),
					arguments_delta: Some(partial_json),
					content_index: Some(index),
				};
				state.tool_index += 1;
				state.saw_tool_calls = true;
				state.visible = true;
				output.push(delta_chunk(None, None, vec![delta], index));
			}
		}
	}

	if chunk.done == Some(true) {
		// TS updates `output.stopReason`/usage here but only pushes the
		// terminal event after the read loop, so keep the finalization for EOF.
		let input = chunk.prompt_eval_count.unwrap_or(0);
		let output_tokens = chunk.eval_count.unwrap_or(0);
		let usage = TokenUsage {
			input,
			output: output_tokens,
			cache_read: 0,
			cache_write: 0,
			total_tokens: input.saturating_add(output_tokens),
			reasoning_tokens: None,
			..TokenUsage::default()
		};
		state.terminal = Some((
			Some(usage),
			map_done_reason(chunk.done_reason.as_deref(), state.saw_tool_calls),
		));
	}

	Ok(output)
}

/// Turn a raw byte stream of NDJSON records into parsed chunks.
///
/// Records may be split across transport chunks, including in the middle of a
/// UTF-8 sequence; the stream may end without a trailing newline. A malformed
/// record, an incomplete EOF tail or a transport failure ends the turn with a
/// terminal [`StopReason::Error`] chunk that keeps the partial output, matching
/// the TS adapter's `error` event.
fn ndjson_stream<S, B, E>(body: S, secrets: Vec<String>) -> ChunkStream
where
	S: Stream<Item = Result<B, E>> + Send + 'static,
	B: AsRef<[u8]> + Send + 'static,
	E: Display + Send + 'static,
{
	let state = NdjsonState {
		source: Box::pin(body),
		buffer: Vec::new(),
		queue: VecDeque::new(),
		parse: ParseState::default(),
		finished: false,
		secrets,
	};
	Box::pin(futures::stream::unfold(Some(state), |state| async move {
		let mut state = state?;
		loop {
			if let Some(item) = state.queue.pop_front() {
				return Some((item, Some(state)));
			}
			if state.finished {
				return None;
			}
			match state.source.next().await {
				Some(Ok(bytes)) => {
					let incoming = bytes.as_ref();
					// Reject growth of an unterminated record before allocating;
					// chunks carrying newlines are drained below and covered by the
					// tail bound.
					if !incoming.contains(&b'\n')
						&& state.buffer.len() + incoming.len() > MAX_RECORD_BYTES
					{
						state.queue.push_back(Ok(error_chunk(format!(
							"Ollama stream record exceeded {MAX_RECORD_BYTES} bytes"
						))));
						state.finished = true;
						continue;
					}
					state.buffer.extend_from_slice(incoming);
					// Complete records are drained below; only the unterminated tail
					// counts against the per-record ceiling.
					let tail = match state.buffer.iter().rposition(|byte| *byte == b'\n') {
						Some(position) => state.buffer.len() - position - 1,
						None => state.buffer.len(),
					};
					if tail > MAX_RECORD_BYTES {
						state.queue.push_back(Ok(error_chunk(format!(
							"Ollama stream record exceeded {MAX_RECORD_BYTES} bytes"
						))));
						state.finished = true;
						continue;
					}
					for line in take_lines(&mut state.buffer) {
						match process_line(&line, &mut state.parse, &state.secrets) {
							Ok(chunks) => state.queue.extend(chunks.into_iter().map(Ok)),
							Err(err) => {
								state.queue.push_back(Ok(error_chunk(err.to_string())));
								state.finished = true;
								break;
							},
						}
					}
				},
				Some(Err(err)) => {
					state.queue.push_back(Ok(error_chunk(err.to_string())));
					state.finished = true;
				},
				None => {
					let mut fatal = false;
					if !state.buffer.is_empty() {
						let tail = std::mem::take(&mut state.buffer);
						match process_line(&tail, &mut state.parse, &state.secrets) {
							Ok(chunks) => state.queue.extend(chunks.into_iter().map(Ok)),
							Err(err) => {
								state.queue.push_back(Ok(error_chunk(err.to_string())));
								fatal = true;
							},
						}
					}
					if !fatal {
						let (usage, raw_stop) = state
							.parse
							.terminal
							.take()
							.unwrap_or((None, StopReason::Stop));
						let (stop, error_message) = finalize_stop(&state.parse, raw_stop);
						state
							.queue
							.push_back(Ok(done_chunk(usage, stop, error_message)));
					}
					state.finished = true;
				},
			}
		}
	}))
}

#[cfg(test)]
mod tests {
	use super::*;

	fn message_tool(name: Option<&str>, call_id: &str) -> Message {
		let mut message = Message::tool(call_id, "result");
		message.name = name.map(str::to_string);
		message
	}

	#[test]
	fn base_url_normalization_matches_ts() {
		assert_eq!(normalize_base_url(""), "https://ollama.com");
		assert_eq!(normalize_base_url(" http://localhost:11434/ "), "http://localhost:11434");
		assert_eq!(
			normalize_base_url("http://localhost:11434/api/"),
			"http://localhost:11434"
		);
	}

	#[test]
	fn tool_result_name_falls_back_to_the_originating_call() {
		let mut assistant = Message::assistant("");
		assistant.tool_calls = vec![crate::state::ToolCall {
			id: "call_1".to_string(),
			name: "read".to_string(),
			arguments: json!({"path": "a"}),
		}];
		let messages = vec![Message::user("hi"), assistant, message_tool(None, "call_1")];
		let names = tool_call_names(&messages);
		let converted = convert_message(&messages[2], false, &names, "system");
		assert_eq!(converted.tool_name.as_deref(), Some("read"));
	}

	#[test]
	fn vision_guard_omits_images_without_failing_support_check() {
		let image = ContentPart::image("aGVsbG8=", "image/png").expect("image");
		let mut message = Message::user_with_parts(vec![
			ContentPart::text("look"),
			image,
		]);
		message.content = "look".to_string();
		let provider = OllamaProvider::with_client(
			reqwest::Client::new(),
			"http://localhost:11434",
			"",
			"m",
		);
		provider.check_support(&[message.clone()]).expect("omitted image is allowed");
		let (content, images) = plain_content(&message, false);
		assert_eq!(
			content,
			format!("look\n{NON_VISION_IMAGE_PLACEHOLDER}")
		);
		assert!(images.is_none());
	}

	#[test]
	fn schema_sanitizer_widens_boolean_subschemas_and_strips_closed_objects() {
		let schema = json!({
			"type": "object",
			"additionalProperties": false,
			"properties": {
				"a": true,
				"b": {"type": ["string", "null"]},
				"c": {"type": ["string", "number"]},
			},
			"required": true,
		});
		let sanitized = sanitize_schema(&schema);
		assert!(sanitized.get("additionalProperties").is_none());
		assert_eq!(
			sanitized["properties"]["a"]["anyOf"].as_array().unwrap().len(),
			6
		);
		assert_eq!(sanitized["properties"]["b"]["type"], "string");
		assert_eq!(
			sanitized["properties"]["c"]["allOf"][0]["anyOf"]
				.as_array()
				.unwrap()
				.len(),
			2
		);
	}

	#[test]
	fn done_reason_and_terminal_finalization_match_ts() {
		assert_eq!(map_done_reason(Some("length"), false), StopReason::Length);
		assert_eq!(map_done_reason(Some("tool_calls"), false), StopReason::ToolUse);
		assert_eq!(map_done_reason(Some("load"), false), StopReason::Error);
		assert_eq!(map_done_reason(None, true), StopReason::ToolUse);
		assert_eq!(map_done_reason(Some("stop"), true), StopReason::Stop);
		assert_eq!(map_done_reason(None, false), StopReason::Stop);

		let mut with_tools = ParseState::default();
		with_tools.saw_tool_calls = true;
		assert_eq!(
			finalize_stop(&with_tools, StopReason::Stop),
			(StopReason::ToolUse, None)
		);

		let empty_length = finalize_stop(&ParseState::default(), StopReason::Length);
		assert_eq!(empty_length.0, StopReason::Error);
		assert_eq!(
			empty_length.1.as_deref(),
			Some(EMPTY_OLLAMA_LENGTH_COMPLETION_MESSAGE)
		);

		let load = finalize_stop(&ParseState::default(), StopReason::Error);
		assert_eq!(load.0, StopReason::Error);
		assert_eq!(
			load.1.as_deref(),
			Some(EMPTY_OLLAMA_LOAD_COMPLETION_MESSAGE)
		);
	}

	#[test]
	fn reasoning_effort_maps_through_the_cloud_effort_map() {
		let provider = OllamaProvider::with_client(
			reqwest::Client::new(),
			"http://localhost:11434",
			"",
			"m",
		)
		.with_cloud(true)
		.with_effort_map([("high".to_string(), "max".to_string())])
		.with_reasoning(Some("high".to_string()));
		assert_eq!(
			provider.think(),
			Some(OllamaThink::Effort("max".to_string()))
		);
		let provider = provider.with_reasoning(Some("xhigh".to_string()));
		assert_eq!(
			provider.think(),
			Some(OllamaThink::Effort("high".to_string()))
		);
		let provider = provider
			.with_model_reasoning(true)
			.with_disable_reasoning(true);
		assert_eq!(provider.think(), Some(OllamaThink::Bool(false)));
	}

	#[test]
	fn assistant_history_keeps_text_and_thinking_blocks() {
		let mut message = Message::assistant("");
		message.parts = vec![
			ContentPart::thinking("plan"),
			ContentPart::Text {
				text: "answer".to_string(),
			},
		];
		let converted = convert_message(&message, false, &HashMap::new(), "system");
		assert_eq!(converted.role, "assistant");
		assert_eq!(converted.content, "answer");
		assert_eq!(converted.thinking.as_deref(), Some("plan"));
	}
}
