//! Ollama `/api/chat` provider (`packages/ai/src/providers/ollama.ts`).
//!
//! Native NDJSON streaming, Ollama-shaped history conversion and the
//! vision/cloud quirks of the TypeScript source. The caller owns cancellation:
//! dropping the returned stream drops the HTTP response body and closes the
//! connection, exactly like the other provider adapters in this crate.

use std::{
	collections::{BTreeMap, HashMap, VecDeque},
	fmt::Display,
	future::Future,
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicU64, Ordering},
	},
	time::Duration,
};

/// Boxed future alias for async callbacks and hooks without extra lifetime dependencies.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// TS `onPayload?: (payload: unknown, model?: Model<Api>) => unknown | undefined | Promise<unknown | undefined>`.
/// Receives the serialized request body and the target model id, returning an optional replacement body or an error.
pub type OllamaPayloadHook = Arc<
	dyn for<'a> Fn(&'a Value, &'a str) -> BoxFuture<'a, Result<Option<Value>, ProviderError>>
		+ Send
		+ Sync,
>;

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
/// Default fetch retry backoff schedule for transient Ollama errors (`ollama.ts:458`).
pub const OLLAMA_RETRY_DELAYS_MS: [u64; 3] = [2000, 5000, 10000];
/// Maximum HTTP request attempts for retryable statuses (`fetch-retry.ts:145`).
pub const MAX_FETCH_ATTEMPTS: usize = 5;
/// Maximum Retry-After delay honored before aborting retry loop (`fetch-retry.ts:167`).
pub const MAX_RETRY_DELAY_MS: u64 = 60_000;
/// Maximum attempts for benign empty completion retries (`empty-completion-retry.ts:18`).
pub const MAX_EMPTY_COMPLETION_RETRIES: usize = 2;
/// Base delay for empty completion retry backoff (`empty-completion-retry.ts:19`).
pub const EMPTY_COMPLETION_BASE_DELAY_MS: u64 = 500;
/// Default maximum gap between streamed NDJSON events before timeout (`idle-iterator.ts:14`).
pub const DEFAULT_STREAM_IDLE_TIMEOUT_MS: u64 = 300_000;
/// Default maximum wait for the first streamed event before timeout (`idle-iterator.ts:24`).
pub const DEFAULT_STREAM_FIRST_EVENT_TIMEOUT_MS: u64 = 300_000;
/// Maximum characters of an error body or malformed record kept for diagnostics.
const ERROR_BODY_CHARS: usize = 2048;
/// Hard ceiling on bytes read from an HTTP error body before diagnostics.
const ERROR_BODY_BYTES: usize = 64 * 1024;
/// Hard ceiling on one NDJSON record; overflow is a terminal error, never a
/// silently dropped or truncated success.
const MAX_RECORD_BYTES: usize = 8 * 1024 * 1024;
/// Aggregate cap for a tool-call section spanning many NDJSON records; the
/// per-record cap alone cannot bound a section that never closes.
pub const MAX_SECTION_BYTES: usize = 8 * 1024 * 1024;
/// Ollama Cloud rejects `num_predict` above this value with HTTP 400.
const OLLAMA_CLOUD_NUM_PREDICT_CAP: u64 = 65_536;
/// `vision-guard.ts` placeholder for an image sent to a text-only model.
pub const NON_VISION_IMAGE_PLACEHOLDER: &str = "[image omitted: model does not support vision]";
const EMPTY_OLLAMA_LENGTH_COMPLETION_MESSAGE: &str =
	"Model returned no content: prompt filled the context window; raise Ollama num_ctx or shorten the prompt.";
const EMPTY_OLLAMA_LOAD_COMPLETION_MESSAGE: &str =
	"Ollama loaded the model but generated nothing (done_reason: load): the request contained no user-role message.";
/// Resolve an optional duration from environment variable names (`idle-iterator.ts:32`).
/// A value of 0 explicitly disables the timeout; missing/invalid uses default.
pub fn resolve_env_timeout(env_vars: &[&str], default_ms: u64) -> Option<Duration> {
	for var in env_vars {
		if let Ok(val) = std::env::var(var) {
			if let Ok(num) = val.trim().parse::<u64>() {
				if num == 0 {
					return None;
				}
				return Some(Duration::from_millis(num));
			}
		}
	}
	Some(Duration::from_millis(default_ms))
}


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

/// Best effort for byte-capped reads: drop the incomplete trailing token so a
/// credential cut mid-token is unlikely to survive. Diagnostics stay bounded
/// and redacted regardless; this is a heuristic, not a prefix guarantee.
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
#[derive(Clone)]
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
	/// Configured gateway identity; overrides the `ollama`/`ollama-cloud` default
	/// so custom providers are stamped and cross-model replays classify right.
	provider_id: Option<String>,
	/// Optional request payload replacement hook (`ollama.ts:577-580`).
	on_payload: Option<OllamaPayloadHook>,
	/// Pre-response and first-event stream budget (`idle-iterator.ts:24`).
	first_event_timeout: Option<Duration>,
	/// Idle gap budget between streamed stream chunks (`idle-iterator.ts:14`).
	idle_timeout: Option<Duration>,
}

/// True when a tool call has empty or whitespace-only id or name (`transform-messages.ts:166-168`).
pub fn is_malformed_tool_call(call: &crate::state::ToolCall) -> bool {
	call.id.trim().is_empty() || call.name.trim().is_empty()
}

/// Strip malformed tool calls and their matching tool results (`transform-messages.ts:170-229`).
pub fn sanitize_malformed_tool_calls(messages: &[Message]) -> Vec<Message> {
	let mut drop_queue: Vec<String> = Vec::new();
	let mut sanitized = Vec::with_capacity(messages.len());

	for msg in messages {
		match msg.role {
			Role::Assistant => {
				let had_calls = !msg.tool_calls.is_empty()
					|| msg.parts.iter().any(|part| matches!(part, ContentPart::ToolCall { .. }));
				if !had_calls {
					sanitized.push(msg.clone());
					continue;
				}
				let mut new_msg = msg.clone();
				new_msg.tool_calls.retain(|call| {
					if is_malformed_tool_call(call) {
						if !call.id.trim().is_empty() {
							drop_queue.push(call.id.clone());
						}
						false
					} else {
						true
					}
				});
				new_msg.parts.retain(|part| match part {
					ContentPart::ToolCall { id, name, .. } => {
						let call = crate::state::ToolCall {
							id: id.clone(),
							name: name.clone(),
							arguments: Value::Null,
						};
						if is_malformed_tool_call(&call) {
							if !id.trim().is_empty() {
								drop_queue.push(id.clone());
							}
							false
						} else {
							true
						}
					},
					_ => true,
				});
				// `transform-messages.ts:170-229`: a turn that existed only to
				// carry malformed calls disappears entirely.
				if new_msg.tool_calls.is_empty()
					&& new_msg.content.trim().is_empty()
					&& !new_msg
						.parts
						.iter()
						.any(|part| matches!(part, ContentPart::Image { .. }))
				{
					continue;
				}
				sanitized.push(new_msg);
			},
			Role::Tool => {
				if let Some(id) = &msg.tool_call_id
					&& let Some(pos) = drop_queue.iter().position(|drop_id| drop_id == id)
				{
					drop_queue.remove(pos);
					continue;
				}
				sanitized.push(msg.clone());
			},
			_ => {
				drop_queue.clear();
				sanitized.push(msg.clone());
			},
		}
	}
	sanitized
}

pub fn preferred_dialect(model_id: &str) -> &'static str {
	let lower = model_id.to_ascii_lowercase();
	if lower.contains("claude") || lower.contains("anthropic") {
		"anthropic"
	} else if lower.contains("gemini") {
		"gemini"
	} else if lower.contains("gemma") {
		"gemma"
	} else if lower.contains("harmony") || lower.contains("gpt-oss") {
		"harmony"
	} else if lower.contains("deepseek") {
		"deepseek"
	} else if lower.contains("qwen") {
		"qwen3"
	} else if lower.contains("kimi") || lower.contains("moonshot") {
		"kimi"
	} else if lower.contains("glm") {
		"glm"
	} else if lower.contains("hermes") {
		"hermes"
	} else if lower.contains("minimax") {
		"minimax"
	} else {
		"xml"
	}
}

/// TS `renderDemotedThinking`: wrap reasoning text in the target model's dialect fences (`demotion.ts:30-41`).
pub fn demote_thinking(model_id: &str, thinking: &str) -> String {
	let dialect = preferred_dialect(model_id);
	match dialect {
		"anthropic" => thinking.to_string(),
		"harmony" | "gemma" | "deepseek" | "glm" | "kimi" | "hermes" | "qwen3" => {
			format!("<think>\n{thinking}\n</think>")
		}
		"gemini" => {
			format!("```thinking\n{thinking}\n```")
		}
		"minimax" | "xml" | _ => {
			format!("<thinking>{thinking}</thinking>")
		}
	}
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
			provider_id: None,
			on_payload: None,
			first_event_timeout: resolve_env_timeout(
				&[
					"PI_OPENAI_STREAM_FIRST_EVENT_TIMEOUT_MS",
					"PI_STREAM_FIRST_EVENT_TIMEOUT_MS",
				],
				DEFAULT_STREAM_FIRST_EVENT_TIMEOUT_MS,
			),
			idle_timeout: resolve_env_timeout(
				&[
					"PI_OPENAI_STREAM_IDLE_TIMEOUT_MS",
					"PI_STREAM_IDLE_TIMEOUT_MS",
				],
				DEFAULT_STREAM_IDLE_TIMEOUT_MS,
			),
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

	/// Stamp a configured gateway/provider id instead of the `ollama` dialect
	/// default. Empty values keep the default.
	pub fn with_provider_id(mut self, provider_id: Option<String>) -> Self {
		self.provider_id = provider_id.filter(|value| !value.trim().is_empty());
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
	/// Attach an asynchronous request payload replacement hook (`ollama.ts:577-580`).
	pub fn with_on_payload(mut self, hook: OllamaPayloadHook) -> Self {
		self.on_payload = Some(hook);
		self
	}

	/// Override first-event and idle stream watchdog timeouts (`idle-iterator.ts`).
	pub fn with_stream_timeouts(
		mut self,
		first_event_timeout: Option<Duration>,
		idle_timeout: Option<Duration>,
	) -> Self {
		self.first_event_timeout = first_event_timeout;
		self.idle_timeout = idle_timeout;
		self
	}

	/// Effective first-event budget (floored at idle timeout per `idle-iterator.ts:90`).
	/// `None` disables the watchdog exactly like the TS `0` sentinel.
	fn effective_first_event_timeout(&self) -> Option<Duration> {
		match (self.first_event_timeout, self.idle_timeout) {
			(Some(first), Some(idle)) => Some(first.max(idle)),
			(Some(first), None) => Some(first),
			(None, Some(idle)) => Some(idle),
			(None, None) => None,
		}
	}

	/// Effective stream idle gap budget; `None` disables the watchdog.
	fn effective_idle_timeout(&self) -> Option<Duration> {
		self.idle_timeout
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

	/// Check whether a historical message originated from this exact provider, API and model.
	fn is_same_model(&self, message: &Message) -> bool {
		message.provider.as_deref() == self.provider()
			&& message.api.as_deref() == self.api()
			&& message.model.as_deref() == self.model()
	}

	/// Transform historical messages for Ollama (`transform-messages.ts:470-1083`):
	/// 1. Demotes foreign/cross-model thinking to text using target dialect fences.
	/// 2. Drops cross-model redacted thinking and whitespace-only thinking.
	/// 3. Drops malformed tool calls and matched tool results.
	/// 4. Ensures tool-call / tool-result pairing: synthesizes missing results
	///    ("No result provided" or "aborted" for stop errors) and re-emits orphan
	///    results as user-role `<stale-tool-result>` blocks.
	fn transform_messages(&self, messages: &[Message]) -> Vec<Message> {
		let sanitized = sanitize_malformed_tool_calls(messages);

		// Pass 1: thinking demotion & filter truncated thinking-only assistant turns
		let mut preprocessed = Vec::with_capacity(sanitized.len());
		for msg in sanitized {
			if msg.role == Role::Assistant {
				let is_truncated = matches!(
					msg.stop_reason,
					Some(StopReason::Length) | Some(StopReason::Error) | Some(StopReason::Aborted)
				);
				let has_text_or_calls = !msg.content.trim().is_empty()
					|| !msg.tool_calls.is_empty()
					|| msg.parts.iter().any(|part| {
						matches!(part, ContentPart::Text { .. } | ContentPart::ToolCall { .. })
					});
				if is_truncated && !has_text_or_calls {
					continue;
				}

				if self.is_same_model(&msg) {
					preprocessed.push(msg);
				} else {
					let mut new_msg = msg;
					let mut demoted = Vec::with_capacity(new_msg.parts.len());
					for part in std::mem::take(&mut new_msg.parts) {
						match part {
							ContentPart::Thinking { thinking, .. } => {
								if thinking.trim().is_empty() {
									continue;
								}
								demoted.push(ContentPart::Text {
									text: demote_thinking(&self.model, &thinking),
								});
							},
							// Dropped cross-model per transform-messages.ts:728-746.
							ContentPart::RedactedThinking { .. } => {},
							other => demoted.push(other),
						}
					}
					if let Some(ContentPart::Text { text }) = demoted.last_mut() {
						*text = text.trim_end().to_string();
					}
					new_msg.parts = demoted;
					preprocessed.push(new_msg);
				}
			} else {
				preprocessed.push(msg);
			}
		}

		// Pass 2: tool-call and tool-result pairing (`transform-messages.ts:845-1083`).
		let mut output = Vec::with_capacity(preprocessed.len());
		let mut pending: Vec<(String, String)> = Vec::new();
		let mut aborted = false;

		fn flush_pending(
			pending: &mut Vec<(String, String)>,
			aborted: bool,
			output: &mut Vec<Message>,
		) {
			let text = if aborted { "aborted" } else { "No result provided" };
			for (id, name) in pending.drain(..) {
				output.push(Message::tool_with_status(id, name, text, true));
			}
		}

		for msg in preprocessed {
			if msg.role == Role::Tool {
				if let Some(id) = msg.tool_call_id.as_deref()
					&& let Some(pos) = pending.iter().position(|(pending_id, _)| pending_id == id)
				{
					pending.remove(pos);
					output.push(msg);
					continue;
				}
				// An orphan result inside an open call window cannot be paired.
				if !pending.is_empty() {
					continue;
				}
				// `convertMessages`: a result with no preceding call becomes a
				// user-role `<stale-tool-result>` block for this dialect.
				let tool_name = msg.name.as_deref().unwrap_or("unknown");
				let id = msg.tool_call_id.as_deref().unwrap_or("unknown");
				let error_attr = if msg.is_error { " is-error=\"true\"" } else { "" };
				let stale_text = format!(
					"<stale-tool-result tool=\"{tool_name}\" id=\"{id}\"{error_attr}>\n{}\n</stale-tool-result>",
					msg.content
				);
				output.push(Message::user(stale_text));
				continue;
			}

			if !pending.is_empty() {
				flush_pending(&mut pending, aborted, &mut output);
			}
			if msg.role == Role::Assistant {
				aborted = matches!(
					msg.stop_reason,
					Some(StopReason::Error) | Some(StopReason::Aborted)
				);
				for part in &msg.parts {
					if let ContentPart::ToolCall { id, name, .. } = part
						&& !pending.iter().any(|(pending_id, _)| pending_id == id)
					{
						pending.push((id.clone(), name.clone()));
					}
				}
				for call in &msg.tool_calls {
					if !pending.iter().any(|(pending_id, _)| pending_id == &call.id) {
						pending.push((call.id.clone(), call.name.clone()));
					}
				}
			}
			output.push(msg);
		}

		if !pending.is_empty() {
			flush_pending(&mut pending, aborted, &mut output);
		}

		output
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

	fn request_value(
		&self,
		messages: &[Message],
		tools: &[Value],
		tool_choice: &ToolChoice,
		stream: bool,
	) -> Result<Value, ProviderError> {
		let req = self.request(messages, tools, tool_choice, stream)?;
		serde_json::to_value(&req).map_err(|err| ProviderError::Protocol(err.to_string()))
	}

	fn convert_messages(&self, messages: &[Message]) -> Vec<OllamaWireMessage> {
		let transformed = self.transform_messages(messages);
		let messages = &transformed;
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

	async fn send_request_with_retry(
		&self,
		payload: &Value,
	) -> Result<reqwest::Response, ProviderError> {
		let pre_response_timeout = self.effective_first_event_timeout();
		let max_attempts = MAX_FETCH_ATTEMPTS;
		let secrets = credential_values(self.api_key.as_deref());

		for attempt in 0..max_attempts {
			let mut req = self.http.post(self.endpoint()).json(payload);
			if let Some(api_key) = self.api_key.as_deref() {
				req = req.bearer_auth(api_key);
			}

			let response = match pre_response_timeout {
				Some(limit) => match tokio::time::timeout(limit, req.send()).await {
					Ok(Ok(resp)) => resp,
					Ok(Err(err)) => {
						if attempt + 1 < max_attempts {
							let delay_ms = OLLAMA_RETRY_DELAYS_MS
								.get(attempt)
								.copied()
								.unwrap_or(10_000);
							tokio::time::sleep(Duration::from_millis(delay_ms)).await;
							continue;
						}
						return Err(network_error(err));
					},
					Err(_) => {
						if attempt + 1 < max_attempts {
							let delay_ms = OLLAMA_RETRY_DELAYS_MS
								.get(attempt)
								.copied()
								.unwrap_or(10_000);
							tokio::time::sleep(Duration::from_millis(delay_ms)).await;
							continue;
						}
						return Err(ProviderError::Network(format!(
							"Ollama request timed out after {limit:?}"
						)));
					},
				},
				None => match req.send().await {
					Ok(resp) => resp,
					Err(err) => {
						if attempt + 1 < max_attempts {
							let delay_ms = OLLAMA_RETRY_DELAYS_MS
								.get(attempt)
								.copied()
								.unwrap_or(10_000);
							tokio::time::sleep(Duration::from_millis(delay_ms)).await;
							continue;
						}
						return Err(network_error(err));
					},
				},
			};

			let status = response.status();
			if status.is_success() {
				return Ok(response);
			}

			let headers = response.headers().clone();
			let body_info = match read_body_bounded(response, ERROR_BODY_BYTES).await {
				Ok(body) => body,
				Err(err) => {
					return Err(ProviderError::HttpStatus(
						status.as_u16(),
						format!("<failed to read error body: {err}>"),
					));
				}
			};

			let is_retryable = should_retry_ollama_response(status.as_u16(), &body_info.text);
			if is_retryable && attempt + 1 < max_attempts {
				let retry_hint = extract_retry_hint(&headers, &body_info.text);
				let default_delay = Duration::from_millis(
					OLLAMA_RETRY_DELAYS_MS
						.get(attempt)
						.copied()
						.unwrap_or(10_000),
				);
				let delay = retry_hint
					.map(|hint| hint.min(Duration::from_millis(MAX_RETRY_DELAY_MS)))
					.unwrap_or(default_delay);
				tokio::time::sleep(delay).await;
				continue;
			}

			let redacted = bounded_redaction(&body_info.text, &secrets, body_info.truncated);
			return Err(ProviderError::HttpStatus(status.as_u16(), redacted));
		}

		Err(ProviderError::Network("maximum retry attempts exceeded".to_string()))
	}

	async fn stream_once(
		&self,
		messages: &[Message],
		tools: &[Value],
		tool_choice: &ToolChoice,
	) -> Result<ChunkStream, ProviderError> {
		let mut payload = self.request_value(messages, tools, tool_choice, true)?;
		if let Some(hook) = &self.on_payload {
			if let Some(replaced) = hook(&payload, &self.model).await? {
				payload = replaced;
			}
		}

		let response = self.send_request_with_retry(&payload).await?;
		let secrets = credential_values(self.api_key.as_deref());
		let body = response
			.bytes_stream()
			.map(|item| item.map_err(|err| err.to_string()));

		let raw_stream = ndjson_stream(body, secrets);
		let pattern = get_stream_markup_healing_pattern(self.provider().unwrap_or(""), &self.model);
		let healed_stream = with_markup_healing(raw_stream, pattern);
		let watched_stream = with_stream_watchdog(
			healed_stream,
			self.effective_first_event_timeout(),
			self.effective_idle_timeout(),
		);
		Ok(watched_stream)
	}

	async fn stream_with_empty_retry(
		&self,
		messages: &[Message],
		tools: &[Value],
		tool_choice: &ToolChoice,
	) -> Result<ChunkStream, ProviderError> {
		let provider = self.clone();
		let messages = messages.to_vec();
		let tools = tools.to_vec();
		let tool_choice = tool_choice.clone();

		struct RetryState {
			provider: OllamaProvider,
			messages: Vec<Message>,
			tools: Vec<Value>,
			tool_choice: ToolChoice,
			current_stream: Option<ChunkStream>,
			buffered: VecDeque<StreamChunk>,
			committed: bool,
			empty_retries: usize,
			finished: bool,
		}

		let init_stream = provider.stream_once(&messages, &tools, &tool_choice).await?;

		let state = RetryState {
			provider,
			messages,
			tools,
			tool_choice,
			current_stream: Some(init_stream),
			buffered: VecDeque::new(),
			committed: false,
			empty_retries: 0,
			finished: false,
		};

		Ok(Box::pin(futures::stream::unfold(Some(state), |state| async move {
			let mut state = state?;
			loop {
				if let Some(chunk) = state.buffered.pop_front() {
					return Some((Ok(chunk), Some(state)));
				}

				if state.finished {
					return None;
				}

				let next_item = match state.current_stream.as_mut() {
					Some(stream) => stream.next().await,
					None => return None,
				};

				match next_item {
					Some(Ok(chunk)) => {
						if state.committed {
							return Some((Ok(chunk), Some(state)));
						}

						if is_meaningful_chunk(&chunk) {
							state.committed = true;
							state.buffered.push_back(chunk);
							if let Some(c) = state.buffered.pop_front() {
								return Some((Ok(c), Some(state)));
							}
							continue;
						}

						if chunk.is_done {
							let output_tokens = chunk.usage.as_ref().map(|usage| usage.output).unwrap_or(0);
							let is_stop = chunk.stop_reason == Some(StopReason::Stop);
							let can_retry_empty = !state.committed
								&& is_stop
								&& chunk.error_message.is_none()
								&& output_tokens <= 1
								&& state.empty_retries < MAX_EMPTY_COMPLETION_RETRIES;

							if can_retry_empty {
								let delay_ms = EMPTY_COMPLETION_BASE_DELAY_MS * (1 << state.empty_retries);
								tokio::time::sleep(Duration::from_millis(delay_ms)).await;
								state.empty_retries += 1;
								state.buffered.clear();
								match state
									.provider
									.stream_once(&state.messages, &state.tools, &state.tool_choice)
									.await
								{
									Ok(new_stream) => {
										state.current_stream = Some(new_stream);
										continue;
									}
									Err(err) => {
										state.finished = true;
										return Some((Err(err), None));
									}
								}
							}

							state.buffered.push_back(chunk);
							state.finished = true;
							if let Some(c) = state.buffered.pop_front() {
								return Some((Ok(c), Some(state)));
							}
							return None;
						}

						state.buffered.push_back(chunk);
					}
					Some(Err(err)) => {
						state.finished = true;
						return Some((Err(err), None));
					}
					None => {
						state.finished = true;
						return None;
					}
				}
			}
		})))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMarkupHealingPattern {
	Kimi,
	Dsml,
	Thinking,
}

pub fn model_may_leak_kimi_tool_calls(provider: &str, model_id: &str) -> bool {
	if provider == "kimi-code" || provider == "moonshot" {
		return true;
	}
	let lower = model_id.to_ascii_lowercase();
	lower.contains("kimi")
		&& (lower.contains("k2")
			|| lower.contains("-k2")
			|| lower.contains("_k2")
			|| lower.contains(".k2")
			|| lower.contains("/k2"))
}

pub fn model_may_leak_dsml_tool_calls(provider: &str, model_id: &str) -> bool {
	let lower = model_id.to_ascii_lowercase();
	if !lower.contains("deepseek") {
		return false;
	}
	matches!(
		provider,
		"ollama"
			| "ollama-cloud"
			| "nvidia"
			| "deepseek"
			| "fireworks"
			| "nanogpt"
			| "opencode-go"
			| "openrouter"
	)
}

pub fn get_stream_markup_healing_pattern(
	provider: &str,
	model_id: &str,
) -> StreamMarkupHealingPattern {
	if model_may_leak_kimi_tool_calls(provider, model_id) {
		StreamMarkupHealingPattern::Kimi
	} else if model_may_leak_dsml_tool_calls(provider, model_id) {
		StreamMarkupHealingPattern::Dsml
	} else {
		StreamMarkupHealingPattern::Thinking
	}
}

pub fn normalize_kimi_function_name(raw_id: &str) -> String {
	let before_index = raw_id.split(':').next().unwrap_or(raw_id);
	let name = before_index.rsplit('.').next().unwrap_or(before_index).trim();
	name.to_string()
}

fn partial_suffix_overlap(text: &str, tag: &str) -> usize {
	let text = text.as_bytes();
	let tag = tag.as_bytes();
	let max_len = text.len().saturating_sub(1).min(tag.len().saturating_sub(1));
	for len in (1..=max_len).rev() {
		if text.ends_with(&tag[..len]) {
			return len;
		}
	}
	0
}

fn partial_suffix_overlap_any(text: &str, tags: &[&str]) -> usize {
	let mut best = 0;
	for tag in tags {
		best = best.max(partial_suffix_overlap(text, tag));
	}
	best
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealedToolCall {
	pub id: String,
	pub name: String,
	pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkupHealingEvent {
	Text(String),
	Thinking(String),
	ToolCall(HealedToolCall),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InbandScanEvent {
	Text(String),
	ThinkingDelta(String),
	ThinkingStart,
	ThinkingEnd,
	ToolCall {
		id: String,
		name: String,
		arguments: String,
	},
}

struct FencedThinkingResult {
	thinking: String,
	closed: bool,
	rest: String,
}

/// Maximum bytes held back while a fenced block or partial tag is unresolved.
/// Without it a stream that never closes its fence retains every record forever.
pub const MAX_STREAM_HOLD_BYTES: usize = 1024 * 1024;

#[derive(Default)]
struct FencedThinkingScanner {
	/// Unconsumed tail after the last committed line.
	buffer: String,
	/// Open fence descriptor (`char` + tick count); empty while outside.
	inner: String,
}

impl FencedThinkingScanner {
	fn feed(
		&mut self,
		text: &str,
		is_final: bool,
	) -> Result<FencedThinkingResult, ProviderError> {
		if self.buffer.len() + text.len() > MAX_STREAM_HOLD_BYTES {
			return Err(ProviderError::Protocol(format!(
				"fenced thinking block exceeded {MAX_STREAM_HOLD_BYTES} bytes without closing"
			)));
		}
		self.buffer.push_str(text);
		let mut thinking = String::new();

		while let Some(nl) = self.buffer.find('\n') {
			let line = self.buffer[..nl].to_string();
			if self.inner.is_empty()
				&& let Some(close_rest) = self.close_rest(&line)
			{
				let rest = format!("{close_rest}{}", &self.buffer[nl..]);
				self.reset();
				return Ok(FencedThinkingResult {
					thinking,
					closed: true,
					rest,
				});
			}
			thinking.push_str(&self.buffer[..=nl]);
			self.update_inner(&line);
			self.buffer = self.buffer[nl + 1..].to_string();
		}

		if !self.inner.is_empty() {
			thinking.push_str(&self.buffer);
			self.buffer.clear();
			return Ok(FencedThinkingResult {
				thinking,
				closed: false,
				rest: String::new(),
			});
		}

		let close = if is_final {
			self.close_rest_final(&self.buffer)
		} else {
			self.close_rest_streaming_tail(&self.buffer)
		};
		if let Some(rest) = close {
			let rest = rest.to_string();
			self.reset();
			return Ok(FencedThinkingResult {
				thinking,
				closed: true,
				rest,
			});
		}
		if !is_final && self.must_hold(&self.buffer) {
			return Ok(FencedThinkingResult {
				thinking,
				closed: false,
				rest: String::new(),
			});
		}

		thinking.push_str(&self.buffer);
		self.buffer.clear();
		if is_final {
			self.reset();
		}
		Ok(FencedThinkingResult {
			thinking,
			closed: false,
			rest: String::new(),
		})
	}

	fn parse_backtick_lead<'a>(&self, line: &'a str) -> Option<(usize, &'a str)> {
		let spaces = line.chars().take_while(|c| *c == ' ').count();
		if spaces > 3 {
			return None;
		}
		let after_spaces = &line[spaces..];
		let ticks = after_spaces.chars().take_while(|c| *c == '`').count();
		if ticks < 3 {
			return None;
		}
		Some((ticks, &after_spaces[ticks..]))
	}

	fn is_lang_token(s: &str) -> bool {
		!s.is_empty()
			&& s.chars()
				.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '+' || c == '#' || c == '-')
	}

	fn close_rest<'a>(&self, line: &'a str) -> Option<&'a str> {
		let (_, rest) = self.parse_backtick_lead(line)?;
		let trimmed = rest.trim();
		if trimmed.is_empty() {
			return Some("");
		}
		if Self::is_lang_token(trimmed) {
			return None;
		}
		Some(rest)
	}

	fn close_rest_final<'a>(&self, tail: &'a str) -> Option<&'a str> {
		let (_, rest) = self.parse_backtick_lead(tail)?;
		if rest.trim().is_empty() {
			Some("")
		} else {
			Some(rest)
		}
	}

	fn close_rest_streaming_tail<'a>(&self, tail: &'a str) -> Option<&'a str> {
		let (_, rest) = self.parse_backtick_lead(tail)?;
		let trimmed = rest.trim();
		if trimmed.is_empty() || Self::is_lang_token(trimmed) {
			return None;
		}
		Some(rest)
	}

	fn must_hold(&self, tail: &str) -> bool {
		let spaces = tail.chars().take_while(|c| *c == ' ').count();
		if spaces > 3 {
			return false;
		}
		let after_spaces = &tail[spaces..];
		let ticks = after_spaces.chars().take_while(|c| *c == '`').count();
		let rest = &after_spaces[ticks..];
		let trimmed = rest.trim();
		if trimmed.is_empty() {
			return ticks >= 1 || tail.len() <= 3;
		}
		ticks >= 3 && Self::is_lang_token(trimmed)
	}

	fn reset(&mut self) {
		self.buffer.clear();
		self.inner.clear();
	}

	fn update_inner(&mut self, line: &str) {
		let spaces = line.chars().take_while(|c| *c == ' ').count();
		if spaces > 3 {
			return;
		}
		let after = &line[spaces..];
		let (ticks, fence_char) = match after.chars().next() {
			Some('`') => (after.chars().take_while(|c| *c == '`').count(), '`'),
			Some('~') => (after.chars().take_while(|c| *c == '~').count(), '~'),
			_ => return,
		};
		if ticks < 3 {
			return;
		}
		let info = after[ticks..].trim();
		if self.inner.is_empty() {
			self.inner = format!("{fence_char}{ticks}");
		} else {
			let inner_char = self.inner.chars().next().unwrap_or('`');
			let inner_len = self.inner[1..].parse::<usize>().unwrap_or(3);
			if inner_char == fence_char && ticks >= inner_len && info.is_empty() {
				self.inner.clear();
			}
		}
	}
}

struct TagDef {
	open: &'static str,
	close: &'static str,
	fenced: bool,
}

static THINKING_TAGS: [TagDef; 7] = [
	TagDef { open: "<think>", close: "</think>", fenced: false },
	TagDef { open: "<thinking>", close: "</thinking>", fenced: false },
	TagDef { open: "<scratchpad>", close: "</scratchpad>", fenced: false },
	TagDef { open: "```thinking\n", close: "```", fenced: true },
	TagDef { open: "<|channel>thought\n", close: "<channel|>", fenced: false },
	TagDef {
		open: "<|start|>assistant<|channel|>analysis<|message|>",
		close: "<|end|>",
		fenced: false,
	},
	TagDef { open: "<|channel|>analysis<|message|>", close: "<|end|>", fenced: false },
];

static THINKING_OPENS: [&str; 7] = [
	"<think>",
	"<thinking>",
	"<scratchpad>",
	"```thinking\n",
	"<|channel>thought\n",
	"<|start|>assistant<|channel|>analysis<|message|>",
	"<|channel|>analysis<|message|>",
];

enum VisibleHit {
	Tag { index: usize, tag: &'static TagDef },
	Code { index: usize, ticks: usize },
	Hold { index: usize },
	None,
}

/// Walk char boundaries only: byte-offset slicing at an arbitrary position
/// would panic on multi-byte UTF-8 (`é`, emoji, fullwidth DSML).
fn scan_visible(buffer: &str, final_: bool) -> VisibleHit {
	for (i, ch) in buffer.char_indices() {
		let sub = &buffer[i..];
		if let Some(tag) = THINKING_TAGS.iter().find(|tag| sub.starts_with(tag.open)) {
			return VisibleHit::Tag { index: i, tag };
		}
		if !final_
			&& THINKING_OPENS
				.iter()
				.any(|open| open.len() > sub.len() && open.starts_with(sub))
		{
			return VisibleHit::Hold { index: i };
		}
		if ch == '`' {
			let ticks = sub.bytes().take_while(|byte| *byte == b'`').count();
			if !final_ && i + ticks == buffer.len() {
				return VisibleHit::Hold { index: i };
			}
			return VisibleHit::Code { index: i, ticks };
		}
	}
	VisibleHit::None
}

fn find_fence_close_end(buffer: &str, ticks: usize, final_: bool) -> Option<usize> {
	let mut start = 0;
	while start <= buffer.len() {
		let nl = buffer[start..].find('\n');
		let (line, terminated) = match nl {
			Some(idx) => (&buffer[start..start + idx], true),
			None => (&buffer[start..], false),
		};
		let trimmed = line.trim();
		if trimmed.len() >= ticks && trimmed.bytes().all(|byte| byte == b'`') && (terminated || final_)
		{
			return Some(if terminated { start + line.len() + 1 } else { buffer.len() });
		}
		if !terminated {
			break;
		}
		start += line.len() + 1;
	}
	None
}

fn find_backtick_run(buffer: &str, from: usize, ticks: usize) -> Option<usize> {
	let mut i = from;
	while i < buffer.len() {
		let pos = buffer[i..].find('`')?;
		let idx = i + pos;
		let run = buffer[idx..].bytes().take_while(|byte| *byte == b'`').count();
		if run == ticks {
			return Some(idx);
		}
		i = idx + run;
	}
	None
}

fn trailing_backtick_run(buffer: &str) -> usize {
	buffer.bytes().rev().take_while(|byte| *byte == b'`').count()
}

fn trailing_line_indent(text: &str, prior: i32) -> i32 {
	let last_nl = text.rfind('\n');
	let mut indent = if last_nl.is_none() { prior } else { 0 };
	let start = last_nl.map(|idx| idx + 1).unwrap_or(0);
	for ch in text[start..].chars() {
		if indent == -1 {
			break;
		}
		if ch == ' ' {
			indent += 1;
		} else {
			indent = -1;
		}
	}
	indent
}

fn push_thinking(delta: &str, events: &mut Vec<InbandScanEvent>) {
	if !delta.is_empty() {
		events.push(InbandScanEvent::ThinkingDelta(delta.to_string()));
	}
}

#[derive(Default)]
pub struct ThinkingInbandScanner {
	buffer: String,
	close_tag: Option<&'static str>,
	fenced: Option<FencedThinkingScanner>,
	code_ticks: usize,
	code_fenced: bool,
	line_indent: i32,
}

impl ThinkingInbandScanner {
	pub fn feed(&mut self, text: &str) -> Result<Vec<InbandScanEvent>, ProviderError> {
		if text.is_empty() {
			return Ok(Vec::new());
		}
		if self.buffer.len() + text.len() > MAX_STREAM_HOLD_BYTES {
			return Err(ProviderError::Protocol(format!(
				"thinking markup held {MAX_STREAM_HOLD_BYTES} bytes without closing"
			)));
		}
		self.buffer.push_str(text);
		self.consume(false)
	}

	pub fn flush(&mut self) -> Result<Vec<InbandScanEvent>, ProviderError> {
		let mut events = self.consume(true)?;
		if !self.buffer.is_empty() {
			if self.close_tag.is_some() {
				let tail = std::mem::take(&mut self.buffer);
				push_thinking(&tail, &mut events);
				events.push(InbandScanEvent::ThinkingEnd);
			} else {
				events.push(InbandScanEvent::Text(std::mem::take(&mut self.buffer)));
			}
		}
		self.buffer.clear();
		self.close_tag = None;
		Ok(events)
	}

	fn consume(&mut self, final_: bool) -> Result<Vec<InbandScanEvent>, ProviderError> {
		let mut events = Vec::new();
		loop {
			if self.fenced.is_some() {
				let result = {
					let fenced = self.fenced.as_mut().expect("checked above");
					fenced.feed(&self.buffer, final_)?
				};
				self.buffer = if result.closed { result.rest } else { String::new() };
				push_thinking(&result.thinking, &mut events);
				if result.closed || final_ {
					events.push(InbandScanEvent::ThinkingEnd);
					self.close_tag = None;
					self.fenced = None;
				}
				if self.fenced.is_some() {
					break;
				}
				continue;
			}

			if self.buffer.is_empty() {
				break;
			}

			if let Some(close_tag) = self.close_tag {
				if let Some(close_idx) = self.buffer.find(close_tag) {
					let pre = self.buffer[..close_idx].to_string();
					push_thinking(&pre, &mut events);
					self.buffer = self.buffer[close_idx + close_tag.len()..].to_string();
					events.push(InbandScanEvent::ThinkingEnd);
					self.close_tag = None;
					continue;
				}
				let hold = if final_ {
					0
				} else {
					partial_suffix_overlap_any(&self.buffer, &[close_tag])
				};
				let emit_end = self.buffer.len() - hold;
				let pre = self.buffer[..emit_end].to_string();
				push_thinking(&pre, &mut events);
				self.buffer = self.buffer[emit_end..].to_string();
				break;
			}

			if self.code_ticks > 0 {
				if self.emit_code(final_, &mut events) {
					continue;
				}
				break;
			}

			match scan_visible(&self.buffer, final_) {
				VisibleHit::None => {
					let all = std::mem::take(&mut self.buffer);
					self.emit_text(&all, &mut events);
					break;
				},
				VisibleHit::Hold { index } => {
					let pre = self.buffer[..index].to_string();
					self.emit_text(&pre, &mut events);
					self.buffer = self.buffer[index..].to_string();
					break;
				},
				VisibleHit::Code { index, ticks } => {
					let fenced = ticks >= 3 && (0..=3).contains(&self.line_indent);
					let pre = self.buffer[..index + ticks].to_string();
					self.emit_text(&pre, &mut events);
					self.buffer = self.buffer[index + ticks..].to_string();
					self.code_ticks = ticks;
					self.code_fenced = fenced;
					continue;
				},
				VisibleHit::Tag { index, tag } => {
					let pre = self.buffer[..index].to_string();
					self.emit_text(&pre, &mut events);
					self.buffer = self.buffer[index + tag.open.len()..].to_string();
					self.close_tag = Some(tag.close);
					if tag.fenced {
						self.fenced = Some(FencedThinkingScanner::default());
					}
					events.push(InbandScanEvent::ThinkingStart);
				},
			}
		}
		Ok(events)
	}

	fn emit_code(&mut self, final_: bool, events: &mut Vec<InbandScanEvent>) -> bool {
		if self.code_fenced {
			if let Some(end) = find_fence_close_end(&self.buffer, self.code_ticks, final_) {
				let chunk = self.buffer[..end].to_string();
				self.emit_text(&chunk, events);
				self.buffer = self.buffer[end..].to_string();
				self.code_ticks = 0;
				self.code_fenced = false;
				return true;
			}
			if final_ {
				let chunk = std::mem::take(&mut self.buffer);
				self.emit_text(&chunk, events);
				self.code_ticks = 0;
				self.code_fenced = false;
				return false;
			}
			if let Some(last_nl) = self.buffer.rfind('\n') {
				let chunk = self.buffer[..last_nl + 1].to_string();
				self.emit_text(&chunk, events);
				self.buffer = self.buffer[last_nl + 1..].to_string();
			}
			return false;
		}

		if let Some(close) = find_backtick_run(&self.buffer, 0, self.code_ticks)
			&& (final_ || close + self.code_ticks < self.buffer.len())
		{
			let end = close + self.code_ticks;
			let chunk = self.buffer[..end].to_string();
			self.emit_text(&chunk, events);
			self.buffer = self.buffer[end..].to_string();
			self.code_ticks = 0;
			return true;
		}
		let hold = if final_ { 0 } else { trailing_backtick_run(&self.buffer) };
		let emit_end = self.buffer.len() - hold;
		let chunk = self.buffer[..emit_end].to_string();
		self.emit_text(&chunk, events);
		self.buffer = self.buffer[emit_end..].to_string();
		if final_ {
			self.code_ticks = 0;
		}
		false
	}

	fn emit_text(&mut self, text: &str, events: &mut Vec<InbandScanEvent>) {
		if !text.is_empty() {
			events.push(InbandScanEvent::Text(text.to_string()));
			self.line_indent = trailing_line_indent(text, self.line_indent);
		}
	}
}

const KIMI_SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
const KIMI_SECTION_END: &str = "<|tool_calls_section_end|>";
const KIMI_CALL_BEGIN: &str = "<|tool_call_begin|>";
const KIMI_CALL_END: &str = "<|tool_call_end|>";
const KIMI_ARG_BEGIN: &str = "<|tool_call_argument_begin|>";

const KIMI_TOKENS: [&str; 5] = [
	KIMI_SECTION_BEGIN,
	KIMI_SECTION_END,
	KIMI_CALL_BEGIN,
	KIMI_CALL_END,
	KIMI_ARG_BEGIN,
];

#[derive(Default)]
pub struct KimiInbandScanner {
	buffer: String,
	in_section: bool,
}

impl KimiInbandScanner {
	pub fn feed(&mut self, text: &str) -> Result<Vec<InbandScanEvent>, ProviderError> {
		if text.is_empty() {
			return Ok(Vec::new());
		}
		if self.in_section && self.buffer.len() + text.len() > MAX_SECTION_BYTES {
			return Err(ProviderError::Protocol(format!(
				"Kimi tool-call section exceeded {MAX_SECTION_BYTES} bytes without closing"
			)));
		}
		self.buffer.push_str(text);
		self.consume(false)
	}

	pub fn flush(&mut self) -> Result<Vec<InbandScanEvent>, ProviderError> {
		self.consume(true)
	}

	fn consume(&mut self, final_: bool) -> Result<Vec<InbandScanEvent>, ProviderError> {
		let mut events = Vec::new();
		loop {
			if !self.in_section {
				if let Some(pos) = self.buffer.find(KIMI_SECTION_BEGIN) {
					if pos > 0 {
						events.push(InbandScanEvent::Text(self.buffer[..pos].to_string()));
					}
					self.buffer = self.buffer[pos + KIMI_SECTION_BEGIN.len()..].to_string();
					self.in_section = true;
					continue;
				}
				let hold = if final_ {
					0
				} else {
					partial_suffix_overlap_any(&self.buffer, &KIMI_TOKENS)
				};
				let emit_len = self.buffer.len() - hold;
				if emit_len > 0 {
					events.push(InbandScanEvent::Text(self.buffer[..emit_len].to_string()));
					self.buffer = self.buffer[emit_len..].to_string();
				}
				break;
			} else {
				if let Some(end_pos) = self.buffer.find(KIMI_SECTION_END) {
					let inner = &self.buffer[..end_pos];
					self.parse_calls(inner, &mut events);
					self.buffer = self.buffer[end_pos + KIMI_SECTION_END.len()..].to_string();
					self.in_section = false;
					continue;
				}
				if let Some(call_end_pos) = self.buffer.find(KIMI_CALL_END) {
					let slice = &self.buffer[..call_end_pos + KIMI_CALL_END.len()];
					self.parse_calls(slice, &mut events);
					self.buffer = self.buffer[call_end_pos + KIMI_CALL_END.len()..].to_string();
					continue;
				}
				if final_ {
					let tail = std::mem::take(&mut self.buffer);
					self.parse_calls(&tail, &mut events);
					self.in_section = false;
				}
				break;
			}
		}
		Ok(events)
	}

	fn parse_calls(&self, s: &str, events: &mut Vec<InbandScanEvent>) {
		let mut rest = s;
		while let Some(begin) = rest.find(KIMI_CALL_BEGIN) {
			let after_begin = &rest[begin + KIMI_CALL_BEGIN.len()..];
			let Some(arg_idx) = after_begin.find(KIMI_ARG_BEGIN) else {
				break;
			};
			let raw_name = after_begin[..arg_idx].trim();
			let name = normalize_kimi_function_name(raw_name);
			let after_arg = &after_begin[arg_idx + KIMI_ARG_BEGIN.len()..];
			let Some(end_idx) = after_arg.find(KIMI_CALL_END) else {
				break;
			};
			let raw_args = after_arg[..end_idx].trim();
			let arguments = if raw_args.is_empty() {
				"{}".to_string()
			} else {
				raw_args.to_string()
			};
			let id = format!("call_{}", &uuid_like_hex()[..24]);
			events.push(InbandScanEvent::ToolCall { id, name, arguments });
			rest = &after_arg[end_idx + KIMI_CALL_END.len()..];
		}
	}
}

const DSML_OPENS: [&str; 2] = ["<｜DSML｜tool_calls>", "<|DSML|tool_calls>"];
const DSML_CLOSES: [&str; 2] = ["</｜DSML｜tool_calls>", "</|DSML|tool_calls>"];
const DSML_INVOKE_OPENS: [&str; 2] = ["<｜DSML｜invoke", "<|DSML|invoke"];
const DSML_INVOKE_CLOSES: [&str; 2] = ["</｜DSML｜invoke>", "</|DSML|invoke>"];
const DSML_PARAM_OPENS: [&str; 2] = ["<｜DSML｜parameter", "<|DSML|parameter"];
const DSML_PARAM_CLOSES: [&str; 2] = ["</｜DSML｜parameter>", "</|DSML|parameter>"];

#[derive(Default)]
pub struct DeepSeekInbandScanner {
	buffer: String,
	in_section: bool,
}

impl DeepSeekInbandScanner {
	pub fn feed(&mut self, text: &str) -> Result<Vec<InbandScanEvent>, ProviderError> {
		if text.is_empty() {
			return Ok(Vec::new());
		}
		if self.in_section && self.buffer.len() + text.len() > MAX_SECTION_BYTES {
			return Err(ProviderError::Protocol(format!(
				"DSML tool-call section exceeded {MAX_SECTION_BYTES} bytes without closing"
			)));
		}
		self.buffer.push_str(text);
		self.consume(false)
	}

	pub fn flush(&mut self) -> Result<Vec<InbandScanEvent>, ProviderError> {
		self.consume(true)
	}

	fn consume(&mut self, final_: bool) -> Result<Vec<InbandScanEvent>, ProviderError> {
		let mut events = Vec::new();
		loop {
			if !self.in_section {
				let open_hit = DSML_OPENS
					.iter()
					.filter_map(|op| self.buffer.find(op).map(|idx| (idx, *op)))
					.min_by_key(|(idx, _)| *idx);
				if let Some((pos, op)) = open_hit {
					if pos > 0 {
						events.push(InbandScanEvent::Text(self.buffer[..pos].to_string()));
					}
					self.buffer = self.buffer[pos + op.len()..].to_string();
					self.in_section = true;
					continue;
				}
				let hold = if final_ {
					0
				} else {
					partial_suffix_overlap_any(&self.buffer, &DSML_OPENS)
				};
				let emit_len = self.buffer.len() - hold;
				if emit_len > 0 {
					events.push(InbandScanEvent::Text(self.buffer[..emit_len].to_string()));
					self.buffer = self.buffer[emit_len..].to_string();
				}
				break;
			} else {
				let close_hit = DSML_CLOSES
					.iter()
					.filter_map(|cl| self.buffer.find(cl).map(|idx| (idx, *cl)))
					.min_by_key(|(idx, _)| *idx);
				if let Some((pos, cl)) = close_hit {
					let inner = &self.buffer[..pos];
					self.parse_invokes(inner, &mut events);
					self.buffer = self.buffer[pos + cl.len()..].to_string();
					self.in_section = false;
					continue;
				}
				let invoke_close_hit = DSML_INVOKE_CLOSES
					.iter()
					.filter_map(|cl| self.buffer.find(cl).map(|idx| (idx, *cl)))
					.min_by_key(|(idx, _)| *idx);
				if let Some((pos, cl)) = invoke_close_hit {
					let slice = &self.buffer[..pos + cl.len()];
					self.parse_invokes(slice, &mut events);
					self.buffer = self.buffer[pos + cl.len()..].to_string();
					continue;
				}
				if final_ {
					let tail = std::mem::take(&mut self.buffer);
					self.parse_invokes(&tail, &mut events);
					self.in_section = false;
				}
				break;
			}
		}
		Ok(events)
	}

	fn parse_invokes(&self, s: &str, events: &mut Vec<InbandScanEvent>) {
		let mut rest = s;
		while let Some((pos, op)) = DSML_INVOKE_OPENS
			.iter()
			.filter_map(|op| rest.find(op).map(|idx| (idx, *op)))
			.min_by_key(|(idx, _)| *idx)
		{
			let after_op = &rest[pos + op.len()..];
			let Some(tag_end) = after_op.find('>') else {
				break;
			};
			let attr_part = &after_op[..tag_end];
			let name = extract_attr(attr_part, "name").unwrap_or_else(|| "unknown".to_string());
			let body_start = &after_op[tag_end + 1..];
			let Some((close_pos, close_tag)) = DSML_INVOKE_CLOSES
				.iter()
				.filter_map(|cl| body_start.find(cl).map(|idx| (idx, *cl)))
				.min_by_key(|(idx, _)| *idx)
			else {
				break;
			};
			let body = &body_start[..close_pos];
			let arguments = self.parse_parameters(body);
			let id = format!("call_{}", &uuid_like_hex()[..24]);
			events.push(InbandScanEvent::ToolCall { id, name, arguments });
			rest = &body_start[close_pos + close_tag.len()..];
		}
	}

	fn parse_parameters(&self, s: &str) -> String {
		let mut map = serde_json::Map::new();
		let mut rest = s;
		while let Some((pos, op)) = DSML_PARAM_OPENS
			.iter()
			.filter_map(|op| rest.find(op).map(|idx| (idx, *op)))
			.min_by_key(|(idx, _)| *idx)
		{
			let after_op = &rest[pos + op.len()..];
			let Some(tag_end) = after_op.find('>') else {
				break;
			};
			let attr_part = &after_op[..tag_end];
			let name = extract_attr(attr_part, "name").unwrap_or_default();
			if let Some(val_str) = extract_attr(attr_part, "string") {
				map.insert(name, Value::String(val_str));
				rest = &after_op[tag_end + 1..];
				continue;
			}
			let body_start = &after_op[tag_end + 1..];
			if let Some((close_pos, close_tag)) = DSML_PARAM_CLOSES
				.iter()
				.filter_map(|cl| body_start.find(cl).map(|idx| (idx, *cl)))
				.min_by_key(|(idx, _)| *idx)
			{
				let raw_val = body_start[..close_pos].trim();
				let parsed = serde_json::from_str(raw_val)
					.unwrap_or_else(|_| Value::String(raw_val.to_string()));
				map.insert(name, parsed);
				rest = &body_start[close_pos + close_tag.len()..];
			} else {
				break;
			}
		}
		Value::Object(map).to_string()
	}
}

fn extract_attr(s: &str, attr_name: &str) -> Option<String> {
	let key = format!("{attr_name}=\"");
	if let Some(idx) = s.find(&key) {
		let after = &s[idx + key.len()..];
		if let Some(end) = after.find('"') {
			return Some(after[..end].to_string());
		}
	}
	let key_single = format!("{attr_name}='");
	if let Some(idx) = s.find(&key_single) {
		let after = &s[idx + key_single.len()..];
		if let Some(end) = after.find('\'') {
			return Some(after[..end].to_string());
		}
	}
	None
}

fn uuid_like_hex() -> String {
	static COUNTER: AtomicU64 = AtomicU64::new(1);
	let count = COUNTER.fetch_add(1, Ordering::Relaxed);
	let now = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map(|d| d.as_nanos())
		.unwrap_or(0);
	format!("{:016x}{:016x}", now, count)
}

pub struct MarkupHealer {
	kimi: Option<KimiInbandScanner>,
	dsml: Option<DeepSeekInbandScanner>,
	thinking: ThinkingInbandScanner,
}

impl MarkupHealer {
	pub fn new(pattern: StreamMarkupHealingPattern) -> Self {
		let kimi = if pattern == StreamMarkupHealingPattern::Kimi {
			Some(KimiInbandScanner::default())
		} else {
			None
		};
		let dsml = if pattern == StreamMarkupHealingPattern::Dsml {
			Some(DeepSeekInbandScanner::default())
		} else {
			None
		};
		Self {
			kimi,
			dsml,
			thinking: ThinkingInbandScanner::default(),
		}
	}

	pub fn feed_events(
		&mut self,
		text: &str,
	) -> Result<Vec<MarkupHealingEvent>, ProviderError> {
		if text.is_empty() {
			return Ok(Vec::new());
		}
		let scan_events = if self.kimi.is_some() {
			let tool_events = self.kimi.as_mut().expect("checked above").feed(text)?;
			self.heal_thinking(tool_events)?
		} else if self.dsml.is_some() {
			let tool_events = self.dsml.as_mut().expect("checked above").feed(text)?;
			self.heal_thinking(tool_events)?
		} else {
			self.thinking.feed(text)?
		};
		Ok(self.convert_events(scan_events))
	}

	pub fn feed_events_without_calls(
		&mut self,
		text: &str,
	) -> Result<Vec<MarkupHealingEvent>, ProviderError> {
		Ok(self
			.feed_events(text)?
			.into_iter()
			.filter(|event| !matches!(event, MarkupHealingEvent::ToolCall(_)))
			.collect())
	}

	pub fn flush_events(&mut self) -> Result<Vec<MarkupHealingEvent>, ProviderError> {
		let mut scan_events = Vec::new();
		if self.kimi.is_some() {
			let tool_events = self.kimi.as_mut().expect("checked above").flush()?;
			scan_events.extend(self.heal_thinking(tool_events)?);
		} else if self.dsml.is_some() {
			let tool_events = self.dsml.as_mut().expect("checked above").flush()?;
			scan_events.extend(self.heal_thinking(tool_events)?);
		}
		scan_events.extend(self.thinking.flush()?);
		Ok(self.convert_events(scan_events))
	}

	fn heal_thinking(
		&mut self,
		events: Vec<InbandScanEvent>,
	) -> Result<Vec<InbandScanEvent>, ProviderError> {
		let mut out = Vec::new();
		for event in events {
			match event {
				InbandScanEvent::Text(text) => out.extend(self.thinking.feed(&text)?),
				other => out.push(other),
			}
		}
		Ok(out)
	}

	fn convert_events(&self, events: Vec<InbandScanEvent>) -> Vec<MarkupHealingEvent> {
		let mut out = Vec::new();
		for event in events {
			match event {
				InbandScanEvent::Text(text) => out.push(MarkupHealingEvent::Text(text)),
				InbandScanEvent::ThinkingDelta(delta) => {
					if !delta.is_empty() {
						out.push(MarkupHealingEvent::Thinking(delta));
					}
				}
				InbandScanEvent::ToolCall { id, name, arguments } => {
					out.push(MarkupHealingEvent::ToolCall(HealedToolCall {
						id,
						name,
						arguments,
					}));
				}
				InbandScanEvent::ThinkingStart | InbandScanEvent::ThinkingEnd => {}
			}
		}
		out
	}
}

fn with_markup_healing(
	stream: ChunkStream,
	pattern: StreamMarkupHealingPattern,
) -> ChunkStream {
	struct HealingState {
		stream: ChunkStream,
		healer: MarkupHealer,
		saw_native_tool_calls: bool,
		saw_healed_tool_calls: bool,
		tool_index: usize,
		pending_queue: VecDeque<StreamChunk>,
		finished: bool,
	}

	let state = HealingState {
		stream,
		healer: MarkupHealer::new(pattern),
		saw_native_tool_calls: false,
		saw_healed_tool_calls: false,
		tool_index: 0,
		pending_queue: VecDeque::new(),
		finished: false,
	};

	Box::pin(futures::stream::unfold(Some(state), |state| async move {
		let mut state = state?;
		loop {
			if let Some(chunk) = state.pending_queue.pop_front() {
				return Some((Ok(chunk), Some(state)));
			}
			if state.finished {
				return None;
			}

			match state.stream.next().await {
				Some(Ok(chunk)) => {
					if chunk.is_done {
						let flush_events = match state.healer.flush_events() {
							Ok(events) => events,
							Err(err) => {
								state.finished = true;
								return Some((Err(err), None));
							},
						};
						for ev in flush_events {
							match ev {
								MarkupHealingEvent::Text(t) => {
									state.pending_queue.push_back(delta_chunk(Some(t), None, Vec::new(), 0));
								}
								MarkupHealingEvent::Thinking(th) => {
									state.pending_queue.push_back(delta_chunk(None, Some(th), Vec::new(), 0));
								}
								MarkupHealingEvent::ToolCall(call) => {
									state.saw_healed_tool_calls = true;
									let idx = state.tool_index;
									state.tool_index += 1;
									state.pending_queue.push_back(delta_chunk(
										None,
										None,
										vec![ToolCallDelta {
											index: idx,
											id: Some(call.id),
											name: Some(call.name),
											arguments_delta: Some(call.arguments),
											content_index: Some(0),
										}],
										0,
									));
								}
							}
						}

						let final_stop_reason = if state.saw_healed_tool_calls
							&& chunk.stop_reason == Some(StopReason::Stop)
						{
							StopReason::ToolUse
						} else {
							chunk.stop_reason.unwrap_or(StopReason::Stop)
						};

						state.pending_queue.push_back(done_chunk(
							chunk.usage,
							final_stop_reason,
							chunk.error_message,
						));
						state.finished = true;
						if let Some(c) = state.pending_queue.pop_front() {
							return Some((Ok(c), Some(state)));
						}
						return None;
					}

					if !chunk.tool_call_deltas.is_empty() {
						state.saw_native_tool_calls = true;
					}
					let content_idx = chunk.content_index.unwrap_or(0);
					let events = if let Some(text) = &chunk.content {
						let result = if state.saw_native_tool_calls {
							state.healer.feed_events_without_calls(text)
						} else {
							state.healer.feed_events(text)
						};
						match result {
							Ok(events) => events,
							Err(err) => {
								state.finished = true;
								return Some((Err(err), None));
							},
						}
					} else {
						Vec::new()
					};

					let mut emitted = false;
					for ev in events {
						match ev {
							MarkupHealingEvent::Text(t) => {
								state.pending_queue.push_back(delta_chunk(Some(t), None, Vec::new(), content_idx));
								emitted = true;
							}
							MarkupHealingEvent::Thinking(th) => {
								state.pending_queue.push_back(delta_chunk(None, Some(th), Vec::new(), content_idx));
								emitted = true;
							}
							MarkupHealingEvent::ToolCall(call) => {
								state.saw_healed_tool_calls = true;
								let idx = state.tool_index;
								state.tool_index += 1;
								state.pending_queue.push_back(delta_chunk(
									None,
									None,
									vec![ToolCallDelta {
										index: idx,
										id: Some(call.id),
										name: Some(call.name),
										arguments_delta: Some(call.arguments),
										content_index: Some(content_idx),
									}],
									content_idx,
								));
								emitted = true;
							}
						}
					}

					if chunk.reasoning.is_some() || !chunk.tool_call_deltas.is_empty() {
						state.pending_queue.push_back(delta_chunk(
							None,
							chunk.reasoning,
							chunk.tool_call_deltas,
							content_idx,
						));
						emitted = true;
					}

					if emitted {
						if let Some(c) = state.pending_queue.pop_front() {
							return Some((Ok(c), Some(state)));
						}
					}
				}
				Some(Err(err)) => {
					state.finished = true;
					return Some((Err(err), None));
				}
				None => {
					state.finished = true;
					return None;
				}
			}
		}
	}))
}

fn is_progress_chunk(chunk: &StreamChunk) -> bool {
	chunk.content.as_ref().map(|s| !s.is_empty()).unwrap_or(false)
		|| chunk.reasoning.as_ref().map(|s| !s.is_empty()).unwrap_or(false)
		|| !chunk.tool_call_deltas.is_empty()
		|| chunk.is_done
}

fn is_meaningful_chunk(chunk: &StreamChunk) -> bool {
	if chunk.is_done {
		return false;
	}
	chunk.content.as_ref().map(|s| !s.trim().is_empty()).unwrap_or(false)
		|| chunk.reasoning.as_ref().map(|s| !s.trim().is_empty()).unwrap_or(false)
		|| !chunk.tool_call_deltas.is_empty()
}

fn with_stream_watchdog(
	stream: ChunkStream,
	first_event_timeout: Option<Duration>,
	idle_timeout: Option<Duration>,
) -> ChunkStream {
	let start_time = tokio::time::Instant::now();

	struct WatchdogState {
		stream: ChunkStream,
		first_deadline: Option<tokio::time::Instant>,
		idle_timeout: Option<Duration>,
		awaiting_first_item: bool,
		last_progress: tokio::time::Instant,
	}

	let state = WatchdogState {
		stream,
		first_deadline: first_event_timeout.map(|d| start_time + d),
		idle_timeout,
		awaiting_first_item: true,
		last_progress: tokio::time::Instant::now(),
	};

	Box::pin(futures::stream::unfold(Some(state), |state| async move {
		let mut state = state?;
		let now = tokio::time::Instant::now();

		let timeout_duration = if state.awaiting_first_item {
			if let Some(deadline) = state.first_deadline {
				if now >= deadline {
					return Some((
						Err(ProviderError::Network(
							"Ollama stream timed out waiting for first event".to_string(),
						)),
						None,
					));
				}
				Some(deadline - now)
			} else {
				None
			}
		} else if let Some(idle) = state.idle_timeout {
			let deadline = state.last_progress + idle;
			if now >= deadline {
				return Some((
					Err(ProviderError::Network(
						"Ollama stream idle timeout exceeded".to_string(),
					)),
					None,
				));
			}
			Some(deadline - now)
		} else {
			None
		};

		let next_item = if let Some(duration) = timeout_duration {
			match tokio::time::timeout(duration, state.stream.next()).await {
				Ok(item) => item,
				Err(_) => {
					let msg = if state.awaiting_first_item {
						"Ollama stream timed out waiting for first event"
					} else {
						"Ollama stream idle timeout exceeded"
					};
					return Some((Err(ProviderError::Network(msg.to_string())), None));
				}
			}
		} else {
			state.stream.next().await
		};

		match next_item {
			Some(Ok(chunk)) => {
				if is_progress_chunk(&chunk) {
					state.awaiting_first_item = false;
					state.last_progress = tokio::time::Instant::now();
				}
				Some((Ok(chunk), Some(state)))
			}
			Some(Err(err)) => Some((Err(err), None)),
			None => None,
		}
	}))
}

pub fn is_llama_cpp_parse_error(body: &str) -> bool {
	let lower = body.to_ascii_lowercase();
	lower.contains("failed to parse tool call arguments as json")
		|| lower.contains("[json.exception.parse_error.101]")
}

pub fn should_retry_ollama_response(status: u16, body_text: &str) -> bool {
	if (500..=599).contains(&status) {
		return !is_llama_cpp_parse_error(body_text);
	}
	status == 429
}

pub fn extract_retry_hint(headers: &reqwest::header::HeaderMap, body: &str) -> Option<Duration> {
	if let Some(val) = headers.get("retry-after-ms").and_then(|v| v.to_str().ok()) {
		if let Ok(ms) = val.trim().parse::<u64>() {
			return Some(Duration::from_millis(ms));
		}
	}
	if let Some(val) = headers.get("retry-after").and_then(|v| v.to_str().ok()) {
		if let Ok(secs) = val.trim().parse::<u64>() {
			return Some(Duration::from_secs(secs));
		}
	}
	if let Some(val) = headers.get("x-ratelimit-reset-ms").and_then(|v| v.to_str().ok()) {
		if let Ok(ms) = val.trim().parse::<u64>() {
			return Some(Duration::from_millis(ms));
		}
	}
	if let Some(val) = headers.get("x-ratelimit-reset").and_then(|v| v.to_str().ok()) {
		if let Ok(secs) = val.trim().parse::<u64>() {
			return Some(Duration::from_secs(secs));
		}
	}
	if let Some(val) = headers.get("x-ratelimit-reset-after").and_then(|v| v.to_str().ok()) {
		if let Ok(secs) = val.trim().parse::<u64>() {
			return Some(Duration::from_secs(secs));
		}
	}
	let lower = body.to_ascii_lowercase();
	for keyword in &["try again in ", "retry in ", "reset in ", "please retry after "] {
		if let Some(idx) = lower.find(keyword) {
			let rest = &lower[idx + keyword.len()..];
			let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
			if let Ok(val) = num_str.parse::<u64>() {
				let after_num = rest[num_str.len()..].trim_start();
				if after_num.starts_with("ms") || after_num.starts_with("millisecond") {
					return Some(Duration::from_millis(val));
				} else {
					return Some(Duration::from_secs(val));
				}
			}
		}
	}
	None
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

	/// TS `model.provider`: configured gateway id, else local `ollama` /
	/// `ollama-cloud`.
	fn provider(&self) -> Option<&str> {
		self
			.provider_id
			.as_deref()
			.or(Some(if self.cloud { "ollama-cloud" } else { "ollama" }))
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
		self.stream_with_empty_retry(messages, tools, tool_choice).await
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
	/// Bytes after the last newline; checked before any segment is copied so a
	/// completed record cannot exceed the ceiling.
	pending_record: usize,
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
		pending_record: 0,
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
					// Validate every segment before copying it: a chunk carrying a
					// newline must not smuggle a complete record past the ceiling.
					let mut rest = bytes.as_ref();
					loop {
						let newline = rest.iter().position(|byte| *byte == b'\n');
						let (segment, tail) = match newline {
							Some(position) => (&rest[..position], &rest[position + 1..]),
							None => (rest, &rest[rest.len()..]),
						};
						if state.pending_record + segment.len() > MAX_RECORD_BYTES {
							state.queue.push_back(Ok(error_chunk(format!(
								"Ollama stream record exceeded {MAX_RECORD_BYTES} bytes"
							))));
							state.finished = true;
							break;
						}
						state.buffer.extend_from_slice(segment);
						state.pending_record += segment.len();
						if newline.is_none() {
							break;
						}
						state.buffer.push(b'\n');
						state.pending_record = 0;
						rest = tail;
						if rest.is_empty() {
							break;
						}
					}
					if state.finished {
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
