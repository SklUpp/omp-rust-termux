//! Persisted child contract and cold-revive factory.
//!
//! Mirrors `packages/coding-agent/src/task/persisted-revive.ts`: a parked child
//! keeps only its transcript (`<session root>/<id>.jsonl`) and the
//! `session_init`-equivalent contract recorded beside it
//! (`<session root>/<id>.init.json`). Reviving reopens the JSONL, replays the
//! real messages into a fresh [`AgentState`], and rebuilds an [`AgentLoop`] from
//! the persisted contract instead of the current top-level session options — a
//! bare reopen would otherwise resurrect a wrong (root) session.
//!
//! No persisted contract, or a recorded workspace that no longer exists, means
//! the child stays transcript-only (`history://<id>`); [`PersistedChildSession::load`]
//! returns `None` in exactly those cases.

use std::{
	fs,
	io::{BufRead, BufReader, Write},
	path::{Path, PathBuf},
	sync::Arc,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
	loop_module::AgentLoop,
	provider::ModelProvider,
	state::{
		AgentConfig, AgentState, Message, PlanModeGuard, Role, ToolApprover,
	},
	supervisor::Supervisor,
	tools::registry::ToolRegistry,
};

/// Contract file suffix, next to the child transcript.
pub const SESSION_INIT_SUFFIX: &str = ".init.json";
/// Marker written when a hard kill tombstones an agent id.
pub const TOMBSTONE_SUFFIX: &str = ".tombstone";

const DEFAULT_MAX_TURNS: usize = 20;
const DEFAULT_CONTEXT_LIMIT: usize = 100_000;

fn default_max_turns() -> usize {
	DEFAULT_MAX_TURNS
}

fn default_context_limit() -> usize {
	DEFAULT_CONTEXT_LIMIT
}

/// One child transcript path in `root`.
#[must_use]
pub fn transcript_path(root: &Path, id: &str) -> PathBuf {
	root.join(format!("{id}.jsonl"))
}

/// One child contract path in `root`.
#[must_use]
pub fn init_path(root: &Path, id: &str) -> PathBuf {
	root.join(format!("{id}{SESSION_INIT_SUFFIX}"))
}

/// One tombstone marker path in `root`.
#[must_use]
pub fn tombstone_path(root: &Path, id: &str) -> PathBuf {
	root.join(format!("{id}{TOMBSTONE_SUFFIX}"))
}

fn valid_id(id: &str) -> bool {
	!id.is_empty()
		&& id
			.chars()
			.all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Persisted `session_init` contract for one subagent.
///
/// Field names and meaning follow the TypeScript `SessionInitEntry`. The two
/// roster-only fields (`displayName`, `parentId`) and the workspace (`cwd`) are
/// extra: the TypeScript build reads them from the session header and the agent
/// registry, while a standalone transcript needs them recorded once to restore
/// a parked row after a process restart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersistedChildSession {
	pub id: String,
	/// Agent definition name (for example `scout` or `reviewer`).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub agent: Option<String>,
	/// Initial task text this child ran.
	#[serde(default)]
	pub task: String,
	/// Full system prompt sent to the model.
	#[serde(default)]
	pub system_prompt: String,
	/// Tool names the child was offered (explicit grants on revive).
	#[serde(default)]
	pub tools: Vec<String>,
	/// Semantic model role declared by the agent, retained after resolution.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub model_role: Option<String>,
	/// Initially resolved `provider/model` selector.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub resolved_model: Option<String>,
	/// Structured output schema, when one was requested.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub output_schema: Option<Value>,
	/// Enforcement policy recorded with the output schema.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub output_schema_mode: Option<String>,
	/// Whether revival must retain only the explicitly persisted tool names.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub restrict_tool_names: Option<bool>,
	/// Spawn allowlist the child ran with (`""` = none, `"*"` = any, else CSV).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub spawns: Option<String>,
	/// The agent's `readSummarize` setting.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub read_summarize: Option<bool>,
	/// Effective advisor: `"on"` = advisor-role model, else an explicit pattern.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub advisor: Option<String>,
	/// Whether the agent definition is read-only.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub read_only: Option<bool>,
	/// Recorded workspace. A gone directory makes the contract unusable.
	pub cwd: PathBuf,
	#[serde(default = "default_max_turns")]
	pub max_turns: usize,
	#[serde(default = "default_context_limit")]
	pub context_limit: usize,
	/// Roster-only display name (`task.name`); never an identity.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub display_name: Option<String>,
	/// Roster-only parent agent id.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub parent_id: Option<String>,
}

impl PersistedChildSession {
	/// Write the contract next to the child transcript.
	pub fn write(&self, root: &Path) -> Result<PathBuf, String> {
		if !valid_id(&self.id) {
			return Err(format!("invalid child id for session contract: {:?}", self.id));
		}
		fs::create_dir_all(root)
			.map_err(|error| format!("cannot create {}: {error}", root.display()))?;
		let path = init_path(root, &self.id);
		let tmp = path.with_extension("json.tmp");
		let bytes = serde_json::to_vec_pretty(self)
			.map_err(|error| format!("cannot serialize child contract: {error}"))?;
		fs::write(&tmp, bytes)
			.map_err(|error| format!("cannot write {}: {error}", tmp.display()))?;
		fs::rename(&tmp, &path)
			.map_err(|error| format!("cannot commit {}: {error}", path.display()))?;
		Ok(path)
	}

	/// Load a usable contract for `id` from `root`.
	///
	/// `None` means the child is transcript-only: no contract was recorded, the
	/// contract does not belong to `id`, the transcript is missing, or the
	/// recorded workspace is gone (isolated/merged worktree, moved dir). A
	/// tombstoned id is not revivable either.
	pub fn load(root: &Path, id: &str) -> Option<Self> {
		if !valid_id(id) || tombstone_path(root, id).exists() {
			return None;
		}
		let raw = fs::read(init_path(root, id)).ok()?;
		let parsed: Self = serde_json::from_slice(&raw).ok()?;
		if parsed.id != id || !parsed.cwd.is_dir() {
			return None;
		}
		if !transcript_path(root, id).is_file() {
			return None;
		}
		Some(parsed)
	}

	/// Reopen the transcript and contract for one parked child.
	pub fn reopen(root: &Path, id: &str) -> Option<(Self, Vec<Message>)> {
		let contract = Self::load(root, id)?;
		let messages = load_transcript(&transcript_path(root, id)).ok()?;
		Some((contract, messages))
	}
}

/// Read a child transcript into real messages.
///
/// The executor writes `{"type":"message","message":{...}}` lines; a raw
/// message object is accepted too so older/foreign files are not fatal. A
/// malformed line is skipped rather than destroying the whole replay, matching
/// a crash-truncated trailing write.
pub fn load_transcript(path: &Path) -> Result<Vec<Message>, String> {
	let file =
		fs::File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
	let mut messages = Vec::new();
	for line in BufReader::new(file).lines() {
		let Ok(line) = line else { continue };
		let line = line.trim();
		if line.is_empty() {
			continue;
		}
		let Ok(value) = serde_json::from_str::<Value>(line) else {
			continue;
		};
		let candidate = value.get("message").cloned().unwrap_or(value);
		if let Ok(message) = serde_json::from_value::<Message>(candidate) {
			messages.push(message);
		}
	}
	Ok(messages)
}

/// Append new messages to a child transcript in the executor's exact format.
pub fn append_messages(path: &Path, messages: &[&Message]) {
	if messages.is_empty() {
		return;
	}
	let mut file = match fs::OpenOptions::new().create(true).append(true).open(path) {
		Ok(file) => file,
		Err(error) => {
			eprintln!("warning: cannot open child transcript {}: {error}", path.display());
			return;
		},
	};
	for message in messages {
		let Ok(message) = serde_json::to_value(message) else {
			continue;
		};
		let mut line = serde_json::json!({ "type": "message", "message": message }).to_string();
		line.push('\n');
		if let Err(error) = file.write_all(line.as_bytes()) {
			eprintln!("warning: cannot append child transcript {}: {error}", path.display());
			return;
		}
	}
}

/// A rebuilt child: the loop and the replayed state it runs against.
pub struct RevivedChild {
	pub agent_loop: AgentLoop,
	pub state:     AgentState,
}

/// Host-owned cold-revive factory (upstream
/// `PersistedSubagentReviverFactory`).
///
/// Provider/auth construction stays host-owned exactly as in the TypeScript
/// reviver context: the factory receives the persisted contract so it can honor
/// `modelRole`/`resolvedModel`; the default implementation uses the provider it
/// was built with and applies the contract's tool grants.
pub trait PersistedChildReviverFactory: Send + Sync {
	fn revive(
		&self,
		contract: &PersistedChildSession,
		replayed: Vec<Message>,
		supervisor: Option<&Arc<Supervisor>>,
	) -> Result<RevivedChild, String>;
}

/// Default factory: one provider snapshot plus the session tool template.
pub struct DefaultPersistedReviverFactory {
	provider: Arc<dyn ModelProvider>,
	registry: ToolRegistry,
	approver: Option<Arc<dyn ToolApprover>>,
	plan:     Option<Arc<dyn PlanModeGuard>>,
}

impl DefaultPersistedReviverFactory {
	#[must_use]
	pub fn new(provider: Arc<dyn ModelProvider>, registry: ToolRegistry) -> Self {
		Self { provider, registry, approver: None, plan: None }
	}

	#[must_use]
	pub fn with_approver(mut self, approver: Option<Arc<dyn ToolApprover>>) -> Self {
		self.approver = approver;
		self
	}

	#[must_use]
	pub fn with_plan(mut self, plan: Option<Arc<dyn PlanModeGuard>>) -> Self {
		self.plan = plan;
		self
	}
}

/// Apply the persisted explicit tool grants to the host template.
///
/// Mirrors the TypeScript replay: the enabled set recorded at spawn is an
/// explicit grant list, `yield` is the framework's mandatory return channel,
/// and a read-only agent never regains `write`.
fn apply_contract_tools(registry: &mut ToolRegistry, contract: &PersistedChildSession) {
	let mut allowed: std::collections::HashSet<&str> =
		contract.tools.iter().map(String::as_str).collect();
	allowed.insert("yield");
	if contract.read_only == Some(true) {
		allowed.remove("write");
	}
	for name in registry.names() {
		if !allowed.contains(name.as_str()) {
			registry.remove(&name);
		}
	}
}

impl PersistedChildReviverFactory for DefaultPersistedReviverFactory {
	fn revive(
		&self,
		contract: &PersistedChildSession,
		replayed: Vec<Message>,
		supervisor: Option<&Arc<Supervisor>>,
	) -> Result<RevivedChild, String> {
		if !contract.cwd.is_dir() {
			return Err(format!(
				"recorded workspace {} no longer exists; agent \"{}\" is transcript-only \
				 (history://{})",
				contract.cwd.display(),
				contract.id,
				contract.id
			));
		}
		let mut registry = self.registry.clone();
		apply_contract_tools(&mut registry, contract);
		let replayed_turns = replayed.iter().filter(|m| m.role == Role::Assistant).count();
		let config = AgentConfig {
			max_turns:     contract.max_turns.saturating_add(replayed_turns),
			context_limit: contract.context_limit.max(1),
			system_prompt: if contract.system_prompt.is_empty() {
				None
			} else {
				Some(contract.system_prompt.clone())
			},
			cwd:           contract.cwd.clone(),
		};
		let mut agent_loop = match &self.approver {
			Some(approver) => AgentLoop::with_approver(
				config,
				Arc::clone(&self.provider),
				registry,
				Arc::clone(approver),
			),
			None => AgentLoop::new(config, Arc::clone(&self.provider), registry),
		};
		if let Some(plan) = &self.plan {
			agent_loop = agent_loop.with_plan(Some(Arc::clone(plan)));
		}
		if let Some(supervisor) = supervisor {
			agent_loop = agent_loop
				.with_background_results(supervisor.owner_results_sink(Some(&contract.id)))
				.with_peer_inbox(supervisor.mailbox(), contract.id.clone());
		}
		Ok(RevivedChild {
			agent_loop,
			state: AgentState {
				messages:    replayed,
				total_turns: replayed_turns,
			},
		})
	}
}
