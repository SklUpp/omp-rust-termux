//! Subagent park/revive lifecycle owner.
//!
//! Mirrors `packages/coding-agent/src/registry/agent-lifecycle.ts` for the
//! subset the Rust supervisor can support without touching its settle,
//! accounting, or retention paths:
//!
//! - `adopt` takes a finished keep-alive child (`idle`) and keeps its record,
//!   transcript, and persisted contract.
//! - an idle TTL (default 420_000 ms, upstream `task/agentIdleTtlMs`) parks it:
//!   the live session is dropped, the transcript and contract stay on disk.
//! - `ensureLive`/`send` revives a parked child from the persisted contract and
//!   JSONL replay, coalescing concurrent revives into one live session.
//! - a hard kill tombstones the id: its transcript stays readable through
//!   `history://`, but it can never be messaged or revived again.
//!
//! The manager holds only a [`Weak<Supervisor>`]; all state it needs comes from
//! the supervisor's existing read accessors plus its own files. The supervisor's
//! record is never mutated, so the five-minute retention path keeps working for
//! normal jobs.

use std::{
	collections::{HashMap, HashSet},
	fs,
	path::{Path, PathBuf},
	sync::{
		Arc, LazyLock, Mutex, Weak,
		atomic::{AtomicU64, Ordering},
	},
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use tokio::sync::mpsc;

use crate::{
	state::{Message, Role},
	supervisor::Supervisor,
	task_persisted_revive::{
		PersistedChildReviverFactory, PersistedChildSession, RevivedChild, SESSION_INIT_SUFFIX,
		TOMBSTONE_SUFFIX, append_messages, load_transcript, tombstone_path, transcript_path,
	},
};

/// Upstream `task/agentIdleTtlMs` default.
pub const DEFAULT_AGENT_IDLE_TTL_MS: u64 = 420_000;

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map(|duration| duration.as_millis() as u64)
		.unwrap_or_default()
}

/// Lifecycle status exposed by the roster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleStatus {
	Idle,
	Parked,
	Running,
	Tombstoned,
}

impl LifecycleStatus {
	#[must_use]
	pub fn as_str(self) -> &'static str {
		match self {
			Self::Idle => "idle",
			Self::Parked => "parked",
			Self::Running => "running",
			Self::Tombstoned => "tombstoned",
		}
	}
}

/// How a message reached a lifecycle-managed agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryOutcome {
	/// The live revived session accepted the message.
	Injected,
	/// The parked/idle agent was rebuilt and the message started a turn.
	Revived,
}

impl DeliveryOutcome {
	#[must_use]
	pub fn as_str(self) -> &'static str {
		match self {
			Self::Injected => "injected",
			Self::Revived => "revived",
		}
	}
}

/// One roster row for a lifecycle-managed agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifecycleRow {
	pub id:            String,
	pub display_name:  String,
	pub parent_id:     Option<String>,
	pub status:        LifecycleStatus,
	pub unread:        usize,
	pub last_activity: u64,
	pub session_file:  PathBuf,
}

struct Entry {
	display_name:  String,
	parent_id:     Option<String>,
	status:        LifecycleStatus,
	session_file:  PathBuf,
	contract:      Option<PersistedChildSession>,
	/// True when this row was restored from disk, not adopted in-process. A
	/// restored row is pinned to its transcript path: a later spawn that reuses
	/// the bare `task-N` id must not replace it.
	restored:      bool,
	/// Invalidates stale timer/actor callbacks after a park or revive.
	epoch:         u64,
	last_activity: u64,
}

struct LiveSession {
	inbound: mpsc::Sender<String>,
	cancel:  tokio_util::sync::CancellationToken,
	epoch:   u64,
}

struct LifecycleState {
	entries:    HashMap<String, Entry>,
	live:       HashMap<String, LiveSession>,
	tombstones: HashSet<String>,
}

/// Park/revive owner for one supervisor.
pub struct SubagentLifecycleManager {
	supervisor:  Weak<Supervisor>,
	root:        Option<PathBuf>,
	idle_ttl_ms: AtomicU64,
	reviver:     Arc<dyn PersistedChildReviverFactory>,
	state:       Mutex<LifecycleState>,
}

static MANAGERS: LazyLock<Mutex<HashMap<String, Weak<SubagentLifecycleManager>>>> =
	LazyLock::new(|| Mutex::new(HashMap::new()));

fn managers() -> &'static Mutex<HashMap<String, Weak<SubagentLifecycleManager>>> {
	&MANAGERS
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
	mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Attach a manager to `supervisor` and register it for hub delegation.
///
/// Existing `*.init.json` contracts in the supervisor's child-session root are
/// scanned as `parked` rows, so a restarted process restores the same roster.
pub fn attach(
	supervisor: &Arc<Supervisor>,
	idle_ttl_ms: u64,
	reviver: Arc<dyn PersistedChildReviverFactory>,
) -> Arc<SubagentLifecycleManager> {
	let root = supervisor.child_session_root();
	let manager = Arc::new(SubagentLifecycleManager {
		supervisor: Arc::downgrade(supervisor),
		root,
		idle_ttl_ms: AtomicU64::new(idle_ttl_ms),
		reviver,
		state: Mutex::new(LifecycleState {
			entries:    HashMap::new(),
			live:       HashMap::new(),
			tombstones: HashSet::new(),
		}),
	});
	if let Some(root) = manager.root.clone() {
		manager.scan_persisted(&root);
	}
	let mut registry = lock(managers());
	registry.retain(|_, weak| weak.strong_count() > 0);
	registry.insert(supervisor.instance_key().to_string(), Arc::downgrade(&manager));
	manager
}

/// Manager attached to `supervisor`, if any.
#[must_use]
pub fn lookup(supervisor: &Arc<Supervisor>) -> Option<Arc<SubagentLifecycleManager>> {
	let key = supervisor.instance_key();
	let mut registry = lock(managers());
	registry.retain(|_, weak| weak.strong_count() > 0);
	registry.get(key).and_then(|weak| weak.upgrade())
}

impl SubagentLifecycleManager {
	/// Directory child contracts/transcripts live in.
	#[must_use]
	pub fn root(&self) -> Option<&Path> {
		self.root.as_deref()
	}

	/// Current lifecycle status for `id`.
	#[must_use]
	pub fn status(&self, id: &str) -> Option<LifecycleStatus> {
		lock(&self.state).entries.get(id).map(|entry| entry.status)
	}

	/// Whether this manager owns a row (including a tombstone) for `id`.
	#[must_use]
	pub fn knows(&self, id: &str) -> bool {
		let state = lock(&self.state);
		state.entries.contains_key(id) || state.tombstones.contains(id)
	}

	/// Transcript path for `id`, including tombstoned ids (`history://`).
	///
	/// A row restored after a restart has no live entry when it was tombstoned,
	/// so fall back to the on-disk transcript the same way the very first
	/// `history://<id>` read does.
	#[must_use]
	pub fn transcript_path(&self, id: &str) -> Option<PathBuf> {
		let known = lock(&self.state)
			.entries
			.get(id)
			.map(|entry| entry.session_file.clone());
		if known.is_some() {
			return known;
		}
		let path = transcript_path(self.root.as_deref()?, id);
		path.is_file().then_some(path)
	}

	/// Roster rows, newest activity first within each status.
	#[must_use]
	pub fn roster(&self) -> Vec<LifecycleRow> {
		let mailbox = self.supervisor.upgrade().map(|supervisor| supervisor.mailbox());
		let state = lock(&self.state);
		let mut rows: Vec<LifecycleRow> = state
			.entries
			.iter()
			.filter(|(_, entry)| entry.status != LifecycleStatus::Tombstoned)
			.map(|(id, entry)| LifecycleRow {
				id:            id.clone(),
				display_name:  entry.display_name.clone(),
				parent_id:     entry.parent_id.clone(),
				status:        entry.status,
				unread:        mailbox.as_ref().map_or(0, |mailbox| mailbox.unread_count(id)),
				last_activity: entry.last_activity,
				session_file:  entry.session_file.clone(),
			})
			.collect();
		rows.sort_by(|left, right| {
			right
				.last_activity
				.cmp(&left.last_activity)
				.then_with(|| left.id.cmp(&right.id))
		});
		rows
	}

	/// Adopt a finished keep-alive child as `idle`.
	///
	/// The record supplies the real transcript path; `contract` is the
	/// `session_init`-equivalent the executor captured at spawn. When `None`, the
	/// manager tries the persisted file, and otherwise keeps the row
	/// transcript-only (revive will refuse).
	pub fn adopt(
		self: &Arc<Self>,
		id: &str,
		contract: Option<PersistedChildSession>,
	) -> Result<(), String> {
		let supervisor = self
			.supervisor
			.upgrade()
			.ok_or_else(|| "supervisor is gone".to_string())?;
		let record = supervisor
			.children()
			.into_iter()
			.find(|record| record.id == id)
			.ok_or_else(|| format!("unknown child \"{id}\""))?;
		if record.status == crate::supervisor::ChildStatus::Aborted {
			return Err(format!(
				"Agent \"{id}\" was hard-aborted and cannot be adopted or revived."
			));
		}
		let session_file = record
			.session_file
			.clone()
			.ok_or_else(|| format!("child \"{id}\" has no transcript file"))?;
		let root = session_file
			.parent()
			.map(Path::to_path_buf)
			.ok_or_else(|| format!("transcript {} has no parent directory", session_file.display()))?;
		// A resumed session restores `task-N` rows from disk while the fresh
		// supervisor's `next_seq` restarts at 1. The executor skips ids whose
		// transcript/contract already exist, so a collision should be
		// impossible; this is the in-memory half of the same guard. A restored
		// row is pinned to its transcript and contract and is never silently
		// replaced by a different spawn.
		let existing = {
			let state = lock(&self.state);
			if state.tombstones.contains(id) {
				return Err(format!(
					"Agent \"{id}\" was hard-aborted and cannot be adopted or revived."
				));
			}
			state
				.entries
				.get(id)
				.map(|entry| (entry.session_file.clone(), entry.restored, entry.contract.clone()))
		};
		if let Some((existing_file, restored, existing_contract)) = existing {
			if existing_file != session_file {
				return Err(format!(
					"agent id \"{id}\" already names transcript {}; refusing to adopt {}",
					existing_file.display(),
					session_file.display()
				));
			}
			if restored {
				let incoming = contract
					.clone()
					.or_else(|| PersistedChildSession::load(&root, id));
				if incoming.as_ref() != existing_contract.as_ref() {
					return Err(format!(
						"agent \"{id}\" is a restored parked session; refusing to replace it \
						 with a different spawn"
					));
				}
			}
		}
		let contract = match contract {
			Some(contract) => {
				if contract.id != id {
					return Err(format!(
						"contract id {:?} does not match child \"{id}\"",
						contract.id
					));
				}
				contract.write(&root)?;
				Some(contract)
			},
			None => PersistedChildSession::load(&root, id),
		};
		let display_name = contract
			.as_ref()
			.and_then(|contract| contract.display_name.clone())
			.unwrap_or_else(|| record.name.clone());
		let parent_id = contract
			.as_ref()
			.and_then(|contract| contract.parent_id.clone())
			.or_else(|| record.parent_id.clone());
		let epoch = {
			let mut state = lock(&self.state);
			let epoch = state.entries.get(id).map_or(0, |entry| entry.epoch.wrapping_add(1));
			state.entries.insert(
				id.to_string(),
				Entry {
					display_name,
					parent_id,
					status: LifecycleStatus::Idle,
					session_file,
					contract,
					restored: false,
					epoch,
					last_activity: now_ms(),
				},
			);
			epoch
		};
		self.arm_timer(id, epoch);
		Ok(())
	}

	/// Park an agent now (TTL expiry, explicit park, or shutdown).
	pub fn park(&self, id: &str) -> bool {
		let mut state = lock(&self.state);
		let Some(status) = state.entries.get(id).map(|entry| entry.status) else {
			return false;
		};
		if status == LifecycleStatus::Tombstoned {
			return false;
		}
		if let Some(live) = state.live.remove(id) {
			live.cancel.cancel();
		}
		if let Some(entry) = state.entries.get_mut(id) {
			entry.epoch = entry.epoch.wrapping_add(1);
			entry.status = LifecycleStatus::Parked;
			entry.last_activity = now_ms();
		}
		true
	}

	/// Hard kill: keep the transcript readable, refuse every later revive.
	pub fn tombstone(self: &Arc<Self>, id: &str, reason: &str) -> bool {
		// An aborted child is never adopted, so the transcript path has to come
		// from the supervisor record; otherwise its spawn-time contract would be
		// rediscovered as parked after a restart.
		let record_file = self.supervisor.upgrade().and_then(|supervisor| {
			supervisor
				.children()
				.into_iter()
				.find(|record| record.id == id)
				.and_then(|record| record.session_file)
		});
		let mut state = lock(&self.state);
		let session_file = state
			.entries
			.get(id)
			.map(|entry| entry.session_file.clone())
			.or(record_file);
		if let Some(live) = state.live.remove(id) {
			live.cancel.cancel();
		}
		if let Some(entry) = state.entries.get_mut(id) {
			entry.epoch = entry.epoch.wrapping_add(1);
			entry.status = LifecycleStatus::Tombstoned;
			entry.last_activity = now_ms();
		}
		state.tombstones.insert(id.to_string());
		let root = session_file
			.as_deref()
			.and_then(Path::parent)
			.map(Path::to_path_buf)
			.or_else(|| self.root.clone());
		drop(state);
		if let Some(root) = root {
			let _ = fs::write(tombstone_path(&root, id), reason);
		}
		true
	}

	/// Deliver `text` to a lifecycle-managed agent.
	///
	/// A live revived session receives the message through its inbound queue
	/// (`injected`). A parked/idle row is rebuilt from the persisted contract and
	/// JSONL replay, and the message starts the first turn (`revived`). The
	/// whole check-and-spawn runs under one lock, so two concurrent sends revive
	/// exactly one session.
	pub fn send(self: &Arc<Self>, id: &str, text: String) -> Result<DeliveryOutcome, String> {
		if tokio::runtime::Handle::try_current().is_err() {
			return Err("reviving an agent requires a tokio runtime".to_string());
		}
		let supervisor = self.supervisor.upgrade();
		let mut state = lock(&self.state);
		if state.tombstones.contains(id) {
			return Err(format!(
				"Agent \"{id}\" was hard-aborted and cannot be messaged or revived."
			));
		}
		let live_send = state
			.live
			.get(id)
			.map(|live| live.inbound.try_send(text.clone()));
		match live_send {
			Some(Ok(())) => {
				// Close the TTL race: a message queued at the exact expiry instant
				// must not be parked away before the actor dequeues it.
				if let Some(entry) = state.entries.get_mut(id) {
					entry.status = LifecycleStatus::Running;
					entry.last_activity = now_ms();
				}
				return Ok(DeliveryOutcome::Injected);
			},
			Some(Err(mpsc::error::TrySendError::Full(_))) => {
				return Err(format!("Agent \"{id}\" is busy; its session queue is full."));
			},
			Some(Err(mpsc::error::TrySendError::Closed(_))) => {
				state.live.remove(id);
			},
			None => {},
		}
		let Some(entry) = state.entries.get(id) else {
			return Err(format!("Unknown agent \"{id}\" — check `hub` list for live peers."));
		};
		if entry.status == LifecycleStatus::Tombstoned {
			return Err(format!(
				"Agent \"{id}\" was hard-aborted and cannot be messaged or revived."
			));
		}
		let contract = entry.contract.clone().ok_or_else(|| {
			format!(
				"Agent \"{id}\" has no persisted session_init contract; only its transcript \
				 (history://{id}) is available."
			)
		})?;
		let session_file = entry.session_file.clone();
		let replayed = load_transcript(&session_file).map_err(|error| {
			format!("cannot reopen agent \"{id}\" transcript: {error}")
		})?;
		let revived = self.reviver.revive(&contract, replayed, supervisor.as_ref())?;
		let epoch = entry.epoch.wrapping_add(1);
		let (inbound, receiver) = mpsc::channel(64);
		inbound
			.try_send(text)
			.map_err(|_| format!("Agent \"{id}\" session queue is unavailable."))?;
		let cancel = tokio_util::sync::CancellationToken::new();
		state.live.insert(
			id.to_string(),
			LiveSession { inbound, cancel: cancel.clone(), epoch },
		);
		if let Some(entry) = state.entries.get_mut(id) {
			entry.epoch = epoch;
			entry.status = LifecycleStatus::Running;
			entry.last_activity = now_ms();
		}
		// Weak: an idle actor must not keep the manager (and its inbound sender)
		// alive in a cycle after the owner drops it.
		let manager = Arc::downgrade(self);
		let id = id.to_string();
		drop(state);
		// The actor receives the first message immediately; `Revived` is honest
		// because a brand-new live session was built for it.
		tokio::spawn(async move {
			run_actor(manager, id, revived, receiver, cancel, epoch).await;
		});
		Ok(DeliveryOutcome::Revived)
	}

	fn arm_timer(self: &Arc<Self>, id: &str, epoch: u64) {
		let ttl = self.idle_ttl_ms.load(Ordering::SeqCst);
		if ttl == 0 {
			return;
		}
		let Ok(handle) = tokio::runtime::Handle::try_current() else {
			return;
		};
		let manager = Arc::clone(self);
		let id = id.to_string();
		handle.spawn(async move {
			tokio::time::sleep(Duration::from_millis(ttl)).await;
			manager.expire_if_idle(&id, epoch);
		});
	}

	fn expire_if_idle(&self, id: &str, epoch: u64) {
		let mut state = lock(&self.state);
		let eligible = state.entries.get(id).is_some_and(|entry| {
			entry.epoch == epoch && entry.status == LifecycleStatus::Idle
		});
		if !eligible {
			return;
		}
		if let Some(live) = state.live.remove(id) {
			live.cancel.cancel();
		}
		if let Some(entry) = state.entries.get_mut(id) {
			entry.epoch = entry.epoch.wrapping_add(1);
			entry.status = LifecycleStatus::Parked;
			entry.last_activity = now_ms();
		}
	}

	fn note_running(&self, id: &str, epoch: u64) {
		let mut state = lock(&self.state);
		let Some(entry) = state.entries.get_mut(id) else {
			return;
		};
		if entry.epoch != epoch
			|| matches!(entry.status, LifecycleStatus::Tombstoned | LifecycleStatus::Parked)
		{
			return;
		}
		entry.status = LifecycleStatus::Running;
		entry.last_activity = now_ms();
	}

	fn note_idle(self: &Arc<Self>, id: &str, epoch: u64, arm_timer: bool) {
		{
			let mut state = lock(&self.state);
			let Some(entry) = state.entries.get_mut(id) else {
				return;
			};
			if entry.epoch != epoch
				|| matches!(entry.status, LifecycleStatus::Tombstoned | LifecycleStatus::Parked)
			{
				return;
			}
			entry.status = LifecycleStatus::Idle;
			entry.last_activity = now_ms();
		}
		if arm_timer {
			self.arm_timer(id, epoch);
		}
	}

	fn finish_actor(&self, id: &str, epoch: u64) {
		let mut state = lock(&self.state);
		if state.live.get(id).is_some_and(|live| live.epoch == epoch) {
			state.live.remove(id);
		}
		if let Some(entry) = state.entries.get_mut(id)
			&& entry.epoch == epoch
			&& !matches!(
				entry.status,
				LifecycleStatus::Tombstoned | LifecycleStatus::Parked
			)
		{
			entry.status = LifecycleStatus::Idle;
			entry.last_activity = now_ms();
		}
	}

	/// Restore parked rows and tombstones from disk (process restart / resume).
	fn scan_persisted(&self, root: &Path) {
		let Ok(entries) = fs::read_dir(root) else {
			return;
		};
		let mut state = lock(&self.state);
		for entry in entries.flatten() {
			let path = entry.path();
			let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
				continue;
			};
			if let Some(id) = name.strip_suffix(TOMBSTONE_SUFFIX) {
				state.tombstones.insert(id.to_string());
				continue;
			}
			let Some(id) = name.strip_suffix(SESSION_INIT_SUFFIX) else {
				continue;
			};
			if state.entries.contains_key(id) {
				continue;
			}
			if tombstone_path(root, id).exists() {
				state.tombstones.insert(id.to_string());
				continue;
			}
			let Some(contract) = PersistedChildSession::load(root, id) else {
				continue;
			};
			let display_name = contract
				.display_name
				.clone()
				.or_else(|| contract.agent.clone())
				.unwrap_or_else(|| id.to_string());
			state.entries.insert(
				id.to_string(),
				Entry {
					display_name,
					parent_id: contract.parent_id.clone(),
					// A restarted process has no live session: every restored row
					// is parked until a send revives it.
					status: LifecycleStatus::Parked,
					session_file: transcript_path(root, id),
					contract: Some(contract),
					restored: true,
					epoch: 0,
					last_activity: now_ms(),
				},
			);
		}
	}
}

/// Serial worker behind one revived session.
///
/// Messages queue here; each one runs a fresh dialog turn against the same
/// [`crate::state::AgentState`], so a second send during a turn is answered
/// after that turn instead of being dropped or starting a second session.
async fn run_actor(
	manager: Weak<SubagentLifecycleManager>,
	id: String,
	revived: RevivedChild,
	mut receiver: mpsc::Receiver<String>,
	cancel: tokio_util::sync::CancellationToken,
	epoch: u64,
) {
	let RevivedChild {
		agent_loop,
		mut state,
		agent,
		contract,
	} = revived;
	// Buffered mail addressed to this agent is drained at the first turn
	// boundary by the loop's peer inbox; it stays queued here until then.
	while let Some(text) = receiver.recv().await {
		if cancel.is_cancelled() {
			break;
		}
		let Some(manager) = manager.upgrade() else {
			break;
		};
		manager.note_running(&id, epoch);
		let before = state.messages.len();
		let had_system = state.messages.first().is_some_and(|m| m.role == Role::System);
		let request = Message::user(text);
		let turn_result = agent_loop
			.run_dialog_message_with_cancel(
				request.clone(),
				&mut state,
				None,
				None,
				cancel.child_token(),
			)
			.await;
		// TS `runSubagentFollowUpTurn`: a follow-up turn rewrites the same
		// `<outputId>.md` only when it produced a real yield. A hub wake with no
		// yield leaves the last completion artifact untouched (issue #9518).
		if let Ok(turns) = &turn_result
			&& let Some(agent) = agent.as_ref()
			&& let Some(output_id) = contract.published_output_id()
			&& let Some(root) = manager
				.supervisor
				.upgrade()
				.and_then(|supervisor| supervisor.agent_output_root())
			&& let Some(body) =
				crate::tools::task::follow_up_output(&contract, agent, turns)
			&& let Err(error) = crate::tools::agent_output::publish_agent_output_replace(
				&root,
				output_id,
				&body,
			)
		{
			eprintln!(
				"warning: cannot rewrite follow-up agent output {output_id}: {error}"
			);
		}
		if let Err(error) = turn_result {
			// The hub receipt already reported this session as `revived`.
			// A failed turn is real information, not a silent success: report
			// it instead of dropping the error. Drop the partial turn too (it
			// can end in `tool_calls` without results, which the next provider
			// request would reject) but keep the request so a later send can
			// retry a valid conversation.
			eprintln!("warning: revived agent \"{id}\" turn failed: {error}");
			state.messages.truncate(before);
			state.messages.push(request);
		}
		let start = if !had_system
			&& state
				.messages
				.first()
				.is_some_and(|message| message.role == Role::System)
		{
			before + 1
		} else {
			before
		};
		if let Some(path) = manager.transcript_path(&id) {
			let appended: Vec<&Message> =
				state.messages.get(start..).unwrap_or(&[]).iter().collect();
			append_messages(&path, &appended);
		}
		if cancel.is_cancelled() {
			break;
		}
		let queue_empty = receiver.len() == 0;
		manager.note_idle(&id, epoch, queue_empty);
	}
	if let Some(manager) = manager.upgrade() {
		if let Some(supervisor) = manager.supervisor.upgrade() {
			supervisor.clear_revived_output_prefix(&id);
		}
		manager.finish_actor(&id, epoch);
	}
}

