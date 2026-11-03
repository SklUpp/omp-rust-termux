//! Hub park/revive delegation and the `history://` protocol handler.
//!
//! This module is the only bridge between the hub tool and the lifecycle
//! manager. It keeps `hub.rs` to two narrow calls:
//!
//! - `roster` merges persisted parked/idle rows into `hub list` (including rows
//!   restored after a process restart).
//! - `delegate_send` hands a message to a parked/idle agent, reviving it when
//!   needed and reporting `revived`/`injected` honestly.
//!
//! `history://` is the read-only transcript surface (upstream
//! `internal-urls/history-protocol.ts`):
//!
//! - `history://` lists the caller session's agents (id/status/kind/parent/last
//!   activity) plus transcripts found on disk.
//! - `history://<id>` renders a concise markdown transcript, never raw JSONL.
//!
//! Rows come from the lifecycle manager attached to this session's supervisor
//! and from the caller's own `<artifacts>/subagents` directory, so one RPC
//! session can never read another's transcripts by guessing an id. A live
//! child's transcript is the JSONL the executor appends; this build retains no
//! in-memory message array, so the source note says "transcript file" and never
//! pretends the content came from a live session object.

use std::{
	collections::HashSet,
	path::{Path, PathBuf},
	sync::Arc,
	time::{SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;

use super::{
	agent_output::ADVISOR_OUTPUT_STEM,
	history_format::format_history_markdown,
	internal_urls::{
		InternalResource, InternalUrl, InternalUrlRouter, ProtocolHandler, ResolveContext,
	},
};
use crate::{
	subagent_lifecycle::{
		DeliveryOutcome, LifecycleRow, LifecycleStatus, SubagentLifecycleManager, lookup,
	},
	supervisor::{ChildStatus, Supervisor},
	task_persisted_revive::{read_transcript, transcript_path as child_transcript_path},
};

/// Deepest transcript directory scan.
///
/// Exact TS parity: `packages/coding-agent/src/internal-urls/registry-helpers.ts`
/// `sessionFilesFromDisk` walks at most 8 directories below an artifacts dir.
/// Rust's own layout is flat (`<root>/<id>.jsonl`), so this bound is unreachable
/// in practice and no valid spawn layout is truncated.
const MAX_SCAN_DEPTH: usize = 8;

/// Parked/idle roster rows for this supervisor, or an empty list when no
/// lifecycle manager is attached.
#[must_use]
pub fn roster(supervisor: &Arc<Supervisor>) -> Vec<LifecycleRow> {
	lookup(supervisor).map_or_else(Vec::new, |manager| manager.roster())
}

/// Delegate a send to the lifecycle manager.
///
/// `None` means this target is not lifecycle-managed, so the caller keeps its
/// existing live-mailbox path. `Some` reports the real delivery outcome or the
/// exact refusal (tombstone, missing contract, gone workspace).
#[must_use]
pub fn delegate_send(
	supervisor: &Arc<Supervisor>,
	to: &str,
	text: String,
) -> Option<Result<DeliveryOutcome, String>> {
	let manager = lookup(supervisor)?;
	if !manager.knows(to) {
		return None;
	}
	Some(manager.send(to, text))
}

/// One visible history entry: a manager row, a supervisor child, or a
/// transcript file found under the caller's scope.
#[derive(Debug, Clone)]
struct Candidate {
	id:            String,
	status:        String,
	kind:          String,
	parent:        String,
	last_activity: Option<u64>,
	path:          PathBuf,
	/// This row is (or just was) a real live session with a manager-owned,
	/// producer-fed snapshot. Only then may the handler serve memory instead of
	/// the persisted transcript.
	live:          bool,
}

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map(|duration| duration.as_millis() as u64)
		.unwrap_or_default()
}

/// Humanize a last-activity timestamp as `Ns/Nm/Nh/Nd ago`.
fn format_ago(timestamp: u64) -> String {
	let diff_ms = now_ms().saturating_sub(timestamp);
	let secs = diff_ms / 1000;
	if secs < 60 {
		return format!("{secs}s ago");
	}
	let mins = secs / 60;
	if mins < 60 {
		return format!("{mins}m ago");
	}
	let hours = mins / 60;
	if hours < 24 {
		return format!("{hours}h ago");
	}
	format!("{}d ago", hours / 24)
}

fn child_status(status: ChildStatus) -> &'static str {
	match status {
		ChildStatus::Running => "running",
		ChildStatus::Completed => "completed",
		ChildStatus::Failed => "failed",
		ChildStatus::Aborted => "aborted",
	}
}

fn is_advisor(id: &str) -> bool {
	id.starts_with(ADVISOR_OUTPUT_STEM)
}

fn paths_equal(left: &Path, right: &Path) -> bool {
	if left == right {
		return true;
	}
	match (left.canonicalize(), right.canonicalize()) {
		(Ok(left), Ok(right)) => left == right,
		_ => left == right,
	}
}

/// Child transcript directories a caller may read, caller-first.
///
/// A root session resolves `<artifacts>/subagents`; a child's own scope names
/// its transcript directory directly. Nothing outside these roots is scanned.
#[must_use]
fn caller_roots(context: &ResolveContext) -> Vec<PathBuf> {
	fn push(roots: &mut Vec<PathBuf>, path: PathBuf) {
		if !path.as_os_str().is_empty() && !roots.contains(&path) {
			roots.push(path);
		}
	}
	let mut roots = Vec::new();
	if let Some(session_file) = &context.session_file {
		let parent_is_subagents = session_file
			.parent()
			.and_then(Path::file_name)
			.and_then(|name| name.to_str())
			== Some("subagents");
		if parent_is_subagents {
			if let Some(parent) = session_file.parent() {
				push(&mut roots, parent.to_path_buf());
			}
		} else {
			let artifacts = session_file.with_extension("");
			if artifacts.file_name().and_then(|name| name.to_str()) == Some("subagents") {
				push(&mut roots, artifacts);
			} else {
				push(&mut roots, artifacts.join("subagents"));
			}
		}
	}
	for root in context.local_roots.values() {
		if root.file_name().and_then(|name| name.to_str()) == Some("subagents") {
			push(&mut roots, root.clone());
		} else {
			push(&mut roots, root.join("subagents"));
		}
	}
	roots
}

/// `(id, path)` for every `*.jsonl` under `root`, depth-first and stable.
fn scan_transcripts(
	root: &Path,
	signal: &crate::CancellationToken,
) -> Result<Vec<(String, PathBuf)>, String> {
	fn walk(
		dir: &Path,
		depth: usize,
		signal: &crate::CancellationToken,
		found: &mut Vec<(String, PathBuf)>,
	) -> Result<(), String> {
		if signal.is_cancelled() {
			return Err("aborted".to_string());
		}
		let entries = match std::fs::read_dir(dir) {
			Ok(entries) => entries,
			Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
			Err(error) => {
				return Err(format!("history:// cannot scan {}: {error}", dir.display()));
			},
		};
		let mut entries: Vec<_> = entries.flatten().collect();
		entries.sort_by_key(std::fs::DirEntry::file_name);
		for entry in entries {
			let path = entry.path();
			let file_type = match entry.file_type() {
				Ok(file_type) => file_type,
				Err(_) => continue,
			};
			if file_type.is_dir() {
				if depth < MAX_SCAN_DEPTH {
					walk(&path, depth + 1, signal, found)?;
				}
				continue;
			}
			if !file_type.is_file() {
				continue;
			}
			let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
				continue;
			};
			let Some(id) = name.strip_suffix(".jsonl") else {
				continue;
			};
			if id.is_empty() || is_advisor(id) {
				continue;
			}
			found.push((id.to_string(), path));
		}
		Ok(())
	}

	let mut found = Vec::new();
	walk(root, 0, signal, &mut found)?;
	Ok(found)
}

/// Read-only `history://` handler for one session's lifecycle manager.
pub struct HistoryProtocolHandler {
	manager: Arc<SubagentLifecycleManager>,
}

impl HistoryProtocolHandler {
	#[must_use]
	pub fn new(manager: Arc<SubagentLifecycleManager>) -> Self {
		Self { manager }
	}

	/// Whether this handler's manager belongs to the calling session.
	///
	/// A router is per session in production, but the runtime keeps the newest
	/// manager when a router is shared. A caller whose scope names a different
	/// session must not see this manager's rows or transcripts.
	fn manager_visible(&self, context: &ResolveContext) -> bool {
		let Some(root) = self.manager.root() else {
			return true;
		};
		if context.session_file.is_none() {
			return true;
		}
		caller_roots(context)
			.iter()
			.any(|candidate| paths_equal(candidate, root))
	}

	/// Manager/child/disk rows for this call, caller-scoped and deduplicated.
	fn candidates(&self, context: &ResolveContext) -> Result<Vec<Candidate>, String> {
		let signal = &context.signal;
		if signal.is_cancelled() {
			return Err("aborted".to_string());
		}
		let mut candidates: Vec<Candidate> = Vec::new();
		let mut seen: HashSet<String> = HashSet::new();
		let visible = self.manager_visible(context);
		if visible {
			for row in self.manager.history_rows() {
				if is_advisor(&row.id) || !seen.insert(row.id.clone()) {
					continue;
				}
				candidates.push(Candidate {
					id:            row.id,
					status:        row.status.as_str().to_string(),
					kind:          row.agent.unwrap_or_else(|| "—".to_string()),
					parent:        row.parent_id.unwrap_or_else(|| "—".to_string()),
					last_activity: (row.last_activity > 0).then_some(row.last_activity),
					path:          row.session_file,
					live:          matches!(row.status, LifecycleStatus::Running | LifecycleStatus::Idle),
				});
			}
			for record in self.manager.child_records() {
				if is_advisor(&record.id) || !seen.insert(record.id.clone()) {
					continue;
				}
				let Some(path) = record.session_file.clone().or_else(|| {
					self.manager
						.root()
						.map(|root| child_transcript_path(root, &record.id))
				}) else {
					continue;
				};
				candidates.push(Candidate {
					id:            record.id,
					status:        child_status(record.status).to_string(),
					kind:          record.agent,
					parent:        record.parent_id.unwrap_or_else(|| "—".to_string()),
					last_activity: record.started_at,
					path,
					live:          record.status == ChildStatus::Running,
				});
			}
		}
		let mut roots = caller_roots(context);
		if visible
			&& let Some(root) = self.manager.root()
			&& !roots.iter().any(|candidate| paths_equal(candidate, root))
		{
			roots.push(root.to_path_buf());
		}
		for root in roots {
			for (id, path) in scan_transcripts(&root, signal)? {
				if !seen.insert(id.clone()) {
					continue;
				}
				candidates.push(Candidate {
					id,
					status: "on disk".to_string(),
					kind: "—".to_string(),
					parent: "—".to_string(),
					last_activity: None,
					path,
					live:          false,
				});
			}
		}
		Ok(candidates)
	}

	fn render_index(&self, url: &InternalUrl, candidates: &[Candidate]) -> InternalResource {
		let mut lines: Vec<String> = vec!["# Agents".to_string(), String::new()];
		if candidates.is_empty() {
			lines.push("No agents registered.".to_string());
		} else {
			lines.push("| id | status | kind | parent | last activity |".to_string());
			lines.push("|---|---|---|---|---|".to_string());
			for candidate in candidates {
				let last_activity = candidate
					.last_activity
					.map(format_ago)
					.unwrap_or_else(|| "—".to_string());
				lines.push(format!(
					"| {} | {} | {} | {} | {} |",
					candidate.id,
					candidate.status,
					candidate.kind,
					candidate.parent,
					last_activity
				));
			}
			lines.push(String::new());
			lines.push("Read a transcript with `read history://<id>`.".to_string());
		}
		let content = format!("{}\n", lines.join("\n"));
		let mut resource = InternalResource::text(&url.href, content);
		resource.content_type = "text/markdown".to_string();
		resource
	}

	fn source_note(candidate: &Candidate) -> String {
		match candidate.status.as_str() {
			"on disk" => "Source: session file (read-only, on disk)".to_string(),
			"tombstoned" => "Source: session file (read-only, tombstoned)".to_string(),
			status => format!("Source: transcript file (read-only, {status})"),
		}
	}
}

#[async_trait]
impl ProtocolHandler for HistoryProtocolHandler {
	fn scheme(&self) -> &str {
		"history"
	}

	/// Transcripts are append-only data: a snapshot may change between reads
	/// even though this scheme never accepts a write. The resource flag stays
	/// `false` (upstream parity), while `read` suppresses hashline anchors for
	/// `history://` because there is no edit path back through the URI.
	fn immutable(&self) -> bool {
		false
	}

	async fn resolve(
		&self,
		url: &InternalUrl,
		context: &ResolveContext,
	) -> Result<InternalResource, String> {
		if context.signal.is_cancelled() {
			return Err("aborted".to_string());
		}
		let raw = if url.host.is_empty() {
			url.path.trim_start_matches('/')
		} else {
			url.host.as_str()
		};
		let id = raw
			.split(['?', '#'])
			.next()
			.unwrap_or_default()
			.trim()
			.to_string();
		let candidates = self.candidates(context)?;
		if id.is_empty() {
			return Ok(self.render_index(url, &candidates));
		}

		let lower = id.to_lowercase();
		let ordered: Vec<&Candidate> = candidates
			.iter()
			.filter(|candidate| candidate.id == id)
			.chain(
				candidates
					.iter()
					.filter(|candidate| candidate.id != id && candidate.id.to_lowercase() == lower),
			)
			.collect();
		let Some(candidate) = ordered
			.iter()
			.find(|candidate| candidate.path.is_file())
			.copied()
			.or_else(|| ordered.first().copied())
		else {
			let known: Vec<&str> = candidates.iter().map(|candidate| candidate.id.as_str()).collect();
			let known = if known.is_empty() {
				"none".to_string()
			} else {
				known.join(", ")
			};
			return Err(format!(
				"Unknown agent: {id}\nKnown agents: {known}\nList all with history://"
			));
		};

		// A real producer-fed snapshot is served before the JSONL flush. It
		// exists only while live ownership lasts; parked/tombstoned/disk rows
		// always fall through to the persisted transcript above.
		if candidate.live
			&& let Some(messages) = self.manager.live_messages(&candidate.id)
		{
			if context.signal.is_cancelled() {
				return Err("aborted".to_string());
			}
			let title = format!("{} ({})", candidate.id, candidate.status);
			let content = format_history_markdown(&title, &messages);
			let mut resource = InternalResource::text(&url.href, content);
			resource.content_type = "text/markdown".to_string();
			resource.source_path = Some(candidate.path.clone());
			resource.notes = Some(vec![format!(
				"Source: live session (read-only, {})",
				candidate.status
			)]);
			return Ok(resource);
		}
		if !candidate.path.is_file() {
			return Err(format!(
				"Agent {} has no transcript: session is gone and no session file was retained",
				candidate.id
			));
		}
		if context.signal.is_cancelled() {
			return Err("aborted".to_string());
		}
		let path = candidate.path.clone();
		let read_path = path.clone();
		let messages = tokio::task::spawn_blocking(move || read_transcript(&read_path))
			.await
			.map_err(|error| format!("history://{}: transcript read task failed: {error}", candidate.id))?
			.map_err(|error| format!("history://{}: {error}", candidate.id))?;

		let title = format!("{} ({})", candidate.id, candidate.status);
		let content = format_history_markdown(&title, &messages);
		let mut resource = InternalResource::text(&url.href, content);
		resource.content_type = "text/markdown".to_string();
		resource.source_path = Some(path);
		resource.notes = Some(vec![Self::source_note(candidate)]);
		Ok(resource)
	}
}

/// Mount `history://` on a session router.
///
/// The scheme is reserved: a host URI bridge can never register its own
/// `history://` and shadow the built-in transcripts.
pub fn register_history(
	router: &InternalUrlRouter,
	manager: Arc<SubagentLifecycleManager>,
) -> Result<(), String> {
	router
		.register_core(Arc::new(HistoryProtocolHandler::new(manager)))
		.map(|_| ())
}

/// Transcript path for an id, used by tests/embedders that read the raw JSONL.
#[must_use]
pub fn transcript_path(supervisor: &Arc<Supervisor>, id: &str) -> Option<PathBuf> {
	lookup(supervisor)?.transcript_path(id)
}
