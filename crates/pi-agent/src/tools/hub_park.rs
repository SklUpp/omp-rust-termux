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
//! `history://<id>` reads the agent's real JSONL transcript, which stays
//! available for parked and tombstoned ids alike.

use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;

use super::internal_urls::{InternalResource, InternalUrl, InternalUrlRouter, ProtocolHandler, ResolveContext};
use crate::{
	subagent_lifecycle::{DeliveryOutcome, LifecycleRow, SubagentLifecycleManager, lookup},
	supervisor::Supervisor,
};

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

/// Read-only `history://<agent-id>` handler over the lifecycle transcripts.
pub struct HistoryProtocolHandler {
	manager: Arc<SubagentLifecycleManager>,
}

impl HistoryProtocolHandler {
	#[must_use]
	pub fn new(manager: Arc<SubagentLifecycleManager>) -> Self {
		Self { manager }
	}
}

#[async_trait]
impl ProtocolHandler for HistoryProtocolHandler {
	fn scheme(&self) -> &str {
		"history"
	}

	fn immutable(&self) -> bool {
		true
	}

	async fn resolve(
		&self,
		url: &InternalUrl,
		_context: &ResolveContext,
	) -> Result<InternalResource, String> {
		let id = if url.host.is_empty() {
			url.path.trim_start_matches('/')
		} else {
			url.host.as_str()
		};
		if id.is_empty() {
			return Err("history:// needs an agent id: history://<agent-id>".to_string());
		}
		let path = self
			.manager
			.transcript_path(id)
			.ok_or_else(|| format!("history://{id}: unknown agent"))?;
		let content = tokio::fs::read_to_string(&path)
			.await
			.map_err(|error| format!("history://{id}: cannot read {}: {error}", path.display()))?;
		let mut resource = InternalResource::text(&url.href, content);
		resource.source_path = Some(path);
		resource.content_type = "text/plain".to_string();
		Ok(resource)
	}
}

/// Mount `history://` on a session router.
pub fn register_history(
	router: &InternalUrlRouter,
	manager: Arc<SubagentLifecycleManager>,
) -> Result<(), String> {
	router
		.register(Arc::new(HistoryProtocolHandler::new(manager)))
		.map(|_| ())
}

/// Transcript path for an id, used by tests/embedders that read the raw JSONL.
#[must_use]
pub fn transcript_path(supervisor: &Arc<Supervisor>, id: &str) -> Option<PathBuf> {
	lookup(supervisor)?.transcript_path(id)
}
