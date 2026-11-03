//! Host implementation of the per-child advisor seam.
//!
//! The trigger contract is the TypeScript one: `task.rs` calls these methods at
//! the same boundaries as `setOnTurnEnd` / `onPrimaryTurnEnd`
//! (`session-advisors.ts:337-354`): once at spawn with the initial user message,
//! once per settled turn after tool results (never for an aborted turn), and
//! once when the child session ends. Advice is delivered through the root host's
//! card sink, so a child card is persisted/rendered exactly like a root card and
//! never starts a hidden child turn; per-child transcripts are written beside the
//! child session log by [`AdvisorManager`].

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pi_agent::Message;
use pi_agent::subagent_advisor::{ChildAdvisorContext, ChildAdvisorHook};
use pi_agent::tools::task_catalog::HandoffTarget;

use crate::advisor_child::{apply_to_config, resolve_agent_advisor};
use crate::advisor_identity::AdvisorIdentities;
use crate::advisor_runtime::{
	AdvisorExecutor, AdvisorManager, AdvisorManagerOptions, AdvisorRoute, CardSink, RouteResolver,
};
use crate::agent::RunRequest;
use crate::catalog::Catalog;
use crate::config::Config;
use crate::runtime::CliAdvisorExecutor;

/// One advisor runtime per live child session, owned by the root run.
pub struct ChildAdvisorHost {
	config:      Config,
	catalog:     Arc<Catalog>,
	approvals:   crate::approve::Approvals,
	identities:  Arc<AdvisorIdentities>,
	cwd:         PathBuf,
	agent_dir:   PathBuf,
	card_sink:   Option<CardSink>,
	service_tier: BTreeMap<String, String>,
	handle:      tokio::runtime::Handle,
	children:    Mutex<HashMap<String, Arc<AdvisorManager>>>,
}

impl ChildAdvisorHost {
	pub fn new(
		request: &RunRequest,
		handle: tokio::runtime::Handle,
		identities: Arc<AdvisorIdentities>,
	) -> Self {
		Self {
			config: request.config.clone(),
			catalog: Arc::clone(&request.catalog),
			approvals: request.approvals.clone(),
			identities,
			cwd: request.cwd.clone(),
			agent_dir: crate::dirs::agent_dir(),
			card_sink: request
				.advisor_card_sink
				.as_ref()
				.map(|sink| Arc::clone(&sink.0)),
			service_tier: crate::advisor_identity::service_tier_map(
				Some(request.config.advisor.tier_setting()),
				&request.service_tier_by_family,
			),
			handle,
			children: Mutex::new(HashMap::new()),
		}
	}

	/// Build the child roster on first use. Absent opt-in returns `None`; no
	/// provider is contacted and no hidden work starts.
	fn manager_for(&self, child: &ChildAdvisorContext<'_>) -> Option<Arc<AdvisorManager>> {
		// TS `subagentAdvisorEnabled` requires `advisor.enabled` before the
		// per-agent opt-in is even consulted.
		if !self.config.advisor.enabled() {
			return None;
		}
		let mut children = self
			.children
			.lock()
			.unwrap_or_else(|poison| poison.into_inner());
		if let Some(manager) = children.get(child.child_id) {
			return Some(Arc::clone(manager));
		}
		let definition = if child.definition_advisor {
			Some(
				child
					.definition_pattern
					.map(|pattern| HandoffTarget::Pattern(pattern.to_string()))
					.unwrap_or(HandoffTarget::Default),
			)
		} else {
			None
		};
		let selection = resolve_agent_advisor(
			self
				.config
				.advisor
				.agent_advisor_for(child.agent.unwrap_or("")),
			definition.as_ref(),
		)?;
		let mut config = self.config.clone();
		apply_to_config(&mut config, &selection);
		let pattern_override = selection.model.clone();
		let catalog = Arc::clone(&self.catalog);
		let resolver_config = config.clone();
		let resolver: RouteResolver = Arc::new(move |pattern| {
			let pattern = if pattern == "@advisor" {
				pattern_override.clone().unwrap_or_else(|| "@advisor".to_string())
			} else {
				pattern.to_string()
			};
			let switch =
				crate::runtime::resolve_model_switch(&catalog, &resolver_config, &pattern, None)?;
			let api_key = resolver_config.resolve_api_key(&switch.provider);
			Ok(AdvisorRoute {
				provider: switch.provider,
				model:    switch.model,
				effort:   switch.effort,
				api_key,
			})
		});
		let executor: Arc<dyn AdvisorExecutor> = Arc::new(CliAdvisorExecutor);
		let discovered =
			crate::advisor_config::discover(&self.cwd, Some(self.agent_dir.as_path()));
		let roster = if discovered.advisors.is_empty() {
			vec![crate::advisor_config::AdvisorEntry {
				name:         "default".to_string(),
				model:        None,
				tools:        None,
				instructions: None,
				enabled:      None,
			}]
		} else {
			discovered.advisors
		};
		let watchdog_docs = crate::advisor_config::discover_watchdog_docs(
			&self.cwd,
			Some(self.agent_dir.as_path()),
		);
		let session_file = child.session_file.map(std::path::Path::to_path_buf);
		let manager = AdvisorManager::new(AdvisorManagerOptions {
			cwd: self.cwd.clone(),
			agent_dir: Some(self.agent_dir.clone()),
			config: config.clone(),
			catalog: Arc::clone(&self.catalog),
			approvals: self.approvals.clone(),
			primary_session_id: session_file
				.as_ref()
				.map(|path| path.display().to_string())
				.or_else(|| Some(child.child_id.to_string())),
			primary_cache_key: None,
			session_file,
			service_tier: self.service_tier.clone(),
			context_prompt: None,
			watchdog_prompt: (!watchdog_docs.is_empty()).then(|| watchdog_docs.join("\n\n")),
			shared_instructions: discovered.shared_instructions,
			roster,
			executor,
			resolver,
			identities: Arc::clone(&self.identities),
			handle: self.handle.clone(),
		});
		manager.configure_run(
			Some(child.child_id.to_string()),
			None,
			child
				.session_file
				.map(std::path::Path::to_path_buf),
			self.card_sink.clone(),
			None,
		);
		children.insert(child.child_id.to_string(), Arc::clone(&manager));
		Some(manager)
	}
}

impl ChildAdvisorHook for ChildAdvisorHost {
	fn on_child_start(&self, child: &ChildAdvisorContext<'_>, initial: &Message) {
		if let Some(manager) = self.manager_for(child) {
			manager.begin_primary_run(&[]);
			manager.push_user(initial);
		}
	}

	fn on_child_turn(
		&self,
		child: &ChildAdvisorContext<'_>,
		message: &Message,
		tool_results: &[pi_agent::ToolResult],
		_will_continue: bool,
	) {
		let Some(manager) = self
			.children
			.lock()
			.unwrap_or_else(|poison| poison.into_inner())
			.get(child.child_id)
			.cloned()
		else {
			return;
		};
		manager.push_turn(message, tool_results);
		// TS `onPrimaryTurnEnd` waits for the backlog after feeding the turn
		// (30s). The hook contract is synchronous, so the wait runs inline and
		// bounded; it starts no detached task that could outlive the child.
		if let Some(threshold) = self.config.advisor.sync_backlog() {
			if manager.backlog() >= threshold {
				manager.sync_backlog(threshold, std::time::Duration::from_secs(30));
			}
		}
	}

	fn on_child_end(&self, child: &ChildAdvisorContext<'_>) {
		if let Some(manager) = self
			.children
			.lock()
			.unwrap_or_else(|poison| poison.into_inner())
			.remove(child.child_id)
		{
			manager.dispose();
		}
	}
}
