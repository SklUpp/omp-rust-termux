//! Plan-yolo host core: one-shot plan mode armed before the first prompt, an
//! auto-approving `xd://propose` handler, and a two-phase hand-off that keeps
//! the write guard active until the execution provider was actually built.
//!
//! TS reference: `session/prewalk.ts:252-348`. The host owns tool-set
//! augmentation and provider assignment; this module owns the shared gate and
//! the pending approval so no caller can exit plan mode before a successful
//! switch is guaranteed.

use std::{
	path::{Path, PathBuf},
	sync::{
		Arc, Mutex,
		atomic::{AtomicBool, Ordering},
	},
};

use pi_agent::{AgentContext, ToolError, ToolResult};
use serde_json::json;

use crate::{
	acp::plan::{self, AcpPlanState, DEFAULT_PLAN_FILE_URL, PlanGate, PlanWorkflow},
	agent::PlanModeHooks,
	args::Cli,
	config::Config,
	runtime::{self, CliThinking, ModelSwitch},
};

/// TS `DEFAULT_PLAN_YOLO_TARGET`.
pub const DEFAULT_TARGET: &str = "@smol";

const PLAN_YOLO_HANDOFF_PROMPT: &str = include_str!(
	"../../../packages/coding-agent/src/prompts/system/plan-yolo-handoff.md"
);

/// One resolved plan-yolo arm. Credentials and the endpoint are re-derived when
/// the switch actually fires, exactly like prewalk.
#[derive(Debug, Clone)]
pub struct PlanYoloArm {
	pub target: ModelSwitch,
	/// Selector the target's `:suffix` spelled, if any.
	pub thinking: Option<CliThinking>,
}

/// Resolve the explicit startup flags. Unlike prewalk, a missing target or
/// missing configured auth is a hard startup error (TS `main.ts:1267-1283`).
pub fn resolve_arm(
	cli: &Cli,
	config: &Config,
	catalog: &crate::catalog::Catalog,
	explicit_api_key: Option<&str>,
) -> Result<Option<PlanYoloArm>, String> {
	if !cli.plan_yolo {
		if cli.plan_yolo_into.is_some() {
			return Err("--plan-yolo-into requires --plan-yolo".to_string());
		}
		return Ok(None);
	}
	let requested = cli.plan_yolo_into.as_deref().unwrap_or(DEFAULT_TARGET);
	let target = runtime::resolve_model_switch(catalog, config, requested, None)
		.map_err(|error| format!("plan-yolo target: {error}"))?;
	// `--api-key` belongs to the root route; reuse it only for that provider.
	let api_key_override = explicit_api_key.filter(|_| {
		cli.provider
			.as_deref()
			.is_some_and(|provider| provider == target.provider.as_str())
	});
	if !runtime::provider_has_configured_auth(
		catalog,
		config,
		&target.provider,
		api_key_override,
	) {
		return Err(format!(
			"plan-yolo target {}/{} has no configured API key",
			target.provider, target.model
		));
	}
	let thinking = target.thinking;
	Ok(Some(PlanYoloArm { target, thinking }))
}

/// Snapshot taken before plan mode is published. The host rolls this back when
/// applying the augmented tool set fails; a failed arm must not leave the
/// session permanently in plan mode.
#[derive(Debug)]
pub struct PlanYoloArming {
	previous: PlanGate,
}

/// Shared one-shot state. Hosts keep one `Arc<PlanYoloShared>`; the write guard,
/// proposal handler and reminder policy all read the same gate.
#[derive(Debug)]
pub struct PlanYoloShared {
	gate: Arc<Mutex<PlanGate>>,
	arm: PlanYoloArm,
	pending: Mutex<Option<PlanYoloPending>>,
	armed: AtomicBool,
	completed: AtomicBool,
}

/// A validated plan whose execution switch has not been applied yet. Review
/// content is retained so [`PlanYoloShared::commit_pending`] can prove the
/// artifact did not change between approval and switch.
#[derive(Debug, Clone)]
pub struct PlanYoloPending {
	target: ModelSwitch,
	thinking: Option<CliThinking>,
	plan_file_path: String,
	title: String,
	reviewed_content: String,
}

impl PlanYoloPending {
	#[must_use]
	pub fn target(&self) -> &ModelSwitch {
		&self.target
	}

	#[must_use]
	pub fn thinking(&self) -> Option<CliThinking> {
		self.thinking
	}

	#[must_use]
	pub fn plan_file_path(&self) -> &str {
		&self.plan_file_path
	}

	#[must_use]
	pub fn title(&self) -> &str {
		&self.title
	}
}

/// Host-facing hand-off material returned only after the provider build
/// succeeded and plan mode was exited atomically.
#[derive(Debug, Clone)]
pub struct PlanYoloCommit {
	pub target: ModelSwitch,
	pub thinking: Option<CliThinking>,
	pub handoff: String,
	pub notice: String,
	pub plan_file_path: String,
}

impl PlanYoloShared {
	#[must_use]
	pub fn new(gate: Arc<Mutex<PlanGate>>, arm: PlanYoloArm) -> Self {
		Self {
			gate,
			arm,
			pending: Mutex::new(None),
			armed: AtomicBool::new(false),
			completed: AtomicBool::new(false),
		}
	}

	#[must_use]
	pub fn gate(&self) -> Arc<Mutex<PlanGate>> {
		Arc::clone(&self.gate)
	}

	#[must_use]
	pub fn is_armed(&self) -> bool {
		self.armed.load(Ordering::Acquire)
	}

	#[must_use]
	pub fn has_pending(&self) -> bool {
		self.pending
			.lock()
			.unwrap_or_else(|poison| poison.into_inner())
			.is_some()
	}

	/// Publish plan mode before the first prompt. Returns the snapshot to pass
	/// to [`Self::rollback_arm`] if the host's tool-set update fails.
	///
	/// The mode state is published first, matching TS: `SessionTools` must see
	/// plan mode before `write` is added to a restricted selection.
	pub fn arm(&self) -> Option<PlanYoloArming> {
		if self.completed.load(Ordering::Acquire)
			|| self.armed.swap(true, Ordering::AcqRel)
		{
			return None;
		}
		let mut gate = self
			.gate
			.lock()
			.unwrap_or_else(|poison| poison.into_inner());
		let previous = gate.clone();
		let plan_file_path = gate
			.plan_reference_for_injection()
			.unwrap_or_else(|| DEFAULT_PLAN_FILE_URL.to_string());
		// TS plan-yolo sets this state directly, independently of
		// `plan.enabled`; the explicit flag is the grant.
		gate.state = Some(AcpPlanState {
			enabled: true,
			plan_file_path,
			workflow: PlanWorkflow::Parallel,
			reentry: previous.state.is_some(),
		});
		gate.settle = plan::PlanSettleState::default();
		gate.discard_queued_continuation();
		gate.generation = gate.generation.wrapping_add(1);
		Some(PlanYoloArming { previous })
	}

	/// Restore the pre-arm gate state after a failed tool-set publication.
	pub fn rollback_arm(&self, arming: PlanYoloArming) {
		*self
			.gate
			.lock()
			.unwrap_or_else(|poison| poison.into_inner()) = arming.previous;
		self.pending
			.lock()
			.unwrap_or_else(|poison| poison.into_inner())
			.take();
		self.armed.store(false, Ordering::Release);
	}

	/// Discard a queued approval without touching plan mode. A cancelled or
	/// failed turn must be able to ask for a fresh proposal.
	pub fn cancel_pending(&self) {
		self.pending
			.lock()
			.unwrap_or_else(|poison| poison.into_inner())
			.take();
	}

	/// Atomically validate, build and commit the execution switch.
	///
	/// The guard is exited only after `build` succeeds, so an auth, config or
	/// wire-build failure leaves plan mode active with the pending approval
	/// intact. The artifact is re-read through the contained reader and must
	/// byte-match the reviewed version. The caller must not independently exit
	/// the gate; this is the only transition to execution.
	pub fn commit_pending<T>(
		&self,
		artifacts: Option<&Path>,
		build: impl FnOnce(&PlanYoloPending) -> Result<T, String>,
	) -> Result<(PlanYoloCommit, T), String> {
		let mut pending_slot = self
			.pending
			.lock()
			.unwrap_or_else(|poison| poison.into_inner());
		let Some(pending) = pending_slot.as_ref() else {
			return Err("plan-yolo has no approved plan pending a switch".to_string());
		};
		let mut gate = self
			.gate
			.lock()
			.unwrap_or_else(|poison| poison.into_inner());
		if !gate.state.as_ref().is_some_and(|state| state.enabled) {
			*pending_slot = None;
			return Err("plan-yolo plan mode is no longer active".to_string());
		}
		let root = artifacts.ok_or_else(|| {
			"plan-yolo cannot switch: the session has no artifact directory".to_string()
		})?;
		let relative = pi_agent::tools::local_artifact_relative(&pending.plan_file_path)
			.ok_or_else(|| {
				format!(
					"plan-yolo cannot switch: {} is not a local artifact path",
					pending.plan_file_path
				)
			})?;
		let current = pi_agent::tools::read_local_artifact(root, &relative).map_err(|error| {
			format!(
				"plan-yolo cannot revalidate {}: {error}",
				pending.plan_file_path
			)
		})?;
		if current != pending.reviewed_content {
			*pending_slot = None;
			return Err(format!(
				"plan-yolo: {} changed after approval; re-propose the updated plan",
				pending.plan_file_path
			));
		}
		// Build while the gate is still active. `Err` returns with both the
		// pending approval and the guard untouched.
		let built = build(pending).map_err(|error| {
			format!(
				"plan-yolo could not build the execution provider for {}/{}: {error}",
				pending.target.provider, pending.target.model
			)
		})?;
		let commit = PlanYoloCommit {
			target: pending.target.clone(),
			thinking: pending.thinking,
			handoff: render_handoff(&pending.plan_file_path, &pending.title),
			notice: format!(
				"Plan-yolo: plan approved, switched to {}/{} to implement \"{}\".",
				pending.target.provider, pending.target.model, pending.title
			),
			plan_file_path: pending.plan_file_path.clone(),
		};
		gate.exit_after_approval(&pending.plan_file_path);
		*pending_slot = None;
		self.armed.store(false, Ordering::Release);
		self.completed.store(true, Ordering::Release);
		Ok((commit, built))
	}

	/// Queue the reviewed plan for the host's turn-boundary switch. The gate
	/// stays active here; only [`Self::commit_pending`] exits it.
	fn queue_pending(&self, plan: plan::ResolvedPlan) -> Result<(), String> {
		let mut pending = self
			.pending
			.lock()
			.unwrap_or_else(|poison| poison.into_inner());
		if pending.is_some() {
			return Err(
				"plan-yolo approval is already pending the execution switch".to_string(),
			);
		}
		let gate = self
			.gate
			.lock()
			.unwrap_or_else(|poison| poison.into_inner());
		if !gate.state.as_ref().is_some_and(|state| state.enabled) {
			return Err("plan mode is not active".to_string());
		}
		*pending = Some(PlanYoloPending {
			target: self.arm.target.clone(),
			thinking: self.arm.thinking,
			plan_file_path: plan.plan_file_path,
			title: plan.title,
			reviewed_content: plan.plan_content,
		});
		Ok(())
	}
}

/// Host plan-mode wiring. `plan_hooks` reuses the existing guard, reminder
/// policy and proposal trait instead of adding a parallel enforcement path.
#[must_use]
pub fn plan_hooks(
	shared: &Arc<PlanYoloShared>,
	artifacts: Option<PathBuf>,
) -> PlanModeHooks {
	let gate = shared.gate();
	PlanModeHooks {
		guard:   Some(Arc::new(plan::PlanWriteGuard::new(
			Arc::clone(&gate),
			artifacts.clone(),
		))),
		propose: Some(Arc::new(PlanYoloProposalHandler {
			shared:    Arc::clone(shared),
			artifacts,
		})),
		policy:  Some(Arc::new(plan::PlanReminderPolicy::new(gate))),
	}
}

/// Add `write` to a restricted selection only when tools are enabled and the
/// builtin is actually absent. `None` means "all builtins", which already
/// includes `write`; `--no-tools` never gains a transport tool.
#[must_use]
pub fn augment_tools(
	tools_enabled: bool,
	selected: Option<&[String]>,
) -> Option<Vec<String>> {
	let mut tools = selected?.to_vec();
	if !tools_enabled || tools.iter().any(|tool| tool == "write") {
		return Some(tools);
	}
	tools.push("write".to_string());
	Some(tools)
}

/// Auto-approving proposal handler. It validates and queues, but deliberately
/// never exits plan mode: cancellation or a failed switch must leave the guard
/// in place.
pub struct PlanYoloProposalHandler {
	shared:    Arc<PlanYoloShared>,
	artifacts: Option<PathBuf>,
}

#[async_trait::async_trait]
impl pi_agent::state::ProposeHandler for PlanYoloProposalHandler {
	async fn propose(
		&self,
		body: &str,
		ctx: &AgentContext,
	) -> Result<ToolResult, ToolError> {
		if self.shared.has_pending() {
			return Err(ToolError::Rejected(
				"Plan approval is already pending the execution switch.".to_string(),
			));
		}
		let state_path = {
			let gate = self
				.shared
				.gate
				.lock()
				.unwrap_or_else(|poison| poison.into_inner());
			match gate.state.as_ref().filter(|state| state.enabled) {
				Some(state) => state.plan_file_path.clone(),
				None => {
					return Err(ToolError::Rejected(
						"No plan is awaiting approval — xd://propose only accepts a plan title while plan mode is active."
							.to_string(),
					));
				},
			}
		};
		let root = self.artifacts.as_deref().ok_or_else(|| {
			ToolError::ExecutionFailed(
				"Plan artifacts need a persisted session directory".to_string(),
			)
		})?;
		let proposal = plan::parse_plan_proposal(body);
		let target_url = proposal
			.slug
			.as_deref()
			.map(plan::plan_file_url_for_slug)
			.unwrap_or(state_path);
		// `local://` is read-only for the model, so the one inline-content path
		// goes through the same fd-contained artifact writer as plan mode.
		if let Some(content) = &proposal.content {
			let relative = pi_agent::tools::local_artifact_relative(&target_url)
				.ok_or_else(|| {
					ToolError::ExecutionFailed(format!(
						"{target_url} is not a local artifact path"
					))
				})?;
			pi_agent::tools::write_artifact(
				root,
				&relative,
				content.as_bytes(),
				&ctx.cancellation_token,
			)
			.await
			.map_err(|error| {
				ToolError::ExecutionFailed(format!(
					"Failed to write {target_url}: {error}"
				))
			})?;
		}
		let resolved = resolve_reviewed_plan(&proposal, &target_url, root)?;
		self.shared
			.queue_pending(resolved.clone())
			.map_err(ToolError::Rejected)?;
		let details = json!({
			"planFilePath": resolved.plan_file_path,
			"title": resolved.title,
			"planExists": true,
			"executionSwitchPending": true,
		});
		Ok(ToolResult {
			tool_call_id: String::new(),
			content: format!(
				"Plan approved at {}. The execution switch is pending; finish this turn and wait for the mode change before implementing.",
				resolved.plan_file_path
			),
			is_error: false,
			terminate: false,
			details: Some(details),
		})
	}
}

/// Contained `resolveApprovedPlan`: plan content is read through the artifact
/// fd walk, never through the working tree.
fn resolve_reviewed_plan(
	proposal: &plan::PlanProposal,
	target_url: &str,
	root: &Path,
) -> Result<plan::ResolvedPlan, ToolError> {
	plan::resolve_approved_plan(
		proposal.title.as_deref(),
		target_url,
		&|url| {
			let relative = pi_agent::tools::local_artifact_relative(url)?;
			pi_agent::tools::read_local_artifact(root, &relative).ok()
		},
		&|| plan::list_local_plan_files(root),
	)
	.map_err(ToolError::Rejected)
}

fn render_handoff(plan_file_path: &str, title: &str) -> String {
	PLAN_YOLO_HANDOFF_PROMPT
		.replace("{{title}}", title)
		.replace("{{planFilePath}}", plan_file_path)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn target() -> ModelSwitch {
		ModelSwitch {
			provider: "mock".to_string(),
			model:    "fast".to_string(),
			effort:   None,
			thinking: None,
		}
	}

	fn shared() -> Arc<PlanYoloShared> {
		Arc::new(PlanYoloShared::new(
			Arc::new(Mutex::new(PlanGate::new(false))),
			PlanYoloArm { target: target(), thinking: None },
		))
	}

	fn temp_root(name: &str) -> PathBuf {
		let path = std::env::temp_dir().join(format!(
			"omp-plan-yolo-{name}-{}-{}",
			std::process::id(),
			std::time::SystemTime::now()
				.duration_since(std::time::UNIX_EPOCH)
				.expect("clock")
				.as_nanos()
		));
		std::fs::create_dir_all(&path).expect("temp root");
		path
	}

	#[test]
	fn arm_publishes_plan_mode_and_rollback_restores_the_gate() {
		let shared = shared();
		let arming = shared.arm().expect("arm");
		{
			let gate = shared.gate();
			let gate = gate.lock().unwrap();
			let state = gate.state.as_ref().expect("plan mode");
			assert!(state.enabled);
			assert_eq!(state.plan_file_path, DEFAULT_PLAN_FILE_URL);
		}
		shared.rollback_arm(arming);
		assert!(!shared.is_armed());
		assert!(shared.gate().lock().unwrap().state.is_none());
		// TS resets `#planYoloArmed` on tool-set failure, so a retry is legal.
		assert!(shared.arm().is_some());
	}

	#[test]
	fn augment_tools_adds_write_only_to_an_enabled_restricted_set() {
		assert_eq!(augment_tools(true, None), None);
		assert_eq!(
			augment_tools(true, Some(&["read".to_string()])),
			Some(vec!["read".to_string(), "write".to_string()])
		);
		assert_eq!(
			augment_tools(true, Some(&["read".to_string(), "write".to_string()])),
			Some(vec!["read".to_string(), "write".to_string()])
		);
		assert_eq!(
			augment_tools(false, Some(&["read".to_string()])),
			Some(vec!["read".to_string()])
		);
	}

	#[test]
	fn failed_switch_keeps_the_guard_and_pending_approval() {
		let shared = shared();
		let _arming = shared.arm().expect("arm");
		let root = temp_root("failed-switch");
		std::fs::write(root.join("PLAN.md"), "# Plan\n\nDo it.\n").expect("plan");
		*shared.pending.lock().unwrap() = Some(PlanYoloPending {
			target: target(),
			thinking: None,
			plan_file_path: "local://PLAN.md".to_string(),
			title: "Plan".to_string(),
			reviewed_content: "# Plan\n\nDo it.\n".to_string(),
		});

		let error = shared
			.commit_pending(Some(&root), |_| Err("no key".to_string()))
			.expect_err("build must fail");
		assert!(error.contains("no key"));
		assert!(shared.gate().lock().unwrap().state.as_ref().is_some_and(|state| state.enabled));
		assert!(shared.has_pending());
		let _ = std::fs::remove_dir_all(root);
	}

	#[test]
	fn successful_switch_is_atomic_and_renders_the_handoff() {
		let shared = shared();
		let _arming = shared.arm().expect("arm");
		let root = temp_root("successful-switch");
		let content = "# Plan\n\nDo it.\n";
		std::fs::write(root.join("PLAN.md"), content).expect("plan");
		*shared.pending.lock().unwrap() = Some(PlanYoloPending {
			target: target(),
			thinking: None,
			plan_file_path: "local://PLAN.md".to_string(),
			title: "Plan".to_string(),
			reviewed_content: content.to_string(),
		});

		let (commit, built) = shared
			.commit_pending(Some(&root), |_| Ok(7))
			.expect("switch");
		assert_eq!(built, 7);
		assert!(commit.handoff.contains("local://PLAN.md"));
		assert!(commit.handoff.contains("Plan"));
		assert!(commit.notice.contains("mock/fast"));
		assert!(shared.gate().lock().unwrap().state.is_none());
		assert!(!shared.has_pending());
		// One-shot: a completed plan-yolo never re-arms.
		assert!(shared.arm().is_none());
		let _ = std::fs::remove_dir_all(root);
	}

	#[test]
	fn changed_plan_fails_revalidation_without_exiting_the_guard() {
		let shared = shared();
		let _arming = shared.arm().expect("arm");
		let root = temp_root("changed-plan");
		std::fs::write(root.join("PLAN.md"), "# Changed\n").expect("plan");
		*shared.pending.lock().unwrap() = Some(PlanYoloPending {
			target: target(),
			thinking: None,
			plan_file_path: "local://PLAN.md".to_string(),
			title: "Plan".to_string(),
			reviewed_content: "# Original\n".to_string(),
		});

		let error = shared
			.commit_pending(Some(&root), |_| Ok(()))
			.expect_err("changed plan");
		assert!(error.contains("changed after approval"));
		assert!(shared.gate().lock().unwrap().state.as_ref().is_some_and(|state| state.enabled));
		assert!(!shared.has_pending());
		let _ = std::fs::remove_dir_all(root);
	}
}
