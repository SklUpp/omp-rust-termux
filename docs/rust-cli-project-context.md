# Rust CLI architecture

Native Rust `omp` for Android arm64/Bionic. This is a separate implementation, not a Bun launcher. [Parity plan](rust-cli-parity-plan.md) owns scope and evidence; [verification procedure](provider-checkpoint.md) owns build, review and publication gates.

## Boundary

- `crates/omp-cli`: CLI, configuration, model/auth resolution, sessions, approvals, RPC/ACP adapters and runtime composition.
- `crates/pi-agent`: provider loop, typed messages/events, tools, supervision and protocol clients.
- `pi-iso`, `pi-walker`, `pi-grep-pattern`, `pi-natives`: isolation, traversal, matching and native services.
- TypeScript under `packages/` is the behavioral reference. Match observable protocols, not its internal class structure.

Normal execution must not start Bun/Node. JavaScript starts lazily for explicit JS eval or browser execution. Interpreters, SSH, language/debug servers, Chromium and external accounts remain prerequisites; missing prerequisites produce real errors.

Interactive TUI, desktop control, plugin/hook/extension execution, local audio playback/capture and WebRTC are excluded. File-backed skills and configuration discovery are not extension execution. Installed `$PREFIX/bin/omp`, installer and existing Android release packaging remain unchanged.

## Execution and ownership

`main`, RPC and ACP construct `agent::RunRequest`. `runtime::PiAgentRunner` resolves the model, credentials, policies and registry, then drives `pi_agent::AgentLoop`. Forward live bounded events; do not reconstruct a stream after completion.

A `RunnerSession` owns mutable runtime resources. Scope selection is session ID, then session file, then one anonymous scope. Its `Supervisor` owns children, process jobs, delivery queues and usage ledger. Closing one session must not cancel another session's work.

Every production `AgentRunner` wrapper must forward lifecycle methods as well as `run`. Default trait methods are not proof that a concrete wrapper supports advisor, cancellation or session teardown.

Each tool call starts and ends once. Partial text, tool results, usage and error state survive cancellation and provider failure. Task failure and abort are distinct. Exactly-once background accounting uses `resultKey`, not task ID or output filename.

## Tool registry and policy

Internal registry names are stable. Wire aliases affect model schemas only: `edit` may expose `apply_patch`; approvals and allowlists still resolve the internal capability. Ambiguous wire aliases fail closed.

Selection follows tools-enabled, configured allowlist, feature gate and caller capability ceiling. Explicit disabled tools fail clearly. Host tools cannot replace reserved builtins or internal URL namespaces. Root-only tools are mounted after child templates are created; children receive their own rebound state.

Approval and capability are separate. Yolo cannot bypass Plan write containment, a restricted worker's read-only LSP ceiling or a caller's tool ceiling. `--add-dir` is a workspace feature, not a sandbox; `--allow-home` controls startup directory behavior, not permissions.

## Sessions and persistence

`session::SessionStore` writes JSONL with stable IDs, parent linkage, provider metadata and ordered typed content. Resume/fork must preserve text, images, tool calls/results and signed/redacted thinking. Rewind branches conversation history; it does not restore Git or filesystem state.

Named `agent://` outputs are separate from live `history://` transcripts. Outputs use stable allocation and no-clobber publication. History reads caller-authorized live snapshots or persisted journals; a display path is not read or mutation authority.

Internal URL access pins authorized roots and resolves through validated descriptors. Android execute-only ancestors require `O_PATH`; final readable descriptors are reopened relative to the pin and identity-checked. Missing session identity grants no foreign manager access.

Artifact publication stages complete bytes, then uses hard link or native no-replace rename. Foreign destinations remain intact; unsupported publication fails explicitly. No unchecked copy fallback. These checks are not a same-UID filesystem sandbox.

## Providers and credentials

Native dispatch reuses model metadata and provider-specific HTTP/stream adapters. Endpoint changes must not inherit another endpoint's credentials. Preserve message ordering, tool-call pairing, thinking signatures, terminal reasons and actual served-model usage.

Restricted security workers pin an exact OAuth row. Multiple accounts without an explicit pin are ambiguous; hashing a session ID is not an account pin. Resolve credentials before each HTTP request; an allowed authentication retry refreshes the same row and retries that request, never the whole agent session.

Redact the credential actually used by each request, including retries and error streams. Empty secrets must not corrupt error text. Ordinary provider behavior stays unchanged when no callback is configured.

New provider features remain paused. Correctness fixes required by the current block are allowed; provider presence does not establish full dialect parity.

## Current subsystems

| Subsystem | Runtime contract | Source |
|---|---|---|
| Plan/prewalk | Explicit policy transitions, required-tool reminders, persisted model changes and contained proposal writes | `plan_yolo.rs`, `prewalk.rs`, `acp/plan.rs` |
| Delegation | Session-owned jobs, park/revive, isolation, background delivery and deduplicated usage | `pi-agent/src/supervisor.rs`, `tools/task.rs`, `hub_park.rs` |
| Eval | Interpreters started on demand, child-specific provider selection, caller-scoped artifacts and tool calls | `eval_host.rs`, `pi-agent/src/eval/` |
| Protocols | MCP stdio/HTTP/SSE, LSP and DAP with bounded cancellation/reap | `pi-agent/src/mcp/`, `lsp/`, `dap/` |
| Security scan | Opt-in local coordinator, restricted worker, persistent journal, once-only bundle publication, read-only `security://` | `security/`, `security_runtime.rs`, `auth/security.rs` |
| Local learning | Backend-gated lessons, managed skills, caller-scoped snapshots, skill/memory resources and restricted capture | `autolearn.rs`, `local_memory.rs`, `managed_skills.rs`, `skills.rs` |
| Advisor | Separate model/session identity, backlog, guarded advice, child opt-in, cards and independent cost | C4 isolated candidate; not yet integrated/accepted |

The plan records acceptance. This table describes architecture, not completion.

## Storage and safety limits

- GC defaults to dry-run. Apply uses a whole-run guard, no-clobber archive publication and explicit failures. Cross-filesystem artifact moves refuse with `EXDEV`; no lossy copy/delete fallback.
- Rust GC executions serialize through a stable guard. Unmodified TypeScript recovery is outside that guarantee; already-open writers can append to cold sessions.
- Security project keys use the canonical root, a bounded readable slug and 12-hex hash. The corrected slug collapses runs of forbidden characters. Earlier Rust stores under affected Unicode roots need explicit reconciliation; no automatic migration exists.
- Secrets, auth databases and raw user transcripts never belong in documentation or test reports. Use isolated HOME fixtures and localhost providers.
- Temporary artifacts belong under `$HOME/task-tmp`. A shared Cargo target is a cache, not binary provenance: pin each tested executable to a unique path and record its hash.
