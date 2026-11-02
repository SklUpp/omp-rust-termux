# Provider checkpoint — 2026-11-03

## Decision

Provider implementation is paused at the user's request. Do not start new provider features, dialects or refactoring without a new instruction. Priority moves to Plan-yolo and remaining C1–C7 work. This checkpoint is not full C8 completion.

## Verified baseline and publication boundary

- Last fully CI- and Android-runtime-verified revision: `dab7ed8e8dc706d503da247adc1fe271a28b8304`.
- CI: https://github.com/SklUpp/omp-termux/actions/runs/41485151065 — host, native and Android jobs successful.
- Downloaded binary: `/data/data/com.termux/files/usr/tmp/omp-dab7ed8e8d.Gg8ZlW/omp`.
- SHA-256: `665edd6ad40d08bca3909c8fd8cdd032b202b88c59561b590380550128e3b2d2`, matched manifest. Actual RPC thinking-cache smoke passed three scenarios, exit 0, 2.03 s. Installed CLI was not replaced.
- Later published Anthropic priority revision: `0da55ccd934951bd661d58c7b37f5bf402f9bb64`.
- Its CI: https://github.com/SklUpp/omp-termux/actions/runs/41528328842 — native and Android successful; host failed compiling a test-only `CliChildModelFactory` initializer missing `service_tier_by_family` at `crates/omp-cli/src/runtime.rs:6342`.
- Fix authored as `26005d554f`, integrated as `601b60aa49`: supplies an empty tier map to that cancelled-factory fixture. Full CI verification of this correction remains pending. Do not call `0da55ccd93` fully green or artifact-accepted.
- Later Plan-yolo changes are outside this provider checkpoint. Preserve the constructor correction when integrating them.

## Locally accepted behavior

### Shared provider/runtime contracts

- `crates/pi-agent/src/provider.rs`: `ModelProvider`, `StreamChunk`, indexed block events and `StreamAssembler`.
- `crates/pi-agent/src/state.rs`: shared `Message`, ordered content parts, signed/redacted thinking, tool calls, usage, attribution and terminal reasons.
- Native dispatch selects explicit provider APIs; unknown APIs fail before contact rather than silently falling back to OpenAI.
- Root and child routing, endpoint/key isolation and typed replay use these contracts. Shared types do not imply all helper implementations are consolidated.

### Ollama

- Native `/api/chat` NDJSON, history conversion, malformed tool/history healing, dialect thinking demotion, retry hints and empty-completion handling.
- Terminal completion waits for EOF; records after `done:true` remain processed.
- Adapter-level async payload hook receives resolved model metadata. This does not implement CLI extension/hook execution.
- Default-client redirects disabled; injected clients own their redirect/TLS policy. Configured header values are redacted from diagnostics.
- Bounded parser uses no native recursion; delimiter processing O(n log n), memory/output O(n). Deep nesting, nested siblings, malformed HTTP dates and partial wrappers covered.
- Evidence: 47 wire tests passed, followed by the final wrapped-Qwen request regression and dialect regression, one each; 12 source-module regressions passed. Correctness and scoped security findings closed.
- Actual Android Ollama/Hermes NDJSON and branched foreign-thinking replay passed on the previously verified `9848ffab82cef5afe109bac712313ce8072702ed` artifact; that revision is included in the later verified baseline.
- Inlined retry, dialect/healing and history helpers remain in `provider_ollama.rs`. Consolidation/refactoring is paused, not a new implementation task.

### Anthropic priority slice

- Source-exact HTTP 400/429 fast-mode rejection downgrade, terminal `disabledFeatures`, sticky family-tier reset, active-branch persistence, resume/fork, real child inheritance and live notices.
- `--service-tier` overlays OpenAI family only. Anthropic priority comes from `tier.anthropic` or live fast-mode selection. Explicit `none`/null in a higher configuration layer clears the lower layer.
- Statusless errors inside HTTP 200 SSE surface without downgrade retry. Notices do not enter model guidance or cause extra provider requests.
- Local evidence: priority suite 8/8, RPC fast-mode 3/3, native Anthropic 33/33, root/child dispatch 4/4, all passing.
- Parent actual ACP smoke: exit 0, 1.48 s; nine valid JSON-RPC frames, exactly one downgrade notification, two first-turn provider requests (rejection/retry), one non-priority follow-up request, no notice-induced provider request.
- Harness: `/data/data/com.termux/files/usr/tmp/acp_priority_smoke.py`, accepts `OMP_BIN`. Correctness and scoped security findings, including ACP notice mapping, closed.
- This is local acceptance, not successful full CI/artifact acceptance of the later publication.

## Retained work; not waived

- Full Anthropic request/stream parity remains open: remaining gateway/account branches, request options, fallback/served-model attribution, server-tool handling and source-specific recovery contracts require re-audit against current implementation before coding. Earlier gap lists may be stale; passing priority tests does not close them.
- Other provider dialects and remaining C8 cloud/browser/image integrations are not authorized for new implementation while paused.
- No real external-account acceptance is claimed; localhost protocol fixtures are distinct from live service validation.
- On resumption, first verify the constructor correction in exact-SHA full CI and exercise the affected Android artifact. Then reconcile remaining contracts against TS source and the canonical `docs/rust-cli-parity-plan.md`; do not restart already accepted work.

## Ownership and retained workspace

- Original provider owner: `WireNativeProviderDispatch`; ordinary worktree `/data/data/com.termux/files/home/omp-anthropic-priority`, clean at author commit `26005d554f` when paused.
- Worktrees and task-created smoke scripts retained. No package installation, installed-CLI replacement, tag or release performed for this checkpoint.
