# Rust CLI parity plan

[Architecture](rust-cli-project-context.md) · [Verification procedure](provider-checkpoint.md)

## Objective and scope

Implement the retained TypeScript CLI behavior in native Rust for Android arm64/Bionic. Match invocation, results, errors, cancellation and persisted state.

The current block completes advisor and local learning/managed skills. Fix existing defects before integration. Do not add new features during this block.

The full project remains incomplete. Provider feature work is paused, not waived. External credentials, servers and interpreters are prerequisites, not substitute implementations.

Excluded: interactive TUI, desktop control, plugin/hook execution, local audio capture/playback and WebRTC. Existing installers and the installed CLI remain unchanged.

## Completion rules

A requirement closes only with implementation, independent review, relevant tests, the actual runtime path and exact-commit CI. Android-specific behavior requires Android execution.

Source review is not runtime proof. A successful Android build is not host-test success. A missing prerequisite must produce the correct error without side effects.
Every upstream registry entry must have equivalent Rust behavior or an explicit exclusion: modes, flags, subcommands, tools, wire dialects and configuration/session paths. Use real TypeScript session and protocol fixtures, not source-text assertions. Excluded requests fail with a specific reason and no side effects. Never accept a rejected option as a no-op.


Do not add a `session` subcommand. Use existing resume/fork/continue flags and RPC session methods. Rewind changes conversation history, not Git or filesystem state.

## Current evidence

| Revision or slice | Verified | Limit |
|---|---|---|
| `9537f2d61844159cfa4a188596d7801352ad6651` | [CI 41591200807](https://github.com/SklUpp/omp-termux/actions/runs/41591200807): host, native and Android pass; saved-agent output producer/read/resume passes on the downloaded Android binary | Last fully accepted published checkpoint, not proof of later changes |
| Same Android artifact | SHA-256 `f77d9d83a6ae45079be4d8cc19ac70f88ff99dd06d19daa425c8fbc8d20ccb48` matches its manifest | Historical artifact |
| History URLs | 21 protocol tests pass; actual live read, persisted read, park, resume, revive and isolation pass | Local evidence; later integrated commits need exact-commit CI |
| GC apply | Actual isolated-HOME dry-run/apply/repeat/collision scenario passes | Cross-filesystem artifact moves refuse; earlier phone test link timeout is not a test pass |
| Local security scan | 14 coordinator tests, 7 HTTP authentication tests and 29 mounted-runtime assertions pass | Local OAuth fixture, not a live account; cloud actions remain open |
| C7 `e10cae8998a40f6350537ce24dcc1e9125dfcb0b` | [CI 41720874309](https://github.com/SklUpp/omp-termux/actions/runs/41720874309): host, native and Android pass; integrated in `d17fb8a908` | Downloaded-artifact runtime acceptance remains open |
| C4 advisor | Pinned `170345d12b` binary passed CLI text/JSON, RPC, ACP, child opt-in, failure and cancellation; integrated in `b18fa21c01` | Standalone smoke does not verify the integrated executable |
| Integrated `961bb343bc` | [CI 41733981790](https://github.com/SklUpp/omp-rust-termux/actions/runs/41733981790): native tests pass | Host and Android compilation fail: duplicate child history seeding moves `initial_message` before the advisor callback; corrected candidate needs fresh checks |
| Integrated corrections | Local `cargo check --offline --locked -j1` passes for `pi-agent --lib --tests` and `omp-cli --tests`; correctness and security reviews close advisor/MCP isolation defects | Behavioral regressions, exact-commit CI and downloaded Android runtime acceptance remain open |
| Integrated `937abd6a6b` | [Rust CI 41767764302](https://github.com/SklUpp/omp-rust-termux/actions/runs/41767764302) passes host, native and Android; downloaded Android binary SHA-256 `cad6a7a9460601ec9951be6051d94e3f5326ae0e4323a4629b9fc925a58f6763` passes advisor, local learning, security publication and in-flight cancellation smoke | Local mock services, not live accounts; remaining category contracts stay open |

No row closes an entire category unless every retained contract below is verified.

## Retained contracts

### C1 — Files, processes and session data

- Internal URLs: caller-scoped `agent://`, `history://`, `artifact://`, `local://`, `memory://`, `skill://` and supported protocol resources. Preserve real producers, selectors, provenance, read-only behavior and access boundaries.
- `@file`: text up to 5 MiB, images up to 25 MiB, directory listings and piped stdin before attached file text.
- SQLite: schema/sample, primary-key or rowid lookup, paged `where`/`order` and raw `SELECT`. Enforce `query_only` and the authorizer.
- Saved agent outputs: stable named and nested output IDs, no overwrite of foreign files, failed/aborted output handling and fresh-process resume. Output IDs differ from lifecycle IDs and `resultKey`.
- History: live producer snapshots, persisted fallback, nested IDs, scope isolation and actual lifecycle transitions. Do not present a renamed file as live history.
- Artifacts: bounded reads, spilling, descriptor-based containment, cancellation and atomic no-clobber publication. Preserve foreign destinations.
- Archives: existing Python standard-library ZIP/package aliases, tar variants and compressed streams work. Unsupported formats must report a missing backend, not return binary data as text.
- Document conversion remains open for PDF, DOCX, PPTX, XLSX and EPUB. Legacy DOC/PPT/XLS/RTF are excluded. New converter dependencies require approval.
- Conflict resolution: session-owned conflict history, numeric IDs, bulk directives, stale-write refusal and cross-turn/reset isolation. Existing 28-case and numeric regressions remain accepted evidence.
- Preserve `conflict://<N>[/ours|/theirs|/base]` reads and `write conflict://<N>` / `conflict://*`. Keep the accepted apply-patch corpus adapter; never edit the corpus to make tests pass.
- Processes: PTY interaction, readiness by log/port, bounded 25 MiB logs, cancellation/reap and no orphan child. Reject `detached+pty`. Preserve broker-owned cleanup and same-name reuse.
- Julia: ordered candidate selection, bounded readiness, one retained interpreter and no false ready. Real cold-start fixtures use an explicit budget; production deadline tests remain separate.
- Session JSONL: lossless TypeScript resume/fork with IDs, parents, text, images, tool calls/results, signed/redacted thinking and usage. Do not modify the source session during fork.

### C2 — Delegation, lifecycle and accounting

- `hub`: owner-scoped direct children, park/revive, first message/job/timeout/steer wait, hung-job cancellation and exactly-once delivery.
- Background work: nested ownership, session teardown, interruption and bounded queues. A completed job remains observable until consumed.
- Usage: account once by `resultKey`, including hub consumption, automatic delivery, nested results and aborted work. Do not claim one global linear conversation.
- Isolation: clean, dirty, untracked, symlink, conflict, merge-back and abort paths preserve data. Preserve accepted transcript-only rewind and atomic publication behavior.
- Eval: persistent state, timeout, cancellation, background completion and output routing. Bind child models to the existing provider factory, including nested calls.
- Start interpreters only for explicit eval. Require zero JavaScript processes before that call, one managed interpreter per kernel and cleanup on cancellation or EOF.
- Checkpoint/rewind: preserve branch metadata through resume/fork and retain abandoned JSONL entries. Do not restore Git or files.

### C3 — MCP, LSP, DAP, RPC and ACP

- MCP protocol `2025-11-25`: initialization, discovery, calls, resources, cancellation, EOF and cleanup over stdio and HTTP/SSE.
- Consume the upstream MCP JSON configuration path and schema. Use `mcp__<server>_<tool>` names and fail-closed launch/approval gates. Return failures as tool errors, not empty results.
- Client-side cancellation is required. A `notifications/cancelled` notification is optional for this protocol version. Deduplicate only identical terminal payloads.
- Preserve authentication, same-origin redirects, session-ID rotation, safe read-only recovery, mutating-request no-replay and overall deadlines.
- Resource pagination is bounded to 64 pages and 10,000 entries. Release transport locks before awaiting other work.
- LSP/DAP: actual invocation, typed results, cancellation and bounded shutdown/reap. Fixtures prove the protocol, not availability of an uninstalled external server.
- RPC: streaming events, session switching and authentication failure before ready. ACP: session isolation, Plan guards, approvals and cancellation.
- The existing RPC integration harness must run against `--mode rpc`; protocol types alone do not close this consumer contract.
- Existing scoped LSP/DAP and MCP resource/redirect/recovery/cancellation tests remain evidence. Remaining transport and interactive-approval combinations are open.

### C4 — CLI policies and advisor

- `--plan <model>` selects a role override with the same argument/environment precedence as `--smol` and `--slow`. It does not enter Plan mode.
- Keep `--prewalk`, `--no-prewalk`, `--prewalk-into`, `--plan-yolo` and `--plan-yolo-into` distinct. `--plan-yolo-into` requires `--plan-yolo`; default prewalk target is `@smol`.
- Plan reminder continuation stays within the current prompt. Required tool choice is one-shot; text/yield does not reset the waiting gate. Limit reminders to three events.
- ACP Plan writes remain confined to `local://` proposal artifacts until approval. Preserve the `xd://propose` composite policy, typed AST handling, child capabilities and durable model transitions.
- `--smol`/`--slow` preserve argument/environment precedence. `--no-lsp` applies to normal requests; explicitly restricted workers follow their own verified contract.
- `--external-thinking` enables `think` and disables native reasoning. `--print-thoughts` applies only to print output.
- `--service-tier` validates the exact enum. Omit empty, `none` and `inherit`; send supported values only to the OpenAI family.
- `--hide-thinking` is TUI-only upstream behavior. Do not invent a headless filter.
- `--add-dir` adds workspace roots, not confinement. `--allow-home` controls startup directory behavior, not permissions.
- Verify `--no-pty`, `--no-rules`, `--models` and existing export behavior against their exact TypeScript callers before closing them.
- Advisor: roster precedence, advisor/slow model selection, separate identity/context/cache, tool guards, backlog limits, advice delivery and independent accounting.
- Preserve advisor child opt-in, paused-roster behavior, reset generation, failure versus abort and bounded final review. Deliver late cards through CLI/RPC/ACP without duplicates.
- Each flag needs real invocation, rejected-value behavior, no rejection side effects and applicable wire assertions.

### C5 — Local commands and storage

- `stats`, `render`, `export`: real session input, output files/stdout and JSON. Corrupt databases exit 1 before stdout or ready.
- `compress`, `cleanse`, `commit`, `ttsr`: preserve actual model/Git/ruleset behavior. Do not replace them with invented shared abstractions.
- `images`, `gallery`, `grievances`, `gc`, `setup`: preserve approvals, scoped file output and error behavior. Do not delete data automatically.
- Gallery uses native ANSI output without `vhs`. Speech acceptance is file generation, not playback.
- GC apply: blob references/grace period, retention selection, gzip/artifacts, history/FTS, stats lineage, both WAL checkpoints and lock cleanup.
- GC cross-filesystem artifact moves explicitly refuse with `EXDEV`. Rust executions share a stable guard; unmodified TypeScript recovery is outside that guarantee.
- Cold-session compression cannot prevent writes through already-open file descriptors. Do not describe it as a sandbox.

### C6 — Shared remote backends and security

- SSH host add/remove/list and actual `ssh://` read/write/grep use isolated test configuration. Remote availability remains a prerequisite.
- GitHub operations use `gh`, exact auth errors and approval separation for reads/mutations. Implement real `issue://` and `pr://` consumers; refusal alone is not full parity.
- Search commands and `web_search` share a backend. Credentials and network access are runtime prerequisites.
- Local `security_scan`: disabled by default, normal allowlist selection, preflight/start/status/cancel/validate, restricted worker, persistent journal and once-only publication.
- Preserve `--no-tools`, explicit disabled errors, reserved-name protection, real API `sessionFile`, forced worker LSP with read-only actions and child inheritance.
- Select worker tools from `SECURITY_SESSION_TOOLS`, excluding `security_publish` from that configured set. Mount the scan-specific publication tool separately. Force `lsp.enabled` on and inherit the read-only action ceiling in children.
- Exact OAuth selection uses an explicit row or the sole eligible row. A session hash is not a stored account pin. Refresh the same row per failed request; never replay the whole scan session.
- `security://` is read-only. Project identity uses the canonical root, readable slug and 12-hex hash. Earlier stores with runs of forbidden characters can need explicit reconciliation after the slug correction.
- Cloud security actions, broker and gateway behavior remain open. Local fixture success does not establish live-account acceptance.

### C7 — Memory and skills

- Local lessons and managed skills use directory storage, not plugin execution. Honor backend and autolearn gates, top-level-only tools and reserved names.
- Use one skill snapshot per session. Discovery, disabled extensions, prompt listings, slash commands, autoload and URL reads must agree.
- Preserve authored-skill precedence, strict names, safe frontmatter, atomic updates, containment, cancellation and honest partial outcomes.
- `skill://` and `memory://` expose real resources. Do not fabricate a summary from lessons or grant writes through source metadata.
- Autoload guidance is hidden guidance, not another user assignment. Capture a resolver for each child; later session changes must not replace it.
- Capture excludes MCP, extensions and host tools, including inherited child capabilities. Normal requests retain their tools.
- Plan, abort and both initial/current goal state gate automatic capture. Do not duplicate user prompts in persistence.
- Memory schema migration must be explicit and loss-safe. Backend-gated retain/edit/recall/reflect, Mnemopi and Hindsight remain unfinished.

### C8 — Paused provider and optional dependency work

- Keep auth/discovery complexity low. Verify each of the 14 wire dialects with request, stream, error and cancellation fixtures.
- Retain cloud `say`, `generate_image`, `share`, `usage`, `token` and account-dependent broker/gateway contracts. Do not substitute generic OAuth for a service-specific flow.
- `tiny-models` requires ONNX dependency approval. Browser work requires working external CDP; installed Chromium is not a validated backend.
- `inspect_image` remains deferred without decoder/resvg/WebP approval. External credentials do not justify removing interfaces.
- No new dependency, installed-CLI replacement, tag, release or PR is authorized by this plan.

## Integration and remaining gates

1. Finish current C4/C7 fixes and independent review without adding scope.
2. Run the actual changed paths on identified binaries. Keep failure artifacts and hashes.
3. Integrate scoped commits and reconcile every external worktree against base. Report unrelated user changes separately.
4. Verify the integrated commit in CI and on the downloaded Android artifact.
5. Close remaining categories only against their complete retained contracts.

Final acceptance includes cancellation/leak checks, lazy JavaScript startup, two model/tool turns, resume/fork and applicable MCP/RPC/ACP/task/hub/SSH scenarios.

Measure startup, CPU and PSS against the installed Bun CLI with matched 8-turn and 30-turn histories. Use the same device and isolated configuration. No Rust-specific threshold exists. Bun packaging targets are not Rust acceptance criteria.

`android/scripts/bench.sh` measures Bun and cannot establish Rust performance. Add a live credential-backed smoke only when credentials are available.
