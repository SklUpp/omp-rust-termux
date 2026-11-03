# Rust CLI verification and integration

[Architecture](rust-cli-project-context.md) · [Scope and evidence](rust-cli-parity-plan.md)

## Working model

Each task has an isolated worktree, one implementation owner and an independent reviewer. Reuse both within that task. Shared files require explicit interfaces and parent-side integration, not mixed task ownership.

Finish the current implementation, fix review findings, exercise its real path, then integrate. READY and source review are not runtime acceptance. Do not begin new functionality while closing the current block.

Before integration, inspect tracked and untracked changes in every task worktree. Commit all intended changes and reconcile each unique commit with base. Preserve unrelated user work; do not silently commit or merge it. A clean tree alone does not prove its commits were integrated.

No force-push, reset, worktree clean, deletion, dependency installation, installed-CLI replacement, tag, release or PR without separate authorization. Superseded files are preserved under `$HOME/.trash-<date>/`; moving them does not reclaim storage.

## Local checks

Termux arm64 uses Bionic. Resolve executables through `command -v`; do not assume systemd, Docker, glibc or writable `/tmp`.

```sh
export TMPDIR="$HOME/task-tmp"
mkdir -p "$TMPDIR"
env CARGO_TARGET_DIR="$HOME/omp-work/target" \
  CARGO_INCREMENTAL=1 RUSTC_BOOTSTRAP=1 \
  RUSTFLAGS='-C link-arg=-Wl,--threads=1' \
  nice -n 15 cargo check --offline --locked -j1 -p omp-cli -p pi-agent --tests
```

Run one scoped Cargo command at a time, bounded to 600 seconds. Full/heavy suites run in Actions. A link timeout means no test result; do not repeat an unchanged failing or heavy command without a new hypothesis.

`cargo check` does not produce an executable. For a scoped runtime check, build the CLI with the same environment:

```sh
env TMPDIR="$HOME/task-tmp" CARGO_TARGET_DIR="$HOME/omp-work/target" \
  CARGO_INCREMENTAL=1 RUSTC_BOOTSTRAP=1 \
  RUSTFLAGS='-C link-arg=-Wl,--threads=1' \
  nice -n 15 cargo build --offline --locked -j1 -p omp-cli --bin omp
```

After a successful build, copy the executable to a unique HOME path and record SHA-256 before another worktree uses the shared target. Test that pinned copy. Another Cargo invocation can replace `target/debug/omp`.

## Exact-SHA CI

`.github/workflows/rust-agent.yml` is authoritative for toolchain, dependencies and gates. It checks/tests `pi-agent`, `omp-cli`, native matching and support crates, then builds Android arm64 release output.

Push triggers apply only to `main` and `feat/rust-agent-termux-bionic`; other candidate branches need dispatch:

```sh
git push origin HEAD:refs/heads/ci/<task>
gh workflow run rust-agent.yml --repo SklUpp/omp-rust-termux --ref ci/<task>
gh run watch <RUN_ID> --repo SklUpp/omp-rust-termux --exit-status
```

Confirm the run's `headSha` equals the candidate. A green Android build does not imply green host tests. An older green run never verifies newer changes.

## Android artifact

```sh
ART="$HOME/task-tmp/omp-art-<RUN_ID>"
mkdir -p "$ART"
gh run download <RUN_ID> --repo SklUpp/omp-rust-termux \
  -n omp-aarch64-linux-android -D "$ART"
chmod +x "$ART/omp"
python3 - "$ART/omp.sha256" "$ART/omp" <<'PY'
import hashlib, pathlib, sys
manifest, binary = map(pathlib.Path, sys.argv[1:])
expected = manifest.read_text().split()[0]
with binary.open('rb') as stream:
    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
assert len(expected) == 64 and actual == expected, 'artifact hash mismatch'
print(actual)
PY
file "$ART/omp"
```

The manifest contains a build-relative path, not the flat downloaded filename. Compare its digest to the actual executable. Download failures and partial archives are not artifacts accepted for execution. Never log signed download URLs or credentials.

## Runtime acceptance

Exercise invocation, typed result, failure, cancellation and persisted state. Use real CLI/RPC/ACP surfaces and localhost fixtures; mocks must emit valid protocol data and assertions must observe consumers.

| Changed surface | Runnable acceptance |
|---|---|
| Saved agent outputs | `OMP_BIN="$ART/omp" bash scripts/agent-uri-home-harness.sh` |
| Live/persisted history | `OMP_BIN="$ART/omp" bash scripts/history-uri-home-harness.sh` |
| Local security scan | `OMP_BIN="$ART/omp" bash scripts/security-scan-home-harness.sh` |
| GC apply | `sh scripts/gc-apply-smoke.sh "$ART/omp"` |
| Advisor | `OMP_BIN="$ART/omp" python3 scripts/advisor-smoke-home.py` after C4 integration |
| Local learning/skills | `python3 scripts/local-learning-home-smoke.py --omp "$ART/omp" --keep` after C7 integration |

Keep failed fixtures and exact executable hashes. Remove task-created diagnostic scaffolding after proof. Do not rerun unchanged suites after exact green CI; run the missing Android-specific paths.

## Closure

1. Close independent findings; keep explicit unsupported/paused boundaries.
2. Integrate scoped commits, resolving shared-file conflicts without dropping behavior.
3. Verify the integrated SHA, not just individual task branches.
4. Match manifest/hash and exercise affected Android paths.
5. Reconcile external worktrees: clean current-block files and no unaccounted unique work. Report unrelated changes separately.
6. Update only the plan's evidence table. Do not create another checkpoint diary.

Performance remains a separate gate: matched 8-turn and 30-turn histories, same device/config/provider fixture, startup/CPU/PSS against the installed Bun CLI. `android/scripts/bench.sh` measures Bun and cannot establish Rust performance. No threshold or improvement claim without measurements.
