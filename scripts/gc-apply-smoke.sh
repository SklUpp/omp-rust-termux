#!/bin/sh
# HOME-confined GC apply smoke. Never touches the real agent dir: every
# invocation passes an explicit --agent-dir under a fresh mktemp tree.
#
# Usage: scripts/gc-apply-smoke.sh /path/to/omp
# Exits 0 only when dry-run, apply, repeat-apply and collision-error behavior
# are all observed. Cleans up only its own mktemp tree.
set -eu

OMP_BIN=${1:-${OMP_BIN:-}}
if [ -z "$OMP_BIN" ] || [ ! -x "$OMP_BIN" ]; then
	echo "usage: $0 /path/to/omp" >&2
	exit 2
fi

SMOKE_ROOT=$(mktemp -d "$HOME/gc-apply-smoke.XXXXXX")
trap 'rm -rf "$SMOKE_ROOT"' EXIT HUP INT TERM

AGENT="$SMOKE_ROOT/agent"
BLOBS="$AGENT/blobs"
SESSIONS="$AGENT/sessions/project"
mkdir -p "$BLOBS" "$SESSIONS"

hash() {
	printf '%s' "$1" | sha256sum | cut -d' ' -f1
}

REFERENCED=$(hash referenced)
ORPHAN=$(hash orphan)
FRESH=$(hash fresh)

printf 'ref' >"$BLOBS/$REFERENCED"
printf 'orphan!' >"$BLOBS/$ORPHAN"
printf 'fresh' >"$BLOBS/$FRESH"
printf '{"type":"message","message":{"content":"blob:sha256:%s"}}\n' "$REFERENCED" \
	>"$SESSIONS/ref.jsonl"
touch -t 202001010000 "$BLOBS/$REFERENCED" "$BLOBS/$ORPHAN"

expect() {
	grep -Fq "$1" "$2" || {
		echo "smoke: expected $1 in $2" >&2
		cat "$2" >&2
		exit 1
	}
}

assert_absent() {
	[ ! -e "$1" ] || {
		echo "smoke: unexpected $1" >&2
		exit 1
	}
}

lock_absent() {
	assert_absent "$AGENT/gc.lock"
	assert_absent "$AGENT/gc.lock.break"
}

# 1. Dry-run reports the candidate without deleting it.
"$OMP_BIN" gc --agent-dir "$AGENT" --json --blobs >"$SMOKE_ROOT/dry.json"
expect '"apply": false' "$SMOKE_ROOT/dry.json"
expect '"wouldDelete": 1' "$SMOKE_ROOT/dry.json"
expect '"deleted": 0' "$SMOKE_ROOT/dry.json"
expect '"bytes": 7' "$SMOKE_ROOT/dry.json"
[ -e "$BLOBS/$ORPHAN" ] || {
	echo "smoke: dry-run deleted the orphan" >&2
	exit 1
}
lock_absent

# 2. Apply deletes only the old unreferenced candidate.
"$OMP_BIN" gc --agent-dir "$AGENT" --json --blobs --apply >"$SMOKE_ROOT/apply.json"
expect '"apply": true' "$SMOKE_ROOT/apply.json"
expect '"wouldDelete": 1' "$SMOKE_ROOT/apply.json"
expect '"deleted": 1' "$SMOKE_ROOT/apply.json"
expect '"bytes": 7' "$SMOKE_ROOT/apply.json"
assert_absent "$BLOBS/$ORPHAN"
[ -e "$BLOBS/$REFERENCED" ] || {
	echo "smoke: referenced blob was deleted" >&2
	exit 1
}
[ -e "$BLOBS/$FRESH" ] || {
	echo "smoke: fresh blob was deleted" >&2
	exit 1
}
lock_absent

# 3. Repeat apply is idempotent.
"$OMP_BIN" gc --agent-dir "$AGENT" --json --blobs --apply >"$SMOKE_ROOT/repeat.json"
expect '"wouldDelete": 0' "$SMOKE_ROOT/repeat.json"
expect '"deleted": 0' "$SMOKE_ROOT/repeat.json"
lock_absent

# 4. Archive apply round-trips gzip plus artifacts.
ARCHIVE_CONTENT='{"type":"session","id":"s1","cwd":"/tmp/project"}
{"type":"message","message":{"role":"assistant","stopReason":"stop","content":[]}}
'
printf '%s' "$ARCHIVE_CONTENT" >"$SMOKE_ROOT/expected.jsonl"
cp "$SMOKE_ROOT/expected.jsonl" "$SESSIONS/s1.jsonl"
mkdir -p "$SESSIONS/s1/artifacts"
printf 'artifact' >"$SESSIONS/s1/artifacts/note.txt"
touch -t 202001010000 "$SESSIONS/s1.jsonl"

"$OMP_BIN" gc --agent-dir "$AGENT" --json --archive --cold-archive-after-days 30 \
	--retain-newest-global 0 --retain-newest-per-cwd 0 >"$SMOKE_ROOT/archive-dry.json"
expect '"wouldArchive": 1' "$SMOKE_ROOT/archive-dry.json"
expect '"archived": 0' "$SMOKE_ROOT/archive-dry.json"
[ -e "$SESSIONS/s1.jsonl" ] || {
	echo "smoke: archive dry-run moved the session" >&2
	exit 1
}
lock_absent

"$OMP_BIN" gc --agent-dir "$AGENT" --json --archive --cold-archive-after-days 30 \
	--retain-newest-global 0 --retain-newest-per-cwd 0 --apply >"$SMOKE_ROOT/archive-apply.json"
expect '"archived": 1' "$SMOKE_ROOT/archive-apply.json"
assert_absent "$SESSIONS/s1.jsonl"
ARCHIVED="$AGENT/archive/sessions/project/s1.jsonl.gz"
[ -e "$ARCHIVED" ] || {
	echo "smoke: missing archived session $ARCHIVED" >&2
	exit 1
}
[ -e "$AGENT/archive/sessions/project/s1/artifacts/note.txt" ] || {
	echo "smoke: artifacts were not moved" >&2
	exit 1
}
gzip -dc "$ARCHIVED" >"$SMOKE_ROOT/roundtrip.jsonl"
cmp "$SMOKE_ROOT/roundtrip.jsonl" "$SMOKE_ROOT/expected.jsonl"
lock_absent

"$OMP_BIN" gc --agent-dir "$AGENT" --json --archive --cold-archive-after-days 30 \
	--retain-newest-global 0 --retain-newest-per-cwd 0 --apply >"$SMOKE_ROOT/archive-repeat.json"
expect '"wouldArchive": 0' "$SMOKE_ROOT/archive-repeat.json"
expect '"archived": 0' "$SMOKE_ROOT/archive-repeat.json"
lock_absent

# 5. Foreign archive destination is preserved and the source survives.
printf 'foreign' >"$AGENT/archive/sessions/project/s2.jsonl.gz"
printf '{"type":"session","id":"s2"}\n{"type":"message","message":{"role":"assistant","stopReason":"stop","content":[]}}\n' \
	>"$SESSIONS/s2.jsonl"
touch -t 202001010000 "$SESSIONS/s2.jsonl"
if "$OMP_BIN" gc --agent-dir "$AGENT" --json --archive --cold-archive-after-days 30 \
	--retain-newest-global 0 --retain-newest-per-cwd 0 --apply \
	>"$SMOKE_ROOT/collision.json" 2>"$SMOKE_ROOT/collision.err"; then
	echo "smoke: collision apply unexpectedly exited 0" >&2
	exit 1
fi
grep -Fq 'GC completed with 1 error' "$SMOKE_ROOT/collision.err" || {
	echo "smoke: collision error missing" >&2
	cat "$SMOKE_ROOT/collision.err" >&2
	exit 1
}
expect '"archived": 0' "$SMOKE_ROOT/collision.json"
[ -e "$SESSIONS/s2.jsonl" ] || {
	echo "smoke: collision removed the source" >&2
	exit 1
}
[ "$(cat "$AGENT/archive/sessions/project/s2.jsonl.gz")" = foreign ] || {
	echo "smoke: foreign destination changed" >&2
	exit 1
}
lock_absent

echo "gc apply smoke: PASS ($SMOKE_ROOT)"
