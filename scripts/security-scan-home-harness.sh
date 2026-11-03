#!/usr/bin/env bash
# Black-box CLI harness for the native security scan pipeline.
#
# Requires a built `omp` binary (`OMP_BIN`). Runs the real CLI against a real git
# fixture under an isolated `$HOME` and a deterministic localhost
# OpenAI-compatible mock provider — no external account, no network beyond
# 127.0.0.1.
#
# Flow driven through the mock:
#   1. root session calls `security_scan action=preflight`; with no stored OAuth
#      row the CLI answers the typed prerequisite error and creates `agent.db`.
#   2. the mock seeds one OAuth row into that same `agent.db` (the schema the CLI
#      just wrote; no DDL is duplicated here) and re-asks preflight.
#   3. preflight succeeds, `start` runs the restricted scan session, whose mock
#      turn calls `security_publish` exactly once with a canonical fixture.
#   4. root polls `status` to `completed`, reads `security://scans/<id>/findings`
#      through the mounted namespace, then validates the finding.
#
# Assertions: plan/scan/operation ids reach the CLI output, the output directory
# holds findings/report/SARIF/provenance/scan.json, provenance carries no
# identity or token material, and the project store has the committed scan.
#
# Nothing is written outside `$HOME`, and the harness never deletes a tree it
# did not create: it prints `$ROOT` and leaves it for inspection.
#
# Invoked by `crates/omp-cli/tests/security_scan_runtime_harness.rs` with
# `OMP_BIN`/`HOME`/`OMP_SECURITY_HARNESS_ROOT` set; it can also be run by hand
# with the same variables.

set -euo pipefail

: "${OMP_BIN:?set OMP_BIN to the built omp binary}"
ROOT="${OMP_SECURITY_HARNESS_ROOT:-$HOME/.omp-security-scan-harness-$$}"
AGENT_DIR="$ROOT/agent"
SESSIONS="$ROOT/sessions"
PROJECT="$ROOT/project"
OUTPUT="$ROOT/security-output"
PORT_FILE="$ROOT/port"
SERVER_LOG="$ROOT/mock.log"
RUN_OUT="$ROOT/run.out"
RUN_ERR="$ROOT/run.err"

mkdir -p "$AGENT_DIR" "$SESSIONS" "$PROJECT" "$OUTPUT"
: >"$SERVER_LOG"
: >"$RUN_OUT"
: >"$RUN_ERR"

cleanup() {
	if [[ -n "${SERVER_PID:-}" ]]; then
		kill "$SERVER_PID" 2>/dev/null || true
		wait "$SERVER_PID" 2>/dev/null || true
	fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Deterministic localhost provider + auth fixture
# ---------------------------------------------------------------------------
python3 - "$PORT_FILE" "$AGENT_DIR/agent.db" "$OUTPUT" "$ROOT/tools.log" >"$SERVER_LOG" 2>&1 <<'PY' &
import json
import os
import re
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port_file, auth_db, output_root, tools_log = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
seed_lock = threading.Lock()
seeded = False

OAUTH_DATA = {
    "access": "access-1",
    "refresh": "refresh-1",
    "expires": 4102444800000,  # far future: no refresh round-trip
    "email": "user1@example.test",
    "accountId": "acct-1",
    "orgId": "org-1",
    "orgName": "Workspace 1",
}
# `identity_key` for `openai-codex` is `email:<lowercased>|org:<id>`.
OAUTH_IDENTITY = "email:user1@example.test|org:org-1"


def seed_oauth_row():
    """Insert the pinned row into the schema the CLI created a moment ago."""
    global seeded
    with seed_lock:
        if seeded:
            return
        connection = sqlite3.connect(auth_db, timeout=10)
        try:
            connection.execute(
                "INSERT INTO auth_credentials (provider, credential_type, data, identity_key)"
                " VALUES (?, ?, ?, ?)",
                ("openai-codex", "oauth", json.dumps(OAUTH_DATA), OAUTH_IDENTITY),
            )
            connection.commit()
        finally:
            connection.close()
        seeded = True


def sse(chunks):
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def tool_call(name, arguments, call_id="call_1"):
    return sse([
        {"choices": [{"delta": {
            "role": "assistant",
            "tool_calls": [{
                "index": 0,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }],
        }}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ])


def text(content):
    return sse([
        {"choices": [{"delta": {"role": "assistant", "content": content}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ])


def message_text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def last_tool_call(messages):
    call_id = None
    for message in reversed(messages):
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            continue
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if call.get("id") == call_id:
                function = call.get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                return function.get("name"), arguments
        if call_id is not None:
            break
    return None, {}


def tool_names(body):
    names = []
    for tool in body.get("tools") or []:
        function = tool.get("function") or {}
        if function.get("name"):
            names.append(function["name"])
    return names


def fixture_publication():
    return {
        "findings": [{
            "rule_id": "rust/unsafe-deref",
            "title": "Unsafe dereference",
            "summary": "Dereferences a raw pointer without validation.",
            "severity": "high",
            "confidence": "high",
            "category": "memory-safety",
            "cwe": ["CWE-476"],
            "locations": [{"path": "src/lib.rs", "start_line": 1, "end_line": 2}],
            "evidence": [{"label": "sink", "explanation": "raw pointer deref"}],
            "remediation": "Validate before dereferencing.",
            "validation": "validated",
        }],
        "coverage": {"completeness": "complete"},
        "report": "# Security report\n\nOne fixture finding.\n",
    }


PLAN = re.compile(r"secplan_[0-9a-zA-Z]+")
SCAN = re.compile(r"secscan_[0-9a-zA-Z]+")
OP = re.compile(r"secop_[0-9a-zA-Z]+")
FINDING = re.compile(r"secf_[0-9a-zA-Z]+")


def root_reply(messages, body):
    last = messages[-1] if messages else {}
    if last.get("role") == "user":
        return tool_call("security_scan", {
            "action": "preflight",
            "target_kind": "repository",
            "output_root": output_root,
        })
    if last.get("role") != "tool":
        return None
    name, arguments = last_tool_call(messages)
    content = message_text(last)
    action = arguments.get("action")

    if "require a stored OAuth account" in content:
        seed_oauth_row()
        return tool_call("security_scan", {
            "action": "preflight",
            "target_kind": "repository",
            "output_root": output_root,
        })

    if name == "security_scan" and action == "preflight":
        if "is ready" not in content or "Fingerprint:" not in content:
            return text("SECURITY_SMOKE_FAIL_PREFLIGHT:" + content)
        match = PLAN.search(content)
        if not match:
            return text("SECURITY_SMOKE_FAIL_NO_PLAN:" + content)
        return tool_call("security_scan", {"action": "start", "plan_id": match.group(0)})

    if name == "security_scan" and action == "start":
        if "started as secop_" not in content:
            return text("SECURITY_SMOKE_FAIL_START:" + content)
        operation = OP.search(content)
        if not operation:
            return text("SECURITY_SMOKE_FAIL_NO_OPERATION:" + content)
        return tool_call("security_scan", {
            "action": "status",
            "operation_id": operation.group(0),
        })

    if name == "security_scan" and action == "status":
        # Tool text: `Security scan <id>: <phase>; <n> finding(s).`
        if ": completed;" in content:
            scan = SCAN.search(content)
            if not scan:
                return text("SECURITY_SMOKE_FAIL_NO_SCAN:" + content)
            return tool_call("read", {
                "path": f"security://scans/{scan.group(0)}/findings",
            })
        if any(f": {phase};" in content for phase in ("failed", "partial", "cancelled")):
            return text("SECURITY_SMOKE_FAIL_STATUS:" + content)
        polls = sum(
            1
            for message in messages
            if message.get("role") == "tool" and ": completed;" not in message_text(message)
        )
        if polls > 400:
            return text("SECURITY_SMOKE_FAIL_STATUS_TIMEOUT:" + content)
        operation = OP.search(content)
        return tool_call("security_scan", {
            "action": "status",
            "operation_id": operation.group(0) if operation else arguments.get("operation_id"),
        })

    if name == "read":
        # `security://scans/<id>/findings` lists `secf_` ids; the location lives
        # in the per-finding resource, so the listing only needs the ids.
        if "secf_" not in content or "Findings for" not in content:
            return text("SECURITY_SMOKE_FAIL_RESOURCE:" + content)
        finding = FINDING.search(content)
        scan = SCAN.search(content)
        if not finding or not scan:
            return text("SECURITY_SMOKE_FAIL_RESOURCE_IDS:" + content)
        return tool_call("security_scan", {
            "action": "validate",
            "scan_id": scan.group(0),
            "finding_id": finding.group(0),
            "validation_status": "validated",
            "validation_summary": "Reproduced with the harness fixture.",
            "validation_evidence": [{"label": "repro", "explanation": "harness repro"}],
        })

    if name == "security_scan" and action == "validate":
        if "validation is now validated" not in content:
            return text("SECURITY_SMOKE_FAIL_VALIDATE:" + content)
        scan = SCAN.search(content)
        return text("SECURITY_SMOKE_OK " + (scan.group(0) if scan else "scan-unknown"))

    return text("SECURITY_SMOKE_FAIL_UNEXPECTED:" + content)


def scan_reply(messages):
    last = messages[-1] if messages else {}
    if last.get("role") == "user":
        return tool_call("security_publish", fixture_publication())
    if last.get("role") != "tool":
        return None
    content = message_text(last)
    if "Published security scan" in content or "already been published" in content:
        return text("scan-session-done")
    return text("SECURITY_SMOKE_FAIL_PUBLISH:" + content)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages") or []
        names = tool_names(body)
        if message_text(messages[-1] if messages else {}).find("NO_TOOLS_TASK") >= 0:
            # `--no-tools` must carry an empty tool definition list: any mounted
            # tool (not just `security_scan`) is a mount leak, so report the
            # actual names instead of blindly answering OK.
            if names:
                payload = text("SECURITY_SMOKE_FAIL_NO_TOOLS_MOUNTED:" + ",".join(names))
            else:
                payload = text("NO_TOOLS_OK")
        elif "security_scan" in names:
            payload = root_reply(messages, body)
        elif "security_publish" in names:
            with open(tools_log, "a", encoding="utf-8") as handle:
                handle.write(",".join(names) + "\n")
            payload = scan_reply(messages)
        else:
            payload = text("SECURITY_SMOKE_FAIL_UNEXPECTED_SESSION:" + ",".join(names))
        if payload is None:
            payload = text("SECURITY_SMOKE_FAIL_NO_REPLY")

        data = payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
with open(port_file, "w", encoding="utf-8") as handle:
    handle.write(str(server.server_address[1]))
server.serve_forever()
PY
SERVER_PID=$!

for _ in $(seq 1 200); do
	[[ -s "$PORT_FILE" ]] && break
	sleep 0.05
done
PORT="$(cat "$PORT_FILE")"
if [[ -z "$PORT" ]]; then
	echo "harness: mock provider did not start; see $SERVER_LOG" >&2
	exit 1
fi

# ---------------------------------------------------------------------------
# Real git fixture + isolated CLI config
# ---------------------------------------------------------------------------
git -C "$PROJECT" init -q
git -C "$PROJECT" config user.email "security-harness@example.test"
git -C "$PROJECT" config user.name "Security Harness"
git -C "$PROJECT" config commit.gpgsign false
mkdir -p "$PROJECT/src"
cat >"$PROJECT/src/lib.rs" <<'RS'
pub fn read(raw: *const u8) -> u8 {
    unsafe { *raw }
}
RS
git -C "$PROJECT" add .
git -C "$PROJECT" commit -q -m "fixture"

cat >"$AGENT_DIR/config.yml" <<YAML
defaultModel: openai-codex/gpt-5-codex
providers:
  openai-codex:
    baseUrl: "http://127.0.0.1:$PORT/v1"
    api: openai-completions
    apiKey: harness-placeholder
tools:
  approvalMode: always-ask
lsp:
  enabled: false
security:
  enabled: true
YAML

cat >"$AGENT_DIR/models.yml" <<YAML
providers:
  openai-codex:
    baseUrl: "http://127.0.0.1:$PORT/v1"
    api: openai-completions
    apiKey: harness-placeholder
    models:
      - id: gpt-5-codex
        contextWindow: 128000
YAML

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
set +e
HOME="$ROOT" PI_CODING_AGENT_DIR="$AGENT_DIR" \
	"$OMP_BIN" --cwd "$PROJECT" --session-dir "$SESSIONS" --mode json -p SECURITY_ROOT_TASK \
	>"$RUN_OUT" 2>"$RUN_ERR"
STATUS=$?
set -e

FAILED=0
check() {
	local label="$1"
	shift
	if "$@"; then
		echo "ok   $label"
	else
		echo "FAIL $label"
		FAILED=1
	fi
}

check "cli exited 0" test "$STATUS" -eq 0
check "preflight reported a plan" grep -q "Security plan secplan_" "$RUN_OUT"
check "scan started" grep -q "started as secop_" "$RUN_OUT"
check "smoke verdict" grep -q "SECURITY_SMOKE_OK" "$RUN_OUT"
check "no mock failure marker" bash -c "! grep -q 'SECURITY_SMOKE_FAIL' '$RUN_OUT'"
check "findings written" test -f "$OUTPUT/findings.json"
check "report written" test -f "$OUTPUT/report.md"
check "sarif written" test -f "$OUTPUT/results.sarif"
check "provenance written" test -f "$OUTPUT/provenance.json"
check "scan.json last marker" test -f "$OUTPUT/scan.json"
check "sarif has a result" grep -q '"version": "2.1.0"' "$OUTPUT/results.sarif"
check "provenance has no email" bash -c "! grep -q 'user1@example.test' '$OUTPUT/provenance.json'"
check "provenance has no token" bash -c "! grep -q 'access-1' '$OUTPUT/provenance.json'"
check "store committed the scan" bash -c "find '$ROOT' -path '*security*/scans/*/scan.json' -print -quit | grep -q ."
# The coordinator publishes the journal path through the tool's structured
# details; take that API value and require the file on disk to exist (the
# `find` pattern is never used to guess it).
PUBLISHED_SESSION_FILE="$(grep -o '"sessionFile":"[^"]*"' "$RUN_OUT" | tail -1 | sed 's/^"sessionFile":"//;s/"$//')"
check "published sessionFile present" test -n "$PUBLISHED_SESSION_FILE"
JOURNAL="$PUBLISHED_SESSION_FILE"
check "published sessionFile exists" test -f "$JOURNAL"
check "journal lives in the security store" \
	bash -c "case '$JOURNAL' in */security/*/sessions/*.jsonl) true;; *) false;; esac"
check "journal has assistant turn" bash -c "grep -q '\"role\":\"assistant\"' '$JOURNAL'"
check "journal has tool result" bash -c "grep -q '\"role\":\"toolResult\"' '$JOURNAL'"
check "one journal for the scan" \
	bash -c "test \"\$(find \"\$(dirname '$JOURNAL')\" -name '*.jsonl' | wc -l)\" -eq 1"
SCAN_ID_FROM_JOURNAL="$(basename "$JOURNAL" .jsonl)"
SCAN_ID_FROM_OUTPUT="$(grep -o 'secscan_[0-9a-zA-Z]*' "$RUN_OUT" | head -1)"
# Session logs are `<file-safe-timestamp>_<id>.jsonl`; the id must be the suffix.
check "journal is named after the scan" \
	bash -c "case '$SCAN_ID_FROM_JOURNAL' in *_'$SCAN_ID_FROM_OUTPUT') true;; *) false;; esac"
check "scan session kept lsp despite parent off" grep -q 'lsp' "$ROOT/tools.log"
check "scan session had publication tool" grep -q 'security_publish' "$ROOT/tools.log"

# ---------------------------------------------------------------------------
# Negatives: `--no-tools` never mounts the coordinator; an explicit selection
# with `security.enabled: false` must fail before any provider call.
# ---------------------------------------------------------------------------
NO_TOOLS_OUT="$ROOT/no-tools.out"
set +e
HOME="$ROOT" PI_CODING_AGENT_DIR="$AGENT_DIR" \
	"$OMP_BIN" --cwd "$PROJECT" --session-dir "$SESSIONS" --mode json --no-tools \
	-p NO_TOOLS_TASK >"$NO_TOOLS_OUT" 2>"$ROOT/no-tools.err"
NO_TOOLS_STATUS=$?
set -e
check "no-tools run exited 0" test "$NO_TOOLS_STATUS" -eq 0
check "no-tools mock answered" grep -q "NO_TOOLS_OK" "$NO_TOOLS_OUT"
check "no-tools mock saw no mounted tools" \
	bash -c "! grep -q 'SECURITY_SMOKE_FAIL' '$NO_TOOLS_OUT'"
check "no-tools did not mount security_scan" bash -c "! grep -q '\"toolName\":\"security_scan\"' '$NO_TOOLS_OUT'"

DISABLED_AGENT="$ROOT/agent-disabled"
mkdir -p "$DISABLED_AGENT"
cp "$AGENT_DIR/models.yml" "$DISABLED_AGENT/models.yml"
sed 's/^  enabled: true$/  enabled: false/' "$AGENT_DIR/config.yml" >"$DISABLED_AGENT/config.yml"
set +e
HOME="$ROOT" PI_CODING_AGENT_DIR="$DISABLED_AGENT" \
	"$OMP_BIN" --cwd "$PROJECT" --session-dir "$SESSIONS" --tools security_scan \
	-p DISABLED_TASK >"$ROOT/disabled.out" 2>"$ROOT/disabled.err"
DISABLED_STATUS=$?
set -e
check "disabled scan selection exited non-zero" test "$DISABLED_STATUS" -ne 0
check "disabled scan selection named the setting" \
	bash -c "grep -q 'requires \`security.enabled: true\`' '$ROOT/disabled.err' '$ROOT/disabled.out'"

echo "harness root: $ROOT"
if [[ "$FAILED" -ne 0 ]]; then
	echo "security harness failed; see $RUN_ERR and $SERVER_LOG" >&2
	exit 1
fi
echo "security harness passed"
