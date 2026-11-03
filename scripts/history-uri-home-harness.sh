#!/usr/bin/env bash
# Black-box CLI harness for `history://` (live, parked, resumed, isolated).
#
# Requires a built `omp` binary. Runs the real CLI against a localhost
# OpenAI-compatible mock under an isolated $HOME:
#   1. root spawns an ASYNC child. The child's first provider call answers
#      "live-child-answer" plus a harmless real `bash true` tool call, so the
#      second request is reachable; that second request blocks on the root's
#      history read. The root must see the in-turn answer in `history://task-1`
#      BEFORE releasing the child, and a bounded probe of the exact fixture
#      transcript must prove the marker is not journaled yet. After release the
#      child settles, is adopted, and history falls back to the journaled
#      transcript.
#   2. a fresh CLI process resumes the session, reads the parked row, cold-
#      revives the same id with `hub send`, and reads its history again.
#   3. a second, independent session spawns its own `task-1` and must read only
#      its own transcript (same id, different session artifacts).
#
# Nothing is written outside $HOME. Hand-run recipe, not a cargo test.

set -euo pipefail

: "${OMP_BIN:?set OMP_BIN to the built omp binary}"
ROOT="${OMP_HARNESS_ROOT:-$HOME/.omp-history-uri-harness-$$}"
AGENT_DIR="$ROOT/agent"
SESSIONS="$ROOT/sessions"
PROJECT="$ROOT/project"
PORT_FILE="$ROOT/port"
SERVER_LOG="$ROOT/mock.log"

mkdir -p "$AGENT_DIR" "$SESSIONS" "$PROJECT"
: >"$SERVER_LOG"

cleanup() {
	if [[ -n "${SERVER_PID:-}" ]]; then
		kill "$SERVER_PID" 2>/dev/null || true
		wait "$SERVER_PID" 2>/dev/null || true
	fi
}
trap cleanup EXIT

python3 - "$PORT_FILE" "$SESSIONS" >"$SERVER_LOG" 2>&1 <<'PY' &
import json
import shlex
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port_file = sys.argv[1]
sessions = sys.argv[2]
release_child = threading.Event()

# Exact fixture transcript under the harness-owned sessions root. No HOME scan.
MARKER_PROBE = f'''set -- {shlex.quote(sessions)}/*/subagents/task-1.jsonl
if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then echo MARKER_PROBE_BAD_COUNT; exit 0; fi
if grep -qs 'live-child-answer' "$1"; then echo MARKER_JOURNALED; else echo MARKER_ABSENT; fi'''


def sse(chunks):
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def tool_call(name, arguments, call_id="call_1", content=None):
    delta = {
        "role": "assistant",
        "tool_calls": [{
            "index": 0,
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }
    if content is not None:
        delta["content"] = content
    return sse([
        {"choices": [{"delta": delta}]},
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


def last_user(messages):
    for message in reversed(messages):
        if message.get("role") == "user":
            return message_text(message)
    return ""


def last_tool_call(messages):
    """(name, arguments) answered by the last tool result; None when not a tool turn."""
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


def count_tool_calls(messages, name, op=None, path=None):
    count = 0
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != name:
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            if op is not None and arguments.get("op") != op:
                continue
            if path is not None and arguments.get("path") != path:
                continue
            count += 1
    return count


def root_reply(messages):
    last = messages[-1] if messages else {}
    if last.get("role") == "user":
        return tool_call("task", {"name": "Live", "task": "CHILD_WORK"})
    if last.get("role") != "tool":
        return None
    name, arguments = last_tool_call(messages)
    content = message_text(last)
    waits = count_tool_calls(messages, "hub", op="wait")
    reads = count_tool_calls(messages, "read", path="history://task-1")

    if name == "task":
        if '"async"' not in content or '"running"' not in content:
            return text("PHASE1_FAIL_TASK_NOT_ASYNC:" + content)
        return tool_call("read", {"path": "history://"})
    if name == "read" and "# Agents" in content:
        return tool_call("read", {"path": "history://task-1"})
    if name == "read" and waits == 0:
        if "live-child-answer" in content:
            # Prove this turn is live and not yet in the journal before release.
            return tool_call("bash", {"command": MARKER_PROBE})
        if reads < 8:
            return tool_call("read", {"path": "history://task-1"})
        return text("PHASE1_FAIL_NO_LIVE")
    if name == "bash":
        if "MARKER_ABSENT" in content:
            release_child.set()
            return tool_call("hub", {"op": "wait", "ids": ["task-1"], "timeoutMs": 10000})
        return text("PHASE1_FAIL_MARKER_PROBE:" + content)
    if name == "hub" and arguments.get("op") == "wait":
        return tool_call("read", {"path": "history://task-1"})
    if name == "read" and waits >= 1:
        if "live-child-answer" in content:
            return text("PHASE1_OK:" + content)
        return text("PHASE1_FAIL_AFTER_SETTLE")
    return text("DONE")


def resume_reply(messages):
    last = messages[-1] if messages else {}
    if last.get("role") == "user":
        return tool_call("read", {"path": "history://task-1"})
    if last.get("role") != "tool":
        return None
    name, arguments = last_tool_call(messages)
    content = message_text(last)

    if name == "read" and "resume-body-C" not in content:
        return tool_call("hub", {"op": "send", "to": "task-1", "message": "REVIVE_RESUME"})
    if name == "hub" and arguments.get("op") == "send":
        return tool_call("hub", {"op": "wait", "ids": ["task-1"], "timeoutMs": 10000})
    if name == "hub" and arguments.get("op") == "wait":
        return tool_call("read", {"path": "history://task-1"})
    if name == "read" and "resume-body-C" in content:
        return text("PHASE2_OK:" + content)
    return text("DONE")


def isolated_reply(messages):
    last = messages[-1] if messages else {}
    if last.get("role") == "user":
        return tool_call("task", {"name": "Iso", "task": "CHILD_WORK_ISOLATED"})
    if last.get("role") != "tool":
        return None
    name, arguments = last_tool_call(messages)
    content = message_text(last)

    if name == "task":
        return tool_call("hub", {"op": "wait", "ids": ["task-1"], "timeoutMs": 10000})
    if name == "hub" and arguments.get("op") == "wait":
        return tool_call("read", {"path": "history://task-1"})
    if name == "read":
        if "iso-body-D" in content:
            return text("PHASE3_OK:" + content)
        return text("PHASE3_FAIL")
    return text("DONE")


def child_live_reply(messages):
    """First call answers plus a real tool turn; second waits for the root's live read."""
    assistant_turns = sum(1 for message in messages if message.get("role") == "assistant")
    if assistant_turns == 0:
        return tool_call("bash", {"command": "true"}, content="live-child-answer")
    release_child.wait(timeout=20)
    return tool_call("yield", {"result": {"data": "child-body-A"}})


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages") or []
        user = last_user(messages)
        payload = None

        if "ROOT_TASK_ISOLATED" in user:
            payload = isolated_reply(messages)
        elif "READ_HISTORY_AFTER_TASK" in user:
            payload = resume_reply(messages)
        elif "ROOT_TASK" in user:
            payload = root_reply(messages)
        elif "CHILD_WORK_ISOLATED" in user:
            payload = text("iso-body-D")
        elif "REVIVE_RESUME" in user:
            payload = text("resume-body-C")
        elif "REVIVE_WORK" in user:
            payload = text("revived-body-B")
        elif "CHILD_WORK" in user:
            payload = child_live_reply(messages)

        if payload is None:
            payload = text("DONE")

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

for _ in $(seq 1 100); do
	[[ -s "$PORT_FILE" ]] && break
	sleep 0.05
done
PORT="$(cat "$PORT_FILE")"

cat >"$AGENT_DIR/config.yml" <<YAML
defaultModel: mock/mock-model
providers:
  mock:
    baseUrl: "http://127.0.0.1:$PORT/v1"
    api: openai-completions
    apiKey: test-key
tools:
  approvalMode: yolo
YAML

cat >"$AGENT_DIR/models.yml" <<YAML
providers:
  mock:
    baseUrl: "http://127.0.0.1:$PORT/v1"
    api: openai-completions
    apiKey: test-key
    models:
      - id: mock-model
        contextWindow: 128000
YAML

run_cli() {
	HOME="$ROOT" PI_CODING_AGENT_DIR="$AGENT_DIR" \
		"$OMP_BIN" --cwd "$PROJECT" --session-dir "$SESSIONS" --yolo "$@"
}

# Phase 1: async spawn -> gated in-turn answer visible live -> settle/adopt.
run_cli -p ROOT_TASK >"$ROOT/first.out" 2>"$ROOT/first.err"
grep -q "PHASE1_OK" "$ROOT/first.out" || {
	echo "phase 1 live history read failed" >&2
	cat "$ROOT/first.out" "$ROOT/first.err" >&2
	exit 1
}
grep -q "live-child-answer" "$ROOT/first.out" || {
	echo "phase 1 did not observe the in-turn live answer" >&2
	exit 1
}

SESSION_FILE="$(find "$SESSIONS" -name '*.jsonl' -type f -not -path '*/subagents/*' | sort | tail -n 1)"
[[ -n "$SESSION_FILE" ]] || { echo "no session file" >&2; exit 1; }
ARTIFACTS="${SESSION_FILE%.jsonl}"
[[ -f "$ARTIFACTS/subagents/task-1.jsonl" ]] || { echo "child transcript missing" >&2; exit 1; }
[[ -f "$ARTIFACTS/subagents/task-1.init.json" ]] || { echo "child contract missing" >&2; exit 1; }
grep -q "live-child-answer" "$ARTIFACTS/subagents/task-1.jsonl" || {
	echo "in-turn answer was not journaled after settle" >&2
	exit 1
}

# Phase 2: fresh process -> parked read -> cold revive -> history again.
SESSION_ID="$(basename "$SESSION_FILE" .jsonl)"
SESSION_ID="${SESSION_ID##*_}"
run_cli --resume "$SESSION_ID" -p READ_HISTORY_AFTER_TASK >"$ROOT/resume.out" 2>"$ROOT/resume.err"
grep -q "PHASE2_OK" "$ROOT/resume.out" || {
	echo "phase 2 resumed history/revive failed" >&2
	cat "$ROOT/resume.out" "$ROOT/resume.err" >&2
	exit 1
}
grep -q "live-child-answer" "$ROOT/resume.out" || { echo "resume lost live answer" >&2; exit 1; }
grep -q "resume-body-C" "$ROOT/resume.out" || { echo "resume missing cold-revive body" >&2; exit 1; }

# Phase 3: different session, same task-1 id, isolated artifacts.
run_cli -p ROOT_TASK_ISOLATED >"$ROOT/isolated.out" 2>"$ROOT/isolated.err"
grep -q "PHASE3_OK" "$ROOT/isolated.out" || {
	echo "phase 3 isolated history failed" >&2
	cat "$ROOT/isolated.out" "$ROOT/isolated.err" >&2
	exit 1
}
grep -q "iso-body-D" "$ROOT/isolated.out" || { echo "isolated session lost its body" >&2; exit 1; }
! grep -q "live-child-answer" "$ROOT/isolated.out" || { echo "isolated session read phase 1 transcript" >&2; exit 1; }
! grep -q "resume-body-C" "$ROOT/isolated.out" || { echo "isolated session read phase 2 transcript" >&2; exit 1; }

echo "HARNESS_OK root=$ROOT session=$SESSION_FILE"
