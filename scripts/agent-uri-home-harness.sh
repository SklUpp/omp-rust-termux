#!/usr/bin/env bash
# Black-box CLI harness for saved agent outputs.
#
# Requires a built `omp` binary. Runs the real CLI twice against a localhost
# OpenAI-compatible mock under an isolated $HOME:
#   1. root asks for a child task; child yields; root finishes.
#   2. a fresh CLI process resolves the saved output with `--resume` and a real
#      `read agent://CliChild` tool call.
#
# Verifies the producer file is in the session artifacts root (not
# `<session>/subagents`) and that the second process can read it after restart.
# Nothing is written outside $HOME. The harness is intentionally not a cargo
# test so it can be run by hand against a release binary.

set -euo pipefail

: "${OMP_BIN:?set OMP_BIN to the built omp binary}"
ROOT="${OMP_HARNESS_ROOT:-$HOME/.omp-agent-uri-harness-$$}"
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

python3 - "$PORT_FILE" >"$SERVER_LOG" 2>&1 <<'PY' &
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port_file = sys.argv[1]


def tool_call(name, arguments):
    call = {
        "choices": [{
            "delta": {
                "role": "assistant",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }],
            },
        }]
    }
    finish = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
    return f"data: {json.dumps(call)}\n\ndata: {json.dumps(finish)}\n\ndata: [DONE]\n\n"


def text(content):
    first = {"choices": [{"delta": {"role": "assistant", "content": content}}]}
    finish = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    return f"data: {json.dumps(first)}\n\ndata: {json.dumps(finish)}\n\ndata: [DONE]\n\n"


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


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages") or []
        last = messages[-1] if messages else {}
        role = last.get("role")
        if role == "tool":
            # The wire carries no tool name on tool results; both roots' tool
            # results are echoed. The second process only passes when the read
            # result contains the saved child payload.
            payload = text("READ_RESULT:" + message_text(last))
        else:
            user = message_text(last)
            if "CHILD_WORK" in user:
                payload = tool_call("yield", {"result": {"data": "child-ok"}})
            elif "READ_AFTER_TASK" in user:
                payload = tool_call("read", {"path": "agent://CliChild"})
            elif "ROOT_TASK" in user:
                payload = tool_call("task", {"name": "CliChild", "task": "CHILD_WORK"})
            else:
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

run_cli -p ROOT_TASK >"$ROOT/first.out" 2>"$ROOT/first.err"
SESSION_FILE="$(find "$SESSIONS" -name '*.jsonl' -type f -not -path '*/subagents/*' | sort | tail -n 1)"
[[ -n "$SESSION_FILE" ]] || { echo "no session file" >&2; exit 1; }
ARTIFACTS="${SESSION_FILE%.jsonl}"
CHILD_OUTPUT="$ARTIFACTS/CliChild.md"
[[ -f "$CHILD_OUTPUT" ]] || { echo "missing $CHILD_OUTPUT" >&2; exit 1; }
grep -q "child-ok" "$CHILD_OUTPUT" || { echo "child output missing payload" >&2; exit 1; }
[[ ! -e "$ARTIFACTS/subagents/CliChild.md" ]] || { echo "output leaked into subagents" >&2; exit 1; }
[[ -f "$ARTIFACTS/subagents/task-1.jsonl" ]] || { echo "child transcript missing" >&2; exit 1; }

SESSION_ID="$(basename "$SESSION_FILE" .jsonl)"
SESSION_ID="${SESSION_ID##*_}"

run_cli --resume "$SESSION_ID" -p READ_AFTER_TASK >"$ROOT/resume.out" 2>"$ROOT/resume.err"
grep -q "READ_RESULT:.*child-ok" "$ROOT/resume.out" ||
	{ echo "resumed agent:// read did not return the saved output" >&2; exit 1; }

echo "HARNESS_OK root=$ROOT output=$CHILD_OUTPUT"
