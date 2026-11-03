#!/usr/bin/env bash
# C7 local-learning HOME smoke test.
#
# Runs the real binary against a mock provider with an isolated HOME:
#   1. `learn` writes learned.md,
#   2. `manage_skill` writes a managed skill,
#   3. `/skill:<name>` injects the user-invocation template,
#   4. `read skill://<name>` resolves through the session snapshot,
#   5. the system prompt carries the memory and auto-learn suffix sections.
#
# Usage: OMP_BIN=/path/to/omp scripts/c7-home-smoke.sh
set -euo pipefail

OMP_BIN="${OMP_BIN:-target/debug/omp}"
ROOT="$(mktemp -d "${TMPDIR:-/tmp}/omp-c7-home.XXXXXX")"
cleanup() { rm -rf "$ROOT"; }
trap cleanup EXIT

AGENT_DIR="$ROOT/agent"
PROJECT="$ROOT/project"
SESSIONS="$ROOT/sessions"
PORT_FILE="$ROOT/port"
REQUESTS_LOG="$ROOT/requests.jsonl"
mkdir -p "$AGENT_DIR" "$PROJECT" "$SESSIONS"

python3 - "$PORT_FILE" "$REQUESTS_LOG" <<'PY' &
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port_file, requests_log = sys.argv[1], sys.argv[2]


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


def tool_call(call_id, name, arguments):
    payload = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }
    return ("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n").encode()


def text(value):
    payload = {"choices": [{"delta": {"content": value}}]}
    return ("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n").encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        with open(requests_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(body) + "\n")
        messages = body.get("messages") or []
        last = messages[-1] if messages else {}
        if last.get("role") != "user":
            data = text("DONE")
        else:
            user = message_text(last)
            if "CAPTURE" in user:
                data = tool_call(
                    "call-learn",
                    "learn",
                    {
                        "memory": "Always run the formatter before committing",
                        "context": "smoke",
                    },
                )
            elif "CREATE THE SKILL" in user:
                data = tool_call(
                    "call-manage",
                    "manage_skill",
                    {
                        "action": "create",
                        "name": "captured",
                        "description": "Captured procedure",
                        "body": "# Captured\nrun it",
                    },
                )
            elif "READ THE SKILL" in user:
                data = tool_call("call-read", "read", {"path": "skill://captured"})
            else:
                data = text("DONE")
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
trap 'kill "$SERVER_PID" 2>/dev/null || true; cleanup' EXIT

for _ in $(seq 1 100); do
	[[ -s "$PORT_FILE" ]] && break
	sleep 0.05
done
PORT="$(cat "$PORT_FILE")"
[[ -n "$PORT" ]] || { echo "mock provider did not start" >&2; exit 1; }

cat >"$AGENT_DIR/config.yml" <<YAML
defaultModel: mock/mock-model
memory:
  backend: local
autolearn:
  enabled: true
providers:
  mock:
    baseUrl: "http://127.0.0.1:$PORT/v1"
    api: openai-completions
    apiKey: test-key
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

HOME="$ROOT" PI_CODING_AGENT_DIR="$AGENT_DIR" OMP_SHARE_DIR="$ROOT/share" \
	"$OMP_BIN" --cwd "$PROJECT" --session-dir "$SESSIONS" --no-session \
	--tools read,learn,manage_skill --yolo --max-time 30s --mode text -p \
	"CAPTURE the lesson" "CREATE THE SKILL now" "USE /skill:captured now" "READ THE SKILL" \
	>"$ROOT/run.out" 2>"$ROOT/run.err"

LEARNED="$(find "$AGENT_DIR/memories" -name learned.md -type f | head -n 1)"
[[ -n "$LEARNED" ]] || { echo "learn did not write learned.md" >&2; exit 1; }
grep -q "Always run the formatter before committing" "$LEARNED" ||
	{ echo "learned.md lost the lesson" >&2; exit 1; }

SKILL="$AGENT_DIR/managed-skills/captured/SKILL.md"
[[ -f "$SKILL" ]] || { echo "manage_skill did not write the skill" >&2; exit 1; }
grep -q "# Captured" "$SKILL" || { echo "managed skill body missing" >&2; exit 1; }

python3 - "$REQUESTS_LOG" <<'PY'
import json
import sys

requests = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
assert len(requests) == 7, f"expected 7 provider calls, got {len(requests)}"


def text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "")
        for part in content or []
        if isinstance(part, dict) and part.get("type") == "text"
    )


system = text(requests[0]["messages"][0])
assert "## Auto-Learn (experimental)" in system, system
assert "manage_skill" in system and "learn" in system, system

later = text(requests[2]["messages"][0])
assert "# Memory Guidance" in later, later
assert "Always run the formatter before committing" in later, later

invocation = text(requests[4]["messages"][-1])
assert 'User invoked the "captured" skill' in invocation, invocation
assert "# Captured" in invocation, invocation

tool_result = requests[6]["messages"][-1]
assert tool_result.get("name") == "read", tool_result
assert "# Captured" in text(tool_result), tool_result
PY

echo "c7 home smoke: ok"
