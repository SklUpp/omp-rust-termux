#!/usr/bin/env python3
"""HOME-confined advisor pipeline smoke test (CLI text, CLI JSON, RPC, ACP).

Invocation:
    OMP_BIN=target/debug/omp python3 scripts/advisor-smoke-home.py
    python3 scripts/advisor-smoke-home.py /path/to/omp

Everything is created under ``$HOME/task-tmp/advisor-smoke-<pid>`` (fallback:
the platform temp dir). Nothing outside that directory is touched; sessions,
transcripts, the mock server log, and captured stdout/stderr stay there for
inspection. No process is killed with a forced signal unless a phase times out;
a timeout fails the step and the child is terminated.

Phases (all assert observable behavior, never source shape):
 1. CLI text:        `--advisor` succeeds, the blocker card is visible.
 2. CLI JSON:        card pair on the wire, one persisted `custom_message`,
                     one `__advisor.reviewer.jsonl`, no hidden primary turn.
 3. CLI error:       advisor 429 is contained: exit 0, no fabricated card.
 4. RPC:             ready -> prompt -> turn_end -> get_advisor_status,
                     advisor frames on the wire, EOF dispose exits 0.
 5. ACP:             initialize -> session/new -> session/prompt, advisor
                     update visible, session/cancel returns, EOF exits 0.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TIMEOUT = 90


class MockModel:
    """Scripted chat-completions endpoint: primary, advisor advise, advisor final."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.advisor_notes_sent = 0
        self.advisor_error = False
        self.primary_delay = 0.0
        self.lock = threading.Lock()
        mock = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):  # keep the harness quiet
                pass

            def do_POST(self):  # noqa: N802 - http.server API
                length = int(self.headers.get("content-length", "0"))
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = {}
                with mock.lock:
                    mock.requests.append(payload)
                    system = json.dumps(payload.get("messages", [{}])[:1])
                    advisor = "passive advisor" in system
                    if not advisor and mock.primary_delay:
                        delay = mock.primary_delay
                    else:
                        delay = 0.0
                    if advisor and mock.advisor_error:
                        chunk = (
                            'data: {"error":{"type":"insufficient_quota",'
                            '"message":"429 rate limit exceeded"}}\n\ndata: [DONE]\n\n'
                        )
                    elif advisor and mock.advisor_notes_sent == 0:
                        mock.advisor_notes_sent += 1
                        args = json.dumps(
                            {"note": "write target deleted", "severity": "blocker"}
                        )
                        chunk = (
                            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                            '"id":"call-adv","type":"function","function":{"name":"advise",'
                            f'"arguments":{json.dumps(args)}}}}}]}}]}}\n\n'
                            "data: [DONE]\n\n"
                        )
                    else:
                        text = "noted" if advisor else "done"
                        chunk = (
                            'data: {"choices":[{"delta":{"content":'
                            f'"{text}"'
                            '},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
                        )
                if delay:
                    time.sleep(delay)
                encoded = chunk.encode()
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(encoded)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def count(self) -> int:
        with self.lock:
            return len(self.requests)


def base_dir() -> Path:
    home = os.environ.get("HOME")
    root = Path(home) / "task-tmp" if home else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="advisor-smoke-", dir=root))


def write_config(base: Path, port: int) -> tuple[Path, Path, Path, Path]:
    agent_dir = base / "agent"
    cwd = base / "work"
    sessions = base / "sessions"
    for directory in (agent_dir, cwd / ".omp", sessions):
        directory.mkdir(parents=True, exist_ok=True)
    (agent_dir / "config.yml").write_text(
        "defaultModel: mock/primary\n"
        "modelRoles:\n"
        "  advisor: mock/advisor\n"
        "  slow: mock/slow\n"
        f"providers:\n  mock:\n    baseUrl: \"http://127.0.0.1:{port}/v1\"\n"
        "    api: openai-completions\n    apiKey: test-key\n"
    )
    (agent_dir / "models.yml").write_text(
        f"providers:\n  mock:\n    baseUrl: \"http://127.0.0.1:{port}/v1\"\n"
        "    api: openai-completions\n    apiKey: test-key\n    models:\n"
        "      - id: primary\n        contextWindow: 128000\n"
        "      - id: advisor\n        contextWindow: 128000\n"
        "      - id: slow\n        contextWindow: 128000\n"
    )
    (cwd / ".omp" / "WATCHDOG.yml").write_text(
        "instructions: SMOKE FOCUS MARKER\n"
        "advisors:\n  - name: reviewer\n    model: mock/advisor\n    tools:\n      - read\n"
    )
    return agent_dir, cwd, sessions, base


def omp_env(agent_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    env["HOME"] = str(agent_dir.parent)
    return env


def cli_run(binary: str, agent_dir: Path, cwd: Path, sessions: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            binary,
            "--cwd", str(cwd),
            "--session-dir", str(sessions),
            "--no-tools",
            "--no-title",
            "--advisor",
            "--mode", "text",
            "say hi",
        ],
        env=omp_env(agent_dir),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


def session_logs(sessions: Path) -> list[Path]:
    return sorted(path for path in sessions.rglob("*.jsonl") if "__advisor" not in path.name)


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def read_lines(proc: subprocess.Popen[str], lines: queue.Queue, timeout: float, until) -> list[dict]:
    """Collect decoded frames until `until(frame)` is true (or timeout)."""
    seen: list[dict] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            raw = lines.get(timeout=max(0.05, deadline - time.time()))
        except queue.Empty:
            continue
        if raw is None:
            break
        raw = raw.strip()
        if not raw:
            continue
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            continue
        seen.append(frame)
        if until(frame):
            break
    return seen


def spawn_with_reader(binary: str, args: list[str], env: dict[str, str]) -> tuple[
    subprocess.Popen[str], queue.Queue
]:
    proc = subprocess.Popen(
        args,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    return proc, lines


def send(proc: subprocess.Popen[str], frame: dict) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(frame) + "\n")
    proc.stdin.flush()


def close_stdin(proc: subprocess.Popen[str], timeout: float = TIMEOUT) -> int:
    assert proc.stdin is not None
    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=10)
        fail("process did not exit after EOF (dispose path hung)")
    return proc.returncode


def phase_cli_text(binary: str, mock: MockModel, base: Path) -> Path:
    agent_dir, cwd, sessions, _ = write_config(base, mock.port)
    out = cli_run(binary, agent_dir, cwd, sessions)
    if out.returncode != 0:
        fail(f"CLI text exited {out.returncode}: {out.stderr}")
    if "advisor:" not in out.stderr or "write target deleted" not in out.stderr:
        fail(f"CLI text must show the card on stderr: {out.stderr!r}")
    logs = session_logs(sessions)
    if len(logs) != 1:
        fail(f"expected one session log, got {logs}")
    log = logs[0].read_text()
    if '"customType":"advisor"' not in log or "write target deleted" not in log:
        fail("CLI text must persist the card as a custom_message")
    transcript = logs[0].with_suffix("") / "__advisor.reviewer.jsonl"
    if not transcript.is_file():
        fail(f"advisor transcript missing: {transcript}")
    if mock.count() != 3:
        fail(f"expected primary+advise+settle requests, got {mock.count()} (hidden turn?)")
    print(f"PASS cli-text card persisted ({logs[0]})")
    return base


def phase_cli_json(binary: str, mock: MockModel, base: Path) -> None:
    agent_dir, cwd, sessions, _ = write_config(base / "json", mock.port)
    out = subprocess.run(
        [
            binary,
            "--cwd", str(cwd),
            "--session-dir", str(sessions),
            "--no-tools",
            "--no-title",
            "--advisor",
            "--mode", "json",
            "say hi",
        ],
        env=omp_env(agent_dir),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    if out.returncode != 0:
        fail(f"CLI json exited {out.returncode}: {out.stderr}")
    frames = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
    pairs = [
        frame for frame in frames
        if frame.get("type") in {"message_start", "message_end"}
        and isinstance(frame.get("message"), dict)
        and frame["message"].get("customType") == "advisor"
    ]
    if len(pairs) != 2:
        fail(f"expected one advisor card pair, got {len(pairs)}: {pairs}")
    body = json.dumps(pairs)
    if "write target deleted" not in body:
        fail("card text missing from the wire")
    print("PASS cli-json advisor card pair on the wire")


def phase_cli_error(binary: str, mock: MockModel, base: Path) -> None:
    mock.advisor_error = True
    try:
        agent_dir, cwd, sessions, _ = write_config(base / "error", mock.port)
        out = cli_run(binary, agent_dir, cwd, sessions)
        if out.returncode != 0:
            fail(f"advisor failure must not fail the session: {out.stderr}")
        if "<advisory" in out.stderr or "write target deleted" in out.stderr:
            fail("a failed advisor call must not fabricate a card")
        if mock.count() > 3:
            fail(f"quota latch retried forever: {mock.count()} requests")
        print("PASS cli-error quota contained, exit 0, no card")
    finally:
        mock.advisor_error = False


def phase_rpc(binary: str, mock: MockModel, base: Path) -> None:
    agent_dir, cwd, sessions, _ = write_config(base / "rpc", mock.port)
    proc, lines = spawn_with_reader(
        binary,
        [
            binary,
            "--cwd", str(cwd),
            "--session-dir", str(sessions),
            "--no-tools",
            "--no-title",
            "--advisor",
            "--mode", "rpc",
        ],
        omp_env(agent_dir),
    )
    ready = read_lines(proc, lines, 20, lambda frame: frame.get("type") == "ready")
    if not ready:
        fail("RPC never sent the ready frame")
    send(proc, {"id": "1", "type": "prompt", "message": "say hi"})
    turn = read_lines(
        proc,
        lines,
        TIMEOUT,
        lambda frame: frame.get("type") == "response" and frame.get("command") == "prompt",
    )
    if not any(frame.get("type") == "turn_end" for frame in turn):
        fail("RPC prompt completed without a turn_end frame")
    if not any(
        frame.get("type") == "message_end"
        and isinstance(frame.get("message"), dict)
        and frame["message"].get("customType") == "advisor"
        for frame in turn
    ):
        fail("RPC wire must carry the advisor card as a custom message")
    send(proc, {"id": "2", "type": "get_advisor_status"})
    status = read_lines(
        proc,
        lines,
        30,
        lambda frame: frame.get("type") == "response"
        and frame.get("command") == "get_advisor_status",
    )
    response = next(
        (frame for frame in status if frame.get("command") == "get_advisor_status"), None
    )
    if response is None or not response.get("success"):
        fail(f"get_advisor_status failed: {response}")
    data = response.get("data") or {}
    if not data.get("configured") or not data.get("status"):
        fail(f"advisor status must report the live roster: {data}")
    code = close_stdin(proc)
    if code != 0:
        fail(f"RPC dispose exited {code}")
    print("PASS rpc card + advisor status + dispose")


def phase_acp(binary: str, mock: MockModel, base: Path) -> None:
    agent_dir, cwd, sessions, _ = write_config(base / "acp", mock.port)
    proc, lines = spawn_with_reader(
        binary,
        [
            binary,
            "--cwd", str(cwd),
            "--session-dir", str(sessions),
            "--no-tools",
            "--no-title",
            "--advisor",
            "acp",
        ],
        omp_env(agent_dir),
    )
    send(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": 1, "clientCapabilities": {}},
        },
    )
    init = read_lines(proc, lines, 20, lambda frame: frame.get("id") == 1)
    if not init:
        fail("ACP initialize produced no response")
    send(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": {"cwd": str(cwd), "mcpServers": []},
        },
    )
    new = read_lines(proc, lines, 30, lambda frame: frame.get("id") == 2)
    created = next((frame for frame in new if frame.get("id") == 2), None)
    if created is None or "error" in created:
        fail(f"session/new failed: {created}")
    session_id = (created.get("result") or {}).get("sessionId")
    if not session_id:
        fail(f"session/new returned no sessionId: {created}")
    send(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {"sessionId": session_id, "prompt": [{"type": "text", "text": "say hi"}]},
        },
    )
    updates = read_lines(
        proc,
        lines,
        TIMEOUT,
        lambda frame: frame.get("method") == "session/update"
        and "advisor"
        in json.dumps((frame.get("params") or {}).get("update") or {}).get("_meta", {})
        or frame.get("id") == 3,
    )
    if not any(
        "advisor" in json.dumps((frame.get("params") or {}).get("update") or {}).get("_meta", {})
        for frame in updates
        if frame.get("method") == "session/update"
    ):
        fail("ACP must surface the advice card in a session/update")
    send(
        proc,
        {
            "jsonrpc": "2.0",
            "method": "session/cancel",
            "params": {"sessionId": session_id},
        },
    )
    code = close_stdin(proc)
    if code != 0:
        fail(f"ACP dispose exited {code}")
    print("PASS acp advisor update + cancel + dispose")


def main() -> None:
    binary = os.environ.get("OMP_BIN") or (sys.argv[1] if len(sys.argv) > 1 else "target/debug/omp")
    binary = shutil.which(binary) or binary
    if not Path(binary).is_file():
        raise SystemExit(f"omp binary not found: {binary}")
    base = base_dir()
    print(f"artifacts: {base}")
    mock = MockModel()
    mock.start()
    try:
        phase_cli_text(binary, mock, base)
        phase_cli_json(binary, mock, base)
        phase_cli_error(binary, mock, base)
        phase_rpc(binary, mock, base)
        phase_acp(binary, mock, base)
    finally:
        mock.stop()
    print("PASS advisor smoke (CLI text/json, error, RPC, ACP)")


if __name__ == "__main__":
    main()
