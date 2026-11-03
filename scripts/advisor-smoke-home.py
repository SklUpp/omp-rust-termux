#!/usr/bin/env python3
"""HOME-confined advisor pipeline smoke test (CLI text, CLI JSON, RPC, ACP).

Invocation:
    OMP_BIN=target/debug/omp python3 scripts/advisor-smoke-home.py
    python3 scripts/advisor-smoke-home.py /path/to/omp

Everything is created under ``$HOME/task-tmp/advisor-smoke-<pid>`` (HOME required;
TMPDIR is used only when it is set and is not ``/tmp``). Nothing outside that
directory is touched; sessions, transcripts, the mock server log, and captured
stdout/stderr stay there for inspection. No process is killed with a forced
signal unless a phase times out; a timeout fails the step and the child is
terminated.

Phases (all assert observable behavior, never source shape):
 1. CLI text:        `--advisor` succeeds, the blocker card is visible.
 2. CLI JSON:        card pair on the wire, one persisted `custom_message`,
                     one `__advisor.reviewer.jsonl`, no hidden primary turn.
 3. CLI error:       advisor 429 is contained: exit 0, no fabricated card.
 4. RPC:             ready -> prompt -> turn_end -> get_advisor_status,
                     advisor frames on the wire, EOF dispose exits 0.
 5. ACP:             initialize -> session/new -> session/prompt, advisor
                     update visible, session/cancel returns, EOF exits 0.
 6. Child:           root advisor off, real task spawn with `advisor: true`,
                     child provider call, advice card, child-local transcript,
                     no advisor usage in the parent session log.
 7. Child failure:   scripted child provider error is contained.
 8. Child cancel:    skipped until a deterministic mock gate exists.
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
        self.child_error = False
        self.child_phase = False
        self.child_gate = False
        self.child_gate_seen = threading.Event()
        self.child_gate_release = threading.Event()
        self.child_gate_timed_out = False
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
                    messages = payload.get("messages", [])
                    system = json.dumps(messages[:1])
                    advisor = "passive advisor" in system
                    child = "CHILD-SMOKE-MARKER" in system
                    has_tool_result = any(
                        message.get("role") == "tool" for message in messages
                    )
                    if child and mock.child_gate:
                        mock.child_gate_seen.set()
                        if mock.child_gate_release.wait(timeout=10):
                            # Abort observed: close without a scripted assistant
                            # chunk, so no fabricated card/transcript can follow.
                            self.close_connection = True
                            return
                        mock.child_gate_timed_out = True
                    if not advisor and mock.primary_delay:
                        delay = mock.primary_delay
                    else:
                        delay = 0.0
                    usage = (
                        'data: {"choices":[],"usage":{"prompt_tokens":19191,'
                        '"completion_tokens":1919,"total_tokens":21110}}\n\n'
                    )
                    if advisor and mock.advisor_error:
                        chunk = (
                            'data: {"error":{"type":"insufficient_quota",'
                            '"message":"429 rate limit exceeded"}}\n\ndata: [DONE]\n\n'
                        )
                    elif advisor and mock.advisor_notes_sent == 0:
                        mock.advisor_notes_sent += 1
                        args = json.dumps(
                            {
                                "note": (
                                    "child advice marker"
                                    if mock.child_phase
                                    else "write target deleted"
                                ),
                                "severity": "blocker",
                            }
                        )
                        chunk = (
                            "data: "
                            + json.dumps(
                                {
                                    "choices": [
                                        {
                                            "delta": {
                                                "tool_calls": [
                                                    {
                                                        "index": 0,
                                                        "id": "call-adv",
                                                        "type": "function",
                                                        "function": {
                                                            "name": "advise",
                                                            "arguments": args,
                                                        },
                                                    }
                                                ]
                                            }
                                        }
                                    ]
                                }
                            )
                            + "\n\n"
                            + usage
                            + "data: [DONE]\n\n"
                        )
                    elif advisor:
                        chunk = (
                            'data: {"choices":[{"delta":{"content":"noted"},'
                            '"finish_reason":"stop"}]}\n\n'
                            + usage
                            + "data: [DONE]\n\n"
                        )
                    elif child and mock.child_error:
                        chunk = (
                            'data: {"error":{"type":"server_error",'
                            '"message":"child provider failed"}}\n\n'
                            "data: [DONE]\n\n"
                        )
                    elif child:
                        chunk = (
                            'data: {"choices":[{"delta":{"content":"child finished"},'
                            '"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
                        )
                    elif mock.child_phase and not has_tool_result:
                        args = json.dumps(
                            {
                                "name": "child-smoke-1",
                                "agent": "child-smoke",
                                "task": "do child work",
                            }
                        )
                        chunk = (
                            "data: "
                            + json.dumps(
                                {
                                    "choices": [
                                        {
                                            "delta": {
                                                "tool_calls": [
                                                    {
                                                        "index": 0,
                                                        "id": "call-task",
                                                        "type": "function",
                                                        "function": {
                                                            "name": "task",
                                                            "arguments": args,
                                                        },
                                                    }
                                                ]
                                            }
                                        }
                                    ]
                                }
                            )
                            + "\n\n"
                            + "data: [DONE]\n\n"
                        )
                    else:
                        chunk = (
                            'data: {"choices":[{"delta":{"content":"root done"},'
                            '"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
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

    def begin_phase(self) -> None:
        with self.lock:
            self.requests.clear()
            self.advisor_notes_sent = 0
            self.advisor_error = False
            self.child_error = False
            self.child_phase = False
            self.child_gate = False
            self.child_gate_timed_out = False
            self.child_gate_seen.clear()
            self.child_gate_release.clear()

    def dump(self, path: Path) -> None:
        with self.lock:
            path.write_text(json.dumps(self.requests, indent=2))

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def count(self) -> int:
        with self.lock:
            return len(self.requests)


def base_dir() -> Path:
    home = os.environ.get("HOME")
    if home:
        root = Path(home) / "task-tmp"
    else:
        tmp = os.environ.get("TMPDIR")
        if not tmp:
            raise SystemExit("HOME or TMPDIR is required for advisor smoke artifacts")
        root = Path(tmp)
        if root.resolve() == Path("/tmp"):
            raise SystemExit("refusing /tmp outside HOME for advisor smoke artifacts")
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="advisor-smoke-", dir=root))


def write_config(base: Path, port: int) -> tuple[Path, Path, Path, Path]:
    agent_dir = base / "agent"
    cwd = base / "work"
    sessions = base / "sessions"
    for directory in (agent_dir, cwd / ".omp", cwd / ".omp" / "agents", sessions):
        directory.mkdir(parents=True, exist_ok=True)
    (cwd / ".omp" / "agents" / "child-smoke.md").write_text(
        "---\n"
        "name: child-smoke\n"
        "description: Child advisor smoke agent\n"
        "tools: [read]\n"
        "blocking: true\n"
        "advisor: true\n"
        "---\n"
        "You are the CHILD-SMOKE-MARKER child. Do exactly one short turn.\n"
    )
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
            "--yolo",
            "--mode", "text",
            "--advisor",
            "say hi",
        ],
        env=omp_env(agent_dir),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


def child_run(binary: str, agent_dir: Path, cwd: Path, sessions: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            binary,
            "--cwd", str(cwd),
            "--session-dir", str(sessions),
            "--no-title",
            "--yolo",
            "--mode", "json",
            "spawn one child-smoke task",
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


ENV_ARTIFACT_KEYS = (
    "HOME",
    "PI_CODING_AGENT_DIR",
    "OMP_AGENT_DIR",
    "PI_PROFILE",
    "OMP_PROFILE",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_CACHE_HOME",
)


def dump_env(base: Path, agent_dir: Path, cwd: Path, sessions: Path, binary: str) -> None:
    """Record which profile/agent dir the fixture process really selected."""
    env = omp_env(agent_dir)
    watchdog_candidates: list[dict] = []
    current = cwd
    while True:
        for parent in (current / ".omp", current):
            for name in ("WATCHDOG.yml", "WATCHDOG.yaml"):
                path = parent / name
                if path.is_file():
                    watchdog_candidates.append(
                        {"path": str(path), "bytes": path.stat().st_size}
                    )
        if current == Path(env["HOME"]):
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    for name in ("WATCHDOG.yml", "WATCHDOG.yaml"):
        path = agent_dir / name
        if path.is_file():
            watchdog_candidates.append({"path": str(path), "bytes": path.stat().st_size})
    artifact = {
        "binary": binary,
        "agent_dir": str(agent_dir),
        "cwd": str(cwd),
        "sessions": str(sessions),
        "cli_text_args": [
            binary,
            "--cwd", str(cwd),
            "--session-dir", str(sessions),
            "--no-tools",
            "--no-title",
            "--yolo",
            "--mode", "text",
            "--advisor",
            "say hi",
        ],
        "watchdog_candidates": watchdog_candidates,
        "env": {key: env.get(key) for key in ENV_ARTIFACT_KEYS},
    }
    (base / "fixture-env.json").write_text(json.dumps(artifact, indent=2) + "\n")


def diagnose_cli_failure(
    binary: str,
    agent_dir: Path,
    cwd: Path,
    sessions: Path,
    base: Path,
) -> None:
    """Probe the same env/config: status before and after the same prompt."""
    report: dict = {
        "error": None,
        "ready": False,
        "pre_prompt_status": None,
        "status": None,
        "frames": [],
    }
    proc = None
    try:
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
        report["frames"].extend(ready)
        report["ready"] = any(frame.get("type") == "ready" for frame in ready)
        if report["ready"]:
            send(proc, {"id": "diag-1", "type": "get_advisor_status"})
            pre = read_lines(
                proc,
                lines,
                30,
                lambda frame: frame.get("type") == "response"
                and frame.get("command") == "get_advisor_status",
            )
            report["frames"].extend(pre)
            report["pre_prompt_status"] = next(
                (
                    frame
                    for frame in pre
                    if frame.get("type") == "response"
                    and frame.get("command") == "get_advisor_status"
                ),
                None,
            )
            # The manager is lazy: status before a prompt only reports the empty
            # slot, never the config. Run the same prompt and probe status after
            # `agent_start` but before the terminal `turn_end`; the manager is
            # created before the first provider call.
            send(proc, {"id": "diag-2", "type": "prompt", "message": "say hi"})
            started = read_lines(
                proc,
                lines,
                20,
                lambda frame: frame.get("type") == "agent_start",
            )
            report["frames"].extend(started)
            send(proc, {"id": "diag-active", "type": "get_advisor_status"})
            turn = read_lines(
                proc,
                lines,
                TIMEOUT,
                lambda frame: frame.get("type") == "turn_end",
            )
            report["frames"].extend(turn)
            report["active_status"] = next(
                (
                    frame
                    for frame in turn
                    if frame.get("type") == "response"
                    and frame.get("id") == "diag-active"
                ),
                None,
            )
            report["terminal_turn_end"] = any(
                frame.get("type") == "turn_end" for frame in turn
            )
            if not report["terminal_turn_end"]:
                report["error"] = (
                    "diagnostic prompt ended without a terminal turn_end "
                    "(timeout or EOF); post-prompt status was not sampled"
                )
            else:
                send(proc, {"id": "diag-3", "type": "get_advisor_status"})
                status = read_lines(
                    proc,
                    lines,
                    30,
                    lambda frame: frame.get("type") == "response"
                    and frame.get("command") == "get_advisor_status",
                )
                report["frames"].extend(status)
                report["status"] = next(
                    (
                        frame
                        for frame in status
                        if frame.get("type") == "response"
                        and frame.get("command") == "get_advisor_status"
                    ),
                    None,
                )
    except (OSError, subprocess.SubprocessError, AssertionError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if proc is not None:
            try:
                report["exit"] = close_stdin(proc, timeout=20)
            except (subprocess.SubprocessError, AssertionError) as error:
                report["exit_error"] = f"{type(error).__name__}: {error}"
        (base / "cli-text-rpc-status.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )


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
    dump_env(base, agent_dir, cwd, sessions, binary)
    out = cli_run(binary, agent_dir, cwd, sessions)
    (base / "cli-text.stdout").write_text(out.stdout)
    (base / "cli-text.stderr").write_text(out.stderr)
    mock.dump(base / "mock-requests.json")
    if out.returncode != 0:
        diagnose_cli_failure(binary, agent_dir, cwd, sessions, base)
        (base / "mock-requests-after-diagnostic.json").write_text(
            json.dumps(mock.requests, indent=2) + "\n"
        )
        fail(f"CLI text exited {out.returncode}: {out.stderr}")
    if "advisor:" not in out.stderr or "write target deleted" not in out.stderr:
        systems = [
            json.dumps(request.get("messages", [])[:1])[:200]
            for request in mock.requests
        ]
        diagnose_cli_failure(binary, agent_dir, cwd, sessions, base)
        (base / "mock-requests-after-diagnostic.json").write_text(
            json.dumps(mock.requests, indent=2) + "\n"
        )
        fail(
            f"CLI text must show the card on stderr: {out.stderr!r}; "
            f"provider requests={len(systems)} {systems}"
        )
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
            "--yolo",
            "--mode", "json",
            "--advisor",
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
            "--yolo",
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
        lambda frame: frame.get("type") == "turn_end",
    )
    if not any(frame.get("type") == "turn_end" for frame in turn):
        fail("RPC prompt completed without a turn_end frame")
    # The transport response may precede `turn_end`; advisor cards may settle
    # just after it. The status response is the post-turn barrier that drains
    # those late frames before we assert the card.
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
    frames = turn + status
    (base / "rpc" / "rpc-frames.json").write_text(
        json.dumps(frames, indent=2) + "\n"
    )
    mock.dump(base / "rpc" / "rpc-mock-requests.json")
    code = close_stdin(proc)
    tail = read_lines(proc, lines, 5, lambda _frame: False)
    frames += tail
    (base / "rpc" / "rpc-frames.json").write_text(
        json.dumps(frames, indent=2) + "\n"
    )
    if not any(frame.get("type") == "message_end"
        and isinstance(frame.get("message"), dict)
        and frame["message"].get("customType") == "advisor"
        for frame in frames
    ):
        fail("RPC wire must carry the advisor card as a custom message")
    if response is None or not response.get("success"):
        fail(f"get_advisor_status failed: {response}")
    data = response.get("data") or {}
    if not data.get("configured") or not data.get("status"):
        fail(f"advisor status must report the live roster: {data}")
    if code != 0:
        fail(f"RPC dispose exited {code}")
    print("PASS rpc card + advisor status + dispose")


def acp_update_has_advisor(frame: dict) -> bool:
    params = frame.get("params")
    update = params.get("update") if isinstance(params, dict) else None
    if not isinstance(update, dict):
        return False
    meta = update.get("_meta")
    if isinstance(meta, dict):
        return "advisor" in json.dumps(meta, sort_keys=True)
    if isinstance(meta, str):
        return "advisor" in meta
    return False


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
            "--yolo",
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
        lambda frame: (
            frame.get("method") == "session/update" and acp_update_has_advisor(frame)
        )
        or frame.get("id") == 3,
    )
    if not any(
        acp_update_has_advisor(frame)
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


def phase_child(binary: str, mock: MockModel, base: Path) -> None:
    agent_dir, cwd, sessions, _ = write_config(base / "child", mock.port)
    mock.child_phase = True
    out = child_run(binary, agent_dir, cwd, sessions)
    if out.returncode != 0:
        fail(f"child advisor run exited {out.returncode}: {out.stderr}")
    frames = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
    cards = [
        frame for frame in frames
        if frame.get("type") == "message_end"
        and isinstance(frame.get("message"), dict)
        and frame["message"].get("customType") == "advisor"
    ]
    if not any("child advice marker" in json.dumps(card) for card in cards):
        fail(f"child advice card missing: {cards}")
    requests = list(mock.requests)
    child_requests = [
        request for request in requests
        if "CHILD-SMOKE-MARKER" in json.dumps(request.get("messages", [])[:1])
        and "passive advisor" not in json.dumps(request.get("messages", [])[:1])
    ]
    advisor_requests = [
        request for request in requests
        if "passive advisor" in json.dumps(request.get("messages", [])[:1])
    ]
    if not child_requests:
        fail("mock never received a real child provider request")
    if not advisor_requests:
        fail("mock never received a child advisor provider request")
    if any(
        "spawn one child-smoke task" in json.dumps(request.get("messages", []))
        for request in advisor_requests
    ):
        fail("root advisor ran despite --advisor being off")
    if not any(
        "child finished" in json.dumps(request.get("messages", []))
        for request in advisor_requests
    ):
        fail("child advisor requests do not contain the child's completed turn")
    root_logs = [
        path for path in session_logs(sessions)
        if "subagents" not in path.parts and "__advisor" not in path.name
    ]
    if len(root_logs) != 1:
        fail(f"expected one root session log, got {root_logs}")
    root_log = root_logs[0].read_text()
    if "child-smoke" not in root_log:
        fail("root session log does not record the real child task spawn")
    if "child advice marker" not in root_log:
        fail("child advice card was not persisted by the parent host")
    root_advisors = [
        path for path in sessions.rglob("__advisor*.jsonl")
        if "subagents" not in path.parts
    ]
    if root_advisors:
        fail(f"root advisor ran despite --advisor being off: {root_advisors}")
    child_logs = [
        path for path in session_logs(sessions)
        if "subagents" in path.parts
    ]
    if not child_logs:
        fail("child session log missing under subagents")
    child_text = "\n".join(path.read_text() for path in child_logs)
    if "19191" not in child_text:
        fail("child provider usage is not recorded in the child session log")
    child_advisors = [
        path for path in sessions.rglob("__advisor*.jsonl")
        if "subagents" in path.parts
    ]
    if not child_advisors:
        fail("child-local __advisor transcript missing")
    if "child advice marker" not in child_advisors[0].read_text():
        fail("child advisor transcript does not contain its delivered note")
    if "19191" in root_log:
        fail("advisor usage leaked into the parent session log")
    print(f"PASS child advisor opt-in, card, provider calls, transcript ({child_advisors[0]})")


def phase_child_failure(binary: str, mock: MockModel, base: Path) -> None:
    agent_dir, cwd, sessions, _ = write_config(base / "child-failure", mock.port)
    mock.child_phase = True
    mock.child_error = True
    out = child_run(binary, agent_dir, cwd, sessions)
    if out.returncode != 0:
        fail(f"contained child failure exited {out.returncode}: {out.stderr}")
    requests = list(mock.requests)
    if not any(
        "CHILD-SMOKE-MARKER" in json.dumps(request.get("messages", [])[:1])
        for request in requests
    ):
        fail("child failure phase never reached the child provider")
    if "child advice marker" in out.stdout:
        fail("failed child advisor fabricated a card")
    root_advisors = [
        path for path in sessions.rglob("__advisor*.jsonl")
        if "subagents" not in path.parts
    ]
    if root_advisors:
        fail(f"root advisor ran despite --advisor being off: {root_advisors}")
    print("PASS child provider failure contained")


def phase_child_cancel(binary: str, mock: MockModel, base: Path) -> None:
    agent_dir, cwd, sessions, _ = write_config(base / "child-cancel", mock.port)
    mock.child_phase = True
    mock.child_gate = True
    proc, lines = spawn_with_reader(
        binary,
        [
            binary,
            "--cwd", str(cwd),
            "--session-dir", str(sessions),
            "--no-title",
            "--yolo",
            "--mode", "rpc",
        ],
        omp_env(agent_dir),
    )
    ready = read_lines(proc, lines, 20, lambda frame: frame.get("type") == "ready")
    if not ready:
        fail("child-cancel RPC never sent ready")
    send(proc, {"id": "1", "type": "prompt", "message": "spawn one child-smoke task"})
    if not mock.child_gate_seen.wait(TIMEOUT):
        mock.child_gate_release.set()
        close_stdin(proc)
        fail("child stream gate never opened")
    abort_frames: list[dict] = []
    after_abort: list[dict] = []
    try:
        send(proc, {"id": "2", "type": "abort"})
        # Release immediately after sending abort: the command is already on
        # the wire; waiting for its response first could deadlock behind the
        # gated child provider stream.
        mock.child_gate_release.set()
        abort_frames = read_lines(
            proc,
            lines,
            20,
            lambda frame: frame.get("type") == "response"
            and frame.get("command") == "abort",
        )
        if not any(
            frame.get("type") == "response" and frame.get("command") == "abort"
            for frame in abort_frames
        ):
            fail("RPC abort produced no response")
        if mock.child_gate_timed_out:
            fail("child stream gate timed out before abort")
        mock.child_gate_release.set()
        terminal = ready + abort_frames
        if not any(
            frame.get("type") in {"turn_end", "error"} for frame in terminal
        ):
            after_abort = read_lines(
                proc,
                lines,
                TIMEOUT,
                lambda frame: frame.get("type") in {"turn_end", "error"},
            )
    finally:
        mock.child_gate_release.set()
    frames = ready + abort_frames + after_abort
    if not any(frame.get("type") in {"turn_end", "error"} for frame in frames):
        fail("aborted child task never produced a terminal frame")
    primary_requests = [
        request
        for request in list(mock.requests)
        if "CHILD-SMOKE-MARKER" not in json.dumps(request.get("messages", [])[:1])
        and "passive advisor" not in json.dumps(request.get("messages", [])[:1])
    ]
    if len(primary_requests) != 1:
        fail(
            "abort did not stop the root before a second primary turn "
            f"({len(primary_requests)} primary requests)"
        )
    if any(
        isinstance(frame.get("message"), dict)
        and frame["message"].get("customType") == "advisor"
        for frame in frames
    ):
        fail("aborted child task fabricated an advisor card")
    if any(
        "passive advisor" in json.dumps(request.get("messages", [])[:1])
        for request in list(mock.requests)
    ):
        fail("aborted child spawned an advisor provider request")
    transcripts = list(sessions.rglob("__advisor*.jsonl"))
    if transcripts:
        fail(f"aborted child left an advisor transcript: {transcripts}")
    code = close_stdin(proc)
    if code != 0:
        fail(f"child-cancel RPC dispose exited {code}")
    print("PASS child cancellation gate, RPC abort, no card/transcript")


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
        mock.begin_phase()
        phase_cli_text(binary, mock, base)
        mock.begin_phase()
        phase_cli_json(binary, mock, base)
        mock.begin_phase()
        phase_cli_error(binary, mock, base)
        mock.begin_phase()
        phase_rpc(binary, mock, base)
        mock.begin_phase()
        phase_acp(binary, mock, base)
        mock.begin_phase()
        phase_child(binary, mock, base)
        mock.begin_phase()
        phase_child_failure(binary, mock, base)
        phase_child_cancel(binary, mock, base)
    finally:
        mock.stop()
    print("PASS advisor smoke (CLI text/json, error, RPC, ACP, child)")


if __name__ == "__main__":
    main()
