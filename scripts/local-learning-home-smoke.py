#!/usr/bin/env python3
"""C7 local-learning HOME smoke against a real `omp` binary.

Not a Cargo test. Parent runs it after a build:

    OMP_BIN=./target/debug/omp python3 scripts/local-learning-home-smoke.py --keep

It creates an isolated HOME/config/state tree (never the caller's real agent
dir), starts a localhost OpenAI-compatible provider, and drives `omp acp` over
stdio. Artifacts stay in the printed root when the run fails; on success the
root is removed unless --keep is passed.

Covered:
- `learn` writes learned.md and a managed skill; the same ACP session refreshes
  its snapshot, so the skill is prompt-visible and readable via skill://.
- memory summary and learned lessons are injected and readable separately; the
  learned lesson still loads after memory_summary.md is moved aside.
- a top-level disabledExtensions entry hides a skill from the prompt listing,
  slash expansion, and the provider wire.
- autoContinue: a normal tool turn triggers a capture turn; a cancelled turn,
  an active plan-mode turn, and turns started in goal mode (or still in it do
  not).
- the capture turn mounts no MCP tool and does not spawn the MCP server again;
  the real MCP process marker file proves it.

Not covered: autoloadSkills/task children and RPC-only paths; this script uses
the ACP transport and one provider. It is a smoke, not a suite replacement.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ACP_FIXTURE = REPO / "crates/omp-cli/tests/acp_accept.py"
REQUEST_TIMEOUT = 25.0
PROMPT_TIMEOUT = 40.0


def load_acp_fixture():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("c7_acp_fixture", ACP_FIXTURE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


acp_fixture = load_acp_fixture()
Provider = acp_fixture.Provider
text_of = acp_fixture.text_of

LEARN_MARKER = "SMOKE LESSON FROM LEARN"
SUMMARY_MARKER = "SMOKE SUMMARY TEXT"
SKILL_BODY_MARKER = "SMOKE MANAGED BODY"
BLOCKED_BODY_MARKER = "SMOKE BLOCKED BODY"
CAPTURE_TOOLS = {"learn", "manage_skill"}


class SmokeProvider(Provider):
    """Deterministic tool calls for the smoke prompts."""

    def frames(self, body):
        messages = body.get("messages") or []
        last_user = ""
        tool_after_user = False
        for message in messages:
            role = message.get("role")
            if role == "tool":
                tool_after_user = True
            elif role == "user":
                last_user = text_of(message.get("content"))
                tool_after_user = False
        created = int(time.time())
        base = {
            "id": "chatcmpl-smoke",
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model,
        }

        def chunk(delta, finish=None):
            payload = dict(base)
            payload["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
            return "data: " + json.dumps(payload)

        def tool_call(call_id, name, arguments):
            yield chunk(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            )
            yield chunk({}, "tool_calls")
            yield "data: [DONE]"

        def text(value):
            yield chunk({"role": "assistant", "content": value})
            yield chunk({}, "stop")
            yield "data: [DONE]"

        if "LEARN NOW" in last_user and not tool_after_user:
            yield from tool_call(
                "call_learn",
                "learn",
                {
                    "memory": LEARN_MARKER,
                    "context": "local-learning home smoke",
                    "skill": {
                        "action": "create",
                        "name": "captured",
                        "description": "Smoke managed skill",
                        "body": SKILL_BODY_MARKER,
                    },
                },
            )
            return
        if "READ SKILL" in last_user and not tool_after_user:
            yield from tool_call("call_read_skill", "read", {"path": "skill://captured"})
            return
        if "READ SUMMARY" in last_user and not tool_after_user:
            yield from tool_call("call_read_summary", "read", {"path": "memory://root"})
            return
        if "READ LEARNED" in last_user and not tool_after_user:
            yield from tool_call(
                "call_read_learned", "read", {"path": "memory://root/learned.md"}
            )
            return
        if "NORMAL TURN" in last_user and not tool_after_user:
            yield from tool_call("call_read_probe", "read", {"path": "probe.txt"})
            return
        if "PLAN TURN" in last_user and not tool_after_user:
            yield from tool_call("call_read_plan", "read", {"path": "probe.txt"})
            return
        if "START GOAL" in last_user and not tool_after_user:
            yield from tool_call(
                "call_goal_create",
                "goal",
                {"op": "create", "objective": "smoke objective", "token_budget": 1000000},
            )
            return
        if "GOAL ACTIVE" in last_user and not tool_after_user:
            yield from tool_call("call_read_goal", "read", {"path": "probe.txt"})
            return
        if "CANCEL TURN" in last_user and not tool_after_user:
            yield chunk({"role": "assistant", "content": "cancel smoke pending"})
            time.sleep(1.2)
            yield chunk({}, "stop")
            yield "data: [DONE]"
            return
        yield from text("smoke ack")


class AcpClient:
    """Minimal ACP stdio client; the fixture Provider owns the model side."""

    def __init__(self, binary: Path, root: Path, env: dict, cwd: Path) -> None:
        self.proc = subprocess.Popen(
            [str(binary), "acp"],
            cwd=str(cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.frames: "queue.Queue[dict]" = queue.Queue()
        self.raw: list[dict] = []
        self.stderr: list[str] = []
        self.next_id = 1
        self.lock = threading.Lock()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr.append(line.rstrip())

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                self.frames.put({"__bad_line__": line})
                continue
            with self.lock:
                self.raw.append(frame)
            self.frames.put(frame)

    def send(self, frame: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(frame) + "\n")
        self.proc.stdin.flush()

    def notify(self, method: str, params: dict | None = None) -> None:
        frame = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            frame["params"] = params
        self.send(frame)

    def wait_frame(self, predicate, timeout=REQUEST_TIMEOUT):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"timeout waiting for frame; raw={self.raw[-8:]!r} "
                    f"stderr={self.stderr[-8:]!r}"
                )
            try:
                frame = self.frames.get(timeout=min(0.25, remaining))
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise AssertionError(
                        f"ACP exited {self.proc.returncode}; stderr={self.stderr[-20:]!r}"
                    )
                continue
            if predicate(frame):
                return frame

    def request(self, method, params=None, timeout=REQUEST_TIMEOUT):
        request_id = self.next_id
        self.next_id += 1
        frame = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            frame["params"] = params
        self.send(frame)
        response = self.wait_frame(
            lambda f: f.get("id") == request_id and ("result" in f or "error" in f),
            timeout,
        )
        if "error" in response:
            raise AssertionError(f"{method} failed: {response['error']}")
        return response.get("result", {})

    def prompt(self, session_id: str, text: str, timeout=PROMPT_TIMEOUT) -> dict:
        request_id = self.next_id
        self.next_id += 1
        self.send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": text}],
                },
            }
        )
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"prompt {text!r} timed out; raw={self.raw[-8:]!r}")
            current = self.wait_frame(lambda _f: True, remaining)
            method = current.get("method")
            if method == "session/request_permission":
                options = current.get("params", {}).get("options", [])
                option_id = next(
                    (option.get("optionId") for option in options if option.get("optionId") == "allow_once"),
                    options[0].get("optionId") if options else "allow_once",
                )
                self.send(
                    {
                        "jsonrpc": "2.0",
                        "id": current["id"],
                        "result": {"outcome": {"outcome": "selected", "optionId": option_id}},
                    }
                )
                continue
            if method == "elicitation/create":
                self.send(
                    {
                        "jsonrpc": "2.0",
                        "id": current["id"],
                        "result": {"action": "accept", "content": {"value": "Approve and execute"}},
                    }
                )
                continue
            if current.get("id") == request_id and ("result" in current or "error" in current):
                if "error" in current:
                    raise AssertionError(f"prompt {text!r} failed: {current['error']}")
                return current.get("result", {})

    def barrier(self, session_id: str) -> None:
        """Round-trip a session command after a terminal prompt.

        Capture runs inside the turn worker before the prompt response, so a
        completed session round-trip proves the actor and its capture work have
        drained. Never use a sleep as absence evidence.
        """
        self.request("session/set_mode", {"sessionId": session_id, "modeId": "default"})

    def cancel_prompt(self, session_id: str, text: str, timeout=PROMPT_TIMEOUT) -> dict:
        request_id = self.next_id
        self.next_id += 1
        self.send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": text}],
                },
            }
        )
        self.wait_frame(
            lambda f: f.get("method") == "session/update"
            and f.get("params", {}).get("sessionId") == session_id
            and f["params"]["update"].get("sessionUpdate") == "agent_message_chunk",
            timeout,
        )
        self.notify("session/cancel", {"sessionId": session_id})
        response = self.wait_frame(
            lambda f: f.get("id") == request_id and ("result" in f or "error" in f),
            timeout,
        )
        if "error" in response:
            raise AssertionError(f"cancelled prompt failed: {response['error']}")
        return response.get("result", {})

    def close(self) -> int:
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        try:
            return self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return self.proc.wait(timeout=5)


def tool_names(body: dict) -> set[str]:
    names = set()
    for tool in body.get("tools") or []:
        name = tool.get("name") or tool.get("function", {}).get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def last_user(body: dict) -> str:
    text = ""
    for message in body.get("messages") or []:
        if message.get("role") == "user":
            text = text_of(message.get("content"))
    return text


def bodies_json(provider: Provider) -> str:
    with provider.lock:
        bodies = list(provider.bodies)
    return json.dumps(bodies, ensure_ascii=False)


def first_body(provider: Provider) -> list[dict]:
    with provider.lock:
        return list(provider.bodies)


def requests_for(provider: Provider, marker: str) -> list[dict]:
    with provider.lock:
        bodies = list(provider.bodies)
    return [body for body in bodies if marker in last_user(body)]


def capture_requests(provider: Provider, marker: str = "") -> list[dict]:
    with provider.lock:
        bodies = list(provider.bodies)
    return [
        body
        for body in bodies
        if tool_names(body)
        and tool_names(body) <= CAPTURE_TOOLS
        and (not marker or marker in json.dumps(body, ensure_ascii=False))
    ]


def write_config(agent_dir: Path, provider: Provider) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    config = (
        "defaultModel: mock/mock-model\n"
        "memory:\n"
        "  backend: local\n"
        "autolearn:\n"
        "  enabled: true\n"
        "  autoContinue: true\n"
        "  minToolCalls: 1\n"
        "goal:\n"
        "  enabled: true\n"
        "  continuationModes: [interactive]\n"
        "disabledExtensions:\n"
        "  - skill:blocked\n"
        "tools: [read, goal, learn, manage_skill]\n"
        "providers:\n"
        "  mock:\n"
        f"    baseUrl: \"{provider.base_url}\"\n"
        "    api: openai-completions\n"
        "    apiKey: test-key\n"
    )
    (agent_dir / "config.yml").write_text(config, encoding="utf-8")
    models = (
        "providers:\n"
        "  mock:\n"
        f"    baseUrl: \"{provider.base_url}\"\n"
        "    api: openai-completions\n"
        "    apiKey: test-key\n"
        "    models:\n"
        "      - id: mock-model\n"
        "        contextWindow: 128000\n"
    )
    (agent_dir / "models.yml").write_text(models, encoding="utf-8")


def write_mcp(project: Path, marker: Path) -> None:
    script = project / ".omp/mcp-smoke-server.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "import json, sys\n"
        f"open({str(marker)!r}, 'a').write('started\\n')\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if not line:\n"
        "        continue\n"
        "    message = json.loads(line)\n"
        "    method = message.get('method')\n"
        "    if method == 'initialize':\n"
        "        result = {'protocolVersion': '2024-11-05', 'capabilities': {}, 'serverInfo': {'name': 'mock', 'version': '1'}}\n"
        "    elif method == 'tools/list':\n"
        "        result = {'tools': [{'name': 'echo', 'description': 'echo', 'inputSchema': {'type': 'object', 'properties': {}}}]}\n"
        "    else:\n"
        "        continue\n"
        "    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'result': result}) + '\\n')\n"
        "    sys.stdout.flush()\n",
        encoding="utf-8",
    )
    (project / ".omp/mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "smoke": {
                        "command": sys.executable,
                        "args": [str(script), str(marker)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def marker_lines(marker: Path) -> int:
    if not marker.exists():
        return 0
    return len(marker.read_text(encoding="utf-8").splitlines())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omp", default=os.environ.get("OMP_BIN"), help="real omp binary")
    parser.add_argument("--keep", action="store_true", help="keep the temp HOME on success")
    args = parser.parse_args(argv)
    if not args.omp:
        parser.error("pass --omp or set OMP_BIN to the built binary")
    binary = Path(args.omp).resolve()
    if not binary.is_file():
        parser.error(f"OMP_BIN is not a file: {binary}")

    base = Path(os.environ.get("HOME") or Path.home()) / ".c7-local-learning-smoke"
    root = base / f"run-{os.getpid()}-{int(time.time())}"
    root.mkdir(parents=True)
    config_dir = root / "config"
    agent_dir = config_dir / "agent"
    data = root / "data"
    state = root / "state"
    tmp = root / "tmp"
    project = root / "workspace"
    sessions = root / "sessions"
    marker = root / "mcp-started.txt"
    for path in (agent_dir, data, state, tmp, project, sessions):
        path.mkdir(parents=True, exist_ok=True)
    (project / "probe.txt").write_text("probe\n", encoding="utf-8")
    project_skill = project / ".omp/skills/blocked/SKILL.md"
    project_skill.parent.mkdir(parents=True, exist_ok=True)
    project_skill.write_text(
        "---\nname: blocked\ndescription: Disabled smoke skill\n---\n"
        f"{BLOCKED_BODY_MARKER}\n",
        encoding="utf-8",
    )

    provider = SmokeProvider()
    provider.start()
    client = None
    try:
        write_config(agent_dir, provider)
        write_mcp(project, marker)
        env = dict(os.environ)
        env.update(
            HOME=str(root),
            PI_CONFIG_DIR=str(config_dir),
            PI_CODING_AGENT_DIR=str(agent_dir),
            XDG_DATA_HOME=str(data),
            XDG_STATE_HOME=str(state),
            TMPDIR=str(tmp),
            NO_COLOR="1",
            TERM="dumb",
        )
        client = AcpClient(binary, root, env, project)
        client.request(
            "initialize",
            {"protocolVersion": 1, "clientCapabilities": {}},
        )
        session_id = client.request(
            "session/new", {"cwd": str(project), "mcpServers": []}
        )["sessionId"]

        response = client.prompt(session_id, "LEARN NOW")
        assert response.get("stopReason") == "end_turn", response
        learned_files = list(agent_dir.rglob("learned.md"))
        assert learned_files, f"learn did not write learned.md under {agent_dir}"
        learned_file = learned_files[0]
        assert LEARN_MARKER in learned_file.read_text(encoding="utf-8")
        managed = list(agent_dir.rglob("managed-skills/captured/SKILL.md"))
        assert managed, "learn did not create the managed skill"
        assert SKILL_BODY_MARKER in managed[0].read_text(encoding="utf-8")

        memory_root = learned_file.parent
        (memory_root / "memory_summary.md").write_text(SUMMARY_MARKER + "\n", encoding="utf-8")

        client.prompt(session_id, "READ SKILL")
        text = bodies_json(provider)
        assert SKILL_BODY_MARKER in text, "skill:// read did not reach the provider"
        prompt_visible = any(
            "Smoke managed skill" in str(body.get("messages", [{}])[0])
            for body in first_body(provider)
        )
        assert prompt_visible, "managed skill must be prompt-visible after the refresh"

        client.prompt(session_id, "READ SUMMARY")
        text = bodies_json(provider)
        assert SUMMARY_MARKER in text, "memory://root did not serve the summary"
        guidance = requests_for(provider, "READ SUMMARY")[-1]
        system = str(guidance.get("messages", [{}])[0])
        assert "# Memory Guidance" in system
        assert "Memory summary:" in system

        client.prompt(session_id, "READ LEARNED")
        assert LEARN_MARKER in bodies_json(provider)
        backup = memory_root / "memory_summary.md.bak"
        (memory_root / "memory_summary.md").rename(backup)
        try:
            client.prompt(session_id, "READ LEARNED")
            without_summary = requests_for(provider, "READ LEARNED")[-1]
            system = str(without_summary.get("messages", [{}])[0])
            assert "Memory summary:" not in system, "summary must be omitted when absent"
            assert LEARN_MARKER in bodies_json(provider), "learned must not depend on summary"
        finally:
            backup.rename(memory_root / "memory_summary.md")

        client.prompt(session_id, "/skill:blocked")
        disabled = requests_for(provider, "/skill:blocked")
        assert disabled, "disabled prompt never reached the provider"
        wire = bodies_json(provider)
        assert BLOCKED_BODY_MARKER not in wire, "disabled skill body reached the provider"
        assert "blocked" not in str(disabled[-1].get("messages", [{}])[0]), "skill listed"
        assert last_user(disabled[-1]) == "/skill:blocked", "slash text must stay literal"

        before_normal = marker_lines(marker)
        client.prompt(session_id, "NORMAL TURN")
        normal = requests_for(provider, "NORMAL TURN")
        assert any(name.startswith("mcp__") for body in normal for name in tool_names(body)), (
            "normal turn must mount the MCP tool"
        )
        capture = capture_requests(provider, "NORMAL TURN")
        assert capture, "normal tool turn must trigger the autoContinue capture"
        assert all(
            not any(name.startswith("mcp__") for name in tool_names(body))
            for body in capture
        ), "capture must not mount MCP tools"
        after_capture = marker_lines(marker)
        assert after_capture == 1, (
            f"exactly one MCP process must start (before={before_normal}, "
            f"after={after_capture}); capture must not spawn another"
        )

        cancelled = client.cancel_prompt(session_id, "CANCEL TURN")
        assert cancelled.get("stopReason") == "cancelled", cancelled
        client.barrier(session_id)
        assert not capture_requests(provider, "CANCEL TURN"), "cancelled turn must not capture"

        client.request("session/set_mode", {"sessionId": session_id, "modeId": "plan"})
        client.prompt(session_id, "PLAN TURN")
        assert not capture_requests(provider, "PLAN TURN"), "active plan must block capture"
        client.request("session/set_mode", {"sessionId": session_id, "modeId": "default"})

        client.prompt(session_id, "START GOAL")
        assert not capture_requests(provider, "START GOAL"), "an active goal must block capture"
        client.prompt(session_id, "GOAL ACTIVE")
        assert not capture_requests(provider, "GOAL ACTIVE"), (
            "a turn started in goal mode must block capture"
        )

        exit_code = client.close()
        stderr_tail = client.stderr[-20:]
        client = None
        assert exit_code == 0, f"ACP exited {exit_code}; stderr={stderr_tail!r}"
        print(f"local learning home smoke passed: {root}")
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
        return 0
    except BaseException:
        print(f"local learning home smoke failed; artifacts kept at {root}", file=sys.stderr)
        raise
    finally:
        provider.stop()
        if client is not None:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
