"""エージェント向けの MCP（tools/soundsep_mcp.py）とスキル導入（tools/install_skill.py）の検証。GPU とモデルは使わない。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import soundsep_control as control
import soundsep_mcp as mcp


def _call(method, params=None):
    return mcp.dispatch({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}, mcp.Server())


def _tool(name, arguments=None):
    result = _call("tools/call", {"name": name, "arguments": arguments or {}})
    return result["isError"], result["content"][0]["text"]


class McpConfigTests(unittest.TestCase):
    def test_registration_uses_only_relative_paths(self):
        entry = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["soundsep"]
        for value in [entry["command"], *entry["args"]]:
            self.assertFalse(Path(value).is_absolute(), value)
        self.assertTrue((ROOT / entry["args"][-1]).is_file())

    def test_app_root_is_derived_from_the_script_location(self):
        self.assertEqual(control.APP_ROOT, ROOT)
        self.assertEqual(mcp.JOBS, ROOT / "output" / "_agent_jobs")


class SkillInstallTests(unittest.TestCase):
    """MCP を使わない導入（スキル）の検証。"""

    def test_install_writes_the_app_location_and_remove_only_touches_its_own_file(self):
        import install_skill
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as tmp:
            target = install_skill.install(tmp)
            text = target.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\nname: soundsep\n"))
            self.assertIn(f'$root = "{ROOT}"', text)
            self.assertNotIn(install_skill.ROOT_LINE, text)
            install_skill.install(tmp)  # 自分が導入したものは更新できる
            self.assertEqual(install_skill.remove(tmp), target)
            self.assertFalse(target.parent.exists())

            target.parent.mkdir()
            target.write_text("別のスキル", encoding="utf-8")
            with self.assertRaises(SystemExit):
                install_skill.install(tmp)
            with self.assertRaises(SystemExit):
                install_skill.remove(tmp)
            self.assertEqual(target.read_text(encoding="utf-8"), "別のスキル")

    def test_project_skill_holds_no_absolute_path(self):
        import install_skill
        text = install_skill.SOURCE.read_text(encoding="utf-8")
        self.assertEqual(text.count(install_skill.ROOT_LINE), 1)
        self.assertNotIn(":\\", text)
        self.assertNotIn(":/", text)


class McpProtocolTests(unittest.TestCase):
    def test_initialize_and_tool_list(self):
        self.assertEqual(_call("initialize", {"protocolVersion": "2025-06-18"})["protocolVersion"], "2025-06-18")
        names = [tool["name"] for tool in _call("tools/list")["tools"]]
        self.assertEqual(names, ["soundsep_status", "soundsep_models", "soundsep_separate", "soundsep_subseparate",
                                 "soundsep_midi", "soundsep_task_status", "soundsep_cancel"])

    def test_models_lists_the_app_catalog(self):
        error, text = _tool("soundsep_models")
        self.assertFalse(error, text)
        entries = json.loads(text)
        self.assertTrue(entries)
        self.assertTrue(all(isinstance(entry["downloaded"], bool) for entry in entries))

    def test_bad_requests_are_reported_as_tool_errors(self):
        self.assertTrue(_tool("soundsep_task_status", {"task_id": "000000000000"})[0])
        self.assertTrue(_tool("soundsep_task_status", {"task_id": "../x"})[0])
        self.assertTrue(_tool("soundsep_separate", {"input": "x.wav"})[0])  # out_dir がない
        with self.assertRaises(mcp.ProtocolError):
            _call("tools/call", {"name": "no_such_tool"})

    def test_stdio_round_trip_from_the_repository_root(self):
        entry = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["soundsep"]
        requests = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
        done = subprocess.run([sys.executable, *entry["args"]], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                              input="".join(json.dumps(item) + "\n" for item in requests), timeout=60)
        lines = [json.loads(line) for line in done.stdout.splitlines()]
        self.assertEqual([line["id"] for line in lines], [1, 2])  # 通知には応答しない
        self.assertEqual(lines[0]["result"]["serverInfo"]["name"], "soundsep")
        self.assertEqual(len(lines[1]["result"]["tools"]), 7)


if __name__ == "__main__":
    unittest.main()
