"""Sound Separator の stdio MCP サーバー（JSON-RPC を 1 行 1 メッセージ）。登録名は soundsep。

リポジトリ直下の .mcp.json から、このリポジトリの venv の Python で起動する。
重い処理は soundsep_control.py を別プロセスで起動し、task_id をすぐ返す。進捗は soundsep_task_status で読む。
同時に動かすジョブは 1 本。モデルの取得や既存ファイルの上書きは行わない。
ジョブの状態・PID・ログは output/_agent_jobs/<task_id>/ に置く。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from soundsep_control import APP_ROOT, MODES, QUALITIES, SUBSEP_KINDS, ControlError, models, status

PROTOCOLS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
CONTROL = Path(__file__).resolve().parent / "soundsep_control.py"
JOBS = APP_ROOT / "output" / "_agent_jobs"


def schema(properties=None, required=None):
    return {"type": "object", "properties": properties or {}, "required": required or [], "additionalProperties": False}


PATH_IN = {"type": "string", "description": "既存の音声ファイルの絶対パス（読むだけ）"}
TASK = schema({"task_id": {"type": "string"}}, ["task_id"])
TOOLS = [
    {"name": "soundsep_status", "description": "アプリの場所、torch と GPU の空き、取得済みモデルを読む。分離は実行しない。",
     "inputSchema": schema(), "annotations": {"readOnlyHint": True, "openWorldHint": False}},
    {"name": "soundsep_models", "description": "モデルごとの取得済みかどうかを返す。取得は行わない。",
     "inputSchema": schema(), "annotations": {"readOnlyHint": True, "openWorldHint": False}},
    {"name": "soundsep_separate", "description": "音源をステムへ分けるジョブを開始し、task_id を返す。music は vocals/drums/bass/guitar/piano/other の 6 本。出力は入力と同じレート・長さの float WAV と manifest.json。exact は「元音源 − 全ステムの和」を other に足して足し戻しを一致させる（music で全ステムのときだけ）。",
     "inputSchema": schema({"input": PATH_IN,
                            "out_dir": {"type": "string", "description": "出力フォルダの絶対パス。同名の WAV があれば失敗する"},
                            "mode": {"type": "string", "enum": list(MODES), "default": "music"},
                            "quality": {"type": "string", "enum": list(QUALITIES), "default": "high"},
                            "model_id": {"type": "string", "description": "省略時はモードの既定モデル"},
                            "stems": {"type": "array", "items": {"type": "string"}, "description": "mega で抽出するカテゴリなど。省略時は全ステム"},
                            "exact": {"type": "boolean", "default": False}}, ["input", "out_dir"]),
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}},
    {"name": "soundsep_subseparate", "description": "ステムをさらに分けるジョブを開始する。vocals は lead/back、drums は kick/snare/cymbals/toms。lead と back の割り当ては曲と区間で入れ替わることがあり、manifest.json の window_rms（5 秒窓の音量）で確かめる。",
     "inputSchema": schema({"input": PATH_IN, "out_dir": {"type": "string"},
                            "kind": {"type": "string", "enum": list(SUBSEP_KINDS)}}, ["input", "out_dir", "kind"]),
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}},
    {"name": "soundsep_midi", "description": "basic-pitch で WAV を採譜し MIDI を書くジョブを開始する。和音も拾うが誤りを含むので、結果は検査する。",
     "inputSchema": schema({"input": PATH_IN, "out": {"type": "string", "description": "出力 .mid の絶対パス。既存なら失敗する"},
                            "onset": {"type": "number", "minimum": 0.05, "maximum": 0.95, "default": 0.5},
                            "frame": {"type": "number", "minimum": 0.05, "maximum": 0.95},
                            "min_note_ms": {"type": "number", "minimum": 10, "maximum": 2000, "default": 120}}, ["input", "out"]),
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}},
    {"name": "soundsep_task_status", "description": "ジョブの状態、進捗、結果、ログの末尾 8 行を読む。",
     "inputSchema": TASK, "annotations": {"readOnlyHint": True, "openWorldHint": False}},
    {"name": "soundsep_cancel", "description": "ジョブのプロセスを止める。書き出し済みのファイルは残す。",
     "inputSchema": TASK, "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}},
]


def _alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"], capture_output=True, text=True)
    return f'"{int(pid)}"' in out.stdout


def _read_job(task_id):
    if not isinstance(task_id, str) or not task_id.isalnum():
        raise ControlError("task_id が正しくありません")
    path = JOBS / task_id / "job.json"
    if not path.is_file():
        raise ControlError("そのジョブはありません")
    for _ in range(5):
        try:
            return path, json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            time.sleep(0.05)
    raise ControlError("ジョブの状態を読めません")


def _pid(job_dir):
    try:
        return int((job_dir / "pid.txt").read_text(encoding="ascii"))
    except (OSError, ValueError):
        return None


def _running():
    if not JOBS.is_dir():
        return None
    for job in JOBS.glob("*/job.json"):
        try:
            data = json.loads(job.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pid = _pid(job.parent)
        if data.get("state") == "running" and pid and _alive(pid):
            return job.parent.name
    return None


class Server:
    def _start(self, command_args):
        busy = _running()
        if busy:
            raise ControlError(f"別のジョブが実行中です: {busy}")
        task_id = uuid.uuid4().hex[:12]
        job_dir = JOBS / task_id
        job_dir.mkdir(parents=True)
        job_file = job_dir / "job.json"
        # 状態は起動前に書く。子プロセスも同じファイルを更新するので、PID は別ファイルに置く。
        job_file.write_text(json.dumps({"state": "running", "started": time.time(), "command": command_args}, ensure_ascii=False), encoding="utf-8")
        command = [sys.executable, "-X", "utf8", str(CONTROL), "--job-file", str(job_file), *command_args]
        log = open(job_dir / "log.txt", "w", encoding="utf-8")
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        (job_dir / "pid.txt").write_text(str(proc.pid), encoding="ascii")
        return {"task_id": task_id, "state": "running"}

    def separate(self, input, out_dir, mode="music", quality="high", model_id=None, stems=None, exact=False):
        args = ["separate", "--input", input, "--out", out_dir, "--mode", mode, "--quality", quality]
        if model_id:
            args += ["--model", model_id]
        if stems:
            args += ["--stems", *stems]
        if exact:
            args.append("--exact")
        return self._start(args)

    def subseparate(self, input, out_dir, kind):
        return self._start(["subseparate", "--input", input, "--out", out_dir, "--kind", kind])

    def midi(self, input, out, onset=0.5, frame=None, min_note_ms=120):
        args = ["midi", "--input", input, "--out", out, "--onset", str(onset), "--min-note-ms", str(min_note_ms)]
        if frame is not None:
            args += ["--frame", str(frame)]
        return self._start(args)

    def task_status(self, task_id):
        path, data = _read_job(task_id)
        pid = _pid(path.parent)
        if data.get("state") == "running" and pid and not _alive(pid):
            time.sleep(0.3)  # 終了直前の書き込みを待つ
            _, data = _read_job(task_id)
            if data.get("state") == "running":
                data["state"] = "failed"
                data.setdefault("error", "プロセスが結果を書かずに終了しました")
        log = path.parent / "log.txt"
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:] if log.is_file() else []
        return {"task_id": task_id, **{k: v for k, v in data.items() if k != "command"}, "log_tail": tail}

    def cancel(self, task_id):
        path, data = _read_job(task_id)
        pid = _pid(path.parent)
        if data.get("state") != "running" or not pid or not _alive(pid):
            return {"task_id": task_id, "state": data.get("state"), "cancelled": False}
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
        data.update(state="cancelled", updated=time.time())
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return {"task_id": task_id, "state": "cancelled", "cancelled": True}


class ProtocolError(Exception):
    def __init__(self, code, message):
        self.code, self.message = code, message


def dispatch(request, server):
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
        raise ProtocolError(-32600, "Invalid JSON-RPC request")
    method, params = request["method"], request.get("params", {})
    if not isinstance(params, dict):
        raise ProtocolError(-32602, "params must be an object")
    if method.startswith("notifications/"):
        return None
    if method == "initialize":
        requested = params.get("protocolVersion")
        return {"protocolVersion": requested if requested in PROTOCOLS else "2025-03-26",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "soundsep", "version": "1.0.0"},
                "instructions": "ローカルの Sound Separator。分離は task_id をすぐ返すので soundsep_task_status で待つ。ジョブは 1 本ずつ。モデルは取得済みのものだけを使う。"}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method != "tools/call":
        raise ProtocolError(-32601, "Method not found")
    functions = {"soundsep_status": status, "soundsep_models": models,
                 "soundsep_separate": server.separate, "soundsep_subseparate": server.subseparate, "soundsep_midi": server.midi,
                 "soundsep_task_status": server.task_status, "soundsep_cancel": server.cancel}
    name, args = params.get("name"), params.get("arguments", {})
    if name not in functions:
        raise ProtocolError(-32602, "Unknown tool")
    if not isinstance(args, dict):
        raise ProtocolError(-32602, "arguments must be an object")
    try:
        value = functions[name](**args)
        return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, allow_nan=False)}], "isError": False}
    except (ControlError, OSError, ValueError, KeyError, TypeError) as exc:
        return {"content": [{"type": "text", "text": str(exc)}], "isError": True}


def serve(server):
    for line in sys.stdin:
        request_id, has_id = None, True
        try:
            if len(line) > 1024 * 1024:
                raise ProtocolError(-32600, "Request exceeds 1 MiB")
            try:
                request = json.loads(line)
            except ValueError:
                raise ProtocolError(-32700, "Invalid JSON")
            if isinstance(request, dict):
                request_id, has_id = request.get("id"), "id" in request
            result = dispatch(request, server)
            if not has_id:
                continue
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
        except ProtocolError as exc:
            if not has_id:
                continue
            response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": exc.code, "message": exc.message}}
        except Exception as exc:
            if not has_id:
                continue
            response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    serve(Server())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
