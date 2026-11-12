"""Sound Separator の分離エンジンを、画面なしで呼ぶ CLI（エージェントやスクリプト向け）。

このリポジトリの venv の Python で実行する（setup.bat の後）。サーバー（run.bat）の起動は要らない。
モデルの取得は行わない。未取得のモデルが要るときはエラーで止まる（取得はアプリ画面の「AIモデルの管理」）。

  .venv/Scripts/python.exe -X utf8 tools/soundsep_control.py status
  .venv/Scripts/python.exe -X utf8 tools/soundsep_control.py models
  .venv/Scripts/python.exe -X utf8 tools/soundsep_control.py separate --input <音声> --out <新しいフォルダ> [--mode music] [--quality high] [--exact]
  .venv/Scripts/python.exe -X utf8 tools/soundsep_control.py subseparate --input <WAV> --out <新しいフォルダ> --kind vocals|drums
  .venv/Scripts/python.exe -X utf8 tools/soundsep_control.py midi --input <WAV> --out <出力.mid>

出力は入力と同じサンプルレート・長さの 32bit float WAV と manifest.json。既存のファイルは上書きしない。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # このファイルの位置から求める（絶対パスを持たない）
MODEL_RATE = 44100
QUALITIES = ("fast", "standard", "high")
MODES = ("music", "vocals", "dme", "mega")
SUBSEP_KINDS = ("vocals", "drums")


class ControlError(Exception):
    pass


def _app_path():
    """app/ を import 先に加える。"""
    app_dir = str(APP_ROOT / "app")
    if not (APP_ROOT / "app" / "separator.py").is_file():
        raise ControlError(f"分離エンジンが見つかりません: {APP_ROOT}")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)


def _engine():
    _app_path()
    import separator  # noqa: PLC0415
    return separator, separator.Engine()


class Progress:
    """進捗を標準エラーと、指定があればジョブファイルへ書く。"""

    def __init__(self, job_file=None):
        self.job_file = Path(job_file) if job_file else None
        self.last = None

    def __call__(self, pct, message=""):
        state = (int(pct), str(message))
        if state == self.last:
            return
        self.last = state
        print(f"[{state[0]:3d}%] {state[1]}", file=sys.stderr, flush=True)
        self.update(progress=state[0], message=state[1])

    def update(self, **fields):
        if not self.job_file:
            return
        try:
            data = json.loads(self.job_file.read_text(encoding="utf-8")) if self.job_file.is_file() else {}
        except (OSError, ValueError):
            data = {}
        data.update(fields, updated=time.time())
        tmp = self.job_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        for _ in range(5):  # 読み手が開いている瞬間は置き換えに失敗することがある
            try:
                os.replace(tmp, self.job_file)
                return
            except OSError:
                time.sleep(0.05)


def _resample(audio, source_rate, target_rate):
    """audio: (samples, channels) float32。"""
    import numpy as np
    if source_rate == target_rate:
        return audio
    try:
        import soxr
        return soxr.resample(audio, source_rate, target_rate, quality="VHQ").astype(np.float32)
    except ImportError:
        from math import gcd
        from scipy.signal import resample_poly
        factor = gcd(int(source_rate), int(target_rate))
        return resample_poly(audio, target_rate // factor, source_rate // factor, axis=0).astype(np.float32)


def _fit(audio, samples):
    import numpy as np
    if audio.shape[0] > samples:
        return audio[:samples]
    if audio.shape[0] < samples:
        return np.pad(audio, ((0, samples - audio.shape[0]), (0, 0)))
    return audio


def _db(value):
    import math
    return round(20 * math.log10(value), 2) if value > 0 else None


def _measure(audio):
    import numpy as np
    return {"peak_dbfs": _db(float(np.abs(audio).max())), "rms_dbfs": _db(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))))}


def _read(path):
    import soundfile as sf
    path = Path(path)
    if not path.is_file():
        raise ControlError(f"入力が見つかりません: {path}")
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[0] < rate // 2:
        raise ControlError("0.5秒以上の音声が必要です")
    return audio, rate


def _prepare_out(out_dir, names):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clash = [name for name in names if (out_dir / name).exists()]
    if clash:
        raise ControlError(f"出力先に同名のファイルがあります（上書きしません）: {', '.join(clash)}")
    return out_dir


def _write(out_dir, name, audio, rate):
    import soundfile as sf
    path = out_dir / f"{name}.wav"
    sf.write(str(path), audio, rate, subtype="FLOAT")
    return {"file": path.name, "samples": int(audio.shape[0]), "channels": int(audio.shape[1]), **_measure(audio)}


def _to_model(audio, rate):
    import numpy as np
    return np.ascontiguousarray(_resample(audio, rate, MODEL_RATE).T)


def _from_model(data, rate, samples):
    return _fit(_resample(data.T, MODEL_RATE, rate), samples)


def status():
    root = APP_ROOT
    info = {"root": str(root), "python": sys.executable}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.cuda.is_available())
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            info.update(gpu=torch.cuda.get_device_name(0), vram_free_gb=round(free / 1024**3, 2), vram_total_gb=round(total / 1024**3, 2))
    except ImportError as exc:
        info["torch_error"] = str(exc)
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        info["app_commit"] = commit.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["app_commit"] = None
    info["models"] = {entry["id"]: entry["downloaded"] for entry in models()}
    return info


def models():
    _app_path()
    from model_downloads import catalog  # noqa: PLC0415
    return [{"id": e["id"], "label": e.get("label"), "downloaded": bool(e.get("downloaded"))} for e in catalog()]


def separate(input_path, out_dir, mode="music", quality="high", model_id=None, stems=None, exact=False,
             job_file=None):
    """音源をステムへ分ける。exact=True は「元音源 − 全ステムの和」を other へ足し、足し戻しを一致させる。"""
    import numpy as np
    if mode not in MODES or quality not in QUALITIES:
        raise ControlError("mode または quality の指定が正しくありません")
    progress = Progress(job_file)
    audio, rate = _read(input_path)
    separator, engine = _engine()
    try:
        model_id, specs = separator.validate_request(mode, model_id, stems)
    except ValueError as exc:
        raise ControlError(str(exc)) from exc
    names = [item["name"] for item in specs]
    if exact and (mode != "music" or stems or "other" not in names):
        raise ControlError("exact は mode=music で全ステムを出すときだけ使えます")
    out_dir = _prepare_out(out_dir, [f"{n}.wav" for n in names] + ["manifest.json"])
    started = time.time()
    try:
        result = engine.separate(_to_model(audio, rate), mode=mode, quality=quality, progress_cb=progress,
                                 model_id=model_id, selected_stems=stems)
    except (RuntimeError, ValueError) as exc:
        raise ControlError(str(exc)) from exc
    native = {name: _from_model(result[name], rate, audio.shape[0]) for name in names}
    manifest = {"input": str(Path(input_path).resolve()), "sample_rate": rate, "samples": int(audio.shape[0]),
                "mode": mode, "model_id": model_id, "quality": quality, "device": engine.device,
                "model_rate": MODEL_RATE, "resampled": rate != MODEL_RATE}
    if mode in ("music", "vocals", "dme") and not stems:
        residual = audio - sum(native.values())
        manifest["sum_error_before"] = {"max_abs": float(np.abs(residual).max()), **_measure(residual)}
        if exact:
            native["other"] = native["other"] + residual
            after = audio - sum(native.values())
            manifest["sum_error_after"] = {"max_abs": float(np.abs(after).max())}
            manifest["exact"] = "元音源 − 全ステムの和 を other に加算"
    manifest["stems"] = [_write(out_dir, name, native[name], rate) for name in names]
    manifest["elapsed_seconds"] = round(time.time() - started, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    progress.update(progress=100, message="完了")
    return {"out_dir": str(out_dir), **manifest}


def _window_report(parts, rate, seconds=5.0):
    """窓ごとの各ステムの RMS。リードとバックの入れ替わりを見つけるための材料。"""
    import numpy as np
    size = int(rate * seconds)
    length = next(iter(parts.values())).shape[0]
    rows = []
    for start in range(0, length, size):
        row = {"start_s": round(start / rate, 2)}
        for name, data in parts.items():
            chunk = data[start:start + size].astype(np.float64)
            row[name] = _db(float(np.sqrt(np.mean(chunk ** 2)))) if len(chunk) else None
        rows.append(row)
    return rows


def subseparate(input_path, out_dir, kind, job_file=None):
    """ボーカルをリード/バックへ、ドラムをキック/スネア/シンバル/タムへ分ける。"""
    if kind not in SUBSEP_KINDS:
        raise ControlError("kind は vocals か drums です")
    progress = Progress(job_file)
    audio, rate = _read(input_path)
    names = ["lead", "back"] if kind == "vocals" else ["kick", "snare", "cymbals", "toms"]
    out_dir = _prepare_out(out_dir, [f"{n}.wav" for n in names] + ["manifest.json"])
    _, engine = _engine()
    started = time.time()
    model_audio = _to_model(audio, rate)
    if model_audio.shape[0] == 1:
        import numpy as np
        model_audio = np.repeat(model_audio, 2, axis=0)
    try:
        result = engine.subseparate(model_audio, kind, progress_cb=progress)
    except (RuntimeError, ValueError) as exc:
        raise ControlError(str(exc)) from exc
    native = {}
    for name in names:
        data = _from_model(result[name], rate, audio.shape[0])
        native[name] = data.mean(axis=1, keepdims=True) if audio.shape[1] == 1 else data
    manifest = {"input": str(Path(input_path).resolve()), "sample_rate": rate, "samples": int(audio.shape[0]),
                "kind": kind, "device": engine.device, "resampled": rate != MODEL_RATE}
    if kind == "vocals":
        # リードは「入力 − バック」。入力のレートで引き直し、2本の和を入力に一致させる。
        native["lead"] = audio - native["back"]
        manifest["note"] = "lead と back の割り当ては曲と区間で入れ替わることがある。window_rms で確かめてから使う。"
    manifest["window_rms"] = _window_report(native, rate)
    manifest["stems"] = [_write(out_dir, name, native[name], rate) for name in names]
    manifest["elapsed_seconds"] = round(time.time() - started, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    progress.update(progress=100, message="完了")
    return {"out_dir": str(out_dir), **manifest}


def midi(input_path, out_path, onset=0.5, frame=None, min_note_ms=120.0, job_file=None):
    """basic-pitch で WAV を採譜し、MIDI を書く。和音も拾うが、誤りを含む。"""
    progress = Progress(job_file)
    if not Path(input_path).is_file():
        raise ControlError(f"入力が見つかりません: {input_path}")
    out_path = Path(out_path)
    if out_path.exists():
        raise ControlError(f"出力先に同名のファイルがあります（上書きしません）: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _app_path()
    from midi_export import audio_to_midi_bytes  # noqa: PLC0415
    progress(10, "採譜中 (basic-pitch)...")
    started = time.time()
    data = audio_to_midi_bytes(str(input_path), onset=onset, frame=frame, min_note_ms=min_note_ms)
    out_path.write_bytes(data)
    summary = {"out": str(out_path), "bytes": len(data), "onset": onset, "frame": frame, "min_note_ms": min_note_ms,
               "elapsed_seconds": round(time.time() - started, 1)}
    try:
        import pretty_midi
        notes = [n for inst in pretty_midi.PrettyMIDI(str(out_path)).instruments for n in inst.notes]
        if notes:
            summary.update(notes=len(notes), pitch_min=min(n.pitch for n in notes), pitch_max=max(n.pitch for n in notes),
                           first_s=round(min(n.start for n in notes), 3), last_s=round(max(n.end for n in notes), 3))
        else:
            summary["notes"] = 0
    except ImportError:
        pass
    progress.update(progress=100, message="完了")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-file", help="進捗と結果を書く JSON（MCP が使う）")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("models")
    p = sub.add_parser("separate")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=MODES, default="music")
    p.add_argument("--quality", choices=QUALITIES, default="high")
    p.add_argument("--model")
    p.add_argument("--stems", nargs="+")
    p.add_argument("--exact", action="store_true")
    p = sub.add_parser("subseparate")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kind", choices=SUBSEP_KINDS, required=True)
    p = sub.add_parser("midi")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--onset", type=float, default=0.5)
    p.add_argument("--frame", type=float)
    p.add_argument("--min-note-ms", type=float, default=120.0)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    # 本体や MSST が処理中に標準出力へ書くので、標準出力は結果の JSON だけにする
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    progress = Progress(args.job_file)
    try:
        if args.command == "status":
            result = status()
        elif args.command == "models":
            result = models()
        elif args.command == "separate":
            result = separate(args.input, args.out, args.mode, args.quality, args.model, args.stems, args.exact,
                              args.job_file)
        elif args.command == "subseparate":
            result = subseparate(args.input, args.out, args.kind, args.job_file)
        else:
            result = midi(args.input, args.out, args.onset, args.frame, args.min_note_ms, args.job_file)
    except ControlError as exc:
        progress.update(state="failed", error=str(exc))
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # 想定外の失敗もジョブに残す
        progress.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    if isinstance(result, dict) and "window_rms" in result:
        brief = {k: v for k, v in result.items() if k != "window_rms"}
    else:
        brief = result
    progress.update(state="done", result=brief)
    print(json.dumps(brief, ensure_ascii=False, indent=1), file=real_stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
