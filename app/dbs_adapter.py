# coding: utf-8
"""公式 DBS-TIGER-DnR のローカル推論。モデルキャッシュは Engine が管理する。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import uuid

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REVISION = "22da7d4cd3ea63f9365181008bff1c0d2ab6fb2b"
MODEL_URL = f"https://huggingface.co/JusperLee/DBS-TIGER-DnR/resolve/{REVISION}/"
MODEL_FILES = {
    "config": {
        "file": "dbs_tiger_dnr_config.json",
        "url": MODEL_URL + "config.json",
        "size": 957,
        "sha256": "c7f7893323e78f9b99747a13090558a13c4f4b77383a38acb9979709fd04d1d8",
    },
    "weights": {
        "file": "dbs_tiger_dnr.safetensors",
        "url": MODEL_URL + "model.safetensors",
        "size": 4111836,
        "sha256": "3d8205743f2a6f5a6704f5d4104d7cf5bf7bf02dab2fd728da31f4c69225cadb",
    },
}
SAMPLE_RATE = 44100


def downloaded() -> bool:
    """一覧表示用。重みをロードせず、必要ファイルの存在とサイズを調べる。"""
    return all(
        (MODELS_DIR / spec["file"]).is_file()
        and (MODELS_DIR / spec["file"]).stat().st_size == spec["size"]
        for spec in MODEL_FILES.values()
    )


def _checked_file(spec, progress_cb=None):
    """固定配布元から一時ファイルへ取得し、SHA-256 検証後だけ公開する。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / spec["file"]
    if target.is_file():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if target.stat().st_size == spec["size"] and digest == spec["sha256"]:
            return target
        raise RuntimeError(f"DBS モデルが破損しています。{target.name} を確認してください")
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex[:8] + ".partial")
    try:
        if progress_cb:
            progress_cb(0, "DBS-TIGER モデルをダウンロード中... (0%)")
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "SoundSeparator/1.0"})
        received, last_report = 0, 0.0
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            while True:
                block = response.read(256 * 1024)
                if not block:
                    break
                stream.write(block)
                digest.update(block)
                received += len(block)
                now = time.monotonic()
                if progress_cb and now - last_report >= 0.25:
                    percent = min(100, int(100 * received / spec["size"]))
                    progress_cb(0, f"DBS-TIGER モデルをダウンロード中... ({percent}%)")
                    last_report = now
        if received != spec["size"] or digest.hexdigest() != spec["sha256"]:
            raise RuntimeError("DBS モデルのダウンロード検証に失敗しました。再試行してください")
        os.replace(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def load_model(device="cpu", progress_cb=None):
    """認証不要の固定重みを読み込む。モデルは呼び出し側で保持する。"""
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("DBS の実行には safetensors が必要です。setup.bat で依存を更新してください") from error
    config_path = _checked_file(MODEL_FILES["config"], progress_cb)
    weight_path = _checked_file(MODEL_FILES["weights"], progress_cb)
    if progress_cb:
        progress_cb(0, "DBS-TIGER モデルを読み込み中...")
    engine_path = str(BASE_DIR / "engine")
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    from dbs.models import DBSTIGER

    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    model = DBSTIGER(**config)
    model.load_state_dict(load_file(str(weight_path), device="cpu"), strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def separate(audio, quality="standard", device="cpu", progress_cb=None, model=None):
    """44.1kHz (ch,n) → モノラルの dialog / effect / music (1,n)。

    ステレオ入力は平均してモノラルへ変換する。GPU上には6秒分だけ載せ、
    結果とoverlap-addの蓄積はCPU上で行う。品質は窓の重なり量を調整する。
    """
    if quality not in ("fast", "standard", "high"):
        raise ValueError("品質は fast / standard / high を指定してください")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] not in (1, 2) or audio.shape[1] == 0:
        raise ValueError("DBS の入力は (1または2チャンネル, サンプル数) が必要です")
    if not np.isfinite(audio).all():
        raise ValueError("音声に無効な値が含まれています")
    if model is None:
        model = load_model(device=device, progress_cb=progress_cb)
    mono = audio.mean(axis=0, dtype=np.float32)
    length = mono.shape[0]
    chunk = 6 * SAMPLE_RATE
    hop = int({"fast": 6, "standard": 3, "high": 1.5}[quality] * SAMPLE_RATE)
    starts = []
    start = 0
    while True:
        starts.append(start)
        if start + chunk >= length:
            break
        start += hop
    output = np.zeros((3, length), dtype=np.float32)
    weight = np.zeros(length, dtype=np.float32)
    for index, start in enumerate(starts):
        valid = min(chunk, length - start)
        piece = torch.from_numpy(mono[start:start + valid].copy()).view(1, 1, -1).to(device)
        # 公式 chunked 推論と同じ短音声・末尾のゼロ埋め条件。
        if valid < chunk and (length > chunk or length <= chunk // 2):
            piece = F.pad(piece, (0, chunk - valid))
        estimate = model(piece)
        if isinstance(estimate, (tuple, list)):
            estimate = estimate[0]
        estimate = estimate[0, :, :valid].float().cpu().numpy()
        if estimate.shape != (3, valid) or not np.isfinite(estimate).all():
            raise RuntimeError("DBS が正しい形の音声を返しませんでした")
        output[:, start:start + valid] += estimate
        weight[start:start + valid] += 1
        if progress_cb:
            percent = int(97 * (index + 1) / len(starts))
            progress_cb(percent, f"分離処理中 (DBS-TIGER・モノラル)... ({percent}%)")
        del piece, estimate
    output /= weight
    return {"dialog": output[2:3], "effect": output[1:2], "music": output[0:1]}
