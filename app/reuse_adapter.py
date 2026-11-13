"""NVIDIA NemotronLabs RE-USE のローカル音声強調。

重みは利用者が別途ダウンロードする。推論時のネットワーク接続は行わない。
ステレオはモノラルへ平均してから処理する。話者や歌の完全な保持は保証しない。
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from reuse_runtime.generator_SEMamba_time_d4 import SEMamba
from reuse_runtime.stfts import mag_phase_stft, mag_phase_istft

from i18n import LocalizedFileNotFoundError, LocalizedRuntimeError, LocalizedValueError, Status

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "nemotron-reuse" / "model.safetensors"
CONFIG = Path(__file__).resolve().parent / "reuse_runtime" / "config.json"
MODEL_BYTES = 38592940
MODEL_SHA256 = "06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1"
MODEL_URL = ("https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/resolve/"
             "31f82f2efad1095034492f236e8dac9239ba713b/model.safetensors")


def downloaded(weights_path=DEFAULT_WEIGHTS):
    """取得済みかをサイズで簡易確認する（取得時にSHA-256を検証すること）。"""
    path = Path(weights_path)
    return path.is_file() and path.stat().st_size == MODEL_BYTES


def load_model(weights_path=DEFAULT_WEIGHTS, device=None):
    path = Path(weights_path)
    if not downloaded(path):
        raise LocalizedFileNotFoundError(
            "error.model_missing_nemotron_reuse",
            "RE-USEのモデルがありません。モデル管理から明示的にダウンロードしてください。")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    model = SEMamba(config)
    # safetensors はpickleを実行しない。ローカルファイルだけを読み込む。
    model.load_state_dict(load_file(os.fspath(path), device="cpu"), strict=True)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), config, device


def _even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def _run_chunk(model, config, samples, sample_rate):
    stft = config["stft_cfg"]
    ratio = stft["sampling_rate"]
    n_fft = _even(stft["n_fft"] * sample_rate // ratio)
    hop = _even(stft["hop_size"] * sample_rate // ratio)
    win = _even(stft["win_size"] * sample_rate // ratio)
    compress = config["model_cfg"]["compress_factor"]
    original_length = samples.shape[-1]
    # center=True の reflect padding に必要な最低長。
    if original_length <= n_fft // 2:
        samples = F.pad(samples, (0, n_fft // 2 + 1 - original_length))
    mag, phase, _ = mag_phase_stft(samples, n_fft, hop, win,
                                   compress_factor=compress, center=True, addeps=False)
    out_mag, out_phase, _ = model(mag, phase)
    # NVIDIA公式推論の sweep artifact 対策。
    zero_portion = (torch.expm1(F.relu(out_mag)) == 0).float().mean(dim=1)
    out_mag = out_mag.masked_fill((zero_portion > 0.5).unsqueeze(1), 0)
    output = mag_phase_istft(out_mag, out_phase, n_fft, hop, win,
                             compress_factor=compress, length=samples.shape[-1])
    if output.shape[-1] < original_length:
        output = F.pad(output, (0, original_length - output.shape[-1]))
    return output[..., :original_length]


def enhance(audio, sample_rate=44100, weights_path=DEFAULT_WEIGHTS, progress_cb=None,
            model_bundle=None, chunk_seconds=2.0, overlap=0.5):
    """(channels, samples) のfloat配列を受け、(1, samples) を返す。"""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] not in (1, 2) or audio.shape[1] == 0:
        raise LocalizedValueError("error.reuse_bad_input", "モノラルまたはステレオの音声が必要です")
    if not np.isfinite(audio).all():
        raise LocalizedValueError("error.reuse_non_finite", "音声に無効な値が含まれています")
    if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 48000:
        raise LocalizedValueError("error.reuse_bad_rate", "RE-USEは8〜48kHzの音声に対応しています")
    if not 0 <= overlap < 1 or not 0.1 <= chunk_seconds <= 30:
        raise LocalizedValueError("error.reuse_bad_chunks", "分割処理の設定が不正です")
    if model_bundle is None:
        model_bundle = load_model(weights_path)
    model, config, device = model_bundle
    mono = torch.from_numpy(audio.mean(axis=0, keepdims=True)).to(device)
    length = mono.shape[-1]
    chunk = max(1, int(chunk_seconds * sample_rate))
    stride = max(1, int(chunk * (1 - overlap)))
    result = torch.zeros_like(mono)
    weights = torch.zeros_like(mono)
    starts = list(range(0, max(1, length - chunk + 1), stride))
    if starts[-1] + chunk < length:
        starts.append(max(0, length - chunk))
    with torch.inference_mode():
        for i, start in enumerate(starts):
            end = min(length, start + chunk)
            enhanced = _run_chunk(model, config, mono[:, start:end], sample_rate)
            # 両端でもゼロ除算しないよう、微小な下限を設ける。
            window = torch.hann_window(end - start, device=device, periodic=False).clamp_min(1e-5)
            result[:, start:end] += enhanced * window
            weights[:, start:end] += window
            if progress_cb:
                progress_cb(int((i + 1) * 100 / len(starts)), Status("status.reuse_enhancing", "RE-USEで音声を補正中"))
    result = (result / weights.clamp_min(1e-5)).cpu().numpy()
    if not np.isfinite(result).all():
        raise LocalizedRuntimeError("error.reuse_bad_output", "RE-USEの出力に無効な値が含まれています")
    return result.astype(np.float32, copy=False)
