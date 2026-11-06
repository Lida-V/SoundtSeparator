# coding: utf-8
"""モデル一覧と明示的な取得。推論時にネットワークへアクセスしない。"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import threading
import urllib.request
import uuid

from separator import (MODELS_DIR, BANDIT_CKPT, BSROFORMER_CKPT, MELROFORMER_CKPT,
                       MEGA_CKPT, DEREVERB_CKPT, KARAOKE_CKPT,
                       DRUMSEP_CKPT_URL, DRUMSEP_CKPT_FILE, CDX23_URL, CDX23_FILES)
import dbs_adapter

ROOT = Path(MODELS_DIR)
DIAR_REV = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
DIAR_URL = f"https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/{DIAR_REV}/"
HF_DIAR = "https://huggingface.co/nvidia/Nemotron-3-Diarization"
HF_REUSE = "https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE"
REUSE_REV = "31f82f2efad1095034492f236e8dac9239ba713b"
REUSE_URL = HF_REUSE + "/resolve/" + REUSE_REV + "/"


def _file(path, url, *, sha256=None, size=None):
    return {"path": Path(path), "url": url, "sha256": sha256, "size": size}


def _model_files(spec):
    result = [_file(ROOT / spec["file"], spec["url"], sha256=spec.get("sha256"))]
    if spec.get("config_url"):
        result.append(_file(spec["config"], spec["config_url"]))
    return result


def htdemucs_path():
    local = ROOT / "5c90dfd2-34c22ccb.th"
    if local.is_file() and local.stat().st_size > 0:
        return local
    # 旧バージョンがtorch.hubへ取得した重みもそのまま利用する。
    import torch
    legacy = Path(torch.hub.get_dir()) / "checkpoints" / local.name
    return legacy if legacy.is_file() and legacy.stat().st_size > 0 else None


def whisper_path(name):
    target = ROOT / "whisper" / f"{name}.pt"
    if target.is_file() and target.stat().st_size > 0:
        return target
    cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper" / f"{name}.pt"
    return cache if cache.is_file() and cache.stat().st_size > 0 else None


def _whisper_file(name):
    hashes = {"small": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794", "medium": "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1", "large-v3": "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb"}
    url = f"https://openaipublic.azureedge.net/main/whisper/models/{hashes[name]}/{name}.pt"
    return _file(ROOT / "whisper" / f"{name}.pt", url, sha256=url.split("/")[-2])


def _group(id, label, description, files=(), **kwargs):
    return {"id": id, "label": label, "description": description, "files": list(files), **kwargs}


def _groups():
    diar_files = [_file(ROOT / "nemotron-3-diarization" / name, DIAR_URL + name,
                        sha256=("c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6" if name == "model.safetensors" else None),
                        size=(396954592 if name == "model.safetensors" else None))
                  for name in ("config.json", "processor_config.json", "model.safetensors")]
    dbs_files = [_file(ROOT / s["file"], s["url"], sha256=s["sha256"], size=s["size"]) for s in dbs_adapter.MODEL_FILES.values()]
    demucs = _file(ROOT / "5c90dfd2-34c22ccb.th", "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th")
    groups = [
        _group("bandit", "BandIt Plus", "声・音楽・効果音の分離。", _model_files(BANDIT_CKPT), download_mb=149, source_url="https://github.com/ZFTurbo/Music-Source-Separation-Training"),
        _group("dbs_dnr", "DBS-TIGER DnR", "声・音楽・効果音をモノラル分離。", dbs_files, download_mb=5, source_url="https://huggingface.co/JusperLee/DBS-TIGER-DnR"),
        _group("bs_roformer", "BS-RoFormer", "ボーカルと伴奏を分離。6ステムではHTDemucsも必要。", _model_files(BSROFORMER_CKPT), download_mb=640, source_url="https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md"),
        _group("mel_band_roformer", "MelBand RoFormer", "ボーカルと伴奏を分離。6ステムではHTDemucsも必要。", _model_files(MELROFORMER_CKPT), download_mb=913, source_url="https://huggingface.co/KimberleyJSN/melbandroformer"),
        _group("mega_53", "MVSep Mega 53 v1", "53の重複するカテゴリから対象を抽出。", _model_files(MEGA_CKPT), download_mb=1369, source_url="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21"),
        _group("htdemucs_6s", "HTDemucs 6ステム", "音楽6ステムでRoFormerと併用。", [demucs], download_mb=55, source_url="https://github.com/facebookresearch/demucs"),
        _group("cdx23", "CDX23", "BandIt高品質で使う3モデル。", [_file(ROOT / f, CDX23_URL + f) for f in CDX23_FILES], download_mb=162, source_url="https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing"),
        _group("drumsep", "DrumSep", "ドラムをキック・スネア等に細分化。", [_file(ROOT / DRUMSEP_CKPT_FILE, DRUMSEP_CKPT_URL)], download_mb=168, source_url="https://github.com/ZFTurbo/Music-Source-Separation-Training"),
        _group("karaoke", "Karaoke MelBand RoFormer", "リード・バックボーカルの推定。", _model_files(KARAOKE_CKPT), download_mb=913, source_url="https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models"),
        _group("dereverb", "MelBand RoFormer Dereverb", "残響除去。", _model_files(DEREVERB_CKPT), download_mb=913, source_url="https://huggingface.co/anvuew/dereverb_mel_band_roformer"),
        _group("nemotron_diarization", "Nemotron-3-Diarization", "最大8話者の話者区間を推定。", diar_files, download_mb=397, source_url=HF_DIAR),
        _group("nemotron_reuse", "NemotronLabs RE-USE", "音声のノイズや残響を抑えるモノラル音声強調。", [
            _file(ROOT / "nemotron-reuse" / "config.json", REUSE_URL + "config.json", size=830),
            _file(ROOT / "nemotron-reuse" / "model.safetensors", REUSE_URL + "model.safetensors", size=38592940, sha256="06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1"),
        ], download_mb=39, source_url=HF_REUSE),
    ]
    for size, name, mb in (("small", "small", 484), ("medium", "medium", 1530), ("large", "large-v3", 3090)):
        groups.append(_group("whisper_" + size, "Whisper " + name, "文字起こし用。", [_whisper_file(name)], download_mb=mb, source_url="https://github.com/openai/whisper", whisper_name=name))
    # these packages contain their own model weights, installed as part of setup.
    groups += [
        _group("resemblyzer", "Resemblyzer", "従来の話者推定。重みはライブラリに同梱。", downloadable=False, bundled_module="resemblyzer", source_url="https://github.com/resemble-ai/Resemblyzer"),
        _group("basic_pitch", "Basic Pitch", "MIDI採譜。重みはライブラリに同梱。", downloadable=False, bundled_module="basic_pitch", source_url="https://github.com/spotify/basic-pitch"),
    ]
    # 数値は公開元で確認できたものだけを記載し、未測定の値を最低要件にしない。
    common = "共通：Windows・Python 3.11の本アプリ環境。必要メモリ容量は未計測。"
    general = "CPUまたはCUDA対応NVIDIA GPU。GPUを推奨。最低VRAMは未公表。"
    for group in groups:
        ident = group["id"]
        if ident == "mega_53":
            detail = "推奨：NVIDIA GPU、VRAM 16GB以上（モデル提供元の推奨）。CPU使用時は非常に低速。"
        elif ident == "nemotron_reuse":
            detail = ("公式実装：Linux・NVIDIA Ampere。アプリ独自のPyTorch実装はWindows / RTX 4090で合成音のGPU推論を確認。"
                      "CPU推論および最低RAM・VRAMは未検証。モノラル出力。")
        elif ident == "nemotron_diarization":
            detail = ("必要：Transformers 5.18.0と16kHzモノラル入力（自動変換）。"
                      "Windows / RTX 4090で実音声のGPU推論を確認。CPUおよび最低RAM・VRAMは未検証。")
        elif ident.startswith("whisper_"):
            amounts = {"whisper_small": "約2GB", "whisper_medium": "約5GB", "whisper_large": "約10GB"}
            detail = f"公式のVRAM目安：{amounts[ident]}（GPU、条件により変動）。CPU実行も可能。"
        elif ident == "resemblyzer":
            detail = "CPUまたはCUDA対応NVIDIA GPU。必要RAM・VRAMは未計測。"
        elif ident == "basic_pitch":
            detail = "ONNX RuntimeでCPU実行。必要RAMは未計測。"
        else:
            detail = general
        group["requirements"] = [common, detail]
    return groups


def _present(spec):
    path = spec["path"]
    return path.is_file() and (path.stat().st_size == spec["size"] if spec.get("size") else path.stat().st_size > 0)


def available(group):
    if group.get("bundled_module"):
        return importlib.util.find_spec(group["bundled_module"]) is not None
    if group["id"] == "htdemucs_6s" and htdemucs_path():
        return True
    if group.get("whisper_name") and whisper_path(group["whisper_name"]):
        return True
    return bool(group["files"]) and all(_present(spec) for spec in group["files"])


def catalog():
    return [{**{k: v for k, v in g.items() if k not in {"files", "whisper_name", "bundled_module"}},
             "downloaded": available(g), "downloadable": g.get("downloadable", True)} for g in _groups()]


def _fetch(spec, callback):
    target = spec["path"]
    if _present(spec):
        return
    if target.exists():
        raise RuntimeError(f"既存ファイルのサイズが不正です: {target.name}。内容を確認してください")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + "." + uuid.uuid4().hex + ".partial")
    try:
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "SoundSeparator/1.0"})
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as out:
            total = spec.get("size") or int(response.headers.get("Content-Length", 0))
            count = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                digest.update(block)
                count += len(block)
                callback(count, total)
        if count == 0 or (total and count != total):
            raise RuntimeError(f"{target.name} のダウンロードが途中で終了しました")
        if spec.get("sha256") and digest.hexdigest() != spec["sha256"]:
            raise RuntimeError(f"{target.name} のSHA-256検証に失敗しました")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def download(group_id, progress):
    group = next((g for g in _groups() if g["id"] == group_id), None)
    if group is None:
        raise ValueError("モデルが見つかりません")
    if not group.get("downloadable", True):
        raise ValueError("このモデルはセットアップに同梱されています")
    if available(group):
        progress(100, "取得済みです")
        return
    files = group["files"]
    for index, spec in enumerate(files):
        def cb(count, total):
            fraction = min(count / total, 1) if total else 0
            progress(int((index + fraction) * 100 / len(files)), f"{spec['path'].name} を取得中 ({count // 1048576} MB)")
        _fetch(spec, cb)
        progress(int((index + 1) * 100 / len(files)), f"{spec['path'].name} を確認しました")
