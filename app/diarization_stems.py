# coding: utf-8
"""話者区間に基づく音声の時間振り分け。

声が重なる区間は混ざったまま overlap に入り、音響的な話者分離は行わない。
単独話者・重なり・未検出の全出力を加算すると入力になる。
"""
from __future__ import annotations

import math
from itertools import groupby

import numpy as np


def _validate_inputs(audio, sr, turns, fade_ms):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] not in (1, 2) or not len(audio):
        raise ValueError("0秒より長いモノラルまたはステレオ音声が必要です")
    if not np.isfinite(audio).all():
        raise ValueError("音声に無効な数値が含まれています")
    if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("サンプルレートが正しくありません")
    if isinstance(fade_ms, bool) or not isinstance(fade_ms, (int, float)) or not math.isfinite(fade_ms) or not 0 <= fade_ms <= 1000:
        raise ValueError("フェード長は0〜1000ミリ秒で指定してください")
    if not isinstance(turns, (list, tuple)):
        raise ValueError("話者区間の形式が正しくありません")
    return audio


def _intervals(n_samples, sr, turns):
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("話者区間の形式が正しくありません")
        speaker = turn.get("speaker")
        if not isinstance(speaker, str) or len(speaker) != 3 or speaker[:2] != "話者" or speaker[2] not in "ABCDEFGH":
            raise ValueError("話者名は話者A〜話者Hで指定してください")
        try:
            start, end = float(turn["start"]), float(turn["end"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("話者区間の時刻が正しくありません") from exc
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("話者区間の時刻が正しくありません")
        # 時刻を先に音声長内へ制限し、極端な値からの巨大整数化を防ぐ。
        duration = n_samples / sr
        start = min(max(start, 0.0), duration)
        end = min(max(end, 0.0), duration)
        begin = min(n_samples, max(0, round(start * sr)))
        finish = min(n_samples, max(0, round(end * sr)))
        if begin < finish:
            yield begin, finish, ord(speaker[2]) - ord("A")


def _classify_samples(n_samples, sr, turns):
    # 8話者をビットとして扱い、同一話者の区間重複を1人にまとめる。
    active = np.zeros(n_samples, dtype=np.uint8)
    for begin, finish, index in _intervals(n_samples, sr, turns):
        active[begin:finish] |= np.uint8(1 << index)

    # 0=未検出、1..8=単独話者、9=複数話者。lookup表でサンプル毎の分類を省メモリ化。
    lut = np.full(256, 9, dtype=np.uint8)
    lut[0] = 0
    for index in range(8):
        lut[1 << index] = index + 1
    return lut[active]


def summarize_diarization_activity(n_samples, sr, turns):
    """フェード前の区間分類を返す。重なりあり判定は音量ではなく話者数による。"""
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("0秒より長い音声が必要です")
    if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("サンプルレートが正しくありません")
    if not isinstance(turns, (list, tuple)):
        raise ValueError("話者区間の形式が正しくありません")
    events = []
    for start, end, speaker in _intervals(n_samples, sr, turns):
        events.append((start, speaker, 1))
        events.append((end, speaker, -1))
    events.sort(key=lambda event: event[0])
    counts = [0] * 8
    summary = {"overlap_samples": 0, "residual_samples": 0, "solo_samples": {}}

    def accumulate(width):
        active = [i for i, count in enumerate(counts) if count > 0]
        if not active:
            summary["residual_samples"] += width
        elif len(active) > 1:
            summary["overlap_samples"] += width
        else:
            key = "speaker_" + chr(ord("a") + active[0])
            summary["solo_samples"][key] = summary["solo_samples"].get(key, 0) + width

    previous = 0
    for position, group in groupby(events, key=lambda event: event[0]):
        accumulate(position - previous)
        for _, speaker, delta in group:
            counts[speaker] += delta
        previous = position
    accumulate(n_samples - previous)
    return summary


def iter_diarization_stems(audio, sr, turns, fade_ms=10):
    """入力と同じ尺・チャンネル数の ``(名前, 音声)`` を順番に返す。

    ``audio`` は (samples, channels)、``turns`` は ``start``, ``end`` 秒と
    ``speaker`` (話者A〜話者H) を持つ辞書のリスト。単独区間の存在する話者だけを
    speaker_a〜speaker_h とし、overlap と residual は常に返す。

    隣り合う区間だけを短くクロスフェードし、ステムの和を入力と一致させる。
    前の結果を保存して解放してから次を取得すると長尺時のメモリを抑えられる。
    """
    audio = _validate_inputs(audio, sr, turns, fade_ms)
    n_samples = len(audio)
    classes = _classify_samples(n_samples, sr, turns)
    present = set(np.unique(classes).tolist())
    boundaries = np.flatnonzero(classes[1:] != classes[:-1]) + 1
    # 各境界の左右は隣の境界の中点以内に制限し、ごく短い区間でもフェード同士を重ねない。
    fade_samples = int(round(float(fade_ms) * sr / 1000.0))
    half = fade_samples // 2
    transitions = []
    if half:
        for i, boundary in enumerate(boundaries):
            prev_boundary = int(boundaries[i - 1]) if i else 0
            next_boundary = int(boundaries[i + 1]) if i + 1 < len(boundaries) else n_samples
            boundary = int(boundary)
            left = min(half, (boundary - prev_boundary) // 2)
            right = min(half, (next_boundary - boundary) // 2)
            start, end = boundary - left, boundary + right
            if end - start >= 2:
                transitions.append((start, end, int(classes[boundary - 1]), int(classes[boundary])))

    stems = [(f"speaker_{chr(ord('a') + i - 1)}", i) for i in range(1, 9) if i in present]
    stems.extend([("overlap", 9), ("residual", 0)])
    for name, category in stems:
        gain = (classes == category).astype(np.float32)
        for start, end, before, after in transitions:
            if category == before or category == after:
                fade = np.linspace(0.0, 1.0, end - start, dtype=np.float32)
                gain[start:end] = (1.0 - fade) if category == before else fade
        result = np.multiply(audio, gain[:, None], dtype=np.float32)
        yield name, result
        del result, gain
