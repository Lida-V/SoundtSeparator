# coding: utf-8
"""セリフ書き出しエンジン。

Whisper (ローカル) で音声トラックを文字起こしし、セリフ単位で
- 連番WAVクリップ (clips/0001.wav ...)
- SRT字幕 (subtitles.srt)
- TTS学習用テキスト (metadata.csv — LJSpeech形式 `ファイル名|テキスト`)
- アノテーション (annotations.json / annotations.csv — 開始/終了/長さ/テキスト)
を生成して ZIP にまとめる。
"""
import csv
import hashlib
import json
import math
import os
import threading
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODELS = {"small": "small", "medium": "medium", "large": "large-v3"}
MODEL_ROOT = Path(__file__).resolve().parent.parent / "models"
CLIP_PAD_SECONDS = 0.15  # セリフ前後の余白
FADE_SECONDS = 0.01


class Transcriber:
    def __init__(self):
        self._models = {}
        self._lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_model(self, size, progress_cb=None):
        import whisper

        name = MODELS.get(size, "medium")
        if name not in self._models:
            # 名前で load_model を呼ぶとネットワークから暗黙に取得される。
            # 管理画面か既存 Whisper キャッシュにある検証済みファイルだけを使う。
            expected = whisper._MODELS[name].split("/")[-2]
            cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
            candidates = [MODEL_ROOT / "whisper" / f"{name}.pt",
                          cache / "whisper" / f"{name}.pt"]
            model_path = None
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                digest = hashlib.sha256()
                with candidate.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() == expected:
                    model_path = candidate
                    break
            if model_path is None:
                raise RuntimeError(f"Whisper {name} の検証済みモデルがありません。モデル管理からダウンロードしてください。")
            if progress_cb:
                progress_cb(0, f"音声認識モデルを読込中 (Whisper {name})...")
            self._models[name] = whisper.load_model(str(model_path), device=self.device)
        return self._models[name]

    def transcribe(self, audio, sr, model_size="medium", language=None, progress_cb=None):
        """audio: (samples, channels) float32
        返り値: (segments [{start,end,text}], words [{start,end,word}], 言語)"""
        with self._lock:
            model = self._get_model(model_size, progress_cb)
            mono = audio.mean(axis=1).astype(np.float32)

            # Whisper は 16kHz 前提。librosa より軽い線形補間で十分 (認識用途)
            import whisper.audio as wa

            if sr != wa.SAMPLE_RATE:
                n_out = int(len(mono) * wa.SAMPLE_RATE / sr)
                x_old = np.linspace(0, 1, len(mono), dtype=np.float64)
                x_new = np.linspace(0, 1, n_out, dtype=np.float64)
                mono16 = np.interp(x_new, x_old, mono).astype(np.float32)
            else:
                mono16 = mono

            if progress_cb:
                progress_cb(10, "文字起こし中... (長さによっては数分かかります)")
            result = model.transcribe(
                mono16,
                language=language or None,
                fp16=self.device == "cuda",
                verbose=None,
                word_timestamps=True,
            )

        segments = []
        words = []
        for seg in result.get("segments", []):
            text = seg["text"].strip()
            if text:
                segments.append({
                    "start": round(float(seg["start"]), 3),
                    "end": round(float(seg["end"]), 3),
                    "text": text,
                })
            for w in seg.get("words", []):
                words.append({
                    "start": round(float(w["start"]), 3),
                    "end": round(float(w["end"]), 3),
                    "word": w["word"],
                })
        return segments, words, result.get("language", language or "")


_SENTENCE_END = ("。", "．", ".", "!", "?", "！", "？")

_voice_encoder = None
SPEAKER_NAMES = [f"話者{c}" for c in "ABCDEFGH"]


def _assign_diarization_segments(segments, turns):
    """発話区間に最も長く重なる話者を割り当てる。重なりがなければ未割当。"""
    for segment in segments:
        overlaps = {}
        for turn in turns:
            overlap = max(0.0, min(float(segment["end"]), turn["end"]) -
                          max(float(segment["start"]), turn["start"]))
            if overlap:
                overlaps[turn["speaker"]] = overlaps.get(turn["speaker"], 0.0) + overlap
        segment["speaker"] = max(overlaps, key=overlaps.get) if overlaps else ""
    return segments


class NemotronDiarizer:
    """モデル管理で明示取得したローカル重みのみを利用する話者区間推定。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._processor = None
        self._model = None

    def _load(self):
        path = MODEL_ROOT / "nemotron-3-diarization"
        if not all((path / name).is_file() for name in
                   ("config.json", "processor_config.json", "model.safetensors")):
            raise RuntimeError("Nemotron-3-Diarization がありません。モデル管理からダウンロードしてください。")
        if self._model is None:
            try:
                from transformers import AutoModelForAudioFrameClassification, AutoProcessor
                processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
                model = AutoModelForAudioFrameClassification.from_pretrained(str(path), local_files_only=True)
            except ImportError as exc:
                raise RuntimeError("Nemotron の実行環境がありません。セットアップ手順で追加してください。") from exc
            self._processor = processor
            self._model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        return self._processor, self._model

    def diarize(self, audio, sr, progress_cb=None):
        """音声から、重複を保持した話者区間 [{start,end,speaker}] を返す。

        入力は (samples, channels)。モデルへの入力だけを16kHzモノラルに変換し、
        元の音声データには変更を加えない。重複する別話者の区間は統合しない。
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[1] not in (1, 2) or not np.isfinite(audio).all():
            raise ValueError("モノラルまたはステレオの正常な音声が必要です")
        if not isinstance(sr, (int, np.integer)) or sr <= 0:
            raise ValueError("サンプルレートが正しくありません")
        if len(audio) < sr // 10:
            raise RuntimeError("Nemotron の話者分離には0.1秒以上の音声が必要です。")
        with self._lock:
            if progress_cb:
                progress_cb(0, "Nemotron で話者区間を分析中...")
            processor, model = self._load()
            mono = audio.mean(axis=1)
            rate = processor.feature_extractor.sampling_rate
            if sr != rate:
                import soxr
                mono = soxr.resample(mono, sr, rate, quality="HQ").astype(np.float32)
            if len(mono) < rate // 10:
                raise RuntimeError("Nemotron の話者分離には0.1秒以上の音声が必要です。")
            with torch.inference_mode():
                if len(mono) <= rate * 30:
                    inputs = processor(mono, sampling_rate=rate, return_tensors="pt").to(model.device, dtype=model.dtype)
                    logits = model(**inputs).logits
                    detected = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
                else:
                    # 公式ストリーミングAPI: チャンク間で speaker_cache を引き継ぎ、
                    # 話者IDを維持する。推論結果はCPUへ退避し、VRAM使用を抑える。
                    processor.set_streaming_mode("low_latency")
                    first_len = processor.num_samples_first_audio_chunk
                    inputs = processor(mono[:first_len], sampling_rate=rate, is_streaming=True,
                                       is_first_audio_chunk=True, return_tensors="pt")
                    outputs = model(**inputs.to(model.device, dtype=model.dtype))
                    cache = outputs.speaker_cache
                    chunks = [outputs.logits.cpu()]
                    frame_idx = processor.num_mel_frames_per_step
                    start = processor.audio_chunk_start(frame_idx)
                    while start + processor.num_samples_per_audio_chunk <= len(mono):
                        end = start + processor.num_samples_per_audio_chunk
                        inputs = processor(mono[start:end], sampling_rate=rate, is_streaming=True,
                                           is_first_audio_chunk=False, return_tensors="pt")
                        outputs = model(**inputs.to(model.device, dtype=model.dtype), speaker_cache=cache)
                        chunks.append(outputs.logits.cpu())
                        cache = outputs.speaker_cache
                        frame_idx += processor.num_mel_frames_per_step
                        start = processor.audio_chunk_start(frame_idx)
                        if progress_cb:
                            progress_cb(min(95, int(95 * start / len(mono))), "Nemotron で話者区間を分析中...")
                    inputs = processor(mono[start:], sampling_rate=rate, is_streaming=True,
                                       is_first_audio_chunk=False, is_last_audio_chunk=True, return_tensors="pt")
                    outputs = model(**inputs.to(model.device, dtype=model.dtype), speaker_cache=cache)
                    chunks.append(outputs.logits.cpu())
                    detected = processor.extract_speaker_dict(torch.cat(chunks, dim=1))[0]
            turns = []
            for item in detected:
                speaker = int(item["Speaker"])
                if 0 <= speaker < len(SPEAKER_NAMES):
                    start, end = float(item["Start"]), float(item["End"])
                    if not math.isfinite(start) or not math.isfinite(end):
                        raise RuntimeError("Nemotron から不正な話者区間が返されました")
                    start, end = max(0.0, start), min(len(audio) / sr, end)
                    if start < end:
                        turns.append({"start": start, "end": end, "speaker": SPEAKER_NAMES[speaker]})
            if progress_cb:
                progress_cb(100, "話者区間の分析が完了しました")
            return turns

    def assign(self, audio, sr, segments, progress_cb=None):
        if not segments:
            return segments

        def report(percent, message):
            if progress_cb:
                progress_cb(80 + int(percent * 0.15), message)

        turns = self.diarize(audio, sr, progress_cb=report)
        return _assign_diarization_segments(segments, turns)


nemotron_diarizer = NemotronDiarizer()


def assign_speakers(audio, sr, segments, num_speakers=0, progress_cb=None):
    """各セグメントに話者ラベルを付与する (resemblyzer埋め込み + 階層クラスタリング)。

    num_speakers: 0=自動判定 (距離しきい値)、2以上=指定人数
    """
    global _voice_encoder
    if len(segments) < 2:
        for s in segments:
            s["speaker"] = SPEAKER_NAMES[0]
        return segments

    import torch
    from resemblyzer import VoiceEncoder
    from scipy.cluster.hierarchy import fcluster, linkage

    if progress_cb:
        progress_cb(80, "話者を分析中...")
    if _voice_encoder is None:
        _voice_encoder = VoiceEncoder("cuda" if torch.cuda.is_available() else "cpu")

    # 16kHz モノラルへ (線形補間で十分)
    mono = audio.mean(axis=1).astype(np.float64)
    n16 = int(len(mono) * 16000 / sr)
    mono16 = np.interp(
        np.linspace(0, 1, n16), np.linspace(0, 1, len(mono)), mono
    ).astype(np.float32)

    embs = []
    for seg in segments:
        s = max(0, int(seg["start"] * 16000))
        e = min(n16, int(seg["end"] * 16000))
        clip = mono16[s:e]
        if len(clip) < 6400:  # 0.4秒未満はパディング
            clip = np.pad(clip, (0, 6400 - len(clip)))
        embs.append(_voice_encoder.embed_utterance(clip))
    embs = np.stack(embs)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9

    z = linkage(embs, method="average", metric="cosine")
    if num_speakers and num_speakers >= 2:
        labels = fcluster(z, t=min(num_speakers, len(segments)), criterion="maxclust")
    else:
        labels = fcluster(z, t=0.25, criterion="distance")  # コサイン距離しきい値

    # 出現順に 話者A, B, ... を割り当て
    order = {}
    for lb in labels:
        if lb not in order:
            order[lb] = SPEAKER_NAMES[min(len(order), len(SPEAKER_NAMES) - 1)]
    for seg, lb in zip(segments, labels):
        seg["speaker"] = order[lb]
    return segments


def segment_words(words, min_len=2.0, max_len=10.0, gap_split=0.6):
    """単語タイムスタンプから、min_len〜max_len 秒に収まるセリフ区間を組み立てる。

    優先順位: 文末記号での区切り > 無音ギャップ > 単語境界での強制分割。
    テキストは単語境界で正しく分割されるため TTS/RVC の学習データに安全。
    """
    if not words:
        return []

    # --- 1) 文候補に分割 (文末記号 or ギャップ) ---
    sentences = []
    cur = [words[0]]
    for prev, w in zip(words, words[1:]):
        if prev["word"].strip().endswith(_SENTENCE_END) or (w["start"] - prev["end"]) >= gap_split:
            sentences.append(cur)
            cur = [w]
        else:
            cur.append(w)
    sentences.append(cur)

    # --- 2) max_len を超える文は中央付近の最大ギャップで再帰分割 ---
    def split_long(ws):
        dur = ws[-1]["end"] - ws[0]["start"]
        if dur <= max_len or len(ws) < 2:
            return [ws]
        best_gap, best_i = -1.0, None
        for i in range(1, len(ws)):
            pos = ws[i]["start"] - ws[0]["start"]
            if pos < dur * 0.2 or pos > dur * 0.8:
                continue
            gap = ws[i]["start"] - ws[i - 1]["end"]
            if gap > best_gap:
                best_gap, best_i = gap, i
        if best_i is None:
            best_i = len(ws) // 2
        return split_long(ws[:best_i]) + split_long(ws[best_i:])

    sentences = [part for s in sentences for part in split_long(s)]

    # --- 3) min_len に満たない文は隣と結合 (max_len を超えない範囲で) ---
    chunks = []
    for s in sentences:
        if chunks:
            last = chunks[-1]
            gap = s[0]["start"] - last[-1]["end"]
            combined = s[-1]["end"] - last[0]["start"]
            last_dur = last[-1]["end"] - last[0]["start"]
            cur_dur = s[-1]["end"] - s[0]["start"]
            if combined <= max_len and (last_dur < min_len or cur_dur < min_len or gap < 0.3):
                last.extend(s)
                continue
        chunks.append(list(s))

    # --- 4) それでも min_len 未満の断片は、隣接チャンクと結合 (max_len の1割超過まで許容) ---
    tolerance = max_len * 1.1
    i = 0
    while i < len(chunks):
        dur = chunks[i][-1]["end"] - chunks[i][0]["start"]
        if dur >= min_len or len(chunks) < 2:
            i += 1
            continue
        prev_ok = i > 0 and (chunks[i][-1]["end"] - chunks[i - 1][0]["start"]) <= tolerance
        next_ok = i < len(chunks) - 1 and (chunks[i + 1][-1]["end"] - chunks[i][0]["start"]) <= tolerance
        if next_ok:  # 前方結合を優先 (次のセリフの頭に付ける方が自然)
            chunks[i + 1] = chunks[i] + chunks[i + 1]
            chunks.pop(i)
        elif prev_ok:
            chunks[i - 1].extend(chunks[i])
            chunks.pop(i)
        else:
            i += 1  # 結合先がない場合はそのまま (UI側で⚠表示)

    segments = []
    for c in chunks:
        text = "".join(w["word"] for w in c).strip()
        if not text:
            continue
        segments.append({
            "start": round(c[0]["start"], 3),
            "end": round(c[-1]["end"], 3),
            "text": text,
        })
    return segments


def _fmt_srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_dialogue(audio, sr, segments, out_dir, base_name="dialogue",
                    make_clips=True, make_srt=True, make_tts=True, mono_clips=True):
    """segments からセリフ別クリップ・SRT・TTSメタデータ・アノテーションを生成し、ZIPのパスを返す。

    audio: (samples, channels) float32 (編集済みトラック)
    """
    os.makedirs(out_dir, exist_ok=True)
    clips_dir = os.path.join(out_dir, "clips")
    n_samples = audio.shape[0]
    fade = int(FADE_SECONDS * sr)

    annotations = []
    has_speakers = any(seg.get("speaker") for seg in segments)

    if make_clips:
        os.makedirs(clips_dir, exist_ok=True)
    for i, seg in enumerate(segments, start=1):
        s = max(0, int((seg["start"] - CLIP_PAD_SECONDS) * sr))
        e = min(n_samples, int((seg["end"] + CLIP_PAD_SECONDS) * sr))
        if e - s < int(0.05 * sr):
            continue
        speaker = seg.get("speaker") or ""
        # 話者分離時は話者別フォルダに配置
        rel = f"{speaker}/{base_name}_{i:04d}.wav" if speaker else f"{base_name}_{i:04d}.wav"
        if make_clips:
            clip = audio[s:e].copy()
            if mono_clips and clip.shape[1] > 1:
                clip = clip.mean(axis=1, keepdims=True)
            f = min(fade, len(clip) // 2)
            if f > 0:
                ramp = np.linspace(0, 1, f, dtype=np.float32)[:, None]
                clip[:f] *= ramp
                clip[-f:] *= ramp[::-1]
            dst = os.path.join(clips_dir, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            sf.write(dst, np.clip(clip, -1, 1), sr, subtype="PCM_16")
        annotations.append({
            "index": i,
            "file": rel if make_clips else None,
            "start": seg["start"],
            "end": seg["end"],
            "duration": round(seg["end"] - seg["start"], 3),
            "speaker": speaker,
            "text": seg["text"],
        })

    if make_srt:
        with open(os.path.join(out_dir, "subtitles.srt"), "w", encoding="utf-8-sig") as f:
            for i, seg in enumerate(segments, start=1):
                prefix = f"{seg['speaker']}: " if has_speakers and seg.get("speaker") else ""
                f.write(f"{i}\n{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}\n{prefix}{seg['text']}\n\n")

    if make_tts:
        # LJSpeech 形式: ファイル名(拡張子なし)|テキスト。話者分離時は話者列を追加
        with open(os.path.join(out_dir, "metadata.csv"), "w", encoding="utf-8", newline="") as f:
            for a in annotations:
                if a["file"]:
                    base = os.path.splitext(a["file"])[0]
                    if has_speakers:
                        f.write(f"{base}|{a['speaker']}|{a['text']}\n")
                    else:
                        f.write(f"{base}|{a['text']}\n")

    with open(os.path.join(out_dir, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "annotations.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "file", "start", "end", "duration", "speaker", "text"])
        for a in annotations:
            w.writerow([a["index"], a["file"] or "", a["start"], a["end"], a["duration"], a["speaker"], a["text"]])

    zip_path = os.path.join(out_dir, "dialogue_export.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_dir):
            for name in files:
                if name == "dialogue_export.zip":
                    continue
                full = os.path.join(root, name)
                zf.write(full, os.path.relpath(full, out_dir))
    return zip_path, annotations


transcriber = Transcriber()
