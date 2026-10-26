# coding: utf-8
"""Sound Separator - ローカル音源分離アプリのWebサーバー。

- 分離モード dme:   話し声 / 効果音 / BGM (BandIt Plus)
- 分離モード music: ボーカル / ドラム / ベース / ギター / ピアノ / その他 (BS-Roformer + htdemucs_6s)
- /api/denoise: 選択範囲をノイズプロファイルとしたスペクトラルゲート除去 (noisereduce)
"""
import io
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from separator import SAMPLE_RATE, engine, validate_request
from transcriber import assign_speakers, export_dialogue, segment_words, transcriber

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "output")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBSEP_SPECS = {
    "drums": [
        {"name": "kick", "label": "キック", "color": "#ef4444"},
        {"name": "snare", "label": "スネア", "color": "#fb923c"},
        {"name": "toms", "label": "タム", "color": "#eab308"},
        {"name": "cymbals", "label": "シンバル", "color": "#a3e635"},
    ],
    "vocals": [
        {"name": "lead", "label": "リードVo", "color": "#f472b6"},
        {"name": "back", "label": "バックVo", "color": "#c084fc"},
    ],
}
app = FastAPI(title="Sound Separator")

jobs: dict = {}
jobs_lock = threading.Lock()


def set_job(job_id, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


def decode_to_wav(src_path: str, dst_path: str, preview_start=0.0, preview_duration=None):
    """ffmpegで任意の音声/動画ファイルを 44.1kHz ステレオ WAV に変換する。"""
    cmd = ["ffmpeg", "-y", "-i", src_path]
    # 入力の後に -ss を置き、動画も指定時刻のサンプルから正確に切り出す。
    if preview_duration is not None:
        cmd += ["-ss", str(preview_start), "-t", str(preview_duration)]
    cmd += [
        "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE),
        "-acodec", "pcm_f32le", dst_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not os.path.isfile(dst_path):
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError("音声の読み込みに失敗しました: " + " / ".join(tail))


def resample_audio(audio, source_rate, target_rate):
    """(samples, channels) を時間を保って変換する。モデル入力は44.1kHz固定。"""
    if source_rate == target_rate:
        return audio
    from scipy.signal import resample_poly
    factor = math.gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // factor, source_rate // factor, axis=0).astype(np.float32)


def process_job(job_id: str, input_path: str, mode: str, quality: str,
                model_id=None, selected_stems=None, preview_start=0.0, preview_duration=None):
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    try:
        set_job(job_id, status="processing", progress=0, message="音声を読み込み中...")
        mix_path = os.path.join(job_dir, "original.wav")
        decode_to_wav(input_path, mix_path, preview_start, preview_duration)

        audio, sr = sf.read(mix_path, dtype="float32", always_2d=True)
        audio = audio.T  # (channels, samples)
        duration = audio.shape[1] / sr
        if duration < 0.5:
            raise RuntimeError("音声が短すぎます (0.5秒以上必要です)")
        set_job(job_id, duration=duration, sample_rate=sr, progress=0, message="分離エンジンの順番を待っています...")

        def cb(pct, msg):
            set_job(job_id, progress=pct, message=msg)

        t0 = time.time()
        stems = engine.separate(audio, mode=mode, quality=quality, progress_cb=cb,
                                model_id=model_id, selected_stems=selected_stems)

        set_job(job_id, progress=100, message="ファイルを書き出し中...")
        with jobs_lock:
            specs = list(jobs[job_id]["stem_specs"])
        for spec in specs:
            data = stems[spec["name"]].T  # (samples, channels)
            if data.shape[0] != audio.shape[1] or not np.isfinite(data).all():
                raise RuntimeError("モデルの出力長または数値が不正です")
            # AI出力のピークを切り捨てず、細部も保つ。保存時の音量はエディタで調整できる。
            sf.write(os.path.join(job_dir, f"{spec['name']}.wav"), data, sr, subtype="FLOAT")

        set_job(
            job_id,
            status="done",
            message="完了",
            elapsed=round(time.time() - t0, 1),
            output_channels=int(next(iter(stems.values())).shape[0]),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        set_job(job_id, status="error", message=str(e))


@app.get("/api/models")
def get_models():
    return {
        "models": engine.model_catalog(),
        "defaults": {"dme": "bandit", "music": "bs_roformer", "vocals": "bs_roformer", "mega": "mega_53"},
        "runtime": engine.status(),
    }


@app.post("/api/models/release")
def release_models():
    if not engine.release_models():
        raise HTTPException(409, "分離・修復の処理中はモデルを解放できません")
    return engine.status()


def validate_preview(preview_start, preview_duration):
    if not math.isfinite(preview_start) or preview_start < 0:
        raise ValueError("試し分離の開始秒は0以上の有限値で指定してください")
    if preview_duration is not None:
        if not math.isfinite(preview_duration) or not 5 <= preview_duration <= 60:
            raise ValueError("試し分離の長さは5〜60秒で指定してください")
    elif preview_start != 0:
        raise ValueError("開始秒を指定する場合は試し分離の長さも指定してください")


@app.post("/api/jobs")
def create_job(
    file: UploadFile = File(...),
    mode: str = Form("dme"),
    quality: str = Form("standard"),
    model_id: str | None = Form(None),
    selected_stems: str | None = Form(None),
    preview_start: float = Form(0.0),
    preview_duration: float | None = Form(None),
):
    if quality not in ("fast", "standard", "high"):
        raise HTTPException(400, "品質は高速・標準・高品質から選んでください")
    try:
        selection = json.loads(selected_stems) if selected_stems is not None else None
        model_id, specs = validate_request(mode, model_id, selection)
        validate_preview(preview_start, preview_duration)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if not file.filename:
        raise HTTPException(400, "音声・動画ファイルを選択してください")
    model = next(m for m in engine.model_catalog() if m["id"] == model_id)
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "input")[1].lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", ext):
        ext = ".bin"
    input_path = os.path.join(job_dir, "input" + ext)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "filename": file.filename,
            "mode": mode,
            "quality": quality,
            "model_id": model_id,
            "model_label": model["label"],
            "stem_specs": specs,
            "selected_stems": [s["name"] for s in specs],
            "preview_start": preview_start,
            "preview_duration": preview_duration,
            "is_preview": preview_duration is not None,
            "model_notes": model["notes"],
            "status": "queued",
            "progress": 0,
            "message": "待機中...",
            "created": time.time(),
        }

    threading.Thread(target=process_job, args=(job_id, input_path, mode, quality, model_id,
                     [s["name"] for s in specs], preview_start, preview_duration), daemon=True).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        job = dict(job)
    if job["status"] == "done" and job.get("kind") not in ("dialogue", "subsep"):
        job["stems"] = [
            {**spec, "url": f"/api/jobs/{job_id}/audio/{spec['name']}.wav"}
            for spec in job["stem_specs"]
        ]
        job["original_url"] = f"/api/jobs/{job_id}/audio/original.wav"
        job["zip_url"] = f"/api/jobs/{job_id}/download.zip"
    return job


@app.get("/api/jobs/{job_id}/audio/{filename}")
def get_audio(job_id: str, filename: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None or job.get("status") != "done":
            raise HTTPException(404)
        specs = job.get("stem_specs", []) if job.get("kind") != "subsep" else job.get("stems", [])
        allowed = {s["name"] + ".wav" for s in specs} | {"original.wav"}
    if not re.fullmatch(r"[a-zA-Z0-9_-]+\.wav", filename) or filename not in allowed:
        raise HTTPException(404)
    path = os.path.join(OUTPUT_DIR, job_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/jobs/{job_id}/download.zip")
def download_zip(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        job = dict(job) if job else None
    if job is None or job.get("status") != "done" or "stem_specs" not in job:
        raise HTTPException(404)
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    zip_path = os.path.join(job_dir, "stems.zip")
    if not os.path.isfile(zip_path):
        base = os.path.splitext(os.path.basename((job.get("filename") or "audio").replace("\\", "/")))[0]
        base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base) or "audio"
        if job.get("is_preview"):
            base += f"_preview_{job['preview_start']:g}s"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for spec in job["stem_specs"]:
                src = os.path.join(job_dir, f"{spec['name']}.wav")
                if os.path.isfile(src):
                    zf.write(src, f"{base}_{spec['label']}.wav")
            # モデル・区間・モノラル等の条件を素材と一緒に残す。
            manifest = {k: job.get(k) for k in ("filename", "mode", "model_id", "model_label",
                        "quality", "duration", "elapsed", "is_preview", "preview_start",
                        "preview_duration", "output_channels", "selected_stems", "model_notes")}
            manifest["sample_rate"] = SAMPLE_RATE
            manifest["wav_subtype"] = "FLOAT"
            zf.writestr("separation.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return FileResponse(zip_path, media_type="application/zip", filename="stems.zip")


def process_subsep_job(job_id: str, wav_path: str, kind: str):
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    try:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        audio = resample_audio(audio, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
        if audio.shape[1] == 1:  # モノラルはステレオ化 (分離モデルはステレオ前提)
            audio = np.repeat(audio, 2, axis=1)

        def cb(pct, msg):
            set_job(job_id, progress=pct, message=msg)

        stems = engine.subseparate(audio.T, kind, progress_cb=cb)

        set_job(job_id, progress=100, message="ファイルを書き出し中...")
        out = []
        for spec in SUBSEP_SPECS[kind]:
            data = np.clip(stems[spec["name"]].T, -1.0, 1.0)
            sf.write(os.path.join(job_dir, f"{spec['name']}.wav"), data, sr, subtype="PCM_16")
            out.append({**spec, "url": f"/api/jobs/{job_id}/audio/{spec['name']}.wav"})
        set_job(job_id, status="done", message="完了", stems=out)
    except Exception as e:
        import traceback
        traceback.print_exc()
        set_job(job_id, status="error", message=str(e))


@app.post("/api/subseparate")
async def subseparate_api(file: UploadFile = File(...), kind: str = Form(...)):
    if kind not in SUBSEP_SPECS:
        raise HTTPException(400, "invalid kind")
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    wav_path = os.path.join(job_dir, "subsep_input.wav")
    with open(wav_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "kind": "subsep",
            "status": "processing",
            "progress": 0,
            "message": "準備中...",
            "created": time.time(),
        }
    threading.Thread(target=process_subsep_job, args=(job_id, wav_path, kind), daemon=True).start()
    return {"id": job_id}


def process_dialogue_job(job_id: str, wav_path: str, opts: dict):
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    try:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)

        def cb(pct, msg):
            set_job(job_id, progress=pct, message=msg)

        if opts["action"] == "analyze":
            # 文字起こし + 長さ調整 → セグメント一覧を返す (ファイルは作らない)
            segments, words, lang = transcriber.transcribe(
                audio, sr,
                model_size=opts["model_size"],
                language=opts["language"] or None,
                progress_cb=cb,
            )
            if opts["auto_adjust"] and words:
                segments = segment_words(words, min_len=opts["min_len"], max_len=opts["max_len"])
            if not segments:
                raise RuntimeError("セリフを検出できませんでした (対象トラックに音声が含まれているか確認してください)")
            if opts["diarize"]:
                segments = assign_speakers(audio, sr, segments, num_speakers=opts["num_speakers"], progress_cb=cb)
            annotated = [
                {"index": i, "start": s["start"], "end": s["end"],
                 "duration": round(s["end"] - s["start"], 3),
                 "speaker": s.get("speaker", ""), "text": s["text"]}
                for i, s in enumerate(segments, start=1)
            ]
            set_job(job_id, status="done", progress=100, message="完了",
                    segments=annotated, language=lang)
        else:
            # 確認・調整済みのセグメントでファイル書き出し
            segments = opts["segments"]
            if not segments:
                raise RuntimeError("書き出すセリフがありません")
            set_job(job_id, progress=50, message="クリップ・字幕・メタデータを書き出し中...")
            out_dir = os.path.join(job_dir, "dialogue")
            _, annotations = export_dialogue(
                audio, sr, segments, out_dir,
                base_name=opts["base_name"],
                make_clips=opts["make_clips"],
                make_srt=opts["make_srt"],
                make_tts=opts["make_tts"],
                mono_clips=opts["mono_clips"],
            )
            set_job(job_id, status="done", progress=100, message="完了",
                    segments=annotations,
                    zip_url=f"/api/jobs/{job_id}/dialogue.zip")
    except Exception as e:
        import traceback
        traceback.print_exc()
        set_job(job_id, status="error", message=str(e))


@app.post("/api/dialogue-export")
async def dialogue_export_api(
    file: UploadFile = File(...),
    action: str = Form("analyze"),
    model_size: str = Form("medium"),
    language: str = Form(""),
    base_name: str = Form("dialogue"),
    make_clips: bool = Form(True),
    make_srt: bool = Form(True),
    make_tts: bool = Form(True),
    mono_clips: bool = Form(True),
    auto_adjust: bool = Form(True),
    min_len: float = Form(2.0),
    max_len: float = Form(10.0),
    diarize: bool = Form(False),
    num_speakers: int = Form(0),
    segments: str = Form(""),
):
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    wav_path = os.path.join(job_dir, "dialogue_input.wav")
    with open(wav_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ファイル名に使えない文字を除去
    safe_base = "".join(c for c in base_name if c not in '\\/:*?"<>|').strip() or "dialogue"

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "kind": "dialogue",
            "status": "processing",
            "progress": 0,
            "message": "準備中...",
            "created": time.time(),
        }
    parsed_segments = []
    if segments:
        import json as _json
        try:
            for s in _json.loads(segments):
                st, en = float(s["start"]), float(s["end"])
                text = str(s.get("text", "")).strip()
                if en - st >= 0.05 and text:
                    seg = {"start": st, "end": en, "text": text}
                    spk = str(s.get("speaker", "")).strip()
                    if spk:
                        seg["speaker"] = "".join(c for c in spk if c not in '\\/:*?"<>|')[:20]
                    parsed_segments.append(seg)
        except (ValueError, KeyError, TypeError):
            raise HTTPException(400, "セグメントデータが不正です")

    opts = {
        "action": action if action in ("analyze", "export") else "analyze",
        "model_size": model_size,
        "language": language,
        "base_name": safe_base,
        "make_clips": make_clips,
        "make_srt": make_srt,
        "make_tts": make_tts,
        "mono_clips": mono_clips,
        "auto_adjust": auto_adjust,
        "min_len": float(np.clip(min_len, 0.3, 60.0)),
        "max_len": float(np.clip(max_len, 1.0, 120.0)),
        "diarize": diarize,
        "num_speakers": int(np.clip(num_speakers, 0, 8)),
        "segments": parsed_segments,
    }
    threading.Thread(target=process_dialogue_job, args=(job_id, wav_path, opts), daemon=True).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}/dialogue.zip")
def download_dialogue_zip(job_id: str):
    path = os.path.join(OUTPUT_DIR, job_id, "dialogue", "dialogue_export.zip")
    if not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path, media_type="application/zip", filename="dialogue_export.zip")


@app.post("/api/midi")
def midi_export_api(
    file: UploadFile = File(...),
    onset: float = Form(0.5),
    min_note_ms: float = Form(120.0),
):
    """トラックをAI採譜してMIDIを返す (basic-pitch)。sync def (イベントループを塞がない)。"""
    from midi_export import audio_to_midi_bytes

    tmp_dir = os.path.join(OUTPUT_DIR, "_midi_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, uuid.uuid4().hex[:12] + ".wav")
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        data = audio_to_midi_bytes(tmp_path, onset=onset, min_note_ms=min_note_ms)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return Response(content=data, media_type="audio/midi")


@app.post("/api/denoise")
def denoise(
    file: UploadFile = File(...),
    noise_start: float = Form(...),
    noise_end: float = Form(...),
    strength: float = Form(1.0),
):
    """選択範囲 (noise_start〜noise_end 秒) をノイズプロファイルとして学習し、
    トラック全体からスペクトラルゲーティングで除去して返す。
    重い同期処理のため sync def (FastAPIがスレッドプールで実行し、イベントループを塞がない)。"""
    import noisereduce as nr

    raw = file.file.read()
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)

    s = max(0, int(noise_start * sr))
    e = min(len(audio), int(noise_end * sr))
    if e - s < int(0.05 * sr):
        raise HTTPException(400, "ノイズ区間が短すぎます (0.05秒以上選択してください)")

    noise = audio[s:e]
    strength = float(np.clip(strength, 0.1, 1.0))

    out = np.stack(
        [
            nr.reduce_noise(
                y=audio[:, c],
                sr=sr,
                y_noise=noise[:, c],
                stationary=True,
                prop_decrease=strength,
            )
            for c in range(audio.shape[1])
        ],
        axis=1,
    ).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, np.clip(out, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.post("/api/effect")
def apply_effect(
    file: UploadFile = File(...),
    effect: str = Form(...),
    sensitivity: float = Form(0.5),
    base_freq: float = Form(50.0),
    target_lufs: float = Form(-14.0),
    param: float = Form(0.0),
):
    """範囲修復エフェクト。クライアントは選択範囲+前後コンテキストのWAVを送り、
    処理済みの同じ長さのWAVを受け取って選択範囲だけをクロスフェードで書き戻す。
    重い同期処理のため sync def (イベントループを塞がない)。"""
    from repair import declick, dehum

    raw = file.file.read()
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)

    headers = {}
    if effect == "declick":
        out = declick(audio, sensitivity=sensitivity, sr=sr)
    elif effect == "dehum":
        out = dehum(audio, base_freq=base_freq, sr=sr)
    elif effect == "dereverb":
        model_audio = resample_audio(audio, sr, SAMPLE_RATE)
        out = engine.dereverb(model_audio.T).T  # (samples, ch) <-> (ch, samples)
        out = resample_audio(out, SAMPLE_RATE, sr)[:len(audio)]
        if len(out) < len(audio):
            out = np.pad(out, ((0, len(audio) - len(out)), (0, 0)))
    elif effect == "pitchshift":
        # トラック全体のピッチシフト (半音単位、長さ不変)
        import librosa

        steps = float(np.clip(param, -12.0, 12.0))
        out = np.stack(
            [librosa.effects.pitch_shift(y=audio[:, c].astype(np.float64), sr=sr, n_steps=steps)
             for c in range(audio.shape[1])],
            axis=1,
        ).astype(np.float32)
    elif effect == "tempo":
        # テンポ変更 (長さが変わる)。rate>1で速く/短く
        import librosa

        rate = float(np.clip(param, 0.25, 4.0))
        out = np.stack(
            [librosa.effects.time_stretch(y=audio[:, c].astype(np.float64), rate=rate)
             for c in range(audio.shape[1])],
            axis=1,
        ).astype(np.float32)
    elif effect == "loudnorm":
        # ITU-R BS.1770 準拠のラウドネス測定 → 目標LUFSへゲイン調整
        import pyloudnorm

        meter = pyloudnorm.Meter(sr)
        measured = float(meter.integrated_loudness(audio.astype(np.float64)))
        if not np.isfinite(measured) or measured < -70:
            raise HTTPException(400, "音が小さすぎてラウドネスを測定できません")
        gain_db = float(np.clip(target_lufs, -36.0, -5.0)) - measured
        gain = 10 ** (gain_db / 20)
        # クリップ防止: ピークが -1dBTP を超えないようゲインを制限
        peak = float(np.abs(audio).max()) * gain
        limit = 10 ** (-1 / 20)
        if peak > limit:
            gain *= limit / peak
        out = audio * gain
        headers = {
            "X-Measured-LUFS": f"{measured:.1f}",
            "X-Applied-Gain-DB": f"{20 * np.log10(gain):.1f}",
        }
    else:
        raise HTTPException(400, "unknown effect")

    buf = io.BytesIO()
    sf.write(buf, np.clip(out, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
