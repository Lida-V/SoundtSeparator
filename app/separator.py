# coding: utf-8
"""音源分離エンジン。

用途別の分離モードを提供する:

- dme:   映像・動画音声向け。話し声 / 効果音 / BGM の3ステム。
         BandIt Plus (MSST, DnR SDR 11.50) を使用。
         high 品質では CDX23 (Demucs4) 3モデルとのアンサンブル。
- music: 音楽向け。ボーカル / ドラム / ベース / ギター / ピアノ / その他 の6ステム。
         選択したRoFormerでボーカル抽出後、残りを htdemucs_6s で分割する多段構成。
- vocals: ボーカルと伴奏の2ステム (BS-RoFormer / MelBand RoFormer)。
- mega: MVSep Mega 53カテゴリから選択した楽器のみを抽出。
"""
import copy
import gc
import hashlib
import os
import sys
import threading
import urllib.request
import uuid

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSST_DIR = os.path.join(BASE_DIR, "engine", "msst")
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, MSST_DIR)

from demucs.apply import apply_model  # noqa: E402
from demucs.pretrained import get_model as demucs_get_model  # noqa: E402
from demucs.states import load_model as demucs_load_model  # noqa: E402

import utils.model_utils as msst_model_utils  # noqa: E402  (MSST)
from utils.model_utils import demix as msst_demix  # noqa: E402  (MSST)
from utils.settings import get_model_from_config  # noqa: E402  (MSST)

SAMPLE_RATE = 44100

# ---------------------------------------------------------------------------
# モデル定義
# ---------------------------------------------------------------------------
BANDIT_CKPT = {
    "file": "model_bandit_plus_dnr_sdr_11.47.chpt",
    "url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt",
    "config": os.path.join(MSST_DIR, "configs", "config_dnr_bandit_bsrnn_multi_mus64.yaml"),
    "type": "bandit",
}
BSROFORMER_CKPT = {
    "file": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "config": os.path.join(MSST_DIR, "configs", "viperx", "model_bs_roformer_ep_317_sdr_12.9755.yaml"),
    "type": "bs_roformer",
}
MELROFORMER_CKPT = {
    "file": "MelBandRoformer.ckpt",
    "url": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
    "config": os.path.join(MSST_DIR, "configs", "KimberleyJensen", "config_vocals_mel_band_roformer_kj.yaml"),
    "type": "mel_band_roformer",
}
MEGA_CKPT = {
    "file": "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt",
    "url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.21/mvsep_mega_model_bs_roformer_53_stems_v1.ckpt",
    "sha256": "c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f",
    "config": os.path.join(MODELS_DIR, "mvsep_mega_model_bs_roformer_53_stems.yaml"),
    "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.21/mvsep_mega_model_bs_roformer_53_stems.yaml",
    "type": "bs_roformer",
}
DEREVERB_CKPT = {
    "file": "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    "url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    "config": os.path.join(MODELS_DIR, "dereverb_mel_band_roformer_anvuew.yaml"),
    "config_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml",
    "type": "mel_band_roformer",
}
KARAOKE_CKPT = {
    "file": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "config": os.path.join(MODELS_DIR, "config_mel_band_roformer_karaoke.yaml"),
    "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/config_mel_band_roformer_karaoke.yaml",
    "type": "mel_band_roformer",
}
DRUMSEP_CKPT_FILE = "model_drumsep.th"
DRUMSEP_CKPT_URL = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.5/model_drumsep.th"
CDX23_URL = "https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/releases/download/v.1.0.0/"
CDX23_FILES = ["97d170e1-a778de4a.th", "97d170e1-dbb4db15.th", "97d170e1-e41a5468.th"]

# カタログとAPIの入力検証はモデルのロード・通信を行わない。
def _stem(name, label, color):
    return {"name": name, "label": label, "color": color}


DME_STEMS = [_stem("dialog", "話し声", "#38bdf8"), _stem("effect", "効果音", "#fbbf24"), _stem("music", "BGM", "#c084fc")]
VOCAL_STEMS = [_stem("vocals", "ボーカル", "#f472b6"), _stem("instrumental", "伴奏", "#38bdf8")]
MUSIC_STEMS = [
    _stem("vocals", "ボーカル", "#f472b6"), _stem("drums", "ドラム", "#fb923c"),
    _stem("bass", "ベース", "#34d399"), _stem("guitar", "ギター", "#fbbf24"),
    _stem("piano", "ピアノ", "#38bdf8"), _stem("other", "その他", "#a78bfa"),
]
_MEGA_LABELS = {
    "accordion": "アコーディオン", "acoustic-guitar": "アコースティックギター", "back-vocal": "バックボーカル",
    "banjo": "バンジョー", "bass": "ベース", "bassoon": "ファゴット", "bells": "ベル", "bowed_strings": "弓奏弦楽器",
    "brass": "金管楽器", "cello": "チェロ", "clarinet": "クラリネット", "congas": "コンガ", "digital-piano": "電子ピアノ",
    "dobro": "ドブロ", "double-bass": "コントラバス", "drums": "ドラム", "electric-guitar": "エレキギター",
    "flute": "フルート", "french-horn": "ホルン", "glockenspiel": "グロッケン", "guitar": "ギター", "harmonica": "ハーモニカ",
    "harp": "ハープ", "harpsichord": "チェンバロ", "hh": "ハイハット", "keys": "鍵盤楽器", "kick": "キック",
    "lead-vocal": "リードボーカル", "mandolin": "マンドリン", "marimba": "マリンバ", "oboe": "オーボエ", "organ": "オルガン",
    "percussion": "パーカッション", "piano": "ピアノ", "saxophone": "サックス", "sitar": "シタール", "snare": "スネア",
    "strings": "ストリングス", "synth": "シンセ", "tambourine": "タンバリン", "timpani": "ティンパニ", "toms": "タム",
    "triangle": "トライアングル", "trombone": "トロンボーン", "trumpet": "トランペット", "tuba": "チューバ", "ukulele": "ウクレレ",
    "viola": "ビオラ", "violin": "バイオリン", "vocal": "ボーカル全体", "wind": "管楽器", "wind-chimes": "ウィンドチャイム", "woodwind": "木管楽器",
}
MEGA_STEMS = [_stem(name, label, ("#38bdf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa")[i % 5]) for i, (name, label) in enumerate(_MEGA_LABELS.items())]
DEFAULT_MODELS = {"dme": "bandit", "music": "bs_roformer", "vocals": "bs_roformer", "mega": "mega_53"}
MODEL_CATALOG = [
    {"id": "bandit", "label": "BandIt Plus", "modes": ["dme"], "description": "映像の話し声・効果音・BGMをステレオで分離。", "stems": DME_STEMS,
     "download_mb": 149, "vram_gb": 4, "channels": "stereo", "source_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training", "notes": "高品質はCDX23とのアンサンブル（追加ダウンロード約162MB）。", "_spec": BANDIT_CKPT},
    {"id": "dbs_dnr", "label": "DBS-TIGER DnR", "modes": ["dme"], "description": "入力に応じた帯域分割による、軽量な映像3分離。", "stems": DME_STEMS,
     "download_mb": 5, "vram_gb": 1, "channels": "mono", "source_url": "https://github.com/JusperLee/Dynamic-Band-Splitting", "notes": "研究モデル。出力はモノラルで、左右の定位は保持されません。"},
    {"id": "bs_roformer", "label": "BS-RoFormer · viperx", "modes": ["music", "vocals"], "description": "ボーカル分離。6ステムではDemucsによる楽器分離も実行。", "stems": VOCAL_STEMS,
     "download_mb": 640, "vram_gb": 6, "channels": "stereo", "source_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md", "notes": "6ステム時はhtdemucs_6sも初回取得（約55MB）。", "_spec": BSROFORMER_CKPT},
    {"id": "mel_band_roformer", "label": "MelBand RoFormer · Kimberley Jensen", "modes": ["music", "vocals"], "description": "メル帯域を使うボーカル分離。BS-RoFormerと聴き比べる候補。", "stems": VOCAL_STEMS,
     "download_mb": 913, "vram_gb": 6, "channels": "stereo", "source_url": "https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model", "notes": "モデル間の優劣は音源によります。6ステム時はhtdemucs_6sを併用。", "_spec": MELROFORMER_CKPT},
    {"id": "mega_53", "label": "MVSep Mega · 53カテゴリ", "modes": ["mega"], "description": "弦・管・鍵盤・シンセなど53カテゴリから選択して抽出。", "stems": MEGA_STEMS,
     "download_mb": 1369, "vram_gb": 16, "channels": "stereo", "source_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21", "notes": "カテゴリは重複します。全ステムを加算しても元音源には戻りません。個別の専用モデルが高精度な場合があります。", "_spec": MEGA_CKPT},
]


def validate_request(mode="dme", model_id=None, selected_stems=None):
    """モデル/モードの組合せを検証し、確定モデルIDと出力仕様を返す。"""
    if mode not in DEFAULT_MODELS:
        raise ValueError("分離モードが正しくありません")
    model_id = model_id or DEFAULT_MODELS[mode]
    entry = next((item for item in MODEL_CATALOG if item["id"] == model_id), None)
    if entry is None or mode not in entry["modes"]:
        raise ValueError("選択したモデルはこの分離モードに対応していません")
    specs = MUSIC_STEMS if mode == "music" else entry["stems"]
    if selected_stems is not None:
        if not isinstance(selected_stems, (list, tuple)) or not selected_stems or any(not isinstance(name, str) for name in selected_stems):
            raise ValueError("抽出するステムを1つ以上選択してください")
        names = set(selected_stems)
        if len(names) != len(selected_stems) or names - {item["name"] for item in specs}:
            raise ValueError("抽出ステムの指定が正しくありません")
        specs = [item for item in specs if item["name"] in names]
    return model_id, copy.deepcopy(specs)

# demucs チャンク処理 (進捗報告用の外部分割)
CHUNK_SECONDS = 40
CHUNK_OVERLAP_SECONDS = 4

# ---------------------------------------------------------------------------
# MSST demix の進捗を tqdm シムで取得する
# ---------------------------------------------------------------------------
_demix_progress_cb = None


class _CallbackPbar:
    def __init__(self, total=None, **kwargs):
        self.total = max(total or 1, 1)
        self.n = 0

    def update(self, n=1):
        self.n += n
        if _demix_progress_cb:
            _demix_progress_cb(min(self.n / self.total, 1.0))

    def close(self):
        pass


msst_model_utils.tqdm = _CallbackPbar


def _download(url, filename, progress_cb=None, label="", sha256=None):
    """取得途中のファイルをモデルとして扱わず、成功時のみ置き換える。"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        if sha256:
            digest = hashlib.sha256()
            with open(path, "rb") as existing:
                for block in iter(lambda: existing.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != sha256:
                raise RuntimeError(f"モデルのチェックサムが一致しません。{filename} を確認してください")
        return path
    temporary = path + "." + uuid.uuid4().hex + ".partial"
    try:
        if progress_cb:
            progress_cb(0, f"モデルをダウンロード中 ({label})...")
        request = urllib.request.Request(url, headers={"User-Agent": "SoundtSeparator/1.0"})
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=60) as response, open(temporary, "wb") as target:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            last_mb = -1
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
                digest.update(block)
                downloaded += len(block)
                megabytes = downloaded // (1024 * 1024)
                if progress_cb and megabytes != last_mb:
                    size = f" / {total / (1024 * 1024):.0f} MB" if total else " MB"
                    progress_cb(0, f"モデルをダウンロード中 ({label}): {megabytes}{size}")
                    last_mb = megabytes
        if not downloaded or (total and downloaded != total):
            raise RuntimeError("モデルのダウンロードが途中で終了しました。もう一度実行してください")
        if sha256 and digest.hexdigest() != sha256:
            raise RuntimeError("モデルのチェックサムが一致しません。もう一度実行してください")
        os.replace(temporary, path)
    finally:
        if os.path.isfile(temporary):
            os.remove(temporary)
    return path


def _fit_audio(data, reference):
    """推論端数を揃え、無効なモデル出力は保存前に検出する。"""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[0] not in {1, reference.shape[0]} or not np.isfinite(data).all():
        raise RuntimeError("分離モデルから不正な音声が返されました")
    data = data[:, :reference.shape[1]]
    if data.shape[1] < reference.shape[1]:
        data = np.pad(data, ((0, 0), (0, reference.shape[1] - data.shape[1])))
    return np.ascontiguousarray(data)


class _SelectedMegaHeads(torch.nn.Module):
    """共有Transformerを使い、選択した楽器の出力ヘッドだけを計算する。"""
    def __init__(self, model, indices):
        super().__init__()
        self.model = model
        self.indices = indices

    def forward(self, audio):
        return self.model(audio, active_stem_ids=self.indices)


def _load_state_dict_flexible(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state", "state_dict", "model_state_dict"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
    model.load_state_dict(sd)


class Engine:
    """全モデルを遅延ロード・キャッシュする分離エンジン。1ジョブずつ実行。"""

    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self._active_key = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def model_catalog(self):
        cached = self._cache.copy()
        result = []
        for entry in MODEL_CATALOG:
            item = {key: copy.deepcopy(value) for key, value in entry.items() if not key.startswith("_")}
            item["notes"] = [item["notes"]] if isinstance(item["notes"], str) else item["notes"]
            spec = entry.get("_spec")
            if spec:
                path = os.path.join(MODELS_DIR, spec["file"])
                item["downloaded"] = os.path.isfile(path) and os.path.getsize(path) > 0
                item["cached"] = spec["file"] in cached
            else:
                from dbs_adapter import downloaded
                item["downloaded"] = downloaded()
                item["cached"] = entry["id"] in cached
            result.append(item)
        return result

    def status(self):
        status = {"device": self.device, "gpu_name": None, "vram_total_gb": None,
                  "vram_allocated_gb": None, "vram_free_gb": None,
                  "cached_models": list(self._cache.copy()), "busy": self._lock.locked()}
        if self.device.startswith("cuda"):
            try:
                free, total = torch.cuda.mem_get_info(self.device)
                status.update(gpu_name=torch.cuda.get_device_name(self.device),
                              vram_total_gb=round(total / 1024**3, 2),
                              vram_allocated_gb=round(torch.cuda.memory_allocated(self.device) / 1024**3, 2),
                              vram_free_gb=round(free / 1024**3, 2))
            except RuntimeError:
                pass
        return status

    @staticmethod
    def _move_cached(value, device):
        # MSST=(model,config)、CDX23=[model,...]、Demucs/DBS=model
        if isinstance(value, tuple):
            value[0].to(device)
        elif isinstance(value, list):
            for model in value:
                model.to(device)
        else:
            value.to(device)

    def _activate(self, key):
        """別モデルをCPUへ退避し、GPU上に前段の重みを溜めない。"""
        if self._active_key != key:
            for cached_key, value in self._cache.copy().items():
                if cached_key != key:
                    self._move_cached(value, "cpu")
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            self._active_key = key
        if key in self._cache:
            self._move_cached(self._cache[key], self.device)

    def _release_models(self):
        self._cache.clear()
        self._active_key = None
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def release_models(self):
        """処理中のモデルは解放しない。モデルファイルは保持する。"""
        if not self._lock.acquire(blocking=False):
            return False
        try:
            self._release_models()
            return True
        finally:
            self._lock.release()

    # ------------------------------------------------------------------ モデルロード
    def _get_msst(self, spec, progress_cb=None, label=""):
        key = spec["file"]
        self._activate(key)
        if key not in self._cache:
            if "config_url" in spec and not os.path.isfile(spec["config"]):
                _download(spec["config_url"], os.path.basename(spec["config"]), progress_cb, label + " 設定")
            path = _download(spec["url"], spec["file"], progress_cb, label, sha256=spec.get("sha256"))
            if progress_cb:
                progress_cb(0, f"モデルを読み込み中 ({label})...")
            model, config = get_model_from_config(spec["type"], spec["config"])
            # 公開configの学習用バッチ設定を引き継がず、推論のVRAMを抑える。
            config.inference.batch_size = 1
            _load_state_dict_flexible(model, path)
            model = model.to(self.device).eval()
            self._cache[key] = (model, config)
        return self._cache[key]

    def _get_htdemucs6s(self, progress_cb=None):
        self._activate("htdemucs_6s")
        if "htdemucs_6s" not in self._cache:
            if progress_cb:
                progress_cb(0, "モデルをダウンロード中 (htdemucs_6s)...")
            model = demucs_get_model("htdemucs_6s")
            model.to(self.device).eval()
            self._cache["htdemucs_6s"] = model
        return self._cache["htdemucs_6s"]

    def _get_cdx23(self, progress_cb=None):
        self._activate("cdx23")
        if "cdx23" not in self._cache:
            models = []
            for f in CDX23_FILES:
                path = _download(CDX23_URL + f, f, progress_cb, "CDX23")
                model = _load_demucs_checkpoint(path)
                model.to(self.device).eval()
                models.append(model)
            self._cache["cdx23"] = models
        return self._cache["cdx23"]

    # ------------------------------------------------------------------ 分離 (公開API)
    def separate(self, audio, mode="dme", quality="standard", progress_cb=None,
                 model_id=None, selected_stems=None):
        """audio: float32 ndarray (channels, samples) @44100Hz
        返り値: dict stem名 -> ndarray (channels, samples)
        """
        model_id, specs = validate_request(mode, model_id, selected_stems)
        if quality not in {"fast", "standard", "high"}:
            raise ValueError("品質設定が正しくありません")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[0] not in {1, 2} or audio.shape[1] < SAMPLE_RATE // 2:
            raise ValueError("0.5秒以上のモノラルまたはステレオ音声が必要です")
        if not np.isfinite(audio).all():
            raise ValueError("音声に無効な数値が含まれています")
        original_channels = audio.shape[0]
        if original_channels == 1:
            audio = np.repeat(audio, 2, axis=0)
        audio = np.ascontiguousarray(audio)
        with self._lock:
            try:
                if model_id == "dbs_dnr":
                    from dbs_adapter import load_model, separate
                    self._activate("dbs_dnr")
                    if "dbs_dnr" not in self._cache:
                        self._cache["dbs_dnr"] = load_model(self.device, progress_cb)
                    result = separate(audio, quality, self.device, progress_cb, model=self._cache["dbs_dnr"])
                elif mode == "mega":
                    result = self._separate_mega(audio, quality, progress_cb, [item["name"] for item in specs])
                elif mode == "music":
                    result = self._separate_music(audio, quality, progress_cb, model_id)
                elif mode == "vocals":
                    result = self._separate_vocals(audio, quality, progress_cb, model_id)
                else:
                    result = self._separate_dme(audio, quality, progress_cb)
                output = {}
                for item in specs:
                    name = item["name"]
                    data = _fit_audio(result[name], audio)
                    output[name] = data.mean(axis=0, keepdims=True) if original_channels == 1 else data
                return output
            except torch.cuda.OutOfMemoryError as exc:
                self._release_models()
                raise RuntimeError("GPUメモリが不足しました。他のGPUアプリを閉じるか、短い試聴区間で再実行してください") from exc

    # ------------------------------------------------------------------ DME モード
    def _separate_dme(self, audio, quality, progress_cb):
        overlap = {"fast": 2, "standard": 4, "high": 6}.get(quality, 4)
        use_ensemble = quality == "high"

        model, config = self._get_msst(BANDIT_CKPT, progress_cb, "BandIt Plus")
        config.inference.num_overlap = overlap

        bandit_span = 0.55 if use_ensemble else 0.92

        def cb(frac):
            if progress_cb:
                pct = int(100 * frac * bandit_span)
                progress_cb(pct, f"分離処理中 (BandIt Plus)... ({pct}%)")

        res = self._run_msst(model, config, BANDIT_CKPT["type"], audio, cb)
        stems = {
            "dialog": res["speech"],
            "effect": res["effects"],
            "music": res["music"],
        }

        if use_ensemble:
            cdx = self._separate_cdx23(audio, progress_cb, base_pct=55, span=40)
            for k in stems:
                n = min(stems[k].shape[1], cdx[k].shape[1], audio.shape[1])
                stems[k] = 0.6 * stems[k][:, :n] + 0.4 * cdx[k][:, :n]

        return stems

    def _separate_cdx23(self, audio, progress_cb, base_pct=0, span=90):
        models = self._get_cdx23(progress_cb)
        out = _chunked_demucs(
            models, audio, self.device, overlap=0.75,
            progress_cb=(lambda f: progress_cb(
                int(base_pct + span * f), f"分離処理中 (アンサンブル)... ({int(base_pct + span * f)}%)"
            )) if progress_cb else None,
        )
        # CDX23 の出力順: 0=music, 1=effect, 2=dialog
        return {"dialog": out[2], "effect": out[1], "music": out[0]}

    # ------------------------------------------------------------------ 音楽モード
    def _separate_vocals(self, audio, quality, progress_cb, model_id="bs_roformer", span=92):
        spec = MELROFORMER_CKPT if model_id == "mel_band_roformer" else BSROFORMER_CKPT
        label = "MelBand RoFormer" if model_id == "mel_band_roformer" else "BS-RoFormer"
        model, config = self._get_msst(spec, progress_cb, label)
        config.inference.num_overlap = {"fast": 2, "standard": 4, "high": 6}[quality]

        def cb(frac):
            if progress_cb:
                pct = int(span * frac)
                progress_cb(pct, f"ボーカルを分離中 ({label})... ({pct}%)")

        res = self._run_msst(model, config, spec["type"], audio, cb)
        vocals = _fit_audio(res["vocals"], audio)
        # 伴奏を差分から作り、同位置での足し戻しを保つ。
        return {"vocals": vocals, "instrumental": audio - vocals}

    def _separate_mega(self, audio, quality, progress_cb, selected_stems):
        model, config = self._get_msst(MEGA_CKPT, progress_cb, "MVSep Mega 53")
        config = copy.deepcopy(config)
        all_names = list(config.training.instruments)
        indices = [all_names.index(name) for name in selected_stems]
        config.training.instruments = selected_stems
        config.training.target_instrument = None
        config.inference.num_overlap = {"fast": 2, "standard": 4, "high": 6}[quality]

        def cb(frac):
            if progress_cb:
                pct = int(92 * frac)
                progress_cb(pct, f"楽器を抽出中 (Mega 53・{len(indices)}カテゴリ)... ({pct}%)")

        return self._run_msst(_SelectedMegaHeads(model, indices), config, MEGA_CKPT["type"], audio, cb)

    def _separate_music(self, audio, quality, progress_cb, model_id="bs_roformer"):
        dem_overlap = {"fast": 0.25, "standard": 0.55, "high": 0.8}.get(quality, 0.55)

        # --- stage 1: 選択したRoFormerでボーカル抽出 ---
        pair = self._separate_vocals(audio, quality, progress_cb, model_id, span=50)
        vocals, instrumental = pair["vocals"], pair["instrumental"]

        # --- stage 2: 残りを htdemucs_6s で楽器分割 ---
        model6 = self._get_htdemucs6s(progress_cb)

        def cb2(frac):
            if progress_cb:
                pct = int(50 + 45 * frac)
                progress_cb(pct, f"楽器を分離中 (Demucs 6stem)... ({pct}%)")

        out = _chunked_demucs([model6], instrumental, self.device, overlap=dem_overlap, progress_cb=cb2)
        # htdemucs_6s の出力順: drums, bass, other, vocals, guitar, piano
        idx = {name: i for i, name in enumerate(model6.sources)}
        stems = {
            "vocals": vocals,
            "drums": out[idx["drums"]],
            "bass": out[idx["bass"]],
            "guitar": out[idx["guitar"]],
            "piano": out[idx["piano"]],
            # 段間の残留ボーカルは "その他" に合算して完全再構成を保つ
            "other": out[idx["other"]] + out[idx["vocals"]],
        }
        return stems

    # ------------------------------------------------------------------ トラックの追加分離
    def subseparate(self, audio, kind, progress_cb=None):
        """既存トラックをさらに分離する。
        kind="drums":  kick / snare / cymbals / toms (DrumSep htdemucs)
        kind="vocals": lead / back (MelBand Roformer Karaoke)
        audio: (channels, samples) → dict stem名 -> (channels, samples)
        """
        with self._lock:
            if kind == "drums":
                self._activate("drumsep")
                if "drumsep" not in self._cache:
                    path = _download(DRUMSEP_CKPT_URL, DRUMSEP_CKPT_FILE, progress_cb, "DrumSep")
                    if progress_cb:
                        progress_cb(0, "モデルを読み込み中 (DrumSep)...")
                    model = _load_demucs_checkpoint(path)
                    model.to(self.device).eval()
                    self._cache["drumsep"] = model
                model = self._cache["drumsep"]

                def cb(frac):
                    if progress_cb:
                        progress_cb(int(100 * frac), f"ドラムを細分化中... ({int(100 * frac)}%)")

                out = _chunked_demucs([model], audio, self.device, overlap=0.55, progress_cb=cb)
                # DrumSep のソース名はスペイン語 (bombo=キック, redoblante=スネア, platillos=シンバル)
                name_map = {"bombo": "kick", "redoblante": "snare", "platillos": "cymbals", "toms": "toms"}
                return {name_map.get(name, name): out[i] for i, name in enumerate(model.sources)}

            if kind == "vocals":
                model, config = self._get_msst(KARAOKE_CKPT, progress_cb, "Karaoke Roformer")

                def cb(frac):
                    if progress_cb:
                        progress_cb(int(100 * frac), f"リード/バックを分離中... ({int(100 * frac)}%)")

                res = self._run_msst(model, config, KARAOKE_CKPT["type"], audio, cb)
                # target: karaoke = 入力からリードボーカルを除いたもの
                # (ボーカルステムに適用した場合 ≈ バックボーカル)
                back = res["karaoke"][:, : audio.shape[1]]
                if back.shape[1] < audio.shape[1]:
                    back = np.pad(back, ((0, 0), (0, audio.shape[1] - back.shape[1])))
                lead = audio - back
                return {"lead": lead, "back": back}

            raise ValueError(f"unknown subseparate kind: {kind}")

    # ------------------------------------------------------------------ リバーブ除去
    def dereverb(self, audio, progress_cb=None):
        """audio: (channels, samples) → リバーブ除去済み (channels, samples)"""
        with self._lock:
            model, config = self._get_msst(DEREVERB_CKPT, progress_cb, "Dereverb Roformer")

            def cb(frac):
                if progress_cb:
                    progress_cb(int(100 * frac), "リバーブ除去中...")

            res = self._run_msst(model, config, DEREVERB_CKPT["type"], audio, cb)
            out = res["noreverb"][:, : audio.shape[1]]
            if out.shape[1] < audio.shape[1]:
                out = np.pad(out, ((0, 0), (0, audio.shape[1] - out.shape[1])))
            return out

    # ------------------------------------------------------------------ MSST 実行
    def _run_msst(self, model, config, model_type, audio, frac_cb):
        global _demix_progress_cb
        _demix_progress_cb = frac_cb
        try:
            mix = torch.from_numpy(audio)
            res = msst_demix(config, model, mix, self.device, model_type, pbar=True)
        finally:
            _demix_progress_cb = None
        return {k: v.astype(np.float32) for k, v in res.items()}


def _load_demucs_checkpoint(path):
    """PyTorch 2.6+ は weights_only=True が既定でこのチェックポイント
    (HTDemucsクラスをpickle) を読めないため、一時的に既定を戻して読み込む。
    モデルは ZFTurbo の GitHub リリース (固定URL) 由来のみ。"""
    orig_load = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = patched
    try:
        return demucs_load_model(path)
    finally:
        torch.load = orig_load


def _chunked_demucs(models, audio, device, overlap=0.55, progress_cb=None):
    """demucs系モデル(複数ならアンサンブル平均)をチャンク+クロスフェードで適用。
    返り値: ndarray (stems, channels, samples)"""
    n_sources = len(models[0].sources)
    n_samples = audio.shape[1]
    chunk = CHUNK_SECONDS * SAMPLE_RATE
    hop = chunk - CHUNK_OVERLAP_SECONDS * SAMPLE_RATE

    starts = list(range(0, max(n_samples - CHUNK_OVERLAP_SECONDS * SAMPLE_RATE, 1), hop))
    if not starts:
        starts = [0]

    out = np.zeros((n_sources, audio.shape[0], n_samples), dtype=np.float32)
    weight = np.zeros(n_samples, dtype=np.float32)

    total_steps = len(starts) * len(models)
    step = 0
    for si, start in enumerate(starts):
        end = min(start + chunk, n_samples)
        seg = audio[:, start:end]
        seg_t = torch.from_numpy(np.expand_dims(seg, 0)).float().to(device)

        seg_outs = []
        for model in models:
            with torch.no_grad():
                res = apply_model(model, seg_t, shifts=1, overlap=overlap)[0].cpu().numpy()
            seg_outs.append(res)
            step += 1
            if progress_cb:
                progress_cb(step / total_steps)
        seg_out = np.mean(seg_outs, axis=0)

        # 三角窓でクロスフェード合成
        w = np.ones(end - start, dtype=np.float32)
        fade = CHUNK_OVERLAP_SECONDS * SAMPLE_RATE
        if si > 0:
            ramp = min(fade, end - start)
            w[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        if start + chunk < n_samples:
            ramp = min(fade, end - start)
            w[-ramp:] = np.minimum(w[-ramp:], np.linspace(1.0, 0.0, ramp, dtype=np.float32))
        out[:, :, start:end] += seg_out * w
        weight[start:end] += w

    weight = np.maximum(weight, 1e-8)
    out /= weight
    return out


engine = Engine()
