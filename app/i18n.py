# coding: utf-8
"""サーバー側ローカライズ (既存APIへの追加のみ・上流互換)。

方針:
- ja は正本。カタログ (description/notes) とステム label は ja の元データを
  そのまま返し、en / zh-CN のときだけ下の辞書で上書きする (?lang= または
  Accept-Language で解決)。
- ジョブメッセージとHTTPエラーは ja 文字列 (message / detail) を変えずに
  翻訳キー (status_key / detail_key + params) を併設し、クライアント側で
  翻訳する。古いクライアントは従来どおり ja 文字列だけを見る。
"""
SUPPORTED = ("ja", "en", "zh-CN")


def normalize(tag):
    """言語タグをサポート対象に正規化する。対象外は None。"""
    value = str(tag or "").strip().lower()
    if not value:
        return None
    if value == "ja" or value.startswith("ja-"):
        return "ja"
    if value.startswith("zh"):
        return "zh-CN"
    if value == "en" or value.startswith("en-"):
        return "en"
    return None


def _accept_language_tags(header):
    """Accept-Language を q値降順のタグ列にする。"""
    if not header:
        return []
    entries = []
    for part in header.split(","):
        tokens = part.split(";")
        tag = tokens[0].strip()
        if not tag:
            continue
        quality = 1.0
        for param in tokens[1:]:
            param = param.strip()
            if param.startswith("q="):
                try:
                    quality = float(param[2:])
                except ValueError:
                    pass
        entries.append((quality, tag))
    entries.sort(key=lambda item: -item[0])
    return [tag for _, tag in entries]


def resolve_lang(lang_param, accept_language=None):
    """?lang= を優先し、なければ Accept-Language、最後は ja。"""
    candidates = ([lang_param] if lang_param else []) + _accept_language_tags(accept_language)
    for candidate in candidates:
        locale = normalize(candidate)
        if locale:
            return locale
    return "ja"


class _Localized:
    """ja正本テンプレートとクライアント翻訳キーを併持する基底。"""

    def __init__(self, key, template, params=None):
        self.i18n_key = key
        self.i18n_params = params or None
        self.text = template.format(**params) if params else template


class Status(_Localized):
    """進捗メッセージ。ジョブの message (ja) と status_key に展開される。"""

    def __init__(self, key, template, **params):
        _Localized.__init__(self, key, template, params or None)


class LocalizedValueError(_Localized, ValueError):
    def __init__(self, key, template, **params):
        _Localized.__init__(self, key, template, params or None)
        ValueError.__init__(self, self.text)


class LocalizedRuntimeError(_Localized, RuntimeError):
    def __init__(self, key, template, **params):
        _Localized.__init__(self, key, template, params or None)
        RuntimeError.__init__(self, self.text)


class LocalizedFileNotFoundError(_Localized, FileNotFoundError):
    def __init__(self, key, template, **params):
        _Localized.__init__(self, key, template, params or None)
        FileNotFoundError.__init__(self, self.text)


# ---------------------------------------------------------------------------
# モデルカタログの翻訳 (キーは MODEL_CATALOG の id。ja は元データを使うため省略)
# ---------------------------------------------------------------------------
MODEL_TEXT = {
    "bandit": {
        "description": {
            "en": "Separates video dialogue, effects and BGM in stereo.",
            "zh-CN": "将视频中的对白、音效与BGM以立体声分离。",
        },
        "notes": {
            "en": "High quality runs an ensemble with CDX23 (extra download of about 162 MB).",
            "zh-CN": "高品质模式与CDX23集成（额外下载约162MB）。",
        },
    },
    "dbs_dnr": {
        "description": {
            "en": "Lightweight 3-stem video separation with input-adaptive band splitting.",
            "zh-CN": "基于输入自适应频带分割的轻量级视频三分轨分离。",
        },
        "notes": {
            "en": "Research model. Output is mono; stereo positioning is not preserved.",
            "zh-CN": "研究用模型。输出为单声道，不保留左右声像。",
        },
    },
    "bs_roformer": {
        "description": {
            "en": "Vocal separation. In 6-stem mode, instruments are also split with Demucs.",
            "zh-CN": "人声分离。六分轨模式下还会用Demucs分离乐器。",
        },
        "notes": {
            "en": "In 6-stem mode, also fetch htdemucs_6s (about 55 MB).",
            "zh-CN": "六分轨模式下还请先获取 htdemucs_6s（约55MB）。",
        },
    },
    "mel_band_roformer": {
        "description": {
            "en": "Vocal separation using mel bands. A candidate to compare against BS-RoFormer by ear.",
            "zh-CN": "利用梅尔频带的人声分离，可与BS-RoFormer对比试听。",
        },
        "notes": {
            "en": "Which model sounds better depends on the source. In 6-stem mode, htdemucs_6s is used as well.",
            "zh-CN": "模型优劣因音源而异。六分轨模式下会并用 htdemucs_6s。",
        },
    },
    "mega_53": {
        "description": {
            "en": "Pick and extract from 53 categories: strings, winds, keys, synths and more.",
            "zh-CN": "从弦乐、管乐、键盘、合成器等53个类别中选择并提取。",
        },
        "notes": {
            "en": "Categories overlap. Summing all stems does not reconstruct the original. A dedicated single-instrument model can be more accurate.",
            "zh-CN": "类别之间有重叠，全部分轨相加也无法还原原音源。专用模型可能精度更高。",
        },
    },
}

# ---------------------------------------------------------------------------
# ステム表示名の翻訳 (キーはステムの name。ja は separator.py の元データを使う)
# ---------------------------------------------------------------------------
STEM_LABELS = {
    # dme / vocals / music
    "dialog": {"en": "Dialogue", "zh-CN": "对白"},
    "effect": {"en": "Effects", "zh-CN": "音效"},
    "music": {"en": "Music", "zh-CN": "BGM"},
    "vocals": {"en": "Vocals", "zh-CN": "人声"},
    "instrumental": {"en": "Instrumental", "zh-CN": "伴奏"},
    "drums": {"en": "Drums", "zh-CN": "鼓"},
    "bass": {"en": "Bass", "zh-CN": "贝斯"},
    "guitar": {"en": "Guitar", "zh-CN": "吉他"},
    "piano": {"en": "Piano", "zh-CN": "钢琴"},
    "other": {"en": "Other", "zh-CN": "其他"},
    # トラック追加分離
    "kick": {"en": "Kick", "zh-CN": "底鼓"},
    "snare": {"en": "Snare", "zh-CN": "军鼓"},
    "toms": {"en": "Toms", "zh-CN": "嗵鼓"},
    "cymbals": {"en": "Cymbals", "zh-CN": "镲片"},
    "lead": {"en": "Lead vocal", "zh-CN": "主唱"},
    "back": {"en": "Backing vocals", "zh-CN": "伴唱"},
    # 話者・重なりステム (diarization_stems)
    "speaker_a": {"en": "Speaker A (solo)", "zh-CN": "说话人A（单独）"},
    "speaker_b": {"en": "Speaker B (solo)", "zh-CN": "说话人B（单独）"},
    "speaker_c": {"en": "Speaker C (solo)", "zh-CN": "说话人C（单独）"},
    "speaker_d": {"en": "Speaker D (solo)", "zh-CN": "说话人D（单独）"},
    "speaker_e": {"en": "Speaker E (solo)", "zh-CN": "说话人E（单独）"},
    "speaker_f": {"en": "Speaker F (solo)", "zh-CN": "说话人F（单独）"},
    "speaker_g": {"en": "Speaker G (solo)", "zh-CN": "说话人G（单独）"},
    "speaker_h": {"en": "Speaker H (solo)", "zh-CN": "说话人H（单独）"},
    "overlap": {"en": "Overlap (multiple speakers)", "zh-CN": "重叠（多说话人）"},
    "residual": {"en": "Undetected / other", "zh-CN": "未检出·其他"},
    # MVSep Mega 53
    "accordion": {"en": "Accordion", "zh-CN": "手风琴"},
    "acoustic-guitar": {"en": "Acoustic guitar", "zh-CN": "原声吉他"},
    "back-vocal": {"en": "Backing vocals", "zh-CN": "伴唱"},
    "banjo": {"en": "Banjo", "zh-CN": "班卓琴"},
    "bassoon": {"en": "Bassoon", "zh-CN": "巴松管"},
    "bells": {"en": "Bells", "zh-CN": "铃铛"},
    "bowed_strings": {"en": "Bowed strings", "zh-CN": "弓弦乐器"},
    "brass": {"en": "Brass", "zh-CN": "铜管乐器"},
    "cello": {"en": "Cello", "zh-CN": "大提琴"},
    "clarinet": {"en": "Clarinet", "zh-CN": "单簧管"},
    "congas": {"en": "Congas", "zh-CN": "康加鼓"},
    "digital-piano": {"en": "Digital piano", "zh-CN": "电钢琴"},
    "dobro": {"en": "Dobro", "zh-CN": "多布罗琴"},
    "double-bass": {"en": "Double bass", "zh-CN": "低音提琴"},
    "electric-guitar": {"en": "Electric guitar", "zh-CN": "电吉他"},
    "flute": {"en": "Flute", "zh-CN": "长笛"},
    "french-horn": {"en": "French horn", "zh-CN": "圆号"},
    "glockenspiel": {"en": "Glockenspiel", "zh-CN": "钟琴"},
    "harmonica": {"en": "Harmonica", "zh-CN": "口琴"},
    "harp": {"en": "Harp", "zh-CN": "竖琴"},
    "harpsichord": {"en": "Harpsichord", "zh-CN": "大键琴"},
    "hh": {"en": "Hi-hat", "zh-CN": "踩镲"},
    "keys": {"en": "Keys", "zh-CN": "键盘乐器"},
    "lead-vocal": {"en": "Lead vocal", "zh-CN": "主唱"},
    "mandolin": {"en": "Mandolin", "zh-CN": "曼陀林"},
    "marimba": {"en": "Marimba", "zh-CN": "马林巴"},
    "oboe": {"en": "Oboe", "zh-CN": "双簧管"},
    "organ": {"en": "Organ", "zh-CN": "管风琴"},
    "percussion": {"en": "Percussion", "zh-CN": "打击乐"},
    "saxophone": {"en": "Saxophone", "zh-CN": "萨克斯"},
    "sitar": {"en": "Sitar", "zh-CN": "西塔琴"},
    "strings": {"en": "Strings", "zh-CN": "弦乐"},
    "synth": {"en": "Synth", "zh-CN": "合成器"},
    "tambourine": {"en": "Tambourine", "zh-CN": "铃鼓"},
    "timpani": {"en": "Timpani", "zh-CN": "定音鼓"},
    "triangle": {"en": "Triangle", "zh-CN": "三角铁"},
    "trombone": {"en": "Trombone", "zh-CN": "长号"},
    "trumpet": {"en": "Trumpet", "zh-CN": "小号"},
    "tuba": {"en": "Tuba", "zh-CN": "大号"},
    "ukulele": {"en": "Ukulele", "zh-CN": "尤克里里"},
    "viola": {"en": "Viola", "zh-CN": "中提琴"},
    "violin": {"en": "Violin", "zh-CN": "小提琴"},
    "vocal": {"en": "All vocals", "zh-CN": "人声全体"},
    "wind": {"en": "Wind", "zh-CN": "管乐器"},
    "wind-chimes": {"en": "Wind chimes", "zh-CN": "风铃"},
    "woodwind": {"en": "Woodwind", "zh-CN": "木管乐器"},
}


def _tr(table, key, lang, fallback):
    entry = table.get(key)
    if entry:
        value = entry.get(lang)
        if value:
            return value
    return fallback


# ---------------------------------------------------------------------------
# モデル管理カタログの翻訳 (キーは model_downloads のグループ id。
# ja は model_downloads.py の元データを使うため省略。
# 画面に表示される label / description / gpu_summary のみ。requirements 等の
# 詳細テキストは README 側に集約されているためここでは扱わない)
# ---------------------------------------------------------------------------
DOWNLOADS_TEXT = {
    "bandit": {
        "description": {
            "en": "Separates speech, music and effects.",
            "zh-CN": "分离人声、音乐与音效。",
        },
        "gpu_summary": {
            "en": "Recommended VRAM: 6 GB+ (provisional)",
            "zh-CN": "推荐显存：6 GB以上（暂定）",
        },
    },
    "dbs_dnr": {
        "description": {
            "en": "Separates speech, music and effects (mono).",
            "zh-CN": "分离人声、音乐与音效（单声道）。",
        },
        "gpu_summary": {
            "en": "Recommended VRAM: 4 GB+ (provisional)",
            "zh-CN": "推荐显存：4 GB以上（暂定）",
        },
    },
    "bs_roformer": {
        "description": {
            "en": "Separates vocals from backing. 6-stem mode also needs HTDemucs.",
            "zh-CN": "分离人声与伴奏。六分轨模式还需 HTDemucs。",
        },
        "gpu_summary": {
            "en": "Recommended VRAM: 8 GB+ (provisional)",
            "zh-CN": "推荐显存：8 GB以上（暂定）",
        },
    },
    "mel_band_roformer": {
        "description": {
            "en": "Separates vocals from backing. 6-stem mode also needs HTDemucs.",
            "zh-CN": "分离人声与伴奏。六分轨模式还需 HTDemucs。",
        },
        "gpu_summary": {
            "en": "Recommended VRAM: 8 GB+ (provisional)",
            "zh-CN": "推荐显存：8 GB以上（暂定）",
        },
    },
    "mega_53": {
        "description": {
            "en": "Extracts targets from 53 overlapping categories.",
            "zh-CN": "从53个相互重叠的类别中提取目标。",
        },
        "gpu_summary": {
            "en": "Recommended VRAM: 16 GB+ (vendor recommendation)",
            "zh-CN": "推荐显存：16 GB以上（官方推荐）",
        },
    },
    "htdemucs_6s": {
        "label": {
            "en": "HTDemucs 6-stem",
            "zh-CN": "HTDemucs 六分轨",
        },
        "description": {
            "en": "Used alongside RoFormer in 6-stem music mode.",
            "zh-CN": "在音乐六分轨模式下与RoFormer并用。",
        },
        "gpu_summary": {
            "en": "Recommended VRAM: 8 GB+ (provisional)",
            "zh-CN": "推荐显存：8 GB以上（暂定）",
        },
    },
    "cdx23": {
        "description": {
            "en": "Three models used by BandIt high quality.",
            "zh-CN": "BandIt高品质模式使用的3个模型。",
        },
        "gpu_summary": {
            "en": "Recommended VRAM: undetermined",
            "zh-CN": "推荐显存：未确定",
        },
    },
    "drumsep": {
        "description": {
            "en": "Splits drums into kick, snare and more.",
            "zh-CN": "将鼓细分为底鼓、军鼓等。",
        },
        "gpu_summary": {
            "en": "Recommended VRAM: 6 GB+ (provisional)",
            "zh-CN": "推荐显存：6 GB以上（暂定）",
        },
    },
    "karaoke": {
        "description": {
            "en": "Estimates lead and backing vocals.",
            "zh-CN": "推断主唱与伴唱。",
        },
        "gpu_summary": {
            "en": "Recommended VRAM: 8 GB+ (provisional)",
            "zh-CN": "推荐显存：8 GB以上（暂定）",
        },
    },
    "dereverb": {
        "description": {
            "en": "Removes reverb.",
            "zh-CN": "去除混响。",
        },
        "gpu_summary": {
            "en": "Recommended VRAM: 8 GB+ (provisional)",
            "zh-CN": "推荐显存：8 GB以上（暂定）",
        },
    },
    "nemotron_diarization": {
        "description": {
            "en": "Estimates speaker regions for up to 8 speakers.",
            "zh-CN": "推断最多8位说话人的说话区间。",
        },
        "gpu_summary": {
            "en": "Recommended VRAM: undetermined (verified on RTX 4090)",
            "zh-CN": "推荐显存：未确定（已在RTX 4090上验证）",
        },
    },
    "nemotron_reuse": {
        "description": {
            "en": "Mono speech enhancement that suppresses noise and reverb.",
            "zh-CN": "抑制噪声和混响的单声道语音增强。",
        },
        "gpu_summary": {
            "en": "Recommended VRAM: undetermined (verified on RTX 4090)",
            "zh-CN": "推荐显存：未确定（已在RTX 4090上验证）",
        },
    },
    "whisper_small": {
        "description": {
            "en": "For transcription.",
            "zh-CN": "用于文字转录。",
        },
        "gpu_summary": {
            "en": "VRAM guide: ~2 GB (official)",
            "zh-CN": "显存参考：约2 GB（官方）",
        },
    },
    "whisper_medium": {
        "description": {
            "en": "For transcription.",
            "zh-CN": "用于文字转录。",
        },
        "gpu_summary": {
            "en": "VRAM guide: ~5 GB (official)",
            "zh-CN": "显存参考：约5 GB（官方）",
        },
    },
    "whisper_large": {
        "description": {
            "en": "For transcription.",
            "zh-CN": "用于文字转录。",
        },
        "gpu_summary": {
            "en": "VRAM guide: ~10 GB (official)",
            "zh-CN": "显存参考：约10 GB（官方）",
        },
    },
    "resemblyzer": {
        "description": {
            "en": "Classic speaker estimation. Weights bundled with the library.",
            "zh-CN": "传统说话人推断。权重内置于库中。",
        },
        "gpu_summary": {
            "en": "Recommended VRAM: undetermined. Also runs on CPU.",
            "zh-CN": "推荐显存：未确定。也可在CPU上运行。",
        },
    },
    "basic_pitch": {
        "description": {
            "en": "MIDI transcription. Weights bundled with the library.",
            "zh-CN": "MIDI记谱。权重内置于库中。",
        },
        "gpu_summary": {
            "en": "No GPU needed (runs on CPU)",
            "zh-CN": "无需GPU（CPU运行）",
        },
    },
}


def localize_stem(spec, lang):
    """ステム仕様の label を表示言語に合わせる (ja は元のまま)。"""
    label = _tr(STEM_LABELS, spec.get("name"), lang, spec.get("label"))
    return {**spec, "label": label}


def download_label(group, lang):
    """モデル管理グループの表示名を表示言語に合わせる (ja は元のまま)。"""
    return _tr(DOWNLOADS_TEXT.get(group.get("id"), {}), "label", lang, group.get("label", ""))


def localize_downloads(groups, lang):
    """/api/model-downloads のカタログ項目の表示テキストを差し替える。"""
    for group in groups:
        text = DOWNLOADS_TEXT.get(group.get("id"), {})
        group["label"] = download_label(group, lang)
        for field in ("description", "gpu_summary"):
            if group.get(field):
                group[field] = _tr(text, field, lang, group[field])
    return groups


def localize_catalog(models, lang):
    """/api/models のカタログ項目の description と notes を差し替える。"""
    for model in models:
        text = MODEL_TEXT.get(model.get("id"), {})
        model["description"] = _tr(text, "description", lang, model.get("description", ""))
        if isinstance(model.get("notes"), list) and model["notes"]:
            model["notes"] = [_tr(text, "notes", lang, model["notes"][0])]
        # gpu_summary は model_downloads 由来のため DOWNLOADS_TEXT で差し替える
        if model.get("gpu_summary"):
            model["gpu_summary"] = _tr(DOWNLOADS_TEXT.get(model.get("id"), {}), "gpu_summary",
                                       lang, model["gpu_summary"])
        if isinstance(model.get("stems"), list):
            model["stems"] = [localize_stem(stem, lang) for stem in model["stems"]]
    return models
