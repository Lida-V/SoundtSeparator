"""1プロセス1モデルのVRAM実測。取得済み重みだけを使う。

例: .venv/Scripts/python.exe docs/benchmarks/bench_vram.py bs_roformer 8 fast
実行前にGPUの利用状況を確認すること。入力は合成音で、実測値は最低要件ではない。
"""
from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
from separator import Engine

name = sys.argv[1]
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
quality = sys.argv[3] if len(sys.argv) > 3 else "standard"
sample_rate = 44100
t = np.arange(round(sample_rate * duration), dtype=np.float32) / sample_rate
audio = np.stack((0.1 * np.sin(2 * np.pi * 220 * t),
                  0.08 * np.sin(2 * np.pi * 330 * t))).astype(np.float32)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
engine = Engine()
torch.cuda.init()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
before = torch.cuda.mem_get_info()
start = time.perf_counter()
if name == "dbs_dnr":
    result = engine.separate(audio, mode="dme", quality=quality, model_id="dbs_dnr")
elif name == "bandit":
    result = engine.separate(audio, mode="dme", quality=quality, model_id="bandit")
elif name in ("bs_roformer", "mel_band_roformer"):
    result = engine.separate(audio, mode="vocals", quality=quality, model_id=name)
elif name == "music_bs":
    result = engine.separate(audio, mode="music", quality=quality, model_id="bs_roformer")
elif name == "mega_2":
    result = engine.separate(audio, mode="mega", quality=quality, model_id="mega_53", selected_stems=["acoustic-guitar", "piano"])
elif name == "drumsep":
    result = engine.subseparate(audio, kind="drums")
elif name == "karaoke":
    result = engine.subseparate(audio, kind="vocals")
elif name == "dereverb":
    result = {"noreverb": engine.dereverb(audio)}
elif name == "reuse":
    from reuse_adapter import enhance
    result = {"enhanced": enhance(audio, sample_rate=sample_rate)}
elif name == "nemotron":
    from transcriber import NemotronDiarizer
    result = {"turns": NemotronDiarizer().diarize(audio.T, sample_rate)}
else:
    raise SystemExit(f"unsupported model: {name}")
torch.cuda.synchronize()
used = torch.cuda.max_memory_allocated() / 1024**3
reserved = torch.cuda.max_memory_reserved() / 1024**3
after = torch.cuda.mem_get_info()
out = {
    "model": name,
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "sample_rate": sample_rate,
    "input_seconds": duration,
    "input_channels": 2,
    "quality": quality if name not in ("drumsep", "karaoke", "dereverb", "reuse", "nemotron") else "default",
    "selected_stems": ["acoustic-guitar", "piano"] if name == "mega_2" else None,
    "seconds_including_model_load": round(time.perf_counter() - start, 3),
    "torch_peak_allocated_gib": round(used, 3),
    "torch_peak_reserved_gib": round(reserved, 3),
    "gpu_free_before_gib": round(before[0] / 1024**3, 3),
    "gpu_free_after_gib": round(after[0] / 1024**3, 3),
    "output_keys": list(result),
}
print("BENCH_RESULT=" + json.dumps(out, ensure_ascii=False), flush=True)
