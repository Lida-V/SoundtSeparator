"""RE-USE を修復APIから適用し、モノラルと時間を確認する。"""
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import soundfile as sf
import torch
from fastapi import UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import reuse_adapter
import server


@unittest.skipUnless(torch.cuda.is_available() and reuse_adapter.downloaded(), "取得済みモデルとCUDAが必要")
class ReuseServerTests(unittest.TestCase):
    def test_real_model_effect_returns_mono_with_same_length(self):
        sr = 44100
        time = np.arange(sr // 2, dtype=np.float32) / sr
        samples = np.stack([.1 * np.sin(2 * np.pi * 250 * time),
                            .08 * np.sin(2 * np.pi * 300 * time)], axis=1)
        payload = io.BytesIO()
        sf.write(payload, samples, sr, format="WAV", subtype="FLOAT")
        response = server.apply_effect(file=UploadFile(io.BytesIO(payload.getvalue()), filename="sample.wav"),
                                       effect="nemotron_reuse")
        self.assertEqual(response.status_code, 200)
        result, out_rate = sf.read(io.BytesIO(response.body), dtype="float32", always_2d=True)
        self.assertEqual((len(result), result.shape[1], out_rate), (len(samples), 1, sr))
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
