"""RE-USEの音声I/Oと明示DLの契約。実音声の品質を評価するテストではない。"""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import reuse_adapter


class _Identity(torch.nn.Module):
    def forward(self, magnitude, phase):
        return magnitude, phase, None


class ReuseAdapterTest(unittest.TestCase):
    def setUp(self):
        config = json.loads(reuse_adapter.CONFIG.read_text(encoding="utf-8"))
        self.model = (_Identity(), config, "cpu")

    def test_mono_mix_length_and_short_tail_preserved(self):
        audio = np.random.default_rng(42).normal(0, 0.02, (2, 12345)).astype(np.float32)
        out = reuse_adapter.enhance(audio, 8000, model_bundle=self.model)
        self.assertEqual(out.shape, (1, audio.shape[1]))
        np.testing.assert_allclose(out[0], audio.mean(axis=0), atol=1e-6, rtol=1e-5)

    def test_chunk_overlap_preserves_mono_audio(self):
        audio = np.random.default_rng(43).normal(0, 0.02, (1, 23457)).astype(np.float32)
        out = reuse_adapter.enhance(audio, 8000, model_bundle=self.model, chunk_seconds=1)
        np.testing.assert_allclose(out, audio, atol=1e-6, rtol=1e-5)

    def test_missing_model_fails_without_download(self):
        with self.assertRaisesRegex(FileNotFoundError, "ダウンロード"):
            reuse_adapter.load_model(Path(__file__).parent / "does-not-exist.safetensors")

    def test_input_validation(self):
        for audio, sample_rate in ((np.zeros((3, 100), dtype=np.float32), 44100),
                                   (np.array([[float("nan")]], dtype=np.float32), 44100),
                                   (np.zeros((1, 100), dtype=np.float32), 1000)):
            with self.subTest(shape=audio.shape, sample_rate=sample_rate):
                with self.assertRaises(ValueError):
                    reuse_adapter.enhance(audio, sample_rate, model_bundle=self.model)


if __name__ == "__main__":
    unittest.main()
