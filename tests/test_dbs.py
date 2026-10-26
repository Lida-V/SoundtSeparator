"""DBS のチャンネル変換・時間整合・キャッシュ検証。モデルの自動DLはしない。"""
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import dbs_adapter

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "output"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


class ProportionalModel(torch.nn.Module):
    """出力順と窓の足し合わせを独立に検証するスタブ。"""
    def forward(self, audio):
        return torch.cat([audio * .2, audio * .3, audio * .5], dim=1)


class DBSAdapterTests(unittest.TestCase):
    def test_channel_order_length_and_chunk_boundaries(self):
        for quality in ("fast", "standard", "high"):
            for seconds in (.5, 2, 4, 6, 7, 13):
                with self.subTest(quality=quality, seconds=seconds):
                    count = int(seconds * 44100)
                    time = np.arange(count, dtype=np.float32) / 44100
                    audio = np.stack([.2 * np.sin(time * 440), .1 * np.cos(time * 560)])
                    result = dbs_adapter.separate(audio, quality=quality, model=ProportionalModel())
                    self.assertEqual(list(result), ["dialog", "effect", "music"])
                    for stem in result.values():
                        self.assertEqual(stem.shape, (1, count))
                        self.assertTrue(np.isfinite(stem).all())
                    np.testing.assert_allclose(sum(result.values())[0], audio.mean(0), atol=2e-8)
                    np.testing.assert_allclose(result["dialog"][0], audio.mean(0) * .5, atol=2e-8)

    def test_invalid_audio_is_rejected(self):
        for audio in (np.zeros(20), np.zeros((2, 0)), np.zeros((3, 20)),
                      np.array([[np.nan]]), np.array([[np.inf]])):
            with self.subTest(shape=audio.shape), self.assertRaises(ValueError):
                dbs_adapter.separate(audio, model=ProportionalModel())

    def test_download_verifies_hash_and_does_not_reuse_partial_files(self):
        data = b"official weights fixture"
        spec = {"file": "fixture.bin", "url": "https://example.test/fixture.bin",
                "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            folder = Path(temporary)
            with patch.object(dbs_adapter, "MODELS_DIR", folder):
                with patch.object(dbs_adapter.urllib.request, "urlopen", return_value=io.BytesIO(data)):
                    path = dbs_adapter._checked_file(spec)
                self.assertEqual(path.read_bytes(), data)
                self.assertEqual(list(folder.glob("*.partial")), [])
                with patch.object(dbs_adapter.urllib.request, "urlopen") as network:
                    self.assertEqual(dbs_adapter._checked_file(spec), path)
                    network.assert_not_called()
                path.write_bytes(b"corrupt")
                with self.assertRaises(RuntimeError):
                    dbs_adapter._checked_file(spec)
                self.assertEqual(path.read_bytes(), b"corrupt")

    def test_failed_download_cleans_only_its_temporary_file(self):
        spec = {"file": "fixture.bin", "url": "https://example.test/fixture.bin",
                "size": 20, "sha256": "0" * 64}
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            folder = Path(temporary)
            unrelated = folder / "unrelated.partial"
            unrelated.write_bytes(b"keep")
            with patch.object(dbs_adapter, "MODELS_DIR", folder):
                with patch.object(dbs_adapter.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")):
                    with self.assertRaises(RuntimeError):
                        dbs_adapter._checked_file(spec)
            self.assertEqual(list(folder.iterdir()), [unrelated])

    @unittest.skipUnless(torch.cuda.is_available() and dbs_adapter.downloaded()
                         and importlib.util.find_spec("safetensors"),
                         "CUDA と取得済みの DBS 重みがあるときだけ実行する")
    def test_official_weights_gpu_smoke(self):
        time = np.arange(8 * 44100, dtype=np.float32) / 44100
        audio = np.stack([.1 * np.sin(2 * np.pi * 440 * time),
                          .1 * np.sin(2 * np.pi * 660 * time)])
        model = dbs_adapter.load_model("cuda:0")
        result = dbs_adapter.separate(audio, device="cuda:0", model=model)
        for stem in result.values():
            self.assertEqual(stem.shape, (1, audio.shape[1]))
            self.assertTrue(np.isfinite(stem).all())
        np.testing.assert_allclose(sum(result.values())[0], audio.mean(0), atol=2e-6)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
