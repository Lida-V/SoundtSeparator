"""明示ダウンロードと推論時のオフライン動作の検証。"""
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import model_downloads as downloads


class DownloadManagerTests(unittest.TestCase):
    def test_verified_download_is_atomic_and_does_not_redownload(self):
        payload = b"fixture model"
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "output") as tmp:
            path = Path(tmp) / "weights.bin"
            spec = downloads._file(path, "https://example.test/weights", sha256=hashlib.sha256(payload).hexdigest(), size=len(payload))
            response = io.BytesIO(payload)
            response.headers = {"Content-Length": str(len(payload))}
            with patch("urllib.request.urlopen", return_value=response) as network:
                downloads._fetch(spec, lambda *_: None)
                self.assertEqual(path.read_bytes(), payload)
                self.assertEqual(list(Path(tmp).glob("*.partial")), [])
                network.assert_called_once()
            with patch("urllib.request.urlopen") as network:
                downloads._fetch(spec, lambda *_: None)
                network.assert_not_called()

    def test_bad_checksum_never_publishes_a_partial_file(self):
        payload = b"untrusted"
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "output") as tmp:
            path = Path(tmp) / "weights.bin"
            response = io.BytesIO(payload)
            response.headers = {"Content-Length": str(len(payload))}
            spec = downloads._file(path, "https://example.test/weights", sha256="0" * 64)
            with patch("urllib.request.urlopen", return_value=response), self.assertRaises(RuntimeError):
                downloads._fetch(spec, lambda *_: None)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(tmp).glob("*.partial")), [])

    def test_catalog_and_unknown_id_do_not_access_network(self):
        with patch("urllib.request.urlopen") as network:
            groups = downloads.catalog()
            self.assertTrue({"nemotron_reuse", "nemotron_diarization", "whisper_medium", "drumsep", "htdemucs_6s"} <= {g["id"] for g in groups})
            with self.assertRaises(ValueError):
                downloads.download("not-an-allowed-model", lambda *_: None)
            network.assert_not_called()

    def test_vram_guidance_distinguishes_legacy_estimates_and_measurements(self):
        groups = {entry["id"]: entry for entry in downloads.catalog()}
        self.assertIn("旧案内", groups["bandit"]["gpu_reference"])
        self.assertIn("RTX 4090実測", groups["bs_roformer"]["gpu_reference"])
        self.assertIn("約1.54 GiB予約", groups["dbs_dnr"]["gpu_reference"])
        self.assertIn("16 GB以上", groups["mega_53"]["gpu_reference"])
        self.assertIn("最低要件ではありません", " ".join(groups["dbs_dnr"]["requirements"]))
        self.assertIn("6 GB以上", groups["bandit"]["gpu_guidance"])
        self.assertIn("未検証", groups["bandit"]["gpu_guidance"])
        self.assertIn("8 GB以上", groups["bs_roformer"]["gpu_guidance"])
        self.assertIn("16 GB以上（モデル提供元の推奨）", groups["mega_53"]["gpu_guidance"])
        self.assertIn("未確定", groups["nemotron_diarization"]["gpu_guidance"])
        self.assertIn("GPUは不要", groups["basic_pitch"]["gpu_guidance"])
        self.assertEqual(groups["bandit"]["requirements"][0], groups["bandit"]["gpu_guidance"])
        self.assertEqual(groups["bandit"]["gpu_summary"], "推奨VRAM：6 GB以上（暫定）")
        self.assertEqual(groups["mega_53"]["gpu_summary"], "推奨VRAM：16 GB以上（公式推奨）")
        self.assertIn("未確定", groups["nemotron_diarization"]["gpu_summary"])
        self.assertIn("GPU不要", groups["basic_pitch"]["gpu_summary"])


if __name__ == "__main__":
    unittest.main()
