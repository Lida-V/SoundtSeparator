"""モデルをロードせず、API境界と区間の正確さを検証する回帰テスト。"""
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import server


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output_patch = patch.object(server, "OUTPUT_DIR", self.temp.name)
        self.output_patch.start()
        with server.jobs_lock:
            server.jobs.clear()

    def tearDown(self):
        with server.jobs_lock:
            server.jobs.clear()
        self.output_patch.stop()
        self.temp.cleanup()

    def create(self, **overrides):
        args = dict(file=UploadFile(io.BytesIO(b"test"), filename="test.wav"),
                    mode="vocals", quality="fast", model_id="bs_roformer",
                    selected_stems=None, preview_start=0.0, preview_duration=None)
        args.update(overrides)
        with patch.object(server.threading, "Thread"):
            return server.create_job(**args)

    def test_rejects_invalid_settings_before_creating_job(self):
        for values in ({"quality": "typo"}, {"model_id": "dbs_dnr"},
                       {"selected_stems": '"vocals"'}, {"selected_stems": "[]"},
                       {"selected_stems": "not-json"}, {"preview_duration": 4},
                       {"preview_start": 1}, {"preview_start": math.inf},
                       {"preview_duration": math.nan}):
            with self.subTest(values=values), self.assertRaises(HTTPException) as error:
                self.create(**values)
            self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.assertEqual(server.jobs, {})

    def test_selected_stem_is_preserved_in_job(self):
        job_id = self.create(selected_stems='["instrumental"]', preview_start=2,
                             preview_duration=5)["id"]
        job = server.get_job(job_id)
        self.assertEqual(job["selected_stems"], ["instrumental"])
        self.assertEqual(job["preview_start"], 2)
        self.assertTrue(job["is_preview"])
        self.assertEqual(job["model_id"], "bs_roformer")
        self.assertEqual(job["status"], "queued")

    def test_dynamic_stem_serving_and_zip_manifest(self):
        job_id = self.create(mode="mega", model_id="mega_53", selected_stems='["acoustic-guitar"]',
                             preview_start=2, preview_duration=5)["id"]
        path = Path(self.temp.name) / job_id
        sf.write(path / "acoustic-guitar.wav", np.zeros((100, 2)), 44100, subtype="FLOAT")
        server.set_job(job_id, status="done", duration=5, output_channels=2)
        job = server.get_job(job_id)
        self.assertEqual([s["name"] for s in job["stems"]], ["acoustic-guitar"])
        self.assertTrue(Path(server.get_audio(job_id, "acoustic-guitar.wav").path).exists())
        with self.assertRaises(HTTPException):
            server.get_audio(job_id, "vocal.wav")
        with self.assertRaises(HTTPException):
            server.get_audio(job_id, "../original.wav")
        result = server.download_zip(job_id)
        with zipfile.ZipFile(result.path) as archive:
            self.assertEqual(len(archive.namelist()), 2)
            manifest = json.loads(archive.read("separation.json"))
            self.assertEqual(manifest["selected_stems"], ["acoustic-guitar"])
            self.assertEqual(manifest["preview_start"], 2)
            self.assertEqual(manifest["wav_subtype"], "FLOAT")

    def test_pending_job_cannot_be_downloaded(self):
        job_id = self.create()["id"]
        with self.assertRaises(HTTPException):
            server.download_zip(job_id)
        with self.assertRaises(HTTPException):
            server.get_audio(job_id, "original.wav")

    def test_preview_decode_has_exact_start_and_length(self):
        sr = 44100
        time = np.arange(sr * 8) / sr
        audio = np.stack((0.2 * np.sin(2 * np.pi * 440 * time),
                          0.1 * np.sin(2 * np.pi * 880 * time)), axis=1).astype(np.float32)
        src, dst = Path(self.temp.name) / "input.wav", Path(self.temp.name) / "clip.wav"
        sf.write(src, audio, sr, subtype="FLOAT")
        server.decode_to_wav(str(src), str(dst), 1.25, 5)
        clip, actual_sr = sf.read(dst, dtype="float32", always_2d=True)
        self.assertEqual(actual_sr, sr)
        np.testing.assert_allclose(clip, audio[int(1.25 * sr):int(6.25 * sr)], atol=1e-6)

    def test_busy_model_release_returns_conflict(self):
        with patch.object(server.engine, "release_models", return_value=False):
            with self.assertRaises(HTTPException) as error:
                server.release_models()
        self.assertEqual(error.exception.status_code, 409)

    def test_model_resampling_preserves_duration_and_frequency(self):
        sr = 48000
        time = np.arange(sr, dtype=np.float32) / sr
        audio = np.stack([0.2 * np.sin(2 * np.pi * 440 * time)] * 2, axis=1)
        result = server.resample_audio(audio, sr, 44100)
        self.assertEqual(result.shape, (44100, 2))
        peak = np.argmax(np.abs(np.fft.rfft(result[:, 0])))
        self.assertEqual(peak, 440)
        restored = server.resample_audio(result, 44100, sr)
        self.assertEqual(restored.shape, audio.shape)


if __name__ == "__main__":
    unittest.main()
