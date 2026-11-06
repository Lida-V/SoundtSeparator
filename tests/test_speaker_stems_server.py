"""話者の単独・重なり・未検出ステムのジョブ境界を検証する。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import server


class SpeakerStemJobTests(unittest.TestCase):
    def test_stereo_stems_are_retrievable_and_sum_to_input(self):
        sr = 8000
        t = np.arange(sr * 3, dtype=np.float32) / sr
        audio = np.stack([.1 * np.sin(2 * np.pi * 220 * t),
                          .15 * np.sin(2 * np.pi * 340 * t)], axis=1)
        turns = [{"start": .2, "end": 1.5, "speaker": "話者A"},
                 {"start": 1., "end": 2., "speaker": "話者B"}]
        root = Path(__file__).resolve().parents[1] / "output"
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            job_id = "test_speaker_job"
            job_dir = Path(tmp) / job_id
            job_dir.mkdir()
            wav = job_dir / "input.wav"
            sf.write(wav, audio, sr, subtype="FLOAT")
            with patch.object(server, "OUTPUT_DIR", tmp), \
                 patch.object(server.nemotron_diarizer, "diarize", return_value=turns):
                with server.jobs_lock:
                    server.jobs[job_id] = {"id": job_id, "kind": "speaker_stems", "status": "queued"}
                try:
                    server.process_speaker_stems_job(job_id, str(wav))
                    job = server.get_job(job_id)
                    self.assertEqual(job["status"], "done", job.get("message"))
                    self.assertEqual({s["name"] for s in job["stems"]},
                                     {"speaker_a", "speaker_b", "overlap", "residual"})
                    total = np.zeros_like(audio)
                    for item in job["stems"]:
                        server.get_audio(job_id, item["name"] + ".wav")
                        wave, sample_rate = sf.read(job_dir / (item["name"] + ".wav"), dtype="float32", always_2d=True)
                        self.assertEqual((sample_rate, wave.shape), (sr, audio.shape))
                        total += wave
                    np.testing.assert_allclose(total, audio, atol=3e-7)
                    self.assertEqual(job["warnings"], [])
                finally:
                    with server.jobs_lock:
                        server.jobs.pop(job_id, None)


if __name__ == "__main__":
    unittest.main()
