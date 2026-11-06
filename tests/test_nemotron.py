"""話者区間の取り扱いと、モデルの暗黙ダウンロード防止を検証する。"""
import hashlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import transcriber


class NemotronDiarizationTests(unittest.TestCase):
    def test_diarize_preserves_overlapping_speakers_without_transcription(self):
        class FakeInputs(dict):
            attention_mask = None

            def to(self, *_args, **_kwargs):
                return self

        class FakeProcessor:
            feature_extractor = types.SimpleNamespace(sampling_rate=16000)

            def __call__(self, *_args, **_kwargs):
                return FakeInputs()

            def extract_speaker_dict(self, *_args):
                return [[{"Start": 0.1, "End": 0.8, "Speaker": 0},
                         {"Start": 0.5, "End": 1.2, "Speaker": 1}]]

        class FakeModel:
            device = torch.device("cpu")
            dtype = torch.float32

            def __call__(self, **_kwargs):
                return types.SimpleNamespace(logits=torch.zeros(1, 100, 8))

        diarizer = transcriber.NemotronDiarizer()
        with mock.patch.object(diarizer, "_load", return_value=(FakeProcessor(), FakeModel())):
            result = diarizer.diarize(np.zeros((16000, 2), dtype=np.float32), 16000)
        self.assertEqual(result, [{"start": 0.1, "end": 0.8, "speaker": "話者A"},
                                  {"start": 0.5, "end": 1.0, "speaker": "話者B"}])

    def test_assign_reuses_diarize_without_clobbering_other_segment_fields(self):
        segments = [{"start": 0., "end": 2., "text": "テスト"}]
        diarizer = transcriber.NemotronDiarizer()
        with mock.patch.object(diarizer, "diarize", return_value=[{"start": 0.5, "end": 1.5, "speaker": "話者B"}]) as run:
            result = diarizer.assign(np.zeros((10, 1), dtype=np.float32), 16000, segments)
        self.assertEqual(result[0], {"start": 0., "end": 2., "text": "テスト", "speaker": "話者B"})
        run.assert_called_once()

    def test_long_audio_stream_keeps_speaker_cache_across_chunks(self):
        class FakeInputs(dict):
            def to(self, *_args, **_kwargs):
                return self

        class FakeProcessor:
            feature_extractor = types.SimpleNamespace(sampling_rate=10)
            num_samples_first_audio_chunk = 12
            num_samples_per_audio_chunk = 12
            num_mel_frames_per_step = 10

            def __init__(self):
                self.final_calls = 0

            def set_streaming_mode(self, value):
                self.mode = value

            def audio_chunk_start(self, index):
                return index

            def __call__(self, *_args, **kwargs):
                self.final_calls += int(bool(kwargs.get("is_last_audio_chunk")))
                return FakeInputs()

            def extract_speaker_dict(self, logits):
                self.logits = logits
                return [[{"Start": 2.0, "End": 20.0, "Speaker": 0},
                         {"Start": 19.0, "End": 31.0, "Speaker": 1}]]

        class FakeModel:
            device = torch.device("cpu")
            dtype = torch.float32

            def __init__(self):
                self.caches = []

            def __call__(self, **kwargs):
                self.caches.append(kwargs.get("speaker_cache"))
                return types.SimpleNamespace(logits=torch.zeros((1, 1, 8)), speaker_cache=len(self.caches))

        diarizer, processor, model = transcriber.NemotronDiarizer(), FakeProcessor(), FakeModel()
        progress = []
        with mock.patch.object(diarizer, "_load", return_value=(processor, model)):
            turns = diarizer.diarize(np.zeros((320, 1), dtype=np.float32), 10, lambda pct, _msg: progress.append(pct))
        self.assertEqual(processor.mode, "low_latency")
        self.assertEqual(processor.final_calls, 1)
        self.assertGreater(len(model.caches), 2)
        self.assertIsNone(model.caches[0])
        self.assertEqual(model.caches[1:], list(range(1, len(model.caches))))
        self.assertEqual(tuple(processor.logits.shape), (1, len(model.caches), 8))
        self.assertEqual([turn["speaker"] for turn in turns], ["話者A", "話者B"])
        self.assertLess(turns[1]["start"], turns[0]["end"])
        self.assertEqual(progress[-1], 100)
        self.assertEqual(progress, sorted(progress))

    def test_diarize_rejects_invalid_audio_before_model_loading(self):
        diarizer = transcriber.NemotronDiarizer()
        with mock.patch.object(diarizer, "_load") as load:
            for audio, rate in ((np.zeros((100, 3)), 16000),
                                (np.array([[np.nan]]), 16000),
                                (np.zeros((16000, 1)), 0)):
                with self.subTest(shape=audio.shape, rate=rate), self.assertRaises(ValueError):
                    diarizer.diarize(audio, rate)
            load.assert_not_called()

    def test_longest_total_overlap_and_unmatched(self):
        segments = [{"start": 0, "end": 2}, {"start": 2, "end": 4},
                    {"start": 9, "end": 10}]
        turns = [{"start": 0, "end": 0.6, "speaker": "話者A"},
                 {"start": 0.6, "end": 1.5, "speaker": "話者B"},
                 {"start": 1.5, "end": 1.9, "speaker": "話者A"},
                 {"start": 2.1, "end": 3.9, "speaker": "話者C"}]
        out = transcriber._assign_diarization_segments(segments, turns)
        self.assertEqual([s["speaker"] for s in out], ["話者A", "話者C", ""])

    def test_missing_nemotron_never_imports_transformers(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            with mock.patch.object(transcriber, "MODEL_ROOT", Path(tmp)):
                with self.assertRaisesRegex(RuntimeError, "モデル管理"):
                    transcriber.NemotronDiarizer()._load()

    def test_whisper_requires_verified_local_file(self):
        content = b"test model content"
        sha = hashlib.sha256(content).hexdigest()
        fake = types.SimpleNamespace(_MODELS={"small": f"https://example.invalid/{sha}/small.pt"},
                                     load_model=mock.Mock(return_value="loaded"))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            with mock.patch.object(transcriber, "MODEL_ROOT", root), \
                 mock.patch.object(Path, "home", return_value=root / "home"), \
                 mock.patch.dict(os.environ, {"XDG_CACHE_HOME": str(root / "cache")}), \
                 mock.patch.dict(sys.modules, {"whisper": fake}):
                engine = transcriber.Transcriber()
                with self.assertRaisesRegex(RuntimeError, "モデル管理"):
                    engine._get_model("small")
                fake.load_model.assert_not_called()
                target = root / "whisper" / "small.pt"
                target.parent.mkdir()
                target.write_bytes(content)
                self.assertEqual(engine._get_model("small"), "loaded")
                self.assertEqual(fake.load_model.call_args.args[0], str(target))

                target.unlink()
                cached = root / "cache" / "whisper" / "small.pt"
                cached.parent.mkdir(parents=True)
                cached.write_bytes(content)
                engine = transcriber.Transcriber()
                self.assertEqual(engine._get_model("small"), "loaded")
                self.assertEqual(fake.load_model.call_args.args[0], str(cached))


if __name__ == "__main__":
    unittest.main()
