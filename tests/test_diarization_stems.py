"""時間振り分けの重複扱い、フェード、および音声の足し戻しを検証する。"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from diarization_stems import iter_diarization_stems, summarize_diarization_activity


class DiarizationStemTests(unittest.TestCase):
    def test_single_overlap_residual_and_stereo(self):
        rate = 100
        source = np.column_stack([np.arange(100) / 100, -np.arange(100) / 200]).astype(np.float32)
        turns = [{"start": .1, "end": .5, "speaker": "話者A"},
                 {"start": .3, "end": .7, "speaker": "話者B"}]
        stems = dict(iter_diarization_stems(source, rate, turns, fade_ms=0))
        self.assertEqual(set(stems), {"speaker_a", "speaker_b", "overlap", "residual"})
        for name, start, end in (("speaker_a", 10, 30), ("overlap", 30, 50),
                                 ("speaker_b", 50, 70)):
            np.testing.assert_array_equal(stems[name][start:end], source[start:end])
            self.assertFalse(np.any(stems[name][:start]))
            self.assertFalse(np.any(stems[name][end:]))
        np.testing.assert_array_equal(sum(stems.values()), source)
        summary = summarize_diarization_activity(len(source), rate, turns)
        self.assertEqual(summary, {"overlap_samples": 20, "residual_samples": 40,
                                   "solo_samples": {"speaker_a": 20, "speaker_b": 20}})

    def test_same_speaker_overlapping_turns_are_not_two_speakers(self):
        source = np.ones((100, 1), np.float32)
        turns = [{"start": .1, "end": .5, "speaker": "話者A"},
                 {"start": .3, "end": .8, "speaker": "話者A"},
                 {"start": .3, "end": .8, "speaker": "話者A"}]
        stems = dict(iter_diarization_stems(source, 100, turns, fade_ms=0))
        self.assertEqual(float(stems["overlap"].sum()), 0)
        self.assertEqual(float(stems["speaker_a"].sum()), 70)
        summary = summarize_diarization_activity(100, 100, turns)
        self.assertEqual(summary["overlap_samples"], 0)
        self.assertEqual(summary["solo_samples"], {"speaker_a": 70})
        self.assertEqual(summary["residual_samples"], 30)

    def test_crossfades_preserve_sum_even_at_very_short_intervals(self):
        source = np.ones((2000, 2), np.float32)
        source[:, 1] *= -.3
        turns = [{"start": .0, "end": .501, "speaker": "話者A"},
                 {"start": .5, "end": .502, "speaker": "話者B"},
                 {"start": .503, "end": 1.8, "speaker": "話者C"}]
        stems = dict(iter_diarization_stems(source, 1000, turns, fade_ms=10))
        np.testing.assert_allclose(sum(stems.values()), source, atol=1e-7)
        for stem in stems.values():
            self.assertGreaterEqual(float(stem[:, 0].min()), 0)
            self.assertLessEqual(float(stem[:, 0].max()), 1)
            np.testing.assert_allclose(stem[:, 1], -.3 * stem[:, 0], atol=1e-7)

    def test_clip_bounds_and_empty_activity(self):
        source = np.ones((100, 1), np.float32)
        turns = [{"start": -9, "end": .1, "speaker": "話者H"},
                 {"start": 3, "end": 5, "speaker": "話者B"}]
        stems = dict(iter_diarization_stems(source, 100, turns, fade_ms=0))
        self.assertEqual(set(stems), {"speaker_h", "overlap", "residual"})
        self.assertEqual(float(stems["speaker_h"].sum()), 10)
        silent = dict(iter_diarization_stems(source, 100, [], fade_ms=0))
        self.assertEqual(set(silent), {"overlap", "residual"})
        np.testing.assert_array_equal(silent["residual"], source)

    def test_bad_audio_and_turns_rejected(self):
        bad_inputs = [(np.zeros((0, 2)), 100, []), (np.zeros(100), 100, []),
                      (np.zeros((100, 3)), 100, []), (np.array([[np.nan]]), 100, []),
                      (np.zeros((100, 1)), 0, []),
                      (np.zeros((100, 1)), 100, [{"start": 1, "end": 0, "speaker": "話者A"}]),
                      (np.zeros((100, 1)), 100, [{"start": 0, "end": np.inf, "speaker": "話者A"}]),
                      (np.zeros((100, 1)), 100, [{"start": 0, "end": 1, "speaker": "話者I"}])]
        for audio, rate, turns in bad_inputs:
            with self.subTest(rate=rate, turns=turns), self.assertRaises(ValueError):
                list(iter_diarization_stems(audio, rate, turns))


if __name__ == "__main__":
    unittest.main()
