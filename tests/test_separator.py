"""推論の入出力・選択ヘッド・モデルファイル保全を、追加DLなしで検証。"""
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import separator as sep


class ModelSelectionTests(unittest.TestCase):
    def test_default_is_backward_compatible(self):
        self.assertEqual(sep.validate_request("dme")[0], "bandit")
        self.assertEqual(sep.validate_request("music")[0], "bs_roformer")
        self.assertEqual(len(sep.validate_request("music")[1]), 6)

    def test_wrong_mode_model_and_unknown_stems_are_rejected(self):
        for args in [("unknown", None, None), ("dme", "mega_53", None),
                     ("mega", None, ["piano", "missing"]),
                     ("mega", None, ["piano", "piano"]), ("mega", None, [])]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                sep.validate_request(*args)

    def test_selected_stems_keep_catalog_order(self):
        _, specs = sep.validate_request("mega", None, ["synth", "piano"])
        self.assertEqual([item["name"] for item in specs], ["piano", "synth"])
        specs[0]["label"] = "modified"
        self.assertNotEqual(sep.validate_request("mega", None, ["piano"])[1][0]["label"], "modified")

    def test_mega_catalog_matches_fixed_upstream_config(self):
        import yaml
        # モデルを構築せず、head番号と公式configの順序が一致することを確認。
        with open(sep.MEGA_CKPT["config"], encoding="utf-8") as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        self.assertEqual(config["training"]["instruments"], list(sep._MEGA_LABELS))
        self.assertEqual(len(sep.MEGA_STEMS), 53)


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.engine = sep.Engine()
        self.engine.device = "cpu"
        self.audio = np.full((2, sep.SAMPLE_RATE), 0.2, dtype=np.float32)

    def test_vocals_pair_reconstructs_after_length_fit(self):
        config = types.SimpleNamespace(inference=types.SimpleNamespace(num_overlap=2))
        short_vocals = np.full((2, sep.SAMPLE_RATE - 3), 0.07, dtype=np.float32)
        with patch.object(self.engine, "_get_msst", return_value=(object(), config)), \
                patch.object(self.engine, "_run_msst", return_value={"vocals": short_vocals}):
            result = self.engine.separate(self.audio, mode="vocals", model_id="mel_band_roformer")
        np.testing.assert_allclose(result["vocals"] + result["instrumental"], self.audio, atol=1e-7)
        self.assertEqual(result["vocals"].shape, self.audio.shape)
        self.assertEqual(result["vocals"].dtype, np.float32)
        np.testing.assert_array_equal(result["vocals"][:, -3:], 0)

    def test_mono_model_output_remains_mono_for_stereo_input(self):
        result = {name: np.zeros((1, sep.SAMPLE_RATE), dtype=np.float32) for name in ("dialog", "effect", "music")}
        with patch("dbs_adapter.load_model", return_value=object()), patch("dbs_adapter.separate", return_value=result):
            output = self.engine.separate(self.audio, model_id="dbs_dnr")
        self.assertTrue(all(value.shape == (1, sep.SAMPLE_RATE) for value in output.values()))

    def test_invalid_audio_is_rejected_before_loading(self):
        with patch.object(self.engine, "_get_msst") as load:
            for audio in [np.zeros((2, 10)), np.zeros((3, sep.SAMPLE_RATE)),
                          np.full(self.audio.shape, np.nan), np.full(self.audio.shape, np.inf)]:
                with self.subTest(shape=audio.shape), self.assertRaises(ValueError):
                    self.engine.separate(audio)
        load.assert_not_called()

    def test_nonfinite_model_output_is_rejected(self):
        outputs = {item["name"]: self.audio.copy() for item in sep.DME_STEMS}
        outputs["effect"][0, 0] = np.nan
        with patch.object(self.engine, "_separate_dme", return_value=outputs), self.assertRaises(RuntimeError):
            self.engine.separate(self.audio)

    def test_selected_mega_heads_have_correct_order_and_do_not_mutate_config(self):
        from ml_collections import ConfigDict
        config = ConfigDict({"training": {"instruments": list(sep._MEGA_LABELS), "target_instrument": None},
                             "inference": {"num_overlap": 2}})
        selected = ["piano", "synth"]
        observed = {}

        class FakeMega(torch.nn.Module):
            def forward(self, audio, active_stem_ids=None):
                observed["indices"] = active_stem_ids
                return audio[:, None].repeat(1, len(active_stem_ids), 1, 1)

        def run(wrapper, selected_config, kind, audio, cb):
            observed["names"] = list(selected_config.training.instruments)
            wrapper(torch.zeros(1, 2, 8))
            return {name: audio for name in selected_config.training.instruments}

        with patch.object(self.engine, "_get_msst", return_value=(FakeMega(), config)), \
                patch.object(self.engine, "_run_msst", side_effect=run):
            result = self.engine.separate(self.audio, "mega", selected_stems=selected)
        self.assertEqual(observed["names"], selected)
        self.assertEqual(observed["indices"], [list(sep._MEGA_LABELS).index(name) for name in selected])
        self.assertEqual(list(result), selected)
        self.assertEqual(len(config.training.instruments), 53)

    def test_model_release_does_not_interrupt_running_job(self):
        self.engine._lock.acquire()
        try:
            self.assertFalse(self.engine.release_models())
        finally:
            self.engine._lock.release()
        self.engine._cache["sample"] = object()
        self.assertTrue(self.engine.release_models())
        self.assertEqual(self.engine._cache, {})


class DownloadTests(unittest.TestCase):
    def test_download_is_atomic_and_checks_digest(self):
        payload = b"public model data"
        response = io.BytesIO(payload)
        response.headers = {"Content-Length": str(len(payload))}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, patch.object(sep, "MODELS_DIR", directory), \
                patch("urllib.request.urlopen", return_value=response):
            path = sep._download("https://example.com/model", "model.ckpt", sha256=hashlib.sha256(payload).hexdigest())
            self.assertEqual(Path(path).read_bytes(), payload)
            self.assertEqual([item.name for item in Path(directory).iterdir()], ["model.ckpt"])

    def test_partial_or_wrong_digest_is_never_installed(self):
        for length, digest in [(1000, None), (4, "0" * 64)]:
            response = io.BytesIO(b"data")
            response.headers = {"Content-Length": str(length)}
            with self.subTest(length=length), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, \
                    patch.object(sep, "MODELS_DIR", directory), patch("urllib.request.urlopen", return_value=response):
                with self.assertRaises(RuntimeError):
                    sep._download("https://example.com/model", "model.ckpt", sha256=digest)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_corrupt_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, patch.object(sep, "MODELS_DIR", directory), \
                patch("urllib.request.urlopen") as network:
            target = Path(directory) / "model.ckpt"
            target.write_bytes(b"existing")
            with self.assertRaises(RuntimeError):
                sep._download("https://example.com/model", "model.ckpt", sha256="0" * 64)
            self.assertEqual(target.read_bytes(), b"existing")
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
