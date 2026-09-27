"""Tiny synthetic optimizer steps verify plumbing, not research performance."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import torch
from PIL import Image

from riverlab import classification as cls
from riverlab import segmentation as seg
from riverlab.data import validate_splits
from riverlab.io import write_csv, write_json, read_json, read_csv, load_weights, seed_all, digest
from riverlab.models import segmentation_model
from riverlab.cli import main
from riverlab.reconstruct import reconstruct


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="riverlab_test_")
        self.root = Path(self.temp.name)
        seed_all(42, 2)

    def tearDown(self):
        self.temp.cleanup()

    def segmentation_rows(self):
        rows = []
        for i in range(8):
            image = np.zeros((40, 48, 3), np.uint8)
            mask = np.zeros((40, 48), np.uint8)
            mask[10:25, 5:43] = 1
            image[..., 1] = mask * 180
            a, b = self.root / ("im%d.png" % i), self.root / ("mask%d.png" % i)
            Image.fromarray(image).save(a)
            Image.fromarray(mask).save(b)
            rows.append(dict(sample_id=str(i), parent_id=str(i), camera_id="day",
                             image_path=str(a), mask_path=str(b), group_id="g"+str(i),
                             split="train" if i < 4 else "val" if i < 6 else "test",
                             timestamp="2022-06-{:02d}T12:00:00".format(i+1), mode="day"))
        return rows

    def test_segmentation_train_evaluate_and_original_resolution(self):
        rows = self.segmentation_rows()
        manifest = self.root / "seg.csv"
        write_csv(manifest, rows)
        config = dict(manifest=str(manifest), model={"name": "unet", "width": 2},
                      input_size=32, epochs=1, batch_size=2, workers=0, threads=2, seed=42,
                      loss={"dice_weight": .5, "boundary_weight": .1, "focal_gamma": 2}, bootstrap=10)
        train_dir = seg.train(config, self.root / "train", "cpu")
        cp = load_weights(train_dir / "best.pt")
        self.assertEqual(cp["provenance"]["train_ids"], ["0", "1", "2", "3"])
        # Evaluation must not train, and test images are returned at original size.
        with patch.object(torch.Tensor, "backward", side_effect=AssertionError("test trained")):
            metrics = seg.evaluate(train_dir / "best.pt", manifest, self.root / "eval", "cpu")
        self.assertEqual(metrics["n_images"], 2)
        self.assertEqual(read_json(train_dir / "run.json")["status"], "complete")
        prediction = read_csv(self.root / "eval" / "predictions.csv")[0]
        with Image.open(prediction["prediction_path"]) as im:
            self.assertEqual(im.size, (48, 40))
        # The original-resolution evaluator refuses a reused development date.
        rows[-1]["timestamp"] = rows[0]["timestamp"]
        write_csv(manifest, rows)
        with self.assertRaises(ValueError):
            seg.evaluate(train_dir / "best.pt", manifest, self.root / "bad", "cpu", split="test")

    def test_classifier_baselines_and_frozen_threshold(self):
        rows = []
        for i in range(18):
            label = i % 2
            r = dict(sample_id=str(i), parent_id=str(i), group_id="g"+str(i),
                     timestamp="2022-07-{:02d}T12:00:00".format(i+1), camera_id="day",
                     split="train" if i < 8 else "val" if i < 12 else "test",
                     label=label, label_source="reviewed:test_fixture", mode="day", weather="clear",
                     quality_valid=1, area_00=label*.5, area_01=label*.2,
                     upper_00=label*.2, upper_01=label*.1, lower_00=label*.3, lower_01=label*.1)
            rows.append(r)
        features = self.root / "features.csv"
        write_csv(features, rows)
        write_json(features.with_suffix(".meta.json"), {"source": "ground_truth", "bins": 2})
        for kind in ["logistic", "forest", "dual_cnn", "mlp", "area_threshold"]:
            with self.subTest(kind=kind):
                config = dict(features=str(features), model=kind,
                              feature_set="area" if kind == "area_threshold" else "both",
                              epochs=2, batch_size=4, hidden=4, seed=42, threads=2, trees=5, bootstrap=10)
                run = cls.fit(config, self.root / kind, "cpu")
                payload = joblib.load(run / "model.joblib")
                result = cls.evaluate(run / "model.joblib", features, self.root / (kind+"_eval"))
                self.assertEqual(result["n"], 6)
                self.assertEqual(result["threshold"], payload["threshold"])
                self.assertEqual(len(read_csv(self.root/(kind+"_eval")/"predictions.csv")), 6)
        # Extractor configuration must be identical at training and test time.
        write_json(features.with_suffix(".meta.json"), {"source": "ground_truth", "bins": 20})
        with self.assertRaisesRegex(ValueError, "extraction differs"):
            cls.evaluate(self.root/"logistic"/"model.joblib", features, self.root/"mismatch")

    def test_all_available_segmentation_models_forward_offline(self):
        for name in ["unet", "deeplab_mobilenet", "deeplab_xception", "segnet",
                     "fcn_resnet50", "deeplabv3_resnet50", "lraspp_mobilenet"]:
            with self.subTest(name=name):
                model = segmentation_model({"name": name, "width": 2}).eval()
                with torch.inference_mode():
                    logits = model(torch.zeros(1, 3, 64, 64))
                self.assertEqual(tuple(logits.shape), (1, 2, 64, 64))
                self.assertTrue(torch.isfinite(logits).all())

    def test_tiling_covers_edges_for_large_and_small_images(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                result = torch.zeros(len(x), 2, *x.shape[2:])
                result[:, 1] = 2
                return result
        for h, w in [(43, 71), (11, 13)]:
            scores = seg.predict_image(Constant(), np.zeros((h, w, 3), np.uint8),
                                       {"input_size": 32, "tile_size": 32, "tile_stride": 23},
                                       torch.device("cpu"))
            np.testing.assert_allclose(scores, torch.sigmoid(torch.tensor(2.)).item(), atol=1e-6)

    def test_joint_split_keeps_pair_endpoint_frames_together(self):
        frames = [dict(sample_id="f"+str(i), parent_id="f"+str(i), camera_id="day",
                       timestamp="2022-06-{:02d}T12:00:00".format(i+1)) for i in range(12)]
        pairs = [dict(sample_id="pair"+str(i), frame_a="f"+str(i), frame_b="f"+str(i+1),
                      camera_id="day", timestamp_a=frames[i]["timestamp"],
                      timestamp=frames[i+1]["timestamp"], label=str(int(i%4 == 0)))
                 for i in range(0, 12, 2)]
        write_csv(self.root/"frames.csv", frames)
        write_csv(self.root/"pairs.csv", pairs)
        main(["joint-split", "--frames", str(self.root/"frames.csv"), "--pairs",
              str(self.root/"pairs.csv"), "--output", str(self.root/"joint")])
        f, p = read_csv(self.root/"joint"/"frames.csv"), read_csv(self.root/"joint"/"pairs.csv")
        validate_splits(f+p)
        lookup = {r["parent_id"]: r for r in f}
        for r in p:
            self.assertEqual(r["split"], lookup[r["frame_a"]]["split"])
            self.assertEqual(r["split"], lookup[r["frame_b"]]["split"])

    def test_reconstruction_preserves_patch_coordinates(self):
        rows = []
        for x in range(2):
            for y in range(2):
                name = "camera_20220601120000_TIMING_{}{}".format(x, y)
                a, b = self.root/(name+".jpg"), self.root/(name+".png")
                Image.fromarray(np.full((5, 7, 3), 40*(x+2*y), np.uint8)).save(a)
                Image.fromarray(np.full((5, 7), (x+y)%2, np.uint8)).save(b)
                rows.append(dict(sample_id=name, parent_id="camera_20220601120000_TIMING",
                                 image_path=str(a), mask_path=str(b), camera_id="day",
                                 split="train", group_id="g", timestamp="2022-06-01T12:00:00"))
        reconstruct(rows, self.root/"reconstruct", columns=2, nrows=2, layout="xy")
        outputs = read_csv(self.root/"reconstruct"/"manifest.csv")
        with Image.open(outputs[0]["mask_path"]) as im:
            array = np.array(im)
        self.assertEqual(array.shape, (10, 14))
        self.assertEqual(array[:5, 7:].sum(), 35)
        self.assertEqual(array[5:, 7:].sum(), 0)

    def test_full_pair_pipeline_matches_offline_classifier(self):
        from riverlab.pipeline import infer_pairs
        from riverlab.data import provenance
        from riverlab.features import pair_geometry
        frames = self.segmentation_rows()
        frame_csv = self.root/"frames.csv"
        write_csv(frame_csv, frames)
        config = {"model": {"name": "unet", "width": 2}, "input_size": 32}
        model = segmentation_model(config["model"]).eval()
        seg_cp = self.root/"seg.pt"
        torch.save(dict(config=config, provenance=provenance(frames), state_dict=model.state_dict()), seg_cp)
        pairs, features = [], []
        for a, b in [(0, 1), (4, 5), (6, 7)]:
            images = [Image.open(frames[i]["image_path"]).convert("RGB") for i in [a, b]]
            masks = [(seg.predict_image(model, im, config, torch.device("cpu")) >= .5).astype(np.uint8)
                     for im in images]
            for label in [0, 1]:
                pair = dict(sample_id="pair{}_{}".format(a, label), frame_a=str(a), frame_b=str(b),
                            group_id=frames[a]["group_id"], camera_id="day", split=frames[a]["split"],
                            timestamp_a=frames[a]["timestamp"], timestamp=frames[b]["timestamp"],
                            label=label, label_source="reviewed:synthetic", mode="day")
                pairs.append(pair)
                features.append(dict(pair, **pair_geometry(masks, images, bins=2)))
        pair_csv, feature_csv = self.root/"pairs.csv", self.root/"features.csv"
        write_csv(pair_csv, pairs)
        write_csv(feature_csv, features)
        write_json(feature_csv.with_suffix(".meta.json"), dict(
            source="predicted", feature_kind="pair-features", checkpoint_sha256=digest(seg_cp),
            provenance=provenance(frames), bins=2, sigma=0, quality_config={}))
        train = cls.fit(dict(features=str(feature_csv), model="logistic", bootstrap=10),
                        self.root/"classifier", "cpu")
        result = infer_pairs(seg_cp, train/"model.joblib", pair_csv, frame_csv, self.root/"pipeline", warmup=1)
        self.assertEqual(result["n_pairs"], 2)
        self.assertGreater(result["median_seconds"], 0)
        output = read_csv(self.root/"pipeline"/"predictions.csv")
        offline = cls.predict(joblib.load(train/"model.joblib"), [r for r in features if r["split"] == "test"])
        np.testing.assert_allclose([float(r["score"]) for r in output], offline)

    def test_exploratory_labels_need_explicit_opt_in(self):
        rows = [dict(sample_id=str(i), group_id=str(i), timestamp="2022-08-{:02d}".format(i+1),
                     split=["train", "val", "test"][i//2], label=i%2, area_00=i)
                for i in range(6)]
        path = self.root/"unverified.csv"
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, "label_source"):
            cls.fit({"features": str(path)}, self.root/"never_started", "cpu")
