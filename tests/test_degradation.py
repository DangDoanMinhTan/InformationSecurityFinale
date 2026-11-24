import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic_eyes as synthetic
from iris_auth.config import SystemConfig
from iris_auth.degradation import PROFILES, evaluate_webcam, select_channel, webcam_capture
from iris_auth.vault_eval import evaluate_vault, extract_codes
from iris_auth.webcam import OUTPUT_HEIGHT, OUTPUT_WIDTH


def colour(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


class DegradationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images = []
        for name in ("alice", "bob", "carol"):
            for image in synthetic.render_subject(name, 3):
                cls.images.append((name, colour(image)))

    def test_capture_has_webcam_output_shape_and_is_repeatable(self):
        image = self.images[0][1]
        first = webcam_capture(image, PROFILES["typical"], 4)
        second = webcam_capture(image, PROFILES["typical"], 4)
        self.assertEqual(first.shape, (OUTPUT_HEIGHT, OUTPUT_WIDTH))
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, webcam_capture(image, PROFILES["typical"], 5)))

    def test_channels(self):
        image = self.images[0][1]
        proxy = select_channel(image, "nir_proxy")
        self.assertTrue(np.array_equal(proxy[:, :, 0], image[:, :, 2]))
        self.assertIs(select_channel(image, "visible"), image)
        with self.assertRaises(ValueError):
            select_channel(image, "thermal")

    def test_profiles_get_harder(self):
        order = ["good", "typical", "poor"]
        sharp = []
        for name in order:
            capture = webcam_capture(self.images[0][1], PROFILES[name], 1)
            sharp.append(float(cv2.Laplacian(capture, cv2.CV_64F).var()))
        self.assertGreater(sharp[0], sharp[2])

    def test_evaluation_reports_consistent_counts(self):
        config = SystemConfig()
        result = evaluate_webcam(self.images, config, PROFILES["good"], "visible")
        self.assertLessEqual(result.acquired, result.images)
        self.assertEqual(result.images, 9)
        self.assertGreater(result.genuine, 0)
        self.assertGreater(result.impostor, result.genuine)
        self.assertLess(result.genuine_mean, result.impostor_mean)
        self.assertIn("good", result.summary())


class VaultEvaluationTests(unittest.TestCase):
    def test_vault_release_statistics(self):
        config = SystemConfig()
        records = []
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            for name in ("alice", "bob", "carol"):
                for index, image in enumerate(synthetic.render_subject(name, 4)):
                    path = Path(directory) / f"{name}_{index}.png"
                    cv2.imwrite(str(path), image)
                    records.append((name, path))
            codes = extract_codes(records, config, lambda path: cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
        result = evaluate_vault(codes, config, 20)
        self.assertEqual(result.eyes, 3)
        self.assertEqual(result.genuine_trials, 6)
        self.assertEqual(result.genuine_released, 6)
        self.assertEqual(result.impostor_released, 0)
        self.assertEqual(result.key_bits, 140)
        self.assertEqual(result.frr, 0.0)


if __name__ == "__main__":
    unittest.main()
