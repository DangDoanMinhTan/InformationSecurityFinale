import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic_eyes as synthetic
from iris_auth.authenticator import EnrollmentError, IrisAuthenticator, Outcome
from iris_auth.config import SystemConfig
from iris_auth.matching import (
    false_accept_probability,
    hamming_distance,
    hamming_matrix,
    threshold_for_database,
)
from iris_auth.pad import PresentationAttackDetector
from iris_auth.pipeline import IrisPipeline
from iris_auth.segmentation import SegmentationError
from iris_auth.templatestore import TemplateStore, TemplateStoreError

PASSPHRASE = "unit-test-passphrase"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SystemConfig()
        cls.pipeline = IrisPipeline(cls.config)
        cls.alice = [cls.pipeline.extract(image) for image in synthetic.render_subject("alice", 3)]
        cls.bob = [cls.pipeline.extract(image) for image in synthetic.render_subject("bob", 2)]

    def test_code_dimensions(self):
        code = self.alice[0].code
        self.assertEqual(code.size, 2048)
        self.assertEqual(code.bits.shape, (8, 128, 2))
        self.assertGreater(code.usable_fraction(), 0.6)

    def test_segmentation_accuracy(self):
        seed = synthetic.subject_seed("probe-subject")
        rng = np.random.default_rng(5)
        capture = synthetic.random_capture(rng)
        features = self.pipeline.extract(synthetic.render_eye(seed, capture, 11))
        self.assertAlmostEqual(features.segmentation.iris.r, capture.iris_radius, delta=3.0)
        self.assertAlmostEqual(features.segmentation.pupil.r, capture.pupil_ratio * capture.iris_radius, delta=3.0)
        self.assertAlmostEqual(features.segmentation.iris.x, capture.center[0], delta=3.0)

    def test_genuine_below_threshold_and_impostor_above(self):
        threshold = self.config.matching.threshold
        for other in self.alice[1:]:
            self.assertLess(hamming_distance(self.alice[0].code, other.code, self.config.matching).distance, threshold)
        for other in self.bob:
            self.assertGreater(hamming_distance(self.alice[0].code, other.code, self.config.matching).distance, threshold)

    def test_rotation_tolerance(self):
        image = synthetic.render_subject("alice", 1)[0]
        height, width = image.shape
        import cv2

        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 4.0, 1.0)
        rotated = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
        features = self.pipeline.extract(rotated)
        distance = hamming_distance(self.alice[0].code, features.code, self.config.matching).distance
        self.assertLess(distance, self.config.matching.threshold)

    def test_matrix_matches_pairwise(self):
        codes = [f.code for f in self.alice + self.bob]
        matrix = hamming_matrix(codes, self.config.matching)
        for i in range(len(codes)):
            for j in range(len(codes)):
                expected = hamming_distance(codes[i], codes[j], self.config.matching).distance
                self.assertAlmostEqual(float(matrix[i, j]), expected, places=5)

    def test_blank_image_is_rejected(self):
        with self.assertRaises(SegmentationError):
            self.pipeline.extract(np.full((280, 320), 128, dtype=np.uint8))

    def test_blurred_image_fails_quality(self):
        import cv2

        image = synthetic.render_subject("alice", 1)[0]
        features = self.pipeline.extract(cv2.GaussianBlur(image, (0, 0), 7.0))
        self.assertFalse(features.quality.passed)


class DecisionModelTests(unittest.TestCase):
    def test_false_accept_matches_report_table(self):
        matching = SystemConfig().matching
        self.assertAlmostEqual(1.0 / false_accept_probability(0.32, matching), 3.5e7, delta=0.5e7)
        self.assertLess(false_accept_probability(0.30, matching), false_accept_probability(0.33, matching))

    def test_threshold_tightens_with_database_size(self):
        matching = SystemConfig().matching
        small = threshold_for_database(10, matching, 1e-6)
        large = threshold_for_database(10 ** 8, matching, 1e-6)
        self.assertGreater(small, 0.30)
        self.assertLessEqual(small, matching.threshold)
        self.assertLess(large, small)


class PadTests(unittest.TestCase):
    def test_print_attack_detected_and_live_passes(self):
        config = SystemConfig()
        pipeline = IrisPipeline(config)
        detector = PresentationAttackDetector(config.pad)
        live = synthetic.render_subject("carol", 1)[0]
        attack = synthetic.simulate_print_attack(live, 3)
        self.assertTrue(detector.assess(pipeline.extract(live).segmentation).passed)
        self.assertFalse(detector.assess(pipeline.extract(attack).segmentation).passed)

    def test_pupil_reaction(self):
        config = SystemConfig()
        pipeline = IrisPipeline(config)
        detector = PresentationAttackDetector(config.pad)
        dark, bright = synthetic.render_live_pair("carol", 1)
        live = detector.assess(pipeline.extract(dark).segmentation, pipeline.extract(bright).segmentation)
        static = detector.assess(pipeline.extract(dark).segmentation, pipeline.extract(dark).segmentation)
        self.assertGreater(live.pupil_constriction, config.pad.min_pupil_constriction)
        self.assertFalse(static.passed)


class StoreTests(unittest.TestCase):
    def test_roundtrip_and_wrong_passphrase(self):
        pipeline = IrisPipeline()
        code = pipeline.extract(synthetic.render_subject("dave", 1)[0]).code
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store.enc"
            store = TemplateStore(path, PASSPHRASE)
            store.add_user("dave", [code])
            reopened = TemplateStore(path, PASSPHRASE)
            restored = reopened.templates("dave")[0]
            self.assertTrue(np.array_equal(restored.bits, code.bits))
            self.assertTrue(np.array_equal(restored.mask, code.mask))
            self.assertNotIn(code.serialize()["bits"][:40].encode("ascii"), path.read_bytes())
            with self.assertRaises(TemplateStoreError):
                TemplateStore(path, "wrong")

    def test_tampering_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store.enc"
            TemplateStore(path, PASSPHRASE)
            blob = bytearray(path.read_bytes())
            blob[-1] ^= 0x01
            path.write_bytes(bytes(blob))
            with self.assertRaises(TemplateStoreError):
                TemplateStore(path, PASSPHRASE)


class AuthenticatorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = TemplateStore(Path(self.directory.name) / "store.enc", PASSPHRASE)
        self.now = [1000.0]
        base = SystemConfig()
        self.config = replace(base, security=replace(base.security, expose_diagnostics=True))
        self.system = IrisAuthenticator(self.store, self.config, clock=lambda: self.now[0])
        self.alice = synthetic.render_subject("alice", 5)
        self.bob = synthetic.render_subject("bob", 5)
        self.system.enroll("alice", self.alice[:3])
        self.system.enroll("bob", self.bob[:3])

    def tearDown(self):
        self.directory.cleanup()

    def test_genuine_accepted(self):
        result = self.system.verify("alice", self.alice[3])
        self.assertTrue(result.accepted)
        self.assertEqual(result.outcome, Outcome.ACCEPTED)

    def test_impostor_rejected_without_leaking_reason(self):
        result = self.system.verify("alice", self.bob[3])
        self.assertFalse(result.accepted)
        self.assertEqual(result.message, "Access denied")
        unknown = self.system.verify("nobody", self.bob[3])
        self.assertEqual(unknown.message, result.message)

    def test_score_hidden_by_default(self):
        hidden = IrisAuthenticator(self.store, SystemConfig(), clock=lambda: self.now[0])
        self.assertIsNone(hidden.verify("alice", self.alice[3]).diagnostics)

    def test_identification(self):
        self.assertEqual(self.system.identify(self.bob[4]).user_id, "bob")
        outsider = synthetic.render_subject("zed", 1)[0]
        self.assertFalse(self.system.identify(outsider).accepted)

    def test_lockout_and_recovery(self):
        limit = self.system._attempt_limit()
        for _ in range(limit):
            self.assertEqual(self.system.verify("alice", self.bob[3]).outcome, Outcome.NO_MATCH)
        self.assertEqual(self.system.verify("alice", self.alice[3]).outcome, Outcome.LOCKED_OUT)
        self.now[0] += self.config.security.lockout_seconds + 1
        self.assertTrue(self.system.verify("alice", self.alice[3]).accepted)

    def test_lockout_survives_reopening_store(self):
        limit = self.system._attempt_limit()
        for _ in range(limit):
            self.system.verify("alice", self.bob[3])
        reopened = TemplateStore(Path(self.directory.name) / "store.enc", PASSPHRASE)
        again = IrisAuthenticator(reopened, self.config, clock=lambda: self.now[0])
        self.assertEqual(again.verify("alice", self.alice[3]).outcome, Outcome.LOCKED_OUT)

    def test_print_attack_blocked_only_with_pad(self):
        attack = synthetic.simulate_print_attack(self.alice[3], 7)
        undefended = IrisAuthenticator(
            self.store, replace(self.config, pad=replace(self.config.pad, enabled=False)), clock=lambda: self.now[0]
        )
        self.assertTrue(undefended.verify("alice", attack).accepted)
        self.assertEqual(self.system.verify("alice", attack).outcome, Outcome.ATTACK_DETECTED)

    def test_liveness_frame_from_other_eye_rejected(self):
        dark, _ = synthetic.render_live_pair("alice", 1)
        _, other_bright = synthetic.render_live_pair("bob", 1)
        result = self.system.verify("alice", dark, other_bright)
        self.assertEqual(result.outcome, Outcome.ATTACK_DETECTED)

    def test_enrollment_guards(self):
        with self.assertRaises(EnrollmentError):
            self.system.enroll("alice", self.alice[:1])
        with self.assertRaises(EnrollmentError):
            self.system.enroll("mallory", self.alice[3:4])
        with self.assertRaises(EnrollmentError):
            self.system.enroll("mixed", [self.alice[3], self.bob[3]])


if __name__ == "__main__":
    unittest.main()
