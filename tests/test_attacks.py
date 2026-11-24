import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic_eyes as synthetic
from iris_auth.attacks import (
    ATTACK_KINDS,
    AttackReport,
    simulate_attack,
    simulate_pupil_response,
    static_second_frame,
)
from iris_auth.config import SystemConfig
from iris_auth.pad import PresentationAttackDetector, detail_ratio_score
from iris_auth.pipeline import IrisPipeline


class AttackSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SystemConfig()
        cls.pipeline = IrisPipeline(cls.config)
        cls.live = synthetic.render_subject("carol", 1)[0]

    def test_attacks_are_deterministic_and_keep_shape(self):
        for kind in ATTACK_KINDS:
            first = simulate_attack(kind, self.live, 3)
            second = simulate_attack(kind, self.live, 3)
            self.assertEqual(first.shape, self.live.shape)
            self.assertEqual(first.dtype, np.uint8)
            self.assertTrue(np.array_equal(first, second))
            self.assertFalse(np.array_equal(first, simulate_attack(kind, self.live, 4)))

    def test_unknown_attack_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate_attack("mask", self.live, 1)

    def test_halftone_print_is_caught_by_spectral_pad(self):
        detector = PresentationAttackDetector(self.config.pad)
        caught = 0
        for seed in range(6):
            attack = simulate_attack("halftone_print", self.live, seed)
            try:
                segmentation = self.pipeline.extract(attack).segmentation
            except Exception:
                caught += 1
                continue
            if not detector.assess(segmentation).passed:
                caught += 1
        self.assertGreaterEqual(caught, 5)

    def test_detail_ratio_drops_for_recaptured_images(self):
        pad = replace(self.config.pad, min_detail_ratio=None)
        live = detail_ratio_score(self.pipeline.extract(self.live).segmentation, pad)
        blurred = detail_ratio_score(
            self.pipeline.extract(simulate_attack("photo_print", self.live, 2)).segmentation, pad
        )
        self.assertLess(blurred, live)

    def test_simulated_pupil_response_constricts_pupil(self):
        before = self.pipeline.extract(self.live).segmentation
        stimulated = simulate_pupil_response(self.live, before, 0.3)
        after = self.pipeline.extract(stimulated).segmentation
        ratioBefore = before.pupil.r / before.iris.r
        ratioAfter = after.pupil.r / after.iris.r
        self.assertLess(ratioAfter, ratioBefore * 0.85)

    def test_reflex_check_separates_live_from_static_frame(self):
        detector = PresentationAttackDetector(replace(self.config.pad, require_pupil_reaction=True))
        primary = self.pipeline.extract(self.live).segmentation
        live = self.pipeline.extract(simulate_pupil_response(self.live, primary, 0.3)).segmentation
        static = self.pipeline.extract(static_second_frame(self.live, 5)).segmentation
        self.assertTrue(detector.assess(primary, live).passed)
        self.assertFalse(detector.assess(primary, static).passed)


class ReportTests(unittest.TestCase):
    def build(self, rate):
        stats = {
            kind: {
                "trials": 10.0,
                "not_acquired": 1.0,
                "acquired": 9.0,
                "accepted_without_pad": rate,
                "accepted_with_pad": 0.0,
                "accepted_with_reflex": 0.0,
                "pad_miss": 0.0,
            }
            for kind in ATTACK_KINDS
        }
        return AttackReport(
            eyes=5,
            genuine_images=15,
            genuine_pad_rejected=1,
            genuine_accepted_without_pad=5,
            genuine_accepted_with_pad=4,
            genuine_accepted_with_reflex=4,
            genuine_trials=5,
            per_attack=stats,
            spectral_genuine=[1.0],
            spectral_attacks={kind: [3.0] for kind in ATTACK_KINDS},
            skipped=[],
        )

    def test_merge_weights_by_trials(self):
        merged = AttackReport.merge([self.build(0.5), self.build(1.0)])
        self.assertEqual(merged.eyes, 10)
        self.assertAlmostEqual(merged.per_attack["photo_print"]["accepted_without_pad"], 0.75)
        self.assertEqual(merged.per_attack["photo_print"]["trials"], 20.0)

    def test_roundtrip_through_dict(self):
        report = self.build(0.4)
        self.assertEqual(AttackReport.from_dict(report.to_dict()).to_dict(), report.to_dict())
        self.assertEqual(len(report.summary()), 5)


if __name__ == "__main__":
    unittest.main()
