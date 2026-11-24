import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulated_camera import FRAME_HEIGHT, FRAME_WIDTH, visible_frame
import synthetic_eyes
from iris_auth.app_logic import SCENARIOS, SECURITY_SCENARIOS, AppController, display_name
from iris_auth.authenticator import Outcome
from iris_auth.webcam import ASPECT, OUTPUT_HEIGHT, OUTPUT_WIDTH, crop_guide, guide_rect, preview_frame, process_frame


class GeometryTests(unittest.TestCase):
    def test_guide_rect_is_centered_and_inside(self):
        for fraction in (0.15, 0.35, 0.7):
            x, y, w, h = guide_rect((FRAME_HEIGHT, FRAME_WIDTH), fraction)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, FRAME_WIDTH)
            self.assertLessEqual(y + h, FRAME_HEIGHT)
            self.assertAlmostEqual(w / h, ASPECT, delta=0.02)
            self.assertAlmostEqual(x + w / 2, FRAME_WIDTH / 2, delta=1)

    def test_processed_frame_shape(self):
        frame = visible_frame("tester", 1)
        gray = process_frame(frame, 0.35)
        self.assertEqual(gray.shape, (OUTPUT_HEIGHT, OUTPUT_WIDTH))
        self.assertEqual(gray.dtype, np.uint8)
        x, y, w, h = guide_rect(frame.shape, 0.35)
        self.assertEqual(crop_guide(frame, 0.35).shape[:2], (h, w))

    def test_preview_has_fixed_width(self):
        self.assertEqual(preview_frame(visible_frame("tester", 1), 0.35).shape[1], 640)


class ControllerWebcamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.controller = AppController(Path(cls.directory.name))
        cls.controller.prepare()
        cls.threshold, cls.pad = cls.controller.default_settings()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    @staticmethod
    def frames(name, sessions):
        return [process_frame(visible_frame(name, session), 0.35) for session in sessions]

    def test_defaults(self):
        self.assertEqual(self.controller.users(), [])
        self.assertFalse(self.pad)
        self.assertAlmostEqual(self.threshold, 0.36)

    def test_enroll_verify_identify_with_webcam_frames(self):
        for name in ("tester", "friend"):
            for gray in self.frames(name, (1, 2, 3)):
                ok, message, _ = self.controller.analyze_capture(gray)
                self.assertTrue(ok, message)
            message = self.controller.enroll_images(name, self.frames(name, (1, 2, 3)), self.pad, self.threshold)
            self.assertTrue(message.startswith("Enrolled"), message)
        genuine = self.controller.verify_image("tester", self.frames("tester", (4,))[0], None, self.pad, False, self.threshold, "t")
        self.assertEqual(genuine.result.outcome, Outcome.ACCEPTED)
        impostor = self.controller.verify_image("tester", self.frames("friend", (5,))[0], None, self.pad, False, self.threshold, "t")
        self.assertEqual(impostor.result.outcome, Outcome.NO_MATCH)
        found = self.controller.identify_image(self.frames("friend", (6,))[0], None, self.pad, False, self.threshold, "t")
        self.assertEqual(found.result.user_id, "friend")

    def test_blank_capture_is_rejected_with_guidance(self):
        ok, message, _ = self.controller.analyze_capture(np.full((280, 320), 120, dtype=np.uint8))
        self.assertFalse(ok)
        self.assertIn("Move closer", message)

    def test_captures_saved(self):
        paths = self.controller.save_captures("my name!", self.frames("tester", (1,)))
        self.assertTrue(paths[0].exists())
        self.assertNotIn("!", paths[0].name)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.dataset = Path(cls.directory.name) / "dataset"
        names = [f"{index:03d}{side}" for index in range(1, 8) for side in "LR"]
        synthetic_eyes.generate_subjects(cls.dataset, names, 5, 2024)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_import_dataset_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = AppController(Path(directory))
            controller.prepare()
            threshold, _ = controller.default_settings()
            message, example, user = controller.import_dataset(self.dataset, threshold, max_users=6, per_user=3)
            self.assertEqual(len(controller.users()), 6)
            self.assertEqual(user, "Alice_L")
            self.assertIsNotNone(example)
            view = controller.verify(user, example, None, False, False, threshold)
            self.assertEqual(view.result.outcome, Outcome.ACCEPTED)
            self.assertIn("Imported 6 user(s)", message)

    def test_display_names(self):
        self.assertEqual(display_name("001L"), "Alice_L")
        self.assertEqual(display_name("025R"), "Alice2_R")
        self.assertEqual(display_name("plain"), "plain")

    def test_scenarios(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = AppController(Path(directory))
            controller.prepare()
            self.assertFalse(controller.scenarios_ready())
            controller.import_dataset(self.dataset, 0.36)
            self.assertTrue(controller.scenarios_ready())
            expected = {
                "genuine": Outcome.ACCEPTED,
                "impostor": Outcome.NO_MATCH,
                "stranger": Outcome.NO_MATCH,
                "other_eye": Outcome.NO_MATCH,
                "identify_known": Outcome.ACCEPTED,
                "identify_stranger": Outcome.NO_MATCH,
                "blur": None,
                "lockout": Outcome.LOCKED_OUT,
            }
            self.assertEqual({item.key for item in SCENARIOS}, set(expected))
            for key, outcome in expected.items():
                view = controller.run_scenario(key, 0.36)
                if outcome is None:
                    self.assertFalse(view.result.accepted, key)
                else:
                    self.assertEqual(view.result.outcome, outcome, key)
            plan = controller.load_plan()
            security = {
                "vault": Outcome.ACCEPTED,
                "vault_impostor": Outcome.NO_MATCH,
                "print_attack": Outcome.ATTACK_DETECTED,
                "print_attack_open": Outcome.ACCEPTED,
                "revocation": None,
            }
            self.assertEqual({item.key for item in SECURITY_SCENARIOS}, set(security))
            for key, outcome in security.items():
                self.assertIn(key, plan, key)
                view = controller.run_scenario(key, 0.36)
                if outcome is None:
                    self.assertIn("STOLEN TEMPLATE", view.banner)
                    self.assertTrue(any("HD" in line for line in view.details))
                else:
                    self.assertEqual(view.result.outcome, outcome, key)
            controller.reset_database()
            self.assertFalse(controller.scenarios_ready())
            with self.assertRaises(ValueError):
                controller.run_scenario("genuine", 0.36)

    def test_scenarios_repeatable_after_lockout(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = AppController(Path(directory))
            controller.prepare()
            controller.import_dataset(self.dataset, 0.36)
            controller.run_scenario("lockout", 0.36)
            self.assertEqual(controller.run_scenario("genuine", 0.36).result.outcome, Outcome.ACCEPTED)

    def test_evaluate_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = AppController(Path(directory))
            controller.prepare()
            figure, summary = controller.evaluate_dataset(self.dataset, 0.36)
            self.assertTrue(figure.exists())
            self.assertGreater(len(summary), 2)

    def test_encode_only(self):
        probe = next(iter(sorted(self.dataset.rglob("*.png"))))
        with tempfile.TemporaryDirectory() as directory:
            controller = AppController(Path(directory))
            controller.prepare()
            self.assertEqual(controller.encode_only(probe).banner, "IRISCODE EXTRACTED")

    def test_reset_database(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = AppController(Path(directory))
            controller.prepare()
            controller.import_dataset(self.dataset, 0.36, max_users=2, per_user=3)
            controller.reset_database()
            self.assertEqual(controller.users(), [])


if __name__ == "__main__":
    unittest.main()
