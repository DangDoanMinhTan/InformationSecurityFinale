import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic_eyes as synthetic
from iris_auth.authenticator import IrisAuthenticator, Outcome
from iris_auth.config import SystemConfig
from iris_auth.matching import hamming_distance
from iris_auth.pipeline import IrisPipeline
from iris_auth.protection import (
    FuzzyVault,
    ReedSolomon,
    Transform,
    flatten_code,
    hadamard_decode,
    hadamard_encode,
    pack,
)
from iris_auth.templatestore import TemplateStore, TemplateStoreError

PASSPHRASE = "unit-test-passphrase"


class CodingTests(unittest.TestCase):
    def test_reed_solomon_corrects_up_to_capacity(self):
        rng = np.random.default_rng(1)
        coder = ReedSolomon(32, 20)
        capacity = (32 - 20) // 2
        for _ in range(200):
            message = [int(value) for value in rng.integers(0, 128, 20)]
            word = coder.encode(message)
            corrupted = list(word)
            for position in rng.choice(32, capacity, replace=False):
                corrupted[int(position)] ^= int(rng.integers(1, 128))
            self.assertEqual(coder.decode(corrupted), message)

    def test_reed_solomon_rejects_beyond_capacity(self):
        rng = np.random.default_rng(2)
        coder = ReedSolomon(32, 20)
        for _ in range(100):
            message = [int(value) for value in rng.integers(0, 128, 20)]
            corrupted = list(coder.encode(message))
            for position in rng.choice(32, 12, replace=False):
                corrupted[int(position)] ^= int(rng.integers(1, 128))
            decoded = coder.decode(corrupted)
            if decoded is not None:
                self.assertNotEqual(decoded, message)

    def test_hadamard_roundtrip_with_noise(self):
        rng = np.random.default_rng(3)
        for symbol in range(0, 128, 9):
            block = hadamard_encode(symbol)
            noisy = block.copy()
            flips = rng.choice(64, 12, replace=False)
            noisy[flips] ^= 1
            decoded, _ = hadamard_decode(noisy, np.ones(64, dtype=bool))
            self.assertEqual(decoded, symbol)


class ProtectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SystemConfig()
        cls.pipeline = IrisPipeline(cls.config)
        cls.alice = [cls.pipeline.extract(image).code for image in synthetic.render_subject("alice", 5)]
        cls.bob = [cls.pipeline.extract(image).code for image in synthetic.render_subject("bob", 5)]

    def test_vault_releases_key_to_genuine_probe(self):
        vault = FuzzyVault(20)
        record, key = vault.lock(self.alice[0])
        self.assertEqual(vault.unlock(record, self.alice[3], self.config.matching), key)

    def test_vault_refuses_other_eye(self):
        vault = FuzzyVault(20)
        record, _ = vault.lock(self.alice[0])
        for probe in self.bob:
            self.assertIsNone(vault.unlock(record, probe, self.config.matching))

    def test_vault_record_roundtrip_serialization(self):
        vault = FuzzyVault(20)
        record, key = vault.lock(self.alice[0])
        restored = type(record).from_dict(record.to_dict())
        self.assertEqual(vault.unlock(restored, self.alice[2], self.config.matching), key)

    def test_vault_helper_does_not_reveal_code_or_key(self):
        vault = FuzzyVault(20)
        record, key = vault.lock(self.alice[0])
        self.assertNotIn(key.hex(), record.helper)
        self.assertNotEqual(record.helper, pack(flatten_code(self.alice[0])[0]))

    def test_transform_preserves_distance(self):
        matching = self.config.matching
        transform = Transform.derive(b"k" * 32, "alice", b"s" * 16, 1)
        raw = hamming_distance(self.alice[3], self.alice[0], matching)
        protected = transform.distance(self.alice[3], transform.apply(self.alice[0]), matching)
        self.assertAlmostEqual(raw.distance, protected.distance, places=9)
        raw = hamming_distance(self.bob[3], self.alice[0], matching)
        protected = transform.distance(self.bob[3], transform.apply(self.alice[0]), matching)
        self.assertAlmostEqual(raw.distance, protected.distance, places=9)

    def test_transforms_are_unlinkable(self):
        matching = self.config.matching
        first = Transform.derive(b"k" * 32, "alice", b"a" * 16, 1)
        second = Transform.derive(b"k" * 32, "alice", b"a" * 16, 2)
        distance = hamming_distance(first.apply(self.alice[0]), second.apply(self.alice[0]), matching).distance
        self.assertGreater(distance, 0.4)
        self.assertLess(distance, 0.6)

    def test_transform_is_deterministic(self):
        one = Transform.derive(b"k" * 32, "alice", b"a" * 16, 1)
        two = Transform.derive(b"k" * 32, "alice", b"a" * 16, 1)
        self.assertTrue(np.array_equal(one.apply(self.alice[0]).bits, two.apply(self.alice[0]).bits))


class ProtectedStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "store.enc"
        self.store = TemplateStore(self.path, PASSPHRASE)
        self.now = [1000.0]
        base = SystemConfig()
        self.config = replace(base, security=replace(base.security, expose_diagnostics=True))
        self.system = IrisAuthenticator(self.store, self.config, clock=lambda: self.now[0])
        self.alice = synthetic.render_subject("alice", 6)
        self.bob = synthetic.render_subject("bob", 6)
        self.system.enroll("alice", self.alice[:3])
        self.system.enroll("bob", self.bob[:3])

    def tearDown(self):
        self.directory.cleanup()

    def test_stored_templates_differ_from_raw_codes(self):
        stored = self.store.templates("alice")[0]
        raw = self.system.pipeline.extract(self.alice[0]).code
        self.assertFalse(np.array_equal(stored.bits, raw.bits))
        self.assertGreater(self.store.protection_version("alice"), 0)

    def test_verification_works_on_protected_templates(self):
        self.assertTrue(self.system.verify("alice", self.alice[4]).accepted)
        self.assertFalse(self.system.verify("alice", self.bob[4]).accepted)
        self.assertEqual(self.system.identify(self.bob[5]).user_id, "bob")

    def test_protection_survives_reopening(self):
        reopened = TemplateStore(self.path, PASSPHRASE)
        again = IrisAuthenticator(reopened, self.config, clock=lambda: self.now[0])
        self.assertTrue(again.verify("alice", self.alice[4]).accepted)
        self.assertEqual(again.unlock_vault("alice", self.alice[5]).outcome, Outcome.ACCEPTED)

    def test_vault_unlock_genuine_and_impostor(self):
        released = self.system.unlock_vault("alice", self.alice[4])
        self.assertEqual(released.outcome, Outcome.ACCEPTED)
        self.assertEqual(released.secret, "protected record of alice")
        denied = self.system.unlock_vault("alice", self.bob[4])
        self.assertEqual(denied.outcome, Outcome.NO_MATCH)
        self.assertIsNone(denied.secret)

    def test_vault_unknown_user(self):
        result = self.system.unlock_vault("nobody", self.alice[4])
        self.assertEqual(result.outcome, Outcome.UNKNOWN_USER)

    def test_vault_failures_lock_the_account(self):
        for _ in range(self.system._attempt_limit()):
            self.system.unlock_vault("alice", self.bob[4])
        self.assertEqual(self.system.unlock_vault("alice", self.alice[4]).outcome, Outcome.LOCKED_OUT)

    def test_revocation_invalidates_stolen_template(self):
        matching = self.config.matching
        stolen = self.store.templates("alice")
        stolen_version = self.store.protection_version("alice")
        self.store.revoke("alice")
        self.assertFalse(self.store.has_user("alice"))
        self.system.enroll("alice", self.alice[:3])
        self.assertGreater(self.store.protection_version("alice"), stolen_version)
        current = self.store.templates("alice")
        cross = hamming_distance(stolen[0], current[0], matching).distance
        self.assertGreater(cross, 0.4)
        self.assertTrue(self.system.verify("alice", self.alice[4]).accepted)

    def test_revoke_unknown_user(self):
        with self.assertRaises(TemplateStoreError):
            self.store.revoke("nobody")

    def test_reenroll_replace_rotates_protection(self):
        before = self.store.protection_version("alice")
        self.system.enroll("alice", self.alice[:3], replace=True)
        self.assertGreater(self.store.protection_version("alice"), before)
        self.assertTrue(self.system.verify("alice", self.alice[4]).accepted)

    def test_duplicate_enrollment_detected_under_protection(self):
        from iris_auth.authenticator import EnrollmentError

        with self.assertRaises(EnrollmentError):
            self.system.enroll("alice-clone", self.alice[3:5])

    def test_cancelable_can_be_disabled(self):
        plain_config = replace(self.config, security=replace(self.config.security, cancelable_templates=False))
        path = Path(self.directory.name) / "plain.enc"
        plain_store = TemplateStore(path, PASSPHRASE)
        plain = IrisAuthenticator(plain_store, plain_config, clock=lambda: self.now[0])
        plain.enroll("carol", synthetic.render_subject("carol", 3))
        self.assertEqual(plain_store.protection_version("carol"), 0)
        self.assertTrue(plain.verify("carol", synthetic.render_subject("carol", 5)[4]).accepted)


if __name__ == "__main__":
    unittest.main()
