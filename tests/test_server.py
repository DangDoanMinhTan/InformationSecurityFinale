import base64
import ssl
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic_eyes as synthetic
from iris_auth.client import IrisClient, encode_image
from iris_auth.config import SystemConfig
from iris_auth.server import NonceBook, SessionSigner, create_app
from iris_auth.signing import capture_signature
from iris_auth.tls import client_context, ensure_certificate, server_context
from werkzeug.serving import make_server

PASSPHRASE = "unit-test-passphrase"
API_KEY = "device-key-0123456789"
ADMIN_KEY = "admin-key-0123456789"


class NonceTests(unittest.TestCase):
    def test_single_use_and_user_bound(self):
        now = [100.0]
        book = NonceBook(lambda: now[0], ttl=10)
        nonce = book.issue("alice")
        self.assertFalse(book.consume(nonce, "bob"))
        self.assertFalse(book.consume(nonce, "alice"))
        nonce = book.issue("alice")
        self.assertTrue(book.consume(nonce, "alice"))
        self.assertFalse(book.consume(nonce, "alice"))

    def test_expiry_and_capacity(self):
        now = [100.0]
        book = NonceBook(lambda: now[0], ttl=10, limit=2)
        first = book.issue("alice")
        book.issue("alice")
        self.assertIsNone(book.issue("alice"))
        now[0] += 11
        self.assertFalse(book.consume(first, "alice"))
        self.assertIsNotNone(book.issue("alice"))

    def test_session_tokens(self):
        now = [100.0]
        signer = SessionSigner(b"s" * 32, lambda: now[0], ttl=10)
        token = signer.sign("alice")
        self.assertEqual(signer.verify(token), "alice")
        forged = base64.urlsafe_b64encode(base64.urlsafe_b64decode(token).replace(b"alice", b"admin")).decode()
        self.assertIsNone(signer.verify(forged))
        self.assertIsNone(signer.verify("garbage"))
        now[0] += 11
        self.assertIsNone(signer.verify(token))


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.alice = synthetic.render_subject("alice", 6)
        cls.bob = synthetic.render_subject("bob", 6)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.app = create_app(Path(self.directory.name), PASSPHRASE, API_KEY, ADMIN_KEY, SystemConfig())
        self.http = self.app.test_client()
        self.keys = {"X-Api-Key": API_KEY}
        self.admin = {"X-Api-Key": ADMIN_KEY}
        status = self.http.post(
            "/api/enroll",
            json={"user": "alice", "images": [encode_image(item) for item in self.alice[:3]]},
            headers=self.admin,
        ).status_code
        self.assertEqual(status, 201)

    def tearDown(self):
        self.directory.cleanup()

    def nonce(self, user="alice"):
        return self.http.post("/api/challenge", json={"user": user}, headers=self.keys).get_json()["nonce"]

    def attempt(self, image, user="alice", nonce=None, path="/api/verify", headers=None):
        payload = {"user": user, "nonce": nonce or self.nonce(user), "image": encode_image(image)}
        return self.http.post(path, json=payload, headers=headers or self.keys)

    def test_health_is_public(self):
        response = self.http.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["users"], 1)
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_genuine_gets_session_token(self):
        response = self.attempt(self.alice[4])
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["accepted"])
        session = self.http.get("/api/session", headers={"Authorization": f"Bearer {body['token']}"})
        self.assertEqual(session.get_json()["user"], "alice")

    def test_impostor_is_denied_without_token(self):
        response = self.attempt(self.bob[4])
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("token", response.get_json())
        self.assertEqual(response.get_json()["message"], "Access denied")

    def test_replayed_request_is_refused(self):
        nonce = self.nonce()
        self.assertEqual(self.attempt(self.alice[4], nonce=nonce).status_code, 200)
        self.assertEqual(self.attempt(self.alice[4], nonce=nonce).status_code, 409)

    def test_nonce_cannot_be_used_for_another_user(self):
        nonce = self.nonce("alice")
        self.assertEqual(self.attempt(self.bob[4], user="bob", nonce=nonce).status_code, 409)

    def test_missing_or_wrong_key(self):
        self.assertEqual(self.http.post("/api/challenge", json={"user": "alice"}).status_code, 401)
        self.assertEqual(self.http.post("/api/challenge", json={"user": "alice"}, headers={"X-Api-Key": "nope"}).status_code, 401)
        self.assertEqual(self.http.post("/api/enroll", json={}, headers=self.keys).status_code, 401)
        self.assertEqual(self.http.post("/api/revoke", json={"user": "alice"}, headers=self.keys).status_code, 401)

    def test_bad_input_is_rejected(self):
        self.assertEqual(self.http.post("/api/challenge", data="x", headers=self.keys).status_code, 400)
        self.assertEqual(self.http.post("/api/challenge", json={"user": ""}, headers=self.keys).status_code, 400)
        bad = self.http.post("/api/verify", json={"user": "alice", "nonce": self.nonce(), "image": "!!"}, headers=self.keys)
        self.assertEqual(bad.status_code, 400)
        junk = base64.b64encode(b"not an image").decode()
        corrupt = self.http.post("/api/verify", json={"user": "alice", "nonce": self.nonce(), "image": junk}, headers=self.keys)
        self.assertEqual(corrupt.status_code, 400)

    def test_oversized_body(self):
        response = self.http.post("/api/verify", data=b"x" * (9 * 1024 * 1024), headers=self.keys, content_type="application/json")
        self.assertEqual(response.status_code, 413)

    def test_vault_releases_secret_only_to_owner(self):
        genuine = self.attempt(self.alice[4], path="/api/vault").get_json()
        self.assertEqual(genuine["secret"], "protected record of alice")
        denied = self.attempt(self.bob[4], path="/api/vault")
        self.assertEqual(denied.status_code, 403)
        self.assertNotIn("secret", denied.get_json())

    def test_print_attack_blocked_by_server_pad(self):
        attack = synthetic.simulate_print_attack(self.alice[4], 7)
        response = self.attempt(attack)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json()["outcome"], "attack_detected")

    def test_revocation_over_api(self):
        self.assertEqual(self.http.post("/api/revoke", json={"user": "alice"}, headers=self.admin).status_code, 200)
        self.assertEqual(self.http.post("/api/revoke", json={"user": "alice"}, headers=self.admin).status_code, 404)
        self.assertEqual(self.attempt(self.alice[4]).status_code, 403)

    def test_lockout_applies_over_the_network(self):
        limit = self.app.extensions["iris"]["system"]._attempt_limit()
        for _ in range(limit):
            self.attempt(self.bob[4])
        self.assertEqual(self.attempt(self.alice[4]).get_json()["outcome"], "locked_out")

    def test_unknown_route_and_method(self):
        self.assertEqual(self.http.get("/api/nothing").status_code, 404)
        self.assertEqual(self.http.get("/api/verify").status_code, 405)


class SignedCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.alice = synthetic.render_subject("alice", 5)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.key = b"signing-key-0123456789"
        self.app = create_app(Path(self.directory.name), PASSPHRASE, API_KEY, ADMIN_KEY, SystemConfig(), signing_key=self.key)
        self.http = self.app.test_client()
        self.http.post(
            "/api/enroll",
            json={"user": "alice", "images": [encode_image(item) for item in self.alice[:3]]},
            headers={"X-Api-Key": ADMIN_KEY},
        )

    def tearDown(self):
        self.directory.cleanup()

    def payload(self, key=None):
        nonce = self.http.post("/api/challenge", json={"user": "alice"}, headers={"X-Api-Key": API_KEY}).get_json()["nonce"]
        image = encode_image(self.alice[4])
        body = {"user": "alice", "nonce": nonce, "image": image}
        body["signature"] = capture_signature(key or self.key, nonce, "alice", image, None)
        return body

    def post(self, body):
        return self.http.post("/api/verify", json=body, headers={"X-Api-Key": API_KEY})

    def test_signed_capture_is_accepted(self):
        self.assertEqual(self.post(self.payload()).status_code, 200)

    def test_missing_or_wrong_signature_is_refused(self):
        body = self.payload()
        del body["signature"]
        self.assertEqual(self.post(body).status_code, 401)
        self.assertEqual(self.post(self.payload(b"another-signing-key")).status_code, 401)

    def test_image_swapped_in_transit_is_refused(self):
        body = self.payload()
        body["image"] = encode_image(self.alice[3])
        self.assertEqual(self.post(body).status_code, 401)

    def test_refused_tamper_does_not_burn_the_challenge(self):
        body = self.payload()
        tampered = dict(body, image=encode_image(self.alice[3]))
        self.assertEqual(self.post(tampered).status_code, 401)
        self.assertEqual(self.post(body).status_code, 200)

    def test_client_signs_automatically(self):
        client = IrisClient("http://127.0.0.1:1", API_KEY, signing_key=self.key)
        payload = client.capture_payload("alice", self.alice[4], None, "n" * 32)
        self.assertEqual(payload["signature"], capture_signature(self.key, "n" * 32, "alice", payload["image"], None))


class TlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.alice = synthetic.render_subject("alice", 5)
        cls.app = create_app(root, PASSPHRASE, API_KEY, ADMIN_KEY, SystemConfig())
        cls.cert, cls.key = ensure_certificate(root / "tls")
        cls.server = make_server("127.0.0.1", 0, cls.app, threaded=True, ssl_context=server_context(cls.cert, cls.key))
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.client = IrisClient(f"https://localhost:{cls.port}", API_KEY, cls.cert, ADMIN_KEY)
        cls.client.enroll("alice", cls.alice[:3])

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(5)
        cls.directory.cleanup()

    def test_verification_over_tls(self):
        status, body = self.client.verify("alice", self.alice[4])
        self.assertEqual(status, 200)
        self.assertTrue(body["accepted"])

    def test_certificate_generation_is_idempotent(self):
        again = ensure_certificate(Path(self.directory.name) / "tls")
        self.assertEqual(again, (self.cert, self.key))

    def test_plain_http_to_tls_port_fails(self):
        with self.assertRaises((urllib.error.URLError, ConnectionError, OSError)):
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/health", timeout=5).read()

    def test_untrusted_certificate_is_refused(self):
        other = Path(self.directory.name) / "other"
        cert, _ = ensure_certificate(other, ("localhost", "127.0.0.1"))
        client = IrisClient(f"https://localhost:{self.port}", API_KEY, cert)
        with self.assertRaises((urllib.error.URLError, ssl.SSLError)):
            client.call("GET", "/api/health")

    def test_tls_version_floor(self):
        context = client_context(self.cert)
        self.assertGreaterEqual(context.minimum_version, ssl.TLSVersion.TLSv1_2)


if __name__ == "__main__":
    unittest.main()
