import base64
import binascii
import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request

from .audit import AuditLog
from .authenticator import AuthResult, EnrollmentError, IrisAuthenticator
from .config import SystemConfig, real_eye_config
from .io_utils import decode_visible
from .signing import capture_signature
from .templatestore import TemplateStore, TemplateStoreError
from .tls import ensure_certificate, server_context

NONCE_BYTES = 16
NONCE_TTL_SECONDS = 60
SESSION_TTL_SECONDS = 900
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_PENDING_NONCES = 4096
MAX_ENROLL_IMAGES = 8


class NonceBook:
    def __init__(self, clock: Callable[[], float] = time.time, ttl: int = NONCE_TTL_SECONDS, limit: int = MAX_PENDING_NONCES):
        self.clock = clock
        self.ttl = ttl
        self.limit = limit
        self.pending: Dict[str, Tuple[str, float]] = {}
        self.lock = threading.Lock()

    def prune(self) -> None:
        now = self.clock()
        for nonce in [key for key, (_, expires) in self.pending.items() if expires <= now]:
            del self.pending[nonce]

    def issue(self, user: str) -> Optional[str]:
        with self.lock:
            self.prune()
            if len(self.pending) >= self.limit:
                return None
            nonce = secrets.token_hex(NONCE_BYTES)
            self.pending[nonce] = (user, self.clock() + self.ttl)
            return nonce

    def consume(self, nonce: str, user: str) -> bool:
        with self.lock:
            self.prune()
            entry = self.pending.pop(nonce, None)
        return entry is not None and hmac.compare_digest(entry[0], user)


class SessionSigner:
    def __init__(self, secret: Optional[bytes] = None, clock: Callable[[], float] = time.time, ttl: int = SESSION_TTL_SECONDS):
        self.secret = secret or secrets.token_bytes(32)
        self.clock = clock
        self.ttl = ttl

    def sign(self, user: str) -> str:
        expires = int(self.clock()) + self.ttl
        body = f"{user}|{expires}"
        mac = hmac.new(self.secret, body.encode("utf-8"), hashlib.sha256).hexdigest()
        return base64.urlsafe_b64encode(f"{body}|{mac}".encode("utf-8")).decode("ascii")

    def verify(self, token: str) -> Optional[str]:
        try:
            raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
            user, expires, mac = raw.rsplit("|", 2)
        except (ValueError, UnicodeError, binascii.Error):
            return None
        expected = hmac.new(self.secret, f"{user}|{expires}".encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, expected) or int(expires) < self.clock():
            return None
        return user


def decode_image(value: Any) -> np.ndarray:
    if not isinstance(value, str) or not value:
        raise ValueError("image must be a base64 string")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("image is not valid base64")
    return decode_visible(data)


def describe(result: AuthResult) -> Dict[str, Any]:
    return {"accepted": result.accepted, "outcome": result.outcome.value, "message": result.message}


def create_app(
    root: Path,
    passphrase: str,
    api_key: str,
    admin_key: str,
    config: Optional[SystemConfig] = None,
    clock: Callable[[], float] = time.time,
    system: Optional[IrisAuthenticator] = None,
    signing_key: Optional[bytes] = None,
) -> Flask:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    settings = config or real_eye_config()
    settings = replace(settings, pad=replace(settings.pad, enabled=True))
    store = TemplateStore(base / "templates.enc", passphrase)
    authenticator = system or IrisAuthenticator(store, settings, audit=AuditLog(base / "audit.log"), clock=clock)
    nonces = NonceBook(clock)
    signer = SessionSigner(clock=clock)
    guard = threading.Lock()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    app.extensions["iris"] = {"system": authenticator, "nonces": nonces, "signer": signer, "store": authenticator.store}

    def reply(payload: Dict[str, Any], status: int = 200):
        response = jsonify(payload)
        response.status_code = status
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    def authorised(expected: str) -> bool:
        supplied = request.headers.get("X-Api-Key", "")
        return bool(supplied) and hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))

    def body() -> Dict[str, Any]:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError("a JSON object is required")
        return data

    def user_of(data: Dict[str, Any]) -> str:
        user = data.get("user")
        if not isinstance(user, str) or not user.strip() or len(user) > 64:
            raise ValueError("user must be a non-empty string of at most 64 characters")
        return user

    @app.errorhandler(413)
    def too_large(_):
        return reply({"error": "request too large"}, 413)

    @app.errorhandler(404)
    def missing(_):
        return reply({"error": "not found"}, 404)

    @app.errorhandler(405)
    def wrong_method(_):
        return reply({"error": "method not allowed"}, 405)

    @app.get("/api/health")
    def health():
        return reply({"status": "ok", "users": len(authenticator.store.users())})

    @app.post("/api/challenge")
    def challenge():
        if not authorised(api_key):
            return reply({"error": "unauthorised"}, 401)
        try:
            user = user_of(body())
        except ValueError as error:
            return reply({"error": str(error)}, 400)
        nonce = nonces.issue(user)
        if nonce is None:
            return reply({"error": "too many pending challenges"}, 429)
        return reply({"nonce": nonce, "expires_in": NONCE_TTL_SECONDS})

    def authenticate(mode: str):
        if not authorised(api_key):
            return reply({"error": "unauthorised"}, 401)
        try:
            data = body()
            user = user_of(data)
            nonce = data.get("nonce")
            if signing_key is not None:
                expected = capture_signature(
                    signing_key,
                    nonce if isinstance(nonce, str) else "",
                    user,
                    data.get("image") if isinstance(data.get("image"), str) else "",
                    data.get("liveness") if isinstance(data.get("liveness"), str) else None,
                )
                supplied = data.get("signature")
                if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
                    return reply({"error": "capture signature invalid"}, 401)
            if not isinstance(nonce, str) or not nonces.consume(nonce, user):
                return reply({"error": "challenge missing, expired or already used"}, 409)
            image = decode_image(data.get("image"))
            liveness = decode_image(data["liveness"]) if data.get("liveness") else None
        except ValueError as error:
            return reply({"error": str(error)}, 400)
        with guard:
            if mode == "vault":
                result = authenticator.unlock_vault(user, image, liveness)
            else:
                result = authenticator.verify(user, image, liveness)
        payload = describe(result)
        if result.accepted:
            payload["token"] = signer.sign(user)
            if mode == "vault":
                payload["secret"] = result.secret
        return reply(payload, 200 if result.accepted else 403)

    @app.post("/api/verify")
    def verify():
        return authenticate("verify")

    @app.post("/api/vault")
    def vault():
        return authenticate("vault")

    @app.get("/api/session")
    def session():
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        user = signer.verify(token) if token else None
        if user is None:
            return reply({"error": "invalid or expired session"}, 401)
        return reply({"user": user})

    @app.post("/api/enroll")
    def enroll():
        if not authorised(admin_key):
            return reply({"error": "unauthorised"}, 401)
        try:
            data = body()
            user = user_of(data)
            images = data.get("images")
            if not isinstance(images, list) or not images or len(images) > MAX_ENROLL_IMAGES:
                raise ValueError(f"images must be a list of 1 to {MAX_ENROLL_IMAGES} base64 strings")
            decoded: List[np.ndarray] = [decode_image(item) for item in images]
        except ValueError as error:
            return reply({"error": str(error)}, 400)
        with guard:
            try:
                features = authenticator.enroll(user, decoded, replace=bool(data.get("replace", False)))
            except EnrollmentError as error:
                return reply({"error": str(error)}, 422)
        return reply({"enrolled": user, "templates": len(features)}, 201)

    @app.post("/api/revoke")
    def revoke():
        if not authorised(admin_key):
            return reply({"error": "unauthorised"}, 401)
        try:
            user = user_of(body())
        except ValueError as error:
            return reply({"error": str(error)}, 400)
        with guard:
            try:
                authenticator.store.revoke(user)
            except TemplateStoreError as error:
                return reply({"error": str(error)}, 404)
        return reply({"revoked": user})

    return app


def serve(
    root: Path,
    passphrase: str,
    api_key: str,
    admin_key: str,
    host: str = "127.0.0.1",
    port: int = 8443,
    tls: bool = True,
    cert_dir: Optional[Path] = None,
    signing_key: Optional[bytes] = None,
) -> None:
    app = create_app(root, passphrase, api_key, admin_key, signing_key=signing_key)
    context = None
    if tls:
        cert, key = ensure_certificate(cert_dir or Path(root) / "tls", ("localhost", "127.0.0.1", host))
        context = server_context(cert, key)
    app.run(host=host, port=port, ssl_context=context, threaded=True, debug=False, use_reloader=False)
