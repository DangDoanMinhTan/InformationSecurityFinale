import base64
import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .signing import capture_signature
from .tls import client_context


def encode_image(image: np.ndarray, quality: int = 92) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("cannot encode image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class IrisClient:
    def __init__(self, base_url: str, api_key: str, ca_file: Optional[Path] = None, admin_key: Optional[str] = None, timeout: float = 60.0, signing_key: Optional[bytes] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.admin_key = admin_key
        self.timeout = timeout
        self.signing_key = signing_key
        self.context: Optional[ssl.SSLContext] = None
        if self.base_url.startswith("https://"):
            if ca_file is None:
                raise ValueError("an https endpoint needs the server certificate (ca_file)")
            self.context = client_context(Path(ca_file))

    def call(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None, key: Optional[str] = None, bearer: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(self.base_url + path, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        if key:
            request.add_header("X-Api-Key", key)
        if bearer:
            request.add_header("Authorization", f"Bearer {bearer}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=self.context) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8") or "{}")

    def challenge(self, user: str) -> str:
        status, body = self.call("POST", "/api/challenge", {"user": user}, self.api_key)
        if status != 200:
            raise RuntimeError(body.get("error", f"challenge failed ({status})"))
        return body["nonce"]

    def capture_payload(self, user: str, image: np.ndarray, liveness: Optional[np.ndarray], nonce: Optional[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"user": user, "nonce": nonce or self.challenge(user), "image": encode_image(image)}
        if liveness is not None:
            payload["liveness"] = encode_image(liveness)
        if self.signing_key is not None:
            payload["signature"] = capture_signature(self.signing_key, payload["nonce"], user, payload["image"], payload.get("liveness"))
        return payload

    def verify(self, user: str, image: np.ndarray, liveness: Optional[np.ndarray] = None, nonce: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
        return self.call("POST", "/api/verify", self.capture_payload(user, image, liveness, nonce), self.api_key)

    def vault(self, user: str, image: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        return self.call("POST", "/api/vault", self.capture_payload(user, image, None, None), self.api_key)

    def enroll(self, user: str, images, replace: bool = False) -> Tuple[int, Dict[str, Any]]:
        payload = {"user": user, "images": [encode_image(item) for item in images], "replace": replace}
        return self.call("POST", "/api/enroll", payload, self.admin_key)

    def revoke(self, user: str) -> Tuple[int, Dict[str, Any]]:
        return self.call("POST", "/api/revoke", {"user": user}, self.admin_key)

    def session(self, token: str) -> Tuple[int, Dict[str, Any]]:
        return self.call("GET", "/api/session", bearer=token)
