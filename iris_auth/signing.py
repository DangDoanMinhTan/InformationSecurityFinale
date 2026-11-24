import hashlib
import hmac
from typing import Optional


def capture_signature(signing_key: bytes, nonce: str, user: str, image: str, liveness: Optional[str]) -> str:
    message = "|".join(
        [
            nonce,
            user,
            hashlib.sha256(image.encode("utf-8")).hexdigest(),
            hashlib.sha256((liveness or "").encode("utf-8")).hexdigest(),
        ]
    )
    return hmac.new(signing_key, message.encode("utf-8"), hashlib.sha256).hexdigest()
