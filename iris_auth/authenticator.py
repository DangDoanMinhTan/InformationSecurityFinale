import hashlib
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .audit import AuditLog
from .config import SystemConfig
from .matching import hamming_distance, threshold_for_database
from .pad import PadResult, PresentationAttackDetector
from .pipeline import IrisFeatures, IrisPipeline
from .protection import FuzzyVault
from .segmentation import SegmentationError
from .templatestore import TemplateStore

IDENTIFY_KEY = "*identify*"
UNKNOWN_KEY = "*unknown*"


class Outcome(str, Enum):
    ACCEPTED = "accepted"
    NO_MATCH = "no_match"
    UNKNOWN_USER = "unknown_user"
    QUALITY_REJECTED = "quality_rejected"
    SEGMENTATION_FAILED = "segmentation_failed"
    ATTACK_DETECTED = "attack_detected"
    LOCKED_OUT = "locked_out"


PUBLIC_MESSAGES = {
    Outcome.ACCEPTED: "Access granted",
    Outcome.LOCKED_OUT: "Too many failed attempts. Try again later",
    Outcome.QUALITY_REJECTED: "Image rejected. Capture the eye again",
    Outcome.SEGMENTATION_FAILED: "Image rejected. Capture the eye again",
}

COUNTED_FAILURES = {Outcome.NO_MATCH, Outcome.UNKNOWN_USER, Outcome.ATTACK_DETECTED}


class EnrollmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class AuthResult:
    accepted: bool
    outcome: Outcome
    user_id: Optional[str]
    diagnostics: Optional[Dict[str, Any]]
    secret: Optional[str] = None

    @property
    def message(self) -> str:
        return PUBLIC_MESSAGES.get(self.outcome, "Access denied")


class IrisAuthenticator:
    def __init__(
        self,
        store: TemplateStore,
        config: SystemConfig = SystemConfig(),
        pipeline: Optional[IrisPipeline] = None,
        audit: Optional[AuditLog] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self.config = config
        self.pipeline = pipeline or IrisPipeline(config)
        self.detector = PresentationAttackDetector(config.pad)
        self.audit = audit or AuditLog(None)
        self.clock = clock

    def _attempt_limit(self) -> int:
        security = self.config.security
        return security.max_failed_attempts_with_pad if self.config.pad.enabled else security.max_failed_attempts

    def _result(
        self,
        event: str,
        subject: str,
        outcome: Outcome,
        user_id: Optional[str],
        details: Dict[str, Any],
        secret: Optional[str] = None,
    ) -> AuthResult:
        self.audit.record(event, subject, outcome.value)
        diagnostics = details if self.config.security.expose_diagnostics else None
        return AuthResult(outcome == Outcome.ACCEPTED, outcome, user_id, diagnostics, secret)

    def _is_locked(self, key: str) -> bool:
        return self.store.locked_until(key) > self.clock()

    def _count_failure(self, key: str) -> None:
        self.store.record_failure(key, self.clock(), self._attempt_limit(), self.config.security.lockout_seconds)

    def _acquire(
        self, image: np.ndarray, liveness_image: Optional[np.ndarray], details: Dict[str, Any]
    ) -> Tuple[Optional[IrisFeatures], Optional[Outcome]]:
        try:
            features = self.pipeline.extract(image)
        except SegmentationError as error:
            details["error"] = str(error)
            return None, Outcome.SEGMENTATION_FAILED
        details["quality"] = features.quality
        if not features.quality.passed:
            return None, Outcome.QUALITY_REJECTED
        if self.config.pad.enabled:
            stimulated = None
            if liveness_image is not None:
                try:
                    stimulated_features = self.pipeline.extract(liveness_image)
                except SegmentationError as error:
                    details["error"] = str(error)
                    return None, Outcome.SEGMENTATION_FAILED
                stimulated = stimulated_features
            result: PadResult = self.detector.assess(
                features.segmentation, stimulated.segmentation if stimulated else None
            )
            reasons = list(result.reasons)
            if stimulated is not None:
                pairing = hamming_distance(features.code, stimulated.code, self.config.matching)
                details["pairing_distance"] = pairing.distance
                if pairing.distance > self.config.matching.threshold:
                    reasons.append("liveness frame belongs to a different eye")
            details["pad"] = result
            details["pad_reasons"] = reasons
            if reasons:
                return None, Outcome.ATTACK_DETECTED
        return features, None

    def enroll(self, user_id: str, images: Sequence[np.ndarray], replace: bool = False) -> List[IrisFeatures]:
        if not user_id.strip():
            raise EnrollmentError("user id must not be empty")
        if not images:
            raise EnrollmentError("at least one image is required")
        if self.store.has_user(user_id) and not replace:
            raise EnrollmentError(f"user already enrolled: {user_id}")
        accepted: List[IrisFeatures] = []
        for position, image in enumerate(images, start=1):
            details: Dict[str, Any] = {}
            features, failure = self._acquire(image, None, details)
            if features is None:
                reason = details.get("error") or "; ".join(details.get("pad_reasons", [])) or "; ".join(
                    details["quality"].failures
                )
                raise EnrollmentError(f"image {position} rejected: {reason}")
            accepted.append(features)
        threshold = self.config.matching.threshold
        for position in range(1, len(accepted)):
            distance = hamming_distance(accepted[0].code, accepted[position].code, self.config.matching).distance
            if distance > threshold:
                raise EnrollmentError(
                    f"image {position + 1} does not belong to the same eye as image 1 (HD {distance:.3f})"
                )
        if self.config.security.reject_duplicate_enrollment:
            for features in accepted:
                other_user, nearest = self.store.nearest(features.code, self.config.matching, exclude=user_id)
                if other_user is not None and nearest.distance <= threshold:
                    raise EnrollmentError("this eye is already enrolled under another identity")
        codes = [features.code for features in accepted]
        vault = self._lock_vault(user_id, codes)
        self.store.add_user(
            user_id,
            codes,
            replace=replace,
            cancelable=self.config.security.cancelable_templates,
            vault=vault,
        )
        self.audit.record("enroll", user_id, "enrolled")
        return accepted

    def verify(
        self, user_id: str, image: np.ndarray, liveness_image: Optional[np.ndarray] = None
    ) -> AuthResult:
        key = user_id if self.store.has_user(user_id) else UNKNOWN_KEY
        details: Dict[str, Any] = {"claimed": user_id}
        if self._is_locked(key):
            return self._result("verify", user_id, Outcome.LOCKED_OUT, None, details)
        features, failure = self._acquire(image, liveness_image, details)
        if features is None:
            if failure in COUNTED_FAILURES:
                self._count_failure(key)
            return self._result("verify", user_id, failure, None, details)
        if key == UNKNOWN_KEY:
            self._count_failure(key)
            return self._result("verify", user_id, Outcome.UNKNOWN_USER, None, details)
        best = self.store.distance(user_id, features.code, self.config.matching)
        details["distance"] = best.distance
        details["threshold"] = self.config.matching.threshold
        details["shift"] = best.shift
        details["overlap"] = best.overlap
        details["features"] = features
        if best.distance <= self.config.matching.threshold:
            self.store.clear_failures(key)
            return self._result("verify", user_id, Outcome.ACCEPTED, user_id, details)
        self._count_failure(key)
        return self._result("verify", user_id, Outcome.NO_MATCH, None, details)

    def identification_threshold(self) -> float:
        return threshold_for_database(
            self.store.template_count(), self.config.matching, self.config.security.target_system_far
        )

    def identify(self, image: np.ndarray, liveness_image: Optional[np.ndarray] = None) -> AuthResult:
        details: Dict[str, Any] = {}
        if self._is_locked(IDENTIFY_KEY):
            return self._result("identify", "-", Outcome.LOCKED_OUT, None, details)
        features, failure = self._acquire(image, liveness_image, details)
        if features is None:
            if failure in COUNTED_FAILURES:
                self._count_failure(IDENTIFY_KEY)
            return self._result("identify", "-", failure, None, details)
        threshold = self.identification_threshold()
        best_user, best_match = self.store.nearest(features.code, self.config.matching)
        best_distance = best_match.distance
        details["distance"] = best_distance
        details["threshold"] = threshold
        details["best_candidate"] = best_user
        details["features"] = features
        if best_user is not None and best_distance <= threshold:
            self.store.clear_failures(IDENTIFY_KEY)
            return self._result("identify", best_user, Outcome.ACCEPTED, best_user, details)
        self._count_failure(IDENTIFY_KEY)
        return self._result("identify", "-", Outcome.NO_MATCH, None, details)

    def _lock_vault(self, user_id: str, codes: Sequence[Any]) -> Dict[str, Any]:
        vault = FuzzyVault(self.config.security.vault_key_symbols)
        symbols = [secrets.randbelow(128) for _ in range(vault.key_symbols)]
        records = []
        key = b""
        for code in codes:
            record, key = vault.lock(code, symbols)
            records.append(record.to_dict())
        nonce = secrets.token_bytes(12)
        message = f"protected record of {user_id}".encode("utf-8")
        blob = AESGCM(hashlib.sha256(key).digest()).encrypt(nonce, message, user_id.encode("utf-8"))
        return {
            "key_bits": vault.key_bits,
            "records": records,
            "nonce": nonce.hex(),
            "blob": blob.hex(),
        }

    def unlock_vault(
        self, user_id: str, image: np.ndarray, liveness_image: Optional[np.ndarray] = None
    ) -> AuthResult:
        key = user_id if self.store.has_user(user_id) else UNKNOWN_KEY
        details: Dict[str, Any] = {"claimed": user_id}
        if self._is_locked(key):
            return self._result("vault", user_id, Outcome.LOCKED_OUT, None, details)
        features, failure = self._acquire(image, liveness_image, details)
        if features is None:
            if failure in COUNTED_FAILURES:
                self._count_failure(key)
            return self._result("vault", user_id, failure, None, details)
        if key == UNKNOWN_KEY or not self.store.vault(user_id):
            self._count_failure(key)
            return self._result("vault", user_id, Outcome.UNKNOWN_USER, None, details)
        vault = FuzzyVault(self.config.security.vault_key_symbols)
        released: Optional[bytes] = None
        for record in self.store.vault_records(user_id):
            released = vault.unlock(record, features.code, self.config.matching)
            if released is not None:
                break
        if released is None:
            self._count_failure(key)
            return self._result("vault", user_id, Outcome.NO_MATCH, None, details)
        entry = self.store.vault(user_id)
        try:
            message = AESGCM(hashlib.sha256(released).digest()).decrypt(
                bytes.fromhex(entry["nonce"]), bytes.fromhex(entry["blob"]), user_id.encode("utf-8")
            ).decode("utf-8")
        except InvalidTag:
            self._count_failure(key)
            return self._result("vault", user_id, Outcome.NO_MATCH, None, details)
        self.store.clear_failures(key)
        details["key_bits"] = entry["key_bits"]
        return self._result("vault", user_id, Outcome.ACCEPTED, user_id, details, secret=message)
