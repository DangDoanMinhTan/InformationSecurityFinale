import base64
import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .config import MatchingConfig
from .encoding import IrisCode
from .matching import MatchResult, hamming_distance
from .protection import Transform, VaultRecord

MAGIC = b"IRISTPL1"
SALT_BYTES = 16
NONCE_BYTES = 12
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1


class TemplateStoreError(RuntimeError):
    pass


class TemplateStore:
    def __init__(self, path, passphrase: str):
        if not passphrase:
            raise TemplateStoreError("a passphrase is required")
        self.path = Path(path)
        self._data: Dict = {}
        if self.path.exists():
            self._salt, self._key = self._open_existing(passphrase)
        else:
            self._salt = secrets.token_bytes(SALT_BYTES)
            self._key = self._derive(passphrase, self._salt)
            self._data = {"version": 1, "shape": None, "users": {}, "lockouts": {}}
            self.save()

    @staticmethod
    def _derive(passphrase: str, salt: bytes) -> bytes:
        kdf = Scrypt(salt=salt, length=32, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
        return kdf.derive(passphrase.encode("utf-8"))

    def _open_existing(self, passphrase: str) -> Tuple[bytes, bytes]:
        blob = self.path.read_bytes()
        header = len(MAGIC) + SALT_BYTES + NONCE_BYTES
        if len(blob) <= header or not blob.startswith(MAGIC):
            raise TemplateStoreError("not a template store file")
        salt = blob[len(MAGIC): len(MAGIC) + SALT_BYTES]
        nonce = blob[len(MAGIC) + SALT_BYTES: header]
        key = self._derive(passphrase, salt)
        try:
            plaintext = AESGCM(key).decrypt(nonce, blob[header:], MAGIC + salt)
        except InvalidTag as error:
            raise TemplateStoreError("wrong passphrase or tampered store") from error
        self._data = json.loads(plaintext.decode("utf-8"))
        return salt, key

    def save(self) -> None:
        nonce = secrets.token_bytes(NONCE_BYTES)
        plaintext = json.dumps(self._data, separators=(",", ":")).encode("utf-8")
        ciphertext = AESGCM(self._key).encrypt(nonce, plaintext, MAGIC + self._salt)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(MAGIC + self._salt + nonce + ciphertext)
            os.replace(temporary, self.path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def users(self) -> List[str]:
        return sorted(self._data["users"])

    def has_user(self, user_id: str) -> bool:
        return user_id in self._data["users"]

    def next_version(self, user_id: str) -> int:
        return int(self._data.setdefault("versions", {}).get(user_id, 0)) + 1

    def new_transform(self, user_id: str) -> Tuple[Transform, Dict]:
        version = self.next_version(user_id)
        salt = secrets.token_bytes(SALT_BYTES)
        transform = Transform.derive(self._key, user_id, salt, version)
        return transform, {"version": version, "salt": base64.b64encode(salt).decode("ascii")}

    def transform_for(self, user_id: str) -> Transform:
        entry = self._data["users"][user_id].get("transform")
        if not entry:
            return Transform.identity()
        return Transform.derive(self._key, user_id, base64.b64decode(entry["salt"]), int(entry["version"]))

    def protection_version(self, user_id: str) -> int:
        entry = self._data["users"][user_id].get("transform")
        return int(entry["version"]) if entry else 0

    def add_user(
        self,
        user_id: str,
        codes: List[IrisCode],
        replace: bool = False,
        cancelable: bool = False,
        vault: Optional[Dict] = None,
    ) -> None:
        if not codes:
            raise TemplateStoreError("at least one template is required")
        if self.has_user(user_id) and not replace:
            raise TemplateStoreError(f"user already enrolled: {user_id}")
        shape = list(codes[0].shape)
        if self._data["shape"] is None:
            self._data["shape"] = shape
        elif self._data["shape"] != shape:
            raise TemplateStoreError("template shape differs from the store")
        entry: Dict = {"enrolled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        stored = codes
        if cancelable:
            transform, meta = self.new_transform(user_id)
            stored = [transform.apply(code) for code in codes]
            entry["transform"] = meta
            self._data.setdefault("versions", {})[user_id] = meta["version"]
        entry["templates"] = [code.serialize() for code in stored]
        if vault is not None:
            entry["vault"] = vault
        self._data["users"][user_id] = entry
        self._data["lockouts"].pop(user_id, None)
        self.save()

    def revoke(self, user_id: str) -> None:
        if not self.has_user(user_id):
            raise TemplateStoreError(f"unknown user: {user_id}")
        versions = self._data.setdefault("versions", {})
        versions[user_id] = max(self.protection_version(user_id), int(versions.get(user_id, 0))) + 1
        del self._data["users"][user_id]
        self._data["lockouts"].pop(user_id, None)
        self.save()

    def vault(self, user_id: str) -> Optional[Dict]:
        if not self.has_user(user_id):
            raise TemplateStoreError(f"unknown user: {user_id}")
        return self._data["users"][user_id].get("vault")

    def vault_records(self, user_id: str) -> List[VaultRecord]:
        vault = self.vault(user_id)
        if not vault:
            return []
        return [VaultRecord.from_dict(item) for item in vault["records"]]

    def distance(self, user_id: str, probe: IrisCode, matching: MatchingConfig) -> MatchResult:
        transform = self.transform_for(user_id)
        best = MatchResult(distance=1.0, shift=0, overlap=0)
        for template in self.templates(user_id):
            if transform.is_identity:
                match = hamming_distance(probe, template, matching)
            else:
                match = transform.distance(probe, template, matching)
            if match.distance < best.distance:
                best = match
        return best

    def nearest(self, probe: IrisCode, matching: MatchingConfig, exclude: Optional[str] = None) -> Tuple[Optional[str], MatchResult]:
        best_user: Optional[str] = None
        best = MatchResult(distance=1.0, shift=0, overlap=0)
        for user in self.users():
            if user == exclude:
                continue
            match = self.distance(user, probe, matching)
            if match.distance < best.distance:
                best = match
                best_user = user
        return best_user, best

    def remove_user(self, user_id: str) -> None:
        if not self.has_user(user_id):
            raise TemplateStoreError(f"unknown user: {user_id}")
        del self._data["users"][user_id]
        self._data["lockouts"].pop(user_id, None)
        self.save()

    def templates(self, user_id: str) -> List[IrisCode]:
        if not self.has_user(user_id):
            raise TemplateStoreError(f"unknown user: {user_id}")
        shape = tuple(self._data["shape"])
        return [IrisCode.deserialize(item, shape) for item in self._data["users"][user_id]["templates"]]

    def all_templates(self) -> List[Tuple[str, IrisCode]]:
        return [(user, code) for user in self.users() for code in self.templates(user)]

    def template_count(self) -> int:
        return sum(len(entry["templates"]) for entry in self._data["users"].values())

    def locked_until(self, key: str) -> float:
        return float(self._data["lockouts"].get(key, {}).get("locked_until", 0.0))

    def failure_count(self, key: str) -> int:
        return int(self._data["lockouts"].get(key, {}).get("failures", 0))

    def record_failure(self, key: str, now: float, limit: int, lockout_seconds: int) -> bool:
        entry = self._data["lockouts"].setdefault(key, {"failures": 0, "locked_until": 0.0})
        if entry["locked_until"] and entry["locked_until"] <= now:
            entry["failures"] = 0
            entry["locked_until"] = 0.0
        entry["failures"] += 1
        locked = entry["failures"] >= limit
        if locked:
            entry["locked_until"] = now + lockout_seconds
        self.save()
        return locked

    def clear_all_failures(self) -> None:
        if self._data["lockouts"]:
            self._data["lockouts"].clear()
            self.save()

    def clear_failures(self, key: str) -> None:
        if key in self._data["lockouts"]:
            del self._data["lockouts"][key]
            self.save()
