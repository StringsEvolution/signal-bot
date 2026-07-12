"""
Credential encryption at rest.

Pocket Option credentials (PO_SESSION / PO_UID) are sensitive: a leaked value
is a full account takeover. This module transparently encrypts credential
blobs before they hit the database and decrypts them on read.

Design goals:
  * Zero new required infra. Uses Fernet (AES-128-CBC + HMAC) from the
    `cryptography` package, which is already a transitive dependency of the
    stack; if it somehow isn't present, we fail SAFE by refusing to store
    plaintext rather than silently downgrading security.
  * Key comes from the CREDENTIAL_ENC_KEY env var (a urlsafe base64 Fernet
    key). Generate one with:
        python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    Put it in your production env / Railway variables. Never commit it.
  * Backward compatible: decrypt() detects and passes through any legacy
    PLAINTEXT rows written before encryption was enabled, so existing users
    are not broken. Re-saving a user (e.g. next /connectpo) upgrades them to
    ciphertext automatically.
  * If no key is configured (local dev), encryption is a transparent no-op so
    the bot still runs — but a loud warning is logged once, and is_enabled()
    lets callers surface that in health checks.
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_ENC_PREFIX = "enc:v1:"   # marker so we can tell ciphertext from legacy plaintext

_fernet = None
_warned_no_key = False


def _get_fernet():
    """Lazily build the Fernet instance from CREDENTIAL_ENC_KEY, or return
    None if no key is configured (dev mode / no-op)."""
    global _fernet, _warned_no_key
    if _fernet is not None:
        return _fernet

    key = os.getenv("CREDENTIAL_ENC_KEY", "").strip()
    if not key:
        if not _warned_no_key:
            logger.warning(
                "CREDENTIAL_ENC_KEY not set — Pocket Option credentials will be "
                "stored WITHOUT encryption. Set CREDENTIAL_ENC_KEY in production. "
                "Generate one: python -c \"from cryptography.fernet import "
                "Fernet; print(Fernet.generate_key().decode())\""
            )
            _warned_no_key = True
        return None

    try:
        from cryptography.fernet import Fernet
        _fernet = Fernet(key.encode() if isinstance(key, str) else key)
        return _fernet
    except Exception as exc:
        # Fail safe: if a key IS configured but unusable, do NOT fall back to
        # plaintext — that would be a silent security downgrade. Raise so the
        # misconfiguration is caught at startup / first save.
        raise RuntimeError(
            f"CREDENTIAL_ENC_KEY is set but invalid ({exc}). Fix the key or "
            f"unset it for local dev."
        )


def is_enabled() -> bool:
    """True if encryption is active (a valid key is configured)."""
    try:
        return _get_fernet() is not None
    except Exception:
        return False


def encrypt_dict(data: dict) -> str:
    """
    Serialize a credentials dict to a string for DB storage. If encryption is
    enabled the output is `enc:v1:<token>`; otherwise it's plain JSON (dev).
    """
    plaintext = json.dumps(data or {}, separators=(",", ":"))
    f = _get_fernet()
    if f is None:
        return plaintext  # dev no-op
    token = f.encrypt(plaintext.encode()).decode()
    return _ENC_PREFIX + token


def decrypt_to_dict(stored) -> dict:
    """
    Inverse of encrypt_dict. Accepts:
      * an `enc:v1:` ciphertext string  -> decrypts
      * a legacy plaintext JSON string  -> parses as-is (back-compat)
      * an already-parsed dict          -> returned unchanged (JSONB rows)
    Never raises on bad input — returns {} so one corrupt row can't crash the
    stream; the caller then simply skips that user.
    """
    if stored is None:
        return {}
    if isinstance(stored, dict):
        return stored
    if not isinstance(stored, str):
        try:
            return dict(stored)
        except Exception:
            return {}

    s = stored.strip()
    if s.startswith(_ENC_PREFIX):
        f = _get_fernet()
        if f is None:
            logger.error(
                "Found encrypted credentials but CREDENTIAL_ENC_KEY is not set "
                "— cannot decrypt. Skipping."
            )
            return {}
        try:
            token = s[len(_ENC_PREFIX):].encode()
            return json.loads(f.decrypt(token).decode())
        except Exception as exc:
            logger.error(f"Credential decryption failed (skipping user): {exc}")
            return {}

    # Legacy plaintext JSON (written before encryption was enabled).
    try:
        return json.loads(s)
    except Exception:
        return {}


def looks_encrypted(stored) -> bool:
    """True if the stored value is already ciphertext (used to detect legacy
    plaintext rows for optional migration)."""
    return isinstance(stored, str) and stored.strip().startswith(_ENC_PREFIX)
