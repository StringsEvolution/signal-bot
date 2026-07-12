"""
User Manager — Stores subscriber Deriv tokens and trade settings in PostgreSQL.
"""

import os
import json
import logging
from typing import Optional
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

import crypto_util

load_dotenv()
logger = logging.getLogger(__name__)


def get_engine():
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/signal_bot")
    return create_engine(db_url, pool_pre_ping=True)


def init_user_tables():
    """Create subscriber tables if not exist."""
    engine = get_engine()
    ddl = """
    CREATE TABLE IF NOT EXISTS subscribers (
        id              SERIAL PRIMARY KEY,
        chat_id         VARCHAR(50) UNIQUE NOT NULL,
        username        VARCHAR(100),
        deriv_token     TEXT,
        trade_amount    NUMERIC(10,2) DEFAULT 10.00,
        auto_trade      BOOLEAN DEFAULT FALSE,
        connected_at    TIMESTAMPTZ,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        updated_at      TIMESTAMPTZ DEFAULT NOW()
    );
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    logger.info("Subscriber tables initialised.")


def save_token(chat_id: str, token: str, username: str = "") -> bool:
    """Save or update a subscriber's Deriv token."""
    engine = get_engine()
    sql = text("""
        INSERT INTO subscribers (chat_id, username, deriv_token, connected_at, updated_at)
        VALUES (:chat_id, :username, :token, NOW(), NOW())
        ON CONFLICT (chat_id) DO UPDATE
            SET deriv_token=EXCLUDED.deriv_token,
                username=EXCLUDED.username,
                connected_at=NOW(),
                updated_at=NOW()
    """)
    try:
        with engine.connect() as conn:
            conn.execute(sql, {"chat_id": chat_id, "username": username, "token": token})
            conn.commit()
        return True
    except Exception as exc:
        logger.error(f"save_token error: {exc}")
        return False


def remove_token(chat_id: str) -> bool:
    """Remove a subscriber's Deriv token."""
    engine = get_engine()
    sql = text("""
        UPDATE subscribers
        SET deriv_token=NULL, connected_at=NULL, updated_at=NOW()
        WHERE chat_id=:chat_id
    """)
    try:
        with engine.connect() as conn:
            conn.execute(sql, {"chat_id": chat_id})
            conn.commit()
        return True
    except Exception as exc:
        logger.error(f"remove_token error: {exc}")
        return False


def set_amount(chat_id: str, amount: float) -> bool:
    """Set a subscriber's default trade amount."""
    engine = get_engine()
    sql = text("""
        INSERT INTO subscribers (chat_id, trade_amount, updated_at)
        VALUES (:chat_id, :amount, NOW())
        ON CONFLICT (chat_id) DO UPDATE
            SET trade_amount=EXCLUDED.trade_amount, updated_at=NOW()
    """)
    try:
        with engine.connect() as conn:
            conn.execute(sql, {"chat_id": chat_id, "amount": amount})
            conn.commit()
        return True
    except Exception as exc:
        logger.error(f"set_amount error: {exc}")
        return False


def get_subscriber(chat_id: str) -> Optional[dict]:
    """Get subscriber info."""
    engine = get_engine()
    sql = text("""
        SELECT chat_id, username, deriv_token, trade_amount, connected_at
        FROM subscribers WHERE chat_id=:chat_id
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(sql, {"chat_id": chat_id}).fetchone()
        if result:
            return {
                "chat_id":      result[0],
                "username":     result[1],
                "deriv_token":  result[2],
                "trade_amount": float(result[3]) if result[3] else 10.0,
                "connected_at": result[4],
            }
        return None
    except Exception as exc:
        logger.error(f"get_subscriber error: {exc}")
        return None


def get_all_connected() -> list:
    """Get all subscribers with connected Deriv accounts."""
    engine = get_engine()
    sql = text("""
        SELECT chat_id, deriv_token, trade_amount
        FROM subscribers
        WHERE deriv_token IS NOT NULL AND deriv_token != ''
    """)
    try:
        with engine.connect() as conn:
            results = conn.execute(sql).fetchall()
        return [
            {"chat_id": r[0], "deriv_token": r[1], "trade_amount": float(r[2])}
            for r in results
        ]
    except Exception as exc:
        logger.error(f"get_all_connected error: {exc}")
        return []


# ---------------------------------------------------------------------------
# Multi-platform layer (ADDITIVE — used for Pocket Option and any future
# broker integrations). Does not touch the `subscribers` table above, which
# stays exactly as-is for the existing Deriv flow.
# ---------------------------------------------------------------------------

def init_platform_tables():
    """Create the generic bot-user + per-platform credential tables."""
    engine = get_engine()
    ddl = """
    CREATE TABLE IF NOT EXISTS bot_users (
        telegram_id  VARCHAR(50) PRIMARY KEY,
        username     VARCHAR(100),
        first_name   VARCHAR(100),
        created_at   TIMESTAMPTZ DEFAULT NOW(),
        updated_at   TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS platform_credentials (
        id           SERIAL PRIMARY KEY,
        telegram_id  VARCHAR(50) NOT NULL,
        platform     VARCHAR(30) NOT NULL,
        credentials  JSONB NOT NULL DEFAULT '{}'::jsonb,
        is_demo      BOOLEAN DEFAULT TRUE,
        assets       TEXT,
        active       BOOLEAN DEFAULT TRUE,
        created_at   TIMESTAMPTZ DEFAULT NOW(),
        updated_at   TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (telegram_id, platform)
    );

    CREATE INDEX IF NOT EXISTS idx_platform_credentials_platform
        ON platform_credentials (platform) WHERE active = TRUE;
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    logger.info("Platform tables initialised.")


class UserManager:
    """
    Generic per-user / per-platform account manager.

    Sits alongside the older module-level functions above (which remain the
    source of truth for the original Deriv flow). This class is what new
    broker integrations — starting with Pocket Option — should use.
    """

    def __init__(self):
        self.engine = get_engine()

    # -- registration ------------------------------------------------------

    def register_user_from_telegram(self, telegram_id, username: str = "",
                                     first_name: str = "") -> bool:
        """
        Called on /start. Upserts a row into bot_users so every Telegram
        user who has ever messaged the bot is known, independent of which
        (if any) trading platform they later connect.
        """
        sql = text("""
            INSERT INTO bot_users (telegram_id, username, first_name, updated_at)
            VALUES (:telegram_id, :username, :first_name, NOW())
            ON CONFLICT (telegram_id) DO UPDATE
                SET username=EXCLUDED.username,
                    first_name=EXCLUDED.first_name,
                    updated_at=NOW()
        """)
        try:
            with self.engine.connect() as conn:
                conn.execute(sql, {
                    "telegram_id": str(telegram_id),
                    "username": username or "",
                    "first_name": first_name or "",
                })
                conn.commit()
            return True
        except Exception as exc:
            logger.error(f"register_user_from_telegram error: {exc}")
            return False

    # -- platform credentials ----------------------------------------------

    def save_platform_credentials(self, telegram_id, platform: str,
                                   credentials: dict, is_demo: bool = True,
                                   assets: Optional[list] = None) -> bool:
        """
        Store per-user broker credentials, e.g. for Pocket Option:
            credentials = {"session": "<PO_SESSION>", "uid": "<PO_UID>"}
        """
        sql = text("""
            INSERT INTO platform_credentials
                (telegram_id, platform, credentials, is_demo, assets, active, updated_at)
            VALUES
                (:telegram_id, :platform, CAST(:credentials AS JSONB), :is_demo, :assets, TRUE, NOW())
            ON CONFLICT (telegram_id, platform) DO UPDATE
                SET credentials=CAST(:credentials AS JSONB),
                    is_demo=EXCLUDED.is_demo,
                    assets=COALESCE(EXCLUDED.assets, platform_credentials.assets),
                    active=TRUE,
                    updated_at=NOW()
        """)
        try:
            with self.engine.connect() as conn:
                conn.execute(sql, {
                    "telegram_id": str(telegram_id),
                    "platform": platform,
                    # Encrypt the credential blob at rest. When encryption is
                    # enabled this stores an `enc:v1:` ciphertext string inside
                    # the JSONB column (a JSON string is valid JSONB); in dev
                    # (no key) it stays plain JSON. Reads decrypt transparently.
                    "credentials": json.dumps(crypto_util.encrypt_dict(credentials)),
                    "is_demo": bool(is_demo),
                    "assets": ",".join(assets) if assets else None,
                })
                conn.commit()
            return True
        except Exception as exc:
            logger.error(f"save_platform_credentials error: {exc}")
            return False

    def get_platform_credentials(self, telegram_id, platform: str) -> Optional[dict]:
        sql = text("""
            SELECT telegram_id, credentials, is_demo, assets, active
            FROM platform_credentials
            WHERE telegram_id=:telegram_id AND platform=:platform
        """)
        try:
            with self.engine.connect() as conn:
                row = conn.execute(sql, {
                    "telegram_id": str(telegram_id), "platform": platform
                }).fetchone()
            if not row:
                return None
            return {
                "telegram_id": row[0],
                "credentials": crypto_util.decrypt_to_dict(row[1]),
                "is_demo":     bool(row[2]),
                "assets":      row[3].split(",") if row[3] else [],
                "active":      bool(row[4]),
            }
        except Exception as exc:
            logger.error(f"get_platform_credentials error: {exc}")
            return None

    def remove_platform_credentials(self, telegram_id, platform: str) -> bool:
        sql = text("""
            UPDATE platform_credentials
            SET active=FALSE, updated_at=NOW()
            WHERE telegram_id=:telegram_id AND platform=:platform
        """)
        try:
            with self.engine.connect() as conn:
                conn.execute(sql, {"telegram_id": str(telegram_id), "platform": platform})
                conn.commit()
            return True
        except Exception as exc:
            logger.error(f"remove_platform_credentials error: {exc}")
            return False

    def deactivate_platform_credentials(self, telegram_id, platform: str,
                                        reason: str = "") -> bool:
        """
        Mark a user's platform credentials INACTIVE (used by the stream
        supervisor's circuit breaker after repeated auth failures, e.g. an
        expired Pocket Option session). The row is kept so the user can
        re-activate by re-running /connectpo; it's just skipped by
        get_all_platform_users until then.
        """
        sql = text("""
            UPDATE platform_credentials
            SET active=FALSE, updated_at=NOW()
            WHERE telegram_id=:telegram_id AND platform=:platform
        """)
        try:
            with self.engine.connect() as conn:
                conn.execute(sql, {"telegram_id": str(telegram_id), "platform": platform})
                conn.commit()
            logger.info(
                f"deactivate_platform_credentials: {telegram_id}/{platform} "
                f"deactivated ({reason or 'no reason given'})."
            )
            return True
        except Exception as exc:
            logger.error(f"deactivate_platform_credentials error: {exc}")
            return False

    def set_platform_assets(self, telegram_id, platform: str, assets: list) -> bool:
        sql = text("""
            UPDATE platform_credentials
            SET assets=:assets, updated_at=NOW()
            WHERE telegram_id=:telegram_id AND platform=:platform
        """)
        try:
            with self.engine.connect() as conn:
                conn.execute(sql, {
                    "telegram_id": str(telegram_id),
                    "platform": platform,
                    "assets": ",".join(assets) if assets else None,
                })
                conn.commit()
            return True
        except Exception as exc:
            logger.error(f"set_platform_assets error: {exc}")
            return False

    def get_all_platform_users(self, platform: str) -> list:
        """
        All users with ACTIVE credentials for a given platform. Used by the
        platform engine (e.g. pocket_option_engine) to know who to open a
        stream for. A missing/blank credential for one user is simply
        skipped by the caller — it never affects other users.
        """
        sql = text("""
            SELECT telegram_id, credentials, is_demo, assets
            FROM platform_credentials
            WHERE platform=:platform AND active=TRUE
        """)
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(sql, {"platform": platform}).fetchall()
            out = []
            for r in rows:
                creds = crypto_util.decrypt_to_dict(r[1])
                if not creds:
                    continue
                out.append({
                    "telegram_id": r[0],
                    "credentials": creds,
                    "is_demo":     bool(r[2]),
                    "assets":      r[3].split(",") if r[3] else [],
                })
            return out
        except Exception as exc:
            logger.error(f"get_all_platform_users error: {exc}")
            return []
