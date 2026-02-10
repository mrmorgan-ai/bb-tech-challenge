import sys
import uuid
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PREDICTIONS_DB, DATA_SCHEMA_VERSION


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    request_id TEXT NOT NULL UNIQUE,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    data_schema_version TEXT NOT NULL,
    features_hash TEXT NOT NULL,
    score REAL NOT NULL,
    predicted_label INTEGER NOT NULL,
    true_label INTEGER,
    latency_ms REAL NOT NULL,
    status TEXT NOT NULL
);
"""

class PredictionLogger:
    """
    Logs prediction events to a local SQLite database.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or PREDICTIONS_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create the predictions table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)

    @contextmanager
    def _connect(self):
        """
        Context manager for SQLite connections.

        Guarantees:
        - Connection is always closed (even on exception)
        - Successful operations are committed
        - Failed operations are rolled back (no partial writes)
        - WAL mode is enabled for concurrent read/write support
        """
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
        finally:
            conn.close()

    def log_prediction(
        self,
        model_name: str,
        model_version: str,
        features_hash: str,
        score: float,
        predicted_label: int,
        true_label: Optional[int] = None,
        latency_ms: float = 0.0,
        status: str = "success",
    ) -> str:
        """
        Log a single prediction event. Returns the generated request_id.

        The request_id (UUID4) is generated here because:
        1. Uniqueness is guaranteed by UUID4 (122 bits of randomness)
        2. The caller doesn't need to manage ID generation
        3. It serves as the receipt. Callers can store it for later lookup
        """
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                f"""INSERT INTO predictions
                    (timestamp, request_id, model_name, model_version,
                    data_schema_version, features_hash, score,
                    predicted_label, true_label, latency_ms, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, request_id, model_name, model_version, 
                        DATA_SCHEMA_VERSION, features_hash, score, predicted_label,
                        true_label, latency_ms, status)
            )
        return request_id

    def get_predictions(self, limit: int = 100, model_name: Optional[str] = None):
        """Retrieve recent predictions as list of dicts."""
        query = "SELECT * FROM predictions"
        params = []
        if model_name:
            query += " WHERE model_name = ?"
            params.append(model_name)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_scores(self, limit: int = 1000):
        """Retrieve scores for monitoring analysis."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT score, predicted_label, latency_ms, status "
                "FROM predictions ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return rows

    def count(self) -> int:
        """Total number of logged predictions."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
