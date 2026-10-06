"""A small SQLite cache for tool results, so repeated searches and fetches are free."""

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_results (
    cache_key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    stored_at TEXT NOT NULL
)
"""


def cache_key(tool_name: str, arguments: dict) -> str:
    """Build a deterministic key from a tool name and its arguments."""
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"


class ToolCache:
    """Stores JSON-serializable tool results with a time-to-live.

    A connection is opened per operation because parallel researchers call
    tools from different threads.
    """

    def __init__(self, path: Path, ttl: timedelta, clock: Callable[[], datetime]):
        """Open (creating if needed) the cache database.

        Args:
            path: SQLite file location.
            ttl: Age after which an entry is treated as missing.
            clock: Returns the current time; injected so expiry is testable.
        """
        self.path = path
        self.ttl = ttl
        self.clock = clock
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(SCHEMA)

    def get(self, tool_name: str, arguments: dict) -> list | dict | None:
        """Return the cached result, or None if absent or older than the TTL."""
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT value, stored_at FROM tool_results WHERE cache_key = ?",
                (cache_key(tool_name, arguments),),
            ).fetchone()
        if row is None or self.clock() - datetime.fromisoformat(row[1]) > self.ttl:
            return None
        return json.loads(row[0])

    def put(self, tool_name: str, arguments: dict, value: list | dict) -> None:
        """Store a result, replacing any earlier entry for the same call."""
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO tool_results VALUES (?, ?, ?)",
                (cache_key(tool_name, arguments), json.dumps(value), self.clock().isoformat()),
            )
