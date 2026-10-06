"""Persistent memory: a knowledge graph of entities and the findings that mention them.

Entities and findings are the nodes; a "mention" is an edge between them. A
later run looks up the entities its question is about and follows the edges to
every finding recorded about them.
"""

import json
import sqlite3
import string
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

from research_analyst.schemas import Finding

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY,
        canonical TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS findings (
        key TEXT PRIMARY KEY,
        statement TEXT NOT NULL,
        source_url TEXT NOT NULL,
        source_title TEXT NOT NULL,
        quote TEXT NOT NULL,
        entities TEXT NOT NULL,
        sub_question TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mentions (
        finding_key TEXT NOT NULL REFERENCES findings(key),
        entity_id INTEGER NOT NULL REFERENCES entities(id),
        PRIMARY KEY (finding_key, entity_id)
    )
    """,
)
LEADING_ARTICLE = "the "
PUNCTUATION_TO_SPACE = str.maketrans(string.punctuation, " " * len(string.punctuation))


class MemoryRecord(BaseModel):
    """A remembered finding with when it was last confirmed."""

    finding: Finding
    source_title: str
    recorded_at: datetime


def canonical_entity(name: str) -> str:
    """Reduce an entity name to the form used to detect duplicates.

    "The EU AI Act", "EU AI act" and "eu-ai-act" all become "eu ai act".
    """
    text = " ".join(name.casefold().translate(PUNCTUATION_TO_SPACE).split())
    return text.removeprefix(LEADING_ARTICLE)


def is_stale(record: MemoryRecord, now: datetime, max_age: timedelta) -> bool:
    """Tell whether a remembered finding is too old to use without re-verifying."""
    return now - record.recorded_at > max_age


class KnowledgeStore:
    """SQLite-backed knowledge graph of entities and findings.

    A connection is opened per operation so the store can be used from any thread.
    """

    def __init__(self, path: Path):
        """Open (creating if needed) the knowledge graph database at path."""
        self.path = path
        with closing(sqlite3.connect(path)) as connection, connection:
            for statement in SCHEMA:
                connection.execute(statement)

    def entity_names(self, limit: int) -> list[str]:
        """List known entity names, most-mentioned first."""
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                """
                SELECT entities.name FROM entities
                JOIN mentions ON mentions.entity_id = entities.id
                GROUP BY entities.id ORDER BY COUNT(*) DESC, entities.name LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [name for (name,) in rows]

    def save(self, findings: list[Finding], titles_by_url: dict[str, str], now: datetime) -> None:
        """Record findings and link them to their entities.

        Saving a finding that is already stored refreshes its timestamp, which is
        how a stale finding becomes fresh again after it is re-verified. Entity
        names that differ only in case, punctuation, or a leading "the" share one node.

        Args:
            findings: Supported findings to remember.
            titles_by_url: Source titles, looked up by each finding's source URL.
            now: The time to record as the moment of confirmation.
        """
        with closing(sqlite3.connect(self.path)) as connection, connection:
            for finding in findings:
                connection.execute(
                    "INSERT OR REPLACE INTO findings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        finding.key,
                        finding.statement,
                        finding.source_url,
                        titles_by_url.get(finding.source_url, finding.source_url),
                        finding.quote,
                        json.dumps(finding.entities),
                        finding.sub_question,
                        now.isoformat(),
                    ),
                )
                for name in finding.entities:
                    canonical = canonical_entity(name)
                    connection.execute(
                        "INSERT OR IGNORE INTO entities (canonical, name) VALUES (?, ?)",
                        (canonical, name),
                    )
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO mentions
                        SELECT ?, id FROM entities WHERE canonical = ?
                        """,
                        (finding.key, canonical),
                    )

    def recall(self, entity_names: list[str]) -> list[MemoryRecord]:
        """Return every finding that mentions any of the named entities, oldest first."""
        canonicals = sorted({canonical_entity(name) for name in entity_names})
        if not canonicals:
            return []
        placeholders = ", ".join("?" for _ in canonicals)
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                f"""
                SELECT DISTINCT findings.statement, findings.source_url, findings.source_title,
                       findings.quote, findings.entities, findings.sub_question,
                       findings.recorded_at
                FROM findings
                JOIN mentions ON mentions.finding_key = findings.key
                JOIN entities ON entities.id = mentions.entity_id
                WHERE entities.canonical IN ({placeholders})
                ORDER BY findings.recorded_at, findings.key
                """,
                canonicals,
            ).fetchall()
        return [
            MemoryRecord(
                finding=Finding(
                    statement=statement,
                    source_url=url,
                    quote=quote,
                    entities=json.loads(entities),
                    sub_question=sub_question,
                    from_memory=True,
                ),
                source_title=title,
                recorded_at=datetime.fromisoformat(recorded_at),
            )
            for statement, url, title, quote, entities, sub_question, recorded_at in rows
        ]
