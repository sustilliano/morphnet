import sqlite3
import json
from collections import deque
from datetime import datetime
from typing import List, Optional

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
except Exception:
    _analyzer = None

class ImmediateMemory:
    """Simple in-memory circular buffer of recent messages."""
    def __init__(self, buffer_size: int = 500):
        self.buffer_size = buffer_size
        self._buffer = deque(maxlen=buffer_size)

    def append(self, speaker: str, text: str) -> None:
        self._buffer.append({"speaker": speaker, "text": text})

    def get_recent(self, k: int) -> List[dict]:
        return list(self._buffer)[-k:]

class SessionStore:
    """SQLite-backed session log with basic tagging."""
    def __init__(self, db_path: str = "memory_lane.db", keywords: Optional[List[str]] = None, vader_threshold: float = 0.5):
        self.conn = sqlite3.connect(db_path)
        self.keywords = keywords or []
        self.vader_threshold = vader_threshold
        self._init_table()

    def _init_table(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS session (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                speaker TEXT,
                text TEXT,
                tags TEXT
            )
            """
        )
        self.conn.commit()

    def insert(self, speaker: str, text: str) -> int:
        tags = []
        for kw in self.keywords:
            if kw.lower() in text.lower():
                tags.append(kw)
        if _analyzer:
            score = _analyzer.polarity_scores(text)["compound"]
            if score >= self.vader_threshold:
                tags.append("positive")
            elif score <= -self.vader_threshold:
                tags.append("negative")
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO session(timestamp, speaker, text, tags) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), speaker, text, json.dumps(tags)),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_recent(self, limit: int = 100) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, timestamp, speaker, text, tags FROM session ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {"id": r[0], "timestamp": r[1], "speaker": r[2], "text": r[3], "tags": json.loads(r[4])}
            for r in rows
        ]

    def add_tag(self, entry_id: int, tag: str) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT tags FROM session WHERE id = ?", (entry_id,))
        row = cur.fetchone()
        if not row:
            return
        tags = json.loads(row[0])
        if tag not in tags:
            tags.append(tag)
        cur.execute("UPDATE session SET tags = ? WHERE id = ?", (json.dumps(tags), entry_id))
        self.conn.commit()

    def search(self, query: str) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, timestamp, speaker, text, tags FROM session WHERE text LIKE ?",
            (f"%{query}%",),
        )
        rows = cur.fetchall()
        return [
            {"id": r[0], "timestamp": r[1], "speaker": r[2], "text": r[3], "tags": json.loads(r[4])}
            for r in rows
        ]

