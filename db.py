import sqlite3
from datetime import datetime

DB_FILE = "jarvis.db"


def connect():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def setup():
    conn = connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            caller_number TEXT,
            caller_name   TEXT,
            is_family     INTEGER NOT NULL DEFAULT 0,
            started_at    TEXT NOT NULL,
            ended_at      TEXT,
            outcome       TEXT,
            connected     INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id INTEGER NOT NULL,
            at      TEXT NOT NULL,
            kind    TEXT NOT NULL,
            detail  TEXT,
            FOREIGN KEY (call_id) REFERENCES calls (id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id     INTEGER,
            kind        TEXT NOT NULL,
            caller_name TEXT,
            content     TEXT NOT NULL,
            urgency     TEXT,
            created_at  TEXT NOT NULL,
            done        INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (call_id) REFERENCES calls (id)
        )
    """)
    try:
        conn.execute("ALTER TABLE calls ADD COLUMN duration_seconds INTEGER")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def now():
    return datetime.now().isoformat(timespec="seconds")


def start_call(caller_number, is_family):
    conn = connect()
    cursor = conn.execute(
        "INSERT INTO calls (caller_number, is_family, started_at) VALUES (?, ?, ?)",
        (caller_number, 1 if is_family else 0, now()),
    )
    conn.commit()
    call_id = cursor.lastrowid
    conn.close()
    return call_id


def log(call_id, kind, detail):
    conn = connect()
    conn.execute(
        "INSERT INTO events (call_id, at, kind, detail) VALUES (?, ?, ?, ?)",
        (call_id, now(), kind, detail),
    )
    conn.commit()
    conn.close()


def end_call(call_id, outcome, connected, caller_name=None):
    conn = connect()
    row = conn.execute("SELECT started_at FROM calls WHERE id = ?", (call_id,)).fetchone()
    started = datetime.fromisoformat(row["started_at"])
    duration = int((datetime.now() - started).total_seconds())
    conn.execute(
        "UPDATE calls SET ended_at = ?, outcome = ?, connected = ?, caller_name = ?, "
        "duration_seconds = ? WHERE id = ?",
        (now(), outcome, 1 if connected else 0, caller_name, duration, call_id),
    )
    conn.commit()
    conn.close()


def save_reminder(call_id, kind, caller_name, content, urgency=None):
    conn = connect()
    conn.execute(
        "INSERT INTO reminders (call_id, kind, caller_name, content, urgency, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (call_id, kind, caller_name, content, urgency, now()),
    )
    conn.commit()
    conn.close()


def replay(call_id):
    conn = connect()
    rows = conn.execute(
        "SELECT at, kind, detail FROM events WHERE call_id = ? ORDER BY id",
        (call_id,),
    ).fetchall()
    conn.close()
    return rows


def reminders_for_call(call_id):
    conn = connect()
    rows = conn.execute(
        "SELECT kind, caller_name, content, urgency FROM reminders WHERE call_id = ? ORDER BY id",
        (call_id,),
    ).fetchall()
    conn.close()
    return rows


def open_reminders():
    conn = connect()
    rows = conn.execute(
        "SELECT id, call_id, kind, caller_name, content, urgency, created_at "
        "FROM reminders WHERE done = 0 ORDER BY id"
    ).fetchall()
    conn.close()
    return rows


def recent_calls(limit):
    conn = connect()
    rows = conn.execute(
        "SELECT id, caller_number, caller_name, started_at, outcome, duration_seconds "
        "FROM calls ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    setup()
    print("Database ready:", DB_FILE)