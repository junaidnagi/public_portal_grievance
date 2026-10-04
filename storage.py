"""SQLite storage scoped to a private recovery token; not an authentication system."""
import hashlib
import json
import sqlite3
import os
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.environ.get('CASE_DB_PATH', str(Path(__file__).parent / 'cases.sqlite3')))

def owner_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

@contextmanager
def connect():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.execute('CREATE TABLE IF NOT EXISTS cases (owner TEXT, id TEXT, body TEXT, PRIMARY KEY(owner,id))')
    try:
        DB_PATH.chmod(0o600)
    except OSError:
        pass
    try:
        with connection:
            yield connection
    finally:
        connection.close()

def save_case(token: str, case: dict) -> None:
    with connect() as db:
        db.execute('INSERT OR REPLACE INTO cases VALUES (?, ?, ?)', (owner_hash(token), case['id'], json.dumps(case, ensure_ascii=False)))

def load_cases(token: str) -> dict:
    with connect() as db:
        rows = db.execute('SELECT id, body FROM cases WHERE owner=?', (owner_hash(token),)).fetchall()
    return {case_id: json.loads(body) for case_id, body in rows}

def delete_case(token: str, case_id: str) -> None:
    with connect() as db:
        db.execute('DELETE FROM cases WHERE owner=? AND id=?', (owner_hash(token), case_id))
