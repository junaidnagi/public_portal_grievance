"""Run citizen-authorized due escalations without an open Streamlit browser.

Use `python scheduler.py` for one cron pass, or `--watch` on an always-on host.
It uses the same protected secrets, cases database and outbox as the app.
No new complaint or authorization is created by this worker.
"""
import argparse
import copy
import json
import logging
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

LOGGER = logging.getLogger('grievance.scheduler')


def run_once(app, case_db: Path) -> dict:
    if not case_db.is_file():
        return {'owners_checked': 0, 'cases_updated': 0}
    with app.feature_database() as db:
        owners = [row[0] for row in db.execute('SELECT DISTINCT owner FROM escalation_jobs').fetchall()]
    updates = 0
    for owner in owners:
        connection = sqlite3.connect(case_db, timeout=10)
        try:
            rows = connection.execute('SELECT id,body FROM cases WHERE owner=?', (owner,)).fetchall()
        finally:
            connection.close()
        original = {case_id: raw for case_id, raw in rows}
        cases = {case_id: json.loads(raw) for case_id, raw in rows}
        changed = app.process_escalations(cases, owner)
        for case_id in changed:
            # A user can edit or close a case while transport is in progress.
            # Read it again in a write transaction and preserve those edits.
            connection = sqlite3.connect(case_db, timeout=10)
            try:
                connection.execute('BEGIN IMMEDIATE')
                row = connection.execute('SELECT body FROM cases WHERE owner=? AND id=?', (owner, case_id)).fetchone()
                if not row:
                    connection.rollback(); continue
                current = cases[case_id]
                if row[0] != original[case_id]:
                    current = json.loads(row[0])
                    with app.feature_database() as jobs:
                        job = jobs.execute('SELECT payload,state,result FROM escalation_jobs WHERE owner=? AND case_id=?', (owner, case_id)).fetchone()
                    if job:
                        payload, state, result = json.loads(job[0]), job[1], json.loads(job[2])
                        # process_escalations already verified the authorization.
                        # Replaying terminal state never dispatches a second email.
                        app.apply_escalation_result(current, payload, state, result)
                connection.execute('UPDATE cases SET body=? WHERE owner=? AND id=?',
                    (json.dumps(current, ensure_ascii=False), owner, case_id))
                connection.commit(); updates += 1
            finally:
                connection.close()
            app.sync_public_analytics(current, owner)
    return {'owners_checked': len(owners), 'cases_updated': updates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='Repeat on an always-on host; Ctrl+C stops the worker.')
    parser.add_argument('--interval', type=int, default=60, help='Seconds between passes, minimum 30.')
    parser.add_argument('--prepare-only', action='store_true', help='Suppress email jobs while processing preparation jobs.')
    parser.add_argument('--cases-db', type=Path, help='Override CASE_DB_PATH; must be the database used by the UI.')
    args = parser.parse_args()
    import app
    import storage
    # Streamlit secrets also work outside `streamlit run`. Remove UI/session
    # dependencies; transport still checks the signed snapshot and owner lock.
    original_st = app.st
    app.st = SimpleNamespace(secrets=original_st.secrets, errors=original_st.errors,
        session_state={'demo_mode': bool(args.prepare_only)},
        caption=lambda *a, **kw: None, warning=lambda *a, **kw: None)
    case_db = args.cases_db or storage.DB_PATH
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        while True:
            try:
                result = run_once(app, case_db)
                LOGGER.info('Checked %d owners; updated %d cases.', result['owners_checked'], result['cases_updated'])
            except Exception as error:
                # Do not log complaint bodies, addresses, documents or secrets.
                LOGGER.error('Worker pass failed (%s); review server configuration.', type(error).__name__)
                if not args.watch: return 1
            if not args.watch: break
            time.sleep(max(30, args.interval))
    except KeyboardInterrupt:
        LOGGER.info('Worker stopped.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
