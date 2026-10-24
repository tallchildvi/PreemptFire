"""
continue from the given scene: all scenes from it onward (in generator order,
is_fire=1 first, then by ascending date) are set back to PENDING.
completed scenes are not touched. scenes before the given one are not touched.

a backup of the db is made before any changes. stop the generator before running this script.

examples:
    python rollback_db.py SCENE_38.9893_-0.5510_20250217
    python rollback_db.py SCENE_38.9893_-0.5510_20250217 --dry-run
"""
import argparse
import sqlite3
import time
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_id", help="scene to continue from (inclusive)")
    ap.add_argument("--db", default="data/processed/dataset_checkpoint.db")
    ap.add_argument("--dry-run", action="store_true", help="only show what would be changed")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"db not found: {db}")

    con = sqlite3.connect(db, timeout=60.0)
    try:
        anchor = con.execute(
            "SELECT is_fire, target_date FROM scenes_tracker WHERE scene_id = ?", (args.scene_id,)
        ).fetchone()
        if anchor is None:
            raise SystemExit(f"scene {args.scene_id} not found in db")
        is_fire, target_date = anchor

        # same order as in the generator: ORDER BY is_fire DESC, target_date ASC
        rows = con.execute(
            """
            SELECT scene_id, status FROM scenes_tracker
            WHERE status != 'COMPLETED'
              AND (is_fire < ? OR (is_fire = ? AND target_date >= ?))
            ORDER BY is_fire DESC, target_date ASC
            """,
            (is_fire, is_fire, target_date),
        ).fetchall()

        if not rows:
            print("no scenes to change")
            return

        counts = Counter(s for _, s in rows)
        print(f"will be set back to PENDING: {len(rows)} scenes {dict(counts)}")
        print(f"  first: {rows[0][0]}\n  last: {rows[-1][0]}")

        if args.dry_run:
            print("[dry-run] no changes made")
            return

        backup = db.with_name(f"{db.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}.db")
        dst = sqlite3.connect(backup)
        with dst:
            con.backup(dst)
        dst.close()
        print(f"backup saved: {backup}")

        cols = {r[1] for r in con.execute("PRAGMA table_info(scenes_tracker)")}
        set_clause = "status='PENDING', patches_count=0, error_msg=NULL, updated_at=?"
        if "attempts" in cols:
            set_clause += ", attempts=0"
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with con:
            con.executemany(
                f"UPDATE scenes_tracker SET {set_clause} WHERE scene_id=?",
                [(now, sid) for sid, _ in rows],
            )
        print("done, you can start the generator")
    finally:
        con.close()


if __name__ == "__main__":
    main()