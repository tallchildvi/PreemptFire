import sqlite3
con = sqlite3.connect("data/processed/dataset_checkpoint.db")
cur = con.execute(
    "UPDATE scenes_tracker SET status='SKIPPED_HANG', "
    "error_msg='hangs in collect_sample, skipped manually' WHERE scene_id=?",
    ("SCENE_42.3700_-5.7712_20170916",),
)
con.commit()
print("rows changed:", cur.rowcount)
con.close()