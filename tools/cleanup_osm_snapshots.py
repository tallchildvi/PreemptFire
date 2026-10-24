import argparse
import re
import time
from pathlib import Path

# andalucia-history.osh_20200314.osm.pbf matches, andalucia-history.osh.pbf does not
SNAPSHOT_RE = re.compile(r"^.+-history\.osh_\d{8}\.osm\.pbf$")


def find_snapshots(folder: Path) -> list[tuple[Path, float, int]]:
    result = []
    for p in folder.iterdir():
        if not p.is_file() or not SNAPSHOT_RE.match(p.name):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        result.append((p, st.st_mtime, st.st_size))
    return sorted(result, key=lambda x: x[1])  # oldest first


def cleanup(folder: Path, max_age_sec: float, keep_newest: int, dry_run: bool) -> None:
    files = find_snapshots(folder)
    protected = {p for p, _, _ in files[len(files) - keep_newest:]} if keep_newest > 0 else set()
    now = time.time()
    deleted = skipped = 0
    freed = 0

    for p, mtime, size in files:
        age = now - mtime
        if p in protected or age < max_age_sec:
            continue
        if dry_run:
            print(f"[dry-run] would delete {p.name} ({size/1e6:.0f} mb, {age/60:.0f} min)")
            continue
        try:
            p.unlink()
            deleted += 1
            freed += size
            print(f"deleted {p.name} ({size/1e6:.0f} mb, age {age/60:.0f} min)")
        except FileNotFoundError:
            continue
        except OSError as e:
            skipped += 1
            print(f"skipped {p.name}: file is in use ({e}), will try on the next pass")

    if deleted or skipped:
        print(f"[{time.strftime('%H:%M:%S')}] deleted {deleted}, freed {freed/1e6:.0f} mb, skipped {skipped}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default="data/osm_raw")
    ap.add_argument("--max-age-minutes", type=float, default=15.0, help="how long a snapshot lives")
    ap.add_argument("--interval-minutes", type=float, default=5.0, help="how often to scan the folder")
    ap.add_argument("--keep-newest", type=int, default=0,
                    help="always keep the n newest snapshots regardless of age (default 0)")
    ap.add_argument("--dry-run", action="store_true", help="only show what would be deleted")
    ap.add_argument("--once", action="store_true", help="do a single pass and exit")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"folder not found: {folder}")

    max_age = args.max_age_minutes * 60
    print(f"cleaning {folder}: snapshots older than {args.max_age_minutes:.0f} min, "
          f"scanning every {args.interval_minutes:.0f} min (ctrl+c to stop)")

    try:
        while True:
            cleanup(folder, max_age, args.keep_newest, args.dry_run)
            if args.once:
                return
            time.sleep(args.interval_minutes * 60)
    except KeyboardInterrupt:
        print("\nstopped by user")


if __name__ == "__main__":
    main()