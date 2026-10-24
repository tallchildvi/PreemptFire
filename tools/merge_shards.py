"""
merges all hdf5 shards from a folder into one file. originals are not changed or deleted.

the schema (datasets, shapes, dtypes, attributes) is taken from the first valid shard.
shards that do not match the schema (other channels, corrupted, cannot be opened) are skipped
and the reasons are printed at the end. the result is written to .tmp and swapped in atomically.

examples:
    python merge_shards.py data/processed/shards_1 data/processed/dataset.h5
    python merge_shards.py data/processed/shards_1 data/processed/dataset.h5 --overwrite
"""
import argparse
import os
from pathlib import Path
import shutil

import h5py
import numpy as np

BATCH = 8  # how many patches to copy at once


def snapshot_attrs(attrs) -> dict:
    """copies attributes keeping h5py compatible types"""
    snap = {}
    for k in attrs.keys():
        v = attrs[k]
        if isinstance(v, (str, bytes, int, float, bool)):
            snap[k] = v
        elif isinstance(v, np.ndarray):
            if v.dtype.kind == "U":
                snap[k] = str(v.item()) if v.ndim == 0 else [str(x) for x in v]
            elif v.dtype.kind == "S":
                snap[k] = v.item().decode("utf-8") if v.ndim == 0 else [x.decode("utf-8") for x in v]
            else:
                snap[k] = np.asarray(v)
        elif isinstance(v, (np.str_, np.bytes_)):
            snap[k] = str(v)
        else:
            snap[k] = v
    return snap


def set_h5_attr(attrs, key: str, val) -> None:
    """writes an hdf5 attribute safely, avoids the dtype('<U...') error"""
    if isinstance(val, str):
        attrs[key] = val
    elif isinstance(val, (int, float, bool, bytes)):
        attrs[key] = val
    elif isinstance(val, (list, tuple)) and all(isinstance(x, str) for x in val):
        dt = h5py.string_dtype(encoding="utf-8")
        attrs.create(key, data=np.array(val, dtype=dt))
    elif isinstance(val, np.ndarray):
        if val.dtype.kind == "U":
            if val.ndim == 0:
                attrs[key] = str(val.item())
            else:
                dt = h5py.string_dtype(encoding="utf-8")
                attrs.create(key, data=np.array([str(x) for x in val], dtype=dt))
        else:
            attrs[key] = val
    else:
        attrs[key] = val


def attrs_equal(ref: dict, attrs) -> bool:
    if set(ref.keys()) != set(attrs.keys()):
        return False
    cur = snapshot_attrs(attrs)
    for k in ref:
        a, b = ref[k], cur[k]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not np.array_equal(np.asarray(a), np.asarray(b)):
                return False
        elif a != b:
            return False
    return True


def read_schema(f: h5py.File) -> dict:
    schema = {}
    for k in f.keys():
        ds = f[k]
        if not isinstance(ds, h5py.Dataset):
            raise ValueError(f"'{k}' is not a dataset")
        schema[k] = {
            "tail": ds.shape[1:],
            "dtype": ds.dtype,
            "chunks": ds.chunks,
            "compression": ds.compression,
            "compression_opts": ds.compression_opts,
            "shuffle": ds.shuffle,
        }
    return schema


def check_compatible(f: h5py.File, schema: dict, ref_attrs: dict) -> str | None:
    """returns the reason for incompatibility, or None if the file fits"""
    keys = set(f.keys())
    if keys != set(schema):
        return f"different set of datasets: {sorted(keys ^ set(schema))}"
    lengths = set()
    for k, spec in schema.items():
        ds = f[k]
        if not isinstance(ds, h5py.Dataset):
            return f"'{k}' is not a dataset"
        if ds.shape[1:] != spec["tail"]:
            return f"{k}: shape {ds.shape[1:]} instead of {spec['tail']}"
        lengths.add(ds.shape[0])
    if len(lengths) != 1:
        return "different number of patches across datasets (file corrupted?)"
    if not attrs_equal(ref_attrs, f.attrs):
        return "attributes (e.g. channel list) differ"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="folder with shards")
    ap.add_argument("output", type=Path, help="path to the final .h5 file")
    ap.add_argument("--pattern", default="*.h5", help="file mask (default *.h5)")
    ap.add_argument("--overwrite", action="store_true", help="allow overwriting an existing output file")
    args = ap.parse_args()

    in_dir, out_path = args.input_dir, args.output
    if not in_dir.is_dir():
        raise SystemExit(f"folder not found: {in_dir}")
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"file already exists: {out_path} (add --overwrite to replace it)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    out_resolved = out_path.resolve()
    files = [
        p for p in sorted(in_dir.glob(args.pattern))
        if p.resolve() != out_resolved and not p.name.endswith(".tmp.h5")
    ]
    if not files:
        raise SystemExit("no files found")
    print(f"found {len(files)} files, checking schema")

    # pass 1: schema and compatibility
    schema = ref_attrs = None
    good: list[tuple[Path, int]] = []
    skipped: list[tuple[str, str]] = []
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                if schema is None:
                    schema = read_schema(f)
                    ref_attrs = snapshot_attrs(f.attrs)
                reason = check_compatible(f, schema, ref_attrs)
                n = next(iter(f.values())).shape[0] if reason is None else 0
        except Exception as e:
            skipped.append((p.name, f"cannot be opened ({e})"))
            continue
        if reason:
            skipped.append((p.name, reason))
        else:
            good.append((p, n))

    if not good:
        raise SystemExit("no compatible files")

    total_patches = sum(n for _, n in good)
    total_size = sum(p.stat().st_size for p, _ in good)
    free = shutil.disk_usage(out_path.parent).free
    print(f"before merge: {len(good)} files, {total_patches} patches, ~{total_size/1e9:.2f} gb, "
          f"free disk space: {free/1e9:.2f} gb")
    if free < total_size * 1.05:
        raise SystemExit("not enough disk space for the result")

    # pass 2: copying
    mismatch_count = 0
    with h5py.File(tmp_path, "w") as out:
        for k, v in ref_attrs.items():
            set_h5_attr(out.attrs, k, v)

        dsets = {}
        for k, spec in schema.items():
            kwargs = {}
            if spec["compression"]:
                kwargs["compression"] = spec["compression"]
                if spec["compression_opts"] is not None:
                    kwargs["compression_opts"] = spec["compression_opts"]
            if spec["shuffle"]:
                kwargs["shuffle"] = True
            dsets[k] = out.create_dataset(
                k,
                shape=(0, *spec["tail"]),
                maxshape=(None, *spec["tail"]),
                dtype=spec["dtype"],
                chunks=spec["chunks"] if spec["chunks"] else True,
                **kwargs,
            )

        written = 0
        for idx, (p, n) in enumerate(good, 1):
            try:
                with h5py.File(p, "r") as f:
                    for k in schema:
                        dsets[k].resize(written + n, axis=0)
                    for i in range(0, n, BATCH):
                        j = min(i + BATCH, n)
                        for k in schema:
                            dsets[k][written + i:written + j] = f[k][i:j]

                    # compare the first and last patch as a sanity check
                    if n > 0 and "X_2d" in schema:
                        for local in {0, n - 1}:
                            a = f["X_2d"][local].astype(schema["X_2d"]["dtype"])
                            b = dsets["X_2d"][written + local]
                            if not np.array_equal(a, b, equal_nan=True):
                                mismatch_count += 1
                                raise RuntimeError(f"sanity check failed (patch {local})")
            except Exception as e:
                raise SystemExit(f"error while copying {p.name}: {e}. "
                                 f"originals are untouched, the partial file {tmp_path.name} can be deleted")
            written += n
            print(f"({idx}/{len(good)}) {p.name}: {n} patches (total {written})")

        if "patch_id" in dsets:
            ids = dsets["patch_id"][:]
            dup = len(ids) - len(set(ids.tolist()))
            if dup:
                print(f"warning: {dup} duplicate patch_id in the final file (same scene in several shards?)")

    with open(tmp_path, "r+b") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp_path, out_path)

    print(f"\ndone: {written} patches from {len(good)} files to {out_path} ({out_path.stat().st_size/1e9:.2f} gb)")
    if skipped:
        print(f"skipped {len(skipped)} files:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()