import argparse
import json
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np

DERIVED_CHANNELS = {
    # spectral indices
    "EVI_T0", "MSI_T0", "NBR_T0", "NBR2_T0", "NDMI_T0", "NDRE_T0", "NDVI_T0", "NMDI_T0",
    # temporal deltas
    "dNBR", "dNDMI", "dNDVI", "dNMDI", "dMSI",
    # sar indices (SAR_VV and SAR_VH stay)
    "SAR_RATIO", "SAR_RVI",
}

BATCH = 8              # how many patches to read at once
POLL_SECONDS = 30      # pause between scans of the input folder
STABLE_SECONDS = 20    # a file counts as ready if it has not changed for this long


def load_names(raw) -> list[str]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return list(json.loads(raw))
    return [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in raw]


def process_file(src: Path, dst: Path, x_dtype: str | None, y_dtype: str | None,
                 compression: str, level: int) -> tuple[int, int]:
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with h5py.File(src, "r") as fin:
        names = load_names(fin.attrs["channel_names_2d"])
        keep_idx = [i for i, n in enumerate(names) if n not in DERIVED_CHANNELS]
        new_names = [names[i] for i in keep_idx]

        if len(keep_idx) == len(names) and x_dtype is None and y_dtype is None:
            shutil.copy2(src, tmp)  # nothing to remove
        else:
            n, c_old, h, w = fin["X_2d"].shape
            c_new = len(keep_idx)
            x_dt = np.dtype(x_dtype) if x_dtype else fin["X_2d"].dtype

            with h5py.File(tmp, "w") as fout:
                for k, v in fin.attrs.items():
                    fout.attrs[k] = v
                fout.attrs["channel_names_2d"] = json.dumps(new_names)

                dx = fout.create_dataset(
                    "X_2d", shape=(n, c_new, h, w), maxshape=(None, c_new, h, w),
                    dtype=x_dt, chunks=(1, c_new, h, w),
                    compression=compression, compression_opts=level, shuffle=True,
                )

                cv = fin["channel_validity"]
                if cv.shape[1] != c_old:
                    raise ValueError(f"channel_validity ({cv.shape[1]}) != X_2d ({c_old})")
                dcv = fout.create_dataset(
                    "channel_validity", shape=(n, c_new), maxshape=(None, c_new),
                    dtype=cv.dtype, chunks=True,
                )

                for i in range(0, n, BATCH):
                    j = min(i + BATCH, n)
                    dx[i:j] = fin["X_2d"][i:j][:, keep_idx].astype(x_dt)
                    dcv[i:j] = cv[i:j][:, keep_idx]

                for key in fin.keys():
                    if key in ("X_2d", "channel_validity"):
                        continue
                    ds = fin[key]
                    if key == "Y" and y_dtype and np.dtype(y_dtype) != ds.dtype:
                        dy = fout.create_dataset(
                            "Y", shape=ds.shape, maxshape=(None, *ds.shape[1:]),
                            dtype=np.dtype(y_dtype), chunks=(1, *ds.shape[1:]),
                            compression=compression, compression_opts=level, shuffle=True,
                        )
                        for i in range(0, ds.shape[0], BATCH):
                            j = min(i + BATCH, ds.shape[0])
                            dy[i:j] = ds[i:j].astype(np.dtype(y_dtype))
                    else:
                        fin.copy(key, fout)

    # verify the new file before swapping it in
    with h5py.File(src, "r") as fin, h5py.File(tmp, "r") as fout:
        names_src = load_names(fin.attrs["channel_names_2d"])
        keep_idx = [i for i, n in enumerate(names_src) if n not in DERIVED_CHANNELS]
        assert load_names(fout.attrs["channel_names_2d"]) == [names_src[i] for i in keep_idx]
        for key in fin.keys():
            assert key in fout, f"dataset {key} is missing"
            assert fin[key].shape[0] == fout[key].shape[0], f"different number of patches in {key}"
        assert fout["X_2d"].shape[1] == len(keep_idx)
        assert fout["channel_validity"].shape[1] == len(keep_idx)
        if fin["X_2d"].shape[0] > 0:
            a = fin["X_2d"][0][keep_idx].astype(np.float32)
            b = fout["X_2d"][0].astype(np.float32)
            tol = 1e-3 if fout["X_2d"].dtype == np.float16 else 1e-6
            assert np.allclose(a, b, rtol=tol, atol=tol, equal_nan=True), "X_2d data does not match"

    os.replace(tmp, dst)
    return src.stat().st_size, dst.stat().st_size


def find_pending(in_dir: Path, out_dir: Path, failed: dict) -> list[tuple[Path, tuple]]:
    """files that are not processed yet, fully written and did not fail in this exact version"""
    pending = []
    now = time.time()
    for src in sorted(in_dir.glob("*.h5")):
        try:
            st = src.stat()
        except FileNotFoundError:
            continue
        dst = out_dir / src.name
        if dst.exists() and dst.stat().st_mtime >= st.st_mtime:
            continue  # already processed
        if now - st.st_mtime < STABLE_SECONDS:
            continue  # file may still be getting written
        sig = (st.st_mtime, st.st_size)
        if failed.get(src.name) == sig:
            continue  # this exact file already failed, wait until it changes
        pending.append((src, sig))
    return pending


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--x-dtype", default=None, help="e.g. float16 (unchanged by default)")
    ap.add_argument("--y-dtype", default=None, help="e.g. float16 (unchanged by default)")
    ap.add_argument("--compression", default="gzip")
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--delete-original", action="store_true")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    if in_dir.resolve() == out_dir.resolve():
        raise SystemExit("input_dir and output_dir must be different folders")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"watching {in_dir} and writing to {out_dir} (ctrl+c to stop)")

    failed: dict[str, tuple] = {}
    tot_in = tot_out = ok = 0
    idle_printed = False

    try:
        while True:
            pending = find_pending(in_dir, out_dir, failed)

            if not pending:
                if not idle_printed:
                    print("no new files, waiting")
                    idle_printed = True
                time.sleep(POLL_SECONDS)
                continue

            idle_printed = False
            for idx, (src, sig) in enumerate(pending, 1):
                dst = out_dir / src.name
                try:
                    size_in, size_out = process_file(src, dst, args.x_dtype, args.y_dtype,
                                                     args.compression, args.level)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"({idx}/{len(pending)}) {src.name}: error, skipped ({e})")
                    failed[src.name] = sig
                    tmp = dst.with_name(dst.name + ".tmp")
                    if tmp.exists():
                        tmp.unlink()
                    continue

                ok += 1
                tot_in += size_in
                tot_out += size_out
                print(f"({idx}/{len(pending)}) {src.name}: {size_in/1e6:.1f} mb to {size_out/1e6:.1f} mb")
                failed.pop(src.name, None)
                if args.delete_original:
                    src.unlink()
    except KeyboardInterrupt:
        print("\nstopped by user")
        for leftover in out_dir.glob("*.h5.tmp"):
            leftover.unlink(missing_ok=True)
        if tot_in:
            print(f"processed this run: {ok} files, {tot_in/1e9:.2f} gb to {tot_out/1e9:.2f} gb "
                  f"(saved {100 * (1 - tot_out / tot_in):.0f}%)")


if __name__ == "__main__":
    main()