import json
from pathlib import Path
from typing import Any
import h5py
import numpy as np


class HDF5Writer:
    """manages streaming append of multimodal wildfire patches into hdf5 archives."""

    def __init__(
        self,
        output_path: Path | str = "data/processed/wildfire_dataset.h5",
        patch_size: int = 256,
        compression: str = "gzip",
        compression_opts: int = 4,
        **kwargs,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.patch_size = patch_size
        self.compression = compression
        self.compression_opts = compression_opts
        self._is_initialized = False

    def _init_storage(
        self,
        sample_patch: dict[str, Any],
        h5_file: h5py.File,
    ) -> None:
        """allocates resizable datasets and stores channel schemas in root attributes."""
        c_2d, h, w = sample_patch["X_2d"].shape
        c_1d = sample_patch["X_1d"].shape[0]
        
        val_vec = sample_patch.get("channel_validity")
        c_val = val_vec.shape[0] if val_vec is not None else c_2d
        c_y = sample_patch["Y"].shape[0]

        # 2d spatial feature tensor: [n, c_2d, h, w]
        h5_file.create_dataset(
            "X_2d",
            shape=(0, c_2d, h, w),
            maxshape=(None, c_2d, h, w),
            dtype="float32",
            chunks=(1, c_2d, h, w),
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

        # 1d environmental context vector: [n, k]
        h5_file.create_dataset(
            "X_1d",
            shape=(0, c_1d),
            maxshape=(None, c_1d),
            dtype="float32",
            chunks=True,
        )

        # 1d sensor modality presence mask: [n, c_2d]
        h5_file.create_dataset(
            "channel_validity",
            shape=(0, c_val),
            maxshape=(None, c_val),
            dtype="float32",
            chunks=True,
        )

        # 2d multi-scale continuous targets: [n, 4, h, w]
        h5_file.create_dataset(
            "Y",
            shape=(0, c_y, h, w),
            maxshape=(None, c_y, h, w),
            dtype="float32",
            chunks=(1, c_y, h, w),
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

        # 2d binary loss mask: [n, h, w]
        h5_file.create_dataset(
            "loss_mask",
            shape=(0, h, w),
            maxshape=(None, h, w),
            dtype="float32",
            chunks=(1, h, w),
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

        # metadata storage as stringified json
        dt_str = h5py.string_dtype(encoding="utf-8")
        h5_file.create_dataset(
            "metadata",
            shape=(0,),
            maxshape=(None,),
            dtype=dt_str,
            chunks=True,
        )
        h5_file.create_dataset(
            "patch_id",
            shape=(0,),
            maxshape=(None,),
            dtype=dt_str,
            chunks=True,
        )

        # persist canonical channel names in hdf5 root attributes
        h5_file.attrs["channel_names_2d"] = json.dumps(
            sample_patch.get("channel_names_2d", [])
        )
        h5_file.attrs["context_names_1d"] = json.dumps(
            sample_patch.get("context_names_1d", [])
        )
        self._is_initialized = True

    def write_patches_batch(self, patches: list[dict[str, Any]]) -> int:
        """appends a batch of patches to the hdf5 storage in a single io pass."""
        if not patches:
            return 0

        batch_size = len(patches)

        # stack batch arrays in memory
        b_x2d = np.stack([p["X_2d"] for p in patches], axis=0).astype(
            np.float32
        )
        b_x1d = np.stack([p["X_1d"] for p in patches], axis=0).astype(
            np.float32
        )
        num_channels_2d = b_x2d.shape[1]
        b_val = np.stack([
            p.get("channel_validity", np.ones(num_channels_2d, dtype=np.float32))
            for p in patches
        ], axis=0).astype(np.float32)

        b_y = np.stack([p["Y"] for p in patches], axis=0).astype(np.float32)
        b_mask = np.stack([p["loss_mask"] for p in patches], axis=0).astype(
            np.float32
        )
        b_ids = [str(p["patch_id"]) for p in patches]
        b_meta = [json.dumps(p["metadata"]) for p in patches]

        with h5py.File(self.output_path, "a") as f:
            if not self._is_initialized and "X_2d" not in f:
                self._init_storage(patches[0], f)
            else:
                self._is_initialized = True

            curr_len = f["X_2d"].shape[0]
            new_len = curr_len + batch_size

            # resize all parallel dataset arrays
            f["X_2d"].resize((new_len, *b_x2d.shape[1:]))
            f["X_1d"].resize((new_len, *b_x1d.shape[1:]))
            f["channel_validity"].resize((new_len, *b_val.shape[1:]))
            f["Y"].resize((new_len, *b_y.shape[1:]))
            f["loss_mask"].resize((new_len, *b_mask.shape[1:]))
            f["metadata"].resize((new_len,))
            f["patch_id"].resize((new_len,))

            # write slice to disk
            f["X_2d"][curr_len:new_len] = b_x2d
            f["X_1d"][curr_len:new_len] = b_x1d
            f["channel_validity"][curr_len:new_len] = b_val
            f["Y"][curr_len:new_len] = b_y
            f["loss_mask"][curr_len:new_len] = b_mask
            f["metadata"][curr_len:new_len] = b_meta
            f["patch_id"][curr_len:new_len] = b_ids

        return batch_size

def inspect_h5_structure(h5_path: Path | str) -> None:
        """inspects and logs datasets, shapes, dtypes, and attributes of the generated hdf5 archive."""
        path = Path(h5_path)
        if not path.exists():
            print(f"  [HDF5 Inspect] File not found at: {path}")
            return

        with h5py.File(path, "r") as f:
            print(f"\n{'='*55}\nHDF5 Archive Inspection: {path.name}\n{'='*55}")
            print("Datasets:")
            for key in f.keys():
                ds = f[key]
                print(f"  - {key:<18} shape: {str(ds.shape):<20} dtype: {ds.dtype}")

            print("\nRoot Attributes:")
            for attr_name, attr_val in f.attrs.items():
                if isinstance(attr_val, str) and (attr_val.startswith("[") or attr_val.startswith("{")):
                    try:
                        parsed = json.loads(attr_val)
                        print(f"  - {attr_name}: {len(parsed)} registered items")
                    except Exception:
                        print(f"  - {attr_name}: {attr_val[:60]}...")
                else:
                    print(f"  - {attr_name}: {attr_val}")
            print(f"{'='*55}\n")    