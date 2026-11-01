import json
from pathlib import Path
from typing import Any
import h5py
import numpy as np

from src.config import PATCH_SIZE_PX

# Safe bounds for IEEE 754 float16 (max representable value is 65504.0)
FP16_MAX = 65500.0
FP16_MIN = -65500.0


class HDF5Writer:
    """Manages streaming append of raw multimodal wildfire patches into HDF5 archives without precomputed targets."""

    def __init__(
        self,
        output_path: Path | str = "data/processed/wildfire_dataset.h5",
        patch_size: int = PATCH_SIZE_PX,
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

    @staticmethod
    def _sanitize_fp16(arr: np.ndarray, min_val: float = FP16_MIN, max_val: float = FP16_MAX) -> np.ndarray:
        """Replaces NaN/Inf and clips values to safe float16 range before casting."""
        cleaned = np.nan_to_num(arr, nan=0.0, posinf=max_val, neginf=min_val)
        return np.clip(cleaned, min_val, max_val).astype(np.float16)

    def _init_storage(
        self,
        sample_patch: dict[str, Any],
        h5_file: h5py.File,
    ) -> None:
        """Allocates resizable datasets for raw 2D/1D features, SCL, and metadata."""
        c_2d, h, w = sample_patch["X_2d"].shape
        c_1d = sample_patch["X_1d"].shape[0]

        val_vec = sample_patch.get("channel_validity")
        c_val = val_vec.shape[0] if val_vec is not None else c_2d

        # Raw 2D spatial feature tensor: [N, 22, H, W]
        h5_file.create_dataset(
            "X_2d",
            shape=(0, c_2d, h, w),
            maxshape=(None, c_2d, h, w),
            dtype="float16",
            chunks=(1, c_2d, h, w),
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=True,
        )

        # Raw 1D meteorological & population context vector: [N, K]
        h5_file.create_dataset(
            "X_1d",
            shape=(0, c_1d),
            maxshape=(None, c_1d),
            dtype="float32",
            chunks=True,
        )

        # 1D sensor modality presence mask: [N, 22]
        h5_file.create_dataset(
            "channel_validity",
            shape=(0, c_val),
            maxshape=(None, c_val),
            dtype="float16",
            chunks=True,
        )

        # Raw 2D categorical scene classification layer: [N, H, W] (uint8: 0..11)
        h5_file.create_dataset(
            "scl",
            shape=(0, h, w),
            maxshape=(None, h, w),
            dtype="uint8",
            chunks=(1, h, w),
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=True,
        )

        # Binary label per scene/patch: [N,] (uint8: 0 or 1)
        h5_file.create_dataset(
            "is_fire",
            shape=(0,),
            maxshape=(None,),
            dtype="uint8",
            chunks=True,
        )

        dt_str = h5py.string_dtype(encoding="utf-8")
        h5_file.create_dataset("metadata", shape=(0,), maxshape=(None,), dtype=dt_str, chunks=True)
        h5_file.create_dataset("patch_id", shape=(0,), maxshape=(None,), dtype=dt_str, chunks=True)

        h5_file.attrs["channel_names_2d"] = json.dumps(sample_patch.get("channel_names_2d", []))
        h5_file.attrs["context_names_1d"] = json.dumps(sample_patch.get("context_names_1d", []))
        self._is_initialized = True

    def write_patches_batch(self, patches: list[dict[str, Any]]) -> int:
        """Appends a batch of raw patches to the HDF5 archive with float16 overflow protection."""
        if not patches:
            return 0

        batch_size = len(patches)

        raw_x2d = np.stack([p["X_2d"] for p in patches], axis=0)
        b_x2d = self._sanitize_fp16(raw_x2d, min_val=FP16_MIN, max_val=FP16_MAX)

        raw_x1d = np.stack([p["X_1d"] for p in patches], axis=0).astype(np.float32)
        b_x1d = np.nan_to_num(raw_x1d, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

        num_channels_2d = b_x2d.shape[1]
        raw_val = np.stack([
            p.get("channel_validity", np.ones(num_channels_2d, dtype=np.float32))
            for p in patches
        ], axis=0)
        b_val = self._sanitize_fp16(raw_val, min_val=0.0, max_val=1.0)

        b_scl = np.stack([p["scl"] for p in patches], axis=0).astype(np.uint8)
        b_fire = np.array([int(p.get("is_fire", p["metadata"]["is_fire"])) for p in patches], dtype=np.uint8)
        b_ids = [str(p["patch_id"]) for p in patches]
        b_meta = [json.dumps(p["metadata"]) for p in patches]

        with h5py.File(self.output_path, "a") as f:
            if not self._is_initialized and "X_2d" not in f:
                self._init_storage(patches[0], f)
            else:
                self._is_initialized = True

            curr_len = f["X_2d"].shape[0]
            new_len = curr_len + batch_size

            f["X_2d"].resize((new_len, *b_x2d.shape[1:]))
            f["X_1d"].resize((new_len, *b_x1d.shape[1:]))
            f["channel_validity"].resize((new_len, *b_val.shape[1:]))
            f["scl"].resize((new_len, *b_scl.shape[1:]))
            f["is_fire"].resize((new_len,))
            f["metadata"].resize((new_len,))
            f["patch_id"].resize((new_len,))

            f["X_2d"][curr_len:new_len] = b_x2d
            f["X_1d"][curr_len:new_len] = b_x1d
            f["channel_validity"][curr_len:new_len] = b_val
            f["scl"][curr_len:new_len] = b_scl
            f["is_fire"][curr_len:new_len] = b_fire
            f["metadata"][curr_len:new_len] = b_meta
            f["patch_id"][curr_len:new_len] = b_ids

        return batch_size


def inspect_h5_structure(h5_path: Path | str) -> None:
    """Inspects and logs datasets, shapes, dtypes, and attributes of the generated HDF5 archive."""
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
                    print(f"  - {attr_name}: {len(parsed)} registered items -> {parsed}")
                except Exception:
                    print(f"  - {attr_name}: {attr_val[:60]}...")
            else:
                print(f"  - {attr_name}: {attr_val}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    import tempfile
    from src.data_pipeline.patch_extractor import PatchExtractor

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "test_raw_shard.h5"
        writer = HDF5Writer(output_path=test_file, patch_size=PATCH_SIZE_PX)
        n_ch = len(PatchExtractor.CANONICAL_2D_CHANNELS)

        mock_patch = {
            "patch_id": "SCENE_RAW_p000",
            "X_2d": np.random.randn(n_ch, PATCH_SIZE_PX, PATCH_SIZE_PX).astype(np.float32),
            "X_1d": np.random.randn(35).astype(np.float32),
            "channel_validity": np.ones(n_ch, dtype=np.float32),
            "scl": np.full((PATCH_SIZE_PX, PATCH_SIZE_PX), 4, dtype=np.uint8),
            "is_fire": 1,
            "metadata": {
                "scene_id": "SCENE_RAW",
                "center_row": 64.0,
                "center_col": 64.0,
                "is_fire": 1,
            },
            "channel_names_2d": PatchExtractor.CANONICAL_2D_CHANNELS,
            "context_names_1d": [f"raw_weather_{i}" for i in range(35)],
        }

        written = writer.write_patches_batch([mock_patch])
        print(f"Wrote {written} raw full-scene patch(es) to temporary HDF5 shard.")
        inspect_h5_structure(test_file)