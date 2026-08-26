import os
import threading
import time
from typing import Any, Dict, List, Optional
import h5py
import numpy as np


class HDF5Writer:
    """writes patches where every raster band, weather feature, and target scale is an independent dataset."""

    def __init__(
        self,
        output_filepath: str,
        patch_size: int = 256,
        compression: str = "lzf",
    ):
        self.output_filepath = output_filepath
        self.patch_size = patch_size
        self.compression = compression
        self._lock = threading.Lock()
        self._initialized = False

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    def _ensure_channel_datasets(self, sample_patch: Dict[str, Any]):
        """dynamically creates separate datasets for each 2d channel, 1d metric, and target scale."""
        if self._initialized or os.path.exists(self.output_filepath):
            self._initialized = True
            return

        with h5py.File(self.output_filepath, "w") as h5:
            str_dtype = h5py.string_dtype(encoding="utf-8")
            h5.attrs["patch_size"] = self.patch_size

            # 1. 2d channels group
            g_2d = h5.create_group("channels_2d")
            for ch_name in sample_patch["channel_names_2d"]:
                dtype = np.uint8 if "MASK" in ch_name else np.float32
                g_2d.create_dataset(
                    ch_name,
                    shape=(0, self.patch_size, self.patch_size),
                    maxshape=(None, self.patch_size, self.patch_size),
                    dtype=dtype,
                    chunks=(1, self.patch_size, self.patch_size),
                    compression=self.compression,
                )

            # loss mask dataset
            g_2d.create_dataset(
                "loss_mask",
                shape=(0, self.patch_size, self.patch_size),
                maxshape=(None, self.patch_size, self.patch_size),
                dtype=np.uint8,
                chunks=(1, self.patch_size, self.patch_size),
                compression=self.compression,
            )

            # 2. 1d context features group
            g_1d = h5.create_group("context_1d")
            for feat_name in sample_patch["context_names_1d"]:
                g_1d.create_dataset(
                    feat_name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.float32,
                    chunks=(256,),
                )

            # 3. targets group (4 scales)
            g_tgt = h5.create_group("target")
            target_scales = ["scale_250m", "scale_1000m", "scale_3000m", "scale_4000m"]
            for scale_name in target_scales:
                g_tgt.create_dataset(
                    scale_name,
                    shape=(0, self.patch_size, self.patch_size),
                    maxshape=(None, self.patch_size, self.patch_size),
                    dtype=np.float32,
                    chunks=(1, self.patch_size, self.patch_size),
                    compression=self.compression,
                )

            # 4. metadata group
            g_meta = h5.create_group("metadata")
            g_meta.create_dataset("patch_id", shape=(0,), maxshape=(None,), dtype=str_dtype, chunks=(256,))
            g_meta.create_dataset("scene_id", shape=(0,), maxshape=(None,), dtype=str_dtype, chunks=(256,))
            g_meta.create_dataset("is_fire", shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=(256,))

        self._initialized = True

    def write_patches_batch(self, patches: List[Dict[str, Any]]) -> int:
        """atomically writes a batch of patches into separate channel datasets."""
        if not patches:
            return 0

        self._ensure_channel_datasets(patches[0])
        batch_size = len(patches)
        target_scales = ["scale_250m", "scale_1000m", "scale_3000m", "scale_4000m"]

        with self._lock:
            with h5py.File(self.output_filepath, "a") as h5:
                # determine current dataset size from one channel
                first_ch = patches[0]["channel_names_2d"][0]
                curr_len = h5["channels_2d"][first_ch].shape[0]
                new_len = curr_len + batch_size

                # 1. write each 2d channel
                for idx, ch_name in enumerate(patches[0]["channel_names_2d"]):
                    dtype = np.uint8 if "MASK" in ch_name else np.float32
                    ch_stack = np.stack([p["X_2d"][idx] for p in patches], axis=0).astype(dtype)
                    ds = h5["channels_2d"][ch_name]
                    ds.resize((new_len, self.patch_size, self.patch_size))
                    ds[curr_len:new_len] = ch_stack

                # write loss mask
                lmask_stack = np.stack([p["loss_mask"] for p in patches], axis=0).astype(np.uint8)
                ds_lmask = h5["channels_2d"]["loss_mask"]
                ds_lmask.resize((new_len, self.patch_size, self.patch_size))
                ds_lmask[curr_len:new_len] = lmask_stack

                # 2. write each 1d context feature
                for idx, feat_name in enumerate(patches[0]["context_names_1d"]):
                    feat_vec = np.array([p["X_1d"][idx] for p in patches], dtype=np.float32)
                    ds = h5["context_1d"][feat_name]
                    ds.resize((new_len,))
                    ds[curr_len:new_len] = feat_vec

                # 3. write each target scale
                for s_idx, scale_name in enumerate(target_scales):
                    tgt_stack = np.stack([p["Y"][s_idx] for p in patches], axis=0).astype(np.float32)
                    ds = h5["target"][scale_name]
                    ds.resize((new_len, self.patch_size, self.patch_size))
                    ds[curr_len:new_len] = tgt_stack

                # 4. write metadata
                ds_patch_id = h5["metadata"]["patch_id"]
                ds_scene_id = h5["metadata"]["scene_id"]
                ds_is_fire = h5["metadata"]["is_fire"]

                ds_patch_id.resize((new_len,))
                ds_scene_id.resize((new_len,))
                ds_is_fire.resize((new_len,))

                ds_patch_id[curr_len:new_len] = [p["patch_id"] for p in patches]
                ds_scene_id[curr_len:new_len] = [p["metadata"]["scene_id"] for p in patches]
                ds_is_fire[curr_len:new_len] = np.array([int(p["metadata"]["is_fire"]) for p in patches], dtype=np.uint8)

        return batch_size


def inspect_h5_structure(filepath: str):
    """prints a detailed structural overview of the generated hdf5 file."""
    print("\n" + "=" * 75)
    print(f" HDF5 DATASET STRUCTURE INSPECTION: {filepath}")
    print(f" File Size: {os.path.getsize(filepath) / (1024 * 1024):.2f} MB")
    print("=" * 75)

    with h5py.File(filepath, "r") as h5:
        total_samples = 0
        for grp_name in ["channels_2d", "target", "context_1d", "metadata"]:
            if grp_name in h5:
                grp = h5[grp_name]
                print(f"\n[Group: /{grp_name}] ({len(grp.keys())} datasets)")
                print("-" * 75)
                for key in sorted(grp.keys()):
                    ds = grp[key]
                    total_samples = ds.shape[0]
                    chunks_str = f"chunk={ds.chunks}" if ds.chunks else "contiguous"
                    comp_str = f"comp={ds.compression}" if ds.compression else "uncompressed"
                    print(f"  |-- {key:<24} shape: {str(ds.shape):<18} dtype: {str(ds.dtype):<10} ({chunks_str}, {comp_str})")

        print("\n" + "=" * 75)
        print(f" Total Stored Patches: {total_samples}")
        print("=" * 75 + "\n")


if __name__ == "__main__":
    from src.data_pipeline.patch_extractor import PatchExtractor
    from src.data_pipeline.single_scene_collector import SingleSceneCollector

    test_h5_path = "data/test_patches_dataset.h5"

    # remove old test file to start fresh
    if os.path.exists(test_h5_path):
        os.remove(test_h5_path)

    print("[1/3] Collecting full scene...")
    collector = SingleSceneCollector()
    extractor = PatchExtractor(patch_size=256, stride=256, max_invalid_ratio=0.20)
    writer = HDF5Writer(output_filepath=test_h5_path, patch_size=256, compression="lzf")

    test_lat, test_lon = 56.7264, -111.3803
    test_date = "2023-06-15"
    sample = collector.collect_sample(lat=test_lat, lon=test_lon, target_date=test_date, is_fire=1)

    if sample:
        print("[2/3] Extracting and writing patches to HDF5...")
        scene_id = f"SCENE_{test_lat:.2f}_{test_lon:.2f}_{test_date.replace('-', '')}"

        # collect all valid patches from generator
        patches_batch = list(extractor.extract_patches(sample, scene_id=scene_id))
        
        # write batch atomically
        written_count = writer.write_patches_batch(patches_batch)
        print(f"Successfully saved {written_count} patches into {test_h5_path}")

        # verify and inspect the written file
        inspect_h5_structure(test_h5_path)
    else:
        print("[Error] Failed to collect master scene sample.")