"""Model registry and inference engine for PreemptFire.

Supports the following models:
- tab_clim_full: CatBoost on weather, 30-day history, and climatology (69 features)
- tab_ssl_full: SimCLR ResNet-based embeddings + CatBoost (101 features)
- tab_full: handcrafted 2D spatial/spectral stats + CatBoost (262 features)
- tab_hist: CatBoost on weather and 30-day history (60 features)
- tab_1d: CatBoost on weather and population (39 features)
- nn_ts: PyTorch GRU on 30-day sequence + static features
- coords: k-NN on geographic coordinates
- persist: 7-day fire persistence baseline
- clim: historical day-of-year fire climatology baseline
- stack_no_img: ensemble of non-image models
- stack_all: ensemble of 2D satellite and 1D models
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from config_and_features import (
    CONFIG,
    RADII,
    build_33ch_array,
    compute_2d_tab_features,
    fit_nb_alpha,
)

LEVELS = ["Low", "Medium", "High"]
COLORS = ["#fee08b", "#f46d43", "#a50026"]

MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "tab_clim_full": {
        "label": "CatBoost: weather, 30d history, climatology (tab_clim_full, 69 features)",
        "requires_2d": False,
        "cbm_prefix": "tab_clim_full",
        "oof_col": "p_tab_clim_full",
        "n_features": 69,
    },
    "tab_ssl_full": {
        "label": "CatBoost + SimCLR 2D satellite (tab_ssl_full, 101 features)",
        "requires_2d": True,
        "cbm_prefix": "tab_ssl_full",
        "oof_col": "p_tab_ssl_full",
        "n_features": 101,
    },
    "tab_full": {
        "label": "CatBoost + 2D scene statistics (tab_full, 262 features)",
        "requires_2d": True,
        "cbm_prefix": "tab_full",
        "oof_col": "p_tab_full",
        "n_features": 262,
    },
    "tab_hist": {
        "label": "CatBoost: weather and 30d history (tab_hist, 60 features)",
        "requires_2d": False,
        "cbm_prefix": "tab_hist",
        "oof_col": "p_tab_hist",
        "n_features": 60,
    },
    "tab_1d": {
        "label": "CatBoost: weather and population only (tab_1d, 39 features)",
        "requires_2d": False,
        "cbm_prefix": "tab_1d",
        "oof_col": "p_tab_1d",
        "n_features": 39,
    },
    "nn_ts": {
        "label": "PyTorch GRU time series + static features (nn_ts)",
        "requires_2d": False,
        "oof_col": "p_nn_ts",
    },
    "coords": {
        "label": "k-NN coordinates baseline (coords)",
        "requires_2d": False,
        "oof_col": "p_coords",
    },
    "persist": {
        "label": "7-day fire persistence baseline (persist)",
        "requires_2d": False,
        "oof_col": "p_persist",
    },
    "clim": {
        "label": "Historical day-of-year climatology baseline (clim)",
        "requires_2d": False,
        "oof_col": "p_clim",
    },
    "stack_no_img": {
        "label": "Ensemble: non-image models (stack_no_img)",
        "requires_2d": False,
        "oof_col": "p_stack_no_img",
    },
    "stack_all": {
        "label": "Ensemble: multimodal 1D + 2D satellite (stack_all)",
        "requires_2d": True,
        "oof_col": "p_stack_with_img",
    },
}


def _build_torch_modules():
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, stride: int = 1):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.SiLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c_out),
            )
            self.skip = (
                nn.Identity()
                if (stride == 1 and c_in == c_out)
                else nn.Sequential(
                    nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                    nn.BatchNorm2d(c_out),
                )
            )
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.conv(x) + self.skip(x))

    class ConvBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def make_encoder(base: int = 24, in_ch: int = 33) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU(inplace=True),
            ResBlock(base, base * 2, 2),
            ResBlock(base * 2, base * 4, 2),
            ResBlock(base * 4, base * 8, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def make_encoder_v5(base: int = 24, in_ch: int = 33) -> nn.Module:
        return nn.Sequential(
            ConvBlock(in_ch, base),
            nn.MaxPool2d(2),
            ConvBlock(base, base * 2),
            nn.MaxPool2d(2),
            ConvBlock(base * 2, base * 4),
            nn.MaxPool2d(2),
            ConvBlock(base * 4, base * 8),
            nn.MaxPool2d(2),
            ConvBlock(base * 8, base * 8),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    class Net(nn.Module):
        is_v5 = False

        def __init__(
            self,
            img: bool = True,
            seq: bool = True,
            n_cnt: int = 4,
            d1: int = 47,
            t_feat: int = 5,
            base: int = 24,
            in_ch: int = 33,
        ):
            super().__init__()
            self.use_img, self.use_seq, self.n_cnt = img, seq, n_cnt
            dim = 0
            if img:
                self.enc = make_encoder(base, in_ch=in_ch)
                dim += base * 8
            if seq:
                self.gru = nn.GRU(t_feat, 48, batch_first=True, bidirectional=True)
                self.tab = nn.Sequential(
                    nn.Linear(d1, 64),
                    nn.LayerNorm(64),
                    nn.SiLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 48),
                    nn.SiLU(),
                )
                dim += 96 + 48
            self.cls_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(dim, 64),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )
            if n_cnt:
                self.cnt_head = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(dim, 64),
                    nn.SiLU(),
                    nn.Linear(64, n_cnt),
                )

        def forward(self, x2, xseq, xst):
            parts = []
            if self.use_img:
                parts.append(self.enc(x2))
            if self.use_seq:
                _, h = self.gru(xseq)
                parts += [torch.cat([h[0], h[1]], dim=1), self.tab(xst)]
            z = torch.cat(parts, dim=1)
            logit = self.cls_head(z).squeeze(1)
            cnt = self.cnt_head(z) if self.n_cnt else None
            return logit, cnt

    class NetV5(nn.Module):
        is_v5 = True

        def __init__(
            self,
            img: bool = False,
            seq: bool = True,
            n_cnt: int = 4,
            d1: int = 47,
            t_feat: int = 5,
            base: int = 24,
            in_ch: int = 33,
        ):
            super().__init__()
            self.img, self.seq, self.n_cnt = img, seq, n_cnt
            if img:
                self.enc = make_encoder_v5(base, in_ch)
            if seq:
                self.gru = nn.GRU(t_feat, 48, batch_first=True)
                self.seq_ln = nn.LayerNorm(48)
            self.mlp = nn.Sequential(
                nn.Linear(d1, 64),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(64, 64),
                nn.GELU(),
            )
            n = 64 + (base * 8 if img else 0) + (48 if seq else 0)
            self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(n, 1 + n_cnt))
            self.log_alpha = nn.Parameter(torch.zeros(n_cnt))

        def forward(self, x2, xseq, xst):
            z = [self.mlp(xst)]
            if self.img:
                z.append(self.enc(x2))
            if self.seq:
                _, h = self.gru(xseq)
                z.append(self.seq_ln(h[-1]))
            out = self.head(torch.cat(z, 1))
            return out[:, 0], out[:, 1:]

    return torch, nn, make_encoder, make_encoder_v5, Net, NetV5


class ModelRegistry:
    def __init__(
        self,
        model_dir: Optional[str | Path] = None,
        oof_path: Optional[str | Path] = None,
        count_metrics_path: Optional[str | Path] = None,
    ):
        self.model_dir = Path(model_dir or CONFIG.get("MODEL_DIR", "models"))
        self.oof_path = self._find_file(oof_path, ["oof_predictions.csv", "data/oof_predictions.csv", "models/oof_predictions.csv"])
        self.count_metrics_path = self._find_file(count_metrics_path, ["count_metrics.csv", "data/count_metrics.csv", "models/count_metrics.csv"])

        self._cb_cache: Dict[str, Tuple[Any, Dict[int, Any]]] = {}
        self._simclr_cache: Optional[Tuple[Any, Any, Any, Any]] = None
        self._nn_ts_cache: Optional[Tuple[Any, Any, Dict[str, Any], Any]] = None
        self._coords_cache: Optional[Tuple[Any, Any]] = None

        self.oof_df: Optional[pd.DataFrame] = None
        if self.oof_path and self.oof_path.exists():
            try:
                self.oof_df = pd.read_csv(self.oof_path)
            except Exception:
                self.oof_df = None

        self.alphas: Dict[str, Dict[int, float]] = self._load_alphas()

    def _find_file(self, explicit: Optional[str | Path], candidates: List[str]) -> Optional[Path]:
        if explicit and Path(explicit).exists():
            return Path(explicit)
        for c in candidates:
            for root in (Path("."), self.model_dir, self.model_dir.parent):
                p = root / c
                if p.exists():
                    return p
        return None

    def _resolve_model_file(self, filename: str) -> Optional[Path]:
        for d in (self.model_dir, Path("."), Path("models"), Path("preemptfire_full_output/models")):
            p = d / filename
            if p.exists():
                return p
        return None

    def _load_alphas(self) -> Dict[str, Dict[int, float]]:
        default_alpha = {5: 8.5, 10: 6.2, 25: 4.1, 50: 3.0}
        alphas: Dict[str, Dict[int, float]] = {m: dict(default_alpha) for m in MODEL_CATALOG}

        meta_p = self._resolve_model_file("meta.json")
        if meta_p and meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text())
                if "alpha" in meta:
                    for k, v in meta["alpha"].items():
                        r = int(str(k).replace("km", ""))
                        alphas["tab_clim_full"][r] = float(v)
            except Exception:
                pass

        if self.oof_df is not None:
            for m in MODEL_CATALOG:
                for R in RADII:
                    y_col = f"y_count_{R}km"
                    mu_col = f"mu_{m}_{R}km"
                    if y_col in self.oof_df.columns and mu_col in self.oof_df.columns:
                        alphas[m][R] = fit_nb_alpha(
                            self.oof_df[y_col].values,
                            self.oof_df[mu_col].values,
                        )
        return alphas

    def status_report(self) -> Dict[str, Dict[str, Any]]:
        report: Dict[str, Dict[str, Any]] = {}
        for m, info in MODEL_CATALOG.items():
            if m in ("persist", "clim"):
                report[m] = {"available": True, "requires_2d": False, "missing_files": [], "label": info["label"]}
            elif "cbm_prefix" in info:
                pref = info["cbm_prefix"]
                req = [f"{pref}_classifier.cbm"] + [f"{pref}_regressor_{R}km.cbm" for R in RADII]
                if m == "tab_ssl_full":
                    req += ["simclr_encoder_weights.pt", "simclr_pca_32.pkl"]
                missing = [f for f in req if self._resolve_model_file(f) is None]
                if missing and m == "tab_clim_full" and self._resolve_model_file("classifier.cbm") is not None:
                    missing = []
                report[m] = {
                    "available": len(missing) == 0,
                    "requires_2d": info["requires_2d"],
                    "missing_files": missing,
                    "label": info["label"],
                }
            elif m == "nn_ts":
                missing = [f for f in ["nn_ts_final_model.pt"] if self._resolve_model_file(f) is None]
                report[m] = {"available": len(missing) == 0, "requires_2d": False, "missing_files": missing, "label": info["label"]}
            elif m == "coords":
                missing = [f for f in ["coords_classifier_knn.pkl", "coords_regressor_knn.pkl"] if self._resolve_model_file(f) is None]
                report[m] = {"available": len(missing) == 0, "requires_2d": False, "missing_files": missing, "label": info["label"]}
            elif m == "stack_no_img":
                report[m] = {"available": True, "requires_2d": False, "missing_files": [], "label": info["label"]}
            elif m == "stack_all":
                ssl_ok = report.get("tab_ssl_full", {}).get("available", False) or report.get("tab_full", {}).get("available", False)
                report[m] = {"available": ssl_ok, "requires_2d": True, "missing_files": [], "label": info["label"]}
        return report

    def available_models(self, include_2d: bool = True) -> List[str]:
        rep = self.status_report()
        return [
            m for m, st in rep.items()
            if st["available"] and (include_2d or not st["requires_2d"])
        ]

    def _load_catboost(self, prefix: str):
        if prefix in self._cb_cache:
            return self._cb_cache[prefix]
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls_path = self._resolve_model_file(f"{prefix}_classifier.cbm")
        if cls_path is None and prefix == "tab_clim_full":
            cls_path = self._resolve_model_file("classifier.cbm")
        if cls_path is None:
            raise FileNotFoundError(f"Missing CatBoost classifier `{prefix}_classifier.cbm` in {self.model_dir}")

        clf = CatBoostClassifier()
        clf.load_model(str(cls_path))

        regs = {}
        for R in RADII:
            rp = self._resolve_model_file(f"{prefix}_regressor_{R}km.cbm")
            if rp is None and prefix == "tab_clim_full":
                rp = self._resolve_model_file(f"regressor_{R}km.cbm")
            if rp is not None:
                reg = CatBoostRegressor()
                reg.load_model(str(rp))
                regs[R] = reg
        self._cb_cache[prefix] = (clf, regs)
        return clf, regs

    def _load_simclr_and_pca(self):
        if self._simclr_cache is not None:
            return self._simclr_cache
        torch, _, make_encoder, make_encoder_v5, _, _ = _build_torch_modules()
        enc_path = self._resolve_model_file("simclr_encoder_weights.pt")
        pca_path = self._resolve_model_file("simclr_pca_32.pkl")
        if enc_path is None or pca_path is None:
            raise FileNotFoundError("Missing `simclr_encoder_weights.pt` or `simclr_pca_32.pkl`.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(str(enc_path), map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = {
                (k[4:] if k.startswith("enc.") else k): v
                for k, v in state["model"].items()
                if k.startswith("enc.") or not k.startswith("proj.")
            }

        is_v5 = isinstance(state, dict) and any(".net." in k for k in state.keys())
        enc = (make_encoder_v5(base=24, in_ch=33) if is_v5 else make_encoder(base=24, in_ch=33)).to(device)
        enc.load_state_dict(state, strict=False)
        enc.eval()

        with open(pca_path, "rb") as f:
            pca_obj = pickle.load(f)

        self._simclr_cache = (torch, enc, pca_obj, device)
        return self._simclr_cache

    def _load_nn_ts(self):
        if self._nn_ts_cache is not None:
            return self._nn_ts_cache
        torch, _, _, _, Net, NetV5 = _build_torch_modules()
        pt_path = self._resolve_model_file("nn_ts_final_model.pt")
        if pt_path is None:
            raise FileNotFoundError("Missing `nn_ts_final_model.pt`.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(pt_path), map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            meta = ckpt
        elif isinstance(ckpt, dict) and any(
            k.startswith("gru.") or k.startswith("cls_head.") or k.startswith("mlp.") or k.startswith("head.")
            for k in ckpt.keys()
        ):
            state_dict = ckpt
            meta = {}
        else:
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt.state_dict()
            meta = ckpt if isinstance(ckpt, dict) else {}

        if "mlp.0.weight" in state_dict:
            d1 = int(state_dict["mlp.0.weight"].shape[1])
            t_feat = int(state_dict["gru.weight_ih_l0"].shape[1]) if "gru.weight_ih_l0" in state_dict else 5
            net = NetV5(img=False, seq=True, n_cnt=4, d1=d1, t_feat=t_feat).to(device)
        else:
            d1 = int(state_dict["tab.0.weight"].shape[1]) if "tab.0.weight" in state_dict else 47
            t_feat = int(state_dict["gru.weight_ih_l0"].shape[1]) if "gru.weight_ih_l0" in state_dict else 5
            net = Net(img=False, seq=True, n_cnt=4, d1=d1, t_feat=t_feat).to(device)
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        self._nn_ts_cache = (torch, net, meta, device)
        return self._nn_ts_cache

    def _load_coords_knn(self):
        if self._coords_cache is not None:
            return self._coords_cache
        cp = self._resolve_model_file("coords_classifier_knn.pkl")
        rp = self._resolve_model_file("coords_regressor_knn.pkl")
        if cp is None or rp is None:
            raise FileNotFoundError("Missing `coords_classifier_knn.pkl` or `coords_regressor_knn.pkl`.")
        with open(cp, "rb") as f:
            knn_cls = pickle.load(f)
        with open(rp, "rb") as f:
            knn_reg = pickle.load(f)
        self._coords_cache = (knn_cls, knn_reg)
        return self._coords_cache

    def compute_2d_embeddings_and_stats(
        self,
        scenes_2d: List[Optional[Dict[str, np.ndarray]]],
        batch_size: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(scenes_2d)
        FT = np.zeros((n, 202), dtype=np.float32)
        EMB32 = np.zeros((n, 32), dtype=np.float32)
        has_2d = np.zeros(n, dtype=bool)

        valid_indices = [i for i, s in enumerate(scenes_2d) if s is not None and "X_2d" in s]
        if not valid_indices:
            return FT, EMB32, has_2d

        tensors_33 = []
        for idx in valid_indices:
            s = scenes_2d[idx]
            x2d = np.asarray(s["X_2d"], dtype=np.float32)
            scl = np.asarray(s.get("scl", np.full(x2d.shape[1:], 4, dtype=np.uint8)), dtype=np.uint8)
            FT[idx] = compute_2d_tab_features(x2d, scl)
            tensors_33.append(build_33ch_array(x2d, scl, size=128))
            has_2d[idx] = True

        try:
            torch, enc, pca_obj, device = self._load_simclr_and_pca()
            arr_all = np.stack(tensors_33, axis=0)
            mean = arr_all.mean(axis=(0, 2, 3), keepdims=True)
            std = arr_all.std(axis=(0, 2, 3), keepdims=True) + 1e-6
            arr_norm = np.clip((arr_all - mean) / std, -6.0, 6.0).astype(np.float32)

            emb192_list = []
            with torch.no_grad():
                for s_i in range(0, len(arr_norm), batch_size):
                    xb = torch.from_numpy(arr_norm[s_i : s_i + batch_size]).to(device)
                    eb = enc(xb).cpu().numpy()
                    emb192_list.append(eb)
            emb192 = np.vstack(emb192_list)

            if isinstance(pca_obj, dict):
                if "scaler" in pca_obj and "pca" in pca_obj:
                    emb32_valid = pca_obj["pca"].transform(pca_obj["scaler"].transform(emb192))
                elif "pca" in pca_obj:
                    emb32_valid = pca_obj["pca"].transform(emb192)
                else:
                    emb32_valid = emb192[:, :32]
            elif hasattr(pca_obj, "transform"):
                emb32_valid = pca_obj.transform(emb192)
            elif isinstance(pca_obj, (tuple, list)) and len(pca_obj) == 2:
                emb32_valid = pca_obj[1].transform(pca_obj[0].transform(emb192))
            else:
                emb32_valid = emb192[:, :32]

            for k, idx in enumerate(valid_indices):
                EMB32[idx] = emb32_valid[k].astype(np.float32)
        except Exception as e:
            print(f"[ModelRegistry] SimCLR embedding note: {e}")

        return FT, EMB32, has_2d

    def predict_model(
        self,
        model_name: str,
        TAB_1D: np.ndarray,
        HIST_TAB: np.ndarray,
        CLIM_FEATS: np.ndarray,
        SEQ: Optional[np.ndarray] = None,
        STATIC_1D: Optional[np.ndarray] = None,
        pts_lat_lon: Optional[np.ndarray] = None,
        FT_2D: Optional[np.ndarray] = None,
        EMB32_2D: Optional[np.ndarray] = None,
        has_2d_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        n = len(TAB_1D)
        TAB_1D = np.nan_to_num(np.asarray(TAB_1D, dtype=np.float32), nan=0.0)
        HIST_TAB = np.nan_to_num(np.asarray(HIST_TAB, dtype=np.float32), nan=0.0)
        CLIM_FEATS = np.nan_to_num(np.asarray(CLIM_FEATS, dtype=np.float32), nan=0.0)

        mu: Dict[int, np.ndarray] = {R: np.zeros(n, dtype=np.float32) for R in RADII}

        if model_name == "persist":
            h7_50 = HIST_TAB[:, 2 * 4 + 3]
            p = np.clip(h7_50 / (h7_50 + 1.0), 1e-3, 0.999).astype(np.float32)
            for ri, R in enumerate(RADII):
                mu[R] = np.clip(HIST_TAB[:, 2 * 4 + ri] / 7.0, 1e-3, None).astype(np.float32)

        elif model_name == "clim":
            clim_rate = np.clip(np.exp(CLIM_FEATS[:, :4]) - 1e-3, 0.0, None)
            clim_p = CLIM_FEATS[:, 4:8]
            p = np.clip(clim_p[:, 3] + 1e-4 * clim_rate[:, 3], 1e-4, 0.999).astype(np.float32)
            for ri, R in enumerate(RADII):
                mu[R] = np.clip(clim_rate[:, ri], 1e-3, None).astype(np.float32)

        elif model_name in ("tab_1d", "tab_hist", "tab_clim_full", "tab_full", "tab_ssl_full"):
            if model_name == "tab_1d":
                X = TAB_1D
            elif model_name == "tab_hist":
                X = np.hstack([TAB_1D, HIST_TAB])
            elif model_name == "tab_clim_full":
                X = np.hstack([TAB_1D, HIST_TAB, CLIM_FEATS])
            elif model_name == "tab_full":
                ft = np.zeros((n, 202), dtype=np.float32) if FT_2D is None else np.nan_to_num(FT_2D, 0.0)
                X = np.hstack([TAB_1D, ft, HIST_TAB])
            elif model_name == "tab_ssl_full":
                emb = np.zeros((n, 32), dtype=np.float32) if EMB32_2D is None else np.nan_to_num(EMB32_2D, 0.0)
                X = np.hstack([TAB_1D, HIST_TAB, CLIM_FEATS, emb])

            clf, regs = self._load_catboost(model_name)
            p = clf.predict_proba(X)[:, 1].astype(np.float32)
            for R in RADII:
                if R in regs:
                    raw_pred = regs[R].predict(X)
                    loss_fn = ""
                    try:
                        loss_fn = str(regs[R].get_params().get("loss_function", "")).lower()
                    except Exception:
                        pass
                    if "poisson" in loss_fn:
                        mu[R] = np.clip(raw_pred, 1e-4, 500.0).astype(np.float32)
                    else:
                        mu[R] = np.expm1(np.clip(raw_pred, 0.0, 12.0)).astype(np.float32)

        elif model_name == "nn_ts":
            torch, net, meta, device = self._load_nn_ts()
            if SEQ is None or STATIC_1D is None:
                raise ValueError("nn_ts requires SEQ and STATIC_1D arrays.")
            seq_arr = np.nan_to_num(SEQ, 0.0).astype(np.float32)
            st_arr = np.nan_to_num(STATIC_1D, 0.0).astype(np.float32)
            s_mean = meta.get("seq_mean", seq_arr.mean(axis=(0, 1), keepdims=True))
            s_std = meta.get("seq_std", seq_arr.std(axis=(0, 1), keepdims=True) + 1e-6)
            t_mean = meta.get("st_mean", st_arr.mean(axis=0, keepdims=True))
            t_std = meta.get("st_std", st_arr.std(axis=0, keepdims=True) + 1e-6)

            seq_n = np.clip((seq_arr - s_mean) / (s_std + 1e-6), -6.0, 6.0).astype(np.float32)
            st_n = np.clip((st_arr - t_mean) / (t_std + 1e-6), -6.0, 6.0).astype(np.float32)

            with torch.no_grad():
                xseq_t = torch.from_numpy(seq_n).to(device)
                xst_t = torch.from_numpy(st_n).to(device)
                logit, cnt = net(None, xseq_t, xst_t)
                p = torch.sigmoid(logit).cpu().numpy().astype(np.float32)
                if getattr(net, "is_v5", False):
                    cnt_np = np.exp(np.clip(cnt.cpu().numpy(), -8.0, 8.0)).astype(np.float32)
                else:
                    cnt_np = np.expm1(np.clip(cnt.cpu().numpy(), 0.0, 12.0)).astype(np.float32)
                for ri, R in enumerate(RADII):
                    mu[R] = cnt_np[:, ri]

        elif model_name == "coords":
            knn_cls, knn_reg = self._load_coords_knn()
            if pts_lat_lon is None:
                raise ValueError("coords requires pts_lat_lon.")
            cxy = np.column_stack([pts_lat_lon[:, 0], pts_lat_lon[:, 1] * np.cos(np.radians(40.0))]).astype(np.float32)
            p = knn_cls.predict_proba(cxy)[:, 1].astype(np.float32)
            if isinstance(knn_reg, dict):
                for R in RADII:
                    rk = R if R in knn_reg else f"{R}km"
                    if rk in knn_reg:
                        mu[R] = np.clip(knn_reg[rk].predict(cxy), 1e-3, 500.0).astype(np.float32)
            elif isinstance(knn_reg, list):
                for ri, R in enumerate(RADII):
                    mu[R] = np.clip(knn_reg[ri].predict(cxy), 1e-3, 500.0).astype(np.float32)
            elif hasattr(knn_reg, "predict"):
                pred_r = np.clip(knn_reg.predict(cxy), 1e-3, 500.0).astype(np.float32)
                if pred_r.ndim == 2 and pred_r.shape[1] == 4:
                    for ri, R in enumerate(RADII):
                        mu[R] = pred_r[:, ri]

        elif model_name in ("stack_no_img", "stack_all"):
            sub_models = ["tab_clim_full", "tab_hist", "persist", "clim"]
            if self._resolve_model_file("nn_ts_final_model.pt") is not None and SEQ is not None:
                sub_models.append("nn_ts")
            if model_name == "stack_all":
                for m2d in ("tab_ssl_full", "tab_full"):
                    if self.status_report().get(m2d, {}).get("available", False):
                        sub_models.append(m2d)
            sub_outs = []
            for sm in sub_models:
                try:
                    sub_outs.append(
                        self.predict_model(
                            sm, TAB_1D, HIST_TAB, CLIM_FEATS, SEQ, STATIC_1D, pts_lat_lon, FT_2D, EMB32_2D, has_2d_mask
                        )
                    )
                except Exception:
                    continue
            if not sub_outs:
                raise RuntimeError(f"No constituent models available for {model_name}")
            p = np.mean([o["p"] for o in sub_outs], axis=0).astype(np.float32)
            for R in RADII:
                mu[R] = np.mean([o["mu"][R] for o in sub_outs], axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unknown model `{model_name}`")

        upper90: Dict[int, np.ndarray] = {}
        model_alphas = self.alphas.get(model_name, {5: 8.5, 10: 6.2, 25: 4.1, 50: 3.0})
        for R in RADII:
            a = max(float(model_alphas.get(R, 4.0)), 1e-4)
            r_nb = 1.0 / a
            prob_nb = r_nb / (r_nb + np.maximum(mu[R], 1e-6))
            upper90[R] = nbinom.ppf(0.90, r_nb, prob_nb).astype(int)

        thresholds, cal_table = self.get_thresholds(model_name, fallback_p=p)
        level_idx = np.digitize(p, thresholds)
        levels = np.array([LEVELS[i] for i in level_idx])

        return {
            "model": model_name,
            "label": MODEL_CATALOG.get(model_name, {}).get("label", model_name),
            "p": p,
            "mu": mu,
            "upper90": upper90,
            "level_idx": level_idx,
            "levels": levels,
            "thresholds": thresholds,
            "calibration_table": cal_table,
        }

    def get_thresholds(
        self,
        model_name: str,
        shares: Tuple[float, float] = (0.60, 0.30),
        fallback_p: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        oof_col = MODEL_CATALOG.get(model_name, {}).get("oof_col", f"p_{model_name}")
        if self.oof_df is not None:
            col = oof_col if oof_col in self.oof_df.columns else (
                "p_tab_clim_full" if "p_tab_clim_full" in self.oof_df.columns else None
            )
            if col is not None:
                p_oof = self.oof_df[col].dropna().values
                t = np.quantile(p_oof, [shares[0], shares[0] + shares[1]])
                lev = np.digitize(p_oof, t)
                y_oof = self.oof_df.loc[self.oof_df[col].notna(), "y"].values if "y" in self.oof_df.columns else np.zeros_like(p_oof)
                c50_oof = (
                    self.oof_df.loc[self.oof_df[col].notna(), "y_count_50km"].values
                    if "y_count_50km" in self.oof_df.columns
                    else np.zeros_like(p_oof)
                )
                tab = pd.DataFrame(
                    {
                        "level": [LEVELS[i] for i in range(3)],
                        "share_of_scenes": [float((lev == i).mean()) for i in range(3)],
                        "fire_within_50km": [float(y_oof[lev == i].mean()) if (lev == i).any() else 0.0 for i in range(3)],
                        "mean_detections_50km": [float(c50_oof[lev == i].mean()) if (lev == i).any() else 0.0 for i in range(3)],
                    }
                )
                return t, tab

        if fallback_p is not None and len(fallback_p) > 0:
            t = np.quantile(fallback_p, [shares[0], shares[0] + shares[1]])
        else:
            t = np.array([0.35, 0.65], dtype=np.float32)
        return t, None
