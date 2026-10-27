"""
Frozen-state re-implementation of the training feature engineering (steps 4-6).

The training scripts fitted scalers / encoders / prototypes / ridge models inside
`main()` and never saved them, so inference cannot call them.  This module splits
every step into

    fit_state(...)   -> run ONCE from the training caches (export_state.py)
    transform(...)   -> run per batch of new plots, using only the frozen state

and `export_state.py` proves parity by regenerating the stored feature matrices
of the held-out plots from the state alone.

Variant differences handled here
  m1: all 5 locations seen; stats come from TRAIN plots of each location.
  m2: 4 dev locations from TRAIN plots; any location with <5 TRAIN plots
      (Crawfordsville / unseen sites) uses label-free stats of ALL its plots
      (the stored reference plots, or the incoming batch for a brand-new site).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

CANOPY_NAMES = [
    "ExG_Mean", "ExG_Std", "NGRDI_Mean", "NGRDI_Std", "VARI_Mean", "GLI_Mean",
    "Green_Canopy_Cover", "Yellow_Senescence_Frac", "PureCanopy_ExG_Mean",
    "PureCanopy_NGRDI_Mean", "Patch_Canopy_Gap_Std", "Patch_Canopy_Min",
]
IMG_KEYS = [f"tp{t}_{k}" for t in (1, 2, 3) for k in ("mean", "std", "canopy")]
PROTO_ALPHAS = [20.0, 80.0, 250.0, 600.0, 1500.0]
PHENO_COLS = ("daysToAnthesis", "GDDToAnthesis")
MIN_REF_PLOTS = 5


# ----------------------------------------------------------------------------- utils
def _clean(a) -> np.ndarray:
    return np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _l2n(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-6, None)


def exp_str(v) -> str:
    """Stable string for an experiment id (4232, 4232.0, '4232' -> '4232')."""
    try:
        f = float(v)
        if np.isfinite(f) and f == int(f):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def parse_hybrid_parents(geno_str: str) -> Tuple[str, str]:
    g = str(geno_str).strip().upper()
    if not g or g == "NAN":
        return "UNKNOWN", "UNKNOWN"
    parts = re.split(r"\s+[X\*\/]\s+|\s*[\*\/]\s*", g)
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return g, g


def clean_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the metadata columns exactly the way training read them."""
    out = df.copy()
    out["location"] = out["location"].astype(str).str.strip()
    out["plot_id"] = out["plot_id"].astype(str).str.strip()
    out["experiment"] = out["experiment"].map(exp_str) if "experiment" in out else "UNK"
    for c in ("range", "row"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(np.float32) if c in out else 0.0
    out["genotype"] = (out["genotype"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
                       if "genotype" in out else "UNKNOWN")
    for c in ("totalStandCount", "plantingDate", *PHENO_COLS):
        if c not in out:
            out[c] = np.nan
    return out


# ----------------------------------------------------------------------------- meta
def compute_meta(df: pd.DataFrame, plot_areas: np.ndarray, stand_median: float, doy_median: float) -> np.ndarray:
    raw_stand = pd.to_numeric(df["totalStandCount"], errors="coerce")
    stand_count = raw_stand.fillna(stand_median).values.astype(np.float32)
    raw_doy = pd.to_datetime(df["plantingDate"], errors="coerce").dt.dayofyear
    planting_doy = raw_doy.fillna(doy_median).values.astype(np.float32)
    safe_area = np.clip(np.asarray(plot_areas, dtype=np.float32) / 10000.0, 1.0, None)
    stand_density = (stand_count / safe_area).astype(np.float32)
    log_stand = np.log1p(np.clip(stand_count, 0.0, None)).astype(np.float32)
    return np.column_stack([stand_count, planting_doy, stand_density, log_stand])


# ----------------------------------------------------------------------------- flight normalisation
def _fit_loc_norm(raw: Dict[str, np.ndarray], ref: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    for key in IMG_KEYS:
        mat = _clean(raw[key])
        mu = np.mean(mat[ref], axis=0, keepdims=True)
        sd = np.std(mat[ref], axis=0, keepdims=True)
        out[key] = (mu, np.where(sd < 1e-5, 1.0, sd))
    return out


def _apply_loc_norm(raw: Dict[str, np.ndarray], rows: np.ndarray, stats, out: Dict[str, np.ndarray]) -> None:
    for key in IMG_KEYS:
        mu, sd = stats[key]
        out[key][rows] = np.clip((_clean(raw[key])[rows] - mu) / sd, -6.0, 6.0)


def ref_mask_for(loc_mask: np.ndarray, train_mask: np.ndarray, variant: str) -> np.ndarray:
    ref = loc_mask & train_mask
    if variant == "m2" and ref.sum() < MIN_REF_PLOTS:
        ref = loc_mask
    return ref


# ----------------------------------------------------------------------------- categorical maps
def _fit_cat_map(c: np.ndarray, y: np.ndarray, l: np.ndarray, m: float, prior: float) -> Dict[str, float]:
    df = pd.DataFrame({"cat": c, "y": y, "loc": l})
    loc_cat = df.groupby(["cat", "loc"])["y"].mean().reset_index()
    stats = loc_cat.groupby("cat")["y"].agg(["count", "mean"])
    tot = df.groupby("cat")["y"].count()
    eff = 0.5 * stats["count"] + 0.5 * tot.reindex(stats.index)
    return ((eff * stats["mean"] + m * prior) / (eff + m)).to_dict()


def _enc(cats: np.ndarray, mp: Dict[str, float], prior: float) -> np.ndarray:
    return np.array([mp.get(c, prior) for c in cats], dtype=np.float32)


# ----------------------------------------------------------------------------- prototypes
def proto_blocks(src: Dict[str, np.ndarray]) -> List[Tuple[str, np.ndarray]]:
    return [
        ("TP1_Mean", src["tp1_mean"]),
        ("TP2_Mean", src["tp2_mean"]),
        ("TP3_Mean", src["tp3_mean"]),
        ("TP3_PatchStd", src["tp3_std"]),
        ("Delta_TP2_TP1", src["tp2_mean"] - src["tp1_mean"]),
        ("Delta_TP3_TP2", src["tp3_mean"] - src["tp2_mean"]),
        ("Delta_TP3_TP1", src["tp3_mean"] - src["tp1_mean"]),
        ("Joint_AllTP_Canopy", np.hstack([src["tp1_canopy"], src["tp2_canopy"], src["tp3_canopy"],
                                         src["tp3_canopy"] - src["tp1_canopy"]])),
    ]


def _loc_target_stats(y: np.ndarray, locs: np.ndarray, train: np.ndarray):
    """per-train-location z-scored target + top/bottom-quartile masks (train rows only)."""
    tl, yt = locs[train], y[train]
    ynorm = np.zeros_like(yt, dtype=np.float32)
    hi = np.zeros(len(yt), dtype=bool)
    lo = np.zeros(len(yt), dtype=bool)
    for loc in np.unique(tl):
        m = tl == loc
        yl = yt[m]
        mu, sd = float(np.mean(yl)), max(float(np.std(yl)), 1.0)
        ynorm[m] = (yl - mu) / sd
        hi[m] = yl >= np.percentile(yl, 75)
        lo[m] = yl <= np.percentile(yl, 25)
    return ynorm, hi, lo


def fit_protos(src: Dict[str, np.ndarray], y, locs, train) -> List[tuple]:
    ynorm, hi_m, lo_m = _loc_target_stats(y, locs, train)
    state = []
    for label, mat in proto_blocks(src):
        m = _clean(mat)
        tmv = np.mean(m[train], axis=0, keepdims=True)
        unit = _l2n(m - tmv)
        tr = unit[train]
        hi = _l2n(np.mean(tr[hi_m], axis=0, keepdims=True))[0]
        lo = _l2n(np.mean(tr[lo_m], axis=0, keepdims=True))[0]
        ridge = RidgeCV(alphas=PROTO_ALPHAS).fit(tr, ynorm)
        state.append((label, tmv, hi, lo, ridge))
    return state


def apply_protos(state: List[tuple], src: Dict[str, np.ndarray]):
    """-> (matrix, suffix-less block labels). Column order per block: SimHigh, SimLow, NetDiff, Ridge."""
    cols, labels = [], []
    for (label, tmv, hi, lo, ridge), (_, mat) in zip(state, proto_blocks(src)):
        unit = _l2n(_clean(mat) - tmv)
        sh, sl = unit @ hi, unit @ lo
        cols.append(np.column_stack([sh, sl, sh - sl, ridge.predict(unit).astype(np.float32)]))
        labels.append(label)
    return np.hstack(cols).astype(np.float32), labels


# ----------------------------------------------------------------------------- spatial reference
def _spatial_ref(rows_mask, df, v3, v2, variant):
    r, w = df["range"].values[rows_mask], df["row"].values[rows_mask]
    return dict(
        ranges=r, rows=w, exps=df["experiment"].values[rows_mask], v3=v3[rows_mask], v2=v2[rows_mask],
        r_mu=float(np.mean(r)), r_sd=max(float(np.std(r)), 1.0),
        w_mu=float(np.mean(w)), w_sd=max(float(np.std(w)), 1.0),
    )


def _spatial_one(ref, r, w, e, v3_i, v2_i):
    dist = np.abs(ref["ranges"] - r) + np.abs(ref["rows"] - w)
    nb = (dist > 0) & (dist <= 2.5) & (ref["exps"] == e)
    if nb.sum() < 2:
        nb = (dist > 0) & (dist <= 3.5)
    if nb.sum() > 0:
        n3, n2 = float(np.mean(ref["v3"][nb])), float(np.mean(ref["v2"][nb]))
    else:
        n3, n2 = v3_i, v2_i
    return n3, n2, (r - ref["r_mu"]) / ref["r_sd"], (w - ref["w_mu"]) / ref["w_sd"]


# ----------------------------------------------------------------------------- names
def gen_names(variant):
    if variant == "m1":
        return ["Loc_TrainSplit_Mean_Prior", "Gen_Hybrid_OOF_Adv", "Gen_Hybrid_OOF_Z", "Gen_Parent1_OOF_Adv",
                "Gen_Parent2_OOF_Adv", "Gen_Parent_GCA_OOF", "Gen_Composite_OOF_Adv", "Gen_LocExperiment_OOF_Adv"]
    return ["Gen_Hybrid_LOLO_Adv", "Gen_Hybrid_LOLO_Z", "Gen_Parent1_LOLO_Adv", "Gen_Parent2_LOLO_Adv",
            "Gen_Parent_GCA_LOLO", "Gen_Composite_LOLO_Adv"]


def meta_bio_names(variant):
    base = ["Meta_TotalStandCount", "Meta_Planting_DOY", "Meta_Stand_Density_10kPx", "Meta_Log_StandCount",
            "Meta_FlightNorm_Stand_Z", "Meta_FlightNorm_Density_Z", "Bio_Effective_Green_Stand_TP2",
            "Bio_Effective_Green_Stand_TP3", "Bio_FlightStand_Plus_TP3Cover", "Bio_LogStand_x_TP3_PureExG",
            "Bio_Canopy_Per_Plant_TP2", "Bio_Canopy_Per_Plant_TP3", "Bio_StayGreen_Ratio_TP3_TP2",
            "Bio_Canopy_Expansion_Ratio_TP2_TP1", "Bio_Senescence_To_Green_TP3", "Bio_GapAdjusted_GreenCover_TP3",
            "Bio_FlightNorm_Season_Vigor", "Bio_Genetics_Plus_CanopyVigor"]
    return base if variant == "m1" else [n for n in base if n != "Meta_Planting_DOY"]


SPATIAL_NAMES = ["Spatial_Neighbor_CanopyVigor_TP3", "Spatial_PlotMinusNeighbor_Vigor_TP3",
                 "Spatial_Neighbor_CanopyVigor_TP2", "Spatial_PlotMinusNeighbor_Vigor_TP2",
                 "Spatial_FieldGrid_Range_Z", "Spatial_FieldGrid_Row_Z"]
CANOPY_BLOCKS = ["TP1", "TP2", "TP3", "Delta_TP2_TP1", "Delta_TP3_TP2", "Delta_TP3_TP1", "Season_AUC"]
CNN_BLOCKS = ["CNN_TP3_Mean", "CNN_TP2_Mean", "CNN_TP1_Mean", "CNN_TP3_PatchStd",
              "CNN_Delta_TP3_TP2", "CNN_Delta_TP2_TP1", "CNN_Delta_TP3_TP1"]


def proto_names(variant, labels):
    tail = "OOF_RidgeVigor" if variant == "m1" else "LOLO_RidgeVigor"
    return [f"Proto_{lb}_{s}" for lb in labels for s in ("SimHigh", "SimLow", "NetDiff", tail)]


def canopy_blocks_from(c1, c2, c3):
    return [("TP1", c1), ("TP2", c2), ("TP3", c3), ("Delta_TP2_TP1", c2 - c1), ("Delta_TP3_TP2", c3 - c2),
            ("Delta_TP3_TP1", c3 - c1), ("Season_AUC", 0.25 * c1 + 0.50 * c2 + 0.25 * c3)]


def cnn_blocks_from(n):
    return [("CNN_TP3_Mean", n["tp3_mean"]), ("CNN_TP2_Mean", n["tp2_mean"]), ("CNN_TP1_Mean", n["tp1_mean"]),
            ("CNN_TP3_PatchStd", n["tp3_std"]), ("CNN_Delta_TP3_TP2", n["tp3_mean"] - n["tp2_mean"]),
            ("CNN_Delta_TP2_TP1", n["tp2_mean"] - n["tp1_mean"]), ("CNN_Delta_TP3_TP1", n["tp3_mean"] - n["tp1_mean"])]


# =============================================================================
#                                   FIT
# =============================================================================
def fit_state(variant: str, raw: Dict[str, np.ndarray], man: pd.DataFrame, y: np.ndarray, splits: np.ndarray,
              feature_names: List[str], x_raw_ref: np.ndarray, x_phys_raw_ref: Optional[np.ndarray],
              gt: Optional[pd.DataFrame] = None, s5_genotypes: Optional[np.ndarray] = None) -> dict:
    locs = man["location"].values.astype(str)
    train = splits == "train"
    S: dict = {"variant": variant, "feature_names": list(feature_names),
               "known_locs": sorted(np.unique(locs).tolist()),
               "train_locs": sorted(np.unique(locs[train]).tolist())}

    # --- meta imputation medians (global, train rows)
    st = pd.to_numeric(man["totalStandCount"], errors="coerce")
    S["stand_median"] = float(st[train].median()) if st[train].notnull().any() else 50.0
    doy = pd.to_datetime(man["plantingDate"], errors="coerce").dt.dayofyear
    S["doy_median"] = float(doy[train].median()) if doy[train].notnull().any() else 130.0
    S["meta_parity_maxdiff"] = float(np.abs(
        compute_meta(man, raw["plot_areas"], S["stand_median"], S["doy_median"]) - raw["meta"]).max())

    # --- flight normalisation stats per location
    S["norm"] = {}
    norm_ref = {k: np.zeros_like(_clean(raw[k])) for k in IMG_KEYS}
    for loc in np.unique(locs):
        lm = locs == loc
        S["norm"][loc] = _fit_loc_norm(raw, ref_mask_for(lm, train, variant))
        _apply_loc_norm(raw, lm, S["norm"][loc], norm_ref)
    c1n, c2n, c3n = norm_ref["tp1_canopy"], norm_ref["tp2_canopy"], norm_ref["tp3_canopy"]

    # --- genetics / pheno
    S["gen"] = _fit_gen(variant, man, y, splits, locs)
    S["pheno"] = _fit_pheno(variant, man, splits, locs, gt)

    # --- meta/bio per-location stand statistics
    S["meta_loc"] = {}
    for loc in np.unique(locs):
        ref = ref_mask_for(locs == loc, train, variant)
        sc, den = raw["meta"][ref, 0], raw["meta"][ref, 2]
        S["meta_loc"][loc] = (float(np.mean(sc)), max(float(np.std(sc)), 1.0),
                              float(np.mean(den)), max(float(np.std(den)), 1e-3))

    # --- spatial reference plots per location
    v3 = 0.5 * (c3n[:, 6] + c3n[:, 8])
    v2 = 0.5 * (c2n[:, 6] + c2n[:, 8])
    S["spatial"] = {loc: _spatial_ref(ref_mask_for(locs == loc, train, variant), man, v3, v2, variant)
                    for loc in np.unique(locs)}

    # --- prototypes on flight-normalised features
    S["proto_norm"] = fit_protos(norm_ref, y, locs, train)

    # --- physical views
    if variant == "m1":
        S["proto_raw"] = fit_protos(raw, y, locs, train)
        S["phys"] = _fit_phys_m1(S, raw, man, x_phys_raw_ref, splits)
    else:
        S["phys"] = _fit_phys_m2(raw, y, locs, train)

    # --- final scaler (identical to the one the training scripts fitted on X_raw[train])
    S["x_scaler"] = StandardScaler().fit(x_raw_ref[train])
    if variant == "m1":
        S["loc_means"] = {loc: float(np.mean(y[train & (locs == loc)])) for loc in S["train_locs"]}
    return S


def _fit_gen(variant, man, y, splits, locs):
    train = splits == "train"
    geno = man["genotype"].values.astype(str)
    p1 = np.array([parse_hybrid_parents(g)[0] for g in geno], dtype=object)
    p2 = np.array([parse_hybrid_parents(g)[1] for g in geno], dtype=object)
    exp_raw = (man["location"] + "_" + man["experiment"]).values.astype(str)

    y_within = np.zeros_like(y, dtype=np.float32)
    y_z = np.zeros_like(y, dtype=np.float32)
    loc_prior = {}
    for loc in np.unique(locs[train]):
        m_tr = (locs == loc) & train
        mu, sd = float(np.mean(y[m_tr])), max(float(np.std(y[m_tr])), 1.0)
        la = locs == loc
        y_within[la] = y[la] - mu
        y_z[la] = (y[la] - mu) / sd
        loc_prior[loc] = mu

    tl = locs[train]

    def fit(cats, target, m):
        prior = float(np.mean(target[train]))
        return {"map": _fit_cat_map(cats[train], target[train], tl, m, prior), "prior": prior}

    G = {"loc_prior": loc_prior,
         "geno": fit(geno, y_within, 4.0), "geno_z": fit(geno, y_z, 4.0),
         "p1": fit(p1, y_within, 6.0), "p2": fit(p2, y_within, 6.0)}
    if variant == "m1":
        G["exp"] = fit(exp_raw, y_within, 8.0)
    G["seen_genos"] = set(G["geno"]["map"].keys())
    return G


def _fit_pheno(variant, man, splits, locs, gt):
    if gt is None:
        return None
    train = splits == "train"
    if "qrCode" not in man or "qrCode" not in gt:
        return None
    df = man[["qrCode"]].merge(gt.drop_duplicates(subset=["qrCode"]), on="qrCode", how="left")
    P = {"med": {}, "loc": {}}
    for col in PHENO_COLS:
        if col not in df.columns:
            continue
        raw_v = pd.to_numeric(df[col], errors="coerce")
        if raw_v[train].notnull().sum() <= 100:
            continue
        med = float(raw_v[train].median())
        filled = raw_v.fillna(med).values.astype(np.float32)
        P["med"][col] = med
        P["loc"][col] = {}
        for loc in np.unique(locs):
            lm = locs == loc
            rm = lm & train
            if rm.sum() < MIN_REF_PLOTS and variant == "m2":
                rm = lm
            P["loc"][col][loc] = (float(np.mean(filled[rm])), max(float(np.std(filled[rm])), 1.0))
    return P if P["med"] else None


# ---- physical views
def _phys_pool_m1(meta, raw_proto, src):
    c1, c2, c3 = src["tp1_canopy"], src["tp2_canopy"], src["tp3_canopy"]
    return np.nan_to_num(np.hstack([meta, raw_proto, c1, c2, c3, c2 - c1, c3 - c2, c3 - c1,
                                    src["tp3_mean"], src["tp2_mean"], src["tp1_mean"]]).astype(np.float32), nan=0.0)


def _fit_phys_m1(S, raw, man, x_phys_raw_ref, splits):
    """The 140 'physical' columns were picked by a train-only ranking that was never saved.
    Re-identify them by matching against the stored matrix on the non-train rows."""
    nt = splits != "train"
    rp, _ = apply_protos(S["proto_raw"], raw)
    pool = _phys_pool_m1(raw["meta"], rp, raw)
    A, B = pool[nt], x_phys_raw_ref[nt]
    idx, worst = [], 0.0
    for j in range(B.shape[1]):
        d = np.abs(A - B[:, [j]]).max(axis=0)
        k = int(np.argmin(d))
        idx.append(k)
        worst = max(worst, float(d[k]))
    scaler = StandardScaler().fit(x_phys_raw_ref[splits == "train"])
    return {"idx": np.array(idx), "scaler": scaler, "match_maxdiff": worst}


def _fit_phys_m2(raw, y, locs, train):
    c1, c2, c3 = raw["tp1_canopy"], raw["tp2_canopy"], raw["tp3_canopy"]
    ynorm, hi_m, lo_m = _loc_target_stats(y, locs, train)
    cnn = [raw["tp1_mean"], raw["tp2_mean"], raw["tp3_mean"], raw["tp2_mean"] - raw["tp1_mean"],
           raw["tp3_mean"] - raw["tp2_mean"], raw["tp3_mean"] - raw["tp1_mean"]]
    protos = []
    for mat in cnn:
        tm = np.mean(mat[train], axis=0, keepdims=True)
        mc = _l2n(mat - tm)
        mt = mc[train]
        hp = _l2n(np.mean(mt[hi_m], axis=0, keepdims=True))[0]
        lp = _l2n(np.mean(mt[lo_m], axis=0, keepdims=True))[0]
        protos.append((tm, hp, lp, RidgeCV(alphas=[50.0, 150.0, 300.0, 600.0]).fit(mt, ynorm)))

    cand = _m2_candidates(raw)
    X_tr, y_tr, tl = cand[train], y[train], locs[train]
    ul = np.unique(tl)
    corr = np.zeros((len(ul), X_tr.shape[1]), dtype=np.float32)
    lmean = np.zeros_like(corr)
    lvar = np.zeros_like(corr)
    for i, loc in enumerate(ul):
        m = tl == loc
        Xs, ys = X_tr[m], y_tr[m]
        yc, Xc = ys - np.mean(ys), Xs - np.mean(Xs, axis=0, keepdims=True)
        den = np.sqrt(np.sum(Xc ** 2, axis=0) * np.sum(yc ** 2))
        corr[i] = np.where(den > 1e-8, (yc @ Xc) / den, 0.0)
        lmean[i], lvar[i] = np.mean(Xs, axis=0), np.var(Xs, axis=0) + 1e-6
    mr = np.mean(corr, axis=0)
    sa = np.maximum(np.sum(corr > 0, axis=0), np.sum(corr < 0, axis=0)) / float(len(ul))
    mnr = np.min(corr * np.sign(mr + 1e-12), axis=0)
    cb = np.var(lmean, axis=0) / np.clip(np.mean(lvar, axis=0), 1e-6, None)
    score = (np.abs(mr) * (sa ** 1.5) * np.where(mnr > 0, 1.25, 0.75)) / (1.0 + 0.35 * cb)
    valid = np.where((cb < 1.25) & (sa >= 0.75))[0]
    top120 = valid[np.argsort(score[valid])[::-1]][:120]

    S = {"protos": protos, "top120": top120}
    raw_full = _phys_m2_raw(S, raw)
    S["scaler"] = StandardScaler().fit(raw_full[train])
    return S


def _m2_candidates(raw):
    c1, c2, c3 = raw["tp1_canopy"], raw["tp2_canopy"], raw["tp3_canopy"]
    canopy_all = np.hstack([c1, c2, c3, c2 - c1, c3 - c2, c3 - c1, 0.25 * c1 + 0.50 * c2 + 0.25 * c3]).astype(np.float32)
    cnn_all = np.hstack([raw["tp3_mean"], raw["tp2_mean"], raw["tp1_mean"], raw["tp3_std"],
                         raw["tp3_mean"] - raw["tp2_mean"], raw["tp2_mean"] - raw["tp1_mean"],
                         raw["tp3_mean"] - raw["tp1_mean"]]).astype(np.float32)
    return np.hstack([canopy_all, cnn_all])


def _phys_m2_raw(P, raw):
    meta = raw["meta"]
    sc, den, lg = meta[:, 0], meta[:, 2], meta[:, 3]
    c1, c2, c3 = raw["tp1_canopy"], raw["tp2_canopy"], raw["tp3_canopy"]
    f1, f2, f3 = c1[:, 6], c2[:, 6], c3[:, 6]
    bio = np.column_stack([
        sc, den, lg, den * f2, den * f3, lg * c3[:, 8],
        f2 / np.clip(sc, 10.0, None), f3 / np.clip(sc, 10.0, None),
        (f3 + 0.02) / (f2 + 0.02), (f2 + 0.02) / (f1 + 0.02), c3[:, 7] / (f3 + 0.02), f3 - c3[:, 10],
        lg * (0.20 * c1[:, 8] + 0.40 * c2[:, 8] + 0.40 * c3[:, 8]),
    ]).astype(np.float32)
    cnn = [raw["tp1_mean"], raw["tp2_mean"], raw["tp3_mean"], raw["tp2_mean"] - raw["tp1_mean"],
           raw["tp3_mean"] - raw["tp2_mean"], raw["tp3_mean"] - raw["tp1_mean"]]
    pl = []
    for (tm, hp, lp, ridge), mat in zip(P["protos"], cnn):
        mc = _l2n(mat - tm)
        sh, sl = mc @ hp, mc @ lp
        pl.append(np.column_stack([sh, sl, sh - sl, ridge.predict(mc).astype(np.float32)]))
    cand = _m2_candidates(raw)
    return np.hstack([bio, np.hstack(pl).astype(np.float32), cand[:, P["top120"]]]).astype(np.float32)


# =============================================================================
#                                 TRANSFORM
# =============================================================================
class BatchStatsError(ValueError):
    pass


def transform(S: dict, man: pd.DataFrame, raw: Dict[str, np.ndarray]) -> dict:
    """
    man : cleaned manifest for the batch (clean_manifest)
    raw : per-plot raw features with the same keys as the training cache
          (tp{1,2,3}_{mean,std,canopy}, plot_areas) -- `meta` is computed here.
    returns dict(X_full, X_phys, flags, raw_cols)   (scaled, ready for the model bundles)
    """
    variant = S["variant"]
    n = len(man)
    locs = man["location"].values.astype(str)
    raw = dict(raw)
    raw["meta"] = compute_meta(man, raw["plot_areas"], S["stand_median"], S["doy_median"])
    flags = [[] for _ in range(n)]
    for i in range(n):
        if pd.isna(pd.to_numeric(man["totalStandCount"].iloc[i], errors="coerce")):
            flags[i].append("stand_count_imputed")
        if pd.isna(pd.to_datetime(man["plantingDate"].iloc[i], errors="coerce")):
            flags[i].append("planting_date_imputed")

    # ---- per-location statistics: frozen where known, otherwise taken from this batch (label-free)
    norm_stats, meta_loc, spat_ref, pheno_loc = {}, {}, {}, {}
    norm_b = {k: np.zeros_like(_clean(raw[k])) for k in IMG_KEYS}
    for loc in np.unique(locs):
        rows = locs == loc
        if loc in S["norm"]:
            norm_stats[loc] = S["norm"][loc]
        else:
            if variant == "m1":
                raise BatchStatsError(f"Model 1 has no reference data for location '{loc}'")
            if rows.sum() < MIN_REF_PLOTS:
                raise BatchStatsError(f"Unseen location '{loc}' needs >= {MIN_REF_PLOTS} plots in the batch "
                                      f"to estimate field statistics (got {int(rows.sum())})")
            norm_stats[loc] = _fit_loc_norm(raw, rows)
            for i in np.where(rows)[0]:
                flags[i].append("field_stats_from_batch")
        _apply_loc_norm(raw, rows, norm_stats[loc], norm_b)
    c1r, c2r, c3r = raw["tp1_canopy"], raw["tp2_canopy"], raw["tp3_canopy"]
    c1n, c2n, c3n = norm_b["tp1_canopy"], norm_b["tp2_canopy"], norm_b["tp3_canopy"]
    v3 = 0.5 * (c3n[:, 6] + c3n[:, 8])
    v2 = 0.5 * (c2n[:, 6] + c2n[:, 8])

    for loc in np.unique(locs):
        rows = locs == loc
        if loc in S["meta_loc"]:
            meta_loc[loc] = S["meta_loc"][loc]
            spat_ref[loc] = S["spatial"][loc]
        else:  # unseen location: label-free batch reference
            sc, den = raw["meta"][rows, 0], raw["meta"][rows, 2]
            meta_loc[loc] = (float(np.mean(sc)), max(float(np.std(sc)), 1.0),
                             float(np.mean(den)), max(float(np.std(den)), 1e-3))
            spat_ref[loc] = _spatial_ref(rows, man, v3, v2, variant)

    cand: Dict[str, np.ndarray] = {}

    # ---- genetics
    G = S["gen"]
    geno = man["genotype"].values.astype(str)
    p1 = np.array([parse_hybrid_parents(g)[0] for g in geno], dtype=object)
    p2 = np.array([parse_hybrid_parents(g)[1] for g in geno], dtype=object)
    e_g, e_gz = _enc(geno, G["geno"]["map"], G["geno"]["prior"]), _enc(geno, G["geno_z"]["map"], G["geno_z"]["prior"])
    e_p1, e_p2 = _enc(p1, G["p1"]["map"], G["p1"]["prior"]), _enc(p2, G["p2"]["map"], G["p2"]["prior"])
    gca = 0.5 * (e_p1 + e_p2)
    seen = np.array([(g in G["seen_genos"]) and g != "UNKNOWN" for g in geno])
    comp = np.where(seen, e_g, gca).astype(np.float32)
    for i in np.where(~seen)[0]:
        flags[i].append("genotype_unseen")
    if variant == "m1":
        exp_raw = (man["location"] + "_" + man["experiment"]).values.astype(str)
        e_exp = _enc(exp_raw, G["exp"]["map"], G["exp"]["prior"])
        for i in np.where(~np.isin(exp_raw, list(G["exp"]["map"].keys())))[0]:
            flags[i].append("experiment_unseen")
        loc_prior = np.array([G["loc_prior"][l] for l in locs], dtype=np.float32)
        gvals = [loc_prior, e_g, e_gz, e_p1, e_p2, gca, comp, e_exp]
    else:
        gvals = [e_g, e_gz, e_p1, e_p2, gca, comp]
    cand.update(dict(zip(gen_names(variant), gvals)))

    # ---- phenology (z-scored per location); neutral 0 if the state lacks it
    P = S["pheno"]
    for col in PHENO_COLS:
        nm = f"Pheno_FlightNorm_{col}"
        if P is None or col not in P["med"]:
            cand[nm] = np.zeros(n, dtype=np.float32)
            for i in range(n):
                flags[i].append("pheno_state_missing")
            continue
        v = pd.to_numeric(man[col], errors="coerce")
        for i in np.where(v.isna().values)[0]:
            flags[i].append("pheno_imputed")
        filled = v.fillna(P["med"][col]).values.astype(np.float32)
        z = np.zeros(n, dtype=np.float32)
        for loc in np.unique(locs):
            rows = locs == loc
            if loc in P["loc"][col]:
                mu, sd = P["loc"][col][loc]
            else:
                mu, sd = float(np.mean(filled[rows])), max(float(np.std(filled[rows])), 1.0)
            z[rows] = (filled[rows] - mu) / sd
        cand[nm] = z

    # ---- meta + bio ratios
    meta = raw["meta"]
    sc, doy, den, lg = meta[:, 0], meta[:, 1], meta[:, 2], meta[:, 3]
    sz = np.zeros(n, dtype=np.float32)
    dz = np.zeros(n, dtype=np.float32)
    for loc in np.unique(locs):
        rows = locs == loc
        s_mu, s_sd, d_mu, d_sd = meta_loc[loc]
        sz[rows] = (sc[rows] - s_mu) / s_sd
        dz[rows] = (den[rows] - d_mu) / d_sd
    f1, f2, f3 = c1r[:, 6], c2r[:, 6], c3r[:, 6]
    y3, gap3 = c3r[:, 7], c3r[:, 10]
    pe1, pe2, pe3 = c1n[:, 8], c2n[:, 8], c3n[:, 8]
    vigor = 0.20 * pe1 + 0.40 * pe2 + 0.40 * pe3
    mb = [sc, doy, den, lg, sz, dz, den * f2, den * f3, sz + c3n[:, 6], lg * pe3,
          f2 / np.clip(sc, 10.0, None), f3 / np.clip(sc, 10.0, None), (f3 + 0.02) / (f2 + 0.02),
          (f2 + 0.02) / (f1 + 0.02), y3 / (f3 + 0.02), f3 - gap3, vigor, comp + 4.0 * vigor]
    mb = np.column_stack(mb).astype(np.float32)
    base_names = meta_bio_names("m1")
    keep = [base_names.index(nm) for nm in meta_bio_names(variant)]
    cand.update({nm: mb[:, j] for nm, j in zip(meta_bio_names(variant), keep)})

    # ---- spatial neighbourhood
    ranges, rows_ = man["range"].values.astype(np.float32), man["row"].values.astype(np.float32)
    exps = man["experiment"].values.astype(str)
    sp = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        n3, n2, rr, rw = _spatial_one(spat_ref[locs[i]], ranges[i], rows_[i], exps[i], v3[i], v2[i])
        sp[i] = [n3, v3[i] - n3, n2, v2[i] - n2, rr, rw]
    cand.update({nm: sp[:, j] for j, nm in enumerate(SPATIAL_NAMES)})

    # ---- prototypes (flight-normalised)
    pm, labels = apply_protos(S["proto_norm"], norm_b)
    pn = proto_names(variant, labels)
    cand.update({nm: pm[:, j] for j, nm in enumerate(pn)})

    # ---- canopy index blocks
    for prefix, blk in canopy_blocks_from(c1n, c2n, c3n):
        for j, nm in enumerate(CANOPY_NAMES):
            cand[f"Canopy_FlightNorm_{prefix}_{nm}"] = blk[:, j]

    # ---- raw CNN channels (only those selected)
    wanted_cnn = [f for f in S["feature_names"] if f.startswith("CNN_")]
    if wanted_cnn:
        blocks = dict(cnn_blocks_from(norm_b))
        for f in wanted_cnn:
            blk, d = re.match(r"(CNN_.+)_D(\d+)$", f).groups()
            cand[f] = blocks[blk][:, int(d)]

    X_raw = np.nan_to_num(np.column_stack([cand[f] for f in S["feature_names"]]).astype(np.float32), nan=0.0)
    X_full = S["x_scaler"].transform(X_raw).astype(np.float32)

    # ---- physical view
    if variant == "m1":
        rp, _ = apply_protos(S["proto_raw"], raw)
        pool = _phys_pool_m1(raw["meta"], rp, raw)
        Xp_raw = pool[:, S["phys"]["idx"]]
        X_phys = S["phys"]["scaler"].transform(Xp_raw).astype(np.float32)
    else:
        Xp_raw = _phys_m2_raw(S["phys"], raw)
        X_phys = S["phys"]["scaler"].transform(Xp_raw).astype(np.float32)

    return {"X_raw": X_raw, "X_full": X_full, "X_phys": X_phys, "X_phys_raw": Xp_raw,
            "flags": flags, "feature_names": S["feature_names"]}
