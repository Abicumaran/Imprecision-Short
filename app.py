
import io
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st


# ============================================================
# Imprecision Short / Repeatability App
# Upload Excel/CSV -> assign conditions -> select analytes/devices
# -> repeatability SD/CV per device + pooled across devices
# -> raw and cleaned outputs + diagnostics + bootstrap CIs
# ============================================================

ID_GUESSES = {
    "batch": ["batch_id", "batchId", "batchID", "Batch ID", "batch"],
    "sample": ["bloodSampleId", "Blood Sample ID", "bloodSampleID", "sample_id", "sampleId"],
    "device": ["deviceId", "Device", "device_id", "serialNumber", "serial_number"],
    "condition": ["Condition", "condition", "Group", "group", "Level", "level"],
}


# Requested Streamlit default ordering. Only columns actually present in the
# uploaded file are selected, but they appear in this order.
DEFAULT_ANALYTE_ORDER = [
    "RBC", "WBC_2", "PLT", "HCT", "HGB", "MCV", "RDW", "MCH", "MCHC",
    "NEUT_2", "LYMPH_2", "MXD_2", "PLT_3", "MCV_3", "RDW_3",
]

AUTO_OUTLIER_METHOD = "Automatic: Shapiro-Wilk -> Gcrit if normal, Robust MAD if non-normal"
GCRIT_OUTLIER_METHOD = "Gcrit Grubbs-like: remove largest |value-mean|/SD if >= Gcrit"
MAD_OUTLIER_METHOD = "Robust MAD modified-z: remove largest robust z if >= threshold"


@dataclass
class AnalysisConfig:
    batch_col: str
    sample_col: str
    device_col: str
    condition_map: Dict[str, List[str]]
    analytes: List[str]
    devices: List[str]
    device_mode: str
    outlier_method: str
    max_outliers_per_group: int
    gcrit_mode: str
    gcrit: float
    gcrit_alpha: float
    gcrit_tail: str
    mad_zcrit: float
    robust_interval_z: float
    make_bootstrap_ci: bool
    n_boot: int
    random_seed: int


def guess_col(cols: List[str], key: str) -> Optional[str]:
    for g in ID_GUESSES.get(key, []):
        if g in cols:
            return g
    # case-insensitive fallback
    lower = {c.lower(): c for c in cols}
    for g in ID_GUESSES.get(key, []):
        if g.lower() in lower:
            return lower[g.lower()]
    return None


def extract_sample_number(x) -> str:
    """
    Default parsing:
    APT-160425D02-IS-1 -> 1
    Anything after 'IS-' up to the next non-alphanumeric/dot/underscore token.
    Falls back to the full sample string if no IS- pattern exists.
    """
    s = str(x)
    m = re.search(r"IS-([A-Za-z0-9._]+)", s)
    if m:
        return str(m.group(1))
    return s


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def robust_sd_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def grubbs_critical_value(n: int, alpha: float = 0.01, tail: str = "Two-sided") -> float:
    """Classical Grubbs critical value for sample size n.

    Two-sided: t = t_(1 - alpha/(2n), n-2)
    One-sided: t = t_(1 - alpha/n, n-2)
    Gcrit = ((n - 1) / sqrt(n)) * sqrt(t^2 / (n - 2 + t^2))
    """
    n = int(n)
    alpha = float(alpha)
    if n < 3 or not np.isfinite(alpha) or alpha <= 0 or alpha >= 1:
        return np.nan
    if str(tail).lower().startswith("one"):
        q = 1.0 - alpha / n
    else:
        q = 1.0 - alpha / (2.0 * n)
    tcrit = stats.t.ppf(q, df=n - 2)
    if not np.isfinite(tcrit):
        return np.nan
    return float(((n - 1) / np.sqrt(n)) * np.sqrt((tcrit ** 2) / (n - 2 + tcrit ** 2)))


def flag_outliers_one_group(
    vals: pd.Series,
    method: str,
    max_remove: int = 1,
    gcrit_mode: str = "Manual Gcrit value",
    gcrit: float = 3.135,
    gcrit_alpha: float = 0.01,
    gcrit_tail: str = "Two-sided",
    mad_zcrit: float = 3.5,
    robust_interval_z: float = 1.96,
) -> Tuple[pd.Series, List[Dict]]:
    """
    Sequentially flags up to max_remove outliers within one
    condition-device-sample replicate cell for one analyte.
    """
    x_all = vals.astype(float)
    flags = pd.Series(False, index=x_all.index)
    logs: List[Dict] = []

    if method == "None" or int(max_remove) <= 0:
        return flags, logs

    remaining = list(x_all[np.isfinite(x_all)].index)
    if len(remaining) < 3:
        return flags, logs

    for step in range(int(max_remove)):
        x = x_all.loc[remaining].to_numpy(dtype=float)
        if len(x) < 3:
            break

        chosen_idx = None
        metric = np.nan
        threshold = np.nan
        direction = ""
        details = ""

        if method.startswith("Gcrit") or method == "Gcrit / Grubbs-like":
            mu = float(np.mean(x))
            sd = float(np.std(x, ddof=1))
            if not np.isfinite(sd) or sd == 0:
                break
            threshold = (
                grubbs_critical_value(len(x), gcrit_alpha, gcrit_tail)
                if str(gcrit_mode).startswith("Automatic")
                else float(gcrit)
            )
            if not np.isfinite(threshold):
                break
            gvals = np.abs(x - mu) / sd
            k = int(np.argmax(gvals))
            metric = float(gvals[k])
            if metric >= threshold:
                chosen_idx = remaining[k]
                direction = "high" if x[k] > mu else "low"
                details = f"G={metric:.4g}; mean={mu:.4g}; sd={sd:.4g}; Gcrit={threshold:.4g}; n={len(x)}"
            else:
                break

        elif method.startswith("Robust MAD"):
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med)))
            if not np.isfinite(mad) or mad == 0:
                break
            modz_signed = 0.6745 * (x - med) / mad
            k = int(np.argmax(np.abs(modz_signed)))
            metric = float(abs(modz_signed[k]))
            threshold = float(mad_zcrit)
            if metric >= threshold:
                chosen_idx = remaining[k]
                direction = "high" if x[k] > med else "low"
                details = f"modified_z={modz_signed[k]:.4g}; median={med:.4g}; MAD={mad:.4g}; threshold={threshold:.4g}"
            else:
                break

        elif method.startswith("95% robust interval"):
            med = float(np.median(x))
            rsd = robust_sd_mad(x)
            if not np.isfinite(rsd) or rsd == 0:
                break
            lo = med - float(robust_interval_z) * rsd
            hi = med + float(robust_interval_z) * rsd
            distances = np.maximum(lo - x, x - hi)
            k = int(np.argmax(distances))
            metric = float(distances[k])
            threshold = 0.0
            if metric > 0:
                chosen_idx = remaining[k]
                direction = "high" if x[k] > hi else "low"
                details = f"value outside robust interval [{lo:.4g}, {hi:.4g}]; median={med:.4g}; robust_SD={rsd:.4g}; z={robust_interval_z}"
            else:
                break

        if chosen_idx is None:
            break

        flags.loc[chosen_idx] = True
        logs.append({
            "removed_order": step + 1,
            "outlier_metric": metric,
            "outlier_threshold": threshold,
            "direction": direction,
            "details": details,
        })
        remaining.remove(chosen_idx)

    return flags.fillna(False), logs

def make_long_working_df(df: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    work = df.copy()
    work["_sample_number"] = work[cfg.sample_col].apply(extract_sample_number).astype(str)
    work["_device"] = work[cfg.device_col].astype(str)
    work["_batch_id"] = work[cfg.batch_col].astype(str)

    # condition mapping from selected sample numbers
    sample_to_condition = {}
    for condition, sample_list in cfg.condition_map.items():
        for s in sample_list:
            sample_to_condition[str(s)] = condition
    work["_condition"] = work["_sample_number"].map(sample_to_condition)

    work = work[work["_condition"].notna()].copy()
    work = work[work["_device"].isin([str(d) for d in cfg.devices])].copy()
    work = coerce_numeric(work, cfg.analytes)
    return work


def _normality_status_from_p(p: float) -> Tuple[object, str]:
    if p is None or not np.isfinite(p):
        return np.nan, "Not testable"
    passed = bool(float(p) >= 0.05)
    return passed, "Normal" if passed else "Not normal"


def build_outlier_decisions(work: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Choose ONE outlier rule per condition x analyte from RAW pooled residual normality.

    Automatic mode is a selector, not a new outlier algorithm:
      - Shapiro-Wilk p >= 0.05 -> existing Gcrit/Grubbs-like rule
      - Shapiro-Wilk p < 0.05  -> existing robust MAD modified-z rule
      - Shapiro unavailable    -> robust MAD fallback

    The decision is based on all selected devices together, using sample x device
    replicate cells. The selected rule is then applied consistently to every
    condition x device x sample replicate cell for that analyte. This avoids
    mixing outlier rules within the same analyte and makes the reported pooled CV
    correspond to one clearly documented method.
    """
    rows = []
    automatic = cfg.outlier_method == AUTO_OUTLIER_METHOD

    for condition in cfg.condition_map.keys():
        cdf = work[work["_condition"] == condition].copy()
        if cdf.empty:
            continue
        for analyte in cfg.analytes:
            diag = diagnostic_tests(
                cdf, analyte,
                ["_sample_number", "_device"],
                ["_device"],
            )
            p = diag.get("shapiro_wilk_p_residuals", np.nan)
            normality_pass, status = _normality_status_from_p(p)
            if automatic:
                method = GCRIT_OUTLIER_METHOD if normality_pass is True else MAD_OUTLIER_METHOD
            else:
                method = cfg.outlier_method

            base = {
                "condition": condition,
                "analyte": analyte,
                "normality_pass_raw": normality_pass,
                "normality_status_raw": status,
                "shapiro_wilk_p_residuals_raw": p,
                "outlier_method_selected": method,
                "selection_scope": "pooled_selected_devices_raw_residuals",
            }
            # Store the same analyte-level decision for pooled and device lookups.
            for decision_device in ["ALL_SELECTED_DEVICES"] + [str(d) for d in cfg.devices]:
                row = base.copy()
                row["decision_device"] = decision_device
                rows.append(row)

    return pd.DataFrame(rows)


def apply_outlier_flags(work: pd.DataFrame, cfg: AnalysisConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flag outliers using either the selected manual rule or the automatic
    Shapiro-Wilk -> Gcrit/MAD selector, without changing the existing Gcrit or
    MAD detection formulas.
    """
    df = work.copy()
    logs = []
    decisions = build_outlier_decisions(work, cfg)
    decision_lookup = {}
    if not decisions.empty:
        for _, r in decisions.iterrows():
            decision_lookup[(str(r["condition"]), str(r["analyte"]), str(r["decision_device"]))] = r.to_dict()

    for analyte in cfg.analytes:
        flag_col = f"_outlier_{analyte}"
        df[flag_col] = False

        group_cols = ["_condition", "_device", "_sample_number"]
        for keys, sub in df.groupby(group_cols, dropna=False):
            condition, device, sample_number = keys
            decision = decision_lookup.get((str(condition), str(analyte), str(device)), {})
            method = decision.get("outlier_method_selected", cfg.outlier_method)

            vals = sub[analyte]
            flags, flag_logs = flag_outliers_one_group(
                vals,
                method,
                cfg.max_outliers_per_group,
                cfg.gcrit_mode,
                cfg.gcrit,
                cfg.gcrit_alpha,
                cfg.gcrit_tail,
                cfg.mad_zcrit,
                cfg.robust_interval_z,
            )
            df.loc[flags.index, flag_col] = flags.values

            if flags.sum() > 0:
                log_by_idx = {idx: info for idx, info in zip(flags[flags].index, flag_logs)}
                for row_idx in flags[flags].index:
                    row = df.loc[row_idx]
                    info = log_by_idx.get(row_idx, {})
                    logs.append({
                        "condition": condition,
                        "device": device,
                        "sample_number": sample_number,
                        "analyte": analyte,
                        "batch_id": row["_batch_id"],
                        "bloodSampleId": row[cfg.sample_col],
                        "deviceId": row[cfg.device_col],
                        "value": row[analyte],
                        "normality_status_raw": decision.get("normality_status_raw", "Not testable"),
                        "shapiro_wilk_p_residuals_raw": decision.get("shapiro_wilk_p_residuals_raw", np.nan),
                        "outlier_method": method,
                        "removed_order": info.get("removed_order", np.nan),
                        "direction": info.get("direction", ""),
                        "outlier_metric": info.get("outlier_metric", np.nan),
                        "outlier_threshold": info.get("outlier_threshold", np.nan),
                        "details": info.get("details", ""),
                        "gcrit_mode": cfg.gcrit_mode if method.startswith("Gcrit") else "",
                        "manual_gcrit": cfg.gcrit if method.startswith("Gcrit") else np.nan,
                        "gcrit_alpha": cfg.gcrit_alpha if method.startswith("Gcrit") else np.nan,
                        "gcrit_tail": cfg.gcrit_tail if method.startswith("Gcrit") else "",
                        "mad_zcrit": cfg.mad_zcrit if method.startswith("Robust MAD") else np.nan,
                        "robust_interval_z": cfg.robust_interval_z if method.startswith("95% robust interval") else np.nan,
                    })

    return df, pd.DataFrame(logs), decisions

def pooled_repeatability_stats(
    df: pd.DataFrame,
    analyte: str,
    group_cols: List[str],
) -> Dict[str, float]:
    """
    Repeatability based on within-cell replicate scatter.

    For each cell, e.g. sample x device:
        mean_cell = mean(y)
        SS_within = sum((y - mean_cell)^2)
    Pooled repeatability variance:
        sigma_repeat^2 = sum SS_within / sum(n_cell - 1)
    CV%:
        100 * SD_repeat / grand mean

    group_cols define replicate cells:
        per-device analysis: ["_sample_number"]
        pooled-all-devices analysis: ["_sample_number", "_device"]
    """
    ydf = df[group_cols + [analyte]].copy()
    ydf[analyte] = pd.to_numeric(ydf[analyte], errors="coerce")
    ydf = ydf[np.isfinite(ydf[analyte])].copy()

    if ydf.empty:
        return dict(
            n_rows=0, n_cells=0, mean=np.nan, median=np.nan,
            sd_repeat=np.nan, cv_repeat_pct=np.nan,
            robust_sd_mad=np.nan, robust_cv_mad_pct=np.nan,
            min_cell_n=np.nan, median_cell_n=np.nan, max_cell_n=np.nan
        )

    ss = 0.0
    dfree = 0
    cell_ns = []
    for _, sub in ydf.groupby(group_cols):
        vals = sub[analyte].dropna().astype(float).values
        n = len(vals)
        if n >= 2:
            ss += float(np.sum((vals - np.mean(vals)) ** 2))
            dfree += n - 1
            cell_ns.append(n)
        elif n == 1:
            cell_ns.append(n)

    sd_repeat = np.sqrt(ss / dfree) if dfree > 0 else np.nan
    center_mean = float(ydf[analyte].mean())
    center_median = float(ydf[analyte].median())
    cv_repeat = 100.0 * sd_repeat / center_mean if np.isfinite(sd_repeat) and center_mean != 0 else np.nan

    # Robust within-cell SD: median of MAD-scaled SDs, then CV by median.
    robust_sds = []
    for _, sub in ydf.groupby(group_cols):
        vals = sub[analyte].dropna().astype(float).values
        if len(vals) >= 2:
            r = robust_sd_mad(vals)
            if np.isfinite(r):
                robust_sds.append(r)
    robust_sd = float(np.median(robust_sds)) if len(robust_sds) else np.nan
    robust_cv = 100.0 * robust_sd / center_median if np.isfinite(robust_sd) and center_median != 0 else np.nan

    return dict(
        n_rows=int(len(ydf)),
        n_cells=int(ydf.groupby(group_cols).ngroups),
        mean=center_mean,
        median=center_median,
        sd_repeat=float(sd_repeat) if np.isfinite(sd_repeat) else np.nan,
        cv_repeat_pct=float(cv_repeat) if np.isfinite(cv_repeat) else np.nan,
        robust_sd_mad=float(robust_sd) if np.isfinite(robust_sd) else np.nan,
        robust_cv_mad_pct=float(robust_cv) if np.isfinite(robust_cv) else np.nan,
        min_cell_n=int(np.min(cell_ns)) if len(cell_ns) else np.nan,
        median_cell_n=float(np.median(cell_ns)) if len(cell_ns) else np.nan,
        max_cell_n=int(np.max(cell_ns)) if len(cell_ns) else np.nan,
    )


def residuals_for_diagnostics(df: pd.DataFrame, analyte: str, group_cols: List[str]) -> pd.Series:
    rows = []
    ydf = df[group_cols + [analyte]].copy()
    ydf[analyte] = pd.to_numeric(ydf[analyte], errors="coerce")
    ydf = ydf[np.isfinite(ydf[analyte])].copy()
    if ydf.empty:
        return pd.Series(dtype=float)
    for _, sub in ydf.groupby(group_cols):
        vals = sub[analyte].astype(float)
        rows.append(vals - vals.mean())
    if not rows:
        return pd.Series(dtype=float)
    return pd.concat(rows)


def diagnostic_tests(df: pd.DataFrame, analyte: str, group_cols: List[str], homogeneity_cols: List[str]) -> Dict[str, float]:
    """
    Diagnostics reported for every condition x analyte x device/scope:

    1) Shapiro-Wilk normality test on within-cell residuals.
       Residuals are y - cell mean, where cells are sample or sample x device.

    2) Levene homogeneity test across the requested homogeneity groups.
       - levene_mean_p: classic Levene using center='mean'.
       - brown_forsythe_median_p: Brown-Forsythe variant using center='median',
         usually more robust for non-normal assay data.

    A p-value < 0.05 is flagged as assumption_check='FAIL'.
    """
    resid = residuals_for_diagnostics(df, analyte, group_cols)
    out = {
        "shapiro_wilk_p_residuals": np.nan,
        "shapiro_wilk_normality_check": "not_tested",
        "levene_mean_p": np.nan,
        "levene_mean_homogeneity_check": "not_tested",
        "brown_forsythe_median_p": np.nan,
        "brown_forsythe_homogeneity_check": "not_tested",
        # Backwards-compatible names kept for old downstream code/templates.
        "shapiro_p_residuals": np.nan,
        "normality_pass_p_ge_0_05": np.nan,
        "brown_forsythe_p": np.nan,
        "homogeneity_pass_p_ge_0_05": np.nan,
    }

    vals = resid.dropna().astype(float).values
    if 3 <= len(vals) <= 5000 and np.std(vals) > 0:
        try:
            p = float(stats.shapiro(vals).pvalue)
            out["shapiro_wilk_p_residuals"] = p
            out["shapiro_wilk_normality_check"] = "PASS_p_ge_0.05" if p >= 0.05 else "FAIL_p_lt_0.05"
            out["shapiro_p_residuals"] = p
            out["normality_pass_p_ge_0_05"] = bool(p >= 0.05)
        except Exception:
            pass

    ydf = df[homogeneity_cols + [analyte]].copy()
    ydf[analyte] = pd.to_numeric(ydf[analyte], errors="coerce")
    ydf = ydf[np.isfinite(ydf[analyte])].copy()
    groups = []
    for _, sub in ydf.groupby(homogeneity_cols):
        group_vals = sub[analyte].dropna().astype(float).values
        if len(group_vals) >= 2:
            groups.append(group_vals)
    if len(groups) >= 2:
        try:
            p_mean = float(stats.levene(*groups, center="mean").pvalue)
            out["levene_mean_p"] = p_mean
            out["levene_mean_homogeneity_check"] = "PASS_p_ge_0.05" if p_mean >= 0.05 else "FAIL_p_lt_0.05"
        except Exception:
            pass
        try:
            p_median = float(stats.levene(*groups, center="median").pvalue)
            out["brown_forsythe_median_p"] = p_median
            out["brown_forsythe_homogeneity_check"] = "PASS_p_ge_0.05" if p_median >= 0.05 else "FAIL_p_lt_0.05"
            out["brown_forsythe_p"] = p_median
            out["homogeneity_pass_p_ge_0_05"] = bool(p_median >= 0.05)
        except Exception:
            pass
    return out


def bootstrap_ci_repeatability(
    df: pd.DataFrame,
    analyte: str,
    group_cols: List[str],
    n_boot: int,
    random_seed: int,
) -> Dict[str, float]:
    """
    Cluster bootstrap over replicate cells.
    Example cells:
      per-device: sample
      pooled: sample x device
    """
    rng = np.random.default_rng(random_seed)
    ydf = df[group_cols + [analyte]].copy()
    ydf[analyte] = pd.to_numeric(ydf[analyte], errors="coerce")
    ydf = ydf[np.isfinite(ydf[analyte])].copy()
    if ydf.empty:
        return {
            "sd_repeat_ci_low": np.nan, "sd_repeat_ci_high": np.nan,
            "cv_repeat_ci_low": np.nan, "cv_repeat_ci_high": np.nan,
        }

    cells = []
    for key, sub in ydf.groupby(group_cols):
        vals = sub[analyte].dropna().astype(float).values
        if len(vals) >= 2:
            cells.append(pd.DataFrame({analyte: vals, "_boot_cell": str(key)}))
    if len(cells) < 2:
        return {
            "sd_repeat_ci_low": np.nan, "sd_repeat_ci_high": np.nan,
            "cv_repeat_ci_low": np.nan, "cv_repeat_ci_high": np.nan,
        }

    sd_vals, cv_vals = [], []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(cells), size=len(cells))
        bdf_parts = []
        for k, i in enumerate(chosen):
            temp = cells[i].copy()
            temp["_resampled_cell"] = k
            bdf_parts.append(temp)
        bdf = pd.concat(bdf_parts, ignore_index=True)
        stats_dict = pooled_repeatability_stats(bdf, analyte, ["_resampled_cell"])
        sd_vals.append(stats_dict["sd_repeat"])
        cv_vals.append(stats_dict["cv_repeat_pct"])

    sd_vals = np.array(sd_vals, dtype=float)
    cv_vals = np.array(cv_vals, dtype=float)
    sd_vals = sd_vals[np.isfinite(sd_vals)]
    cv_vals = cv_vals[np.isfinite(cv_vals)]

    def q(arr, pct):
        return float(np.percentile(arr, pct)) if len(arr) else np.nan

    return {
        "sd_repeat_ci_low": q(sd_vals, 2.5),
        "sd_repeat_ci_high": q(sd_vals, 97.5),
        "cv_repeat_ci_low": q(cv_vals, 2.5),
        "cv_repeat_ci_high": q(cv_vals, 97.5),
    }


def summarize_analysis(work_flagged: pd.DataFrame, cfg: AnalysisConfig, cleaned: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diag_rows = []

    for condition in cfg.condition_map.keys():
        cdf0 = work_flagged[work_flagged["_condition"] == condition].copy()
        if cdf0.empty:
            continue

        for analyte in cfg.analytes:
            flag_col = f"_outlier_{analyte}"
            cdf = cdf0.copy()
            if cleaned and flag_col in cdf.columns:
                cdf = cdf[~cdf[flag_col]].copy()

            include_per_device = cfg.device_mode.startswith("Analyze each")

            if include_per_device:
                # Per device
                for device in sorted(cdf["_device"].dropna().astype(str).unique()):
                    ddf = cdf[cdf["_device"].astype(str) == str(device)].copy()
                    stats_dict = pooled_repeatability_stats(ddf, analyte, ["_sample_number"])
                    diag = diagnostic_tests(ddf, analyte, ["_sample_number"], ["_sample_number"])
                    ci = {}
                    if cfg.make_bootstrap_ci:
                        ci = bootstrap_ci_repeatability(ddf, analyte, ["_sample_number"], cfg.n_boot, cfg.random_seed)

                    rows.append({
                        "dataset": "cleaned_outliers_removed" if cleaned else "raw_no_outlier_removal",
                        "condition": condition,
                        "analyte": analyte,
                        "scope": "per_device",
                        "device": device,
                        "aggregation_method": "within_device_pooled_across_samples",
                        **stats_dict,
                        **ci,
                    })
                    diag_rows.append({
                        "dataset": "cleaned_outliers_removed" if cleaned else "raw_no_outlier_removal",
                        "condition": condition,
                        "analyte": analyte,
                        "scope": "per_device",
                        "device": device,
                        **diag,
                    })

            # Pooled across all devices
            stats_dict = pooled_repeatability_stats(cdf, analyte, ["_sample_number", "_device"])
            diag = diagnostic_tests(cdf, analyte, ["_sample_number", "_device"], ["_device"])
            ci = {}
            if cfg.make_bootstrap_ci:
                ci = bootstrap_ci_repeatability(cdf, analyte, ["_sample_number", "_device"], cfg.n_boot, cfg.random_seed)

            pooled_stats_dict = stats_dict.copy()
            pooled_ci = ci.copy()
            rows.append({
                "dataset": "cleaned_outliers_removed" if cleaned else "raw_no_outlier_removal",
                "condition": condition,
                "analyte": analyte,
                "scope": "pooled_all_devices",
                "device": "ALL_SELECTED_DEVICES",
                "aggregation_method": "pooled_sample_x_device_cells",
                **stats_dict,
                **ci,
            })
            diag_rows.append({
                "dataset": "cleaned_outliers_removed" if cleaned else "raw_no_outlier_removal",
                "condition": condition,
                "analyte": analyte,
                "scope": "pooled_all_devices",
                "device": "ALL_SELECTED_DEVICES",
                **diag,
            })

            if include_per_device:
                # Mean of device SD/CV, useful as a simple descriptive summary
                tmp = pd.DataFrame([r for r in rows if r["dataset"] == ("cleaned_outliers_removed" if cleaned else "raw_no_outlier_removal")
                                    and r["condition"] == condition and r["analyte"] == analyte and r["scope"] == "per_device"])
                if len(tmp) > 0:
                    mean_device_row = {
                        "dataset": "cleaned_outliers_removed" if cleaned else "raw_no_outlier_removal",
                        "condition": condition,
                        "analyte": analyte,
                        "scope": "mean_of_device_summaries",
                        "device": "MEAN_OF_SELECTED_DEVICES",
                        "aggregation_method": "simple_average_of_device_level_SD_and_CV",
                        "n_rows": int(tmp["n_rows"].sum()),
                        "n_cells": int(tmp["n_cells"].sum()),
                        "mean": float(tmp["mean"].mean()),
                        "median": float(tmp["median"].mean()),
                        "sd_repeat": float(tmp["sd_repeat"].mean()),
                        "cv_repeat_pct": float(tmp["cv_repeat_pct"].mean()),
                        "robust_sd_mad": float(tmp["robust_sd_mad"].mean()),
                        "robust_cv_mad_pct": float(tmp["robust_cv_mad_pct"].mean()),
                        "min_cell_n": float(tmp["min_cell_n"].min()),
                        "median_cell_n": float(tmp["median_cell_n"].median()),
                        "max_cell_n": float(tmp["max_cell_n"].max()),
                        "sd_repeat_ci_low": np.nan,
                        "sd_repeat_ci_high": np.nan,
                        "cv_repeat_ci_low": np.nan,
                        "cv_repeat_ci_high": np.nan,
                    }
                    rows.append(mean_device_row)

                    equal_replicates_across_devices = (
                        tmp["n_rows"].nunique(dropna=True) == 1
                        and tmp["n_cells"].nunique(dropna=True) == 1
                        and tmp["min_cell_n"].nunique(dropna=True) == 1
                        and tmp["median_cell_n"].nunique(dropna=True) == 1
                        and tmp["max_cell_n"].nunique(dropna=True) == 1
                    )
                    if equal_replicates_across_devices:
                        auto_row = mean_device_row.copy()
                        auto_row["scope"] = "overall_across_devices_auto"
                        auto_row["device"] = "AUTO_AVERAGE_EQUAL_REPLICATES"
                        auto_row["aggregation_method"] = "average_of_device_summaries_equal_replicates"
                        auto_row["equal_replicates_across_devices"] = True
                    else:
                        auto_row = {
                            "dataset": "cleaned_outliers_removed" if cleaned else "raw_no_outlier_removal",
                            "condition": condition,
                            "analyte": analyte,
                            "scope": "overall_across_devices_auto",
                            "device": "AUTO_POOLED_UNEQUAL_REPLICATES",
                            "aggregation_method": "pooled_sample_x_device_cells_unequal_replicates",
                            "equal_replicates_across_devices": False,
                            **pooled_stats_dict,
                            **pooled_ci,
                        }
                    rows.append(auto_row)

    return pd.DataFrame(rows), pd.DataFrame(diag_rows)

def _fill_aggregate_diagnostics(merged: pd.DataFrame, diag: pd.DataFrame) -> pd.DataFrame:
    """Mean/auto rows do not have separate diagnostic rows; use the matching
    pooled-all-devices diagnostic because those rows summarize the same selected
    across-device data.
    """
    if merged.empty or diag.empty:
        return merged
    diag_cols = [
        "normality_pass_p_ge_0_05",
        "shapiro_wilk_p_residuals",
        "levene_mean_p",
        "brown_forsythe_median_p",
        "brown_forsythe_p",
        "homogeneity_pass_p_ge_0_05",
    ]
    pooled = diag[
        (diag["scope"] == "pooled_all_devices") &
        (diag["device"].astype(str) == "ALL_SELECTED_DEVICES")
    ].copy()
    if pooled.empty:
        return merged
    pooled_lookup = {}
    for _, r in pooled.iterrows():
        pooled_lookup[(str(r["dataset"]), str(r["condition"]), str(r["analyte"]))] = r

    aggregate_scopes = {"mean_of_device_summaries", "overall_across_devices_auto"}
    for idx, row in merged.iterrows():
        if str(row.get("scope", "")) not in aggregate_scopes:
            continue
        pr = pooled_lookup.get((str(row["dataset"]), str(row["condition"]), str(row["analyte"])))
        if pr is None:
            continue
        for col in diag_cols:
            merged.at[idx, col] = pr.get(col, np.nan)
    return merged


def build_final_summary(
    summary: pd.DataFrame,
    diag: pd.DataFrame,
    decisions: pd.DataFrame,
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    """Create the streamlined first Excel sheet.

    The full standard and robust statistics are still computed internally.
    Only one center and one CV are exposed:
      Gcrit -> mean + standard repeatability CV
      Robust MAD/robust interval -> median + MAD robust CV
    """
    if summary.empty:
        return summary.copy()

    merge_keys = ["dataset", "condition", "analyte", "scope", "device"]
    merged = summary.merge(diag, on=merge_keys, how="left") if not diag.empty else summary.copy()
    merged = _fill_aggregate_diagnostics(merged, diag)

    decision_lookup = {}
    if not decisions.empty:
        for _, r in decisions.iterrows():
            decision_lookup[(str(r["condition"]), str(r["analyte"]), str(r["decision_device"]))] = r.to_dict()

    out_rows = []
    for _, r in merged.iterrows():
        device = str(r.get("device", ""))
        scope = str(r.get("scope", ""))
        decision_device = device if scope == "per_device" else "ALL_SELECTED_DEVICES"
        dec = decision_lookup.get((str(r["condition"]), str(r["analyte"]), decision_device), {})
        method = dec.get("outlier_method_selected", cfg.outlier_method)

        if method.startswith("Gcrit"):
            center_name = "mean"
            center_value = r.get("mean", np.nan)
            cv_name = "CV%"
            cv_value = r.get("cv_repeat_pct", np.nan)
        elif method == "None":
            # With no outlier rule, use Shapiro on the displayed dataset to pick
            # the appropriate descriptive family without altering the data.
            normality = r.get("normality_pass_p_ge_0_05", np.nan)
            if pd.notna(normality) and bool(normality):
                center_name, center_value = "mean", r.get("mean", np.nan)
                cv_name, cv_value = "CV%", r.get("cv_repeat_pct", np.nan)
            else:
                center_name, center_value = "median", r.get("median", np.nan)
                cv_name, cv_value = "Robust CV% (MAD)", r.get("robust_cv_mad_pct", np.nan)
        else:
            center_name = "median"
            center_value = r.get("median", np.nan)
            cv_name = "Robust CV% (MAD)"
            cv_value = r.get("robust_cv_mad_pct", np.nan)

        # Display the RAW pooled Shapiro result that actually drove the automatic
        # method choice, so the normal/not-normal label and selected method are
        # always internally consistent in both raw and cleaned rows.
        selection_p = dec.get("shapiro_wilk_p_residuals_raw", r.get("shapiro_wilk_p_residuals", np.nan))
        selection_pass = dec.get("normality_pass_raw", r.get("normality_pass_p_ge_0_05", np.nan))
        _, distribution = _normality_status_from_p(selection_p)
        row = {
            "dataset": r.get("dataset", ""),
            "condition": r.get("condition", ""),
            "analyte": r.get("analyte", ""),
            "device": r.get("device", ""),
            "n_rows": r.get("n_rows", np.nan),
            "n_cells": r.get("n_cells", np.nan),
            "sd_repeat": r.get("sd_repeat", np.nan),
            "mean": r.get("mean", np.nan),
            "median": r.get("median", np.nan),
            "distribution": distribution,
            "outlier_method_selected": method,
            "center_statistic": center_name,
            "center_value": center_value,
            "cv_statistic": cv_name,
            "cv_pct": cv_value,
            "normality_pass_p_ge_0_05": selection_pass,
            "shapiro_wilk_p_residuals": selection_p,
            "levene_mean_p": r.get("levene_mean_p", np.nan),
            "brown_forsythe_median_p": r.get("brown_forsythe_median_p", np.nan),
            "brown_forsythe_p": r.get("brown_forsythe_p", np.nan),
            "homogeneity_pass_p_ge_0_05": r.get("homogeneity_pass_p_ge_0_05", np.nan),
            "aggregation_method": r.get("aggregation_method", ""),
        }
        if cfg.make_bootstrap_ci:
            # The legacy bootstrap is a standard-CV bootstrap. Only expose it
            # when the selected reported CV is the standard CV.
            if cv_name == "CV%":
                row["cv_95ci_low"] = r.get("cv_repeat_ci_low", np.nan)
                row["cv_95ci_high"] = r.get("cv_repeat_ci_high", np.nan)
            else:
                row["cv_95ci_low"] = np.nan
                row["cv_95ci_high"] = np.nan
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def make_excel_output(
    final_summary: pd.DataFrame,
    outlier_log: pd.DataFrame,
    cfg: AnalysisConfig,
    decisions: pd.DataFrame,
) -> bytes:
    """Return ONE combined Excel workbook with exactly three sheets:
    summary_raw_cleaned, outliers, settings.
    """
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        final_summary.to_excel(writer, sheet_name="summary_raw_cleaned", index=False)

        if outlier_log.empty:
            outlier_cols = [
                "condition", "device", "sample_number", "analyte", "batch_id",
                "bloodSampleId", "deviceId", "value", "normality_status_raw",
                "shapiro_wilk_p_residuals_raw", "outlier_method", "removed_order",
                "direction", "outlier_metric", "outlier_threshold", "details",
                "gcrit_mode", "manual_gcrit", "gcrit_alpha", "gcrit_tail",
                "mad_zcrit", "robust_interval_z",
            ]
            pd.DataFrame(columns=outlier_cols).to_excel(writer, sheet_name="outliers", index=False)
        else:
            outlier_log.to_excel(writer, sheet_name="outliers", index=False)

        auto_note = (
            "Shapiro-Wilk on RAW pooled sample x device residuals for each condition/analyte; "
            "p >= 0.05 uses Gcrit, p < 0.05 uses Robust MAD; unavailable Shapiro uses Robust MAD fallback."
            if cfg.outlier_method == AUTO_OUTLIER_METHOD else
            f"Manual override: {cfg.outlier_method}"
        )
        settings = pd.DataFrame({
            "setting": [
                "device_mode", "outlier_method", "automatic_selection_rule",
                "normality_alpha", "max_outliers_per_group", "gcrit_mode",
                "manual_gcrit", "gcrit_alpha", "gcrit_tail", "mad_zcrit",
                "robust_interval_z", "bootstrap_ci", "n_boot", "random_seed",
                "devices", "analytes", "conditions",
            ],
            "value": [
                cfg.device_mode, cfg.outlier_method, auto_note,
                0.05, cfg.max_outliers_per_group, cfg.gcrit_mode,
                cfg.gcrit, cfg.gcrit_alpha, cfg.gcrit_tail, cfg.mad_zcrit,
                cfg.robust_interval_z, cfg.make_bootstrap_ci, cfg.n_boot, cfg.random_seed,
                ", ".join(map(str, cfg.devices)),
                ", ".join(cfg.analytes),
                "; ".join([f"{k}: {','.join(v)}" for k, v in cfg.condition_map.items()]),
            ],
        })
        settings.to_excel(writer, sheet_name="settings", index=False)

        # Light formatting for a clean single-workbook deliverable.
        from openpyxl.styles import Font, PatternFill, Alignment
        wb = writer.book
        header_fill = PatternFill("solid", fgColor="D9EAD3")
        header_font = Font(bold=True)
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            if ws.max_row >= 1 and ws.max_column >= 1:
                ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(vertical="center")
            for col_cells in ws.columns:
                letter = col_cells[0].column_letter
                max_len = 0
                for cell in col_cells[: min(ws.max_row, 250)]:
                    val = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, len(val))
                ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 42)

        # Readable precision without changing stored numeric values.
        ws = wb["summary_raw_cleaned"]
        header_map = {c.value: c.column for c in ws[1]}
        for name in [
            "sd_repeat", "mean", "median", "center_value", "cv_pct", "shapiro_wilk_p_residuals", "levene_mean_p",
            "brown_forsythe_median_p", "brown_forsythe_p", "cv_95ci_low", "cv_95ci_high",
        ]:
            if name in header_map:
                for row in range(2, ws.max_row + 1):
                    ws.cell(row=row, column=header_map[name]).number_format = "0.0000"

    return excel_buf.getvalue()


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Imprecision Short Repeatability App", layout="wide")
st.title("Imprecision Short Repeatability App")
st.caption("Upload Excel/CSV → detect samples/devices → define conditions → choose analytes → repeatability SD/CV per device and pooled across devices.")

with st.expander("What this app calculates", expanded=False):
    st.markdown(
        r"""
**Repeatability per device** is calculated from replicate scatter within each selected sample on the same device.

For each condition × analyte × device:

\[
SD_{repeat} = \\sqrt{\\frac{\\sum_s\\sum_r(y_{sr}-\\bar{y}_s)^2}{\\sum_s(n_s-1)}}
\]

\[
CV_{repeat}\\% = 100 \\times \\frac{SD_{repeat}}{\\bar{y}}
\]

**Pooled across devices** uses the same formula but treats each sample × device as a replicate cell.

The app reports raw results and cleaned results after optional outlier removal.
"""
    )

uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded, engine="openpyxl")
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.subheader("1) Preview uploaded data")
st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
st.dataframe(df.head(25), use_container_width=True)

cols = list(df.columns)
batch_guess = guess_col(cols, "batch") or cols[0]
sample_guess = guess_col(cols, "sample") or cols[0]
device_guess = guess_col(cols, "device") or cols[0]
condition_guess = guess_col(cols, "condition")

st.subheader("2) Confirm ID columns")
c1, c2, c3 = st.columns(3)
with c1:
    batch_col = st.selectbox("Batch ID column", options=cols, index=cols.index(batch_guess))
with c2:
    sample_col = st.selectbox("Blood sample ID column", options=cols, index=cols.index(sample_guess))
with c3:
    device_col = st.selectbox("Device ID column", options=cols, index=cols.index(device_guess))

working_preview = df.copy()
working_preview["_sample_number"] = working_preview[sample_col].apply(extract_sample_number).astype(str)
working_preview["_device"] = working_preview[device_col].astype(str)

st.markdown("**Detected sample numbers from bloodSampleId after `IS-`:**")
sample_counts = (
    working_preview.groupby("_sample_number")
    .agg(n_rows=(batch_col, "size"), devices=("_device", lambda x: ", ".join(sorted(x.astype(str).unique()))))
    .reset_index()
    .sort_values("_sample_number")
)
st.dataframe(sample_counts, use_container_width=True)

st.markdown("**Replicate counts by sample × device:**")
rep_table = pd.crosstab(working_preview["_sample_number"], working_preview["_device"])
st.dataframe(rep_table, use_container_width=True)

st.subheader("3) Define conditions")
condition_mode = st.radio(
    "How should conditions be defined?",
    options=["Manual: assign sample numbers to conditions", "Use an existing condition/level column"],
    index=0 if condition_guess is None else 1,
)

condition_map: Dict[str, List[str]] = {}
all_samples = sorted(working_preview["_sample_number"].dropna().astype(str).unique().tolist())

if condition_mode == "Use an existing condition/level column":
    if condition_guess is None:
        condition_guess = cols[0]
    cond_col = st.selectbox("Condition column", options=cols, index=cols.index(condition_guess))
    temp = working_preview[[cond_col, "_sample_number"]].dropna()
    observed_conditions = sorted(temp[cond_col].astype(str).unique().tolist())
    chosen_conditions = st.multiselect("Conditions to analyze", observed_conditions, default=observed_conditions)
    for cond in chosen_conditions:
        condition_map[str(cond)] = sorted(temp.loc[temp[cond_col].astype(str) == str(cond), "_sample_number"].astype(str).unique().tolist())
else:
    n_conditions = st.number_input("Number of conditions", min_value=1, max_value=20, value=4, step=1)
    default_names = ["Anemic", "Normal", "Low WBC", "Low PLT"]
    used_samples = set()
    for i in range(int(n_conditions)):
        cols_condition = st.columns([1, 3])
        with cols_condition[0]:
            default_name = default_names[i] if i < len(default_names) else f"Condition {i+1}"
            cond_name = st.text_input(f"Condition {i+1} name", value=default_name, key=f"cond_name_{i}")
        with cols_condition[1]:
            remaining_default = [s for s in all_samples if s not in used_samples]
            selected = st.multiselect(
                f"Sample numbers for {cond_name}",
                options=all_samples,
                default=[],
                key=f"cond_samples_{i}",
            )
        if cond_name.strip():
            condition_map[cond_name.strip()] = [str(s) for s in selected]
            used_samples.update(selected)

# Show condition map
condition_map = {k: v for k, v in condition_map.items() if len(v) > 0}
if len(condition_map) == 0:
    st.warning("Assign at least one sample number to at least one condition.")
    st.stop()

st.markdown("**Condition/sample mapping to be analyzed:**")
mapping_rows = []
for cond, samples in condition_map.items():
    mapping_rows.append({"condition": cond, "n_samples": len(samples), "samples": ", ".join(samples)})
st.dataframe(pd.DataFrame(mapping_rows), use_container_width=True)

st.subheader("4) Select devices and analytes")

devices_all = sorted(working_preview["_device"].dropna().astype(str).unique().tolist())
# Default: devices with decent row counts
device_counts = working_preview["_device"].value_counts()
default_devices = [str(d) for d in device_counts[device_counts >= max(3, int(0.05 * len(working_preview)))].index.tolist()]
default_devices = sorted(default_devices) if len(default_devices) else devices_all

device_mode = st.selectbox("Device handling", ["Pool all devices", "Analyze each device separately + pooled"], index=0)
devices = st.multiselect("Devices to include", options=devices_all, default=default_devices)

reserved = {batch_col, sample_col, device_col, "_sample_number", "_device"}
numeric_candidates = []
for c in cols:
    if c in reserved:
        continue
    as_num = pd.to_numeric(df[c], errors="coerce")
    if as_num.notna().sum() >= max(3, int(0.1 * len(df))):
        numeric_candidates.append(c)

# Put the requested analytes first in both the selector and the selected tags.
# Requested default analytes are kept even when they are sparse and therefore
# would not pass the generic >=10% numeric-candidate threshold above.
requested_present = [a for a in DEFAULT_ANALYTE_ORDER if a in cols]
ordered_first = requested_present
numeric_candidates = ordered_first + [c for c in numeric_candidates if c not in ordered_first]
default_analytes = ordered_first.copy()
if not default_analytes:
    default_analytes = numeric_candidates[:8]

analytes = st.multiselect("Analyte columns to analyze", options=numeric_candidates, default=default_analytes)

st.subheader("5) Outliers and confidence intervals")
outlier_method = st.selectbox(
    "Outlier detection method",
    options=[
        AUTO_OUTLIER_METHOD,
        "None",
        GCRIT_OUTLIER_METHOD,
        MAD_OUTLIER_METHOD,
        "95% robust interval: remove most extreme outside median ± z*MAD_SD",
    ],
    index=0,
    help="Automatic mode uses raw Shapiro-Wilk residual normality to choose the existing Gcrit or Robust MAD rule.",
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    max_outliers_per_group = st.selectbox("Max outliers to remove per condition/device/sample/analyte", [0, 1, 2], index=1)
with c2:
    gcrit_mode = st.selectbox("Gcrit mode", ["Manual Gcrit value", "Automatic from n, alpha, and tail"], index=0)
with c3:
    gcrit = st.number_input("Manual Gcrit value", min_value=0.0, value=3.135, step=0.001, format="%.3f")
with c4:
    gcrit_alpha = st.number_input("Automatic Gcrit alpha", min_value=0.0001, max_value=0.2, value=0.01, step=0.001, format="%.4f")

c1, c2, c3, c4 = st.columns(4)
with c1:
    gcrit_tail = st.selectbox("Automatic Gcrit tail", ["Two-sided", "One-sided"], index=0)
with c2:
    mad_zcrit = st.number_input("MAD modified-z threshold", min_value=0.1, value=3.5, step=0.1, format="%.1f")
with c3:
    robust_interval_z = st.number_input("Robust interval z", min_value=0.5, value=1.96, step=0.01)
with c4:
    make_bootstrap_ci = st.checkbox("Report 95% bootstrap CIs", value=False)

n_boot = st.slider("Bootstrap resamples", min_value=100, max_value=20000, value=2000, step=100, disabled=not make_bootstrap_ci)
random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

st.markdown("""
**Outlier outputs:** raw results are always reported; cleaned results remove flagged outliers using the selected rule.  
**Automatic mode:** for each condition × analyte, Shapiro-Wilk is evaluated on raw pooled sample × device residuals across the selected devices. Normal data (`p ≥ 0.05`) use the existing Gcrit rule; non-normal data (`p < 0.05`) use the existing Robust MAD rule. The chosen rule is then applied consistently to that analyte on every selected device. If Shapiro-Wilk cannot be evaluated, Robust MAD is used as a conservative fallback.  
**Gcrit automatic mode:** recalculates Grubbs' critical value from the current replicate-cell size `n`, alpha, and one-/two-sided setting during sequential removal.
""")

st.subheader("6) Run")
run = st.button("Run imprecision short analysis", type="primary")

if run:
    if len(devices) == 0:
        st.error("Select at least one device.")
        st.stop()
    if len(analytes) == 0:
        st.error("Select at least one analyte.")
        st.stop()

    cfg = AnalysisConfig(
        batch_col=batch_col,
        sample_col=sample_col,
        device_col=device_col,
        condition_map=condition_map,
        analytes=analytes,
        devices=devices,
        device_mode=device_mode,
        outlier_method=outlier_method,
        max_outliers_per_group=int(max_outliers_per_group),
        gcrit_mode=gcrit_mode,
        gcrit=float(gcrit),
        gcrit_alpha=float(gcrit_alpha),
        gcrit_tail=gcrit_tail,
        mad_zcrit=float(mad_zcrit),
        robust_interval_z=float(robust_interval_z),
        make_bootstrap_ci=bool(make_bootstrap_ci),
        n_boot=int(n_boot),
        random_seed=int(random_seed),
    )

    with st.spinner("Preparing data and applying outlier flags..."):
        work = make_long_working_df(df, cfg)
        flagged, outlier_log, outlier_decisions = apply_outlier_flags(work, cfg)

    if work.empty:
        st.error("No rows remained after condition/device selection. Check your sample-condition mapping and selected devices.")
        st.stop()

    with st.spinner("Computing repeatability summaries, diagnostics, and bootstrap CIs..."):
        raw_summary, raw_diag = summarize_analysis(flagged, cfg, cleaned=False)
        clean_summary, clean_diag = summarize_analysis(flagged, cfg, cleaned=True)

    raw_final = build_final_summary(raw_summary, raw_diag, outlier_decisions, cfg)
    clean_final = build_final_summary(clean_summary, clean_diag, outlier_decisions, cfg)
    final_summary = pd.concat([raw_final, clean_final], ignore_index=True)

    st.success("Analysis complete.")

    st.markdown("### Summary: raw and automatically cleaned")
    st.caption("The workbook reports one center and one CV only: mean + CV for Gcrit, median + robust MAD CV for robust-MAD cleaning.")
    st.dataframe(final_summary, use_container_width=True)

    st.markdown("### Outliers removed log")
    if outlier_log.empty:
        st.info("No outliers were flagged by the selected/automatic method.")
    else:
        st.dataframe(outlier_log, use_container_width=True)

    st.markdown("### Quick interpretation")
    st.markdown(
        """
- `raw_no_outlier_removal` = the original selected observations before outlier removal.
- `cleaned_outliers_removed` = the same analysis after the automatically selected Gcrit/MAD rule removes flagged observations.
- `distribution` comes from Shapiro-Wilk on within-cell residuals for that displayed dataset.
- Automatic outlier choice is made once per condition × analyte from **raw pooled sample × device residual normality**: normal → Gcrit; non-normal → Robust MAD.
- `center_value` is therefore a mean for Gcrit reporting and a median for Robust MAD reporting.
- `cv_pct` is therefore standard pooled repeatability CV% for Gcrit reporting and robust MAD CV% for Robust MAD reporting.
- `aggregation_method` retains the existing per-device/pooled calculation definition.
"""
    )

    st.markdown("### Download")
    excel_bytes = make_excel_output(final_summary, outlier_log, cfg, outlier_decisions)
    st.download_button(
        label="Download combined Excel results",
        data=excel_bytes,
        file_name="imprecision_short_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

