
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


@dataclass
class AnalysisConfig:
    batch_col: str
    sample_col: str
    device_col: str
    condition_map: Dict[str, List[str]]
    analytes: List[str]
    devices: List[str]
    outlier_method: str
    gcrit: float
    mad_zcrit: float
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


def flag_outliers_one_group(
    vals: pd.Series,
    method: str,
    gcrit: float = 3.135,
    mad_zcrit: float = 3.5,
) -> pd.Series:
    """
    Flags outliers within one analyte-condition-device-sample replicate cell.
    The methods intentionally act within small replicate groups.
    """
    x = vals.astype(float)
    idx = x.index
    finite = x[np.isfinite(x)]
    flags = pd.Series(False, index=idx)

    if len(finite) < 3 or method == "None":
        return flags

    if method == "Gcrit / Grubbs-like":
        mu = finite.mean()
        sd = finite.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return flags
        g = (x - mu).abs() / sd
        flags = g >= gcrit
        return flags.fillna(False)

    if method == "Robust MAD modified-z":
        med = finite.median()
        mad = np.median(np.abs(finite - med))
        if not np.isfinite(mad) or mad == 0:
            return flags
        modified_z = 0.6745 * (x - med).abs() / mad
        flags = modified_z > mad_zcrit
        return flags.fillna(False)

    if method == "95% robust interval; remove most extreme only":
        med = finite.median()
        rsd = robust_sd_mad(finite.values)
        if not np.isfinite(rsd) or rsd == 0:
            return flags
        lo = med - 1.96 * rsd
        hi = med + 1.96 * rsd
        outside = (x < lo) | (x > hi)
        outside = outside.fillna(False)
        if outside.sum() == 0:
            return flags
        # remove only the one most extreme value by robust z distance
        distances = (x - med).abs() / rsd
        candidate_idx = distances[outside].idxmax()
        flags.loc[candidate_idx] = True
        return flags

    return flags


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


def apply_outlier_flags(work: pd.DataFrame, cfg: AnalysisConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns a dataframe with one boolean outlier column per analyte and an outlier log.
    Outliers are detected within condition-device-sample replicate cells, separately by analyte.
    """
    df = work.copy()
    logs = []

    for analyte in cfg.analytes:
        flag_col = f"_outlier_{analyte}"
        df[flag_col] = False

        group_cols = ["_condition", "_device", "_sample_number"]
        for keys, sub in df.groupby(group_cols, dropna=False):
            vals = sub[analyte]
            flags = flag_outliers_one_group(vals, cfg.outlier_method, cfg.gcrit, cfg.mad_zcrit)
            df.loc[flags.index, flag_col] = flags.values

            if flags.sum() > 0:
                condition, device, sample_number = keys
                for row_idx in flags[flags].index:
                    row = df.loc[row_idx]
                    logs.append({
                        "condition": condition,
                        "device": device,
                        "sample_number": sample_number,
                        "analyte": analyte,
                        "batch_id": row["_batch_id"],
                        "bloodSampleId": row[cfg.sample_col],
                        "deviceId": row[cfg.device_col],
                        "value": row[analyte],
                        "outlier_method": cfg.outlier_method,
                        "gcrit": cfg.gcrit if cfg.outlier_method == "Gcrit / Grubbs-like" else np.nan,
                        "mad_zcrit": cfg.mad_zcrit if cfg.outlier_method == "Robust MAD modified-z" else np.nan,
                    })

    return df, pd.DataFrame(logs)


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

                # User-facing final across-device row:
                # - If replicate/cell counts are equal across devices, report the simple average of device SD/CV.
                # - If not equal, report the pooled sample x device estimate, which handles imbalance correctly.
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


def make_zip_outputs(
    raw_summary: pd.DataFrame,
    clean_summary: pd.DataFrame,
    raw_diag: pd.DataFrame,
    clean_diag: pd.DataFrame,
    outlier_log: pd.DataFrame,
    working_data: pd.DataFrame,
    cfg: AnalysisConfig,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        all_summary = pd.concat([raw_summary, clean_summary], ignore_index=True)
        all_diag = pd.concat([raw_diag, clean_diag], ignore_index=True)

        zf.writestr("imprecision_short_summary_raw_and_cleaned.csv", all_summary.to_csv(index=False).encode("utf-8"))
        zf.writestr("imprecision_short_summary_raw_only.csv", raw_summary.to_csv(index=False).encode("utf-8"))
        zf.writestr("imprecision_short_summary_cleaned_only.csv", clean_summary.to_csv(index=False).encode("utf-8"))
        zf.writestr("diagnostics_normality_homogeneity.csv", all_diag.to_csv(index=False).encode("utf-8"))
        zf.writestr("diagnostics_raw_only.csv", raw_diag.to_csv(index=False).encode("utf-8"))
        zf.writestr("diagnostics_cleaned_only.csv", clean_diag.to_csv(index=False).encode("utf-8"))
        zf.writestr("outliers_removed_log.csv", outlier_log.to_csv(index=False).encode("utf-8"))
        zf.writestr("working_data_with_sample_condition_outlier_flags.csv", working_data.to_csv(index=False).encode("utf-8"))

        # Excel workbook too
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            all_summary.to_excel(writer, sheet_name="summary_raw_cleaned", index=False)
            all_diag.to_excel(writer, sheet_name="diagnostics", index=False)
            outlier_log.to_excel(writer, sheet_name="outliers", index=False)
            pd.DataFrame({
                "setting": ["outlier_method", "gcrit", "mad_zcrit", "bootstrap_ci", "n_boot", "devices", "analytes", "conditions"],
                "value": [
                    cfg.outlier_method,
                    cfg.gcrit,
                    cfg.mad_zcrit,
                    cfg.make_bootstrap_ci,
                    cfg.n_boot,
                    ", ".join(map(str, cfg.devices)),
                    ", ".join(cfg.analytes),
                    "; ".join([f"{k}: {','.join(v)}" for k, v in cfg.condition_map.items()]),
                ],
            }).to_excel(writer, sheet_name="settings", index=False)
        zf.writestr("imprecision_short_results.xlsx", excel_buf.getvalue())

        readme = f"""
Imprecision Short Repeatability Outputs

Main formula used:
For each condition x analyte x device, replicate cells are sample numbers.
Within each sample-device cell:
  SS_repeat = sum((y_i - mean_cell)^2)

Pooled repeatability variance:
  sigma_repeat^2 = sum(SS_repeat) / sum(n_cell - 1)

Repeatability SD:
  SD_repeat = sqrt(sigma_repeat^2)

Repeatability CV%:
  CV_repeat% = 100 * SD_repeat / grand_mean

Pooled all devices:
  Same formula, but cells are sample x device, so device-specific replicate scatter is pooled.

Outputs:
- imprecision_short_summary_raw_and_cleaned.csv
- diagnostics_normality_homogeneity.csv, including Shapiro-Wilk, classic Levene, and Brown-Forsythe tests
- outliers_removed_log.csv
- working_data_with_sample_condition_outlier_flags.csv
- imprecision_short_results.xlsx

Outlier mode:
  {cfg.outlier_method}

Bootstrap CIs:
  {cfg.make_bootstrap_ci}, n_boot={cfg.n_boot}
"""
        zf.writestr("README_output_interpretation.txt", readme.encode("utf-8"))
    return buf.getvalue()


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

devices = st.multiselect("Devices to include", options=devices_all, default=default_devices)

reserved = {batch_col, sample_col, device_col, "_sample_number", "_device"}
numeric_candidates = []
for c in cols:
    if c in reserved:
        continue
    as_num = pd.to_numeric(df[c], errors="coerce")
    if as_num.notna().sum() >= max(3, int(0.1 * len(df))):
        numeric_candidates.append(c)

default_analytes = [a for a in ["WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "NEUT", "LYMPH", "MONO", "EOS", "BASO"] if a in numeric_candidates]
if not default_analytes:
    default_analytes = numeric_candidates[:8]

analytes = st.multiselect("Analyte columns to analyze", options=numeric_candidates, default=default_analytes)

st.subheader("5) Outliers and confidence intervals")
c1, c2, c3, c4 = st.columns(4)
with c1:
    outlier_method = st.selectbox(
        "Outlier detection method",
        options=["None", "Gcrit / Grubbs-like", "Robust MAD modified-z", "95% robust interval; remove most extreme only"],
        index=0,
    )
with c2:
    gcrit = st.number_input("Gcrit value", min_value=0.0, value=3.135, step=0.001, format="%.3f")
with c3:
    mad_zcrit = st.number_input("MAD modified-z threshold", min_value=0.0, value=3.5, step=0.1, format="%.1f")
with c4:
    make_bootstrap_ci = st.checkbox("Report 95% bootstrap CIs", value=True)

n_boot = st.slider("Bootstrap resamples", min_value=100, max_value=5000, value=1000, step=100, disabled=not make_bootstrap_ci)
random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

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
        outlier_method=outlier_method,
        gcrit=float(gcrit),
        mad_zcrit=float(mad_zcrit),
        make_bootstrap_ci=bool(make_bootstrap_ci),
        n_boot=int(n_boot),
        random_seed=int(random_seed),
    )

    with st.spinner("Preparing data and applying outlier flags..."):
        work = make_long_working_df(df, cfg)
        flagged, outlier_log = apply_outlier_flags(work, cfg)

    if work.empty:
        st.error("No rows remained after condition/device selection. Check your sample-condition mapping and selected devices.")
        st.stop()

    with st.spinner("Computing repeatability summaries, diagnostics, and bootstrap CIs..."):
        raw_summary, raw_diag = summarize_analysis(flagged, cfg, cleaned=False)
        clean_summary, clean_diag = summarize_analysis(flagged, cfg, cleaned=True)

    all_summary = pd.concat([raw_summary, clean_summary], ignore_index=True)
    all_diag = pd.concat([raw_diag, clean_diag], ignore_index=True)

    st.success("Analysis complete.")

    st.markdown("### Summary: raw and cleaned")
    st.dataframe(all_summary, use_container_width=True)

    st.markdown("### Normality and homogeneity diagnostics")
    st.dataframe(all_diag, use_container_width=True)

    st.markdown("### Outliers removed log")
    if outlier_log.empty:
        st.info("No outliers were flagged by the selected method.")
    else:
        st.dataframe(outlier_log, use_container_width=True)

    st.markdown("### Quick interpretation")
    st.markdown(
        """
- `per_device` = repeatability across selected samples within one device.
- `pooled_all_devices` = repeatability pooled across all selected devices using sample × device cells.
- `mean_of_device_summaries` = simple average of device-level SD/CV summaries.
- `overall_across_devices_auto` = final across-device summary: average of device summaries when replicate counts are equal, otherwise pooled sample × device estimate.
- `raw_no_outlier_removal` = before removing flagged outliers.
- `cleaned_outliers_removed` = after removing analyte-specific flagged outliers.
- `shapiro_wilk_p_residuals < 0.05` suggests residual non-normality.
- `levene_mean_p < 0.05` suggests unequal variance by classic Levene test.
- `brown_forsythe_median_p < 0.05` suggests unequal variance by median-centered Levene/Brown-Forsythe, which is more robust.
"""
    )

    st.markdown("### Downloads")
    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        st.download_button(
            label="Download summary CSV",
            data=all_summary.to_csv(index=False).encode("utf-8"),
            file_name="imprecision_short_summary_raw_and_cleaned.csv",
            mime="text/csv",
        )
    with dc2:
        st.download_button(
            label="Download diagnostics CSV",
            data=all_diag.to_csv(index=False).encode("utf-8"),
            file_name="diagnostics_normality_homogeneity.csv",
            mime="text/csv",
        )
    with dc3:
        st.download_button(
            label="Download outlier log CSV",
            data=outlier_log.to_csv(index=False).encode("utf-8"),
            file_name="outliers_removed_log.csv",
            mime="text/csv",
        )
    with dc4:
        st.download_button(
            label="Download working data CSV",
            data=flagged.to_csv(index=False).encode("utf-8"),
            file_name="working_data_with_sample_condition_outlier_flags.csv",
            mime="text/csv",
        )

    zip_bytes = make_zip_outputs(raw_summary, clean_summary, raw_diag, clean_diag, outlier_log, flagged, cfg)
    st.download_button(
        label="Download ALL results ZIP",
        data=zip_bytes,
        file_name="imprecision_short_repeatability_results.zip",
        mime="application/zip",
    )
