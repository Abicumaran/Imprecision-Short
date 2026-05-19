# Imprecision Short Repeatability App

This Streamlit app analyzes short imprecision / repeatability datasets with columns such as:

- `batch_id`
- `bloodSampleId`
- `deviceId`
- analyte columns such as `WBC`, `RBC`, `HGB`, `HCT`, `PLT`, `NEUT`, `LYMPH`, etc.

It detects sample numbers from `bloodSampleId` using the pattern after `IS-`, for example:

`APT-160425D02-IS-1` -> sample number `1`

## Computations:

For each selected condition × analyte × device:

- Repeatability SD
- Repeatability CV%
- Robust MAD-based SD/CV
- Shapiro-Wilk normality test on within-sample residuals
- Classic Levene homogeneity test using mean-centering
- Brown-Forsythe homogeneity test using median-centering, may be more robust
- Optional 95% bootstrap confidence intervals
- Outlier log with batch ID, sample ID, device, analyte, and value

It also calculates across-device repeatability in three ways:

- `pooled_all_devices`: pooled sample × device repeatability, valid for equal or unequal replicate counts.
- `mean_of_device_summaries`: simple average of device-level SD/CV summaries.
- `overall_across_devices_auto`: final overall row. If replicate counts are equal across devices, this uses the average of device summaries; if replicate counts are unequal, it uses the pooled sample × device estimate.

## Diagnostics columns

The downloaded diagnostics file includes:

- `shapiro_wilk_p_residuals`
- `shapiro_wilk_normality_check`
- `levene_mean_p`
- `levene_mean_homogeneity_check`
- `brown_forsythe_median_p`
- `brown_forsythe_homogeneity_check`

A p-value below 0.05 is marked as a failure of that assumption check, but the app still reports robust MAD CV and bootstrap CI so the analysis does not depend only on normality/homogeneity assumptions. The user can customize to their data context and data patterns. 

## Downloads

The app provides individual CSV download buttons for the user:

- summary CSV
- diagnostics CSV
- outlier log CSV
- working data CSV

It also provides one all-results ZIP containing all CSVs plus an Excel workbook.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or:

```bash
python run_app.py
```


## ZIP outputs

The ZIP contains:

- `imprecision_short_summary_raw_and_cleaned.csv`
- `imprecision_short_summary_raw_only.csv`
- `imprecision_short_summary_cleaned_only.csv`
- `diagnostics_normality_homogeneity.csv`
- `diagnostics_raw_only.csv`
- `diagnostics_cleaned_only.csv`
- `outliers_removed_log.csv`
- `working_data_with_sample_condition_outlier_flags.csv`
- `imprecision_short_results.xlsx`
- `README_output_interpretation.txt`
