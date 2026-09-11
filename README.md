# Imprecision Short Repeatability App — updated

Key UI/output updates only; the existing short-term statistical pipeline is retained.

- Global-flag column selector added under ID-column confirmation.
- `global_flag=TRUE` rows are excluded before statistical analysis, unless the explicit “treat all rows as FALSE” override is selected.
- Global-flag exclusions are exported separately from statistical outliers in the `global flag TRUE` worksheet.
- Manual-condition mode is the default, with 1 condition and a blank Condition 1 name.
- Automatic Gcrit calculation is the default.
- Automatic outlier selection remains Shapiro-Wilk -> Gcrit for normal residuals, Robust MAD for non-normal/not-testable residuals.
- One downloadable Excel workbook only, with sheets: `summary_raw_cleaned`, `outliers`, `global flag TRUE`, `settings`.
- Bootstrap 95% CIs remain off by default.

Run with:

```bash
pip install -r requirements.txt
streamlit run app.py
```
