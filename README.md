# Proxima Short Term — V8

## Upload to GitHub and run on Streamlit

1. Extract this ZIP on your computer.
2. Upload **app.py** and **requirements.txt** from this folder into the root of
   the GitHub repository for this app, replacing the older files. Upload the
   extracted files, not the ZIP itself.
3. In Streamlit Community Cloud, select that repository and set the main file
   path to **app.py**. Select **Python 3.12** under Advanced settings for a new app.
4. Deploy, or reboot the existing Streamlit app after committing both files.

Each app has its own app.py. Use one repository per app, or place each app and
its requirements.txt together in a separate subfolder and point Streamlit at
that subfolder's app.py. Do not overwrite one app with another in the same folder.

No validation_io.py or analysis_core.py upload is needed: their code is bundled
inside app.py. The validated numerical functions are preserved from V7.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Python 3.12 is the tested runtime. Dependencies are pinned in requirements.txt.
See the accompanying V8 validation bundle for test results and known report
limitations. Local validation does not certify a particular cloud account's
deployment or permissions.

Streamlit documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
