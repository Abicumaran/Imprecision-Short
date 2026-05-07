import os
import sys
import subprocess

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(here, "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless=true"]
    subprocess.Popen(cmd)

if __name__ == "__main__":
    main()
