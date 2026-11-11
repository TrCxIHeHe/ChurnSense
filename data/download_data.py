"""Download the IBM Telco Customer Churn dataset.

Usage (from project root):  python data/download_data.py

Source: IBM Cloud Pak for Data sample dataset
  https://github.com/IBM/telco-customer-churn-on-icp4d
  Raw CSV: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

If you are offline, download the CSV manually from the URL above and save it as
data/Telco-Customer-Churn.csv. No synthetic data is ever substituted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data_loader import download_dataset, load_raw_data  # noqa: E402

if __name__ == "__main__":
    config.setup_logging()
    try:
        download_dataset()
        df = load_raw_data(auto_download=False)
        print(f"OK: {df.shape[0]} rows x {df.shape[1]} columns -> {config.DATA_FILE}")
    except Exception as exc:
        print(f"Download failed: {exc}\nManual download: {config.DATA_SOURCE_URL}")
        sys.exit(1)
