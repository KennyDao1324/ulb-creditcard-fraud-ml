from __future__ import annotations

import shutil
import ssl
import urllib.request
from pathlib import Path

import certifi

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "raw" / "creditcard.csv"


if __name__ == "__main__":
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(URL, context=context) as response, DESTINATION.open("wb") as output:
        shutil.copyfileobj(response, output)
    print(f"Downloaded {DESTINATION}")
