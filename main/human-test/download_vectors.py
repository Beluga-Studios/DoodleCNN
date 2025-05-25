import os
import sys
import json
import random
import urllib.request
from pathlib import Path

# Set how many drawings to download from the command line
if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    print("Usage: python download_vectors.py <number_of_drawings>")
    sys.exit(1)

NUM_DRAWINGS = int(sys.argv[1])
print(f"✅ Downloading {NUM_DRAWINGS} drawings per category...")

# Categories you want to download
def load_categories(file_path="../categories.txt"):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

categories = load_categories()


# Destination folder
Path("images").mkdir(exist_ok=True)

for category in categories:
    cat_filename = category.replace(" ", "%20")
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/raw/{cat_filename}.ndjson"
    dest_file = Path(f"vectors/{category}.ndjson")

    try:
        print(f"Downloading {category}...")
        with urllib.request.urlopen(url) as response:
            lines = response.read().decode("utf-8").splitlines()
            random.shuffle(lines)
            selected = lines[:NUM_DRAWINGS]

            with open(dest_file, "w") as out_file:
                for line in selected:
                    out_file.write(line + "\n")

        print(f"✔️  Saved {len(selected)} drawings to {dest_file}")

    except Exception as e:
        print(f"❌ Failed to download {category}: {e}")
