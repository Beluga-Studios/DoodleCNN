import os
import urllib.request
import numpy as np
from tqdm import tqdm

# Your categories here
def load_categories(file_path="../categories.txt"):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

categories = load_categories()

DATA_DIR = "npy_data"
os.makedirs(DATA_DIR, exist_ok=True)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def is_file_valid(filepath):
    try:
        data = np.load(filepath)
        if len(data.shape) == 2 and data.shape[1] == 784:
            return True
        else:
            print(f"❌ Invalid shape for {filepath}: {data.shape}")
            return False
    except Exception as e:
        print(f"❌ Could not load {filepath}: {e}")
        return False

def download_file(url, dest, category):
    print(f"⬇️ Downloading: {category}...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=category) as pbar:
            urllib.request.urlretrieve(url, dest, reporthook=pbar.update_to)
        print(f"✅ Done: {category}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {category}: {e}")
        return False

# 🔁 Main loop
total = len(categories)
kept = 0
re_downloaded = 0
failed = 0

for category in categories:
    filename = category.replace(" ", "%20") + ".npy"
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{filename}"
    dest = os.path.join(DATA_DIR, f"{category}.npy")

    if os.path.exists(dest):
        if is_file_valid(dest):
            print(f"✅ Valid file already exists: {category}")
            kept += 1
        else:
            print(f"⚠️ Corrupted file found, re-downloading: {category}")
            if download_file(url, dest, category):
                re_downloaded += 1
            else:
                failed += 1
    else:
        if download_file(url, dest, category):
            re_downloaded += 1
        else:
            failed += 1

    # ✅ Summary
    total_done = kept + re_downloaded
    percent_done = (total_done / total) * 100
    print(f"📊 Total completed: {total_done}/{total} ({percent_done:.1f}%)")

print("\n📦 Download & Validation Summary")
print(f"✅ Valid files kept: {kept}")
print(f"🔄 Files downloaded/re-downloaded: {re_downloaded}")
print(f"❌ Failed downloads: {failed}")
print(f"📊 Total completed: {total_done}/{total} ({percent_done:.1f}%)")