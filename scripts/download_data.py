# la tuyauterie de donnees
# on ramene les datas
import urllib.request
import json
import os
import sys

manifest_path = os.path.join("data", "databento_downloads", "manifest.json")
with open(manifest_path) as f:
    manifest = json.load(f)

# Find smallest .dbn.zst file
dbn_files = [f for f in manifest["files"] if f["filename"].endswith(".dbn.zst")]
dbn_files.sort(key=lambda x: x["size"])

smallest = dbn_files[0]
print(f"Smallest: {smallest['filename']} ({smallest['size']/1e6:.1f} MB)")

dest = os.path.join("data", "databento_downloads", smallest["filename"])
if os.path.exists(dest):
    print("Already downloaded!")
    sys.exit(0)

url = smallest["urls"]["https"]
print(f"Downloading from {url}...")
print("This may take a few minutes...")

try:
    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        mb = downloaded / 1e6
        print(f"\r  {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print(f"\nDownloaded {os.path.getsize(dest)/1e6:.1f} MB to {dest}")
except Exception as e:
    print(f"\nDownload failed: {e}")
    # Clean up partial file
    if os.path.exists(dest):
        os.remove(dest)
    sys.exit(1)
