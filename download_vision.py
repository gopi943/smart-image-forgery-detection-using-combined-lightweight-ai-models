"""
Download VISION dataset (images only, skip videos).
Shows per-file download progress.

Usage:
    python scripts/download_vision.py --output F:\VISION
"""

import argparse
import os
import sys
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser


BASE_URL = "https://lesc.dinfo.unifi.it/VISION/dataset/"

DEVICES = [
    "D01_Samsung_GalaxyS3Mini", "D02_Apple_iPhone4s", "D03_Huawei_P9",
    "D04_LG_D290", "D05_Apple_iPhone5c", "D06_Apple_iPhone6",
    "D07_Lenovo_P70A", "D08_Samsung_GalaxyTab3", "D09_Apple_iPhone4",
    "D10_Apple_iPhone4s", "D11_Samsung_GalaxyS3", "D12_Sony_XperiaZ1Compact",
    "D13_Apple_iPad2", "D14_Apple_iPhone5c", "D15_Apple_iPhone6",
    "D16_Huawei_P9Lite", "D17_Microsoft_Lumia640LTE", "D18_Apple_iPhone5c",
    "D19_Apple_iPhone6Plus", "D20_Apple_iPadMini", "D21_Wiko_Ridge4G",
    "D22_Samsung_GalaxyTrendPlus", "D23_Asus_Zenfone2Laser",
    "D24_Xiaomi_RedmiNote3", "D25_OnePlus_A3000", "D26_Samsung_GalaxyS3Mini",
    "D27_Samsung_GalaxyS5", "D28_Huawei_P8", "D29_Apple_iPhone5",
    "D30_Huawei_Honor5c", "D31_Samsung_GalaxyS4Mini", "D32_OnePlus_A3003",
    "D33_Huawei_Ascend", "D34_Apple_iPhone5", "D35_Samsung_GalaxyTabA",
]

# Image subfolders to download (skip videos)
IMAGE_SUBDIRS = ["flat", "flatFBH", "flatWA", "nat", "natFBH", "natWA"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, val in attrs:
                if name == "href" and val and not val.startswith("?") and val != "../":
                    self.links.append(val)


def list_remote(url):
    """List links in a remote directory."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        p = LinkParser()
        p.feed(html)
        return p.links
    except Exception as e:
        print(f"\n  [WARN] Cannot list {url}: {e}")
        return []


def download_one(url, path):
    """Download a single file with progress."""
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0
            with open(path, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        kb = downloaded // 1024
                        print(f"\r    Downloading: {os.path.basename(path)} [{kb} KB / {total//1024} KB] {pct}%", end="", flush=True)
        return True
    except Exception as e:
        # Clean up partial file
        if os.path.exists(path):
            os.remove(path)
        print(f"\n  [FAIL] {os.path.basename(path)}: {e}")
        return False


def download_device(device, output_dir):
    """Download all images from one device."""
    device_url = BASE_URL + device + "/"
    total = 0

    # First, list the device directory to find structure
    top_links = list_remote(device_url)

    # Try known image subdirectory patterns
    for subdir in IMAGE_SUBDIRS:
        # Try: device/images/subdir/ or device/subdir/
        for prefix in ["images/", ""]:
            folder_url = device_url + prefix + subdir + "/"
            files = list_remote(folder_url)
            images = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]

            if images:
                print(f"\n  Found {len(images)} images in {prefix}{subdir}/")
                for i, img in enumerate(images, 1):
                    local_path = os.path.join(output_dir, device, subdir, img)
                    if os.path.exists(local_path):
                        total += 1
                        continue
                    success = download_one(folder_url + img, local_path)
                    if success:
                        total += 1
                        print(f" ✓ [{i}/{len(images)}]")
                break  # Found images for this subdir, move to next

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="F:\\VISION")
    parser.add_argument("--start-from", type=int, default=1,
                       help="Start from device number (1-35), useful for resuming")
    args = parser.parse_args()

    print("=" * 60)
    print("VISION Dataset Downloader")
    print(f"Saving to: {args.output}")
    print(f"Devices: {args.start_from} to 35")
    print("=" * 60)

    grand_total = 0
    start_idx = args.start_from - 1

    for i, device in enumerate(DEVICES[start_idx:], start_idx + 1):
        print(f"\n[{i}/35] {device}")
        count = download_device(device, args.output)
        grand_total += count
        print(f"  => {count} images downloaded")

    print(f"\n{'=' * 60}")
    print(f"DONE! Total: {grand_total} images saved to {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
