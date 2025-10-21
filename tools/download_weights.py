#!/usr/bin/env python3
"""Download YOLOX pretrained weights"""

import urllib.request
import sys
from pathlib import Path


def download_file(url, output_path):
    """Download file with progress bar"""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, report_progress)
    print("\nDownload complete!")


def main():
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    # YOLOX-M weights
    url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
    output_path = weights_dir / "yolox_s.pth"

    if output_path.exists():
        print(f"Weights already exist at: {output_path}")
        return

    try:
        download_file(url, output_path)
    except Exception as e:
        print(f"\nError downloading weights: {e}")
        print("\nPlease download manually from:")
        print(url)
        print(f"And save to: {output_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
