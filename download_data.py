"""Download FineWeb training shards from HuggingFace if not already present.

Usage: python download_data.py <dst_dir> [max_shards] [timeout_seconds]
"""
import os
import sys
import time

def download_shards(dst: str, max_shards: int = 32, timeout: float = 240.0) -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not available, skipping download")
        return 0

    os.makedirs(dst, exist_ok=True)
    repo = "willdepueoai/parameter-golf"
    t0 = time.time()
    downloaded = 0
    for i in range(1, max_shards + 1):
        if time.time() - t0 > timeout:
            print(f"Timeout reached after {i-1} shards ({time.time()-t0:.0f}s)")
            break
        dest = os.path.join(dst, f"fineweb_train_{i:06d}.bin")
        if os.path.exists(dest):
            continue
        try:
            src = hf_hub_download(
                repo_id=repo,
                filename=f"fineweb_train_{i:06d}.bin",
                subfolder="datasets/fineweb10B_sp1024",
                repo_type="dataset",
            )
            # Copy (resolve any symlinks from HF cache)
            with open(src, "rb") as fin, open(dest, "wb") as fout:
                fout.write(fin.read())
            downloaded += 1
            elapsed = time.time() - t0
            print(f"Downloaded shard {i:06d} ({downloaded} total, {elapsed:.0f}s)")
        except Exception as e:
            print(f"Failed shard {i}: {e}")
            break
    return downloaded

if __name__ == "__main__":
    dst = sys.argv[1] if len(sys.argv) > 1 else "./data/datasets/fineweb10B_sp1024"
    max_s = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    tmo = float(sys.argv[3]) if len(sys.argv) > 3 else 240.0
    n = download_shards(dst, max_s, tmo)
    total = len([f for f in os.listdir(dst) if f.startswith("fineweb_train_") and f.endswith(".bin")])
    print(f"Download complete: {n} new shards, {total} total train shards in {dst}")
