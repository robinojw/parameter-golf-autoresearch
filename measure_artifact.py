import sys
from pathlib import Path

ARTIFACT_LIMIT = 16_000_000


def _try_import_zstandard():
    try:
        import zstandard as zstd

        return zstd
    except ImportError:
        return None


def _measure_weight_file(weight_path: Path, zstd_module) -> int:
    raw = weight_path.read_bytes()
    if zstd_module is not None:
        cctx = zstd_module.ZstdCompressor()
        compressed = cctx.compress(raw)
        print(f"  {weight_path.name}: {len(raw)} raw -> {len(compressed)} zstd")
        return len(compressed)
    print(
        f"  {weight_path.name}: {len(raw)} raw (zstandard not installed, using raw size)"
    )
    return len(raw)


def measure_artifact(train_script: str = "train_gpt.py") -> int:
    script_path = Path(train_script)
    if not script_path.exists():
        print(f"ERROR: {train_script} not found")
        return 0

    total = script_path.stat().st_size

    weight_files = list(Path(".").glob("*.npz")) + list(Path(".").glob("*.pt"))
    if weight_files:
        zstd_module = _try_import_zstandard()
        for weight_path in weight_files:
            total += _measure_weight_file(weight_path, zstd_module)

    headroom = ARTIFACT_LIMIT - total
    status = "OK" if total <= ARTIFACT_LIMIT else "OVER LIMIT"

    print(f"artifact_bytes: {total}")
    print(f"limit: {ARTIFACT_LIMIT}")
    print(f"headroom: {headroom}")
    print(f"status: {status}")

    return total


def _main() -> None:
    measured_total = measure_artifact()
    over_limit = measured_total > ARTIFACT_LIMIT
    if over_limit:
        sys.exit(1)


if __name__ == "__main__":
    _main()
