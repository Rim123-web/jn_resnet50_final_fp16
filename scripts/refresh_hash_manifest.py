from __future__ import annotations

from pathlib import Path

from common import RUN, sha256, write_csv


EXCLUDED = {
    Path("validation/file_hash_manifest.csv"),
    Path("logs/package_validation.log"),
}


def main():
    rows = []
    for path in sorted(RUN.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(RUN)
        if relative in EXCLUDED:
            continue
        rows.append(
            {
                "relative_path": str(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(RUN / "validation/file_hash_manifest.csv", rows)
    print(f"HASH MANIFEST REFRESH PASS: {len(rows)} files")


if __name__ == "__main__":
    main()
