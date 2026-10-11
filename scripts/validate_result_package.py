from pathlib import Path
import csv
import hashlib
import json

RUN = Path(__file__).resolve().parents[1]
report = json.loads((RUN / "validation/final_validation_report.json").read_text())
if report.get("status") != "PASS":
    raise SystemExit("PACKAGE VALIDATION FAILED")
manifest = RUN / "validation/file_hash_manifest.csv"
if not manifest.exists() or manifest.stat().st_size == 0:
    raise SystemExit("PACKAGE HASH MANIFEST MISSING")
with manifest.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
failures = []
for row in rows:
    path = RUN / row["relative_path"]
    if not path.exists():
        failures.append(f"missing:{row['relative_path']}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        failures.append(f"hash:{row['relative_path']}")
if failures:
    raise SystemExit("PACKAGE HASH VALIDATION FAILED: " + ", ".join(failures[:20]))
print("PACKAGE VALIDATION PASS")
