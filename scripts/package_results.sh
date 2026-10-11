#!/usr/bin/env bash
set -euo pipefail
RUN="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$RUN")"
NAME="$(basename "$RUN")"
ARCHIVE="$PARENT/$NAME.tar.gz"
SIDECAR="$PARENT/${NAME}_archive_sha256.txt"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite archive: $ARCHIVE" >&2
  exit 3
fi
tar -czf "$ARCHIVE" -C "$PARENT" "$NAME"
sha256sum "$ARCHIVE" | tee "$SIDECAR"
echo "$ARCHIVE"
