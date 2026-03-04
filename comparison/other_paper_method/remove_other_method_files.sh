#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FILE_LIST="${SCRIPT_DIR}/FILES.txt"

cd "${REPO_ROOT}"

while IFS= read -r relpath; do
  [[ -z "${relpath}" ]] && continue
  if [[ -e "${relpath}" ]]; then
    rm -rf "${relpath}"
    echo "Removed: ${relpath}"
  else
    echo "Skip (not found): ${relpath}"
  fi
done < "${FILE_LIST}"

echo "Done."

