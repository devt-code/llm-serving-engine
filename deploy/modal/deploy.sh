#!/usr/bin/env bash
# Build the dashboard to be served by the engine itself, then deploy to Modal (see app.py).
# Run from the project's virtualenv after `pip install modal && modal token new`.
set -euo pipefail
cd "$(dirname "$0")/../.."
(cd dashboard && VITE_SAME_ORIGIN=1 npx vite build --outDir dist-server --emptyOutDir)

# The app runs one container at most, so a container still running the previous version (kept alive
# by an open dashboard) would make the new version wait. Stop it once the new version is deployed.
running() {
  python -m modal container list --json | python -c "
import json, sys
print(' '.join(c['container_id'] for c in json.load(sys.stdin) if c['app_name'] == 'llm-serving-engine'))"
}
old=$(running)
python -m modal deploy deploy/modal/app.py
for id in $old; do
  python -m modal container stop --yes "$id"
done
