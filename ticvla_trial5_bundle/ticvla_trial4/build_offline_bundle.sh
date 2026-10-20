#!/usr/bin/env bash
# Run ONCE on a fully configured Brev host before destroying that VM.
# Produces a very large ZIP64 archive with Isaac/installed pip packages + models + exact patched code.
set -Eeuo pipefail
# Cloud VMs often have command-specific NOPASSWD; avoid interactive prompts.
sudo() { command sudo -n "$@"; }
PKG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER=ticvla-isaac
READY_IMAGE=ticvla-trial4-ready:5.0.0
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT="${1:-$HOME/ticvla_trial4_full_offline_${STAMP}.zip}"
STAGE="$HOME/trial4_offline_staging_${STAMP}"
PAYLOAD="$STAGE/ticvla_trial4_full_payload"
[[ ! -e "$OUTPUT" ]] || { echo "Archive already exists: $OUTPUT (refusing to overwrite)" >&2; exit 1; }

[[ -d "$HOME/TIC-VLA/.git" && -d "$HOME/models" ]] || { echo "First complete the fresh VM bootstrap" >&2; exit 1; }
sudo docker inspect "$CONTAINER" >/dev/null || { echo "No $CONTAINER; bootstrap first" >&2; exit 1; }
if sudo docker exec "$CONTAINER" pgrep -af 'run_suite.py|benchmark.py' >/dev/null 2>&1; then
  echo "An evaluation may be running; let it finish before making an immutable offline snapshot." >&2
  exit 1
fi
command -v zip >/dev/null || sudo -n apt-get install -y zip
mkdir -p "$PAYLOAD"
echo "Committing Isaac container with installed pip packages ($READY_IMAGE)..."
sudo docker commit "$CONTAINER" "$READY_IMAGE" >/dev/null
printf 'Saving Docker image (multi-GB)...\n'
sudo docker save "$READY_IMAGE" | gzip -1 > "$PAYLOAD/isaac_image.tar.gz"
printf 'Saving checkpoints/model...\n'
tar -czf "$PAYLOAD/models.tar.gz" -C "$HOME" models
printf 'Saving exact source Git checkout and installed Trial 4 code (excluding outputs/logs)...\n'
tar -czf "$PAYLOAD/repo.tar.gz" -C "$HOME" \
  --exclude='TIC-VLA/.env.container' \
  --exclude='TIC-VLA/trial4_results' \
  --exclude='TIC-VLA/trial4_launchlogs' \
  --exclude='TIC-VLA/DynaNav/benchmark_results' \
  --exclude='TIC-VLA/DynaNav/logs' \
  --exclude='TIC-VLA/DynaNav/output' \
  --exclude='TIC-VLA/outputs' \
  --exclude='TIC-VLA/logs' \
  --exclude='TIC-VLA/trial4_backups' \
  TIC-VLA
# This makes the resulting giant ZIP sufficient as the single uploaded file.
cp -a "$PKG_DIR" "$PAYLOAD/ticvla_trial4"
(
  cd "$PAYLOAD"
  sha256sum isaac_image.tar.gz models.tar.gz repo.tar.gz > SHA256SUMS
)
cat > "$PAYLOAD/RESTORE.txt" <<'EOF'
ON FRESH UBUNTU NVIDIA GPU VM:
  sudo -n apt-get update && sudo -n apt-get install -y unzip
  unzip ticvla_trial4_full_offline_*.zip
  bash ticvla_trial4_full_payload/ticvla_trial4/bootstrap_fresh_vm.sh \
    --accept-eula --offline-dir "$HOME/ticvla_trial4_full_payload"
  bash ticvla_trial4_full_payload/ticvla_trial4/vm.sh run \
    --episodes 20 --repeats 1 --tag t4_first
Requires NVIDIA driver, suitable disk space, and Docker/NVIDIA Toolkit (installer attempts to install host tools if missing).
EOF
printf 'Creating ZIP64 archive; this can be tens of GB and take considerable time...\n'
(cd "$STAGE" && zip -0 -q -r "$OUTPUT" ticvla_trial4_full_payload)
echo "OFFLINE BUNDLE READY: $OUTPUT"
ls -lh "$OUTPUT"
echo "Keep this ZIP on your Mac or persistent cloud storage BEFORE terminating the VM."
echo "Staging files also consume disk at $STAGE; remove only after confirming your archive transferred correctly."
