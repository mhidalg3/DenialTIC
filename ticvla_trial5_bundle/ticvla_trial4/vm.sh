#!/usr/bin/env bash
# Run this on the Brev VM host, not the Mac and not inside an interactive container.
set -euo pipefail
# Never prompt for sudo credentials (including child helpers).
sudo() { command sudo -n "$@"; }
REPO="${TICVLA_REPO:-$HOME/TIC-VLA}"
CONTAINER="${TICVLA_CONTAINER:-ticvla-isaac}"
CROOT="/workspace/TIC-VLA"
usage() {
  cat <<'EOF'
Usage:
  bash vm.sh check
  bash vm.sh run --episodes 20 --repeats 1 --tag t4_first
  bash vm.sh run --episodes 25,45,75,85 --repeats 1 --tag t4_rest
  bash vm.sh audit SESSION_TAG
  bash vm.sh status SESSION_TAG
  bash vm.sh summarize SESSION_TAG
  bash vm.sh export
Runs stay in the detached container if SSH disconnects. Only run one suite.
EOF
}
[[ $# -gt 0 ]] || { usage; exit 2; }
MODE="$1"; shift
if [[ "$MODE" == export ]]; then
  [[ -d "$REPO/trial4_results" ]] || { echo "No Trial 4 results directory" >&2; exit 1; }
  archive="$HOME/ticvla_trial4_results_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
  PKG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  items=(trial4_results)
  for name in trial4_reports trial4_launchlogs trial4_setup_manifest.json trial4_pip_freeze.txt; do
    [[ ! -e "$REPO/$name" ]] || items+=("$name")
  done
  tar -czf "$archive" -C "$REPO" "${items[@]}"
  ls -lh "$archive"
  printf '\nCopy this archive to the Mac using the Brev instance name from brev ls.\n'
  exit 0
fi
sudo true
sudo docker inspect "$CONTAINER" >/dev/null || { echo "Container not found. Do not delete/recreate it; inspect docker ps -a." >&2; exit 1; }
if [[ "$(sudo docker inspect -f '{{.State.Running}}' "$CONTAINER")" != true ]]; then
  sudo docker start "$CONTAINER" >/dev/null
fi
[[ -d "$REPO/DynaNav/trial4" ]] || { echo "Run install.py --apply first" >&2; exit 1; }
# Verify the real bind mount; do not assume that a container sees the host checkout.
ACTUAL=$(sudo docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace/TIC-VLA"}}{{.Source}}{{end}}{{end}}' "$CONTAINER")
[[ "$(realpath "$ACTUAL")" == "$(realpath "$REPO")" ]] || {
  echo "Container mount mismatch: /workspace/TIC-VLA maps to $ACTUAL, not $REPO" >&2; exit 1;
}
case "$MODE" in
  check)
    # Packaging/bootstrap and patch-fixture tests require the actual ZIP root.
    # They must run on the VM host, where vm.sh and bootstrap_fresh_vm.sh live.
    PKG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    echo "[Trial 4] Checking fresh-VM bootstrap scripts and installer on host"
    python3 -m unittest discover -s "$PKG_DIR/tests" -p 'test_bootstrap.py' -v
    python3 -m unittest discover -s "$PKG_DIR/tests" -p 'test_install.py' -v
    echo "[Trial 4] Checking installed runtime, GPU, scheduler and cache behavior inside Isaac Sim"
    # An earlier installer copied host-only tests into DynaNav/trial4/tests.
    # Use explicit test names so existing v3 VMs remain repairable in place.
    sudo docker exec "$CONTAINER" bash -lc '
      set -euo pipefail
      cd /workspace/TIC-VLA
      source .env.container
      export CUDA_VISIBLE_DEVICES=0
      /isaac-sim/python.sh DynaNav/trial4/run_suite.py --check
      /isaac-sim/python.sh -m unittest discover -s DynaNav/trial4/tests -p test_core.py -v
      /isaac-sim/python.sh -m unittest discover -s DynaNav/trial4/tests -p test_adapter.py -v
      /isaac-sim/python.sh -m unittest discover -s DynaNav/trial4/tests -p test_suite.py -v
    '
    ;;
  run)
    mkdir -p "$REPO/trial4_launchlogs"
    STAMP=$(date -u +%Y%m%dT%H%M%S)_$$
    LOG="trial4_launchlogs/$STAMP.log"
    sudo docker exec -d -e "T4_LAUNCH_LOG=$CROOT/$LOG" "$CONTAINER" bash -lc '
      exec >"$T4_LAUNCH_LOG" 2>&1
      set -euo pipefail
      cd /workspace/TIC-VLA
      source .env.container
      export CUDA_VISIBLE_DEVICES=0
      exec /isaac-sim/python.sh -u DynaNav/trial4/run_suite.py "$@"
    ' bash "$@"
    echo "Launched. Follow progress with:"
    printf 'tail -n 80 -F %q\n' "$REPO/$LOG"
    echo "The session's run_state.json records completion or invalid-episode stop."
    ;;
  audit)
    [[ $# -eq 1 && "$1" =~ ^[A-Za-z0-9_-]+$ ]] || { usage; exit 2; }
    sudo docker exec "$CONTAINER" /isaac-sim/python.sh \
      "$CROOT/DynaNav/trial4/audit.py" "$CROOT/trial4_results/$1"
    ;;
  status)
    [[ $# -eq 1 && "$1" =~ ^[A-Za-z0-9_-]+$ ]] || { usage; exit 2; }
    if [[ -f "$REPO/trial4_results/$1/run_state.json" ]]; then
      cat "$REPO/trial4_results/$1/run_state.json"
    else
      echo "No session state yet; inspect the printed launcher log for a startup error."
    fi
    ;;
  summarize)
    [[ $# -ge 1 ]] || { usage; exit 2; }
    paths=()
    for tag in "$@"; do
      [[ "$tag" =~ ^[A-Za-z0-9_-]+$ ]] || { usage; exit 2; }
      paths+=("$CROOT/trial4_results/$tag")
    done
    sudo docker exec "$CONTAINER" /isaac-sim/python.sh \
      "$CROOT/DynaNav/trial4/summarize.py" "${paths[@]}"
    ;;
  *) usage; exit 2 ;;
esac
