#!/usr/bin/env bash
# Brev HOST ONLY. Run Trial 4's established container launcher with Trial 5 protocol.
set -euo pipefail
sudo() { command sudo -n "$@"; }
ROOT="${TICVLA_REPO:-$HOME/TIC-VLA}"
PKG="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CROOT="/workspace/TIC-VLA"
T4="${TICVLA_TRIAL4_PACKAGE:-$HOME/ticvla_trial4}/vm.sh"
mode="${1:-}"; [[ $# -gt 0 ]] && shift || true
case "$mode" in
 check)
   test -f "$ROOT/DynaNav/trial5/install_info.json" || { echo 'Trial 5 overlay is not installed' >&2;exit 1; }
   python3 "$PKG/test_trial5.py" -v
   sudo docker exec ticvla-isaac bash -lc '
     set -euo pipefail
     cd /workspace/TIC-VLA
     source .env.container
     /isaac-sim/python.sh DynaNav/trial4/run_suite.py --check
     /isaac-sim/python.sh DynaNav/trial5/test_trial5.py -v
   '
   ;;
 smoke)
   bash "$T4" run --episodes 17 --seed-start 8800 --repeats 1 --replicates 1 \
        --order-seed 20261018 --on-frames 180 --off-frames 900 \
        --arrival-frames 30 --service-frames 120 --deadline-frames 150 \
        --tag "${1:-t5_smoke_17}"
   ;;
 run)
   bash "$T4" run \
      --manifest-file "$CROOT/DynaNav/trial5/frozen_study.json" \
      --replicates 2 --order-seed 20261018 \
      --on-frames 180 --off-frames 900 --arrival-frames 30 \
      --service-frames 120 --warmup-frames 3 --deadline-frames 150 \
      "$@"
   ;;
 dry-run)
   # Compute a frozen plan and return before starting the actual child episodes.
   # This still creates a session directory; choose a fresh tag for the real run.
   bash "$T4" run \
      --manifest-file "$CROOT/DynaNav/trial5/frozen_study.json" \
      --replicates 2 --order-seed 20261018 \
      --on-frames 180 --off-frames 900 --arrival-frames 30 \
      --service-frames 120 --warmup-frames 3 --deadline-frames 150 \
      --dry-run "$@"
   ;;
 status|audit)
   bash "$T4" "$mode" "$@"
   ;;
 analyze|interim)
   [[ $# -ge 1 ]] || { echo 'Supply at least one session tag' >&2; exit 2; }
   arr=()
   for tag in "$@"; do
     [[ "$tag" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Bad tag' >&2;exit 2; }
     arr+=("$ROOT/trial4_results/$tag")
   done
   out="$ROOT/trial5_reports/$(date -u +%Y%m%dT%H%M%SZ)"
   if [[ "$mode" == interim ]]; then
      python3 "$PKG/analysis.py" --interim-blind --out "$out" "${arr[@]}"
   else
      python3 "$PKG/analysis.py" --out "$out" "${arr[@]}"
   fi
   echo "Report: $out"
   ;;
 export)
   [[ $# -eq 1 ]] || { echo 'Usage: vm5.sh export SESSION_TAG' >&2;exit 2; }
   tag="$1"; [[ "$tag" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
   test -f "$ROOT/trial4_results/$tag/run_state.json" || { echo "Unknown session: $tag" >&2; exit 2; }
   outfile="$HOME/trial5_${tag}_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
   tar -czf "$outfile" -C "$ROOT" "trial4_results/$tag" "DynaNav/trial5" \
       -C "$HOME" ticvla_trial5 ticvla_trial4
   (cd "$(dirname "$outfile")" && sha256sum "$(basename "$outfile")" > "$(basename "$outfile").sha256")
   ls -lh "$outfile"
   echo "Integrity: $outfile.sha256"
   ;;
 *)
   echo 'Usage: vm5.sh check | smoke [TAG] | dry-run --block-start N --block-count K --tag TAG | run ... | status TAG | audit TAG | interim TAG... | analyze TAG... | export TAG' >&2
   exit 2
   ;;
esac
