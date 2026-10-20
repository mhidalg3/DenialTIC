#!/usr/bin/env bash
# Fresh x86_64 Ubuntu GPU VM -> complete TIC-VLA/Isaac Sim 5.0 Trial 4 runtime.
# Safe to rerun after interrupted downloads. Never re-installs NVIDIA host driver.
set -Eeuo pipefail

PKG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$HOME/TIC-VLA"
MODELS="$HOME/models"
CONTAINER="ticvla-isaac"
UPSTREAM="https://github.com/ucla-mobility/TIC-VLA.git"
COMMIT="9fa6f8b66b9e121d5df5df071297bba8e5353ebb"
ISAAC_IMAGE="nvcr.io/nvidia/isaac-sim:5.0.0"
CUDA_IMAGE="nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04"
READY_IMAGE="ticvla-trial4-ready:5.0.0"
OFFLINE_DIR=""
ACCEPT_EULA=0
RUN_SMOKE=0
PLAN_ONLY=0

usage() {
cat <<'EOF'
Usage: bash bootstrap_fresh_vm.sh --accept-eula [--run-smoke]
       bash bootstrap_fresh_vm.sh --accept-eula --offline-dir /path/to/extracted/ticvla_trial4_full_payload
       bash bootstrap_fresh_vm.sh --plan

Online mode downloads the pinned TIC-VLA checkout, Isaac Sim 5.0 Docker image,
InternVL3-1B base model and TIC-VLA evaluation checkpoint, installs DynaNav
Python requirements in Isaac's container Python, applies Trial 4 code and tests.

Before running, optionally set a read-only HF_TOKEN (not saved in the package).
--accept-eula acknowledges the NVIDIA Isaac Sim EULA for container use.
--run-smoke launches the first matched episode block (three rollouts) (costs GPU time).
--offline-dir uses a large archive previously assembled with build_offline_bundle.sh.
Do not invoke these commands from INSIDE a Docker container.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accept-eula) ACCEPT_EULA=1; shift ;;
    --run-smoke) RUN_SMOKE=1; shift ;;
    --offline-dir) [[ $# -ge 2 ]] || { usage; exit 2; }; OFFLINE_DIR="$2"; shift 2 ;;
    --plan) PLAN_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

trap 'echo "Setup failed at line $LINENO. Completed downloads are reusable; rerun this script after fixing the reported error." >&2' ERR

info() { printf '\n[%s] %s\n' "$(date -u '+%H:%M:%S UTC')" "$*"; }
fatal() { echo "FATAL: $*" >&2; exit 1; }

# Never prompt for a sudo password on ephemeral/cloud instances.
# A sudoers validation request (`sudo -v`) can require a password even when
# particular commands have NOPASSWD, so test an actual command instead.
sudo() { command sudo -n "$@"; }

if (( PLAN_ONLY )); then
  cat <<EOF
Trial 4 fresh-VM dependency plan (no changes made):
- VM: Ubuntu x86_64, NVIDIA driver/GPU provisioned by cloud provider
- Host: apt git/python3/venv/curl/unzip, Docker daemon, NVIDIA Container Toolkit
- Source: $UPSTREAM @ $COMMIT
- Simulator: $ISAAC_IMAGE (large; official Docker container, not workstation zip)
- Model: OpenGVLab/InternVL3-1B @ 4415a3b810e636d11dfa86b0e9ba40bb00535aa8
- Checkpoint: handsomeYun/TIC-VLA @ f27fa2f85aa205098080c53bb1272e987e2c1a25 (only TIC-VLA-model.ckpt)
- Evaluation packages: requirements-test.txt into /isaac-sim/python.sh (not requirements-train.txt)
- Output: $REPO, $MODELS, detached Docker container (VM-local, not durable storage) $CONTAINER
- Trial 4 scripts: $PKG_DIR/install.py; $PKG_DIR/vm.sh
- Validation: GPU passthrough, SimulationApp minimal launch, Trial 4 hooks/tests
- External downloads are needed unless --offline-dir points to a previously made full bundle.
EOF
  exit 0
fi

[[ "$ACCEPT_EULA" == 1 ]] || fatal "Pass --accept-eula after reviewing NVIDIA Isaac Sim terms, or --plan."
[[ "$(uname -m)" == "x86_64" ]] || fatal "Expected x86_64 GPU VM, got $(uname -m)."
[[ -f /etc/os-release ]] || fatal "Linux distribution cannot be determined"
. /etc/os-release
[[ "${ID:-}" == ubuntu ]] || fatal "This automatic install targets Ubuntu VMs; found ${ID:-unknown}."
case "${VERSION_ID:-}" in 22.04|24.04) ;; *) fatal "Ubuntu ${VERSION_ID:-unknown} has not been validated by this setup; need 22.04 or 24.04";; esac
command -v nvidia-smi >/dev/null || fatal "No NVIDIA driver/host GPU. Create a GPU-enabled Brev VM first. We do not modify provider-managed GPU drivers."
nvidia-smi -L || fatal "GPU not visible to the host"

# Require free space before pulling many-gigabyte OCI layers and model weights.
FREE_GIB=$(df -Pk "$HOME" | awk 'NR==2 {printf "%.0f", $4/1024/1024}')
REQUIRED_GIB=80
[[ -z "$OFFLINE_DIR" ]] || REQUIRED_GIB=100
if [[ "$FREE_GIB" -lt "$REQUIRED_GIB" ]]; then
  fatal "Only $FREE_GIB GiB available under HOME. Allow >=$REQUIRED_GIB GiB for this setup."
fi
info "Starting Trial 4 fresh-VM setup; available disk: ${FREE_GIB} GiB"

# Some Brev images have these already; do not require Docker or toolkit to be preinstalled.
sudo true || fatal "Non-interactive sudo is unavailable. Ask the VM provider for passwordless admin privileges."
info "Installing minimal host tools if missing"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates curl git unzip python3 python3-venv python3-pip gnupg procps util-linux

info "Verifying uploaded Trial 4 code package SHA-256 checksums"
python3 "$PKG_DIR/verify_package.py"

if ! command -v docker >/dev/null 2>&1; then
  info "Docker not found; installing docker.io"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
fi
if ! sudo docker info >/dev/null 2>&1; then
  info "Docker daemon not active; starting docker service"
  sudo systemctl enable --now docker || fatal "Docker daemon could not be started"
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  info "NVIDIA Container Toolkit missing; installing official NVIDIA apt repository"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
fi
if ! sudo docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
  info "Configuring NVIDIA GPU runtime in Docker"
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi
sudo docker info >/dev/null

info "Validating NVIDIA container runtime"
if [[ -z "$OFFLINE_DIR" ]]; then
  sudo docker run --rm --gpus all --runtime=nvidia "$CUDA_IMAGE" nvidia-smi
else
  info "Offline restoration: deferring container GPU passthrough check until local image has been loaded"
fi

if [[ -n "$OFFLINE_DIR" ]]; then
  info "Restoring full source/model/image payload from $OFFLINE_DIR"
  OFFLINE_DIR=$(realpath "$OFFLINE_DIR")
  [[ -f "$OFFLINE_DIR/SHA256SUMS" ]] || fatal "Missing SHA256SUMS in offline directory"
  (cd "$OFFLINE_DIR" && sha256sum -c SHA256SUMS) || fatal "Offline payload checksum failed"
  [[ ! -e "$REPO" && ! -e "$MODELS" ]] || fatal "Offline restore requires fresh HOME without TIC-VLA/models; move existing paths aside explicitly."
  tar -xzf "$OFFLINE_DIR/repo.tar.gz" -C "$HOME"
  tar -xzf "$OFFLINE_DIR/models.tar.gz" -C "$HOME"
  gzip -dc "$OFFLINE_DIR/isaac_image.tar.gz" | sudo docker load
  ISAAC_IMAGE="$READY_IMAGE"
  sudo docker run --rm --gpus all --runtime=nvidia --entrypoint nvidia-smi "$ISAAC_IMAGE"
else
  info "Cloning exact TIC-VLA source commit"
  if [[ ! -e "$REPO" ]]; then
    git init "$REPO"
    git -C "$REPO" remote add origin "$UPSTREAM"
    git -C "$REPO" fetch --depth=1 origin "$COMMIT"
    git -C "$REPO" checkout --detach FETCH_HEAD
  fi
  [[ -d "$REPO/.git" ]] || fatal "$REPO exists but isn't a Git repository"
  ACTUAL_COMMIT=$(git -C "$REPO" rev-parse HEAD)
  if [[ "$ACTUAL_COMMIT" != "$COMMIT" ]]; then
    if git -C "$REPO" status --porcelain | grep -q .; then
      fatal "Existing $REPO at $ACTUAL_COMMIT has modifications. Fresh VM setup refuses to overwrite code."
    fi
    git -C "$REPO" fetch origin "$COMMIT"
    git -C "$REPO" checkout --detach "$COMMIT"
  fi
  [[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || fatal "Repo SHA mismatch"

  # If scene assets are Git LFS pointers rather than USD files, fetch their contents.
  if find "$REPO/DynaNav/assets" -maxdepth 1 -name '*.usd' -type f -print -quit 2>/dev/null | grep -q .; then
    if grep -l -m1 '^version https://git-lfs.github.com/spec/v1' "$REPO"/DynaNav/assets/*.usd >/dev/null 2>&1; then
      info "Fetching Git LFS scene assets"
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y git-lfs
      git -C "$REPO" lfs install --local
      git -C "$REPO" lfs pull
    fi
  fi

  info "Installing isolated Hugging Face download client (not training dependencies)"
  mkdir -p "$HOME/.venvs"
  python3 -m venv "$HOME/.venvs/ticvla-download"
  "$HOME/.venvs/ticvla-download/bin/python" -m pip install --disable-pip-version-check \
    'huggingface_hub==0.36.0' 'hf_xet>=1.1,<2'
  info "Downloading only the base model and evaluation checkpoint (not the training dataset)"
  "$HOME/.venvs/ticvla-download/bin/python" "$PKG_DIR/download_models.py"
  info "Pulling NVIDIA Isaac Sim 5.0 OCI image; this can be many GB"
  sudo docker pull "$ISAAC_IMAGE"

  info "Configuring released DynaNav benchmark for headless cloud rendering"
  python3 - "$REPO/DynaNav/benchmark.py" <<'PYHEADLESS'
import sys
from pathlib import Path
path = Path(sys.argv[1])
s = path.read_text()
old = 'APP_CONFIG = {"renderer": "RayTracedLighting", "headless": False, "width": 640, "height": 480}'
new = 'APP_CONFIG = {"renderer": "RayTracedLighting", "headless": True, "width": 640, "height": 480}'
if old in s:
    path.write_text(s.replace(old, new, 1))
elif new not in s:
    raise SystemExit("Unexpected DynaNav APP_CONFIG; refusing to guess headless edit")
PYHEADLESS

  info "Applying complete Trial 4 implementation to clean TIC-VLA checkout"
  if ! grep -q 'TICVLA_TRIAL4_INSTALLED' "$REPO/DynaNav/ticvla.py"; then
    python3 "$PKG_DIR/install.py" --repo "$REPO" --apply
  fi
fi

for file in "$MODELS/InternVL3-1B/config.json" "$MODELS/InternVL3-1B/model.safetensors" "$MODELS/TIC-VLA/TIC-VLA-model.ckpt"; do
  [[ -s "$file" ]] || fatal "Required model file missing: $file"
done
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || fatal "Source commit is not pinned: $REPO"
grep -E '^APP_CONFIG.*"headless": True' "$REPO/DynaNav/benchmark.py" >/dev/null || fatal "DynaNav benchmark APP_CONFIG is not headless=True"
for f in DynaNav/ticvla.py DynaNav/benchmark.py DynaNav/behavior/nova_carter_test_ticvla.py; do
  grep -q 'TICVLA_TRIAL4_INSTALLED' "$REPO/$f" || fatal "Trial 4 hook missing: $f"
done

info "Creating Isaac container environment file (contains no auth tokens)"
cat > "$REPO/.env.container" <<'EOF'
export ISAAC_SIM_ROOT=/isaac-sim
export ISAAC_SIM_PYTHON=/isaac-sim/python.sh
export TICVLA_DYNANAV_ROOT=/workspace/TIC-VLA/DynaNav
export TICVLA_BASE_MODEL_PATH=/workspace/models/InternVL3-1B
export TICVLA_CHECKPOINT_PATH=/workspace/models/TIC-VLA/TIC-VLA-model.ckpt
export TICVLA_OUTPUT_DIR=/workspace/TIC-VLA/outputs
export CUDA_VISIBLE_DEVICES=0
EOF

info "Creating host-mounted directories and VM-local cache paths"
mkdir -p "$REPO/logs" "$REPO/trial4_results" \
  "$HOME/docker/isaac-sim/cache/main" \
  "$HOME/docker/isaac-sim/cache/computecache" \
  "$HOME/docker/isaac-sim/cache/kit" \
  "$HOME/docker/isaac-sim/cache/pip" \
  "$HOME/docker/isaac-sim/config" \
  "$HOME/docker/isaac-sim/data" \
  "$HOME/docker/isaac-sim/logs" \
  "$HOME/docker/isaac-sim/pkg" \
  "$HOME/.cache/ov/hub"

if sudo docker inspect "$CONTAINER" >/dev/null 2>&1; then
  MOUNT=$(sudo docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace/TIC-VLA"}}{{.Source}}{{end}}{{end}}' "$CONTAINER")
  [[ "${MOUNT:-}" == "$REPO" ]] || fatal "Existing $CONTAINER has a different repo mount ($MOUNT); refusing to remove it"
  if [[ "$(sudo docker inspect -f '{{.State.Running}}' "$CONTAINER")" != true ]]; then
    sudo docker start "$CONTAINER"
  fi
else
  info "Creating detached persistent Isaac Sim 5.0 container"
  sudo docker run -d \
    --name "$CONTAINER" --user 0:0 \
    --runtime=nvidia --gpus all --network=host \
    --entrypoint /bin/bash \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e ACCEPT_EULA=Y -e NVIDIA_DRIVER_CAPABILITIES=all -e CUDA_VISIBLE_DEVICES=0 \
    -v "$REPO:/workspace/TIC-VLA:rw" \
    -v "$MODELS:/workspace/models:ro" \
    -v "$HOME/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw" \
    -v "$HOME/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
    -v "$HOME/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw" \
    -v "$HOME/docker/isaac-sim/cache/pip:/root/.cache/pip:rw" \
    -v "$HOME/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
    -v "$HOME/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw" \
    -v "$HOME/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw" \
    -v "$HOME/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
    -v "$HOME/.cache/ov/hub:/var/cache/hub:rw" \
    "$ISAAC_IMAGE" -lc 'sleep infinity'
fi

info "Checking CUDA inside Isaac container"
sudo docker exec "$CONTAINER" nvidia-smi
info "Installing container-side system tools (Git is needed by the experiment runner)"
sudo docker exec -u 0 "$CONTAINER" bash -lc '
  set -euo pipefail
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates procps util-linux
  git config --global --add safe.directory /workspace/TIC-VLA
  git -C /workspace/TIC-VLA rev-parse HEAD
'
if [[ -z "$OFFLINE_DIR" ]]; then
  info "Installing DynaNav + TIC-VLA runtime requirements in Isaac Python"
  sudo docker exec "$CONTAINER" bash -lc '
    set -euo pipefail
    cd /workspace/TIC-VLA
    /isaac-sim/python.sh DynaNav/trial4/protect_runtime.py --write /tmp/ticvla-runtime-constraints.txt
    /isaac-sim/python.sh -m pip install --constraint /tmp/ticvla-runtime-constraints.txt -e .
    /isaac-sim/python.sh -m pip install --constraint /tmp/ticvla-runtime-constraints.txt -r requirements-test.txt
    /isaac-sim/python.sh DynaNav/trial4/protect_runtime.py --verify /tmp/ticvla-runtime-constraints.txt
    /isaac-sim/python.sh -m pip freeze > /workspace/TIC-VLA/trial4_pip_freeze.txt
  '
else
  info "Using packaged Python dependencies baked into offline Docker image"
fi

info "Running actual minimal headless Isaac SimulationApp preflight"
sudo docker exec -i "$CONTAINER" /isaac-sim/python.sh - <<'PY'
from isaacsim import SimulationApp
print("Creating SimulationApp...", flush=True)
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
print("SimulationApp started", flush=True)
for _ in range(10):
    app.update()
app.close()
print("SimulationApp closed successfully", flush=True)
PY

info "Running Trial 4 setup validators (real GPU + bundled CPU tests)"
bash "$PKG_DIR/vm.sh" check

info "Recording image and model provenance"
python3 - "$REPO" "$CONTAINER" <<'PY'
import json, subprocess, sys
from pathlib import Path
repo, container = Path(sys.argv[1]), sys.argv[2]
docker = subprocess.check_output(["sudo", "-n", "docker", "inspect", "--format", "{{.Image}}", container], text=True).strip()
manifest = json.loads((repo.parent / "models" / "trial4_model_manifest.json").read_text()) if (repo.parent / "models" / "trial4_model_manifest.json").is_file() else {}
manifest.update({"ticvla_git_commit":subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(), "container_image_sha256":docker, "note":"Actual Linux/NVIDIA drivers are VM-provided. No credentials stored."})
(repo / "trial4_setup_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
PY

info "COMPLETE: freshly provisioned Trial 4 runtime is ready"
printf '%s\n' \
  "Run first block:  bash $PKG_DIR/vm.sh run --episodes 20 --repeats 1 --tag t4_first" \
  "Follow log:           tail -F $REPO/trial4_launchlogs/<STAMP>.log" \
  "Summarize:            bash $PKG_DIR/vm.sh summarize t4_first" \
  "Export results:       bash $PKG_DIR/vm.sh export"
if ((RUN_SMOKE)); then
  info "Launching requested first matched Trial 4 episode block"
  bash "$PKG_DIR/vm.sh" run --episodes 20 --repeats 1 --tag t4_first
fi
