#!/usr/bin/env bash
# Candidate installation, not a claim of validated 5080 compatibility.
set -euo pipefail
model="${1:-}"
if [[ "$model" != "smirk" && "$model" != "deca" ]]; then
  echo "Usage: bash environments/bootstrap_wsl.sh smirk|deca" >&2
  exit 2
fi
project_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${FACEBENCH_PYTHON:-python3.10}"
command -v "$python_bin" >/dev/null || { echo "Install Python 3.10 + venv first, or set FACEBENCH_PYTHON." >&2; exit 2; }
command -v nvcc >/dev/null || { echo "CUDA compiler nvcc is required for PyTorch3D. Prepare a CUDA 12.8-compatible toolchain first." >&2; exit 2; }
"$python_bin" -m venv "$project_root/.venv-$model"
source "$project_root/.venv-$model/bin/activate"
python -m pip install --upgrade pip wheel 'setuptools<70'
python -m pip install numpy==1.26.4
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-build-isolation -r "$project_root/environments/requirements-runtime.txt"
python -m pip install -e "$project_root"
python -m facebench fetch-upstream --model "$model" --destination "$project_root/third_party/$model"
# Do not use stale CUDA 11.x binary wheels on Blackwell. Compile and record the source revision.
if [[ ! -d "$project_root/third_party/pytorch3d/.git" ]]; then
  git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git "$project_root/third_party/pytorch3d"
fi
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 python -m pip install --no-build-isolation "$project_root/third_party/pytorch3d"
mkdir -p "$project_root/outputs/environment"
git -C "$project_root/third_party/pytorch3d" rev-parse HEAD > "$project_root/outputs/environment/pytorch3d-$model-commit.txt"
python -m pip freeze > "$project_root/outputs/environment/$model-freeze.txt"
python -m facebench doctor --render-test --require-cuda --out "$project_root/outputs/environment/$model.json"
echo "CUDA/renderer checks complete. Model assets and full model forward/backward are separate checks."
