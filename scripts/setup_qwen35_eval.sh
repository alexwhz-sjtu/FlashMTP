#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE_PYTHON:-${ROOT}/.venv/bin/python}"
ENV="${EVAL_VENV:-${ROOT}/.venv-qwen35-eval}"
"$BASE" -m venv --system-site-packages "$ENV"
SITE="$($ENV/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
BASE_SITE="$($BASE -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "$BASE_SITE" > "$SITE/base_dependencies.pth"
"$ENV/bin/python" -m pip install 'transformers==5.3.0'
"$ENV/bin/python" -m pip install --no-deps 'flash-linear-attention==0.5.2' 'fla-core==0.5.2'
"$ENV/bin/python" -m pip install --no-build-isolation --no-deps 'causal-conv1d==1.7.0'
"$ENV/bin/python" -c 'import torch, transformers; from transformers import Qwen3_5ForCausalLM; import causal_conv1d; from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule; print(torch.__version__, transformers.__version__)'
