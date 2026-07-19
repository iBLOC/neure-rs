#!/usr/bin/env bash
#
# Build + verify the flash-attn feature in environments with a CUDA
# toolkit installed.
#
# The flash-attn feature pulls in candle-flash-attn (CUDA-only via
# cudarc) and candle-transformers/flash-attn (which itself enables
# candle-transformers/cuda). At runtime the vendored Qwen2/3 modules
# under src/llm/vendor/ and the upstream Llama / Mistral backends
# will call candle_flash_attn::flash_attn instead of the sdpa path.
#
# Usage:
#   scripts/build-flash-attn.sh                # default: check + build + test
#   scripts/build-flash-attn.sh check          # only cargo check
#   scripts/build-flash-attn.sh build          # only cargo build --lib
#   scripts/build-flash-attn.sh test           # only cargo test --lib
#   scripts/build-flash-attn.sh smoke          # cargo check on examples/
#                                              # to confirm vendored model
#                                              # APIs resolve under flash-attn
#
# Exit codes:
#   0   success
#   10  prerequisites missing (nvcc not found)
#   11  nvcc found but CUDA version unsupported
#   20  cargo check failed
#   21  cargo build failed
#   22  cargo test failed
#   23  smoke check failed
#
# Environment overrides:
#   FEATURES   extra cargo features to combine with flash-attn
#              (default: empty; examples: "asr-audio", "voxcpm")
#   NVCC       path to the nvcc binary (default: nvcc on PATH)
#   CUDA_HOME  CUDA toolkit root (default: auto-detected if NVCC is set)

set -euo pipefail

# ---- help ----------------------------------------------------------------

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
    sed -n '2,33p' "$0"
    exit 0
fi

# ---- locate nvcc ---------------------------------------------------------

if [[ -n "${NVCC:-}" ]]; then
    if [[ ! -x "$NVCC" ]]; then
        echo "error: NVCC=$NVCC is not executable" >&2
        exit 10
    fi
else
    if ! command -v nvcc >/dev/null 2>&1; then
        cat >&2 <<'EOF'
error: nvcc not found on PATH.

flash-attn requires the CUDA toolkit (the cudarc build script invokes
`nvcc --version`). Install CUDA first, for example:

    Ubuntu / Debian:
        sudo apt-get install nvidia-cuda-toolkit

    Or via the NVIDIA repo (recommended for newer CUDA versions):
        https://developer.nvidia.com/cuda-downloads

    Conda:
        conda install -c conda-forge cuda-nvcc cudatoolkit-dev

Then re-run this script. If nvcc is installed in a non-standard location,
point NVCC= at it, e.g.:

    NVCC=/usr/local/cuda/bin/nvcc scripts/build-flash-attn.sh
EOF
        exit 10
    fi
    NVCC="$(command -v nvcc)"
fi

# ---- report CUDA version -------------------------------------------------

cuda_version="$("$NVCC" --version | sed -n 's/^.*release \([0-9.]*\).*$/\1/p')"
if [[ -z "$cuda_version" ]]; then
    echo "error: could not parse CUDA version from: $("$NVCC" --version)" >&2
    exit 11
fi

cuda_major="${cuda_version%%.*}"
if (( cuda_major < 11 )); then
    echo "error: CUDA $cuda_version is too old (flash-attn requires >= 11.0)" >&2
    exit 11
fi

echo "==> nvcc:        $NVCC"
echo "==> CUDA:        $cuda_version"
echo "==> features:    flash-attn${FEATURES:+ $FEATURES}"
echo

# ---- feature matrix ------------------------------------------------------

FEATURE_FLAGS="--features flash-attn${FEATURES:+,$FEATURES}"
EXTRA_ENV=()

# Some cudarc builds honour CUDA_HOME for the toolkit root. If the
# caller did not set it, try to derive it from NVCC.
if [[ -z "${CUDA_HOME:-}" && "$NVCC" == */nvcc ]]; then
    candidate="$(dirname "$(dirname "$NVCC")")"
    if [[ -d "$candidate/lib64" || -d "$candidate/lib" ]]; then
        EXTRA_ENV+=("CUDA_HOME=$candidate")
    fi
fi

# ---- subcommands ---------------------------------------------------------

run_with_features() {
    if [[ ${#EXTRA_ENV[@]} -gt 0 ]]; then
        env "${EXTRA_ENV[@]}" "$@"
    else
        "$@"
    fi
}

run_check() {
    echo "==> cargo check --lib $FEATURE_FLAGS"
    run_with_features cargo check --lib "$FEATURE_FLAGS"
}

run_build() {
    echo "==> cargo build --lib $FEATURE_FLAGS"
    run_with_features cargo build --lib "$FEATURE_FLAGS"
}

run_test() {
    echo "==> cargo test --lib $FEATURE_FLAGS"
    run_with_features cargo test --lib "$FEATURE_FLAGS"
}

run_smoke() {
    echo "==> cargo check --examples $FEATURE_FLAGS"
    run_with_features cargo check --examples "$FEATURE_FLAGS"
}

cmd="${1:-all}"

case "$cmd" in
    check)
        run_check || { echo "cargo check failed" >&2; exit 20; }
        ;;
    build)
        run_build || { echo "cargo build failed" >&2; exit 21; }
        ;;
    test)
        run_test || { echo "cargo test failed" >&2; exit 22; }
        ;;
    smoke)
        run_smoke || { echo "smoke check failed" >&2; exit 23; }
        ;;
    all)
        run_check || { echo "cargo check failed" >&2; exit 20; }
        run_build || { echo "cargo build failed" >&2; exit 21; }
        run_test  || { echo "cargo test failed"  >&2; exit 22; }
        ;;
    *)
        echo "error: unknown subcommand '$cmd' (try: check | build | test | smoke | all)" >&2
        exit 2
        ;;
esac

echo
echo "==> flash-attn build OK (CUDA $cuda_version)"