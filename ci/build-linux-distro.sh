#!/usr/bin/env bash
set -euo pipefail

: "${DISTRO:?DISTRO is required}"
: "${RELEASE_VERSION:=dev}"
ROOT=/src
OUT=/out

case "$DISTRO" in
  debian12|debian13)
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git procps python3 python3-pip python3-venv
    ;;
  fedora)
    dnf -y install gcc gcc-c++ make ca-certificates curl git procps-ng python3 python3-pip
    ;;
  arch)
    pacman -Syu --noconfirm --needed base-devel ca-certificates curl git procps-ng python python-pip
    ;;
  *) echo "unsupported build distro: $DISTRO" >&2; exit 2 ;;
esac

# Install a pinned NVIDIA CUDA 13.5 compiler/toolkit component set. This is build-only;
# the published binary uses the host's NVIDIA driver at runtime.
VENV=/tmp/cuda-build-venv
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
  nvidia-cuda-nvcc==13.5.37 \
  nvidia-cuda-runtime==13.5.37 \
  nvidia-cuda-crt==13.5.37 \
  nvidia-cuda-cccl==13.3.5.0.6
NVCC=$(find "$VENV" -type f -name nvcc -print -quit)
[[ -x "$NVCC" ]] || { echo 'CUDA 13.5 nvcc was not installed' >&2; exit 1; }
TOOLKIT_ROOT=$(dirname "$(dirname "$NVCC")")

# build.rs expects a CUDA_HOME-style tree and a link-time libcuda stub. Reuse the
# official NVIDIA CUDA image's stub; it is never included in the release archive.
CUDA_HOME=/tmp/bpm-cuda-home
mkdir -p "$CUDA_HOME/lib64/stubs" "$CUDA_HOME/targets/x86_64-linux/lib/stubs"
ln -sfn "$TOOLKIT_ROOT/include" "$CUDA_HOME/include"
ln -sfn /tmp/libcuda.so "$CUDA_HOME/lib64/stubs/libcuda.so"
export CUDA_HOME NVCC
export CARGO_HOME=/tmp/cargo-home RUSTUP_HOME=/tmp/rustup-home
export PATH="$CARGO_HOME/bin:$PATH"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
  sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path

cd "$ROOT"
export CARGO_TARGET_DIR=/tmp/bpm-target
cargo test --release
cargo build --release --bin bpm
/tmp/bpm-target/release/bpm --version
/tmp/bpm-target/release/bpm --help >/dev/null

version=${RELEASE_VERSION//\//-}
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
install -m 0755 /tmp/bpm-target/release/bpm "$stage/bpm"
cp README.md LICENSE THIRD_PARTY_NOTICES.md "$stage/"
printf 'version=%s\ndistro=%s\narchitecture=x86_64\ncuda_toolkit=13.5.37\n' \
  "$version" "$DISTRO" > "$stage/BUILD-INFO.txt"
mkdir -p "$OUT"
archive="$OUT/bpm-${version}-${DISTRO}-x86_64.tar.gz"
tar -czf "$archive" -C "$stage" .
(cd "$OUT" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
printf 'Created %s\n' "$archive"
