#!/usr/bin/env bash
# Run ONCE on a fresh instance whose image lacks system tools (e.g. the bare cuda:13.0.3 GPU base).
# Installs the system-level deps the autoresearch loop needs. The Python env itself comes from the
# mounted home (model/.venv) — this only covers things the container image must provide.
#   bash research/bootstrap_vm.sh
# (To bake these into the image instead so no per-boot install is needed, see research/Dockerfile.)
set -e

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  command -v sudo >/dev/null 2>&1 && SUDO=sudo || { echo "Not root and no sudo — ask the admin to bake git/build tools into the image (research/Dockerfile)."; exit 1; }
fi

$SUDO apt-get update
# git: the loop commits/branches every cycle. build-essential + pkg-config + CUDA toolkit (already in
# the -devel base): only needed if you have to `uv sync` rebuild the env. ffmpeg/curl/ca-certs: insurance.
$SUDO apt-get install -y --no-install-recommends \
  git ca-certificates curl ffmpeg build-essential pkg-config

echo "bootstrap done: $(git --version)"
