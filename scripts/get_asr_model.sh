#!/usr/bin/env bash
set -euo pipefail
mkdir -p models && cd models
DIR="sherpa-onnx-nemo-ctc-en-conformer-medium"
ARCH="$DIR.tar.bz2"
if [ ! -d "$DIR" ]; then
  wget -q "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$ARCH" -O "$ARCH"
  tar -xjf "$ARCH" && rm "$ARCH"
fi
echo "NeMo CTC English model ready at models/$DIR"
