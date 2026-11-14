#!/usr/bin/env bash
# Download dry speech (LibriSpeech dev-clean, CC BY 4.0) and measured RIRs
# (MIT McDermott Lab IR Survey, Traer & McDermott 2016) into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw && cd data/raw

if [ ! -d LibriSpeech/dev-clean ]; then
  echo "[data] LibriSpeech dev-clean (~340 MB)"
  curl -fL --retry 3 -o dev-clean.tar.gz https://www.openslr.org/resources/12/dev-clean.tar.gz
  tar xzf dev-clean.tar.gz && rm dev-clean.tar.gz
fi
if [ ! -d mit_ir/Audio ]; then
  echo "[data] MIT IR Survey (~11 MB, 271 RIRs)"
  curl -fL --retry 3 -o mit_ir.zip https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip
  unzip -q -o mit_ir.zip -d mit_ir && rm mit_ir.zip
fi
echo "[data] speech files: $(find LibriSpeech -name '*.flac' | wc -l | tr -d ' ')  RIRs: $(find mit_ir -name '*.wav' | wc -l | tr -d ' ')"
