#!/usr/bin/env bash
# End-to-end pipeline: data -> simulate -> train teacher -> distil student ->
# evaluate (sim + real) -> export int8 -> real-time demo on a file.
#
#   scripts/run_pipeline.sh                      # full run (configs/default.yaml)
#   CONFIG=configs/smoke.yaml scripts/run_pipeline.sh   # 2-minute smoke test
#   STAGES="eval export" scripts/run_pipeline.sh  # run selected stages only
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/default.yaml}
STAGES=${STAGES:-"data sim teacher student eval export demo"}
PY=${PY:-.venv/bin/python}
cfg() { $PY -c "from roomest.config import load_config; c=load_config('$CONFIG'); print(c$1)"; }
RUNS=$(cfg "['paths']['runs_dir']"); EXPORT=$(cfg "['paths']['export_dir']")
LOG=logs/$(basename "$CONFIG" .yaml); mkdir -p "$LOG"
has() { [[ " $STAGES " == *" $1 "* ]]; }
step() { echo; echo "=============== $* ==============="; }

if has data;    then step data;    scripts/download_data.sh; fi
if has sim;     then step simulate; $PY -m roomest.simulate --config "$CONFIG"; fi
if has teacher; then step "train teacher (CRNN)"
  $PY -m roomest.train --config "$CONFIG" --name crnn_teacher 2>&1 | tee "$LOG/crnn_teacher.log"; fi
if has student; then step "distil student (TinyTCN)"
  $PY -m roomest.train --config "$CONFIG" --name tcn_student train.model=tcn \
      train.teacher="$RUNS/crnn_teacher/best.pt" 2>&1 | tee "$LOG/tcn_student.log"
  step "student without distillation (ablation)"
  $PY -m roomest.train --config "$CONFIG" --name tcn_scratch train.model=tcn 2>&1 | tee "$LOG/tcn_scratch.log"; fi
if has eval; then
  for run in crnn_teacher tcn_student tcn_scratch; do
    [ -f "$RUNS/$run/best.pt" ] || continue
    step "evaluate $run"
    extra=""; [ "$run" = crnn_teacher ] && extra="--baselines"
    $PY -m roomest.evaluate --ckpt "$RUNS/$run/best.pt" --real $extra
  done; fi
if has export; then
  for run in tcn_student crnn_teacher; do
    step "export $run"; $PY -m roomest.export --ckpt "$RUNS/$run/best.pt"; done; fi
if has demo; then step "real-time demo on a file (fast mode)"
  WAV=$(find data/raw/LibriSpeech -name '*.flac' | sort | tail -1)
  RIR=$(find data/raw/mit_ir -iname '*Classroom*.wav' | sort | head -1)
  $PY -m roomest.realtime --model "$EXPORT/tcn_student" --wav "$WAV" --rir "$RIR" --fast --no-ui; fi
echo; echo "done."
