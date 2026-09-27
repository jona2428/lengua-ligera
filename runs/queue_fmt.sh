#!/bin/bash
# Cola: espera el QAT largo y luego entrena/evalúa los candidatos de formato propio.
cd "$(dirname "$0")/.."
B=target2/release
while tasklist //FI "IMAGENAME eq train.exe" | grep -q train.exe; do sleep 60; done
echo "== QAT largo terminado $(date)" >> runs/queue_fmt.log
for F in trit-e4-g16 trit-e4-g32 trit-e3-g8; do
  INIT=runs/base.llg QAT=$F STEPS=1500 LR=1e-3 WARMUP=100 THREADS=6 EVAL_EVERY=500 OUT=runs/qat_$F.llg $B/train.exe > runs/qat_$F.log 2>&1
  echo "== $F entrenado $(date): $(tail -1 runs/qat_$F.log)" >> runs/queue_fmt.log
done
for F in tern32_long:ternario-g32 trit-e4-g16:trit-e4-g16 trit-e4-g32:trit-e4-g32 trit-e3-g8:trit-e3-g8; do
  N=${F%%:*}; FMT=${F##*:}; M=runs/qat_$N.llg; [ "$N" = tern32_long ] && M=runs/qat_tern32_long.llg
  for AB in 0 8; do
    echo "== eval $N act_bits $AB" >> runs/queue_fmt.log
    ACT_BITS=$AB MODEL=$M FMT=$FMT OUT=runs/$N.lli EVAL_CHARS=2000000 THREADS=6 $B/int_check.exe >> runs/queue_fmt.log 2>&1
  done
done
echo "== FIN $(date)" >> runs/queue_fmt.log
