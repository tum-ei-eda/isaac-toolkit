#!/bin/bash

BENCH=$1
SIM=$2
shift 2
TARGETS=$@

# TODO: validate arguments
# TODO: directory tree?
# TODO: custom base dir
# TODO: vars

FACTOR=${FACTOR:-1}

BASE=${BASE:-$(pwd)}

BUILD_BASE=${BUILD_BASE:-$BASE}
BUILD_DIR=$BUILD_BASE/build_${BENCH}_${SIM}_x${FACTOR}/

SESS_BASE=${SESS_BASE:-$BASE}
SESS_DIR=$SESS_BASE/sess_${BENCH}_${SIM}_x${FACTOR}/

OUT_BASE=${OUT_BASE:-$BASE}
OUT_DIR=$OUT_BASE/out_${BENCH}_${SIM}_x${FACTOR}/

mkdir -p $OUT_BASE

echo OUT_BASE=$OUT_BASE

make SIMULATOR=$SIM FORCE=1 VERBOSE=1 TGC_BSP_DIR=/work/git/colabs/minres/Firmwares/bare-metal-bsp/ TGC_SRC_DIR=/work/git/colabs/minres/TGC-ISS/ OUT_DIR=$OUT_DIR BUILD_DIR=$BUILD_DIR SESS=$SESS_DIR BENCH=$BENCH GLOBAL_SCALE_FACTOR=$FACTOR $TARGETS

if [[ -f $OUT_DIR/stage_timings.csv ]]
then
  python3 ../scripts/gantt_stage_times.py $OUT_DIR/stage_timings.csv $OUT_DIR/stage_timings.md
fi
