BENCH := crc32
# EMBENCH2 := 1

PROG := coremark
PROG_SRCS := main.c embench-iot/src/xgboost/*.c embench-iot/support/beebsc.c
PROG_INCS := -Iembench-iot/support
PROG_DEFS := -DCPU_MHZ=1 -DGLOBAL_SCALE_FACTOR=1 -DWARMUP_HEAT=1

include ../common.mk
