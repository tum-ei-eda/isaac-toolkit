PROG := tvm

TVM_MODEL := ic

DATA_DIR := $(TVM_MODEL)_data
MLF_DIR := mlf

PROG_SRCS := *.c $(DATA_DIR)/*.cc $(MLF_DIR)/codegen/host/src/*.c -lm
PROG_INCS := -I$(MLF_DIR)/runtime/include -I$(MLF_DIR)/codegen/host/include
PROG_DEFS := -DMIN_RUNS=1 -DMAX_RUNS=1 -DCHECK=0

include ../common.mk
