PROG := mceliece348864_encrypt
PROG_SRCS := main.c encrypt.c riscv_profiling.c syndrome_rvv.c
# PROG_DEFS :=
PROG_DEFS := -fno-unroll-loops "-DCRYPTO_NAMESPACE(x)=x" "-D_CRYPTO_NAMESPACE(x)=_\#\#x" -DPSEUDO_RNG -Isubroutines -DRISCV

include ../common.mk
