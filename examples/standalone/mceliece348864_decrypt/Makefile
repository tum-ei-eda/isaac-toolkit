PROG := mceliece348864_decrypt
PROG_SRCS := main.c decrypt.c fft.c fft_tr.c benes.c vec.c bm.c gf.c
# PROG_DEFS :=
PROG_DEFS := -fno-unroll-loops "-DCRYPTO_NAMESPACE(x)=x" "-D_CRYPTO_NAMESPACE(x)=_\#\#x" -DPSEUDO_RNG -Isubroutines

include ../common.mk
