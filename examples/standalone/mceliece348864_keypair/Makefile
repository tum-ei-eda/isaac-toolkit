PROG := mceliece348864_keypair
PROG_SRCS := main.c operations.c sk_gen.c gf.c pk_gen.c controlbits.c benes.c vec.c fft.c fips202.c randombytes.c
# PROG_DEFS :=
PROG_DEFS := -fno-unroll-loops "-DCRYPTO_NAMESPACE(x)=x" "-D_CRYPTO_NAMESPACE(x)=_\#\#x" -DPSEUDO_RNG -Isubroutines

include ../common.mk
