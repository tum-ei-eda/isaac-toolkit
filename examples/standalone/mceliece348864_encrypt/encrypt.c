/*
  This file is for Niederreiter encryption
*/

#include "encrypt.h"

#include "gf.h"
#include "util.h"
#include "params.h"
#include "uint16_sort.h"
#ifndef PSEUDO_RNG
#include "randombytes.h"
#endif  // !PSEUDO_RNG

#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "crypto_declassify.h"
#include "crypto_uint16.h"
#include "crypto_uint32.h"

#ifdef RISCV
#include "riscv_profiling.h"
#include "syndrome_rvv.h"
#endif

static inline crypto_uint16 uint16_is_smaller_declassify(uint16_t t,uint16_t u)
{
  crypto_uint16 mask = crypto_uint16_smaller_mask(t,u);
  crypto_declassify(&mask,sizeof mask);
  return mask;
}

static inline crypto_uint32 uint32_is_equal_declassify(uint32_t t,uint32_t u)
{
  crypto_uint32 mask = crypto_uint32_equal_mask(t,u);
  crypto_declassify(&mask,sizeof mask);
  return mask;
}


#ifdef PSEUDO_RNG
// Simple xorshift64* PRNG (not cryptographically strong)
static uint64_t rng_state = 88172645463325252ull; // default seed, replace with entropy

static uint64_t xorshift64star(void) {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 2685821657736338717ull;
}

void randombytes(unsigned char *buf, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if ((i & 7) == 0) {
            // Generate 64 bits at a time
            uint64_t rnd = xorshift64star();
            for (int j = 0; j < 8 && i + j < n; j++) {
                buf[i + j] = (rnd >> (8 * j)) & 0xFF;
            }
        }
    }
}
#endif  // PSEUDO_RNG

/* output: e, an error vector of weight t */
static void gen_e(unsigned char *e)
{
	int i, j, eq, count;

	union
	{
		uint16_t nums[ SYS_T*2 ];
		unsigned char bytes[ SYS_T*2 * sizeof(uint16_t) ];
	} buf;

	uint16_t ind[ SYS_T ];
	uint64_t e_int[ (SYS_N+63)/64 ];
	uint64_t one = 1;
	uint64_t mask;
	uint64_t val[ SYS_T ];

	while (1)
	{
		randombytes(buf.bytes, sizeof(buf));

		for (i = 0; i < SYS_T*2; i++)
			buf.nums[i] = load_gf(buf.bytes + i*2);

		// moving and counting indices in the correct range

		count = 0;
		for (i = 0; i < SYS_T*2 && count < SYS_T; i++)
			if (uint16_is_smaller_declassify(buf.nums[i],SYS_N))
				ind[ count++ ] = buf.nums[i];

		if (count < SYS_T) continue;

		// check for repetition

		uint16_sort(ind, SYS_T);

		eq = 0;
		for (i = 1; i < SYS_T; i++)
			if (uint32_is_equal_declassify(ind[i-1],ind[i]))
				eq = 1;

		if (eq == 0)
			break;
	}

	for (j = 0; j < SYS_T; j++)
		val[j] = one << (ind[j] & 63);

	for (i = 0; i < (SYS_N+63)/64; i++)
	{
		e_int[i] = 0;

		for (j = 0; j < SYS_T; j++)
		{
			mask = i ^ (ind[j] >> 6);
			mask -= 1;
			mask >>= 63;
			mask = -mask;

			e_int[i] |= val[j] & mask;
		}
	}

	for (i = 0; i < (SYS_N+63)/64 - 1; i++)
		{ store8(e, e_int[i]); e += 8; }

	for (j = 0; j < (SYS_N % 64); j+=8)
		e[ j/8 ] = (e_int[i] >> j) & 0xFF;
}

/* input: public key pk, error vector e */
/* output: syndrome s */
static void syndrome(unsigned char *s, const unsigned char *pk, unsigned char *e)
{
	uint64_t b;

	const uint64_t *pk_ptr;
	const uint64_t *e_ptr = ((uint64_t *) (e + SYND_BYTES));

	int i, j;

	//

	for (i = 0; i < SYND_BYTES; i++)
		s[i] = e[i];

	for (i = 0; i < PK_NROWS; i++)
	{
		pk_ptr = ((uint64_t *) (pk + PK_ROW_BYTES * i));

		b = 0;
		for (j = 0; j < PK_NCOLS/64; j++)
			b ^= pk_ptr[j] & e_ptr[j];

		b ^= ((uint32_t *) &pk_ptr[j])[0] & ((uint32_t *) &e_ptr[j])[0];

		b ^= b >> 32;
		b ^= b >> 16;
		b ^= b >> 8;
		b ^= b >> 4;
		b ^= b >> 2;
		b ^= b >> 1;
		b &= 1;

		s[ i/8 ] ^= (b << (i%8));
	}
}

void encrypt(unsigned char *s, const unsigned char *pk, unsigned char *e)
{
	#ifdef RISCV
	start_count(&encryptGenECount);
	#endif
	gen_e(e);
	#ifdef RISCV
	end_count(&encryptGenECount);
	#endif

#ifdef KAT
  {
    int k;
    printf("encrypt e: positions");
    for (k = 0;k < SYS_N;++k)
      if (e[k/8] & (1 << (k&7)))
        printf(" %d",k);
    printf("\n");
  }
#endif

	#ifdef RISCV
	start_count(&encryptSyndromeCount);
	#endif
	#ifdef RISCV
	syndrome_rvv(s, pk, e);
	#else
	syndrome(s, pk, e);
	#endif
	#ifdef RISCV
	end_count(&encryptSyndromeCount);
	#endif
}
