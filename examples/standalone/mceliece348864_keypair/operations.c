#include "operations.h"

#include "controlbits.h"
#include "randombytes.h"
#include "crypto_hash.h"
// #include "encrypt.h"
// #include "decrypt.h"
#include "params.h"
#include "sk_gen.h"
#include "pk_gen.h"
#include "util.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#ifdef RISCV
#include "riscv_profiling.h"
#endif


// int crypto_kem_enc(
//        unsigned char *c,
//        unsigned char *key,
//        const unsigned char *pk
// )
// {
// 	unsigned char e[ SYS_N/8 ];
// 	unsigned char one_ec[ 1 + SYS_N/8 + SYND_BYTES ] = {1};
//
// 	//
// 	#ifdef RISCV
// 	start_count(&encryptCycleCount);
// 	#endif
// 	encrypt(c, pk, e);
// 	#ifdef RISCV
// 	end_count(&encryptCycleCount);
// 	#endif
//
// 	memcpy(one_ec + 1, e, SYS_N/8);
// 	memcpy(one_ec + 1 + SYS_N/8, c, SYND_BYTES);
//
// 	crypto_hash_32b(key, one_ec, sizeof(one_ec));
//
// 	return 0;
// }
//
// int crypto_kem_dec(
//        unsigned char *key,
//        const unsigned char *c,
//        const unsigned char *sk
// )
// {
// 	int i;
//
// 	unsigned char ret_decrypt = 0;
//
// 	uint16_t m;
//
// 	unsigned char e[ SYS_N/8 ];
// 	unsigned char preimage[ 1 + SYS_N/8 + SYND_BYTES ];
// 	unsigned char *x = preimage;
// 	const unsigned char *s = sk + 40 + IRR_BYTES + COND_BYTES;
//
// 	//
// 	#ifdef RISCV
// 	start_count(&decryptCycleCount);
// 	#endif
// 	ret_decrypt = decrypt(e, sk + 40, c);
// 	#ifdef RISCV
// 	end_count(&decryptCycleCount);
// 	#endif
//
// 	m = ret_decrypt;
// 	m -= 1;
// 	m >>= 8;
//
// 	*x++ = m & 1;
// 	for (i = 0; i < SYS_N/8; i++)
// 		*x++ = (~m & s[i]) | (m & e[i]);
//
// 	for (i = 0; i < SYND_BYTES; i++)
// 		*x++ = c[i];
//
// 	crypto_hash_32b(key, preimage, sizeof(preimage));
//
// 	return 0;
// }

int crypto_kem_keypair
(
       unsigned char *pk,
       unsigned char *sk
)
{
	int i;
	static unsigned char seed[ 33 ] = {64};
	static unsigned char r[ SYS_N/8 + (1 << GFBITS)*sizeof(uint32_t) + SYS_T*2 + 32 ];
	unsigned char *rp, *skp;

	static gf f[ SYS_T ]; // element in GF(2^mt)
	static gf irr[ SYS_T ]; // Goppa polynomial
	static uint32_t perm[ 1 << GFBITS ]; // random permutation as 32-bit integers
	static int16_t pi[ 1 << GFBITS ]; // random permutation

	#ifdef RISCV
	start_count(&keygenCycleCount);
	#endif

	randombytes(seed+1, 32);

	while (1)
	{
    printf("loop\n");
		#ifdef RISCV
		start_count(&keygenSKGenCount);
		#endif

		rp = &r[ sizeof(r)-32 ];
		skp = sk;

		// expanding and updating the seed

    printf("shake\n");
		shake(r, sizeof(r), seed, 33);
		memcpy(skp, seed+1, 32);
		skp += 32 + 8;
		memcpy(seed+1, &r[ sizeof(r)-32 ], 32);

		// generating irreducible polynomial

		rp -= sizeof(f);

    printf("load_gf\n");
		for (i = 0; i < SYS_T; i++)
			f[i] = load_gf(rp + i*2);

    printf("genpoly_gen\n");
		if (genpoly_gen(irr, f)) {
			#ifdef RISCV
			end_count(&keygenSKGenCount);
			#endif
      printf("cont\n");
			continue;
		}

    printf("store_gf\n");
		for (i = 0; i < SYS_T; i++)
			store_gf(skp + i*2, irr[i]);

		skp += IRR_BYTES;

		#ifdef RISCV
		end_count(&keygenSKGenCount);
		#endif

		// generating permutation

		#ifdef RISCV
		start_count(&keygenPKGenCount);
		#endif
		rp -= sizeof(perm);

    printf("load4\n");
		for (i = 0; i < (1 << GFBITS); i++)
			perm[i] = load4(rp + i*4);

    printf("pk_gen\n");
		if (pk_gen(pk, skp - IRR_BYTES, perm, pi)){
			#ifdef RISCV
			end_count(&keygenPKGenCount);
			#endif
      printf("cont\n");
			continue;
		}
    printf("after pk_gen\n");

		#ifdef RISCV
		end_count(&keygenPKGenCount);
		#endif

		#ifdef RISCV
		start_count(&keygenControlBitsCount);
		#endif

		controlbitsfrompermutation(skp, pi, GFBITS, 1 << GFBITS);
		skp += COND_BYTES;

		#ifdef RISCV
		end_count(&keygenControlBitsCount);
		#endif

		// storing the random string s

		rp -= SYS_N/8;
		memcpy(skp, rp, SYS_N/8);

		// storing positions of the 32 pivots

    printf("store8\n");
		store8(sk + 32, 0xFFFFFFFF);

		break;
	}

	#ifdef RISCV
	end_count(&keygenCycleCount);
	#endif

	return 0;
}
