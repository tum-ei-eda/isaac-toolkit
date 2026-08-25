/*
    This file implementes the syndrom function used in the encryption step
    using the RISC-V Vector extension.
    Note, the embedded extension Zve32x is used, which restrict some operation
    and vector sizes. 
*/

#include "syndrome_rvv.h"

// Some constant for the given parameter set
#include "params.h"
#include <stdint.h>

#ifdef RISCV
#include "riscv_profiling.h"
#include <riscv_vector.h> /* __riscv_v_intrinsic */
#endif

static inline vuint8m1_t __riscv_vcpop_v_u8m1(vuint8m1_t op1, size_t vl){
	vuint8m1_t temp;
	asm volatile inline(
		"vsetvli zero, %1, e8, m1, ta, ma \n\t"
		"vcpop.v %0, %2 \n\t"
		: "=&vd"(temp)
		: "r"(vl), "vd"(op1));
	return temp;
}

void print_vec(vuint8m1_t vec, size_t vl){
	vuint8m1_t temp = __riscv_vmv_v_v_u8m1(vec, vl);
	for (int i = 0; i < vl; i++)
	{
		uint8_t ele = __riscv_vmv_x_s_u8m1_u8(temp);
		printf("Element %d has value %02x\n", i, ele);
		temp = __riscv_vslidedown_vx_u8m1(temp, 1, vl);
	}
}

/* input: public key pk, error vector e */
/* output: syndrome s */
void syndrome_rvv(unsigned char *s, const unsigned char *pk, unsigned char *e)
{
	/*for (i = 0; i < SYND_BYTES; i++)
		s[i] = e[i];
	*/
	unsigned char *destination = s;
	const unsigned char *source = e;
	size_t remainingBytes = SYND_BYTES;
	for (size_t vl; remainingBytes > 0; remainingBytes -= vl, source += vl, destination += vl) {
		// Number of vector element for next iteration
		vl = __riscv_vsetvl_e8m8(remainingBytes);
		// Load error vector
		vuint8m8_t errorVector = __riscv_vle8_v_u8m8(source, vl);
		// Store to syndrome
		__riscv_vse8_v_u8m8(destination, errorVector, vl);
		// Decrement the remaining bytes by vl
		// Incement the pointer of error vector and syndrome
	}

	/*uint64_t b;
	const uint64_t *pk_ptr; 
	const uint64_t *e_ptr = ((uint64_t *) (e + SYND_BYTES));
	int i, j;
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
	}*/

	//#ifdef NOT_READY 
	const unsigned char *pk_ptr = pk;
	const unsigned char *e_prt = e + SYND_BYTES;
	size_t vlmax = __riscv_vsetvlmax_e8m1();
	// Preload some masks
	uint8_t bitShifts[8] = {0, 1, 2, 3, 4, 5, 6, 7};
	vuint8m1_t shiftVector = __riscv_vmv_v_x_u8m1(0, vlmax);
	shiftVector = __riscv_vle8_v_u8m1(bitShifts, 8);
	//vbool8_t shiftMask = __riscv_vmsltu_vx_u8m1_b8(shiftVector, 8, 8);

	unsigned int syndromeIndex = 0;
	size_t remainingRows = PK_NROWS;
	
	// Do multiple rows at the same time
	for (size_t vlRow; remainingRows > 0; remainingRows-=vlRow, pk_ptr+=vlRow*PK_ROW_BYTES){
		// Force the number of rows to be a multiple of 8
		// This is due to the easier writing to the syndrome
		vlRow = __riscv_vsetvl_e8m1(remainingRows) & ~0x3;
		//printf("vl %d\n", vlRow);
		//printf("remainingRows %d\n", remainingRows);
		// Set accumulator to 0
		vuint8m1_t acc = __riscv_vmv_v_x_u8m1(0, vlmax);

		// Row times column
		const unsigned char *pk_col = pk_ptr;
		for (int j = 0; j < PK_NCOLS / 8; j++, pk_col++){
			// Load error and pk for each element
			vuint8m1_t errorVector = __riscv_vmv_v_x_u8m1(e_prt[j], vlRow);
			//printf("Error Vector\n");
			//print_vec(errorVector, vlRow);
			vuint8m1_t pkVector = __riscv_vlse8_v_u8m1(pk_col, PK_ROW_BYTES, vlRow);
			//printf("PK Vector\n");
			//print_vec(pkVector, vlRow);
			vuint8m1_t mulVector = __riscv_vand_vv_u8m1(errorVector, pkVector, vlRow);
			//printf("Mul Vector\n");
			//print_vec(mulVector, vlRow);
			acc = __riscv_vxor_vv_u8m1(acc, mulVector, vlRow);
			//printf("Acc Vector\n");
			//print_vec(acc, vlRow);
		}

		// Even parity (xor every bit)
		//printf("Pre pop count\n");
		//print_vec(acc, vlRow);
		acc = __riscv_vcpop_v_u8m1(acc, vlRow);
		//printf("Post pop count\n");
		//print_vec(acc, vlRow);
		acc = __riscv_vand_vx_u8m1(acc, 1, vlRow);

		// Write result to syndrome
		for (int i = 0; i < vlRow; i+=8, syndromeIndex++){
			// Shift to the respective position in the syndrome
			vuint8m1_t redRes = __riscv_vsll_vv_u8m1(acc, shiftVector, 8);
			// Compress the bits to a byte and write to syndrome
			redRes = __riscv_vredor_vs_u8m1_u8m1(redRes, redRes, 8);
			uint8_t syndRes = __riscv_vmv_x_s_u8m1_u8(redRes);
			//printf("syndromeIndex %d\n",syndromeIndex);
			//printf("Had %d\n", s[syndromeIndex]);
			//printf("XORing %x\n", syndRes);
			s[syndromeIndex] ^= syndRes;
			//printf("Now %x\n", s[syndromeIndex]);
			// Shift the vector to the next batch of eigth
			acc = __riscv_vslidedown_vx_u8m1(acc, 8, vlRow);
		}
	}
	//#endif
}