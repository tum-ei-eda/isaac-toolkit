#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define GFBITS 12
#define SYS_N 3488
#define SYS_T 64

#define PK_NROWS (SYS_T*GFBITS)
#define PK_NCOLS (SYS_N - PK_NROWS)
#define PK_ROW_BYTES ((PK_NCOLS + 7)/8)

#define SYND_BYTES ((PK_NROWS + 7)/8)

// #define NUM 1
#define NUM 10

static unsigned char *global_pk = NULL;
static unsigned char *global_e  = NULL;
static unsigned char *global_s  = NULL;

// #define RAND rand
#define RAND fake_rand

static inline int __attribute__((always_inline)) fake_rand() {
    static int cur = 0;
    cur += 1;
    return cur;
}

int mlonmcu_init() {
  srand(0xC0FFEE); // deterministic seed for reproducibility

  // Allocate global buffers
  global_pk = (unsigned char*)malloc(PK_ROW_BYTES * PK_NROWS);
  global_e  = (unsigned char*)malloc(SYS_N/8);
  global_s  = (unsigned char*)malloc(SYND_BYTES);

  // Fill public key with random data
  for (size_t j = 0; j < PK_ROW_BYTES * PK_NROWS; j++) {
    global_pk[j] = (unsigned char)(RAND() & 0xFF);
  }

  // Fill error vector with random data
  for (size_t j = 0; j < SYS_N/8; j++) {
    global_e[j] = (unsigned char)(RAND() & 0xFF);
  }

  return 0;
}

int mlonmcu_deinit() {
  free(global_pk);
  free(global_e);
  free(global_s);
  return 0;
}


/* input: public key pk, error vector e */
/* output: syndrome s */
void syndrome(unsigned char *s, const unsigned char *pk, unsigned char *e)
{
    unsigned char b, row[SYS_N/8];
    const unsigned char *pk_ptr = pk;

    int i, j;

    for (i = 0; i < SYND_BYTES; i++)
        s[i] = 0;

    for (i = 0; i < PK_NROWS; i++)
    {
        for (j = 0; j < SYS_N/8; j++)
            row[j] = 0;

        for (j = 0; j < PK_ROW_BYTES; j++)
            row[ SYS_N/8 - PK_ROW_BYTES + j ] = pk_ptr[j];

        row[i/8] |= 1 << (i%8);

        b = 0;
        for (j = 0; j < SYS_N/8; j++)
            b ^= row[j] & e[j];

        b ^= b >> 4;
        b ^= b >> 2;
        b ^= b >> 1;
        b &= 1;

        s[ i/8 ] |= (b << (i%8));

        pk_ptr += PK_ROW_BYTES;
    }
}

int mlonmcu_run() {
  // printf("RUN\n");
  for (size_t i = 0; i < NUM; i++) {
    // printf("i=%u\n", i);
    syndrome(global_s, global_pk, global_e);
  }

  // Prevent compiler optimizing away results
  unsigned checksum = 0;
  for (size_t i = 0; i < SYND_BYTES; i++) {
    checksum += global_s[i];
  }
  printf("Checksum: %u\n", checksum);

  return 0;
}

int mlonmcu_check() {
  // Could validate expected checksum here
  return 0;
}

int main() {
  mlonmcu_init();
  mlonmcu_run();
  mlonmcu_check();
  mlonmcu_deinit();
  return 0;
}
