#include "randombytes.h"

#ifdef PSEUDO_RNG
#include <stddef.h>
#include <stdint.h>

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
