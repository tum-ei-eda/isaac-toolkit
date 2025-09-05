#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "params.h"
#include "decrypt.h"


// Buffers in .bss / .rodata
static unsigned char s[SYND_BYTES];                   // ciphertext (syndrome)
static unsigned char e[SYS_N/8];                      // error vector output
static const unsigned char sk[IRR_BYTES + PK_NROWS * PK_ROW_BYTES] = {0}; // dummy secret key

int main(void) {
    printf("Running decrypt() with dummy data...\n");

    // Fill s with some dummy ciphertext if needed
    memset(s, 0, sizeof(s));

    // Run decryption
    int ret = decrypt(e, sk, s);

    // Lightweight correctness check
    unsigned long checksum = 0;
    for (size_t i = 0; i < sizeof(e); i++) checksum += e[i];

    printf("decryption return = %d\n", ret);
    printf("checksum(e) = %lu\n", checksum);

    return 0;
}
