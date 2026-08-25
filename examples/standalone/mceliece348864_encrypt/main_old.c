#include <stdio.h>
#include <string.h>

#include "encrypt.h"
#include "params.h"

int main(void) {
    // Buffers with correct sizes
    unsigned char s[SYND_BYTES];                      // syndrome (96 bytes)
    unsigned char pk[PK_NROWS * PK_ROW_BYTES];        // public key (~261 KB)
    unsigned char e[SYS_N/8];                         // error vector (436 bytes)

    // Initialize pk with dummy data
    // for (size_t i = 0; i < sizeof(pk); i++) {
    //     pk[i] = (unsigned char)(i & 0xFF);
    // }

    // Clear s and e
    memset(s, 0, sizeof(s));
    memset(e, 0, sizeof(e));

    // printf("Running encrypt() with dummy data...\n");

    encrypt(s, pk, e);

    // Print syndrome (truncated)
    // printf("syndrome: ");
    // for (int i = 0; i < SYND_BYTES; i++) {
    //     printf("%02X", s[i]);
    // }
    // printf("\n");

    // // Print error vector (truncated)
    // printf("error vector e: ");
    // for (int i = 0; i < 16 && i < sizeof(e); i++) {   // only first 16 bytes
    //     printf("%02X", e[i]);
    // }
    // printf(" ...\n");
    unsigned long checksum_s = 0;
    for (int i = 0; i < sizeof(s); i++) checksum_s += s[i];
    printf("checksum s = %lu\n", checksum_s);
    unsigned long checksum_e = 0;
    for (int i = 0; i < sizeof(e); i++) checksum_e += e[i];
    printf("checksum e = %lu\n", checksum_e);

    return 0;
}
