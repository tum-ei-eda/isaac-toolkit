#include <stdio.h>
#include <string.h>

#include "encrypt.h"
#include "params.h"


// error vector and syndrome live in .bss (auto-zeroed at startup)
static unsigned char s[SYND_BYTES];         // 96 bytes
static unsigned char e[SYS_N/8];            // 436 bytes

// put pk into .rodata as a const array (compiler/linker handles initialization)
static const unsigned char pk[PK_NROWS * PK_ROW_BYTES] = { 0 };

// optional: a small prefilled pattern instead of all-zeros
// static const unsigned char pk[PK_NROWS * PK_ROW_BYTES] = {
//     0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF
// };

int main(void) {
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
    unsigned long checksum_e = 0;
    for (int i = 0; i < sizeof(e); i++) checksum_e += e[i];
#ifdef PRINT
    printf("checksum s = %lu\n", checksum_s);
    printf("checksum e = %lu\n", checksum_e);
#else
    // prevent optimization
    if (checksum_s == 0xDEADBEEF) __asm__ volatile("nop");
    if (checksum_e == 0xDEADBEEF) __asm__ volatile("nop");
#endif  // PRINT

    return 0;
}
