#ifndef RANDOMBYTES_H
#define RANDOMBYTES_H

#include <stddef.h>

#ifndef PSEUDO_RNG
#include "nist/rng.h"
#else
void randombytes(unsigned char *buf, size_t n);
#endif  // PSEUDO_RNG

#endif  // RANDOMBYTES_H
