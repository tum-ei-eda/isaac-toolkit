/*
    This file include the global variable with track the cycle count of a function
    To add more performance tracker add the variable here and in the header file (with extern)
*/
#include "riscv_profiling.h"

//-------------------------
//        Keygen
//-------------------------
cycleCount_t keygenCycleCount = {0,0};
cycleCount_t keygenSKGenCount = {0,0};
cycleCount_t keygenPKGenCount = {0,0};
cycleCount_t keygenControlBitsCount = {0,0};
//-------------------------
//        Encrypt
//-------------------------
cycleCount_t encryptCycleCount = {0,0};
cycleCount_t encryptGenECount = {0,0};
cycleCount_t encryptSyndromeCount = {0,0};
//-------------------------
//        Decrypt
//-------------------------
cycleCount_t decryptCycleCount = {0,0};
cycleCount_t decryptBerlekampDecCount = {0,0};
cycleCount_t decryptReencryptionCount = {0,0};
cycleCount_t decryptWeigthCheckCount = {0,0};
//-------------------------
//     Misc Functions
//-------------------------
cycleCount_t int32_sortCount = {0,0};
cycleCount_t gf_mulCount = {0,0};
cycleCount_t bmCount = {0,0};
cycleCount_t benesCount = {0,0};
cycleCount_t fftCount = {0,0};
cycleCount_t fft_trCount = {0,0};
