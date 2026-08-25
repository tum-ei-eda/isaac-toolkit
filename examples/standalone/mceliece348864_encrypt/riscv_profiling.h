/*
    This file include funtions and macros for profiling code on RISC-V
    The global variables are defined in riscv_profiling.c
*/

#ifndef RISCV_PROFILING_H
#define RISCV_PROFILING_H

#include <stdint.h>
#include <stdio.h>

#ifdef RV32
#define F64S "%llu"
#else
#define F64S "%lu"
#endif

#define STDOUT

typedef struct{
    uint64_t startCycleCount;
    uint64_t totalCycleCount;
} cycleCount_t;

//-------------------------
//        Keygen
//-------------------------
extern cycleCount_t keygenCycleCount;
extern cycleCount_t keygenSKGenCount;
extern cycleCount_t keygenPKGenCount;
extern cycleCount_t keygenControlBitsCount;
//-------------------------
//        Encrypt
//-------------------------
extern cycleCount_t encryptCycleCount;
extern cycleCount_t encryptGenECount;
extern cycleCount_t encryptSyndromeCount;
//-------------------------
//        Decrypt
//-------------------------
extern cycleCount_t decryptCycleCount;
extern cycleCount_t decryptBerlekampDecCount;
extern cycleCount_t decryptReencryptionCount;
extern cycleCount_t decryptWeigthCheckCount;
//-------------------------
//     Misc Functions
//-------------------------
extern cycleCount_t int32_sortCount;
extern cycleCount_t gf_mulCount;
extern cycleCount_t bmCount;
extern cycleCount_t benesCount;
extern cycleCount_t fftCount;
extern cycleCount_t fft_trCount;

static inline uint64_t read_cycle()
{
    #ifdef RV32
        uint32_t h32, l32, temp;
        asm volatile inline(
            ".again%=: \n\t"
            "rdcycleh %0 \n\t"
            "rdcycle %1 \n\t"
            "rdcycleh %2 \n\t"
            "bne %0, %2, .again%= \n\t"
            : "=&r"(h32), "=&r"(l32), "=&r"(temp));
        uint64_t cycleCount = h32;
        return (cycleCount << 32) + l32;
    #else
       uint64_t cycleCount;
        asm volatile inline(
            "rdcycle %0 \n\t"
            : "=&r"(cycleCount));
        return cycleCount;
    #endif
}

static inline void start_count(cycleCount_t *cy){
    cy->startCycleCount = read_cycle();
}
static inline void end_count(cycleCount_t *cy){
    cy->totalCycleCount += read_cycle() - cy->startCycleCount;
}

static void report_cycles()
{
    FILE *fp_rep;

#ifndef STDOUT
    fp_rep = fopen(ARCHTYPE "_" ADDNAME ".csv", "w");
    if(!fp_rep)
        return;
#else
#define fp_rep stdout
#endif  // !STDOUT
    // CSV Header
    fprintf(fp_rep, "keygen total;SK generation;PK generation;control bits;"
        "encrypt total;generate error;syndrome;"
        "decrypt total;Berlekamp decoder;reencryption;weigthcheck;"
        "int32_sort;gf_mul;bm;benes;fft;fft_tr\n");
    // Performance values
    fprintf(fp_rep, F64S ";", keygenCycleCount.totalCycleCount);
    fprintf(fp_rep, F64S ";", keygenSKGenCount.totalCycleCount);
    fprintf(fp_rep, F64S ";", keygenPKGenCount.totalCycleCount);
    fprintf(fp_rep, F64S ";", keygenControlBitsCount.totalCycleCount);

    fprintf(fp_rep, F64S ";", encryptCycleCount.totalCycleCount);
    fprintf(fp_rep, F64S ";", encryptGenECount.totalCycleCount);
    fprintf(fp_rep, F64S ";", encryptSyndromeCount.totalCycleCount);

    fprintf(fp_rep, F64S ";", decryptCycleCount.totalCycleCount);
    fprintf(fp_rep, F64S ";", decryptBerlekampDecCount.totalCycleCount);
    fprintf(fp_rep, F64S ";", decryptReencryptionCount.totalCycleCount);
    fprintf(fp_rep, F64S ";", decryptWeigthCheckCount.totalCycleCount);

    fprintf(fp_rep, F64S ";", int32_sortCount.totalCycleCount);
    fprintf(fp_rep, F64S ";", gf_mulCount.totalCycleCount);
    fprintf(fp_rep, F64S ";", bmCount.totalCycleCount);
    fprintf(fp_rep, F64S ";", benesCount.totalCycleCount);
    fprintf(fp_rep, F64S ";", fftCount.totalCycleCount);
    fprintf(fp_rep, F64S    , fft_trCount.totalCycleCount);

    fclose(fp_rep);
}

#endif
