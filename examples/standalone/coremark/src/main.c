#include <stdio.h>

#if defined(SIM_VICUNA)
#include "uart.h"
#include "terminate_benchmark.h"
#define printf uart_printf
#endif

int coremark_init();
int coremark_run();
int coremark_deinit();
int coremark_check();

int main() {
    printf("Hello World!");
    int res = 0;
    res = coremark_init();
    res = coremark_run();
    res = coremark_deinit();
    res = coremark_check();
#if defined(SIM_VICUNA)
    if (res != 0)
    {
        uart_printf("Test Failed!\n");
        benchmark_failure();

    }
    else
    {
        uart_printf("Test Success!\n");
        benchmark_success();
    }
#endif
    return res;
}
