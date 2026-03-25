#include "support.h"
#include <stdio.h>

#if defined(SIM_VICUNA)
#include "uart.h"
#include "terminate_benchmark.h"
#define printf uart_printf
#endif

void
initialise_board ()
{
}

void __attribute__ ((noinline)) __attribute__ ((externally_visible))
start_trigger ()
{
}

void __attribute__ ((noinline)) __attribute__ ((externally_visible))
stop_trigger ()
{
}

volatile int result = 0;
int correct = 0;

int main() {
  initialise_board();
  initialise_benchmark();
  warm_caches(WARMUP_HEAT);
  start_trigger();
  result = benchmark();
  stop_trigger();
  correct = verify_benchmark(result);
  if (!correct) {
#if defined(SIM_VICUNA)
      benchmark_failure();
#endif
      return -1;
  }
#if defined(SIM_VICUNA)
  benchmark_success();
#endif
  return 0;
}
