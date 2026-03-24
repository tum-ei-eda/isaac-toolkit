#include "support.h"
#include <stdio.h>

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
      return -1;
  }
  return 0;
}
