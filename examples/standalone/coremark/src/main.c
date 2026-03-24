#include <stdio.h>

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
    return res;
}
