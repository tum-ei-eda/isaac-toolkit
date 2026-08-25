#include <stdio.h>

int bench_microfrontend();

int main(int argc, char *argv[])
{
    int ret = bench_microfrontend();
    if (ret != 0)
    {
        printf("Test Failed!\n");

    }
    else
    {
        printf("Test Success!\n");
    }

    return ret;
}
