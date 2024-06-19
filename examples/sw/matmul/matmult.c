#include <stdint.h>
void mat_mult(const int8_t *mat_l, const int8_t *mat_r, int *result, const unsigned int N, const unsigned int K, const unsigned int M)
{
    unsigned int n, k, m;
    unsigned int row, col;
    int accumulator;

    for (m = 0; m < M; m++)
    {
        for (n = 0; n < N; n++)
        {
            row = n*K;
            accumulator = 0;
            for (k = 0; k < K; k++)
            {
                col = k*M;
                accumulator += mat_l[row + k] * mat_r[col + m];
            }
            result[n*M + m] = accumulator;
        }
    }
}
