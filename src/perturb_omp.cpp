#include <cstdlib>
#include <random>
#include <omp.h>

extern "C" {
void perturb_zeros(
    int **zero_list,
    int *row_sizes,
    int rows,
    double p,
    int **output_rows,
    int *output_sizes
) {
    // OpenMP 병렬 영역 시작
    #pragma omp parallel
    {
        std::mt19937 gen(std::random_device{}() + omp_get_thread_num());  // 각 쓰레드마다 독립된 seed
        std::uniform_real_distribution<> dis(0.0, 1.0);

        #pragma omp for
        for (int i = 0; i < rows; ++i) {
            int *row_indices = zero_list[i];
            int row_size = row_sizes[i];

            int *temp_row = (int *)malloc(row_size * sizeof(int));
            if (!temp_row) continue;

            int count = 0;
            for (int j = 0; j < row_size; ++j) {
                if (dis(gen) < p) {
                    temp_row[count++] = row_indices[j];
                }
            }

            if (count > 0) {
                int *final_row = (int *)realloc(temp_row, count * sizeof(int));
                output_rows[i] = (final_row != NULL) ? final_row : temp_row;
                output_sizes[i] = count;
            } else {
                free(temp_row);
                output_rows[i] = NULL;
                output_sizes[i] = 0;
            }
        }
    }
}
}
