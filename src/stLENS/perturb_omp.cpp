#include <cstdlib>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>
#include <omp.h>

extern "C" {
void perturb_zeros(
    int *indptr,
    int *indices,
    int n_rows,
    int n_cols,
    double p,
    int **output_rows,
    int *output_sizes
) {
    #pragma omp parallel
    {
        std::mt19937 gen(std::random_device{}() + omp_get_thread_num());
        std::vector<int> zero_indices;
        zero_indices.reserve(static_cast<size_t>(n_cols));

        #pragma omp for
        for (int i = 0; i < n_rows; ++i) {
            int start = indptr[i];
            int end = indptr[i + 1];
            int nnz_row = end - start;
            int row_size = n_cols - nnz_row;

            if (row_size <= 0) {
                output_rows[i] = NULL;
                output_sizes[i] = 0;
                continue;
            }

            std::binomial_distribution<int> binom(row_size, p);
            int k = binom(gen);

            if (k <= 0) {
                output_rows[i] = NULL;
                output_sizes[i] = 0;
                continue;
            }

            zero_indices.clear();
            int nnz_ptr = start;
            for (int col = 0; col < n_cols; ++col) {
                if (nnz_ptr < end && indices[nnz_ptr] == col) {
                    ++nnz_ptr;
                } else {
                    zero_indices.push_back(col);
                }
            }

            std::unordered_set<int> picked;
            picked.reserve(static_cast<size_t>(k) * 2);
            for (int j = row_size - k; j < row_size; ++j) {
                std::uniform_int_distribution<int> uni(0, j);
                int t = uni(gen);
                if (picked.count(t)) picked.insert(j);
                else                 picked.insert(t);
            }

            int *temp_row = (int *)malloc(k * sizeof(int));
            if (!temp_row) {
                output_rows[i] = NULL;
                output_sizes[i] = 0;
                continue;
            }
            int count = 0;
            for (int local_idx : picked) {
                temp_row[count++] = zero_indices[local_idx];
            }
            std::sort(temp_row, temp_row + count);

            output_rows[i] = temp_row;
            output_sizes[i] = count;
        }
    }
}

void free_perturb_output(int **output_rows, int n_rows) {
    for (int i = 0; i < n_rows; ++i) {
        if (output_rows[i]) {
            free(output_rows[i]);
            output_rows[i] = NULL;
        }
    }
}
}
