#include <cstdlib>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>
#include <omp.h>

extern "C" {

struct CSRMatrix {
    int* indptr;
    int* indices;
    float* data;
    int nnz;
    int n_rows;
    int n_cols;
};

void free_csr(CSRMatrix* mat) {
    if (mat) {
        delete[] mat->indptr;
        delete[] mat->indices;
        delete[] mat->data;
        delete mat;
    }
}

CSRMatrix* sparse_rand_csr(int n_rows, int n_cols, double density) {
    std::vector<std::vector<int>> row_cols(n_rows);
    std::vector<int> row_nnz(n_rows, 0);

    #pragma omp parallel
    {
        std::mt19937 gen(std::random_device{}() + omp_get_thread_num());

        #pragma omp for
        for (int i = 0; i < n_rows; ++i) {
            std::binomial_distribution<int> binom(n_cols, density);
            int k = binom(gen);

            if (k <= 0) continue;

            if (k >= n_cols) {
                row_cols[i].resize(n_cols);
                for (int c = 0; c < n_cols; ++c) row_cols[i][c] = c;
                row_nnz[i] = n_cols;
                continue;
            }

            std::unordered_set<int> picked;
            picked.reserve(static_cast<size_t>(k) * 2);
            for (int j = n_cols - k; j < n_cols; ++j) {
                std::uniform_int_distribution<int> uni(0, j);
                int t = uni(gen);
                if (picked.count(t)) picked.insert(j);
                else                 picked.insert(t);
            }

            row_cols[i].assign(picked.begin(), picked.end());
            std::sort(row_cols[i].begin(), row_cols[i].end());
            row_nnz[i] = static_cast<int>(row_cols[i].size());
        }
    }

    std::vector<int> indptr(n_rows + 1, 0);
    for (int i = 0; i < n_rows; ++i) {
        indptr[i + 1] = indptr[i] + row_nnz[i];
    }
    int total_nnz = indptr[n_rows];

    CSRMatrix* mat = new CSRMatrix;
    mat->n_rows = n_rows;
    mat->n_cols = n_cols;
    mat->nnz = total_nnz;
    mat->indptr = new int[n_rows + 1];
    mat->indices = new int[total_nnz];
    mat->data = new float[total_nnz];

    std::copy(indptr.begin(), indptr.end(), mat->indptr);

    #pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
        int start = mat->indptr[i];
        int size = row_nnz[i];
        std::copy(row_cols[i].begin(), row_cols[i].end(), mat->indices + start);
        for (int j = 0; j < size; ++j) {
            mat->data[start + j] = 1.0f;
        }
    }

    return mat;
}
}
