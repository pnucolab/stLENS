#include <cstdlib>
#include <ctime>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <unordered_map>

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

inline int64_t encode_pair(int i, int j, int n_cols) {
    return static_cast<int64_t>(i) * n_cols + j;
}

CSRMatrix* sparse_rand_csr(int n_rows, int n_cols, double density) {
    int nnz_target = static_cast<int>(std::round(density * n_rows * n_cols));

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> row_dis(0, n_rows - 1);
    std::uniform_int_distribution<int> col_dis(0, n_cols - 1);

    std::unordered_set<int64_t> hash_set;
    hash_set.reserve(nnz_target);

    while (hash_set.size() < static_cast<size_t>(nnz_target)) {
        int i = row_dis(gen);
        int j = col_dis(gen);
        hash_set.insert(encode_pair(i, j, n_cols));
    }

    std::vector<std::vector<int>> row_to_cols(n_rows);
    for (int64_t h : hash_set) {
        int i = static_cast<int>(h / n_cols);
        int j = static_cast<int>(h % n_cols);
        row_to_cols[i].push_back(j);
    }

    std::vector<int> indptr(n_rows + 1, 0);
    std::vector<int> indices;
    std::vector<float> data;

    for (int i = 0; i < n_rows; ++i) {
        std::vector<int>& cols = row_to_cols[i];
        std::sort(cols.begin(), cols.end());  // optional
        for (int j : cols) {
            indices.push_back(j);
            data.push_back(1.0f);
        }
        indptr[i + 1] = indices.size();
    }

    CSRMatrix* mat = new CSRMatrix;
    mat->n_rows = n_rows;
    mat->n_cols = n_cols;
    mat->nnz = static_cast<int>(indices.size());

    mat->indptr = new int[indptr.size()];
    std::copy(indptr.begin(), indptr.end(), mat->indptr);

    mat->indices = new int[indices.size()];
    std::copy(indices.begin(), indices.end(), mat->indices);

    mat->data = new float[data.size()];
    std::copy(data.begin(), data.end(), mat->data);

    return mat;
}
}
