#include <iostream>

void call_cluster_average_kernel(
	int num_row_labels,
	int num_col_labels,
	double* cluster_sum,
	int* cluster_size,
	float* cluster_avg);

std::pair<int, double> call_update_row_labels_kernel(
    int num_rows,
    int num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* matrix,
    int* row_labels,
    const int* col_labels,
    const float* cluster_avg,
    int displacement,
	int num_rows_recv);

std::pair<int, double> call_update_col_labels_kernel(
	int num_rows,
	int num_cols,
	int num_row_labels,
	int num_col_labels,
	const float* matrix,
	const int* row_labels,
	int* col_labels,
	const float* cluster_avg,
	int displacement,
	int num_cols_recv);