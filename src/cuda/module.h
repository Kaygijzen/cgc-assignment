#include <iostream>

std::pair<int, double> best_row_label(
	int num_row_labels,
	int num_col_labels,
	int num_rows,
	int num_cols,
	const float* matrix,
	const float* cluster_avg,
	int i,
	const int* col_labels); 

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