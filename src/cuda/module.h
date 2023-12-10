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