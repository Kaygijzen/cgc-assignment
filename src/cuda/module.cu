#ifdef __CUDACC__
#define CUDA_GLOBAL __global__
#else
#define CUDA_GLOBAL
#endif

#include <cuda.h>
#include "module.h"
#include <stdio.h>
#include <math.h>

__device__ float calculate_dist(float avg, float item) {
	float diff = (avg - item);
	return diff * diff;
}
float calculate_distance(float avg, float item) {
	float diff = (avg - item);
	return diff * diff;
}

// THIS FUNCTION IS TENTATIVE AND NOT IMPLEMENTED
__global__ void cluster_id_kernel(
	int num_rows,
	int num_cols,
	const int* row_labels,
	const int* col_labels,
	int row_displacement,
	int* cluster_ids
) {
	int j = blockDim.x * blockIdx.x + threadIdx.x; 
	int tid = threadIdx.x;

	if (j < num_cols) {
		for (int i = row_displacement; i < num_rows + row_displacement; i++) {
			cluster_ids[i * num_cols + j] = row_labels[i] * col_labels[j];
		}
	}
}

// THIS FUNCTION IS TENTATIVE AND NOT IMPLEMENTED
void call_cluster_id_kernel(
	int num_rows,
	int num_cols,
	int num_row_labels,
	int num_col_labels,
	const float* matrix,
	const int* row_labels,
	const int* col_labels,
	int row_displacement,
	int* cluster_ids) {

	int N = num_cols;

	// Block size and number calculation
	int blockSize = 1024;
  int numBlocks = (N + blockSize - 1) / blockSize;

	// Allocate memory for data on device
	float *d_matrix;
	cudaMalloc(&d_matrix, (num_cols*num_rows)*sizeof(float));
	int *d_cluster_ids;
	cudaMalloc(&d_cluster_ids, (num_cols*num_rows)*sizeof(int));
	int *d_col_labels;
	cudaMalloc(&d_col_labels, num_cols*sizeof(int));
	int *d_row_labels;
	cudaMalloc(&d_row_labels, num_rows*sizeof(int));

	// Copy data to device
	cudaMemcpy(d_row_labels, row_labels, (num_rows)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_labels, col_labels, (num_cols)*sizeof(int), cudaMemcpyHostToDevice);

	cluster_id_kernel <<< numBlocks, blockSize >>>(
		num_rows,
		num_cols,
		d_row_labels,
		d_col_labels,
		row_displacement,
		d_cluster_ids);

	// Copy result from device to host
	cudaMemcpy(cluster_ids, d_cluster_ids, (num_cols*num_rows)*sizeof(int), cudaMemcpyDeviceToHost);
	
	// Free allocated memory
	cudaFree(d_cluster_ids);
	cudaFree(d_col_labels);
	cudaFree(d_row_labels);
}

// THIS FUNCTION IS TENTATIVE AND NOT IMPLEMENTED
__global__ void cluster_sum_size_kernel(
	int num_rows,
	int num_cols,
	const float* matrix,
	const int* cluster_ids,
	double* cluster_sum,
	int* cluster_size,
	int row_displacement,
	int N
) {
	int id = blockDim.x * blockIdx.x + threadIdx.x; 
	int tid = threadIdx.x;

	if (id < (N)) {
		for (int i = row_displacement; i < num_rows + row_displacement; i++) {
			for (int j = 0; j < num_cols; j++) {
				int cluster_id = cluster_ids[i * num_cols + j];
				if (id == cluster_id) {
					cluster_sum[cluster_id] += matrix[cluster_ids[i * num_cols + j]];
					cluster_size[cluster_id] += 1;
				}
			}
		}
	}
};

// THIS FUNCTION IS TENTATIVE AND NOT IMPLEMENTED
void call_cluster_sum_size_kernel(
	int num_rows,
	int num_cols,
	int num_row_labels,
	int num_col_labels,
	const float* matrix,
	int* cluster_ids,
	int row_displacement,
	double* cluster_sum,
	int* cluster_size) {

	int N = num_col_labels * num_row_labels;

	// Block size and number calculation
	int blockSize = 1024;
  int numBlocks = (N + blockSize - 1) / blockSize;

	// Allocate memory for data on device
	float *d_matrix;
	cudaMalloc(&d_matrix, (num_cols*num_rows)*sizeof(float));
	int *d_cluster_ids;
	cudaMalloc(&d_cluster_ids, (num_cols*num_rows)*sizeof(int));
	double *d_cluster_sum;
	cudaMalloc(&d_cluster_sum, (num_row_labels*num_col_labels)*sizeof(double));
	int *d_cluster_size;
	cudaMalloc(&d_cluster_size, (num_row_labels*num_col_labels)*sizeof(int));

	// Copy data to device
	cudaMemcpy(d_matrix, matrix, (num_cols*num_rows)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cluster_ids, cluster_ids, (num_cols*num_rows)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cluster_sum, cluster_sum, (num_col_labels*num_row_labels)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cluster_size, cluster_size, (num_col_labels*num_row_labels)*sizeof(int), cudaMemcpyHostToDevice);
	
	cluster_sum_size_kernel <<< numBlocks, blockSize >>>(
		num_rows,
		num_cols,
		d_matrix,
		d_cluster_ids,
		d_cluster_sum,
		d_cluster_size,
		row_displacement,
		(num_row_labels * num_col_labels));

	// Copy results from device to host
	cudaMemcpy(cluster_sum, d_cluster_sum, (num_row_labels*num_col_labels)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(cluster_size, d_cluster_size, (num_row_labels*num_col_labels)*sizeof(int), cudaMemcpyDeviceToHost);

	// Free allocated memory
	cudaFree(d_matrix);
	cudaFree(d_cluster_ids);
	cudaFree(d_cluster_sum);
	cudaFree(d_cluster_size);
}

__global__ void row_labels_iteration() {

}

std::pair<int, double> call_update_row_labels_kernel(
    int num_rows,
    int num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* matrix,
    int* row_labels,
    const int* col_labels,
    const float* cluster_avg,
    int displacement) {
    int num_updated = 0;
    double total_dist = 0;

    for (int i = 0; i < num_rows; i++) {
        int best_label = -1;
        double best_dist = INFINITY;
        int displaced_i = i + displacement;

        for (int k = 0; k < num_row_labels; k++) {
            double dist = 0;

            for (int j = 0; j < num_cols; j++) {
                float item = matrix[displaced_i * num_cols + j];

                int row_label = k;
                int col_label = col_labels[j];
                float y = cluster_avg[row_label * num_col_labels + col_label];

                dist += calculate_distance(y, item);
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_label = k;
            }
        }

        if (row_labels[i] != best_label) {
            row_labels[i] = best_label;
            num_updated++;
        }

        total_dist += best_dist;
    }

    return {num_updated, total_dist};
}


__global__ void col_labels_iteration(
	const float* matrix, 
	int* col_labels,
	const int* row_labels,
	const float* cluster_avg,
	int num_cols,
	int num_rows,
	int num_col_labels,
	int num_cols_recv,
	int displacement) {

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	__shared__ int s_num_updated[1024];
	__shared__ double s_total_dist[1024];

	for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
		s_num_updated[i] = 0;
		s_total_dist[i] = 0.0;
	}
	__syncthreads();

    if (j < num_cols_recv) {
        int best_label = -1;
        double best_dist = INFINITY;

        for (int k = 0; k < num_col_labels; k++) {
            double dist = 0;

            for (int i = 0; i < num_rows; i++) {
                auto item = matrix[i * num_cols + j + displacement];

                auto row_label = row_labels[i];
                auto col_label = k;
                auto y = cluster_avg[row_label * num_col_labels + col_label];

                dist += calculate_dist(y, item);
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_label = k;
            }
        }

        if (col_labels[j] != best_label) {
            col_labels[j] = best_label;
            s_num_updated[tid]++;
        }

        s_total_dist[tid] += best_dist;
    }

	// TODO: reduce num_updated and total_dist
}

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
	int num_cols_recv) {

	int N = num_cols_recv;

	// Block size and number calculation
	int blockSize = 1024;
    int numBlocks = (N + blockSize - 1) / blockSize;

	// Allocate memory for data on device
	float *d_matrix;
	cudaMalloc(&d_matrix, (num_cols*num_rows)*sizeof(float));
	float *d_cluster_avg;
	cudaMalloc(&d_cluster_avg, (num_row_labels*num_col_labels)*sizeof(float));
	int *d_col_labels;
	cudaMalloc(&d_col_labels, num_cols_recv*sizeof(int));
	int *d_row_labels;
	cudaMalloc(&d_row_labels, num_rows*sizeof(int));

	// Copy data to device
	cudaMemcpy(d_matrix, matrix, (num_cols*num_rows)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cluster_avg, cluster_avg, (num_row_labels*num_col_labels)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_labels, col_labels, num_cols_recv*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row_labels, row_labels, num_rows*sizeof(int), cudaMemcpyHostToDevice);

	// Call kernel
	col_labels_iteration <<< numBlocks, blockSize >>>(
		d_matrix, 
		d_col_labels,
		d_row_labels,
		d_cluster_avg,
		num_cols,
		num_rows,
		num_col_labels,
		num_cols_recv,
		displacement);

	// Copy results from device to host
	cudaMemcpy(col_labels, d_col_labels, num_cols_recv*sizeof(int), cudaMemcpyDeviceToHost);

	// Free allocated memory
	cudaFree(d_matrix);
	cudaFree(d_cluster_avg);
	cudaFree(d_col_labels);
	cudaFree(d_row_labels);

	return {1, 1};
}