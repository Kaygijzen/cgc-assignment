#ifdef __CUDACC__
#define CUDA_GLOBAL __global__
#else
#define CUDA_GLOBAL
#endif

#include <cuda.h>
#include "module.h"
#include <stdio.h>
#include <math.h>

__device__ int d_num_col_labels_updated = 0;
__device__ double d_total_dist_cols = 0;

__global__ void block_dist_row_labels(
	const float* matrix, 
	int i,
	int k,
	const int* col_labels,
	const float* cluster_avg,
	double* dist_array,
	int num_cols,
	int num_col_labels) 
{
	__shared__ double sdata[1024];

	int j = blockDim.x * blockIdx.x + threadIdx.x; 
	int tid = threadIdx.x;

	if (j < num_cols) {
		float item = matrix[i * num_cols + j];

		int row_label = k;
		int col_label = col_labels[j];

		float y = cluster_avg[row_label * num_col_labels + col_label];

		sdata[tid] = (y - item) * (y - item);
	}

	__syncthreads();

	// do reduction in shared mem
	for (int s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) {
		dist_array[blockIdx.x] = sdata[0];
	}
}

std::pair<int, double> best_row_label(
	int num_row_labels,
	int num_col_labels,
	int num_rows,
	int num_cols,
	const float* matrix,
	const float* cluster_avg,
	int i,
	const int* col_labels) {
	int N = num_cols;

	// Block size and number calculation
	int blockSize = 1024;
  int numBlocks = (N + blockSize - 1) / blockSize;
	
	// Number of bytes to allocate for numBlocks
	size_t bytes = numBlocks*sizeof(double);

	// Allocate memory on host
	double *dist_blocks = (double*)malloc(bytes);

	// Allocate memory on device
	double *d_dist_blocks;
	cudaMalloc(&d_dist_blocks, bytes);
	float *d_matrix;
	cudaMalloc(&d_matrix, (num_cols*num_rows)*sizeof(float));
	float *d_cluster_avg;
	cudaMalloc(&d_cluster_avg, (num_row_labels*num_col_labels)*sizeof(float));
	int *d_col_labels;
	cudaMalloc(&d_col_labels, num_cols*sizeof(int));

	// Copy data to device
	cudaMemcpy(d_matrix, matrix, (num_cols*num_rows)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cluster_avg, cluster_avg, (num_row_labels*num_col_labels)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_labels, col_labels, num_cols*sizeof(int), cudaMemcpyHostToDevice);

	int best_label = -1;
	double best_dist = INFINITY;

	for (int k = 0; k < num_row_labels; k++) {
		double dist = 0;

		block_dist_row_labels<<< numBlocks, blockSize >>>(
			d_matrix,
			i,
			k,
			d_col_labels,
			d_cluster_avg,
			d_dist_blocks, 
			num_cols,
			num_col_labels);

		// Copy result from device to host
		cudaMemcpy(dist_blocks, d_dist_blocks, bytes, cudaMemcpyDeviceToHost);

		// Reduce result by summing all block results
		double sum = 0;
		for (int x = 0; x < numBlocks; x++) {
			sum += dist_blocks[x];
		}

		dist = sum;

		if (dist < best_dist) {
			best_dist = dist;
			best_label = k;
		}
	}

	// Free allocated memory
	cudaFree(d_dist_blocks);
	cudaFree(d_matrix);
	cudaFree(d_col_labels);
	free(dist_blocks);

	return {best_label, best_dist};
}


__global__ void col_labels_iteration(
	const float* matrix, 
	int* col_labels,
	const int* row_labels,
	const float* cluster_avg,
	int num_cols,
	int num_rows,
	int num_col_labels,
	int displacement) 
{
	int j = blockDim.x * blockIdx.x + threadIdx.x; 
	int displaced_j = j + displacement;

	if (j < num_cols) {
		int best_label = -1;
		double best_dist = INFINITY;

		for (int k = 0; k < num_col_labels; k++) {
			double dist = 0;

			for (int i = 0; i < num_rows; i++) {
				auto item = matrix[i * num_cols + displaced_j];

				auto row_label = row_labels[i];
				auto col_label = k;
				auto y = cluster_avg[row_label * num_col_labels + col_label];

				dist += (y - item) * (y - item);
			}

			if (dist < best_dist) {
				best_dist = dist;
				best_label = k;
			}
		}

		if (col_labels[displaced_j] != best_label) {
			col_labels[displaced_j] = best_label;
			d_num_col_labels_updated++;
		}

		d_total_dist_cols += best_dist;
	}
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
	cudaMalloc(&d_col_labels, num_cols*sizeof(int));
	int *d_row_labels;
	cudaMalloc(&d_row_labels, num_rows*sizeof(int));

	// Copy data to device
	cudaMemcpy(d_matrix, matrix, (num_cols*num_rows)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cluster_avg, cluster_avg, (num_row_labels*num_col_labels)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_labels, col_labels, num_cols*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row_labels, row_labels, num_rows*sizeof(int), cudaMemcpyHostToDevice);

	// Call kernel
	col_labels_iteration <<< numBlocks, blockSize >>>(
		d_matrix, 
		d_col_labels,
		d_row_labels,
		d_cluster_avg,
		num_cols_recv,
		num_rows,
		num_col_labels,
		displacement);

	// Copy results from device to host
	int num_updated;
	double total_dist;
	cudaMemcpy(col_labels, d_col_labels, num_cols*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&num_updated, d_num_col_labels_updated, sizeof(int));
	cudaMemcpyFromSymbol(&total_dist, d_total_dist_cols, sizeof(double));

	// Free allocated memory
	cudaFree(d_matrix);
	cudaFree(d_cluster_avg);
	cudaFree(d_col_labels);
	cudaFree(d_row_labels);

	return {num_updated, total_dist};
}