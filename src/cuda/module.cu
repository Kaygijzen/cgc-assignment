#ifdef __CUDACC__
#define CUDA_GLOBAL __global__
#else
#define CUDA_GLOBAL
#endif

#include <cuda.h>
#include "module.h"
#include <stdio.h>
#include <math.h>

__device__ float calculate_distance(float avg, float item) {
	float diff = (avg - item);
	return diff * diff;
}

__global__ void cluster_id_kernel(
	int num_rows,
	int num_cols,
	const int* row_labels,
	const int* col_labels,
	int* cluster_ids,
	int num_col_labels,
	int row_displacement
) {
	int j = blockDim.x * blockIdx.x + threadIdx.x; 

	if (j < num_cols) {
		for (int i = 0; i < num_rows; i++) {
			int row_label = row_labels[i + row_displacement];
			int col_label = col_labels[j];
			cluster_ids[i * num_cols + j] = row_label * num_col_labels + col_label;
		}
	}
}

void call_cluster_id_kernel(
	int num_rows,
	int num_cols,
	int num_col_labels,
	const int* row_labels,
	const int* col_labels,
	int* cluster_ids,
	int row_displacement,
	int num_rows_recv) {

	// Block size and number calculation
	int blockSize = 1024;
  	int numBlocks = (num_cols + blockSize - 1) / blockSize;

	// Allocate memory for data on device
	int *d_row_labels;
	cudaMalloc(&d_row_labels, num_rows*sizeof(int));
	int *d_col_labels;
	cudaMalloc(&d_col_labels, num_cols*sizeof(int));
	int *d_cluster_ids;
	cudaMalloc(&d_cluster_ids, (num_rows_recv*num_cols)*sizeof(int));

	// Copy data to device
	cudaMemcpy(d_row_labels, row_labels, num_rows*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_labels, col_labels, num_cols*sizeof(int), cudaMemcpyHostToDevice);
	
	cluster_id_kernel <<< numBlocks, blockSize >>>(
		num_rows_recv,
		num_cols,
		d_row_labels,
		d_col_labels,
		d_cluster_ids,
		num_col_labels,
		row_displacement);

	// Copy results from device to host
	cudaMemcpy(cluster_ids, d_cluster_ids, num_rows_recv*num_cols*sizeof(int), cudaMemcpyDeviceToHost);

	// Free allocated memory
	cudaFree(d_row_labels);
	cudaFree(d_col_labels);
	cudaFree(d_cluster_ids);
}

__global__ void calculate_cluster_avg(
	int num_row_labels,
	int num_col_labels,
	double* cluster_sum,
	int* cluster_size,
	float* cluster_avg) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < num_row_labels && j < num_col_labels) {
		cluster_avg[i * num_col_labels + j] = float(cluster_sum[i * num_col_labels + j]) / float(cluster_size[i * num_col_labels + j]);   
    }
}

void call_cluster_average_kernel(
	int num_row_labels,
	int num_col_labels,
	double* cluster_sum,
	int* cluster_size,
	float* cluster_avg) {

	int N = num_col_labels;
	int num_clusters = num_row_labels * num_col_labels;

	// Block size and number calculation
	dim3 blockSize(32, 32);
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x, (N+blockSize.y -1) / blockSize.y);

	// Allocate memory on device
	double *d_cluster_sum;
	cudaMalloc(&d_cluster_sum, num_clusters*sizeof(double));
	int *d_cluster_size;
	cudaMalloc(&d_cluster_size, num_clusters*sizeof(int));
	float *d_cluster_avg;
	cudaMalloc(&d_cluster_avg, num_clusters*sizeof(float));

	// Copy data to device
	cudaMemcpy(d_cluster_sum, cluster_sum, num_clusters*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cluster_size, cluster_size, num_clusters*sizeof(int), cudaMemcpyHostToDevice);

	calculate_cluster_avg <<< numBlocks, blockSize >>>(
		num_row_labels,
		num_col_labels,
		d_cluster_sum,
		d_cluster_size,
		d_cluster_avg); 

	// Copy result from device to host
	cudaMemcpy(cluster_avg, d_cluster_avg, num_clusters*sizeof(float), cudaMemcpyDeviceToHost);

	// Free allocated memory
	cudaFree(d_cluster_sum);
	cudaFree(d_cluster_size);
	cudaFree(d_cluster_avg);
}

__global__ void calculate_block_distance_row(
	const float* matrix, 
	int i,
	int row_label,
	const int* col_labels,
	const float* cluster_avg,
	double* dist_per_block,
	int num_cols,
	int num_col_labels) 
{
	__shared__ double s_total_dist[1024];

	int j = blockDim.x * blockIdx.x + threadIdx.x; 
	int tid = threadIdx.x;

	if (j < num_cols) {
		float item = matrix[i * num_cols + j];

		int col_label = col_labels[j];

		float y = cluster_avg[row_label * num_col_labels + col_label];

		s_total_dist[tid] = calculate_distance(y, item);
	}

	__syncthreads();

	// do reduction in shared mem
	for (int s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) {
			s_total_dist[tid] += s_total_dist[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) {
		dist_per_block[blockIdx.x] = s_total_dist[0];
	}
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
    int displacement,
	int num_rows_recv) {
		
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
	cudaMalloc(&d_matrix, (num_rows*num_cols)*sizeof(float));
	float *d_cluster_avg;
	cudaMalloc(&d_cluster_avg, (num_row_labels*num_col_labels)*sizeof(float));
	int *d_col_labels;
	cudaMalloc(&d_col_labels, num_cols*sizeof(int));

	// Copy data to device
	cudaMemcpy(d_matrix, matrix, (num_rows*num_cols)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cluster_avg, cluster_avg, (num_row_labels*num_col_labels)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_labels, col_labels, num_cols*sizeof(int), cudaMemcpyHostToDevice);

    int num_updated = 0;
    double total_dist = 0;

    for (int i = 0; i < num_rows_recv; i++) {
        int best_label = -1;
        double best_dist = INFINITY;
        int displaced_i = i + displacement;

        for (int k = 0; k < num_row_labels; k++) {
            double dist = 0;

			calculate_block_distance_row <<< numBlocks, blockSize >>>(
				d_matrix,
				displaced_i,
				k,
				d_col_labels,
				d_cluster_avg,
				d_dist_blocks, 
				num_cols,
				num_col_labels);

			// Copy result from device to host
			cudaMemcpy(dist_blocks, d_dist_blocks, bytes, cudaMemcpyDeviceToHost);

			// Reduce result by summing all block results
			for (int x = 0; x < numBlocks; x++) {
				dist += dist_blocks[x];
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

	// Free allocated memory
	cudaFree(d_dist_blocks);
	cudaFree(d_matrix);
	cudaFree(d_col_labels);
	free(dist_blocks);

	return {num_updated, total_dist};
}


__global__ void col_labels_iteration(
	const float* matrix, 
	int* col_labels,
	const int* row_labels,
	const float* cluster_avg,
	int* num_updated_per_block,
	double* total_dist_per_block,
	int num_cols,
	int num_rows,
	int num_col_labels,
	int num_cols_recv,
	int displacement) {

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	__shared__ int s_num_updated[1024];
	__shared__ double s_total_dist[1024];
	s_num_updated[tid] = 0;
	s_total_dist[tid] = 0.0;

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

                dist += calculate_distance(y, item);
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

	__syncthreads();

	// do reduction in shared mem
	for (int s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) {
			s_num_updated[tid] += s_num_updated[tid + s];
			s_total_dist[tid] += s_total_dist[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) {
		num_updated_per_block[blockIdx.x] = s_num_updated[0];
		total_dist_per_block[blockIdx.x] = s_total_dist[0];
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

	// Allocate memory on host
	int *num_updated_per_block = (int*)malloc(numBlocks*sizeof(int));
	double *total_dist_per_block = (double*)malloc(numBlocks*sizeof(double));

	// Allocate memory for data on device
	float *d_matrix;
	cudaMalloc(&d_matrix, (num_cols*num_rows)*sizeof(float));
	float *d_cluster_avg;
	cudaMalloc(&d_cluster_avg, (num_row_labels*num_col_labels)*sizeof(float));
	int *d_col_labels;
	cudaMalloc(&d_col_labels, num_cols_recv*sizeof(int));
	int *d_row_labels;
	cudaMalloc(&d_row_labels, num_rows*sizeof(int));
	int *d_num_updated_per_block;
	cudaMalloc(&d_num_updated_per_block, numBlocks*sizeof(int));
	double *d_total_dist_per_block;
	cudaMalloc(&d_total_dist_per_block, numBlocks*sizeof(double));

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
		d_num_updated_per_block,
		d_total_dist_per_block,
		num_cols,
		num_rows,
		num_col_labels,
		num_cols_recv,
		displacement);

	// Copy results from device to host
	cudaMemcpy(col_labels, d_col_labels, num_cols_recv*sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(num_updated_per_block, d_num_updated_per_block, numBlocks*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(total_dist_per_block, d_total_dist_per_block, numBlocks*sizeof(double), cudaMemcpyDeviceToHost);

	int num_updated = 0;
	double total_dist = 0;

	// Reduce result by summing all block results
	for (int i = 0; i < numBlocks; i++) {
		num_updated += num_updated_per_block[i];
		total_dist += total_dist_per_block[i];
	}

	// Free allocated memory
	cudaFree(d_matrix);
	cudaFree(d_cluster_avg);
	cudaFree(d_col_labels);
	cudaFree(d_row_labels);

	return {num_updated, total_dist};
}