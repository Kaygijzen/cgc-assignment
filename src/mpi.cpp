#include <chrono>
#include <iostream>

#include "common.h"
#include <mpi.h>

/**
 * This function returns a matrix of size (num_row_labels, num_col_labels)
 * that stores the average value for each combination of row label and
 * column label. In other words, the entry at coordinate (x, y) is the
 * average over all input values having row label x and column label y.
 */
std::vector<float> calculate_cluster_average(
    int num_rows,
    int num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* matrix,
    const label_type* row_labels,
    const label_type* col_labels) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int count = num_rows / size;
    int r = num_rows % size;
    int start, stop;
    if (rank < r) {
        // The first r ranks get count + 1 tasks
        start = rank * (count + 1);
        stop = start + count;
    } else {
        // The remaining size - r ranks get count task
        start = rank * count + r;
        stop = start + (count - 1);
    }

    auto rank_cluster_sum =
        std::vector<double>(num_row_labels * num_col_labels, 0.0);
    auto rank_cluster_size = std::vector<int>(num_row_labels * num_col_labels, 0);

    for (int i = start; i <= stop; i++) {
        for (int j = 0; j < num_cols; j++) {
            auto item = matrix[i * num_cols + j];
            auto row_label = row_labels[i];
            auto col_label = col_labels[j];

            rank_cluster_sum[row_label * num_col_labels + col_label] += item;
            rank_cluster_size[row_label * num_col_labels + col_label] += 1;
        }
    }

    auto cluster_sum =
        std::vector<double>(num_row_labels * num_col_labels, 0.0);
    auto cluster_size = std::vector<int>(num_row_labels * num_col_labels, 0);
    
    for (int i = 0; i < rank_cluster_sum.size(); i++) {
        // Reduce all of the local sums into cluster_sum
        float local_sum = rank_cluster_sum.at(i);
        float global_sum;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD);
        cluster_sum[i] = global_sum;

        // Reduce all of the local sizes into cluster_size
        float local_size = rank_cluster_size.at(i);
        float global_size;
        MPI_Allreduce(&local_size, &global_size, 1, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD);
        cluster_size[i] = global_size;
    }
    
    auto cluster_avg = std::vector<float>(num_row_labels * num_col_labels);

    for (int i = 0; i < num_row_labels; i++) {
        for (int j = 0; j < num_col_labels; j++) {
            auto index = i * num_col_labels + j;
            cluster_avg[index] =
                float(cluster_sum[index]) / float(cluster_size[index]);
        }
    }

    return cluster_avg;
}

float calculate_distance(float avg, float item) {
    float diff = (avg - item);
    return diff * diff;
}

/**
 * Update the labels along the rows of the matrix. This function returns
 * both the number of rows that changed their label and the total distance.
 * If the first return value is zero, then no row was updated.
 */
std::pair<int, double> update_row_labels(
    int num_rows,
    int num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* matrix,
    label_type* row_labels,
    const label_type* col_labels,
    const float* cluster_avg) {
    int num_updated = 0;
    double total_dist = 0;

    for (int i = 0; i < num_rows; i++) {
        int best_label = -1;
        double best_dist = INFINITY;

        for (int k = 0; k < num_row_labels; k++) {
            double dist = 0;

            for (int j = 0; j < num_cols; j++) {
                float item = matrix[i * num_cols + j];

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

/**
 * Update the labels along the columns of the matrix. This function returns
 * the number of columns that changed their label label and the total distance.
 * If the first return value is zero, then no column was updated.
 */
std::pair<int, double> update_col_labels(
    int num_rows,
    int num_cols,
    int num_col_labels,
    const float* matrix,
    const label_type* row_labels,
    label_type* col_labels,
    const float* cluster_avg) {
    int num_updated = 0;
    double total_dist = 0;

    for (int j = 0; j < num_cols; j++) {
        int best_label = -1;
        double best_dist = INFINITY;

        for (int k = 0; k < num_col_labels; k++) {
            double dist = 0;

            for (int i = 0; i < num_rows; i++) {
                auto item = matrix[i * num_cols + j];

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
            num_updated++;
        }

        total_dist += best_dist;
    }

    return {num_updated, total_dist};
}

/**
 * Perform one iteration of the co-clustering algorithm. This function updates
 * the labels in both `row_labels` and `col_labels`, and returns the total
 * number of labels that changed (i.e., the number of rows and columns that
 * were reassigned to a different label).
 */
std::pair<int, double> cluster_serial_iteration(
    int num_rows,
    int num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* matrix,
    label_type* row_labels,
    label_type* col_labels) {
    // Calculate the average value per cluster
    auto cluster_avg = calculate_cluster_average(
        num_rows,
        num_cols,
        num_row_labels,
        num_col_labels,
        matrix,
        row_labels,
        col_labels);

    for (int i = 0; i < cluster_avg.size(); i++) {
        std::cout << "cluster_avg = " << cluster_avg.at(i) << "\n";
    }

    // Update labels along the rows
    auto [num_rows_updated, _] = update_row_labels(
        num_rows,
        num_cols,
        num_row_labels,
        num_col_labels,
        matrix,
        row_labels,
        col_labels,
        cluster_avg.data());

    // Update the labels along the columns
    auto [num_cols_updated, total_dist] = update_col_labels(
        num_rows,
        num_cols,
        num_col_labels,
        matrix,
        row_labels,
        col_labels,
        cluster_avg.data());

    return {num_rows_updated + num_cols_updated, total_dist};
}

/**
 * Repeatedly calls `cluster_serial_iteration` to iteratively update the
 * labels along the rows and columns. This function performs
 * `max_iterations` iterations or until convergence.
 */
void cluster_serial(
    int num_rows,
    int num_cols,
    int num_row_labels,
    int num_col_labels,
    float* matrix,
    label_type* row_labels,
    label_type* col_labels,
    int max_iterations = 25) {
    int iteration = 0;
    auto before = std::chrono::high_resolution_clock::now();

    while (iteration < max_iterations) {
        auto [num_updated, total_dist] = cluster_serial_iteration(
            num_rows,
            num_cols,
            num_row_labels,
            num_col_labels,
            matrix,
            row_labels,
            col_labels);

        iteration++;

        auto average_dist = total_dist / (num_rows * num_cols);
        std::cout << "iteration " << iteration << ": " << num_updated
                  << " labels were updated, average error is " << average_dist
                  << "\n";

        if (num_updated == 0) {
            break;
        }
    }

    auto after = std::chrono::high_resolution_clock::now();
    auto time_seconds = std::chrono::duration<double>(after - before).count();

    std::cout << "clustering time total: " << time_seconds << " seconds\n";
    std::cout << "clustering time per iteration: " << (time_seconds / iteration)
              << " seconds\n";
}

int main(int argc, const char* argv[]) {
    MPI_Init(NULL, NULL);

    std::string output_file;
    std::vector<float> matrix;
    std::vector<label_type> row_labels, col_labels;
    int num_rows = 0, num_cols = 0;
    int num_row_labels = 0, num_col_labels = 0;
    int max_iter = 0;

    auto before = std::chrono::high_resolution_clock::now();

    // Parse arguments
    if (!parse_arguments(
            argc,
            argv,
            &num_rows,
            &num_cols,
            &num_row_labels,
            &num_col_labels,
            &matrix,
            &row_labels,
            &col_labels,
            &output_file,
            &max_iter)) {
        return EXIT_FAILURE;
    }

    // Cluster labels
    cluster_serial(
        num_rows,
        num_cols,
        num_row_labels,
        num_col_labels,
        matrix.data(),
        row_labels.data(),
        col_labels.data(),
        max_iter);

    // Write resulting labels
    write_labels(
        output_file,
        num_rows,
        num_cols,
        row_labels.data(),
        col_labels.data());

    auto after = std::chrono::high_resolution_clock::now();
    auto time_seconds = std::chrono::duration<double>(after - before).count();

    std::cout << "total execution time: " << time_seconds << " seconds\n";

    return EXIT_SUCCESS;
}
