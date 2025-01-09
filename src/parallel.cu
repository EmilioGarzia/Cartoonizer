/*
    @brief Image cartoonization using K-means colors quantization: PARALLEL CUDA VERSION
    @file parallel.cu
    @author Emilio Garzia, Luigi Marino
    @date 2025
    @details nvcc main.cu -o main -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
*/

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

typedef struct {
    float r, g, b;
} Color;

// Assign pixel to centroid kernel
__global__ void assign_pixels_to_centroids(Color* pixels, Color* centroids, int* assignments, int num_pixels, int num_centroids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        float min_dist = FLT_MAX;
        int closest_centroid = 0;
        Color pixel = pixels[idx];

        // Distance among pixels and centroids
        for (int i = 0; i < num_centroids; i++) {
            float dist = (pixel.r - centroids[i].r) * (pixel.r - centroids[i].r) +
                         (pixel.g - centroids[i].g) * (pixel.g - centroids[i].g) +
                         (pixel.b - centroids[i].b) * (pixel.b - centroids[i].b);
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid = i;
            }
        }
        assignments[idx] = closest_centroid;
    }
}

// Update centroids kernel
__global__ void update_centroids(Color* pixels, int* assignments, Color* centroids, int* cluster_sizes, int num_pixels, int num_centroids) {
    __shared__ float shared_centroids_r[256];
    __shared__ float shared_centroids_g[256];
    __shared__ float shared_centroids_b[256];
    __shared__ int shared_cluster_sizes[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < num_centroids) {
        shared_centroids_r[threadIdx.x] = 0.0f;
        shared_centroids_g[threadIdx.x] = 0.0f;
        shared_centroids_b[threadIdx.x] = 0.0f;
        shared_cluster_sizes[threadIdx.x] = 0;
    }
    __syncthreads();

    // Update the centroid after pixels assignment (shared memory)
    if (idx < num_pixels) {
        int cluster_idx = assignments[idx];
        atomicAdd(&shared_centroids_r[cluster_idx], pixels[idx].r);
        atomicAdd(&shared_centroids_g[cluster_idx], pixels[idx].g);
        atomicAdd(&shared_centroids_b[cluster_idx], pixels[idx].b);
        atomicAdd(&shared_cluster_sizes[cluster_idx], 1);
    }
    __syncthreads();

    // Each thread compute the final centroid
    if (threadIdx.x < num_centroids) {
        if (shared_cluster_sizes[threadIdx.x] > 0) {
            centroids[threadIdx.x].r = shared_centroids_r[threadIdx.x] / shared_cluster_sizes[threadIdx.x];
            centroids[threadIdx.x].g = shared_centroids_g[threadIdx.x] / shared_cluster_sizes[threadIdx.x];
            centroids[threadIdx.x].b = shared_centroids_b[threadIdx.x] / shared_cluster_sizes[threadIdx.x];
        }
    }
}

// Image maker kernel
__global__ void create_cartoon_image(Color* pixels, int* assignments, Color* centroids, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        Color pixel = pixels[idx];
        int closest_centroid = assignments[idx];

        // Assign the pixel to the closest cluster
        pixel.r = centroids[closest_centroid].r;
        pixel.g = centroids[closest_centroid].g;
        pixel.b = centroids[closest_centroid].b;

        pixels[idx] = pixel;
    }
}

// driver code
int main() {
    // Load and check input images
    cv::Mat image = cv::imread("images/image.jpg");
    if (image.empty()) {
        printf("ERROR: An error were occurred during the image loading\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int num_pixels = width * height;

    // Input parameters
    const int num_clusters = 20;  // total amount of colors used on the output image
    const int max_iterations = 50;
    const int nThreads = 256;
    const int nBlocks = (num_pixels + nThreads-1)/ nThreads; // grid size based on number of pixels
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Convert the image channel in RGB
    Color* pixels = (Color*)malloc(num_pixels * sizeof(Color));
    for (int i = 0; i < num_pixels; i++) {
        cv::Vec3b color = image.at<cv::Vec3b>(i / width, i % width);
        pixels[i].b = (float)color[0] / 255.0f;
        pixels[i].g = (float)color[1] / 255.0f;
        pixels[i].r = (float)color[2] / 255.0f;
    }

    srand(42);

    // Initial centroids (in GPU)
    Color* centroids = (Color*)malloc(num_clusters * sizeof(Color));
    for (int i = 0; i < num_clusters; i++) {
        centroids[i].r = (float)rand() / RAND_MAX;
        centroids[i].g = (float)rand() / RAND_MAX;
        centroids[i].b = (float)rand() / RAND_MAX;
    }

    Color* d_pixels;
    Color* d_centroids;
    int* d_assignments;
    int* d_cluster_sizes;

    // GPU memory allocation
    cudaMalloc(&d_pixels, num_pixels * sizeof(Color));
    cudaMalloc(&d_centroids, num_clusters * sizeof(Color));
    cudaMalloc(&d_assignments, num_pixels * sizeof(int));
    cudaMalloc(&d_cluster_sizes, num_clusters * sizeof(int));
    cudaMemcpy(d_pixels, pixels, num_pixels * sizeof(Color), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, num_clusters * sizeof(Color), cudaMemcpyHostToDevice);

    // Inizializza cluster_sizes su GPU
    int* cluster_sizes = (int*)malloc(num_clusters * sizeof(int));
    memset(cluster_sizes, 0, num_clusters * sizeof(int));
    cudaMemcpy(d_cluster_sizes, cluster_sizes, num_clusters * sizeof(int), cudaMemcpyHostToDevice);

    // K-means execution
    for (int iter = 0; iter < max_iterations; iter++) {
        // Step 1: Assign pixels to centroids
        assign_pixels_to_centroids<<<(num_pixels+nThreads-1)/nThreads, nThreads>>>(d_pixels, d_centroids, d_assignments, num_pixels, num_clusters);
        cudaDeviceSynchronize();

        // Step 2: Keep the centroids up to date
        update_centroids<<<(num_clusters+nThreads-1)/nThreads, nThreads>>>(d_pixels, d_assignments, d_centroids, d_cluster_sizes, num_pixels, num_clusters);
        cudaDeviceSynchronize();
    }

    // Make the output image using GPU
    create_cartoon_image<<<nBlocks, nThreads>>>(d_pixels, d_assignments, d_centroids, num_pixels);
    cudaDeviceSynchronize();

    // Copy the output image on CPU in order to visualize it
    cudaMemcpy(pixels, d_pixels, num_pixels * sizeof(Color), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_pixels; i++) {
        cv::Vec3b new_color(
            (unsigned char)(pixels[i].b * 255),
            (unsigned char)(pixels[i].g * 255),
            (unsigned char)(pixels[i].r * 255)
        );
        image.at<cv::Vec3b>(i / width, i % width) = new_color;
    }

    cv::imwrite("images/cartoon_image_parallel.jpg", image);

    // Compute and print the elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tempo di esecuzione %fms\n", milliseconds);

    // Free the memories
    free(pixels);
    free(centroids);
    free(cluster_sizes);
    cudaFree(d_pixels);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_cluster_sizes);

    return 0;
}
