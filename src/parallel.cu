/*
 * Cartoonizer: Image quantization using K-Means algorithms
 * @autors Emilio Garzia, Luigi Marino 
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <unistd.h>

// Default parameters
#define DEFAULT_CLUSTERS 7
#define DEFAULT_ITERATIONS 50
#define DEFAULT_SEED 200
#define DEFAULT_THREADS 256

struct Color {
    float r, g, b;
};


// Calcola la distanza euclidea tra due colori
__device__ float compute_distance(const Color& a, const Color& b) {
    return sqrtf((a.r - b.r) * (a.r - b.r) + (a.g - b.g) * (a.g - b.g) + (a.b - b.b) * (a.b - b.b));
}

// Kernel per assegnare i pixel ai centroidi
__global__ void assign_pixels_to_centroids(const Color* pixels, int* assignments, const Color* centroids, int num_pixels, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        float min_dist = FLT_MAX;
        int closest_centroid = 0;
        for (int i = 0; i < k; i++) {
            float dist = compute_distance(pixels[idx], centroids[i]);
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid = i;
            }
        }
        assignments[idx] = closest_centroid;
    }
}

// Kernel per aggiornare i centroidi in base agli assegnamenti dei pixel
__global__ void update_centroids(const Color* pixels, int* assignments, Color* centroids, int* cluster_sizes, int num_pixels, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        int cluster_idx = assignments[idx];
        atomicAdd(&centroids[cluster_idx].r, pixels[idx].r);
        atomicAdd(&centroids[cluster_idx].g, pixels[idx].g);
        atomicAdd(&centroids[cluster_idx].b, pixels[idx].b);
        atomicAdd(&cluster_sizes[cluster_idx], 1);
    }
}

// Kernel per finalizzare i centroidi dividendo per la dimensione del cluster
__global__ void finalize_centroids(Color* centroids, int* cluster_sizes, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k && cluster_sizes[idx] > 0) {
        centroids[idx].r /= cluster_sizes[idx];
        centroids[idx].g /= cluster_sizes[idx];
        centroids[idx].b /= cluster_sizes[idx];
    }
}

void kmeans_gpu(const cv::Mat& image, int k, int max_iter, int threads_per_block, int seed, cv::Mat& output_image) {
    int width = image.cols;
    int height = image.rows;
    int num_pixels = width * height;

    // Converte l'immagine in un vettore di strutture Color
    std::vector<Color> pixels(num_pixels);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec3b color = image.at<cv::Vec3b>(i, j);
            pixels[i * width + j] = { color[2] / 255.0f, color[1] / 255.0f, color[0] / 255.0f };
        }
    }

    // Allocazione memoria su GPU
    Color* d_pixels;
    Color* d_centroids;
    int* d_assignments;
    int* d_cluster_sizes;

    cudaError_t err;

    err = cudaMalloc(&d_pixels, num_pixels * sizeof(Color));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for pixels: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_centroids, k * sizeof(Color));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for centroids: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_assignments, num_pixels * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for assignments: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_cluster_sizes, k * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for cluster_sizes: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Copia i dati dalla memoria host a quella device
    err = cudaMemcpy(d_pixels, pixels.data(), num_pixels * sizeof(Color), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for pixels: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    srand(seed);

    // Inizializza i centroidi casualmente
    std::vector<Color> centroids(k);
    for (int i = 0; i < k; i++) {
        centroids[i] = { float(rand()) / RAND_MAX, float(rand()) / RAND_MAX, float(rand()) / RAND_MAX };
    }

    // Copia i centroidi sulla GPU
    err = cudaMemcpy(d_centroids, centroids.data(), k * sizeof(Color), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for centroids: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Configurazione del kernel
    int blocks_per_grid = (num_pixels + threads_per_block - 1) / threads_per_block;

    for (int iter = 0; iter < max_iter; iter++) {
        // Passo 1: Assegna i pixel ai centroidi
        assign_pixels_to_centroids<<<blocks_per_grid, threads_per_block>>>(d_pixels, d_assignments, d_centroids, num_pixels, k);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed for assign_pixels_to_centroids: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        cudaDeviceSynchronize();

        // Passo 2: Aggiorna i centroidi
        cudaMemset(d_cluster_sizes, 0, k * sizeof(int));  // Reset delle dimensioni dei cluster
        cudaMemset(d_centroids, 0, k * sizeof(Color));   // Reset dei centroidi
        update_centroids<<<blocks_per_grid, threads_per_block>>>(d_pixels, d_assignments, d_centroids, d_cluster_sizes, num_pixels, k);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed for update_centroids: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        cudaDeviceSynchronize();
        
        // Passo 3: Finalizza i centroidi dividendo per la dimensione dei cluster
        finalize_centroids<<<(k + threads_per_block - 1) / threads_per_block, threads_per_block>>>(d_centroids, d_cluster_sizes, k);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed for finalize_centroids: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        cudaDeviceSynchronize();
    }

    // Copia i centroidi finali e gli assegnamenti dal device all'host
    cudaMemcpy(centroids.data(), d_centroids, k * sizeof(Color), cudaMemcpyDeviceToHost);
    std::vector<int> assignments(num_pixels);
    cudaMemcpy(assignments.data(), d_assignments, num_pixels * sizeof(int), cudaMemcpyDeviceToHost);

    // Costruisci l'immagine di output
    output_image = image.clone();
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int cluster_idx = assignments[i * width + j];
            cv::Vec3b new_color(
                static_cast<unsigned char>(centroids[cluster_idx].b * 255),
                static_cast<unsigned char>(centroids[cluster_idx].g * 255),
                static_cast<unsigned char>(centroids[cluster_idx].r * 255)
            );
            output_image.at<cv::Vec3b>(i, j) = new_color;
        }
    }

    // Libera la memoria su GPU
    cudaFree(d_pixels);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_cluster_sizes);
}

// Argument parser
void arg_parser(int argc,char* argv[], int& clusters, int& iterations, int& seed, int& threads_per_block,std::string& input_image_path, std::string& output_image_path){
    int opt;
    while ((opt = getopt(argc, argv, "c:i:s:t:")) != -1) {
        switch (opt) {
            case 'c':
                clusters = std::stoi(optarg);
                break;
            case 'i':
                iterations = std::stoi(optarg);
                break;
            case 's':
                seed = std::stoi(optarg);
                break;
            case 't':
                threads_per_block = std::stoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-c clusters] [-i iterations] [-s seed] [-t threads] <input_image> <output_image>" << std::endl;
        }
    }
    if (optind < argc) input_image_path = argv[optind++];
    if (optind < argc) output_image_path = argv[optind++];
}

// Driver code
int main(int argc, char* argv[]) {
    int clusters = DEFAULT_CLUSTERS;
    int iterations = DEFAULT_ITERATIONS;
    int seed = DEFAULT_SEED;
    int threads_per_block = DEFAULT_THREADS;
    std::string input_image_path = "images/image.jpg";
    std::string output_image_path = "images/cartoon_gpu.jpg";

    arg_parser(argc, argv, clusters, iterations, seed, threads_per_block, input_image_path, output_image_path);

    // Load Image
    if (input_image_path.empty() || output_image_path.empty()) {
        std::cerr << "Error: You must provide input and output image paths!" << std::endl;
        return -1;
    }

    // Carica l'immagine
    cv::Mat image = cv::imread(input_image_path);
    if (image.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    cv::Mat output_image;
    kmeans_gpu(image, clusters, iterations, threads_per_block, seed, output_image);

    // Salva l'immagine di output
    cv::imwrite(output_image_path, output_image);
    std::cout << "GPU Output saved!" << output_image_path << std::endl;

    return 0;
}