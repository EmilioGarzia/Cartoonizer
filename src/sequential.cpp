/*
 * Cartoonizer: Image quantization using K-Means algorithms
 * @autors Emilio Garzia, Luigi Marino 
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// Default parameters
#define DEFAULT_CLUSTERS 7
#define DEFAULT_ITERATIONS 50
#define DEFAULT_SEED 200
#define DEFAULT_THREADS 256

struct Color {
    float r, g, b;
};

// Compute the Euclidean distance between two colors
float euclidean_distance(const Color& a, const Color& b) {
    return std::sqrt((a.r - b.r) * (a.r - b.r) + (a.g - b.g) * (a.g - b.g) + (a.b - b.b) * (a.b - b.b));
}

// Update centroids based on pixel assignments
void update_centroids(const std::vector<Color>& pixels, const std::vector<int>& assignments, std::vector<Color>& centroids, int k) {
    std::vector<int> cluster_sizes(k, 0);
    std::vector<Color> new_centroids(k, {0.0f, 0.0f, 0.0f});

    for (size_t i = 0; i < pixels.size(); i++) {
        int cluster_idx = assignments[i];
        new_centroids[cluster_idx].r += pixels[i].r;
        new_centroids[cluster_idx].g += pixels[i].g;
        new_centroids[cluster_idx].b += pixels[i].b;
        cluster_sizes[cluster_idx]++;
    }

    for (int i = 0; i < k; i++) {
        if (cluster_sizes[i] > 0) {
            new_centroids[i].r /= cluster_sizes[i];
            new_centroids[i].g /= cluster_sizes[i];
            new_centroids[i].b /= cluster_sizes[i];
        }
    }

    centroids = new_centroids;
}

// K-means algorithm for color quantization
void kmeans_cpu(const cv::Mat& image, int k, int max_iter, cv::Mat& output_image,int seed) {
    int width = image.cols;
    int height = image.rows;
    int num_pixels = width * height;

    // Convert image to a vector of Color structs
    std::vector<Color> pixels(num_pixels);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec3b color = image.at<cv::Vec3b>(i, j);
            pixels[i * width + j] = { color[2] / 255.0f, color[1] / 255.0f, color[0] / 255.0f };
        }
    }

    srand(seed);

    // Randomly initialize centroids
    std::vector<Color> centroids(k);
    for (int i = 0; i < k; i++) {
        centroids[i] = { float(rand()) / RAND_MAX, float(rand()) / RAND_MAX, float(rand()) / RAND_MAX };
    }

    std::vector<int> assignments(num_pixels, 0);
    for (int iter = 0; iter < max_iter; iter++) {
        // Step 1: Assign each pixel to the closest centroid
        for (int i = 0; i < num_pixels; i++) {
            float min_dist = std::numeric_limits<float>::max();
            for (int j = 0; j < k; j++) {
                float dist = euclidean_distance(pixels[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    assignments[i] = j;
                }
            }
        }

        // Step 2: Update centroids based on the pixel assignments
        update_centroids(pixels, assignments, centroids, k);
    }

    // Step 3: Create the output image
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

//driver code
int main(int argc, char* argv[]) {
    int clusters = DEFAULT_CLUSTERS;
    int iterations = DEFAULT_ITERATIONS;
    int seed = DEFAULT_SEED;
    int threads_per_block = DEFAULT_THREADS;
    std::string input_image_path = "images/image.jpg";
    std::string output_image_path = "images/cartoon_cpu.jpg";

    arg_parser(argc, argv, clusters, iterations, seed, threads_per_block, input_image_path, output_image_path);

    // Load the image
    cv::Mat image = cv::imread(input_image_path);
    if (image.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    cv::Mat output_image;
    kmeans_cpu(image, clusters, iterations, output_image, seed);

    // Save the output
    cv::imwrite(output_image_path, output_image);
    std::cout << "CPU output saved!" << std::endl;
    return 0;
}
