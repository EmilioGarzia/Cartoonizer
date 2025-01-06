/*
 @brief Image cartoonization using K-means colors quantization: SEQUENTIAL CPU VERSION
 @file sequential.cpp
 @author Emilio Garzia, Luigi Marino
 @date 2025
 @details g++ sequential.cpp -o sequential -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
*/

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    float r, g, b;
} Color;

// Assign pixels to centroids routine
void assign_pixels_to_centroids(Color* pixels, Color* centroids, int* assignments, int num_pixels, int num_centroids) {
    for (int i = 0; i < num_pixels; i++) {
        float min_dist = FLT_MAX;
        int closest_centroid = 0;
        Color pixel = pixels[i];

        for (int j = 0; j < num_centroids; j++) {
            float dist = (pixel.r - centroids[j].r) * (pixel.r - centroids[j].r) +
                         (pixel.g - centroids[j].g) * (pixel.g - centroids[j].g) +
                         (pixel.b - centroids[j].b) * (pixel.b - centroids[j].b);
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid = j;
            }
        }
        assignments[i] = closest_centroid;
    }
}

// Update centroids routine
void update_centroids(Color* pixels, int* assignments, Color* centroids, int* cluster_sizes, int num_pixels, int num_centroids) {
    // Reset centroids and dimensions of the clusters
    for (int i = 0; i < num_centroids; i++) {
        centroids[i].r = 0.0f;
        centroids[i].g = 0.0f;
        centroids[i].b = 0.0f;
        cluster_sizes[i] = 0;
    }

    // Update centroids
    for (int i = 0; i < num_pixels; i++) {
        int cluster_idx = assignments[i];
        centroids[cluster_idx].r += pixels[i].r;
        centroids[cluster_idx].g += pixels[i].g;
        centroids[cluster_idx].b += pixels[i].b;
        cluster_sizes[cluster_idx]++;
    }

    // Compute the new average value
    for (int i = 0; i < num_centroids; i++) {
        if (cluster_sizes[i] > 0) {
            centroids[i].r /= cluster_sizes[i];
            centroids[i].g /= cluster_sizes[i];
            centroids[i].b /= cluster_sizes[i];
        }
    }
}

// Routine that make the output image
void create_cartoon_image(Color* pixels, int* assignments, Color* centroids, int num_pixels) {
    for (int i = 0; i < num_pixels; i++) {
        Color pixel = pixels[i];
        int closest_centroid = assignments[i];

        pixel.r = centroids[closest_centroid].r;
        pixel.g = centroids[closest_centroid].g;
        pixel.b = centroids[closest_centroid].b;

        pixels[i] = pixel;
    }
}

// driver code
int main() {
    const int num_clusters = 20; 
    const int max_iterations = 50;

    cv::Mat image = cv::imread("images/image.jpg");

    if (image.empty()) {
        printf("ERROR: An error were occurred during the image loading\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int num_pixels = width * height;

    double start = (double)clock()/CLOCKS_PER_SEC;

    Color* pixels = (Color*)malloc(num_pixels * sizeof(Color));
    for (int i = 0; i < num_pixels; i++) {
        cv::Vec3b color = image.at<cv::Vec3b>(i / width, i % width);
        pixels[i].b = (float)color[0] / 255.0f;
        pixels[i].g = (float)color[1] / 255.0f;
        pixels[i].r = (float)color[2] / 255.0f;
    }

    srand(42);

    Color* centroids = (Color*)malloc(num_clusters * sizeof(Color));
    for (int i = 0; i < num_clusters; i++) {
        centroids[i].r = (float)rand() / RAND_MAX;
        centroids[i].g = (float)rand() / RAND_MAX;
        centroids[i].b = (float)rand() / RAND_MAX;
    }

    int* assignments = (int*)malloc(num_pixels * sizeof(int));
    int* cluster_sizes = (int*)malloc(num_clusters * sizeof(int));

    // K-means execution
    for (int iter = 0; iter < max_iterations; iter++) {
        assign_pixels_to_centroids(pixels, centroids, assignments, num_pixels, num_clusters);
        update_centroids(pixels, assignments, centroids, cluster_sizes, num_pixels, num_clusters);
    }

    create_cartoon_image(pixels, assignments, centroids, num_pixels);
    for (int i = 0; i < num_pixels; i++) {
        cv::Vec3b new_color(
            (unsigned char)(pixels[i].b * 255),
            (unsigned char)(pixels[i].g * 255),
            (unsigned char)(pixels[i].r * 255)
        );
        image.at<cv::Vec3b>(i / width, i % width) = new_color;
    }

    cv::imwrite("images/cartoon_image.jpg", image);

    double end = (double)clock()/CLOCKS_PER_SEC;
    double delta = end-start;
    std::cout << "Execution time: " << delta << "s" << std::endl;
  
    free(pixels);
    free(centroids);
    free(assignments);
    free(cluster_sizes);

    return 0;
}