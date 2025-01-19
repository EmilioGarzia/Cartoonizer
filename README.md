# Cartoonizer

**Cartoonizer** is a project designed to apply a "cartoon" effect to images using **quantization**. Specifically, we implemented the **K-means** algorithm to achieve this effect.

This is an educational project aimed at showcasing the main advantages of parallel computing using the CUDA framework (GP-GPU). To this end, we developed two different versions of the same program (written in `C++`): one implements a sequential approach (CPU), while the other leverages the CUDA framework (GPU) to apply parallelization.

Below is a summary of the main objectives:

- Implementing the K-means algorithm in `CUDA C++`.
- Developing a sequential version of the same code.
- Performing tests with different inputs, first on the CPU and then on various GPUs.
- Comparing and analyzing the collected results, including using tools such as `NVIDIA Nsight Compute`
- Prepare a report that explains all the work done

# Used technologies

* `CUDA C++`
* `OpenCV 4`
* `NVIDIA Nsight Compute`
* `NVIDIA Nsight Systems`
* `nvprof`

# How to run

`sequential.cpp` contains the source code for the sequential version of the program, basically you could run binary file as shown below:

```shell
./sequential "/path/path/image.extension"
```

By the way, the program has an arguments parser useful to specify different inputs in order to achive the wanted result:

```bash
./sequential -c <clusters> -i <iterations> -s <random_seed> "/path/input_image.jpg" "/path/output_image.jpg"
```

You can run the parallel version in the same way, but in this case there is  an additional input argument, as shown below:

```bash
./parallel -c <clusters> -i <iterations> -s <random_seed> -t <threads_per_block> "/path/input_image.jpg" "/path/output_image.jpg"
```

## Default values

You can skip specifying the input arguments; by default the inputs are:

| Argument | Default value |
|:-:|:-:|
| Clusters | 7 |
| Iterations | 50 |
| Random seed | 200 |
| Threads per block | 256 |

# Author

* *Emilio Garzia*
* *Luigi Marino*
* *Emilio Garzia*
* *Luigi Marino*