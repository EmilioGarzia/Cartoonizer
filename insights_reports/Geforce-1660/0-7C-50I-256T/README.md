# NVIDIA Nsight Compute report

> ⚠️: You have to use Nsight Compute tool, available on NVIDIA official website

To run the UI Nsight Compute tool, execute the command as shown below:

`ncu-ui report.ncu-rep`

# Session's informations

| Input info | Value |
|:-:|:-:|
| Session ID | `0` |
| Image width | `1200px` |
| Image height | `800px` |
| Image channels | `RGB (3 channels)` |
| #Clusters | `7` |
| #Iterations | `50` |
| #Threads | `256` |
| Random seed | `200` |

# Results

| Metric | Value |
|:-:|:-:|
| Execution time (Parallel version)| `~569ms (0.59s)` |
| Execution time (Sequential version)| `~5034ms (5s)` |
| Delta time | `+4438 ms (+4.4s)` |
| Speedup | `8.84` |

# GPU specs

| Property | Value |
|:-:|:-:|
| GPU Name | `NVIDIA GeForce GTX 1660` |
| Architecture | `Turing` |
| Compute Capability | `7.5` |
| Driver Version | `550.120` |
| CUDA Version | `12.4` |
| VRAM  | `6GB GDDR5` |
| CUDA Cores  | `1536` |
| Clock frequency | `1.53GHz` |

# CPU specs

| Property | Value |
|:-:|:-:|
| CPU Name | `Intel(R) Core(TM) i5-9600K` |
| Cores | `6` |
| Threads | `6` |
| Cache size | `192 KiB L1d, 192 KiB L1i, 1.5 MiB L2, 9 MiB L3` |
| Clock frequency | `2.2GHz` |
| RAM | `16Gb DDR4` |
| RAM frequency | `2133MHz` |

# Image result

![input](./input_image.jpg)

> Input Image

![output](output_image.jpg)

> Output Image

# Authors

* Emilio Garzia
* Luigi Marino