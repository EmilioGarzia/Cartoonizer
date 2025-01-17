# NVIDIA Nsight Compute report

> ⚠️: You have to use Nsight Compute tool, available on NVIDIA official website

To run the UI Nsight Compute tool, execute the command as shown below:

`ncu-ui report.ncu-rep`

# Session's informations

| Input info | Value |
|:-:|:-:|
| Session ID | `2` |
| Image width | `3840px` |
| Image height | `2160px` |
| Image channels | `RGB (3 channels)` |
| #Clusters | `8` |
| #Iterations | `50` |
| #Threads | `128` |
| Random seed | `200` |

# Results

| Metric | Value |
|:-:|:-:|
| Execution time (Parallel version)| `~4397ms (4.39s)` |
| Execution time (Sequential version)| `~41379ms (41.37s)` |
| Delta time | `+36982 ms (+36.98s)` |
| Speedup | `9.4` |

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