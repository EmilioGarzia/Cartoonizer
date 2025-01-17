# NVIDIA Nsight Compute report

> ⚠️: You have to use Nsight Compute tool, available on NVIDIA official website

To run the UI Nsight Compute tool, execute the command as shown below:

`ncu-ui report.ncu-rep`

# Session's informations

| Input info | Value |
|:-:|:-:|
| Session ID | `4` |
| Image width | `1200px` |
| Image height | `800px` |
| Image channels | `RGB (3 channels)` |
| #Clusters | `15` |
| #Iterations | `100` |
| #Threads | `128` |
| Random seed | `200` |

# Results

| Metric | Value |
|:-:|:-:|
| Execution time (Parallel version)| `~974ms (0.9s)` |
| Execution time (Sequential version)| `~4860ms (4.86s)` |
| Delta time | `+3886ms (+3.88s)` |
| Speedup | `4.9` |

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