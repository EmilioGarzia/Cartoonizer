# NVIDIA Nsight Compute report

> ⚠️: You have to use Nsight Compute tool, available on NVIDIA official website

To run the UI Nsight Compute tool, execute the command as shown below:

`ncu-ui report.ncu-rep`

# Session's informations

| Input info | Value |
|:-:|:-:|
| Session ID | `3` |
| Image width | `7680px` |
| Image height | `5120px` |
| Image channels | `RGB (3 channels)` |
| #Clusters | `5` |
| #Iterations | `50` |
| #Threads | `64` |
| Random seed | `200` |

# Results

| Metric | Value |
|:-:|:-:|
| Execution time (Parallel version)| `~23428ms (23.4s)` |
| Execution time (Sequential version)| `~ 138567ms (2.3 minutes)` |
| Delta time | `+115139ms (+1.9 minutes)` |
| Speedup | `5.9` |

# GPU specs

| Property | Value |
|:-:|:-:|
| GPU Name | `NVIDIA GeForce 4060` |
| Architecture | `Ada Lovelace` |
| Compute Capability | `8.9` |
| Driver Version | `560.35.03` |
| CUDA Version | `12.6` |
| VRAM  | `8GB GDDR6` |
| CUDA Cores  | `3072` |
| Clock frequency | `2.46GHz` |

# CPU specs

| Property | Value |
|:-:|:-:|
| CPU Name | `AMD Ryzen 5 3600` |
| Cores | `6` |
| Threads | `12` |
| Cache size | `512Kb` |
| Clock frequency | `2.2GHz` |
| RAM | `16Gb DDR4` |
| RAM frequency | `3600MHz` |

# Image result

![input](./input_image.jpg)

> Input Image

![output](output_image.jpg)

> Output Image

# Authors

* Emilio Garzia
* Luigi Marino