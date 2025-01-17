# NVPROF report

> ⚠️: For this specific GPU we used NVPROF, because the compute capability is lower than 7.0

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
| Execution time (Parallel version)| `~12030 (12.03 seconds)` |
| Execution time (Sequential version)| `~105250 (1.75 minutes)` |
| Delta time | `+93220ms (+1.55 minutes)` |
| Speedup | `8.74` |

# GPU specs

| Property | Value |
|:-:|:-:|
| GPU Name | `Quadro K5000` |
| Architecture | `Kepler` |
| Compute Capability | `3.0` |
| Driver Version | `450.66` |
| CUDA Version | `11.0` |
| VRAM  | `4GB GDDR5` |
| CUDA Cores  | `1536` |
| Clock frequency | `0.71GHz` |

# CPU specs

| Property | Value |
|:-:|:-:|
| CPU Name | `Intel(R) Core(TM) i7` |
| Cores | `4` |
| Cache size | `8192KB` |
| Clock frequency | `2.8 GHz` |
| RAM | `7.6GB DDR3` |
| RAM frequency | `~1333MHz` |

# Image result

![input](./input_image.jpg)

> Input Image

![output](output_image.jpg)

> Output Image

# Authors

* Emilio Garzia
* Luigi Marino