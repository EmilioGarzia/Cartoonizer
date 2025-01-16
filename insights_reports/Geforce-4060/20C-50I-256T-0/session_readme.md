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
| #Clusters | `20` |
| #Iterations | `50` |
| #Threads | `256` |

# Used kernels

| Kernel signature | Kernal launch parameters |
|:-:|:-:|
| `assign_pixels_to_centroids()` | `<<<(num_pixels+nThreads-1)/nThreads, nThreads>>>` |
| `update_centroids()` | `<<<(num_clusters+nThreads-1)/nThreads, nThreads>>>` |
| `update_centroids()` | `<<<(num_clusters+nThreads-1)/nThreads, nThreads>>>` |

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