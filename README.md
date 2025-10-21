<div align="center">
<h2>UniAD Model Lightweighting and Performance Comparison</h2>
</div>

- [2024/10]: Manuscript submitted (received on October 31, 2024).
- [2024/12]: Manuscript revised (on December 21, 2024).
- [2025/02]: Manuscript accepted (on February 15, 2025).
- [2025/04]: This work has been published in the Institute of Control, Robotics and Systems (ICROS, SCOPUS)

<p align="center">
    <a href='https://doi.org/10.5302/J.ICROS.2025.24.0249'><img src="https://img.shields.io/badge/Paper-PDF-blue?style=flat&#x26;logo=doi&#x26;logoColor=yello" alt="Paper PDF"></a>
</p>

🚀 This project base on [UniAD](https://github.com/OpenDriveLab/UniAD)
---
## Abstract

Recently developed autonomous driving systems based on deep learning typically operate through modular architectures, where separate modules perform distinct individual tasks. While the UniAD framework proposed in the “Planning-oriented Autonomous Driving” paper addresses the limitations of modular approaches through a unified architecture, its complex transformer structure requires substantial computational resources to function. This paper proposes a lightweight version of UniAD to improve the accessibility of multimodal learning. We reduce the computational complexity by lowering the number of transformer layers and queries, the dimensions, and the BEV spatial resolution. Additionally, we optimize memory usage by limiting sampling queries and enabling page-locked memory settings. Experiments with two versions of the lightweight architecture show significant memory reductions: up to 79.92% in Stage 1 and 38.81% in Stage 2 compared with the original UniAD architecture (52.3 GB and 16.67 GB, respectively). Although the lightweight model suffers an overall performance degradation, we discover that progressive resolution expansion during training can enhance its feature extraction capability, particularly in the initial low-resolution learning phase.


## Keywords
deep learning, autonomous driving, multimodal, lightweighting

<img width="2268" height="612" alt="image" src="https://github.com/user-attachments/assets/68925d75-7801-4b57-bdef-3352c32c2eab" />


### Table 1. Comparison of structures by lightweight version
*The two values in the table represent the structures in stages 1 and 2, respectively.*
| Method | GPU*N (NVIDIA) | Encoder | embed dimensions | layers | heads | FFN channels | BEV resolution | Queue length (stage-1) | BEV queries | Seg queries | Learning Rate | Grid Size | Query dimensions (stage-2) | Pin memory | Memory threshold |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| UniAD[3] | A100*16 | R101 | 256 | 6 | 8 | 2048 | 200x200 | 5 | 900 | 300 | 2e-4 | 512 | 256 | False | 3 |
| DV-1 | 3090*3 | R101 | 128 | 4 | 8 | 1024 | 200x200 | 3 | 900 | 300 | 1e-6<br>1e-4 | 512<br>256 | 128 | False | 3 |
| DV-2 | A100*1 | R50 | 128 | 4 | 4 | 1024 | 50x50<br>200x200 | 3 | 900 | 300 | 2e-4 | 512 | 128 | True | 1 |

### Table 2. Comparison of experimental results by lightweight version
*The two values in the Memory section represent stages 1 and 2, respectively.*
| Method | Memory (GB) | AMOTA ↑ [%] | AMOTP ↓ [m] | mIDS ↓ [count] | IoU-lane ↑ [%] | IoU-road ↑ [%] | minADE ↓ [m] | minFDE ↓ [m] | MR ↓ [%] | EPA ↓ [%] | IoU-o. ↑ [%] | IoU-f. ↑ [%] | VPQ-n. ↑ [%] | VPQ-f. ↑ [%] | avgL2 ↓ [m] | avgCol. ↓ [%] |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| UniAD (Base) | 52.3<br>16.67 | **0.359** | **1.32** | 906 | 0.313 | **0.691** | **0.708** | **1.025** | 0.151 | **0.456** | **63.4** | **40.2** | **54.7** | **33.5** | 1.03 | **0.31** |
| UniAD (Small) | Files not public<br>Test unavailable | 0.241 | 1.488 | 958 | **0.315** | 0.689 | 0.788 | 1.126 | 0.156 | 0.381 | 59.4 | 35.6 | 49.2 | 28.9 | 1.04 | 0.32 |
| DV-1 | 17.9<br>10.4 | 0.000 | 1.946 | 1493 | 0.253 | 0.615 | **0.995** | **1.274** | **0.130** | -0.077 | 25.7 | 9.8 | 15.4 | 5.7 | 1.25 | 0.88 |
| DV-2 | 10.5<br>10.2 | **0.016** | **1.816** | **753** | **0.304** | **0.684** | 1.157 | 1.658 | 0.208 | 0.108 | **43.0** | **22.0** | **29.6** | **15.0** | **0.97** | **0.75** |

## Results Comparison
<img width="4660" height="1289" alt="image" src="https://github.com/user-attachments/assets/e821f36c-7f0e-4221-a986-a82055c5d5be" />
<img width="4660" height="1290" alt="image" src="https://github.com/user-attachments/assets/d4e1bbe5-4674-481f-aded-b45048472f58" />
<img width="4660" height="1289" alt="image" src="https://github.com/user-attachments/assets/12e01a70-f0d5-4e64-bb6b-09c1ef47d2a1" />

