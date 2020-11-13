# OMODL: Off-The-Grid Model Based Deep Learning
OMODL is a model based off-the grid image reconstruction algorithm that uses deep learned priors. The main difference of the proposed scheme with current deep learning strategies is the learning of non-linear annihilation relations in Fourier space. It relies on a model based framework, which allows us to use a significantly smaller deep network, compared to direct approaches that also learn how to invert the forward model. Preliminary comparisons against image domain MoDL approach demonstrates the potential of the off-the-grid formulation. The main benefit of the proposed scheme compared to structured low-rank (SLR) methods is the quite significant reduction in computational complexity.

## Relevant Paper
Pramanik, Aniket, Hemant Aggarwal, and Mathews Jacob, "Off-The-Grid Model Based Deep Learning (O-MODL)", 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI). https://ieeexplore.ieee.org/document/8759403 
 
## Recursive OMODL Network
<img src="omodl.png"  title="hover text">

The network solves for two variables by alternating between

<img src="alternating_steps.png"  title="hover text" width="450px">

### Benefits of OMODL
1. It outperforms strutured low-rank (SLR) methods in terms of performance in SNR.

2. It is three orders of magnitude faster than SLR methods.
<img src="time_complexity.png"  title="hover text" width="600px">

## Code Details
