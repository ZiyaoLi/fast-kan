# FastKAN: Very Fast Implementation (Approximation) of Kolmogorov-Arnold Network

Work in progress and as demo.

This repository contains a very fast implementation of Kolmogorov-Arnold Network (KAN).

The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan).

The implementation of efficient KAN is available [here](https://github.com/Blealtan/efficient-kan).

Demo code and efficient_kan code in this repo is from efficient_kan thanks to Blealtan.

The code is just a demo of how to approximately calculate Kolmogorov-Arnold Networks, with demo quality.

1. Used Gaussian Radial Basis Functions to approximate the B-spline basis, which is the bottleneck of KAN and efficient KAN:

$$b_{i}(u)=\exp\left(-\left(\frac{u-u_i}{h}\right)^2\right)$$

The rationale of doing so is that these RBF functions well approximate the B-spline basis (up to a linear transformation) and are very easy to calculate (as long as the grids are uniform). Results are shown in the figure below (code in [notebook](draw_spline_basis.ipynb)). 

![RBF well approximates 3-order B-spline basis.](img/compare_basis.png)

2. Used LayerNorm to scale inputs to the range of spline grids, so there is no need to adjust the grids.

3. The running speed of forward calculation is approximately 2x compared with efficient_kan. (see [notebook](test_running_time.ipynb), 640us -> 320us on V100)

More importantly this approximation suggests that KAN is equivalent to adding an RBF transformation to the inputs some place in the model. Someone may dig deeper into this for expression or approximation theories.