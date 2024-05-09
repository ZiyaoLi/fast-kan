# FastKAN: Very Fast Implementation (Approximation) of Kolmogorov-Arnold Network

Work in progress and as demo.

This repository contains a very fast implementation of Kolmogorov-Arnold Network (KAN).

The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan).

The implementation of efficient KAN is available [here](https://github.com/Blealtan/efficient-kan).

Demo code and efficient_kan code in this repo is from efficient_kan thanks to Blealtan.

The code is just a demo of how to approximately calculate Kolmogorov-Arnold Networks, with demo quality.

1. I used Gaussian Radial Basis Functions to approximate the B-spline basis, which is the bottleneck of KAN and efficient KAN:

$$b_{i}(u)=\exp(-(u-u_i)^2)$$

The rationale of doing so is that these RBF functions well approximate the B-spline basis (up to a linear transformation) and are very easy to calculate (as long as the grids are uniform). Results are shown in `draw_spline_basis.ipynb`.

2. I used LayerNorm to scale inputs to the range of spline grids, so there is no need (yet to test) to adjust the grids.

3. I tested the network on `MNIST` just like `efficient-kan` (run `train_mnist.py`). The accuracies are similar, yet running faster (63 it/s -> 68 it/s). The acceleration may be more significant on finer discretization of the grids and larger inputs.

More importantly this approximation suggests that KAN is equivalent to adding an RBF transformation to the inputs some place in the model. Someone may dig deeper into this for expression or approximation theories.