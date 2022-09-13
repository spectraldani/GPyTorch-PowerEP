# Sparse Gaussian processes using power expectation propagation for GPyTorch

This repository implements the Power EP algorithm for Sparse GPs as described
by [Bui & Yan & Turner (2017)](https://arxiv.org/abs/1605.07066v3). The code tries to
follow [the original implementation by Bui](https://github.com/thangbui/sparseGP_powerEP) and has comments referencing
the original code for comparison’s sake.

## Requirements

This was implemented against `torch==1.10.2, gpytorch==1.7.0`.

## Limitations

* The code does not implement hyperparameter training through gradients of the approximation of the evidence;
* No support for batched GPs;
* No support for multi-output GPs.