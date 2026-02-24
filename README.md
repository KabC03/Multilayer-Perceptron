# Tiny C++ Perceptron

This is a small, single threaded, CPU based multilayer perceptron (MLP) written in C++.
No external libraries have been used for the actual implementation; only what is in the STL.

The perceptron can be quite memory intensive due to its focus on reducing the number of heap allocations.
Internal structures are pre-allocated ahead of time.


## Features
- CPU only
- Supports a configurable number of layers, each with their own activation function (Relu, Sigmoid, Tanh)
- Forward and backward pass (batch size == 1)
- Saving and loading networks from files


## Structure

**./perceptron** :: Source files for network

**./testing** :: Tests for a 2D function 

**./output** :: Output of saved network states for tests

**./bin** :: Binary files








