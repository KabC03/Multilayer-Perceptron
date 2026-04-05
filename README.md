


# C++ based multilayer perceptron
A single-threaded multilayer perceptron written in C++. Uses only the STL. Pre-allocates all internal structures at startup for predictable memory use.

## Features
- CPU only

- Choose activation functions per layer: ReLU, Sigmoid, or Tanh

- Batch size is always 1

- Save and load networks from files

## Project Structure
| Directory	| What's inside |
| ----- | ----- |
| ./src	| Network source code |
| ./testing	| Various tests for network |
|./bin |	Compiled binaries |


## Requirements
- C++17 or newer. No external libraries.
- Linux or WSL

## Basic Usage
``` cpp
#include "perceptron.h++"

//Perceptron 
using namespace perceptron;

int main(void) {

    //2 input neurons, 4 hidden neurons, 1 output neuron
    std::vector<size_t> layers = {2, 4, 1}; 
    //RELU activation for layer 0->1, then SIGMOID activation for layer 4->1
    std::vector<ACTIVATION_FUNCTION> activations = {RELU, SIGMOID};


    //Create a perceptron with a corresponding layer with activations
    perceptron::Perceptron net(layers, activations);

    //Create input vector with values 0.1, -0.2
    std::vector<float> input = {0.5, -0.2};
    //Pass input vector to network, perform forward pass then extract output
    const std::vector<float>& output = net.forward(input);

    //Calculate loss of network
    float loss = net.mse(expected_output);

    //Perform back propagation with 0.01f learning rate
    net.backward(expected_output, 0.01f);

    //Save network params to a file
    if(net.save_file("model.net")) {
        std::cout << "Successful save!" << "\n";
    }

    //Load a network from a file into a network
    if(net.read_file("model.net")) {
        std::cout << "Successful load!" << "\n";
    }

    return 0;
}
```



## Memory
The network allocates all weights, biases, and gradients when you build it. This means fewer heap allocations during runtime. The tradeoff is higher memory usage. Each layer keeps both forward and backward buffers in memory at the same time.

## Limitations
- Batch size fixed at 1

- No GPU support

- Single-threaded only

- Only supports activations not requiring a Z matrix

## File Format
Each layer gets written as one block in the format:

- Activation function (stored as an integer)

- Weight matrix

- Bias vector


## Testing
The library includes four tests that validate different aspects of the network. Each test is self-contained and prints PASS/FAIL results.

### Run All Tests
```bash
cd testing
chmod +x run_all.sh
./run_all.sh
```

### Individual Tests

| Test | Location | What it tests |
|------|----------|----------------|
| XOR | testing/XOR/xor_test.c++ | Non-linear learning and backpropagation |
| Save/Load | testing/Save_load/save_load_test.c++ | File I/O and weight persistence |
| Iris | testing/IRIS/test_iris.c++ | Multi-class classification on real data |
| MNIST | testing/MNIST/test_mnist.c++ | Scalability on 784-dimension images |

### Running Tests

Use the test runner script:

```bash
cd testing
chmod +x run_tests.sh

# Compile all tests
./run_tests.sh compile

# Run all tests
./run_tests.sh run

# Run a single test
./run_tests.sh run-xor
./run_tests.sh run-iris
./run_tests.sh run-mnist

# Clean compiled binaries
./run_tests.sh clean
```




