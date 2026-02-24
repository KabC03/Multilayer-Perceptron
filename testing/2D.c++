#include "../perceptron/perceptron.h++"
#include "../perceptron/matrix.h++"
#include <math.h>
#include <fstream>

inline float func(float x) {
    return sin(x);
}

/* Test performance learning sin(x) */
int main(void) {
  
    const size_t samples = 1000;
    const float seperator = 0.01;
    const size_t epochs = 3000;
    const float lr = 0.01;

    std::vector<size_t> dims = {1, 32, 32, 1};
    std::vector<perceptron::ACTIVATION_FUNCTION> acts = {
        perceptron::ACTIVATION_FUNCTION::TANH,
        perceptron::ACTIVATION_FUNCTION::TANH,
        perceptron::ACTIVATION_FUNCTION::TANH,
    };
    perceptron::Perceptron network(dims, acts);


    std::vector<float> x;
    x.reserve(samples);
    std::vector<float> y;
    y.reserve(samples);
    std::vector<float> p;
    p.reserve(samples);

    for(size_t i = 0; i < samples; i++) {
        x.push_back(i * seperator);
        y.push_back(func(i * seperator));
    } 

    std::cout << "Running training loop" << "\n";

    for(size_t j = 0; j < epochs; j++) {
        for(size_t i = 0; i < samples; i++) {

            std::vector<float> xVec = {x[i]}; //Network requres vector input
            std::vector<float> yVec = {y[i]}; //Network requres vector input

            network.forward(xVec);
            network.backward(yVec, lr);
        }
    }




    //Network output 
    std::cout << "Running network output" << "\n";
    for(size_t i = 0; i < samples; i++) {
        std::vector<float> xVec = {x[i]}; //Network requres vector input
        std::vector<float> yVec = {y[i]}; //Network requres vector input

        p.push_back(network.forward(xVec)[0]);
    }


    std::ofstream file("./output/out.txt");
    if(!file) {
        std::cout << "Failed to open output test file" << std::endl;
        return -1;
    }
    for(auto item : x) {
        file << item << " ";
    }
    file << "\n";
    for(auto item : y) {
        file << item << " ";
    }
    file << "\n";
    for(auto item : p) {
        file << item << " ";
    }
    file << "\n";

    network.save_file("./output/test1Network.txt");

    return 0;
}

