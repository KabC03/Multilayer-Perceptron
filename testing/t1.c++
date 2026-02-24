#include "../perceptron/perceptron.h++"
#include "../perceptron/matrix.h++"


int main(void) {

    std::vector<size_t> dimensions = {3, 4, 2};
    std::vector<perceptron::ACTIVATION_FUNCTION> act = {
        perceptron::ACTIVATION_FUNCTION::RELU,
        perceptron::ACTIVATION_FUNCTION::RELU,
        perceptron::ACTIVATION_FUNCTION::RELU,
    };
    perceptron::Perceptron perceptron(dimensions, act);

    std::vector<float> x = {0.1, 0.2, 0.3};

    perceptron.save_file("./bin/network.txt");

    perceptron.read_file("./bin/network.txt");

    perceptron.save_file("./bin/network_1.txt");


    auto vec = perceptron.forward(x);
    for(auto item : vec) {
        std::cout << item << ", ";
    }

    return 0;
}

