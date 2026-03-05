#include "../perceptron/perceptron.h++"
#include "../perceptron/matrix.h++"
#include <math.h>
#include <fstream>


int main(void) {

    //Network params
        //Neurons
    std::vector<size_t> neurons = {
        1, 32, 32, 1,
    };
        //Acts
    std::vector<perceptron::ACTIVATION_FUNCTION> acts = {
        perceptron::ACTIVATION_FUNCTION::RELU,
        perceptron::ACTIVATION_FUNCTION::RELU,
        perceptron::ACTIVATION_FUNCTION::TANH,
    };
    float lr = 0.01;
    float start = -1;
    float end = 1;
    float step = 0.01; //Step
    size_t epochs = 1000;
    size_t msePrint = 100;

    perceptron::Perceptron network(neurons, acts);
   
    std::vector<float> x = {0};
    std::vector<float> y = {0};

    for(size_t j = 0; j < epochs; j++) {
        for(float i = start; i < end; i += step) {
            x[0] = i;
            y[0] = sin(i);
            network.forward(x);
            network.backward(y, lr);
        }
        if(j % msePrint == 0) {
            std::cout << network.mse(y) << "\n";
        }
    }
    std::cout << network.mse(y) << "\n";

    return 0;
}


