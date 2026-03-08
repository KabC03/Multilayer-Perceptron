#include "../../perceptron/perceptron.h++"
#include "../../perceptron/matrix.h++"
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
    float start = 5 * -3.141;
    float end = 5 * 3.141;
    float step = 0.001; //Step
    size_t epochs = 100;

    perceptron::Perceptron network(neurons, acts);
   
    std::vector<float> x = {0};
    std::vector<float> y = {0};
    std::vector<float> p = {0};

    //Forward pass and training
    for(size_t j = 0; j < epochs; j++) {
        for(float i = start; i < end; i += step) {
            x[0] = i;
            y[0] = sin(i) / i;
            network.forward(x);
            network.backward(y, lr);
        }
    }



    std::ofstream outFile;
    outFile.open("./output/benchmark_sinc(x).txt");
    if(!outFile) {
        return -1;
    }


    std::vector<float> xs, ys, ps;

    float mse = 0;
    //Forward pass
    for(float i = start; i < end; i += step) {
        x[0] = i;
        y[0] = sin(i) / i;
        p[0] = network.forward(x)[0];
        mse += network.mse(y);

        xs.push_back(x[0]);
        ys.push_back(y[0]);
        ps.push_back(p[0]);
    }

    std::cout << "Average MSE: " << mse << "\n";


    for(float item : xs) {
        outFile << item;
        outFile << " ";
    }
    outFile << "\n";
    for(float item : ys) {
        outFile << item;
        outFile << " ";
    }
    outFile << "\n";
    for(float item : ps) {
        outFile << item;
        outFile << " ";
    }
    outFile << "\n";

    outFile.close();

    return 0;
}


