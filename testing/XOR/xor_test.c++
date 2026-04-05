#include "../../src/perceptron.h++"
#include <iostream>
#include <vector>


int main() {
    std::cout << "XOR Test\n";
    std::cout << "========\n\n";

    //2 inputs, 4 hidden, 1 output
    std::vector<size_t> layers = {2, 4, 1};
    std::vector<perceptron::ACTIVATION_FUNCTION> acts = {perceptron::RELU, perceptron::SIGMOID};
    
    perceptron::Perceptron net(layers, acts);
    
    net.summary();
    
    //XOR data
    std::vector<std::vector<float>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<float>> targets = {
        {0}, {1}, {1}, {0}
    };
    
    //Train
    std::cout << "Training...\n";
    net.train(inputs, targets, 3000, 0.5f, true);
    
    //Test
    std::cout << "\nResults:\n";
    for(int i = 0; i < 4; i++) {
        float out = net.forward(inputs[i])[0];
        int pred = (out > 0.5f) ? 1 : 0;
        
        std::cout << inputs[i][0] << " XOR " << inputs[i][1] 
                  << " = " << out << " (" << pred << ") | expected " 
                  << targets[i][0];
                  
        if(pred == targets[i][0]) {
            std::cout << " PASS\n";
        } else {
            
            std::cout << " FAIL\n";
        }
    }
    
    return 0;
}



