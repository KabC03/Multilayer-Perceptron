#include <iostream>
#include <vector>
#include <cmath>
#include "../../src/perceptron.h++"

int main() {
    std::cout << "Save/Load Test\n";
    std::cout << "==============\n\n";
    
    //Simple network: 2 inputs, 3 hidden, 1 output
    std::vector<size_t> layers = {2, 3, 1};
    std::vector<perceptron::ACTIVATION_FUNCTION> acts = {perceptron::RELU, perceptron::SIGMOID};
    
    perceptron::Perceptron netA(layers, acts);
    
    std::cout << "Network A (before training):\n";
    netA.summary();
    
    //Train on a simple pattern
    std::vector<std::vector<float>> inputs = {{1, 0}, {0, 1}, {1, 1}};
    std::vector<std::vector<float>> targets = {{1}, {1}, {0}};
    
    std::cout << "Training network A...\n";
    netA.train(inputs, targets, 500, 0.3f, false);
    
    //Get output after training
    std::vector<float> testInput = {1, 0};
    float outputBefore = netA.forward(testInput)[0];
    std::cout << "Output for [1,0] before save: " << outputBefore << "\n\n";
    
    //Save to file
    std::cout << "Saving to file...\n";
    if(netA.save_file("./Save_load/test_model.net")) {
        std::cout << "Save successful\n\n";

    } else {
        
        std::cout << "Save FAILED\n";
        return 1;
    }
    
    //Create new network (different size to test that load rebuilds correctly)
    std::vector<size_t> layersWrong = {2, 10, 5, 1};
    std::vector<perceptron::ACTIVATION_FUNCTION> actsWrong = {perceptron::RELU, perceptron::RELU, perceptron::SIGMOID};
    perceptron::Perceptron netB(layersWrong, actsWrong);
    
    std::cout << "Network B (different architecture before load):\n";
    netB.summary();
    
    //Load into netB - should overwrite architecture
    std::cout << "Loading from file...\n";
    if(netB.read_file("./Save_load/test_model.net")) {
        std::cout << "Load successful\n\n";
    } else {
        std::cout << "Load FAILED\n";
        return 1;
    }
    
    std::cout << "Network B (after load):\n";
    netB.summary();
    
    //Get output after load
    float outputAfter = netB.forward(testInput)[0];
    std::cout << "Output for [1,0] after load: " << outputAfter << "\n\n";
    
    //Compare
    float diff = std::abs(outputBefore - outputAfter);
    if(diff < 0.0001f) {
        std::cout << "PASS: Outputs match (diff = " << diff << ")\n";
    } else {

        std::cout << "FAIL: Outputs don't match (diff = " << diff << ")\n";
    }
    
    return 0;
}




