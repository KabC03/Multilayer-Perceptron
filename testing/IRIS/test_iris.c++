#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <random>
#include <chrono>
#include "../../src/perceptron.h++"

int main() {
    std::cout << "Iris Test\n";
    std::cout << "=========\n\n";
    
    //Load the data - file is in same directory (IRIS/ folder)
    std::ifstream file("./IRIS/iris.data");
    if(!file.is_open()) {
        std::cout << "Cant open iris.data, download from 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' and place it in ./testing/IRIS/\n";
        return 1;
    }
    
    std::vector<std::vector<float>> inputs;
    std::vector<int> labels;
    
    std::string line;
    while(std::getline(file, line)) {

        if(line.length() == 0) continue;
        
        //Parse numbers - look for commas
        std::vector<float> vec;
        std::string num = "";
        
        for(int i = 0; i < (int)line.length(); i++) {
            if(line[i] == ',') {
                float val = std::stof(num);
                vec.push_back(val);
                num = "";
            } else {

                num = num + line[i];
            }
        }
        
        //Last thing left is the label
        std::string labelName = num;
        
        //Normalize each feature. Min/max ranges from UCI repo
        vec[0] = (vec[0] - 4.3f) / 3.6f;  //sepal length
        vec[1] = (vec[1] - 2.0f) / 2.4f;  //sepal width
        vec[2] = (vec[2] - 1.0f) / 5.9f;  //petal length
        vec[3] = (vec[3] - 0.1f) / 2.4f;  //petal width
        
        inputs.push_back(vec);
        
        //Turn label string into number
        if(labelName == "Iris-setosa") {
            labels.push_back(0);
        } else if(labelName == "Iris-versicolor") {
            labels.push_back(1);
        } else {
            labels.push_back(2);
        }
    }
    
    std::cout << "Loaded " << inputs.size() << " flowers\n";
    
    //Make a list of indices to shuffle
    std::vector<int> order;
    for(int i = 0; i < (int)inputs.size(); i++) {
        order.push_back(i);
    }
    
    //Shuffle - swap each index with a random one
    std::random_device rd;
    std::mt19937 rng(rd());
    for(int i = (int)order.size() - 1; i > 0; i--) {
        int j = rng() % (i + 1);
        int temp = order[i];
        order[i] = order[j];
        order[j] = temp;
    }
    
    //Split into train and test. 70% train, 30% test
    int trainSize = 105;
    
    std::vector<std::vector<float>> trainIn;
    std::vector<std::vector<float>> trainTarget;
    std::vector<std::vector<float>> testIn;
    std::vector<int> testLabel;
    
    for(int i = 0; i < (int)order.size(); i++) {
        int idx = order[i];

        if(i < trainSize) {
            trainIn.push_back(inputs[idx]);
            
            //One-hot encoding for target
            std::vector<float> target(3, 0.0f);
            target[labels[idx]] = 1.0f;
            trainTarget.push_back(target);

        } else {
            testIn.push_back(inputs[idx]);
            testLabel.push_back(labels[idx]);
        }
    }
    
    std::cout << "Training samples: " << trainIn.size() << "\n";
    std::cout << "Testing samples: " << testIn.size() << "\n\n";
    
    //Build network. 4 inputs, 6 hidden, 3 outputs
    std::vector<size_t> layers = {4, 6, 3};
    std::vector<perceptron::ACTIVATION_FUNCTION> acts = {perceptron::RELU, perceptron::SIGMOID};
    
    perceptron::Perceptron net(layers, acts);
    net.summary();
    
    //Time the training
    std::cout << "Training...\n";
    auto startTime = std::chrono::steady_clock::now();
    
    for(int epoch = 0; epoch < 300; epoch++) {
        float totalLoss = 0.0f;
        
        //Loop through all training samples
        for(int i = 0; i < (int)trainIn.size(); i++) {

            net.forward(trainIn[i]);
            totalLoss = totalLoss + net.mse(trainTarget[i]);
            net.backward(trainTarget[i], 0.15f);
        }
        
        totalLoss = totalLoss / trainIn.size();
        
        //Print every 100 epochs
        if(epoch % 100 == 0 || epoch == 299) {
            std::cout << "Epoch " << epoch << " loss: " << totalLoss << "\n";
        }
    }
    
    auto endTime = std::chrono::steady_clock::now();
    float seconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0f;
    std::cout << "Training took " << seconds << " seconds\n";
    
    //Test on unseen data
    std::cout << "\nTesting...\n";
    int correct = 0;
    for(int i = 0; i < (int)testIn.size(); i++) {

        const std::vector<float>& out = net.forward(testIn[i]);
        
        //Pick class with highest output
        int guess = 0;
        float bestVal = out[0];
        for(int j = 1; j < 3; j++) {
            if(out[j] > bestVal) {
                bestVal = out[j];
                guess = j;
            }
        }
        
        if(guess == testLabel[i]) {
            correct++;
        }
    }
    
    float acc = 100.0f * correct / testIn.size();
    std::cout << "\nAccuracy: " << acc << "%\n";
    
    if(acc > 90.0f) {
        std::cout << "PASS\n";
    } else {
        std::cout << "FAIL\n";
        
    }
    
    return 0;
}



