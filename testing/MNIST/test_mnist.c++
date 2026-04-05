#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <random>
#include <chrono>
#include <cstdint>
#include "../../src/perceptron.h++"

//CONFIGURATION - Change these values
#define TRAIN_SAMPLES 10000   //Max 60000
#define TEST_SAMPLES 1000     //Max 10000
#define EPOCHS 10
#define LEARNING_RATE 0.1f
#define HIDDEN_NEURONS 128

//Simple MNIST reader - no fancy stuff
std::vector<std::vector<float>> read_mnist_images(std::string filename, int maxCount) {
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open()) {
        return {};
    }
    
    //Read header
    uint32_t magic, numImages, rows, cols;
    file.read((char*)&magic, 4);
    file.read((char*)&numImages, 4);
    file.read((char*)&rows, 4);
    file.read((char*)&cols, 4);
    
    //Swap endian (MNIST uses big-endian)
    magic = __builtin_bswap32(magic);
    numImages = __builtin_bswap32(numImages);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);
    
    int limit = maxCount < (int)numImages ? maxCount : numImages;
    
    std::vector<std::vector<float>> images;
    images.reserve(limit);
    
    for(int i = 0; i < limit; i++) {
        std::vector<float> img(rows * cols);
        for(int j = 0; j < rows * cols; j++) {
            unsigned char pixel;
            file.read((char*)&pixel, 1);
            img[j] = pixel / 255.0f;
        }
        images.push_back(img);
    }
    
    return images;
}

std::vector<int> read_mnist_labels(std::string filename, int maxCount) {
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open()) {
        return {};
    }
    
    uint32_t magic, numLabels;
    file.read((char*)&magic, 4);
    file.read((char*)&numLabels, 4);
    
    magic = __builtin_bswap32(magic);
    numLabels = __builtin_bswap32(numLabels);
    
    int limit = maxCount < (int)numLabels ? maxCount : numLabels;
    
    std::vector<int> labels;
    labels.reserve(limit);
    
    for(int i = 0; i < limit; i++) {
        unsigned char label;
        file.read((char*)&label, 1);
        labels.push_back(label);
    }
    
    return labels;
}

int main() {
    std::cout << "MNIST Test\n";
    std::cout << "==========\n\n";
    
    std::cout << "Configuration:\n";
    std::cout << "  Train samples: " << TRAIN_SAMPLES << "\n";
    std::cout << "  Test samples: " << TEST_SAMPLES << "\n";
    std::cout << "  Epochs: " << EPOCHS << "\n";
    std::cout << "  Learning rate: " << LEARNING_RATE << "\n";
    std::cout << "  Hidden neurons: " << HIDDEN_NEURONS << "\n\n";
    
    //Load data - simple paths like Iris
    std::cout << "Loading MNIST data...\n";
    
    auto trainImages = read_mnist_images("MNIST/train-images-idx3-ubyte", TRAIN_SAMPLES);
    if(trainImages.empty()) {
        std::cout << "Cant open MNIST/train-images-idx3-ubyte\n";
        std::cout << "Download from:\n";
        std::cout << "  wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz\n";
        std::cout << "  wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz\n";
        std::cout << "  wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz\n";
        std::cout << "  wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz\n";
        std::cout << "\nThen: gunzip to ./testing/MNIST/\n";
        return 1;
    }
    
    auto trainLabels = read_mnist_labels("MNIST/train-labels-idx1-ubyte", TRAIN_SAMPLES);
    auto testImages = read_mnist_images("MNIST/t10k-images-idx3-ubyte", TEST_SAMPLES);
    auto testLabels = read_mnist_labels("MNIST/t10k-labels-idx1-ubyte", TEST_SAMPLES);
    
    std::cout << "Train: " << trainImages.size() << " samples\n";
    std::cout << "Test: " << testImages.size() << " samples\n\n";
    
    //Build network
    std::vector<size_t> layers = {784, HIDDEN_NEURONS, 10};
    std::vector<perceptron::ACTIVATION_FUNCTION> acts = {perceptron::RELU, perceptron::SIGMOID};
    
    perceptron::Perceptron net(layers, acts);
    net.summary();
    
    //Prepare training targets (one-hot)
    std::vector<std::vector<float>> trainTargets;
    for(int i = 0; i < (int)trainLabels.size(); i++) {
        std::vector<float> target(10, 0.0f);
        target[trainLabels[i]] = 1.0f;
        trainTargets.push_back(target);
    }
    
    //Train
    std::cout << "Training...\n";
    auto startTime = std::chrono::steady_clock::now();
    
    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        float totalLoss = 0.0f;
        
        for(int i = 0; i < (int)trainImages.size(); i++) {
            net.forward(trainImages[i]);
            totalLoss += net.mse(trainTargets[i]);
            net.backward(trainTargets[i], LEARNING_RATE);
        }
        
        totalLoss /= trainImages.size();
        std::cout << "Epoch " << epoch + 1 << " loss: " << totalLoss << "\n";
    }
    
    auto endTime = std::chrono::steady_clock::now();
    float seconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0f;
    std::cout << "Training took " << seconds << " seconds\n";
    
    //Test
    std::cout << "\nTesting...\n";
    int correct = 0;


    for(int i = 0; i < (int)testImages.size(); i++) {
        const std::vector<float>& out = net.forward(testImages[i]);
        
        int guess = 0;
        float best = out[0];
        for(int j = 1; j < 10; j++) {
            if(out[j] > best) {
                best = out[j];
                guess = j;
            }
        }
        
        if(guess == testLabels[i]) {
            correct++;
        }
    }
    
    float acc = 100.0f * correct / testImages.size();
    std::cout << "Accuracy: " << acc << "%\n";
    

    //Show some predictions
    std::cout << "\nSample predictions:\n";
    for(int i = 0; i < 10; i++) {
        const std::vector<float>& out = net.forward(testImages[i]);
        
        int guess = 0;
        for(int j = 1; j < 10; j++) {
            if(out[j] > out[guess]) guess = j;
        }
        std::cout << "  Predicted: " << guess << " | Actual: " << testLabels[i];
        if(guess == testLabels[i]) std::cout << " Y";
        else std::cout << " N";
        std::cout << "\n";
    }
    
    if(acc > 90.0f) {
        std::cout << "\nPASS\n";
    } else if(acc > 80.0f) {
        std::cout << "\nOKAY but could be better\n";
        
    } else {
        std::cout << "\nFAIL\n";
    }
    
    return 0;
}
