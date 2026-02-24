#include <random>
#include <stdexcept>
#include <string>
#include <fstream>
#include "./matrix.h++"


namespace perceptron {
    
    typedef enum ACTIVATION_FUNCTION {
        RELU,
        SIGMOID,
        TANH, 
    } ACTIVATION_FUNCTION;
    
    class Layer {
        public:
        //AOS or SOA doesnt really matter here, since each array encapsulates another structure
        
        /* Forward */
        matrix::Matrix<float> w; //Weights
        matrix::Matrix<float> b; //Biases
        ACTIVATION_FUNCTION act; //Activation function
        
        /* Backward */
        matrix::Matrix<float> dw;
        matrix::Matrix<float> db;
        
        //NOTE: X is not stored since A is just X to the next layer
        matrix::Matrix<float> a; //Activated function
        //Network only supports relu, sigmoid and tanh
        //Means Z doesnt need to be stored, only A
        
        //Preallocated for backpropagation
        matrix::Matrix<float> dA;
        matrix::Matrix<float> dZ; 
    };
    class Perceptron {
        private:
        matrix::Matrix<float> input; //Input to layer
        std::vector<Layer> layers;

        

    
        public:
        /**
         * @brief Construct a neural network
         * 
         * @param neuronsPerLayer :: Vector of neurons in each layer. Index 0 is considered as layer one, and the end is the output layer
         * @param activationFunctionsPerLayer :: Vector of ACTIVATION_FUNCTION's (corrosponding to neurons perlayer)
         */
        Perceptron(std::vector<size_t> &neuronsPerLayer, std::vector<ACTIVATION_FUNCTION> &activationFunctionsPerLayer) {
             
            this->input = matrix::Matrix<float>(neuronsPerLayer.front(), 1);
            
            if(neuronsPerLayer.size() -1 != activationFunctionsPerLayer.size()) {
                throw std::invalid_argument("Incompatable neuron and activation function vector dimensions\n");
            }
             
            std::mt19937 rng(std::random_device{}());
            this->layers.reserve(neuronsPerLayer.size() - 1);
            
            for(size_t i = 0; i < neuronsPerLayer.size() - 1; i++) {
                size_t neuronsCurrent = neuronsPerLayer[i];
                size_t neuronsNext = neuronsPerLayer[i+1];

                ACTIVATION_FUNCTION act = activationFunctionsPerLayer[i];
                
                
                //Construct layer in place
                this->layers.emplace_back();
                Layer &layer = this->layers.back();
                
                
                //TODO should give layer a constructor
                //Garuntees copy elision
                layer.w = matrix::Matrix<float>(neuronsNext, neuronsCurrent);
                layer.b = matrix::Matrix<float>(neuronsNext, 1);
                layer.a = matrix::Matrix<float>(neuronsNext, 1);
                layer.dw = matrix::Matrix<float>(neuronsNext, neuronsCurrent);
                layer.db = matrix::Matrix<float>(neuronsNext, 1);
                layer.act = act;
                
                layer.dA = matrix::Matrix<float>(neuronsCurrent, 1);
                layer.dZ = matrix::Matrix<float>(neuronsNext, 1);


                //Randomise layer
                //RELU layers recieves He randomisation
                //Sigmoid/Tanh recieves Xavier randomisation
                switch(layer.act) {
                    case RELU: {
                        std::normal_distribution<float> dist(0, std::sqrt(((float)2)/neuronsCurrent));

                        for(size_t i = 0; i < neuronsNext; i++) {
                            for(size_t j = 0; j < neuronsCurrent; j++) {
                                
                                layer.w.at(i, j) = dist(rng);
                            }
                        }                        
                        break;
                    } case SIGMOID: {
                        [[fallthrough]];
                    } case TANH: {
                        float nIn = (float)(neuronsCurrent);
                        float nOut = (float)(neuronsNext);

                        float d = sqrt((float)(6)/(nIn + nOut));
                        std::uniform_real_distribution<float> dist(-d, d);
                        for(size_t i = 0; i < neuronsNext; i++) {
                            for(size_t j = 0; j < neuronsCurrent; j++) {
                                layer.w.at(i, j) = dist(rng);
                            }
                        }

                        break;
                    }
                }
                
                
                //Biases at zero
                for(size_t i = 0; i < neuronsNext * 1; i++) {
                    layer.b.at(i, 0) = 0;
                }
            }
        }


        /**
         * @brief Save a network into a file (weights + biases only)
         * 
         * @param fileName :: File name to save the network state too
         * 
         * @return bool :: Indication of if save was successful
         */
        bool save_file(std::string fileName) {
            //TODO: Should work on making this not reopen a file many, many times

            std::ofstream file(fileName); //Clear file
            if(!file) {
                return false;
            }
            file.close();
            for(Layer layer : this->layers) {

                std::ofstream file(fileName, std::ios::app); //Write activation function
                if(!file) {
                    return false;
                }
                file << layer.act << "\n";
                file.close();


                if(!layer.w.append_to_file(fileName)) {
                    return false;
                }

                if(!layer.b.append_to_file(fileName)) {
                    return false;
                }
            }

            return true;
        }

        /**
         * @brief Read a network from a file
         * 
         * @param fileName :: File name to load the network state from
         * 
         * @return bool :: Indication of if save was successful
         */
        bool read_file(std::string fileName) {
            //need to improve error handling
            std::ifstream file(fileName);
            if(!file) {
                return false;
            }
            layers.clear();
            while(1) {
                std::string str;
                if(!std::getline(file, str)) {
                    break;
                }

                ACTIVATION_FUNCTION act = (ACTIVATION_FUNCTION)std::stoi(str);

                //Need to reduce redundant copies here
                matrix::Matrix<float> weight;
                if(!weight.read_float_file_fstream(file)) {
                    return false;
                }

                matrix::Matrix<float> bias;
                if(!bias.read_float_file_fstream(file)) {
                    return false;
                }

                Layer layer = {
                    .act = act,
                    .w = weight,
                    .b = bias,
                };
                size_t neuronsNext = layer.w.get_rows();
                size_t neuronsCurrent = layer.w.get_cols();
                layer.a = matrix::Matrix<float>(neuronsNext, 1);
                layer.dw = matrix::Matrix<float>(neuronsNext, neuronsCurrent);
                layer.db = matrix::Matrix<float>(neuronsNext, 1);
                layer.dA = matrix::Matrix<float>(neuronsCurrent, 1);
                layer.dZ = matrix::Matrix<float>(neuronsNext, 1);

                layers.push_back(layer);
            }
            if(layers.empty()) {
                return false;
            }
            this->input = matrix::Matrix<float>(layers.front().w.get_cols(), 1);
            return true;
        }

        
        /**
         * @brief Forward pass on a neural network (single batch)
         * 
         * @param &x :: Input vector matrix (copied to internal network). Is considered as a row vector internally
         * 
         * @return std::vector<float>& :: Pointer to internal network output
         */
        const std::vector<float> &forward(std::vector<float> &x) {
            
            if(x.size() != this->input.get_vector().size()) {
                throw std::invalid_argument("Network was passed incompatable x dimension\n");
            }
            
            this->input.fill_vector(x);
            matrix::Matrix<float> *inputMatrix = &(this->input);
            
            for(Layer &layer : this->layers) {
                layer.a.multiply(layer.w, *inputMatrix, false, false);
                layer.a.add(layer.b);
                
                switch(layer.act) {
                    case RELU: {
                        layer.a.activate_relu();   
                        break;
                    } case SIGMOID: {
                        layer.a.activate_sigmoid();
                        break;
                    } case TANH: {
                        layer.a.activate_tanh();
                        break;
                    }
                }
                inputMatrix = &(layer.a);
            }            
            return this->layers.back().a.get_vector();
        }
        
        /**
         * @brief Backpropagation implementation. Batch size == 1
         * 
         * @param y :: Expected network output
         * @param lr :: Learning rate during back propagation
         * 
         * @return void :: None 
         */
        void backward(std::vector<float> &y, float lr) {
            
            Layer &out = this->layers.back();
            if(y.size() != out.a.get_vector().size()) {
                throw std::invalid_argument("Network was passed incompatable y dimension\n");
            }

            //Compute gradients dZ, dW, dB
            for(size_t i = this->layers.size() - 1; i != SIZE_MAX; i--) {
                Layer &layer = this->layers[i];

                matrix::Matrix<float> *aPrev = NULL;
                if(i == 0) {
                    aPrev = &(this->input);
                } else {
                    aPrev = &(this->layers[i - 1].a);
                }

                //Find dZ
                if(i == this->layers.size() - 1) { //Back layer

                    //dA = (A - y), dZ = Hadamard(dA, G'(A))
                    for(size_t j = 0; j < layer.dZ.get_rows(); j++) {
                        layer.dZ.at(j, 0) = layer.a.at(j, 0) - y[j];
                    }

                } else { //Hidden layer
                    //dZ = wNext^T * dZNext

                    Layer &next = this->layers[i + 1];
                    layer.dZ.multiply(next.w, next.dZ, true, false);
                }

                //Use dB as temp buffer, gets overwritten anyway and has same size
                //dB = G'(A)
                //dZ = hadamard(dZ, dB)
                for(size_t j = 0; j < layer.db.get_rows(); j++) {
                    layer.db.at(j, 0) = layer.a.at(j, 0);
                }

                switch(layer.act) {
                    case RELU: {
                        layer.db.activate_derivative_relu();
                        break;
                    } case SIGMOID: {
                        layer.db.activate_derivative_sigmoid();
                        break;
                    } case TANH: {
                        layer.db.activate_derivative_tanh();
                        break;
                    }
                }
                layer.dZ.hadamard(layer.db);


                //dW = dZ * aPrev^T
                layer.dw.multiply(layer.dZ, *aPrev, false, true);

                //dB = dZ
                for(size_t j = 0; j < layer.db.get_rows(); j++) {
                    layer.db.at(j, 0) = layer.dZ.at(j, 0);
                }
            }
            //Second pass to update params
            for(Layer &layer : this->layers) {

                //W -= lr * dW
                for(size_t i = 0; i < layer.w.get_rows(); i++) {
                    for(size_t j = 0; j < layer.w.get_cols(); j++) {
                        layer.w.at(i, j) -= lr * layer.dw.at(i, j);
                    }
                }

                for(size_t i = 0; i < layer.b.get_rows(); i++) {
                    layer.b.at(i, 0) -= lr * layer.db.at(i, 0);
                }
            }
            return;
        }




    };
}




