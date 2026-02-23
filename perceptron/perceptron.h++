#include <random>
#include "./matrix.h++"


namespace perceptron {
    
    typedef enum ACTIVATION_FUNCTION {
        RELU,
        
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
                    }
                }
                
                
                //Biases at zero
                for(size_t i = 0; i < neuronsNext * 1; i++) {
                    layer.b.at(i, 0) = 0;
                }
            }
        }
        
        /**
         * @brief Forward pass on a neural network (single batch)
         * 
         * @param &x :: Input vector matrix (copied to internal network). Is considered as a row vector internally
         * 
         * @return std::vector<float>& :: Pointer to internal network output
         */
        const std::vector<float> &forward(std::vector<float> &x) {
            this->input.fill_vector(x);
            matrix::Matrix<float> *inputMatrix = &(this->input);
            
            for(Layer &layer : this->layers) {
                layer.a.multiply(*inputMatrix, layer.w, false, true);
                layer.a.add(layer.b);
                
                switch(layer.act) {
                    case RELU: {
                        layer.a.activate_relu();   
                        break;
                    }
                }
                inputMatrix = &(layer.a);
            }            
            return this->layers.back.a.get_vector();
        }
        
        
        
        
    };
}




