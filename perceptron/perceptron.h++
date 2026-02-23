#include "./matrix.h++"


namespace perceptron {
    
    typedef enum ACTIVATION_FUNCTION {
        RELU,
        
    } ACTIVATION_FUNCTION;
    
    class Layer {
        private:
        
        /* Forward */
        matrix::Matrix<float> w; //Weights
        matrix::Matrix<float> b; //Biases
        ACTIVATION_FUNCTION act; //Activation function
        
        /* Backward */
        matrix::Matrix<float> dw;
        matrix::Matrix<float> db;
        
        matrix::Matrix<float> x; //Input to layer
        matrix::Matrix<float> a; //Activated function
        //Network only supports relu, sigmoid and tanh
        //Means Z doesnt need to be stored, only A
        
        public:
        
    };
    class Perceptron {
        private:
        
        std::vector<Layer> layers;
    
        public:
        
    
    
    };
}




