#include "../../perceptron/perceptron.h++"
#include "../../perceptron/matrix.h++"
#include <math.h>
#include <fstream>
#include <stdint.h>

typedef struct IMAGES {
    uint32_t count;
    uint32_t rows;
    uint32_t cols;
    std::vector<float> data;
} IMAGES;

typedef struct LABELS {
    uint32_t count;
    std::vector<uint8_t> labels;
} LABELS;


int main(int argc, char *argv[]) {

    if(argc != 2) {
        std::cout << "Expect -t (train) or -r (run) flag" << "\n";
        return -1;
    }
    std::string arg = std::string(argv[1]);
    if(arg != "-t" || arg != "-r") {
        std::cout << "Invalid flag pass; expect -t (train) or -r (run) flag" << "\n";
        return -2;
    }



    return 0;
}

