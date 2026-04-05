#!/bin/bash

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BIN_DIR="../bin"
TEST_DIR="."

mkdir -p $BIN_DIR

# Check if files exist
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: $1 not found${NC}"
        return 1
    fi
    return 0
}

# Compile a single test
compile_test() {
    local name=$1
    local source=$2
    
    echo -e "${BLUE}compiling $name...${NC}"
    
    if ! check_file "$source"; then
        return 1
    fi
    
    clang++ -std=c++17 -O3 "$source" -o "$BIN_DIR/$name"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}$name done${NC}"
        return 0
    else
        echo -e "${RED}$name failed${NC}"
        return 1
    fi
}

# Run a test
run_test() {
    local name=$1
    
    if [ ! -f "$BIN_DIR/$name" ]; then
        echo -e "${RED}$name not compiled. Run compile first${NC}"
        return 1
    fi
    
    echo -e "\n${YELLOW}--- $name ---${NC}"
    "$BIN_DIR/$name"
}

# Compile everything
compile_all() {
    echo -e "${BLUE}compiling all tests...${NC}"
    echo "================================"
    
    compile_test "xor_test" "$TEST_DIR/XOR/xor_test.c++"
    compile_test "save_load_test" "$TEST_DIR/Save_load/save_load_test.c++"
    compile_test "iris_test" "$TEST_DIR/IRIS/test_iris.c++"
    compile_test "mnist_test" "$TEST_DIR/MNIST/test_mnist.c++"
    
    echo "================================"
    echo -e "${GREEN}compile complete${NC}"
}

# Run all tests
run_all() {
    echo -e "${BLUE}running all tests...${NC}"
    echo "================================"
    
    run_test "xor_test"
    run_test "save_load_test"
    run_test "iris_test"
    run_test "mnist_test"
    
    echo "================================"
    echo -e "${GREEN}testing complete${NC}"
}

# Run a specific test
run_single() {
    case "$1" in
        xor)
            run_test "xor_test"
            ;;
        save)
            run_test "save_load_test"
            ;;
        iris)
            run_test "iris_test"
            ;;
        mnist)
            run_test "mnist_test"
            ;;
        *)
            echo "options: xor, save, iris, mnist"
            ;;
    esac
}

# Clean binaries
clean() {
    echo -e "${BLUE}cleaning...${NC}"
    rm -rf "$BIN_DIR"
    echo -e "${GREEN}clean done${NC}"
}

# Print usage
usage() {
    echo "options:"
    echo "  compile              - compile all tests"
    echo "  compile-xor          - compile XOR test only"
    echo "  compile-save         - compile Save/Load test only"
    echo "  compile-iris         - compile Iris test only"
    echo "  compile-mnist        - compile MNIST test only"
    echo "  run                  - run all tests"
    echo "  run-xor              - run XOR test"
    echo "  run-save             - run Save/Load test"
    echo "  run-iris             - run Iris test"
    echo "  run-mnist            - run MNIST test"
    echo "  clean                - remove compiled binaries"
    echo "  help                 - show this message"
}

# Main
case "$1" in
    compile)
        compile_all
        ;;
    compile-xor)
        compile_test "xor_test" "$TEST_DIR/XOR/xor_test.c++"
        ;;
    compile-save)
        compile_test "save_load_test" "$TEST_DIR/Save_load/save_load_test.c++"
        ;;
    compile-iris)
        compile_test "iris_test" "$TEST_DIR/IRIS/test_iris.c++"
        ;;
    compile-mnist)
        compile_test "mnist_test" "$TEST_DIR/MNIST/test_mnist.c++"
        ;;
    run)
        run_all
        ;;
    run-xor)
        run_single "xor"
        ;;
    run-save)
        run_single "save"
        ;;
    run-iris)
        run_single "iris"
        ;;
    run-mnist)
        run_single "mnist"
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
esac



