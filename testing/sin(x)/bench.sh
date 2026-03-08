#!/bin/bash
mkdir -p ./bin
clang++ ./testing/sin\(x\)/bench.c++ -O3 -o ./bin/a.out
./bin/a.out
python3 ./testing/sin\(x\)/plot.py

