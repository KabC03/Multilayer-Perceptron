#!/bin/bash
mkdir -p ./bin
clang++ ./testing/sinc\(x\)/bench.c++ -O3 -o ./bin/a.out
./bin/a.out
python3 ./testing/sinc\(x\)/plot.py
