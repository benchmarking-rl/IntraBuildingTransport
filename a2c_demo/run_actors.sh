#!/bin/bash

export CUDA_VISIBLE_DEVICES=""

for i in $(seq 1 10); do
    python rl_demo/actor.py &
done;
wait
