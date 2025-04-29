#!/bin/bash

set -ex

# kill all processes on GPU.
pgrep pt_main_thread | xargs -r kill -9
pgrep python3 | xargs -r kill -9
sleep 3

# remove vllm config file
rm -rf ~/.config/vllm

# Print the GPU memory usage
# so that we know if all GPU processes are killed.
gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
# The memory usage should be 0 MB.
echo "GPU 0 Memory Usage: $gpu_memory_usage MB"
