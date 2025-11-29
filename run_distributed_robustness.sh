#!/bin/bash

# Distributed Robustness Analysis Script
# Usage: ./run_distributed_robustness.sh [num_processes]

NUM_PROCESSES=${1:-2}

echo "🚀 Starting Distributed Robustness Analysis"
echo "Number of processes: $NUM_PROCESSES"
echo "=================================="

# Set environment variables
export HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-$HF_TOKEN}

# Run with accelerate
accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --mixed_precision=fp16 \
    --main_process_port=29500 \
    perturb_multi.py

echo "✅ Distributed analysis completed!"
