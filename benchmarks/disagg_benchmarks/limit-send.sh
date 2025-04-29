#!/bin/bash

# benchmark the overhead of disaggregated prefill.
# methodology:
# - send all request to prefill vLLM instance. It will buffer KV cache.
# - then send all request to decode instance. 
# - The TTFT of decode instance is the overhead.

set -ex

kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  sleep 10

  # remove vllm config file
  rm -rf ~/.config/vllm

  # Print the GPU memory usage
  # so that we know if all GPU processes are killed.
  gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  # The memory usage should be 0 MB.
  echo "GPU 0 Memory Usage: $gpu_memory_usage MB"
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


benchmark() {

  export VLLM_LOGGING_LEVEL=DEBUG
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  # compare chunked prefill with disaggregated prefill

  results_folder="./results"
  model="/root/share/models/Qwen2.5-7B-Instruct"
  # dataset_name="sonnet"
  # dataset_path="../sonnet_4x.txt"
  num_prompts=800
  qps=$1
  prefix_len=50
  input_len=128
  output_len=$2


 
  # let the prefill instance finish prefill
  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name random \
          --random-input-len $input_len \
          --random-output-len 1 \
          --num-prompts $num_prompts \
          --ignore-eos \
          --port 8100 \
          --save-result \
          --result-dir $results_folder \
          --percentile-metrics ttft,tpot,itl,e2el \
          --result-filename prefill.json \
          --request-rate 164 \
  

}


main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)

  pip install quart httpx datasets

  cd "$(dirname "$0")"

  cd ..
  # # create sonnet-4x.txt
  # echo "" > sonnet_4x.txt
  # for _ in {1..4}
  # do
  #   cat sonnet.txt >> sonnet_4x.txt
  # done
  cd disagg_benchmarks

  rm -rf results
  mkdir results

  default_qps=6
  default_output_len=100
  benchmark $default_qps $default_output_len

}


main "$@"
