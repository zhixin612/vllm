#!/bin/bash

set -ex

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
  results_folder="./results"
  model="/root/share/models/Qwen2.5-7B-Instruct"
  num_prompts=2000
  qps=4
  prefix_len=50
  input_len=256
  output_len=256

 
  # let the prefill instance finish prefill
  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name random \
          --percentile-metrics ttft,tpot,itl,e2el \
          --random-input-len $input_len \
          --random-output-len 1 \
          --random-prefix-len $prefix_len \
          --num-prompts $num_prompts \
          --ignore-eos \
          --port 9000 \
          --save-result \
          --result-dir $results_folder \
          --result-filename prefill.json \
          --request-rate "inf" \

  
  # send the request to decode.
  # The TTFT of this command will be the overhead of disagg prefill impl.
  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name random \
          --random-input-len $input_len \
          --random-output-len $output_len \
          --random-prefix-len $prefix_len \
          --percentile-metrics ttft,tpot,itl,e2el \
          --ignore-eos \
          --num-prompts $num_prompts \
          --port 9001 \
          --save-result \
          --result-dir $results_folder \
          --result-filename decode.json \
          --request-rate $qps \

  kill_gpu_processes

  python update.py

}


main() {
#  rm -rf results
#  mkdir results

  benchmark
}


main "$@"
