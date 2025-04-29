#!/bin/bash

set -ex

benchmark() {
  export VLLM_LOGGING_LEVEL=INFO
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  CUDA_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model "/root/share/models/Qwen2.5-7B-Instruct" \
    --port 9001 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --max-num-batched-token 350000 \
    --max_num_seqs 256 \
    --enforce-eager \
    --gpu-memory-utilization 0.95 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e11}'
}


main() {
  default_qps=6
  default_output_len=512
  benchmark $default_qps $default_output_len
}


main "$@"
