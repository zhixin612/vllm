# export VLLM_LOGGING_LEVEL=DEBUG
# export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# model="/root/share/models/Qwen2.5-72B-Instruct"

# CUDA_VISIBLE_DEVICES=2,3,4,5  python3 \
#     -m vllm.entrypoints.openai.api_server \
#     --model $model \
#     --port 8100 \
#     --tensor-parallel-size 4 \
#     --max-num-batched-token 40000 \
#     --enforce-eager \
#     --max_num_seqs 128 \
#     --disable-log-requests \
#     --disable-log-stats \
#     --gpu-memory-utilization 0.99 \
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

model="/root/share/models/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=4  vllm serve\
    /root/share/models/Qwen2.5-7B-Instruct \
    --port 8100 \
    --tensor-parallel-size 1 \
    --max-num-batched-token 128000 \
    --enforce-eager \
    --max_num_seqs 256 \
    --disable-log-requests \
    --disable-log-stats \
    --gpu-memory-utilization 0.8 \