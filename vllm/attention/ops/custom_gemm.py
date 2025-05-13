import torch


def custom_gemm(C: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> None:
    # add fallback here (e.g. when the dtype is not supported, fallback to torch.mm)
    # reference: merge_attn_states.py

    from vllm._custom_ops import custom_gemm
    return custom_gemm(C, A, B)
