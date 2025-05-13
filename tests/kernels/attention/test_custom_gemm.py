import pytest
import torch

from vllm._custom_ops import custom_gemm


# 检查是否有可用的 CUDA 设备
CUDA_AVAILABLE = torch.cuda.is_available()
if not CUDA_AVAILABLE:
    print("警告：没有可用的 CUDA 设备。自定义算子测试将被跳过。")

# Pytest 的 skipif 装饰器
skip_if_cuda_unavailable = pytest.mark.skipif(not CUDA_AVAILABLE, reason="没有可用的 CUDA 设备")


def _check_tensor_equality(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: float, atol: float, config_str: str):
    """Helper function to check tensor equality and assert."""
    are_equal = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    if not are_equal:
        max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
        # For better debugging, you might want to print parts of the tensors
        # print(f"\nTensor 1 (custom) for {config_str} (first 5x5):\n{tensor1[:5,:5]}")
        # print(f"Tensor 2 (reference) for {config_str} (first 5x5):\n{tensor2[:5,:5]}")
    assert are_equal, (
        f"自定义 GEMM 输出与 PyTorch matmul 参考输出不匹配。\n"
        f"配置: {config_str}\n"
        f"最大差异: {torch.max(torch.abs(tensor1 - tensor2)).item() if not are_equal else 'N/A'}"
    )


@skip_if_cuda_unavailable
@pytest.mark.parametrize("M, K, N, dtype_str, rtol, atol", [
    # Float32 tests
    (128, 256, 64, "float32", 1e-3, 1e-3),
    (256, 256, 256, "float32", 1e-3, 1e-3),  # 方阵
    (512, 128, 1024, "float32", 1e-3, 1e-3),
    (1, 512, 1, "float32", 1e-3, 1e-3),     # 向量-向量外积形式
    (1024, 1, 512, "float32", 1e-3, 1e-3),  # 列向量乘以行向量, and a small K
    (1024, 32, 512, "float32", 1e-3, 1e-3),  # 列向量乘以行向量, and a small K
] + [
    # Float16 tests (adjust rtol/atol as needed)
    # (128, 256, 64, "float16", 1e-2, 1e-2),
    # (256, 256, 256, "float16", 1e-2, 1e-2),
    # (511, 127, 1023, "float16", 1e-2, 1e-2),  # 奇数维度
] + [
    # BFloat16 tests
    # (128, 256, 64, "bfloat16", 1e-2, 1e-2),
]
)
def test_custom_gemm(M: int, K: int, N: int, dtype_str: str, rtol: float, atol: float, seed: int = 0):
    """
    测试自定义 GEMM 算子 (C = A @ B) 的不同配置。
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str)
    if dtype is None:
        pytest.skip(f"不支持的 dtype_str: {dtype_str}")

    config_str = f"M={M}, K={K}, N={N}, dtype={dtype}"

    # 检查 GPU 是否支持特定数据类型
    if dtype == torch.float16:
        if not torch.cuda.is_available() or not torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 7:
            pytest.skip(f"当前 GPU 不完全支持 float16 或 CUDA 不可用，跳过测试: {config_str}")
    elif dtype == torch.bfloat16:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            pytest.skip(f"当前 GPU 不支持 bfloat16 或 CUDA 不可用，跳过测试: {config_str}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda"

    # 1. 创建输入张量 A 和 B
    try:
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)
    except Exception as e:
        pytest.skip(f"创建 {dtype} 类型的张量失败 (可能是 GPU 不支持): {e} - 配置: {config_str}")
        return

    # 2. 创建输出张量 C (用于自定义算子)
    C_custom = torch.empty(M, N, dtype=dtype, device=device)

    # 3. 调用自定义 GEMM 算子
    try:
        custom_gemm(C_custom, A, B)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            pytest.skip(f"CUDA out of memory for {config_str}")
            return
        elif "not implemented for" in str(e) and str(dtype) in str(e):
            pytest.skip(f"自定义算子可能未针对 {dtype} 实现: {e} - 配置: {config_str}")
            return
        raise e # 重新抛出其他运行时错误

    # 4. 使用 PyTorch 内置的 matmul 作为参考
    C_reference = torch.matmul(A, B)

    # 5. 比较结果
    _check_tensor_equality(C_custom, C_reference, rtol, atol, config_str)
    print(f"测试通过: {config_str}")
