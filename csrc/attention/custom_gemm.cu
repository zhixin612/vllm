#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// 对于更复杂的核函数，可以考虑创建一个 "custom_gemm_kernels.cuh" 头文件
// #include "custom_gemm_kernels.cuh"

// 常用的宏定义
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace vllm {

// 自定义 GEMM CUDA 核函数
// 计算 C = A * B
// A: [M, K], B: [K, N], C: [M, N]
// TILE_DIM_M, TILE_DIM_N, TILE_DIM_K: 定义每个线程块处理的瓦片大小
// BLOCK_ROWS, BLOCK_COLS: 定义每个线程块中的线程数量 (二维)
template <
    typename scalar_t,
    int TILE_DIM_M,
    int TILE_DIM_N,
    int TILE_DIM_K, // TILE_DIM_K 通常用于共享内存优化中的子块大小
    int BLOCK_ROWS, // blockDim.y
    int BLOCK_COLS  // blockDim.x
>
__global__ void custom_gemm_kernel(
    scalar_t* __restrict__ C_ptr,         // 输出矩阵 C 的指针
    const scalar_t* __restrict__ A_ptr,   // 输入矩阵 A 的指针
    const scalar_t* __restrict__ B_ptr,   // 输入矩阵 B 的指针
    const int M,                          // 矩阵 A 的行数，矩阵 C 的行数
    const int N,                          // 矩阵 B 的列数，矩阵 C 的列数
    const int K,                          // 矩阵 A 的列数，矩阵 B 的行数
    const int stride_a_m,                 // 矩阵 A 行主序下的行跨度 (A.stride(0))
    const int stride_a_k,                 // 矩阵 A 行主序下的列跨度 (A.stride(1))
    const int stride_b_k,                 // 矩阵 B 行主序下的行跨度 (B.stride(0))
    const int stride_b_n,                 // 矩阵 B 行主序下的列跨度 (B.stride(1))
    const int stride_c_m,                 // 矩阵 C 行主序下的行跨度 (C.stride(0))
    const int stride_c_n                  // 矩阵 C 行主序下的列跨度 (C.stride(1))
) {
    // 这是一个基础的 GEMM 实现，每个线程计算输出矩阵 C 中的一个元素。
    // 生产级的 GEMM 会使用共享内存进行分块 (tiling) 以提高访存效率，
    // 并可能使用更复杂的线程协作模式以及 warp 级别的原语。

    // 计算当前线程负责的输出元素在 C 中的全局行索引
    const int global_row_c = blockIdx.y * TILE_DIM_M + threadIdx.y;
    // 计算当前线程负责的输出元素在 C 中的全局列索引
    const int global_col_c = blockIdx.x * TILE_DIM_N + threadIdx.x;

    // 边界检查，确保线程不会越界访问
    if (global_row_c < M && global_col_c < N) {
        scalar_t accumulator = static_cast<scalar_t>(0.0f);
        // 沿 K 维度进行点积计算
        for (int k_idx = 0; k_idx < K; ++k_idx) {
            // A[global_row_c, k_idx] * B[k_idx, global_col_c]
            accumulator += A_ptr[global_row_c * stride_a_m + k_idx * stride_a_k] *
                           B_ptr[k_idx * stride_b_k + global_col_c * stride_b_n];
        }
        C_ptr[global_row_c * stride_c_m + global_col_c * stride_c_n] = accumulator;
    }
}

} // namespace vllm


// 自定义 GEMM 核函数的启动器 (launcher)
template <typename scalar_t>
void custom_gemm_launcher(
    torch::Tensor& C,       // 输出张量 C
    torch::Tensor& A,       // 输入张量 A
    torch::Tensor& B        // 输入张量 B
) {
    // 输入参数校验
    TORCH_CHECK(A.scalar_type() == B.scalar_type() && A.scalar_type() == C.scalar_type(),
                "自定义 GEMM：所有张量必须具有相同的数据类型");
    TORCH_CHECK(A.device() == B.device() && A.device() == C.device() && A.is_cuda(),
                "自定义 GEMM：所有张量必须位于同一个 CUDA 设备上");

    TORCH_CHECK(A.dim() == 2, "自定义 GEMM：张量 A 必须是二维的");
    TORCH_CHECK(B.dim() == 2, "自定义 GEMM：张量 B 必须是二维的");
    TORCH_CHECK(C.dim() == 2, "自定义 GEMM：张量 C 必须是二维的");

    // 获取张量维度
    const int M = A.size(0);
    const int K_A = A.size(1); // A 的 K 维度
    const int K_B = B.size(0); // B 的 K 维度
    const int N = B.size(1);

    TORCH_CHECK(K_A == K_B, "自定义 GEMM：A 的内维度 (K) 必须与 B 的内维度 (K) 匹配");
    TORCH_CHECK(M == C.size(0), "自定义 GEMM：C 的维度 0 (M) 必须与 A 的维度 0 (M) 匹配");
    TORCH_CHECK(N == C.size(1), "自定义 GEMM：C 的维度 1 (N) 必须与 B 的维度 1 (N) 匹配");

    // 获取指向张量数据的原始指针
    scalar_t* C_ptr = reinterpret_cast<scalar_t*>(C.data_ptr());
    const scalar_t* A_ptr = reinterpret_cast<const scalar_t*>(A.data_ptr());
    const scalar_t* B_ptr = reinterpret_cast<const scalar_t*>(B.data_ptr());

    // 获取张量的步长 (stride)
    // PyTorch 张量可能不是内存连续的，使用 stride() 获取正确的内存偏移
    const int stride_a_m = A.stride(0);
    const int stride_a_k = A.stride(1);
    const int stride_b_k = B.stride(0);
    const int stride_b_n = B.stride(1);
    const int stride_c_m = C.stride(0);
    const int stride_c_n = C.stride(1);

    // 定义核函数启动配置
    // 这些值应该根据目标 GPU 架构和具体问题进行调整以获得最佳性能。
    // 为简单起见，这里使用固定的瓦片和块维度。
    // TILE_DIM_M 和 TILE_DIM_N 定义了每个线程块在 M 和 N 维度上计算的输出元素数量。
    // BLOCK_ROWS 和 BLOCK_COLS 定义了每个线程块内的线程布局。
    constexpr int TILE_DIM_M_VAL = 16;
    constexpr int TILE_DIM_N_VAL = 16;
    constexpr int TILE_DIM_K_VAL = 16; // 对于当前简单核函数，K维度瓦片不直接在模板参数中使用，但保留用于未来扩展（如共享内存）
    constexpr int BLOCK_ROWS_VAL = 16; // blockDim.y
    constexpr int BLOCK_COLS_VAL = 16; // blockDim.x

    // 线程块维度
    dim3 threads_per_block(BLOCK_COLS_VAL, BLOCK_ROWS_VAL);
    // 网格维度 (Grid)
    // 计算在 N 维度 (x) 和 M 维度 (y) 上需要的线程块数量
    dim3 num_blocks(
        DIVIDE_ROUND_UP(N, TILE_DIM_N_VAL),
        DIVIDE_ROUND_UP(M, TILE_DIM_M_VAL)
    );

    // 设置当前 CUDA 设备和流
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 启动 CUDA 核函数
    // 注意：custom_gemm_kernel 的模板参数在这里是硬编码的。
    // 在更通用的解决方案中，可能需要像 paged_attention_v1.cu 中的 LAUNCH_PAGED_ATTENTION_V1 那样，
    // 使用宏或进一步的模板化来根据特定条件（如矩阵大小）选择这些参数。
    vllm::custom_gemm_kernel<
        scalar_t,
        TILE_DIM_M_VAL,
        TILE_DIM_N_VAL,
        TILE_DIM_K_VAL,
        BLOCK_ROWS_VAL,
        BLOCK_COLS_VAL
    ><<<num_blocks, threads_per_block, 0, stream>>>(
        C_ptr, A_ptr, B_ptr,
        M, N, K_A, // K_A 即为 K 维度
        stride_a_m, stride_a_k,
        stride_b_k, stride_b_n,
        stride_c_m, stride_c_n
    );

    // 检查 CUDA 核函数启动是否成功 (通常在 debug 模式下有用)
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// 定义一个宏，用于根据数据类型调度到特定的模板化启动器函数
// 这类似于 vLLM 中常见的 DISPATCH_BY_XXX_DTYPE 宏
#define DISPATCH_CUSTOM_GEMM_BY_DTYPE(scalar_dtype_enum, launcher_fn_name, ...) \
  [&] { \
    if (scalar_dtype_enum == at::ScalarType::Float) { \
      launcher_fn_name<float>(__VA_ARGS__); \
    } else if (scalar_dtype_enum == at::ScalarType::Half) { \
      launcher_fn_name<at::Half>(__VA_ARGS__); \
    } else if (scalar_dtype_enum == at::ScalarType::BFloat16) { \
      launcher_fn_name<at::BFloat16>(__VA_ARGS__); \
    } else { \
      TORCH_CHECK(false, "自定义 GEMM：不支持的数据类型: ", scalar_dtype_enum); \
    } \
  }()


// 这是最终暴露给 Python 层调用的 C++ 函数 (通常通过 Pybind11)
void custom_gemm(
    torch::Tensor& C,       // 输出张量 C
    torch::Tensor& A,       // 输入张量 A
    torch::Tensor& B        // 输入张量 B
) {
    // 使用调度宏调用相应数据类型的启动器函数
    DISPATCH_CUSTOM_GEMM_BY_DTYPE(A.scalar_type(), custom_gemm_launcher, C, A, B);
}
