# 2.2 架构特定特性

## 概述

每个 GPU 架构都有其独特的硬件特性。本节详细介绍各架构的关键特性及其在 Prima.cpp 中的应用。

## Pascal 架构 (SM 60, 61)

### 关键特性

- **FP16 存储支持**：可以存储 FP16 数据，但计算仍使用 FP32
- **__dp4a 指令**（SM 6.1+）：4 字节整数点积

### __dp4a 指令详解

```cpp
// dp4a = Dot Product of 4 bytes with Accumulate
// 计算：a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + c

int result = __dp4a(int_a, int_b, int_c);
```

### Prima.cpp 中的实现

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    // AMD GPU 实现
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(RDNA2)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3)
    c = __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    // 软件实现
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
    c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
    return c;

#else  // NVIDIA GPU
#if __CUDA_ARCH__ >= MIN_CC_DP4A
    return __dp4a(a, b, c);  // 硬件指令
#else
    // 软件实现（用于不支持 dp4a 的架构）
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
#endif
}
```

## Volta 架构 (SM 70)

### 关键特性

- **Tensor Core**：首次引入，支持 FP16 矩阵乘法加速
- **独立线程调度**：每个线程有独立的程序计数器
- **增强共享内存**：每 SM 128KB

### Tensor Core 基础

```cpp
// Tensor Core 执行 D = A * B + C
// A, B: FP16 矩阵 (m x k, k x n)
// C, D: FP16 或 FP32 矩阵 (m x n)

// WMMA (Warp Matrix Multiply Accumulate)
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

load_matrix_sync(a_frag, a_ptr, 16);
load_matrix_sync(b_frag, b_ptr, 16);
fill_fragment(c_frag, 0.0f);

mma_sync(c_frag, a_frag, b_frag, c_frag);

store_matrix_sync(c_ptr, c_frag, 16, mem_row_major);
```

### FP16 MMA 可用性

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_VOLTA
#define FP16_MMA_AVAILABLE
#endif
```

## Turing 架构 (SM 75)

### 关键特性

- **INT8 Tensor Core**：支持 INT8 矩阵乘法
- **RT Core**：光线追踪（与 ML 无关）
- **增强的共享内存**：每 SM 64KB，支持更灵活配置

### INT8 MMA 可用性

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_TURING
#define INT8_MMA_AVAILABLE
#endif
```

### Flash Attention 支持

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

#if !(defined(GGML_USE_MUSA) && __MUSA_ARCH__ <= CC_QY1)
#define FLASH_ATTN_AVAILABLE
#endif
```

## Ampere 架构 (SM 80, 86)

### 关键特性

- **TF32 Tensor Core**：FP32 精度的 Tensor Core 加速
- **BF16 支持**：Brain Float 16
- **稀疏矩阵支持**：结构化稀疏
- **异步复制**：直接从全局内存到共享内存

### TF32 矩阵乘法

```cpp
// cuBLAS TF32 模式
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
// 然后使用 cublasSgemm，内部会使用 TF32 加速
```

### 硬件归约指令

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

static __device__ __forceinline__ int warp_reduce_sum(int x) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_AMPERE
    // Ampere+ 使用硬件归约指令
    return __reduce_add_sync(0xffffffff, x);
#else
    // 其他架构使用 shuffle
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
#endif
}
```

### 异步复制

```cpp
// Ampere+ 的异步复制
#if __CUDA_ARCH__ >= 800
__pipeline_memcpy_async(shared_mem, global_mem, size);
__pipeline_commit();
__pipeline_wait_prior(0);
#endif
```

## Ada Lovelace 架构 (SM 89)

### 关键特性

- **FP8 支持**：8-bit 浮点运算
- **增强的 Tensor Core**
- **更高带宽**

### FP8 格式

```cpp
// FP8 有两种格式：
// E4M3: 4-bit 指数，3-bit 尾数（更高精度）
// E5M2: 5-bit 指数，2-bit 尾数（更大范围）

// 使用 CUDA 的 FP8 类型（CUDA 11.8+）
#include <cuda_fp8.h>
__nv_fp8_e4m3 fp8_val;
```

## Hopper 架构 (SM 90)

### 关键特性

- **分布式共享内存**：跨 SM 的共享内存访问
- **Tensor Memory Accelerator**
- **FP8 Tensor Core**
- **线程块集群**

### 线程块集群

```cpp
// Hopper 特有的线程块集群
__cluster_dims__(2, 1, 1)  // 2x1x1 的集群
__global__ void cluster_kernel(...) {
    // 集群内的 Block 可以直接通信
}
```

## 特性可用性总结

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

// FP16 可用（Pascal+）
#if (defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ >= CC_PASCAL
#define FP16_AVAILABLE
#endif

// 快速 FP16（Pascal+，排除 610）
#if defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610
#define FAST_FP16_AVAILABLE
#endif

// FP16 MMA（Volta+）
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_VOLTA
#define FP16_MMA_AVAILABLE
#endif

// INT8 MMA（Turing+）
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_TURING
#define INT8_MMA_AVAILABLE
#endif

// Flash Attention
#if !(defined(GGML_USE_MUSA) && __MUSA_ARCH__ <= CC_QY1)
#define FLASH_ATTN_AVAILABLE
#endif
```

## 练习

1. 查看你系统中 GPU 支持哪些特性
2. 阅读 `ggml/src/ggml-cuda/mma.cuh`，理解 Tensor Core 的使用
3. 阅读 `ggml/src/ggml-cuda/fattn-wmma-f16.cuh`，理解 Flash Attention 的 WMMA 实现

## 下一步

完成本节后，请继续学习 [代码适配技术](03-code-adaptation.md)。
