# 4.1 矩阵乘法 (MMQ)

## 概述

矩阵乘法是 LLM 推理中最核心的计算，占据 80% 以上的计算时间。Prima.cpp 实现了高效的量化矩阵乘法内核。

## 文件位置

```
ggml/src/ggml-cuda/
├── mmq.cu          # 矩阵乘法内核实现
├── mmq.cuh         # 头文件
└── mmvq.cu         # 矩阵-向量乘法
```

## 问题定义

### 标准 GEMM

```
C = A × B

A: M × K 矩阵
B: K × N 矩阵
C: M × N 矩阵

计算量: 2 × M × N × K FLOPs
```

### 量化 GEMM

```
C = dequant(A_q) × dequant(B_q)

A_q: 量化矩阵
B_q: 量化矩阵（或非量化）
```

## 基本优化策略

### 1. 分块计算 (Tiling)

```
将大矩阵分成小块，利用共享内存复用数据

┌─────────────────────────────────────┐
│              B 矩阵                  │
│  ┌─────┬─────┬─────┬─────┐         │
│  │ B00 │ B01 │ B02 │ B03 │         │
│  ├─────┼─────┼─────┼─────┤         │
│  │ B10 │ B11 │ B12 │ B13 │         │
│  └─────┴─────┴─────┴─────┘         │
└─────────────────────────────────────┘
        ×
┌─────────────────┐
│    A 矩阵        │
│  ┌─────┬─────┐  │
│  │ A00 │ A01 │  │
│  ├─────┼─────┤  │
│  │ A10 │ A11 │  │
│  └─────┴─────┘  │
└─────────────────┘
```

### 2. 共享内存使用

```cpp
// 每个线程块加载一个 tile 到共享内存
__shared__ float tile_A[TILE_M][TILE_K];
__shared__ float tile_B[TILE_K][TILE_N];

// 加载阶段
tile_A[ty][tx] = A[row * K + tile_k * TILE_K + tx];
tile_B[ty][tx] = B[(tile_k * TILE_K + ty) * N + col];

__syncthreads();

// 计算阶段
for (int k = 0; k < TILE_K; ++k) {
    sum += tile_A[ty][k] * tile_B[k][tx];
}
```

### 3. 向量化内存访问

```cpp
// 使用 float4 一次加载 4 个浮点数
float4 val = reinterpret_cast<float4*>(ptr)[idx];
```

## Prima.cpp MMQ 实现

### 核心数据结构

```cpp
// 文件：ggml/src/ggml-cuda/mmq.cu

// 线程块配置
#define MMQ_DP4A_MAX_BATCH_SIZE  64
#define MMQ_DP4A_MAX_ROWS_PER_BLOCK 128

// 共享内存布局
struct mmq_args {
    const void * x;       // 量化矩阵 A
    const void * y;       // 量化矩阵 B
    float * dst;          // 输出矩阵 C
    int ncols_x;          // A 的列数
    int nrows_x;          // A 的行数
    int ncols_y;          // B 的列数
};
```

### Q4_0 矩阵乘法内核

```cpp
// 文件：ggml/src/ggml-cuda/mmq.cu（简化示例）

template<int ncols_y>
__global__ void mul_mat_q4_0(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int nrows_y,
    const int nrows_dst) {

    // 1. 计算线程索引
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int col = blockIdx.y * ncols_y + threadIdx.x;

    if (row >= nrows_dst) return;

    // 2. 获取量化块指针
    const block_q4_0 * x = (const block_q4_0 *) vx;
    const block_q8_0 * y = (const block_q8_0 *) vy;

    // 3. 共享内存用于缓存
    __shared__ float block_x[MMQ_DP4A_MAX_ROWS_PER_BLOCK][QK4_0];

    // 4. 加载并反量化 A 矩阵块
    const int blocks_per_row_x = ncols_x / QK4_0;
    const int block_idx = row * blocks_per_row_x;

    // 5. 计算点积
    float sum = 0.0f;

    for (int k = 0; k < blocks_per_row_x; ++k) {
        // 反量化并计算
        const float d_x = __half2float(x[block_idx + k].d);
        const float d_y = __half2float(y[k * nrows_y + col / QK8_0].d);

        // 使用 dp4a 指令加速
        const uint8_t * qs_x = x[block_idx + k].qs;
        const int8_t * qs_y = y[k * nrows_y + col / QK8_0].qs;

        int sum_i = 0;
        for (int i = 0; i < QK4_0 / 2; ++i) {
            const int v_x = (int) qs_x[i];
            const int v_x_lo = (v_x & 0x0F) - 8;
            const int v_x_hi = (v_x >> 4) - 8;

            sum_i = ggml_cuda_dp4a(v_x_lo, *(int *)(qs_y + i * 8), sum_i);
            sum_i = ggml_cuda_dp4a(v_x_hi, *(int *)(qs_y + i * 8 + 4), sum_i);
        }

        sum += d_x * d_y * sum_i;
    }

    // 6. 写回结果
    dst[row * nrows_y + col] = sum;
}
```

### 分块策略详解

```cpp
// 文件：ggml/src/ggml-cuda/mmq.cu

// 分块大小根据架构调整
static int mmq_get_m_block_size(int cc) {
    if (cc >= CC_AMPERE) {
        return 128;  // Ampere+ 使用更大块
    } else if (cc >= CC_TURING) {
        return 64;
    } else {
        return 32;
    }
}

// 启动配置
void launch_mmq(ggml_backend_cuda_context & ctx, ...) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;

    const int m_block = mmq_get_m_block_size(cc);
    const int n_block = 8;  // 每个线程处理 8 列

    dim3 block_dim(32, m_block / 32);
    dim3 grid_dim((nrows_x + m_block - 1) / m_block,
                  (ncols_y + n_block - 1) / n_block);

    mul_mat_q4_0<n_block><<<grid_dim, block_dim, 0, stream>>>(...);
}
```

## 优化技术详解

### 1. Warp 级优化

```cpp
// 使用 Warp 级原语减少同步开销
__device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}
```

### 2. 内存访问优化

```cpp
// 合并访问模式
// 好：相邻线程访问相邻地址
int idx = threadIdx.x;
float val = data[idx];

// 坏：跨步访问
float val = data[idx * stride];

// 使用向量化加载
float4 vals = reinterpret_cast<float4*>(data)[idx];
```

### 3. 寄存器使用优化

```cpp
// 展开循环，使用寄存器存储中间结果
#pragma unroll
for (int k = 0; k < TILE_K; ++k) {
    sum0 += tile_A[ty][k] * tile_B[k][tx * 4 + 0];
    sum1 += tile_A[ty][k] * tile_B[k][tx * 4 + 1];
    sum2 += tile_A[ty][k] * tile_B[k][tx * 4 + 2];
    sum3 += tile_A[ty][k] * tile_B[k][tx * 4 + 3];
}
```

### 4. Tensor Core 使用

```cpp
// 文件：ggml/src/ggml-cuda/mma.cuh

// 使用 WMMA 接口
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

// 加载、计算、存储
load_matrix_sync(a_frag, a_ptr, 16);
load_matrix_sync(b_frag, b_ptr, 16);
fill_fragment(c_frag, 0.0f);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(c_ptr, c_frag, 16, mem_row_major);
```

## 性能分析

### 理论性能

```
RTX 3090 规格：
- FP32 性能: 35.6 TFLOPS
- INT8 性能: 142 TFLOPS (使用 Tensor Core)
- 内存带宽: 936 GB/s

矩阵乘法 (M=1024, N=1024, K=1024):
- 计算量: 2.1 GFLOPS
- 数据量: 12 MB
- 理论时间: 0.06 ms (INT8 Tensor Core)
```

### 实际性能因素

1. **内存带宽**：量化减少数据传输
2. **计算密度**：矩阵大小影响计算/访存比
3. **占用率**：SM 利用率
4. **延迟隐藏**：内存延迟与计算重叠

## 调试技巧

### 使用 Nsight Compute

```bash
# 分析内核性能
ncu --set full -o profile ./llama-cli -m model.gguf -p "test"

# 关键指标
# - 内存吞吐量
# - 计算吞吐量
# - Warp 执行效率
# - 占用率
```

### 性能计数器

```cpp
// 使用 CUDA 事件测量时间
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
launch_mmq(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("MMQ time: %.3f ms\n", ms);
```

## 练习

1. 阅读 `ggml/src/ggml-cuda/mmq.cu` 的完整实现
2. 比较不同量化类型的矩阵乘法性能
3. 使用 Nsight Compute 分析内核瓶颈

## 下一步

完成本节后，请继续学习 [Flash Attention](02-flash-attention.md)。
