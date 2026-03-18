# 4.2 Flash Attention

## 概述

Flash Attention 是一种高效的注意力机制实现，通过优化内存访问模式大幅提升性能。本节详细介绍其原理和 Prima.cpp 中的实现。

## 传统注意力的问题

### 标准注意力计算

```
Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V

步骤：
1. 计算 Q × K^T: [batch, heads, seq_len, seq_len]
2. Softmax: [batch, heads, seq_len, seq_len]
3. 乘以 V: [batch, heads, seq_len, d_v]
```

### 内存问题

```
对于 seq_len = 4096, heads = 32, batch = 1:
- 注意力矩阵大小: 1 × 32 × 4096 × 4096 × 4 bytes = 2 GB
- 无法放入 GPU 显存
- 需要多次读写 HBM
```

## Flash Attention 原理

### 核心思想

1. **分块计算**：将注意力矩阵分成小块，逐块计算
2. **在线 Softmax**：增量计算 Softmax，无需存储完整矩阵
3. **内存重用**：在 SRAM 中完成计算，减少 HBM 访问

### 内存访问对比

```
传统注意力：
HBM → SRAM: Q, K, V (完整)
SRAM → HBM: S = QK^T (完整)
HBM → SRAM: S (完整)
SRAM → HBM: P = softmax(S) (完整)
HBM → SRAM: P, V (完整)
SRAM → HBM: O = PV (完整)

Flash Attention：
HBM → SRAM: Q块, K块, V块 (小块)
SRAM 内计算: QK^T, softmax, PV
SRAM → HBM: O块 (小块)
重复直到完成
```

## 在线 Softmax 算法

### 数学推导

```
标准 Softmax:
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

在线更新：
假设已有 m_old, l_old, O_old
新数据 m_new, l_new, O_new

m_new = max(m_old, m_i)
l_new = l_old * exp(m_old - m_new) + l_i * exp(m_i - m_new)
O_new = O_old * (l_old * exp(m_old - m_new) / l_new) + O_i * (l_i / l_new)
```

### 代码实现

```cpp
// 在线 Softmax 更新
struct softmax_state {
    float max_val;     // 当前最大值
    float sum_exp;     // exp 值之和
    float * output;    // 当前输出
};

__device__ void online_softmax_update(
    softmax_state & state,
    float new_max,
    float new_sum_exp,
    float * new_output,
    int dim) {

    if (new_max > state.max_val) {
        // 需要重新缩放
        float scale = expf(state.max_val - new_max);
        state.sum_exp *= scale;
        for (int i = 0; i < dim; ++i) {
            state.output[i] *= scale;
        }
        state.max_val = new_max;
    }

    float scale = expf(new_max - state.max_val);
    state.sum_exp += new_sum_exp * scale;
    for (int i = 0; i < dim; ++i) {
        state.output[i] += new_output[i] * scale;
    }
}
```

## Prima.cpp Flash Attention 实现

### 实现变体

```
ggml/src/ggml-cuda/
├── fattn.cu           # 主入口
├── fattn-common.cuh   # 公共工具
├── fattn-tile-f16.cu  # Tile-based 实现
├── fattn-tile-f32.cu
├── fattn-vec-f16.cuh  # Vectorized 实现
├── fattn-vec-f32.cuh
├── fattn-wmma-f16.cuh # WMMA/Tensor Core 实现
└── template-instances/ # 模板实例化
```

### Tile-based 实现

```cpp
// 文件：ggml/src/ggml-cuda/fattn-tile-f16.cu（简化示例）

template<int D, int ncols, int parallel_blocks>
__global__ void flash_attn_tile_f16(
    const char * __restrict__ Q,
    const char * __restrict__ K,
    const char * __restrict__ V,
    float * __restrict__ O,
    const int ncols_q,
    const int nrows_k,
    const int ncols_k,
    const float scale) {

    // 1. 计算线程索引
    const int col = blockIdx.x * ncols + threadIdx.x;
    const int row_q = blockIdx.y;
    const int head = blockIdx.z;

    // 2. 共享内存分配
    __shared__ half K_tile[TILE_SIZE][D];
    __shared__ half V_tile[TILE_SIZE][D];

    // 3. 加载 Q 到寄存器
    half Q_local[D];
    for (int i = 0; i < D; ++i) {
        Q_local[i] = ((half *)Q)[row_q * ncols_q * D + col * D + i];
    }

    // 4. 初始化 Softmax 状态
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float O_local[D] = {0};

    // 5. 分块计算
    for (int tile_start = 0; tile_start < nrows_k; tile_start += TILE_SIZE) {
        // 加载 K, V 块到共享内存
        // ...（省略加载代码）

        __syncthreads();

        // 计算 QK^T
        for (int k = 0; k < TILE_SIZE; ++k) {
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += (float)Q_local[d] * (float)K_tile[k][d];
            }
            score *= scale;

            // 在线 Softmax 更新
            float new_max = fmaxf(max_val, score);
            float exp_diff = expf(max_val - new_max);
            float exp_score = expf(score - new_max);

            sum_exp = sum_exp * exp_diff + exp_score;

            for (int d = 0; d < D; ++d) {
                O_local[d] = O_local[d] * exp_diff +
                             (float)V_tile[k][d] * exp_score;
            }

            max_val = new_max;
        }

        __syncthreads();
    }

    // 6. 归一化并写回
    for (int d = 0; d < D; ++d) {
        O[row_q * ncols_q * D + col * D + d] = O_local[d] / sum_exp;
    }
}
```

### WMMA 实现（Tensor Core）

```cpp
// 文件：ggml/src/ggml-cuda/fattn-wmma-f16.cuh

#include <mma.h>
using namespace nvcuda::wmma;

template<int D>
__global__ void flash_attn_wmma_f16(...) {
    // 使用 Tensor Core 加速矩阵乘法

    // WMMA 片段
    fragment<matrix_a, 16, 16, 16, half, row_major> Q_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> K_frag;
    fragment<accumulator, 16, 16, 16, half> S_frag;

    // Q × K^T
    load_matrix_sync(Q_frag, Q_ptr, D);
    load_matrix_sync(K_frag, K_ptr, D);
    fill_fragment(S_frag, 0.0f);
    mma_sync(S_frag, Q_frag, K_frag, S_frag);

    // Softmax（需要从 fragment 中提取数据）
    // ...

    // S × V
    fragment<matrix_b, 16, 16, 16, half, row_major> V_frag;
    fragment<accumulator, 16, 16, 16, half> O_frag;

    load_matrix_sync(V_frag, V_ptr, D);
    mma_sync(O_frag, S_frag, V_frag, O_frag);

    // 写回结果
    store_matrix_sync(O_ptr, O_frag, D, mem_row_major);
}
```

### 模板实例化

```cpp
// 文件：ggml/src/ggml-cuda/template-instances/fattn-wmma-f16-instance-*.cu

// 为不同配置实例化模板
template __global__ void flash_attn_wmma_f16<64>(
    const char *, const char *, const char *, float *, int, int, int, float);

template __global__ void flash_attn_wmma_f16<128>(
    const char *, const char *, const char *, float *, int, int, int, float);
```

## 性能优化技术

### 1. 共享内存优化

```cpp
// 避免 Bank Conflict
__shared__ half K_tile[TILE_SIZE][D + 1];  // +1 避免 Bank Conflict
```

### 2. 寄存器使用

```cpp
// Q 和中间结果存储在寄存器
// 减少共享内存访问
half Q_local[D];
float O_local[D];
```

### 3. Warp 级优化

```cpp
// 使用 Warp 归约计算最大值和求和
__device__ float warp_reduce_max(float val) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}
```

### 4. 多查询注意力 (MQA) 优化

```cpp
// MQA/GQA: 多个查询头共享 KV
// 减少内存访问
if (head % n_group == 0) {
    // 只加载一次 KV
}
```

## 性能对比

```
测试配置：A100, seq_len=4096, heads=32, d=128

标准注意力：
- 内存：2 GB
- 时间：15 ms

Flash Attention v1：
- 内存：40 MB
- 时间：3 ms

Flash Attention v2：
- 内存：40 MB
- 时间：2 ms
```

## 调试技巧

### 数值稳定性

```cpp
// 使用 log-sum-exp 技巧
float log_sum_exp = logf(sum_exp);
float logit = score - max_val - log_sum_exp;
```

### 边界检查

```cpp
// 确保索引在有效范围内
if (row_k >= nrows_k) continue;
```

## 练习

1. 理解在线 Softmax 的数学推导
2. 阅读 `ggml/src/ggml-cuda/fattn-tile-f16.cu`，理解分块计算流程
3. 阅读 `ggml/src/ggml-cuda/fattn-wmma-f16.cuh`，理解 Tensor Core 的使用

## 下一步

完成本节后，请继续学习 [其他核心算子](03-other-kernels.md)。
