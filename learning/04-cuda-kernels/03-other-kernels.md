# 4.3 其他核心算子

## 概述

除了矩阵乘法和 Flash Attention，Prima.cpp 还实现了许多其他核心算子。本节介绍这些算子的实现原理。

## 核心算子列表

```
ggml/src/ggml-cuda/
├── norm.cu          # 归一化 (RMSNorm, LayerNorm)
├── softmax.cu       # Softmax
├── rope.cu          # 旋转位置编码
├── diagmask.cu      # 因果掩码
├── cpy.cu           # 内存复制
├── scale.cu         # 缩放
├── getrows.cu       # 行获取
├── argsort.cu       # 排序
├── argmax.cu        # 最大值索引
└── ...
```

## RMSNorm

### 数学公式

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
```

### 实现

```cpp
// 文件：ggml/src/ggml-cuda/norm.cu

__global__ void rms_norm_f32(
    const float * __restrict__ x,
    float * __restrict__ dst,
    const int ncols,
    const float eps) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // 1. 计算平方和
    float sum = 0.0f;
    for (int col = tid; col < ncols; col += blockDim.x) {
        sum += x[row * ncols + col] * x[row * ncols + col];
    }

    // 2. Warp 归约求和
    sum = warp_reduce_sum(sum);

    // 3. 计算均方根
    float rms = sqrtf(sum / ncols + eps);

    // 4. 归一化
    const float inv_rms = 1.0f / rms;
    for (int col = tid; col < ncols; col += blockDim.x) {
        dst[row * ncols + col] = x[row * ncols + col] * inv_rms;
    }
}
```

### 优化技巧

```cpp
// 使用向量化加载
float4 val = reinterpret_cast<const float4*>(x)[idx];
sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
```

## Softmax

### 数学公式

```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

### 实现

```cpp
// 文件：ggml/src/ggml-cuda/softmax.cu

__global__ void softmax_f32(
    const float * __restrict__ x,
    float * __restrict__ dst,
    const int ncols) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // 1. 找最大值（数值稳定性）
    float max_val = -INFINITY;
    for (int col = tid; col < ncols; col += blockDim.x) {
        max_val = fmaxf(max_val, x[row * ncols + col]);
    }
    max_val = warp_reduce_max(max_val);

    // 2. 计算 exp 和求和
    float sum_exp = 0.0f;
    for (int col = tid; col < ncols; col += blockDim.x) {
        sum_exp += expf(x[row * ncols + col] - max_val);
    }
    sum_exp = warp_reduce_sum(sum_exp);

    // 3. 归一化
    const float inv_sum = 1.0f / sum_exp;
    for (int col = tid; col < ncols; col += blockDim.x) {
        dst[row * ncols + col] = expf(x[row * ncols + col] - max_val) * inv_sum;
    }
}
```

## RoPE (旋转位置编码)

### 数学公式

```
RoPE(x, pos) = x * [cos(pos * freq), sin(pos * freq)]
```

### 实现

```cpp
// 文件：ggml/src/ggml-cuda/rope.cu

__global__ void rope_f32(
    const float * __restrict__ x,
    float * __restrict__ dst,
    const int ncols,
    const int pos,
    const float freq_base,
    const float freq_scale) {

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (col >= ncols) return;

    // 计算频率
    const float theta = pos * powf(freq_base, -2.0f * (col % (ncols / 2)) / ncols) * freq_scale;

    // 计算旋转
    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);

    const float x0 = x[row * ncols + col];
    const float x1 = x[row * ncols + col + ncols / 2];

    // 应用旋转
    if (col < ncols / 2) {
        dst[row * ncols + col] = x0 * cos_theta - x1 * sin_theta;
        dst[row * ncols + col + ncols / 2] = x0 * sin_theta + x1 * cos_theta;
    }
}
```

## 因果掩码 (Causal Mask)

### 实现

```cpp
// 文件：ggml/src/ggml-cuda/diagmask.cu

__global__ void diag_mask_inf_f32(
    float * __restrict__ x,
    const int ncols,
    const int n_past) {

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (col >= ncols) return;

    // 因果掩码：只允许看到之前的 token
    if (col > n_past + row) {
        x[row * ncols + col] = -INFINITY;
    }
}
```

## 内存复制 (Copy)

### 类型转换复制

```cpp
// 文件：ggml/src/ggml-cuda/cpy.cu

// FP32 -> FP16
__global__ void cpy_f32_f16(const float * src, half * dst, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// FP16 -> FP32
__global__ void cpy_f16_f32(const half * src, float * dst, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}
```

## 行获取 (Get Rows)

### 实现

```cpp
// 文件：ggml/src/ggml-cuda/getrows.cu

// 从嵌入矩阵中获取指定行
__global__ void get_rows_f32(
    const float * __restrict__ x,
    const int * __restrict__ rows,
    float * __restrict__ dst,
    const int ncols,
    const int nrows) {

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_idx = blockIdx.y;

    if (col >= ncols) return;

    const int row = rows[row_idx];
    dst[row_idx * ncols + col] = x[row * ncols + col];
}
```

## 排序 (Argsort)

### 实现

```cpp
// 文件：ggml/src/ggml-cuda/argsort.cu

__global__ void argsort_f32(
    const float * __restrict__ x,
    int * __restrict__ dst,
    const int ncols,
    const int order) {  // 0: 升序, 1: 降序

    const int row = blockIdx.x;

    // 初始化索引
    __shared__ int indices[1024];
    __shared__ float values[1024];

    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        indices[i] = i;
        values[i] = x[row * ncols + i];
    }
    __syncthreads();

    // 简单排序（实际使用更高效的算法）
    for (int i = 0; i < ncols; ++i) {
        for (int j = i + 1; j < ncols; ++j) {
            bool swap = (order == 0) ? (values[i] > values[j]) : (values[i] < values[j]);
            if (swap) {
                float tmp_val = values[i];
                values[i] = values[j];
                values[j] = tmp_val;

                int tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
            }
        }
    }

    // 写回结果
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        dst[row * ncols + i] = indices[i];
    }
}
```

## Argmax

### 实现

```cpp
// 文件：ggml/src/ggml-cuda/argmax.cu

__global__ void argmax_f32(
    const float * __restrict__ x,
    int * __restrict__ dst,
    const int ncols) {

    const int row = blockIdx.x;

    float max_val = -INFINITY;
    int max_idx = 0;

    for (int col = 0; col < ncols; ++col) {
        float val = x[row * ncols + col];
        if (val > max_val) {
            max_val = val;
            max_idx = col;
        }
    }

    dst[row] = max_idx;
}
```

## 算子组合示例

### 完整的 Attention 层

```cpp
// Attention 层的算子组合
void attention_layer(
    const float * Q, const float * K, const float * V,
    float * output, int seq_len, int head_dim, int n_heads) {

    // 1. QK^T (矩阵乘法)
    // mmq_kernel<<<...>>>(Q, K, scores, ...);

    // 2. Scale
    // scale_kernel<<<...>>>(scores, 1.0f / sqrt(head_dim), ...);

    // 3. Causal Mask
    // diag_mask_kernel<<<...>>>(scores, ...);

    // 4. Softmax
    // softmax_kernel<<<...>>>(scores, ...);

    // 5. Score × V (矩阵乘法)
    // mmq_kernel<<<...>>>(scores, V, output, ...);
}
```

## 性能优化总结

### 1. 内存访问优化

```cpp
// 合并访问
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 向量化加载
float4 val = reinterpret_cast<const float4*>(ptr)[idx];

// 共享内存缓存
__shared__ float cache[BLOCK_SIZE];
```

### 2. 计算优化

```cpp
// Warp 归约
float sum = warp_reduce_sum(val);

// 展开循环
#pragma unroll
for (int i = 0; i < 4; ++i) { ... }

// 使用快速数学函数
__expf(x)  // 比 expf(x) 快但精度略低
```

### 3. 架构适配

```cpp
// 根据架构选择最优实现
#if __CUDA_ARCH__ >= CC_AMPERE
    // Ampere+ 优化
#elif __CUDA_ARCH__ >= CC_TURING
    // Turing 优化
#else
    // 通用实现
#endif
```

## 练习

1. 阅读 `ggml/src/ggml-cuda/norm.cu`，理解 RMSNorm 的完整实现
2. 阅读 `ggml/src/ggml-cuda/rope.cu`，理解不同 RoPE 变体的实现
3. 尝试优化一个算子，比较优化前后的性能

## 下一步

完成本阶段后，请继续学习 [第五阶段：CPU SIMD 优化](../05-cpu-simd/README.md)。
