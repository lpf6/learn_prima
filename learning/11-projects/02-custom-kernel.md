# 项目 2：自定义内核开发

## 项目概述

开发一个自定义的 CUDA 内核，从需求分析到性能优化，完整体验内核开发流程。

## 项目目标

- 理解内核开发全流程
- 掌握性能分析和优化技巧
- 学会内核集成和测试

## 项目选择

选择一个内核进行开发：

### 选项 1：LayerNorm 内核

```cpp
// 目标：实现高效的 LayerNorm 内核
// 输入：[batch, seq_len, hidden]
// 输出：[batch, seq_len, hidden]

__global__ void layer_norm_kernel(
    const float *input,
    float *output,
    const float *weight,
    const float *bias,
    int batch,
    int seq_len,
    int hidden,
    float eps);
```

### 选项 2：RoPE 内核

```cpp
// 目标：实现旋转位置编码内核
// 输入：Q, K [batch, heads, seq_len, head_dim]
// 输出：RoPE(Q), RoPE(K)

__global__ void rope_kernel(
    half *Q,
    half *K,
    const float *freqs,
    int batch,
    int heads,
    int seq_len,
    int head_dim);
```

### 选项 3：SwiGLU 内核

```cpp
// 目标：实现 SwiGLU 激活函数内核
// 输入：X [batch, seq_len, hidden]
// 输出：SwiGLU(X) = SiLU(gate) * up

__global__ void swiglu_kernel(
    const float *gate,
    const float *up,
    float *output,
    int n);
```

## 开发流程

### 步骤 1：需求分析

```
1. 明确输入输出
   - 张量形状
   - 数据类型（FP16/FP32）
   - 内存布局

2. 确定性能目标
   - 目标吞吐量
   - 延迟要求
   - 显存限制

3. 分析计算特性
   - 计算强度（FLOPs/Byte）
   - 内存访问模式
   - 并行度
```

### 步骤 2：内核设计

```cpp
// 以 LayerNorm 为例

// 1. 线程映射
// 每个 block 处理一个 token
// 每个 thread 处理 hidden 维度的一部分

// 2. 内存访问
// - 全局内存：输入输出
// - 共享内存：归约计算
// - 寄存器：中间结果

// 3. 同步策略
// - __syncthreads() 用于归约
// - 原子操作用于最终累加
```

### 步骤 3：初始实现

```cpp
// LayerNorm 初始版本
__global__ void layer_norm_v1(
    const float *input,
    float *output,
    const float *weight,
    const float *bias,
    int n,
    int hidden,
    float eps) {
    
    int row = blockIdx.x;
    if (row >= n) return;
    
    const float *x = input + row * hidden;
    float *y = output + row * hidden;
    
    // 1. 计算均值
    float sum = 0.0f;
    for (int i = 0; i < hidden; i++) {
        sum += x[i];
    }
    float mean = sum / hidden;
    
    // 2. 计算方差
    float var_sum = 0.0f;
    for (int i = 0; i < hidden; i++) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / hidden;
    
    // 3. 归一化
    float inv_std = rsqrtf(var + eps);
    for (int i = 0; i < hidden; i++) {
        y[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}
```

### 步骤 4：性能优化

```cpp
// 优化版本 1：使用共享内存
__global__ void layer_norm_v2(
    const float *input,
    float *output,
    const float *weight,
    const float *bias,
    int n,
    int hidden,
    float eps) {
    
    int row = blockIdx.x;
    if (row >= n) return;
    
    const float *x = input + row * hidden;
    float *y = output + row * hidden;
    
    extern __shared__ float shared[];
    float *s_mean = shared;
    float *s_var = shared + 1;
    
    int tid = threadIdx.x;
    
    // 1. 并行归约计算均值
    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        local_sum += x[i];
    }
    s_mean[tid] = local_sum;
    __syncthreads();
    
    // 归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mean[tid] += s_mean[tid + stride];
        }
        __syncthreads();
    }
    float mean = s_mean[0] / hidden;
    
    // 2. 并行计算方差
    float local_var = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float diff = x[i] - mean;
        local_var += diff * diff;
    }
    s_var[tid] = local_var;
    __syncthreads();
    
    // 归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_var[tid] += s_var[tid + stride];
        }
        __syncthreads();
    }
    float var = s_var[0] / hidden;
    
    // 3. 归一化（并行）
    float inv_std = rsqrtf(var + eps);
    for (int i = tid; i < hidden; i += blockDim.x) {
        y[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}
```

### 步骤 5：性能测试

```cpp
// 基准测试
void benchmark_layer_norm() {
    int batch = 32;
    int hidden = 4096;
    
    // 分配内存
    float *d_input, *d_output;
    float *d_weight, *d_bias;
    cudaMalloc(&d_input, batch * hidden * sizeof(float));
    cudaMalloc(&d_output, batch * hidden * sizeof(float));
    cudaMalloc(&d_weight, hidden * sizeof(float));
    cudaMalloc(&d_bias, hidden * sizeof(float));
    
    // 初始化
    // ...
    
    // 测试不同版本
    const char *versions[] = {"v1", "v2"};
    
    for (auto version : versions) {
        // 预热
        for (int i = 0; i < 10; i++) {
            layer_norm<<<batch, 256>>>(...);
        }
        cudaDeviceSynchronize();
        
        // 正式测试
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            layer_norm<<<batch, 256>>>(...);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        
        printf("LayerNorm %s: %.2f ms\n", version, elapsed_ms / 100);
    }
}
```

### 步骤 6：集成测试

```cpp
// 单元测试
TEST(LayerNorm, Correctness) {
    int batch = 2;
    int hidden = 128;
    
    // 准备输入
    std::vector<float> h_input(batch * hidden);
    // ... 初始化
    
    // CPU 参考实现
    std::vector<float> h_ref(batch * hidden);
    layer_norm_cpu(h_input.data(), h_ref.data(), ...);
    
    // GPU 实现
    std::vector<float> h_output(batch * hidden);
    layer_norm_gpu(h_input.data(), h_output.data(), ...);
    
    // 比较结果
    float max_diff = 0.0f;
    for (int i = 0; i < batch * hidden; i++) {
        float diff = fabs(h_ref[i] - h_output[i]);
        max_diff = fmax(max_diff, diff);
    }
    
    ASSERT_LT(max_diff, 1e-5);
}
```

## 性能优化技巧

### 1. 减少内存访问

```cpp
// 使用向量化加载
__global__ void optimized_load(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 使用 float4 向量化加载（128-bit）
    float4 val = reinterpret_cast<float4*>(input)[idx];
    reinterpret_cast<float4*>(output)[idx] = val;
}
```

### 2. 使用共享内存

```cpp
// 减少全局内存访问
__shared__ float shared_data[256];

// 加载到共享内存
shared_data[threadIdx.x] = global_data[idx];
__syncthreads();

// 使用共享内存计算
float val = shared_data[threadIdx.x];
```

### 3. 隐藏延迟

```cpp
// 增加 occupancy
// 使用更多寄存器
// 使用更多线程块

// 异步内存拷贝
cudaMemcpyAsync(...);
kernel<<<...>>>(...);  // 与拷贝重叠执行
```

### 4. 使用 Tensor Core

```cpp
// 使用 WMMA API
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::load_matrix_sync(a_frag, a, lda);
wmma::load_matrix_sync(b_frag, b, ldb);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(c, c_frag, ldc, wmma::mem_row_major);
```

## 评估标准

| 标准 | 权重 | 说明 |
|------|------|------|
| **功能正确性** | 30% | 结果与参考实现一致 |
| **性能提升** | 30% | 相比朴素实现有显著提升 |
| **代码质量** | 20% | 代码规范、可读 |
| **优化技巧** | 10% | 使用多种优化技术 |
| **文档完整** | 10% | 清晰的实现文档 |

## 参考资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices/)
- [cuBLAS Source](https://github.com/NVIDIA/cuBLAS)
- [Cutlass Library](https://github.com/NVIDIA/cutlass)
