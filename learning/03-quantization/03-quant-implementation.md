# 3.3 量化实现

## 概述

本节深入 Prima.cpp 中量化的具体实现，包括 CPU 和 CUDA 两个版本。

## 量化实现位置

```
ggml/src/
├── ggml-quants.c          # CPU 量化实现
├── ggml-quants.h          # 量化类型定义
└── ggml-cuda/
    ├── quantize.cu        # CUDA 量化内核
    ├── quantize.cuh
    └── dequantize.cuh     # 反量化内核
```

## CPU 量化实现

### Q4_0 量化

```cpp
// 文件：ggml/src/ggml-quants.c（简化示例）

void quantize_row_q4_0(const float * restrict src, void * restrict dst, int n) {
    block_q4_0 * restrict y = (block_q4_0 *) dst;

    for (int i = 0; i < n; i += QK4_0) {
        // 1. 找到块内最大绝对值
        float amax = 0.0f;
        for (int j = 0; j < QK4_0; ++j) {
            amax = fmaxf(amax, fabsf(src[i + j]));
        }

        // 2. 计算缩放因子
        const float d = amax / 7.0f;  // 4-bit 有符号范围 [-8, 7]
        y[i / QK4_0].d = GGML_FP32_TO_FP16(d);

        // 3. 量化每个值
        for (int j = 0; j < QK4_0; j += 2) {
            // 量化两个值并打包到一个字节
            int v0 = (int) roundf(src[i + j] / d) + 8;
            int v1 = (int) roundf(src[i + j + 1] / d) + 8;

            v0 = fminf(fmaxf(v0, 0), 15);
            v1 = fminf(fmaxf(v1, 0), 15);

            y[i / QK4_0].qs[j / 2] = v0 | (v1 << 4);
        }
    }
}
```

### Q4_0 反量化

```cpp
void dequantize_row_q4_0(const block_q4_0 * restrict x, float * restrict y, int n) {
    for (int i = 0; i < n; i += QK4_0) {
        const block_q4_0 * restrict block = &x[i / QK4_0];
        const float d = GGML_FP16_TO_FP32(block->d);

        for (int j = 0; j < QK4_0; ++j) {
            // 解包 4-bit 值
            const int q = (block->qs[j / 2] >> (4 * (j % 2))) & 0x0F;
            // 转换为有符号整数并反量化
            y[i + j] = d * (q - 8);
        }
    }
}
```

## CUDA 量化实现

### 量化内核

```cpp
// 文件：ggml/src/ggml-cuda/quantize.cu（简化示例）

template<typename T>
__global__ void quantize_kernel(const float * src, void * dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n / T::block_size) return;

    T * dst_block = (T *) dst + i;
    const float * src_block = src + i * T::block_size;

    // 1. 找最大值
    float amax = 0.0f;
    for (int j = 0; j < T::block_size; ++j) {
        amax = fmaxf(amax, fabsf(src_block[j]));
    }

    // 2. 计算缩放因子
    float scale = amax / T::max_value;
    dst_block->d = __float2half(scale);

    // 3. 量化
    for (int j = 0; j < T::block_size; ++j) {
        int q = (int) roundf(src_block[j] / scale);
        // 打包存储...
    }
}
```

### 反量化内核

```cpp
// 文件：ggml/src/ggml-cuda/dequantize.cuh

// Q4_0 反量化
static __device__ __forceinline__ void dequantize_q4_0(
    const void * vx, const int64_t ib, const int iqs, dfloat2 & v) {

    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = __half2float(x[ib].d);

    const int vui = x[ib].qs[iqs];

    v.x = d * (int8_t((vui & 0xF) - 8));
    v.y = d * (int8_t((vui >> 4) - 8));
}
```

## 向量点积（量化矩阵乘法核心）

### 使用 dp4a 的实现

```cpp
// 文件：ggml/src/ggml-cuda/vecdotq.cuh

static __device__ __forceinline__ float vec_dot_q4_0_q8_0(
    const void * __restrict__ vbq, const block_q8_0 * __restrict__ bq8_0, const int iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq;

    float sumf = 0.0f;

    // 获取缩放因子
    const float d = __half2float(bq4_0->d) * __half2float(bq8_0->d);

    // 使用 dp4a 指令计算点积
    for (int i = 0; i < QK4_0 / 8; ++i) {
        const int v0 = bq4_0->qs[iqs + i];
        const int v1 = bq8_0->qs[iqs + i];

        // 将 4-bit 值展开为 8-bit
        int v0_lo = (v0 & 0x0F) - 8;
        int v0_hi = (v0 >> 4) - 8;

        // 使用 dp4a 计算点积
        sumf += d * (ggml_cuda_dp4a(v0_lo, v1, 0) +
                     ggml_cuda_dp4a(v0_hi, v1 >> 32, 0));
    }

    return sumf;
}
```

## 矩阵乘法中的量化处理

### MMQ (Matrix-Matrix Quantized)

```cpp
// 文件：ggml/src/ggml-cuda/mmq.cu（简化示例）

template<typename T>
__global__ void mmq_kernel(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y) {

    // 加载量化矩阵块到共享内存
    __shared__ float block_x[TILE_SIZE][TILE_SIZE];
    __shared__ float block_y[TILE_SIZE][TILE_SIZE];

    // 反量化并计算
    for (int tile = 0; tile < ncols_x / TILE_SIZE; ++tile) {
        // 加载并反量化 x 块
        load_and_dequantize<T>(vx, block_x, tile);

        // 加载并反量化 y 块
        load_and_dequantize<T>(vy, block_y, tile);

        __syncthreads();

        // 计算点积
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += block_x[threadIdx.y][k] * block_y[k][threadIdx.x];
        }

        __syncthreads();
    }

    // 写回结果
    dst[row * ncols_y + col] = sum;
}
```

## 量化工具使用

### 使用 quantize 工具

```bash
# 编译项目
cmake -B build
cmake --build build

# 量化模型
./build/bin/llama-quantize \
    ./models/model-f16.gguf \
    ./models/model-q4_0.gguf \
    Q4_0

# 支持的量化类型
# Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
# Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
# IQ4_NL, IQ3_S, IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S
```

### 量化统计

```bash
# 查看量化统计信息
./build/bin/llama-quantize-stats ./models/model-f16.gguf ./models/model-q4_0.gguf
```

## 量化精度优化技术

### 1. 校准数据选择

```python
# convert_hf_to_gguf.py 中的校准
def quantize_with_calibration(model, calibration_data):
    # 使用代表性数据计算量化参数
    for layer in model.layers:
        activations = run_calibration(layer, calibration_data)
        layer.quant_params = compute_optimal_params(activations)
```

### 2. 重要性加权

```cpp
// IQ 系列使用重要性加权
// 对重要的权重使用更精细的量化
float importance = compute_importance(weight);
int quant_bits = select_quant_bits(importance);
```

### 3. 混合精度

```cpp
// 不同层使用不同精度
enum ggml_type get_optimal_type(const char * name) {
    if (strstr(name, "output")) return GGML_TYPE_F16;  // 输出层保持高精度
    if (strstr(name, "embed")) return GGML_TYPE_Q4_K; // 嵌入层中等精度
    return GGML_TYPE_Q4_0;                             // 其他层低精度
}
```

## 性能对比

### 不同量化类型的推理速度

```
测试环境：RTX 3090, 7B 模型, batch_size=1

FP16:    45 tokens/s  (基准)
Q8_0:    65 tokens/s  (1.4x)
Q6_K:    80 tokens/s  (1.8x)
Q5_K:    95 tokens/s  (2.1x)
Q4_K:    110 tokens/s (2.4x)
Q4_0:    120 tokens/s (2.7x)
Q3_K:    130 tokens/s (2.9x)
```

### 内存占用

```
7B 模型:

FP16:    14 GB
Q8_0:    7.5 GB
Q6_K:    6.0 GB
Q5_K:    5.0 GB
Q4_K:    4.0 GB
Q4_0:    3.8 GB
Q3_K:    3.2 GB
Q2_K:    2.5 GB
```

## 练习

1. 使用 quantize 工具量化一个模型，比较不同量化类型的效果
2. 阅读 `ggml/src/ggml-cuda/vecdotq.cuh`，理解各种量化类型的点积实现
3. 阅读 `ggml/src/ggml-cuda/mmq.cu`，理解量化矩阵乘法的实现

## 下一步

完成本阶段后，请继续学习 [第四阶段：核心 CUDA 内核](../04-cuda-kernels/README.md)。
