# 3.2 量化类型详解

## 概述

Prima.cpp 支持多种量化类型，每种都有其特点和适用场景。本节详细介绍各种量化类型的数据结构和特性。

## 量化类型分类

```
量化类型
├── 基础量化
│   ├── Q4_0: 4-bit 对称量化
│   ├── Q4_1: 4-bit 非对称量化
│   ├── Q5_0: 5-bit 对称量化
│   ├── Q5_1: 5-bit 非对称量化
│   └── Q8_0: 8-bit 对称量化
│
├── K-quant 系列
│   ├── Q2_K: 2-bit 量化
│   ├── Q3_K: 3-bit 量化
│   ├── Q4_K: 4-bit 优化量化
│   ├── Q5_K: 5-bit 优化量化
│   └── Q6_K: 6-bit 量化
│
└── IQ 系列 (Importance-based)
    ├── IQ1_S: 1-bit 量化
    ├── IQ2_XXS/XS/S: 2-bit 变体
    ├── IQ3_XXS/S: 3-bit 变体
    └── IQ4_NL/XS: 4-bit 变体
```

## 基础量化类型

### Q4_0

```cpp
// 文件：ggml/src/ggml-common.h

#define QK4_0 32
#define QR4_0 2

typedef struct {
    ggml_fp16_t d;          // 缩放因子 (FP16)
    uint8_t qs[QK4_0/2];    // 量化值 (32 个 4-bit 值打包成 16 字节)
} block_q4_0;

// 每个 4-bit 值表示 [-8, 7] 范围的有符号整数
// 反量化: x = (q - 8) * d
```

**特点**：
- 块大小：32
- 每元素：4 bits + 16/32 bits scale ≈ 4.5 bits
- 对称量化，无零点

### Q4_1

```cpp
#define QK4_1 32
#define QR4_1 2

typedef struct {
    ggml_fp16_t d;          // 缩放因子
    ggml_fp16_t m;          // 最小值 (零点)
    uint8_t qs[QK4_0/2];    // 量化值
} block_q4_1;

// 反量化: x = q * d + m
```

**特点**：
- 块大小：32
- 每元素：4 bits + 32/32 bits (scale + min) ≈ 5 bits
- 非对称量化，有零点
- 精度比 Q4_0 略高

### Q8_0

```cpp
#define QK8_0 32
#define QR8_0 1

typedef struct {
    ggml_fp16_t d;          // 缩放因子
    int8_t qs[QK8_0];       // 量化值 (32 个 8-bit 有符号整数)
} block_q8_0;

// 反量化: x = q * d
```

**特点**：
- 块大小：32
- 每元素：8 bits + 16/32 bits scale ≈ 8.5 bits
- 精度最高，内存节省最少

## K-quant 系列

K-quant 系列使用更复杂的量化策略，在相同位数下获得更高精度。

### Q4_K

```cpp
#define QK_K 256

typedef struct {
    uint8_t scales[QK_K/16];       // 16 个缩放因子 (每个 4-bit)
    uint8_t qs[QK_K/2];            // 256 个 4-bit 量化值
    ggml_fp16_t d;                  // 全局缩放因子
    ggml_fp16_t dmin;               // 最小值缩放因子
} block_q4_k;

// 分层缩放：
// - 1 个全局缩放因子 d
// - 16 个局部缩放因子 (每个控制 16 个元素)
```

**特点**：
- 块大小：256
- 分层缩放，精度更高
- 每元素约 4.5 bits

### Q6_K

```cpp
typedef struct {
    uint8_t ql[QK_K/2];            // 低 4-bit
    uint8_t qh[QK_K/4];            // 高 2-bit
    int8_t scales[QK_K/16];        // 16 个缩放因子
    ggml_fp16_t d;                  // 全局缩放因子
} block_q6_k;

// 6-bit 量化：ql 存储 bits 0-3，qh 存储 bits 4-5
```

**特点**：
- 块大小：256
- 6-bit 精度
- 每元素约 6.5 bits

## IQ 系列

IQ 系列使用重要性加权和优化的编码方式。

### IQ4_NL

```cpp
#define QK4_NL 32

typedef struct {
    ggml_fp16_t d;
    uint8_t qs[QK4_NL/2];
} block_iq4_nl;

// 使用非线性量化表
static constexpr int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};
```

**特点**：
- 非线性量化值分布
- 更好地表示小数值
- 适合激活值量化

### IQ2_XXS

```cpp
typedef struct {
    uint16_t qs[QK_K/8];    // 2-bit 索引
    ggml_fp16_t d;          // 缩放因子
} block_iq2_xxs;

// 使用查找表将 2-bit 索引映射到量化值
```

**特点**：
- 极低比特率
- 使用优化的查找表
- 适合极端压缩场景

## 量化类型对比

| 类型 | 每元素位数 | 相对精度 | 相对速度 | 推荐场景 |
|------|-----------|----------|----------|----------|
| Q8_0 | 8.5 | 100% | 1.5x | 高精度需求 |
| Q6_K | 6.5 | 98% | 2x | 平衡选择 |
| Q5_1 | 5.5 | 96% | 2.5x | 通用场景 |
| Q5_0 | 5.5 | 95% | 2.5x | 通用场景 |
| Q4_K | 4.5 | 94% | 3x | 推荐默认 |
| Q4_1 | 5.0 | 93% | 3x | 需要零点 |
| Q4_0 | 4.5 | 92% | 3x | 内存优先 |
| Q3_K | 3.5 | 88% | 3.5x | 内存受限 |
| Q2_K | 2.5 | 80% | 4x | 极端压缩 |
| IQ4_NL | 4.5 | 95% | 3x | 激活量化 |
| IQ2_XXS | 2.0 | 75% | 4x | 实验 |

## 选择建议

### 按内存选择

```
内存充足 (>16GB): Q6_K 或 Q8_0
内存适中 (8-16GB): Q4_K 或 Q5_K
内存受限 (<8GB):  Q3_K 或 Q4_0
极端受限:         Q2_K 或 IQ 系列
```

### 按精度需求选择

```
高精度需求: Q8_0 或 Q6_K
平衡需求:   Q4_K 或 Q5_K
速度优先:   Q4_0 或 Q4_1
```

## 量化类型在代码中的使用

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0;  // 块大小
    static constexpr int qr = QR4_0;  // 反量化比率
    static constexpr int qi = QI4_0;  // 每块整数数
};

// 使用示例
template<typename T>
__global__ void dequantize_kernel(const void * vx, float * y, int k) {
    const int i = blockIdx.x;
    const T * x = (const T *) vx;

    // 每个块处理一个量化块
    const int qk = ggml_cuda_type_traits<T>::qk;

    for (int j = threadIdx.x; j < qk; j += blockDim.x) {
        y[i * qk + j] = dequantize(x[i], j);
    }
}
```

## 练习

1. 计算不同量化类型下 7B 模型的内存占用
2. 阅读 `ggml/src/ggml-quants.h`，理解所有量化结构体
3. 比较同一模型使用不同量化类型的推理效果

## 下一步

完成本节后，请继续学习 [量化实现](03-quant-implementation.md)。
