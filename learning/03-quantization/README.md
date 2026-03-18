# 第三阶段：量化技术

## 概述

量化是将高精度浮点数转换为低精度表示的技术，是 LLM 推理优化的核心技术之一。Prima.cpp 支持多种量化格式，可以在保持模型精度的同时大幅减少内存占用和提升推理速度。

## 能力目标

完成本阶段后，你将能够：

### 能做什么

| 能力 | 具体表现 | 相关代码 |
|------|----------|----------|
| **选择量化类型** | 根据需求选择合适的量化格式 | 模型量化 |
| **理解量化原理** | 知道 Q4_0, Q4_K, IQ 系列的区别 | `ggml-quants.h` |
| **使用量化工具** | 使用 llama-quantize 工具量化模型 | 命令行工具 |
| **阅读量化代码** | 理解 CPU 和 CUDA 量化实现 | `ggml-quants.c`, `quantize.cu` |

### 还不能做什么

- 设计新的量化算法
- 优化量化内核性能
- 处理量化精度问题的高级调试

### 实际工作示例

学完本阶段后，你可以：

1. **选择合适的量化类型**
```bash
# 你能根据需求选择量化类型
# 内存充足，追求精度
./llama-quantize model-f16.gguf model-q6_k.gguf Q6_K

# 内存受限，追求速度
./llama-quantize model-f16.gguf model-q4_0.gguf Q4_0

# 平衡选择
./llama-quantize model-f16.gguf model-q4_k.gguf Q4_K
```

2. **理解量化数据结构**
```cpp
// 你能理解这些结构体的设计
typedef struct {
    ggml_fp16_t d;          // 缩放因子
    uint8_t qs[QK4_0/2];    // 量化值
} block_q4_0;
```

3. **估算内存占用**
```cpp
// 你能计算量化后的内存占用
// 7B 模型，Q4_0 量化
// 参数量: 7,000,000,000
// 每参数: 4.5 bits (4 bit 数据 + 0.5 bit 缩放因子)
// 总大小: 7B * 4.5 / 8 ≈ 3.9 GB
```

4. **阅读反量化代码**
```cpp
// 你能理解这段反量化代码
static __device__ void dequantize_q4_0(...) {
    const float d = __half2float(x[ib].d);
    const int vui = x[ib].qs[iqs];
    v.x = d * (int8_t((vui & 0xF) - 8));
    v.y = d * (int8_t((vui >> 4) - 8));
}
```

## 章节目录

1. [量化基础](01-quant-basics.md)
2. [量化类型详解](02-quant-types.md)
3. [量化实现](03-quant-implementation.md)

## 预计学习时间

2 周

## 开始学习

请从 [01-quant-basics.md](01-quant-basics.md) 开始。
