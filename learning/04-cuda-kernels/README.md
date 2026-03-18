# 第四阶段：核心 CUDA 内核

## 概述

本阶段深入 Prima.cpp 中最重要的 CUDA 内核实现，包括矩阵乘法、Flash Attention 等核心算子。

## 能力目标

完成本阶段后，你将能够：

### 能做什么

| 能力 | 具体表现 | 相关代码 |
|------|----------|----------|
| **阅读复杂内核** | 理解 MMQ、Flash Attention 实现 | `mmq.cu`, `fattn.cu` |
| **修改内核参数** | 调整分块大小、优化配置 | 内核调优 |
| **添加简单算子** | 实现新的 CUDA 内核 | 开发扩展 |
| **性能分析** | 使用 Nsight 分析内核性能 | 性能优化 |
| **修复内核 Bug** | 定位和修复内核错误 | 维护工作 |

### 还不能做什么

- 从零设计复杂的多内核系统
- 进行极致性能优化
- 实现新的 Flash Attention 变体

### 实际工作示例

学完本阶段后，你可以：

1. **理解矩阵乘法内核**
```cpp
// 你能理解这段代码的优化思路
template<int ncols_y>
__global__ void mul_mat_q4_0(...) {
    // 分块策略
    // 共享内存使用
    // dp4a 指令加速
    // ...
}
```

2. **修改内核配置**
```cpp
// 你能根据硬件调整这些参数
static int mmq_get_m_block_size(int cc) {
    if (cc >= CC_AMPERE) return 128;
    if (cc >= CC_TURING) return 64;
    return 32;
}
```

3. **添加新的算子内核**
```cpp
// 你能实现类似这样的新算子
__global__ void my_custom_op_kernel(
    const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = custom_operation(input[idx]);
    }
}
```

4. **性能分析和优化**
```bash
# 你能使用这些工具分析性能
ncu --set full ./llama-cli -m model.gguf -p "test"

# 分析关键指标
# - 内存吞吐量
# - 计算吞吐量
# - Warp 执行效率
```

5. **理解 Flash Attention**
```cpp
// 你能理解在线 Softmax 的数学原理
// 和分块计算的实现方式
for (int tile_start = 0; tile_start < nrows_k; tile_start += TILE_SIZE) {
    // 加载 K, V 块
    // 计算 QK^T
    // 在线 Softmax 更新
    // ...
}
```

## 章节目录

1. [矩阵乘法 (MMQ)](01-mmq.md)
2. [Flash Attention](02-flash-attention.md)
3. [其他核心算子](03-other-kernels.md)

## 预计学习时间

3-4 周

## 开始学习

请从 [01-mmq.md](01-mmq.md) 开始。
