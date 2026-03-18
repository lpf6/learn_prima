# 第一阶段：CUDA 基础

## 概述

CUDA（Compute Unified Device Architecture）是 NVIDIA 开发的并行计算平台和编程模型。在开始阅读 prima.cpp 的 CUDA 代码之前，你需要理解以下核心概念：

1. **线程层次结构**：Grid、Block、Thread、Warp
2. **内存模型**：不同类型的内存及其性能特征
3. **同步机制**：线程间的协调方式

## 能力目标

完成本阶段后，你将能够：

### 能做什么

| 能力 | 具体表现 | 相关代码 |
|------|----------|----------|
| **阅读 CUDA 内核代码** | 理解线程索引计算、内存访问模式 | `ggml-cuda/*.cu` |
| **理解基础优化技术** | 知道为什么使用共享内存、Warp 归约 | `common.cuh` |
| **调试简单 CUDA 问题** | 使用 cuda-memcheck 检查内存错误 | 开发调试 |
| **编写简单内核** | 向量运算、简单归约 | 练习项目 |

### 还不能做什么

- 编写复杂的多内核协作程序
- 进行深度性能优化
- 处理架构特定问题

### 实际工作示例

学完本阶段后，你可以：

1. **理解现有代码**
```cpp
// 你能理解这段代码在做什么
static __device__ __forceinline__ int warp_reduce_sum(int x) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}
```

2. **修改简单参数**
```cpp
// 你能理解为什么选择这些参数
const int BLOCK_SIZE = 256;
int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
```

3. **定位基础错误**
```cpp
// 你能发现这种错误：缺少边界检查
__global__ void kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 缺少: if (idx < n)
    data[idx] = 0;  // 可能越界
}
```

## 章节目录

1. [线程层次结构](01-thread-hierarchy.md)
2. [内存模型](02-memory-model.md)
3. [同步机制](03-synchronization.md)
4. [实践练习](04-practice-exercises.md)

## 预计学习时间

2-3 周

## 开始学习

请从 [01-thread-hierarchy.md](01-thread-hierarchy.md) 开始。
