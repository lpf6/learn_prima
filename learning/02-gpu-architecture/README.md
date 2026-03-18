# 第二阶段：GPU 架构与计算能力

## 概述

NVIDIA GPU 有不同的架构世代，每个世代都有特定的计算能力（Compute Capability）。理解这些差异对于编写兼容不同 GPU 的代码至关重要。

## 能力目标

完成本阶段后，你将能够：

### 能做什么

| 能力 | 具体表现 | 相关代码 |
|------|----------|----------|
| **理解架构差异** | 知道 SM 75/80/86/89/90 的特性区别 | `common.cuh` |
| **阅读架构适配代码** | 理解条件编译和运行时检测 | `ggml-cuda.cu` |
| **修改架构支持** | 添加新架构的编译支持 | `CMakeLists.txt` |
| **诊断架构问题** | 解决"no device code"错误 | 调试排错 |

### 还不能做什么

- 编写深度架构优化代码
- 利用 Tensor Core 进行高级优化
- 处理复杂的跨架构兼容性

### 实际工作示例

学完本阶段后，你可以：

1. **添加新架构编译支持**
```cmake
# 你能理解并修改编译配置
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
```

2. **理解架构检测代码**
```cpp
// 你能理解这段代码的作用
static constexpr bool int8_mma_available(const int cc) {
    return cc < CC_OFFSET_AMD && cc >= CC_TURING;
}
```

3. **诊断架构错误**
```
// 你能理解这种错误并解决
ERROR: CUDA kernel has no device code compatible with CUDA arch 750
// 解决方案：添加 -gencode arch=compute_75,code=sm_75
```

4. **查询 GPU 信息**
```bash
# 你能使用这些命令诊断问题
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## 章节目录

1. [计算能力版本](01-compute-capability.md)
2. [架构特定特性](02-arch-specific-features.md)
3. [代码适配技术](03-code-adaptation.md)

## 预计学习时间

1-2 周

## 开始学习

请从 [01-compute-capability.md](01-compute-capability.md) 开始。
