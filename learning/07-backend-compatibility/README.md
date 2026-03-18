# 第七阶段：后端兼容性

## 概述

Prima.cpp 支持多种硬件后端，包括 CUDA、ROCm、Metal、Vulkan 等。本阶段介绍多后端架构和兼容性处理。

## 能力目标

完成本阶段后，你将能够：

### 能做什么

| 能力 | 具体表现 | 相关代码 |
|------|----------|----------|
| **理解后端架构** | 知道 GGML 后端抽象设计 | `ggml-backend.h` |
| **处理后端问题** | 诊断和修复后端相关 Bug | 维护工作 |
| **添加后端支持** | 为新硬件添加基础支持 | 扩展开发 |
| **跨平台调试** | 处理不同平台的兼容性问题 | 调试排错 |

### 还不能做什么

- 从零实现完整后端
- 进行深度后端优化
- 处理极端硬件兼容性

### 实际工作示例

学完本阶段后，你可以：

1. **理解后端接口**
```cpp
// 你能理解后端抽象接口
struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t backend);
    void (*free)(ggml_backend_t backend);
    ggml_status (*graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph);
    // ...
};
```

2. **处理后端选择**
```cpp
// 你能理解后端选择逻辑
ggml_backend_t ggml_backend_init_best(void) {
#if defined(GGML_USE_CUDA)
    if (ggml_cuda_has_device()) {
        return ggml_backend_cuda_init(0);
    }
#endif
    // ...
}
```

3. **诊断后端问题**
```bash
# 你能使用这些方法诊断问题
# 检查 CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 检查 Metal 可用性 (macOS)
system_profiler SPDisplaysDataType
```

4. **理解多后端协作**
```cpp
// 你能理解多后端计算图执行
void ggml_backend_graph_compute_async(
    ggml_backend_t * backends,
    int n_backends,
    struct ggml_cgraph * cgraph) {
    // 分配张量到各后端
    // 执行计算
    // 同步结果
}
```

## 章节目录

1. [多后端架构](01-multi-backend.md)

## 预计学习时间

1-2 周

## 开始学习

请从 [01-multi-backend.md](01-multi-backend.md) 开始。
