# 后端选择指南

## 概述

Prima.cpp 支持多种后端（CUDA、Metal、Vulkan、ROCm/HIP），每种后端都有其适用场景。本章帮助你根据硬件平台、性能需求和功能特性选择合适的后端。

## 1. 后端能力对比

### 1.1 平台支持

| 后端 | NVIDIA | AMD | Intel | Apple | 移动设备 | 跨平台 |
|------|--------|-----|-------|-------|----------|--------|
| **CUDA** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Metal** | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Vulkan** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **ROCm** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |

### 1.2 性能对比

```
性能排名（相对值，以 CUDA 为基准）:

NVIDIA GPU (RTX 4090):
┌─────────────────────────────────────────────────────────────┐
│ CUDA:     ████████████████████████████████ 100%             │
│ Vulkan:   ████████████████████████████░░░░  85%             │
│ Metal:    N/A                                              │
│ ROCm:     N/A                                              │
└─────────────────────────────────────────────────────────────┘

AMD GPU (RX 7900 XTX):
┌─────────────────────────────────────────────────────────────┐
│ CUDA:     N/A                                              │
│ Vulkan:   █████████████████████████████░░░░  80%            │
│ Metal:    N/A                                              │
│ ROCm:     ████████████████████████████████ 100%             │
└─────────────────────────────────────────────────────────────┘

Apple Silicon (M2 Ultra):
┌─────────────────────────────────────────────────────────────┐
│ CUDA:     N/A                                              │
│ Vulkan:   N/A                                              │
│ Metal:    ████████████████████████████████ 100%             │
│ ROCm:     N/A                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 功能特性对比

| 特性 | CUDA | Metal | Vulkan | ROCm |
|------|------|-------|--------|------|
| **FP16 支持** | ✅ | ✅ | ✅ | ✅ |
| **INT8 量化** | ✅ | ✅ | ✅ | ✅ |
| **INT4 量化** | ✅ | ✅ | ✅ | ✅ |
| **Tensor Core** | ✅ | ❌ | ❌ | ✅ (Matrix Core) |
| **统一内存** | ❌ | ✅ | ⚠️ | ⚠️ |
| **多 GPU** | ✅ | ❌ | ✅ | ✅ |
| **动态并行** | ✅ | ❌ | ❌ | ✅ |

## 2. 后端选择决策树

```
选择后端决策流程:
┌─────────────────────────────────────────────────────────────┐
│ 1. 你的 GPU 是什么品牌？                                     │
│                                                            │
│    ├─ NVIDIA ──→ 选择 CUDA（最佳性能）                      │
│    │              备选：Vulkan（跨平台需求）                │
│    │                                                        │
│    ├─ AMD ────→ 选择 ROCm（Linux）                          │
│    │              备选：Vulkan（Windows/跨平台）            │
│    │                                                        │
│    ├─ Apple Silicon ──→ 选择 Metal（唯一选择）              │
│    │                                                        │
│    ├─ Intel ────→ 选择 Vulkan                               │
│    │                                                        │
│    └─ 移动 GPU ──→ 选择 Vulkan (Android)                    │
│                   选择 Metal (iOS)                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 2. 是否有特殊需求？                                          │
│                                                            │
│    ├─ 跨平台部署 ──→ Vulkan                                 │
│    ├─ 最高性能 ────→ 原生后端 (CUDA/ROCm/Metal)             │
│    ├─ 量化推理 ────→ 所有后端都支持                         │
│    └─ 多 GPU 并行 ──→ CUDA / ROCm / Vulkan                  │
└─────────────────────────────────────────────────────────────┘
```

## 3. 各后端详细对比

### 3.1 CUDA 后端

**优势**：
- ✅ 最佳性能（NVIDIA GPU）
- ✅ 最成熟的生态系统
- ✅ 完整的工具链（Nsight、cuda-gdb）
- ✅ Tensor Core 支持
- ✅ 丰富的库（cuBLAS、cuDNN）

**劣势**：
- ❌ 仅限 NVIDIA GPU
- ❌ 无法跨平台

**适用场景**：
- NVIDIA GPU 用户
- 追求最高性能
- 需要 Tensor Core 加速
- 数据中心部署

**性能建议**：
```bash
# 启用 CUDA 优化
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# 使用 Tensor Core
# 在代码中启用 FP16 计算
```

### 3.2 Metal 后端

**优势**：
- ✅ Apple Silicon 原生支持
- ✅ 统一内存架构（零拷贝）
- ✅ 能效比高
- ✅ 支持 iOS/macOS

**劣势**：
- ❌ 仅限 Apple 平台
- ❌ 无 Tensor Core 等效功能
- ❌ 工具链相对简单

**适用场景**：
- Mac 用户（M1/M2/M3）
- iOS 设备部署
- 能效敏感场景

**性能建议**：
```cpp
// 优化 Metal 内存
MTLResourceOptions options = MTLResourceStorageModeShared;
// 使用统一内存，避免拷贝

// 使用 AMX 加速（如果可用）
// 利用 Apple 矩阵扩展
```

### 3.3 Vulkan 后端

**优势**：
- ✅ 真正的跨平台
- ✅ 支持所有主流 GPU
- ✅ 底层硬件访问
- ✅ 移动设备支持

**劣势**：
- ❌ API 复杂度高
- ❌ 性能略低于原生后端
- ❌ 调试困难

**适用场景**：
- 跨平台部署
- 混合 GPU 环境
- Android 设备
- 嵌入式系统

**性能建议**：
```cpp
// 使用专用显存
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT

// 异步计算
// 使用独立的计算和传输队列

// 批处理
// 合并多个命令缓冲
```

### 3.4 ROCm 后端

**优势**：
- ✅ AMD GPU 原生支持
- ✅ HIP 可移植 CUDA 代码
- ✅ Matrix Core 支持
- ✅ 开源生态

**劣势**：
- ❌ 仅限 AMD GPU
- ❌ Linux 支持较好，Windows 较弱
- ❌ 生态系统不如 CUDA

**适用场景**：
- AMD GPU 用户
- Linux 服务器
- 需要 CUDA 代码移植

**性能建议**：
```bash
# 启用 ROCm 优化
export HSA_ENABLE_SDMA=1
export ROCBLAS_LAYER=2

# 使用 Matrix Core
# 在代码中启用 FP16 计算
```

## 4. 性能基准对比

### 4.1 LLaMA-7B 推理性能

```
硬件：RTX 4090 (24GB)
精度：FP16
Batch Size: 1
Sequence Length: 2048

┌─────────────────────────────────────────────────────────────┐
│ 后端        │ Tokens/s │ 相对性能 │ 显存占用 │              │
├─────────────────────────────────────────────────────────────┤
│ CUDA        │   125    │   100%    │  14 GB   │  ⭐⭐⭐⭐⭐      │
│ Vulkan      │   105    │    84%    │  14 GB   │  ⭐⭐⭐⭐       │
└─────────────────────────────────────────────────────────────┘

硬件：RX 7900 XTX (24GB)
精度：FP16
Batch Size: 1
Sequence Length: 2048

┌─────────────────────────────────────────────────────────────┐
│ 后端        │ Tokens/s │ 相对性能 │ 显存占用 │              │
├─────────────────────────────────────────────────────────────┤
│ ROCm        │   110    │   100%    │  14 GB   │  ⭐⭐⭐⭐⭐      │
│ Vulkan      │    88    │    80%    │  14 GB   │  ⭐⭐⭐⭐       │
└─────────────────────────────────────────────────────────────┘

硬件：M2 Ultra (24GB 统一内存)
精度：FP16
Batch Size: 1
Sequence Length: 2048

┌─────────────────────────────────────────────────────────────┐
│ 后端        │ Tokens/s │ 相对性能 │ 内存占用 │              │
├─────────────────────────────────────────────────────────────┤
│ Metal       │    95    │   100%    │  14 GB   │  ⭐⭐⭐⭐⭐      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 量化推理性能

```
硬件：RTX 4090
模型：LLaMA-7B
精度：Q4_0

┌─────────────────────────────────────────────────────────────┐
│ 后端        │ Tokens/s │ 加速比   │ 显存占用 │              │
├─────────────────────────────────────────────────────────────┤
│ CUDA Q4_0   │   180    │  1.44x   │  4.5 GB  │  ⭐⭐⭐⭐⭐      │
│ Vulkan Q4_0 │   145    │  1.38x   │  4.5 GB  │  ⭐⭐⭐⭐       │
└─────────────────────────────────────────────────────────────┘

注：加速比相对于 FP16
```

## 5. 多后端配置

### 5.1 编译时选择

```bash
# 只编译 CUDA 后端
cmake -DGGML_CUDA=ON -DGGML_METAL=OFF -DGGML_VULKAN=OFF ..

# 只编译 Metal 后端
cmake -DGGML_CUDA=OFF -DGGML_METAL=ON -DGGML_VULKAN=OFF ..

# 编译多个后端
cmake -DGGML_CUDA=ON -DGGML_VULKAN=ON ..

# 默认后端选择
cmake -DGGML_BACKEND=CUDA ..
```

### 5.2 运行时选择

```cpp
// 环境变量选择后端
export GGML_BACKEND=CUDA    # CUDA
export GGML_BACKEND=METAL   # Metal
export GGML_BACKEND=VULKAN  # Vulkan
export GGML_BACKEND=ROCm    # ROCm

// 代码中查询可用后端
const char ** backends = ggml_available_backends();
printf("Available backends:\n");
for (int i = 0; backends[i] != NULL; i++) {
    printf("  - %s\n", backends[i]);
}

// 手动选择后端
ggml_backend_t backend = ggml_backend_init_by_name("CUDA");
// 或
ggml_backend_t backend = ggml_backend_init_best();
```

### 5.3 混合后端使用

```cpp
// 使用不同后端处理不同层
void hybrid_inference() {
    // Embedding 层在 CPU
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    
    // Attention 层在 GPU
    ggml_backend_t gpu_backend = ggml_backend_cuda_init(0);
    
    // 分配张量到不同后端
    struct ggml_tensor * embedding = 
        ggml_backend_alloc_tensor(cpu_backend, ...);
    struct ggml_tensor * attention = 
        ggml_backend_alloc_tensor(gpu_backend, ...);
    
    // 调度计算
    ggml_graph_compute_with_ctx(cpu_backend, graph_embedding);
    ggml_graph_compute_with_ctx(gpu_backend, graph_attention);
}
```

## 6. 故障排查

### 6.1 常见问题

**问题 1：CUDA 后端无法初始化**
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查设备可见性
export CUDA_VISIBLE_DEVICES=0,1
```

**问题 2：Metal 后端性能低**
```bash
# 检查是否使用专用 GPU（macOS）
# 系统设置 → 电池 → 自动切换图形

# 检查统一内存使用
# 确保使用 MTLResourceStorageModeShared
```

**问题 3：Vulkan 后端找不到设备**
```bash
# 安装 Vulkan 运行时
# Ubuntu: sudo apt install vulkan-tools
# Windows: 安装 NVIDIA/AMD 驱动

# 检查设备
vulkaninfo | grep deviceName
```

**问题 4：ROCm 后端权限错误**
```bash
# 添加用户到 render 组
sudo usermod -aG render $USER
sudo usermod -aG video $USER

# 检查设备权限
ls -l /dev/kfd
ls -l /dev/dri
```

### 6.2 性能调试

```cpp
// 后端性能监控
struct backend_stats {
    float compute_time;
    float memory_bandwidth;
    float utilization;
};

void profile_backend(ggml_backend_t backend) {
    struct backend_stats stats;
    
    // 收集统计信息
    ggml_backend_get_stats(backend, &stats);
    
    printf("Backend: %s\n", ggml_backend_name(backend));
    printf("  Compute Time: %.2f ms\n", stats.compute_time);
    printf("  Memory Bandwidth: %.2f GB/s\n", stats.memory_bandwidth);
    printf("  Utilization: %.1f%%\n", stats.utilization * 100);
}
```

## 7. 最佳实践

### 7.1 选择建议

```
快速选择指南:

1. 如果你是 NVIDIA GPU 用户:
   → 选择 CUDA（最佳性能）
   
2. 如果你是 AMD GPU 用户 (Linux):
   → 选择 ROCm（最佳性能）
   
3. 如果你是 AMD GPU 用户 (Windows):
   → 选择 Vulkan（最佳兼容性）
   
4. 如果你是 Apple Silicon 用户:
   → 选择 Metal（唯一选择）
   
5. 如果你需要跨平台部署:
   → 选择 Vulkan（最佳兼容性）
   
6. 如果你有混合 GPU 环境:
   → 选择 Vulkan 或 多后端混合
```

### 7.2 性能优化清单

```
性能优化检查清单:

□ 1. 选择正确的后端
  - 根据硬件选择原生后端
  - 考虑跨平台需求

□ 2. 启用量化
  - Q4_0 或 Q8_0
  - 平衡精度和性能

□ 3. 优化内存使用
  - 使用适当的批处理大小
  - 管理 KV Cache 大小

□ 4. 监控性能
  - 使用性能分析工具
  - 识别瓶颈

□ 5. 更新驱动
  - 保持 GPU 驱动最新
  - 更新后端库
```

## 练习

1. 在你的硬件上测试所有可用后端的性能
2. 比较不同量化级别下的性能差异
3. 实现运行时后端切换功能
4. 编写后端性能基准测试工具

## 参考资料

- [GGML Backend API](https://github.com/ggerganov/llama.cpp/tree/master/ggml-backend)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices/)
- [Metal Best Practices](https://developer.apple.com/metal/)
- [Vulkan Best Practices](https://www.khronos.org/vulkan/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
