# 2.1 计算能力版本

## 什么是计算能力？

计算能力（Compute Capability）是 NVIDIA 对 GPU 架构版本的标识，格式为 `X.Y`，其中：

- `X`：主要架构版本
- `Y`：次要特性版本

**注意**：计算能力数值大小并不直接代表性能强弱，而是代表架构特性。

## 主要架构世代

| 计算能力 | 架构代号 | 发布年份 | 代表 GPU |
|----------|----------|----------|----------|
| 6.0 | Pascal | 2016 | Tesla P100 |
| 6.1 | Pascal | 2016 | GTX 1080, Titan X |
| 7.0 | Volta | 2017 | Tesla V100 |
| 7.5 | Turing | 2018 | RTX 2080, T4 |
| 8.0 | Ampere | 2020 | A100 |
| 8.6 | Ampere | 2020 | RTX 3090, RTX 3080 |
| 8.9 | Ada Lovelace | 2022 | RTX 4090 |
| 9.0 | Hopper | 2022 | H100 |

## Prima.cpp 中的架构定义

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

#define CC_PASCAL     600
#define MIN_CC_DP4A   610   // __dp4a 指令最低要求
#define CC_VOLTA      700
#define CC_TURING     750
#define CC_AMPERE     800
#define CC_OFFSET_AMD 1000000
#define CC_RDNA1      (CC_OFFSET_AMD + 1010)
#define CC_RDNA2      (CC_OFFSET_AMD + 1030)
#define CC_RDNA3      (CC_OFFSET_AMD + 1100)
```

## 关键特性对比

### 1. FP16 支持

| 架构 | FP16 支持 | 加速比 |
|------|-----------|--------|
| Pascal (6.0) | 存储格式 | 1x |
| Pascal (6.1) | 计算支持 | 2x |
| Volta+ | Tensor Core | 8x+ |

```cpp
// FP16 可用性检查
#if (defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ >= CC_PASCAL
#define FP16_AVAILABLE
#endif
```

### 2. INT8 支持

| 架构 | INT8 支持 | 关键特性 |
|------|-----------|----------|
| Pascal (6.1+) | __dp4a 指令 | 4字节点积 |
| Turing+ | Tensor Core INT8 | 矩阵加速 |

```cpp
// INT8 MMA 可用性检查
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_TURING
#define INT8_MMA_AVAILABLE
#endif
```

### 3. Tensor Core

| 架构 | Tensor Core 支持 |
|------|------------------|
| Volta (7.0) | FP16 |
| Turing (7.5) | FP16, INT8, INT4 |
| Ampere (8.0) | FP16, BF16, TF32, INT8 |
| Ada (8.9) | FP8 |
| Hopper (9.0) | FP8, FP16, BF16 |

### 4. 共享内存

| 架构 | 每 SM 共享内存 | 每 Block 最大 |
|------|---------------|---------------|
| Pascal | 64KB | 48KB |
| Volta | 128KB | 96KB |
| Turing | 64KB | 64KB |
| Ampere | 164KB | 163KB |
| Hopper | 228KB | 227KB |

## 查询 GPU 信息

### 命令行查询

```bash
# 查询所有 GPU 的计算能力
nvidia-smi --query-gpu=name,compute_cap --format=csv

# 输出示例：
# name, compute_cap
# NVIDIA GeForce RTX 3090, 8.6
# NVIDIA A100-SXM4-40GB, 8.0
```

### 代码查询

```cpp
// 文件：ggml/src/ggml-cuda.cu

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);

    for (int i = 0; i < info.device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        info.devices[i].cc = prop.major * 10 + prop.minor;  // 计算能力
        info.devices[i].nsm = prop.multiProcessorCount;      // SM 数量
        info.devices[i].smpb = prop.sharedMemPerBlock;       // 每 Block 共享内存
        info.devices[i].total_vram = prop.totalGlobalMem;    // 总显存
    }

    return info;
}
```

## 架构特性函数

Prima.cpp 提供了运行时架构特性检查函数：

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

static constexpr bool fast_fp16_available(const int cc) {
    return cc >= CC_PASCAL && cc != 610;
}

static constexpr bool fp16_mma_available(const int cc) {
    return cc < CC_OFFSET_AMD && cc >= CC_VOLTA;
}

static constexpr bool int8_mma_available(const int cc) {
    return cc < CC_OFFSET_AMD && cc >= CC_TURING;
}
```

## 编译时架构指定

### CMake 配置

```cmake
# 指定目标架构
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

# 或使用环境变量
export CUDAARCHS="75;80;86"
```

### Makefile 配置

```makefile
# 指定架构
NVCCFLAGS += -gencode arch=compute_75,code=sm_75
NVCCFLAGS += -gencode arch=compute_80,code=sm_80
NVCCFLAGS += -gencode arch=compute_86,code=sm_86
```

### 虚拟架构 vs 实际架构

```makefile
# 虚拟架构：用于 PTX 前向兼容
-gencode arch=compute_80,code=compute_80

# 实际架构：生成特定 GPU 的 SASS 代码
-gencode arch=compute_80,code=sm_80

# 推荐组合
-gencode arch=compute_80,code=sm_80   # A100
-gencode arch=compute_86,code=sm_86   # RTX 3090
-gencode arch=compute_80,code=compute_80  # 未来 GPU 兼容
```

## AMD GPU 支持

Prima.cpp 也支持 AMD GPU（通过 HIP/ROCm）：

```cpp
// AMD 架构定义
#define CC_OFFSET_AMD 1000000
#define CC_RDNA1      (CC_OFFSET_AMD + 1010)  // RX 5000 系列
#define CC_RDNA2      (CC_OFFSET_AMD + 1030)  // RX 6000 系列
#define CC_RDNA3      (CC_OFFSET_AMD + 1100)  // RX 7000 系列

// 检查是否是 AMD GPU
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    // AMD 特定代码
#endif
```

## 练习

1. 使用 `nvidia-smi` 查询你系统中 GPU 的计算能力
2. 阅读 `ggml/src/ggml-cuda.cu` 中的 `ggml_cuda_init()` 函数
3. 理解为什么需要同时指定虚拟架构和实际架构

## 下一步

完成本节后，请继续学习 [架构特定特性](02-arch-specific-features.md)。
