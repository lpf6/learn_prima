# 2.3 代码适配技术

## 概述

本节介绍如何在代码中适配不同的 GPU 架构，确保代码在各种 GPU 上都能正确运行并发挥最佳性能。

## 适配技术概览

```
┌─────────────────────────────────────────────────────────────┐
│                     代码适配技术                              │
├─────────────────────────────────────────────────────────────┤
│  1. 编译时适配：#if __CUDA_ARCH__ >= XXX                     │
│  2. 运行时适配：if (cc >= XXX)                               │
│  3. 模板特化：template<> struct                              │
│  4. 函数重载：不同架构不同实现                                │
│  5. 多架构编译：-gencode                                     │
└─────────────────────────────────────────────────────────────┘
```

## 编译时适配

### 基本模式

```cpp
// 使用 __CUDA_ARCH__ 宏进行编译时判断

__global__ void my_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

#if __CUDA_ARCH__ >= CC_AMPERE
    // Ampere 及以上架构的优化代码
    float val = __reduce_add_sync(0xffffffff, data[idx]);
#elif __CUDA_ARCH__ >= CC_TURING
    // Turing 架构的代码
    float val = warp_reduce_sum(data[idx]);
#else
    // 通用实现
    float val = generic_reduce(data[idx]);
#endif
}
```

### Prima.cpp 中的实际例子

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

// FP16 最大值比较
static __device__ __forceinline__ half ggml_cuda_hmax(const half a, const half b) {
#ifdef FP16_AVAILABLE
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && CUDART_VERSION < CUDART_HMAX
    // CUDA 版本低于 11.7，使用软件实现
    return __float2half(fmaxf(__half2float(a), __half2float(b)));
#else
    // CUDA 11.7+，使用硬件指令
    return __hmax(a, b);
#endif
#else
    // FP16 不可用，返回默认值
    NO_DEVICE_CODE;
    GGML_UNUSED(b);
    return a;
#endif
}
```

### 编译时架构列表

```cpp
// 获取编译时支持的架构列表
#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

// 在错误信息中显示
#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#endif
```

## 运行时适配

### 设备信息查询

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

// 使用示例
void select_kernel(int cc) {
    if (int8_mma_available(cc)) {
        // 使用 INT8 MMA 内核
        launch_int8_mma_kernel<<<...>>>();
    } else if (fp16_mma_available(cc)) {
        // 使用 FP16 MMA 内核
        launch_fp16_mma_kernel<<<...>>>();
    } else {
        // 使用通用内核
        launch_generic_kernel<<<...>>>();
    }
}
```

### 内核选择策略

```cpp
// 文件：ggml/src/ggml-cuda.cu（简化示例）

enum ggml_cuda_kernel_type {
    KERNEL_TYPE_GENERIC,
    KERNEL_TYPE_FP16_MMA,
    KERNEL_TYPE_INT8_MMA,
};

static ggml_cuda_kernel_type select_kernel_type(int cc) {
    if (int8_mma_available(cc)) {
        return KERNEL_TYPE_INT8_MMA;
    }
    if (fp16_mma_available(cc)) {
        return KERNEL_TYPE_FP16_MMA;
    }
    return KERNEL_TYPE_GENERIC;
}
```

## 模板特化

### 按架构特化

```cpp
// 通用模板
template<int CC>
struct ArchTraits {
    static constexpr bool has_fp16_mma = false;
    static constexpr bool has_int8_mma = false;
};

// Volta 特化
template<>
struct ArchTraits<700> {
    static constexpr bool has_fp16_mma = true;
    static constexpr bool has_int8_mma = false;
};

// Turing 特化
template<>
struct ArchTraits<750> {
    static constexpr bool has_fp16_mma = true;
    static constexpr bool has_int8_mma = true;
};

// Ampere 特化
template<>
struct ArchTraits<800> {
    static constexpr bool has_fp16_mma = true;
    static constexpr bool has_int8_mma = true;
};
```

### Prima.cpp 中的类型特性

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

template <ggml_type type>
struct ggml_cuda_type_traits;

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0;  // 每块元素数
    static constexpr int qr = QR4_0;  // 每元素存储位数
    static constexpr int qi = QI4_0;  // 每块存储整数数
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q8_0> {
    static constexpr int qk = QK8_0;
    static constexpr int qr = QR8_0;
    static constexpr int qi = QI8_0;
};
```

## 多架构编译

### CMake 配置

```cmake
# 文件：ggml/src/CMakeLists.txt（简化示例）

# 设置 CUDA 架构
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90")
endif()

# 添加可移植性标志
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    )
endif()
```

### 条件编译最佳实践

```cpp
// 好的模式：分层检查

__device__ void process_data(float *data) {
    // 第一层：检查 FP16 是否可用
#ifdef FP16_AVAILABLE
    half h_data = __float2half(data[0]);

    // 第二层：检查特定架构特性
#if __CUDA_ARCH__ >= CC_VOLTA
    // Volta+ 优化
    h_data = __hmax(h_data, __float2half(0.0f));
#else
    // 通用实现
    h_data = __float2half(fmaxf(__half2float(h_data), 0.0f));
#endif

#else
    // FP16 不可用，使用 FP32
    data[0] = fmaxf(data[0], 0.0f);
#endif
}
```

## 错误处理

### 不支持的架构

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

[[noreturn]]
static __device__ void no_device_code(
    const char * file_name,
    const int line,
    const char * function_name,
    const int arch,
    const char * arch_list) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    printf("%s:%d: ERROR: HIP kernel %s has no device code compatible with HIP arch %d.\n",
           file_name, line, function_name, arch);
#else
    printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. "
           "ggml-cuda.cu was compiled for: %s\n",
           file_name, line, function_name, arch, arch_list);
#endif
    __trap();  // 终止内核执行
}
```

### 运行时检查

```cpp
// 在主机代码中检查架构支持
bool check_arch_support(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int cc = prop.major * 10 + prop.minor;

    if (cc < 60) {
        printf("Warning: GPU compute capability %d.%d is below minimum (6.0)\n",
               prop.major, prop.minor);
        return false;
    }

    return true;
}
```

## 性能调优

### 根据架构调整参数

```cpp
// 文件：ggml/src/ggml-cuda/mmq.cu（简化示例）

// 根据架构选择最优块大小
int get_optimal_block_size(int cc) {
    if (cc >= CC_AMPERE) {
        return 128;  // Ampere+ 使用较大块
    } else if (cc >= CC_TURING) {
        return 64;   // Turing 使用中等块
    } else {
        return 32;   // 旧架构使用较小块
    }
}

// 根据架构选择共享内存大小
size_t get_shared_mem_size(int cc, int block_size) {
    // 考虑架构的共享内存限制
    if (cc >= CC_AMPERE) {
        return block_size * 32 * sizeof(float);  // Ampere 有更多共享内存
    } else {
        return block_size * 16 * sizeof(float);  // 保守配置
    }
}
```

## 练习

1. 编写一个内核，在不同架构上使用不同的优化策略
2. 阅读 `ggml/src/ggml-cuda/mmq.cu`，理解矩阵乘法的架构适配
3. 阅读 `ggml/src/ggml-cuda/fattn.cu`，理解 Flash Attention 的架构适配

## 下一步

完成本阶段后，请继续学习 [第三阶段：量化技术](../03-quantization/README.md)。
