# 7.1 多后端架构

## 概述

GGML 采用后端抽象架构，支持多种硬件平台。本节介绍后端架构设计和各后端实现。

## 后端架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│                   (llama.cpp API)                            │
├─────────────────────────────────────────────────────────────┤
│                    Backend Abstraction                       │
│                   (ggml-backend.h)                           │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│   CPU    │   CUDA   │   Metal  │  Vulkan  │     ROCm       │
│ Backend  │ Backend  │ Backend  │ Backend  │    Backend     │
├──────────┴──────────┴──────────┴──────────┴────────────────┤
│                    Hardware Layer                            │
│         x86/ARM     NVIDIA     Apple     GPU      AMD       │
└─────────────────────────────────────────────────────────────┘
```

## 后端接口

### 后端结构

```cpp
// 文件：ggml/include/ggml-backend.h

struct ggml_backend {
    ggml_guid_t guid;
    struct ggml_backend_i iface;
    void * context;
};

struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t backend);

    void (*free)(ggml_backend_t backend);

    ggml_backend_buffer_type_t (*get_default_buffer_type)(ggml_backend_t backend);

    void (*set_tensor_async)(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void (*get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);

    bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

    void (*synchronize)(ggml_backend_t backend);

    ggml_status (*graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph);
};
```

### 缓冲区类型

```cpp
struct ggml_backend_buffer_type_i {
    const char * (*get_name)(ggml_backend_buffer_type_t buft);

    void (*free)(ggml_backend_buffer_type_t buft);

    ggml_backend_buffer_t (*alloc_buffer)(ggml_backend_buffer_type_t buft, size_t size);

    size_t (*get_alignment)(ggml_backend_buffer_type_t buft);
    size_t (*get_max_size)(ggml_backend_buffer_type_t buft);

    size_t (*get_alloc_size)(ggml_backend_buffer_type_t buft, struct ggml_tensor * tensor);

    bool (*is_host)(ggml_backend_buffer_type_t buft);
};
```

## CPU 后端

### 实现文件

```
ggml/src/
├── ggml.c           # 核心实现
├── ggml-quants.c    # 量化实现
├── ggml-aarch64.c   # ARM64 优化
└── llamafile/
    └── sgemm.cpp    # SIMD SGEMM
```

### CPU 后端特点

```cpp
// CPU 后端注册
ggml_backend_t ggml_backend_cpu_init(void) {
    struct ggml_backend_cpu_context * ctx = calloc(1, sizeof(*ctx));
    // ...

    ggml_backend_t backend = calloc(1, sizeof(*backend));
    backend->iface = ggml_backend_cpu_i;
    backend->context = ctx;

    return backend;
}
```

### SIMD 调度

```cpp
// 文件：ggml/src/llamafile/sgemm.cpp

void sgemm_kernel_dispatch(
    const float * A, const float * B, float * C,
    int M, int N, int K) {

#if defined(__AVX512F__)
    sgemm_kernel_avx512(A, B, C, M, N, K);
#elif defined(__AVX2__)
    sgemm_kernel_avx2(A, B, C, M, N, K);
#elif defined(__ARM_NEON)
    sgemm_kernel_neon(A, B, C, M, N, K);
#else
    sgemm_kernel_scalar(A, B, C, M, N, K);
#endif
}
```

## CUDA 后端

### 实现文件

```
ggml/src/ggml-cuda/
├── ggml-cuda.cu     # 主入口
├── common.cuh       # 公共工具
├── mmq.cu           # 矩阵乘法
├── fattn.cu         # Flash Attention
├── quantize.cu      # 量化
└── ...              # 其他内核
```

### CUDA 后端特点

```cpp
// 文件：ggml/src/ggml-cuda.cu

struct ggml_backend_cuda_context {
    int device;
    std::string name;
    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS];
    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES];
    std::unique_ptr<ggml_cuda_pool> pools[GGML_CUDA_MAX_DEVICES];
    std::unique_ptr<ggml_cuda_graph> cuda_graph;
};
```

### 多 GPU 支持

```cpp
// 多 GPU 张量分割
void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ...) {
    const int n_devices = ggml_cuda_info().device_count;

    if (n_devices > 1 && split_mode == LLAMA_SPLIT_MODE_ROW) {
        // 按行分割张量到多个 GPU
        for (int i = 0; i < n_devices; ++i) {
            // 每个设备处理一部分行
            ggml_cuda_mul_mat_on_device(ctx, src0, src1, dst, i);
        }
    }
}
```

## Metal 后端 (Apple)

### 实现文件

```
ggml/src/
├── ggml-metal.m     # Metal 后端实现
└── ggml-metal.metal # Metal 着色器
```

### Metal 后端特点

```objc
// 文件：ggml/src/ggml-metal.m

struct ggml_backend_metal_context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;

    // 内核函数
    id<MTLComputePipelineState> pipeline_mul_mm_f32;
    id<MTLComputePipelineState> pipeline_flash_attn;
    // ...
};
```

### Metal 着色器

```metal
// 文件：ggml/src/ggml-metal.metal

kernel void kernel_mul_mm_f32(
    device const float * A [[buffer(0)]],
    device const float * B [[buffer(1)]],
    device float * C [[buffer(2)]],
    constant int & M [[buffer(3)]],
    constant int & N [[buffer(4)]],
    constant int & K [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]) {
    // Metal 内核实现
}
```

## Vulkan 后端

### 实现文件

```
ggml/src/
├── ggml-vulkan.cpp  # Vulkan 后端实现
└── vulkan-shaders/  # Vulkan 着色器
    ├── mul_mat_vec.comp
    ├── rms_norm.comp
    └── ...
```

### Vulkan 后端特点

```cpp
// 文件：ggml/src/ggml-vulkan.cpp

struct ggml_backend_vk_context {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    VkCommandPool command_pool;

    // 着色器模块
    VkShaderModule shader_mul_mat_vec;
    VkShaderModule shader_rms_norm;
    // ...
};
```

### Vulkan 计算着色器

```glsl
// 文件：ggml/src/vulkan-shaders/mul_mat_vec.comp

#version 450

layout(local_size_x = 32) in;

layout(binding = 0) buffer A { float a[]; };
layout(binding = 1) buffer B { float b[]; };
layout(binding = 2) buffer C { float c[]; };

void main() {
    // Vulkan 计算着色器实现
}
```

## ROCm 后端 (AMD)

### HIP 兼容层

```cpp
// 文件：ggml/src/ggml-cuda/vendors/hip.h

// HIP 兼容性宏
#define cudaDeviceProp hipDeviceProp_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
// ...
```

### AMD 架构适配

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    // AMD GPU 特定实现
    #if defined(__gfx906__) || defined(__gfx908__)
        // MI50/MI100
    #elif defined(RDNA2)
        // RX 6000 系列
    #elif defined(RDNA3)
        // RX 7000 系列
    #endif
#endif
```

## 后端调度

### 自动后端选择

```cpp
// 文件：ggml/src/ggml-backend.cpp

ggml_backend_t ggml_backend_init_best(void) {
    // 优先级：CUDA > Metal > Vulkan > CPU

#if defined(GGML_USE_CUDA)
    if (ggml_cuda_has_device()) {
        return ggml_backend_cuda_init(0);
    }
#endif

#if defined(GGML_USE_METAL)
    if (ggml_metal_has_device()) {
        return ggml_backend_metal_init();
    }
#endif

#if defined(GGML_USE_VULKAN)
    if (ggml_vulkan_has_device()) {
        return ggml_backend_vulkan_init(0);
    }
#endif

    return ggml_backend_cpu_init();
}
```

### 混合后端计算

```cpp
// 多后端计算图执行
void ggml_backend_graph_compute_async(
    ggml_backend_t * backends,
    int n_backends,
    struct ggml_cgraph * cgraph) {

    // 分配张量到各后端
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        ggml_backend_t backend = select_backend_for_tensor(node);
        // ...
    }

    // 执行计算
    for (int i = 0; i < n_backends; ++i) {
        ggml_backend_synchronize(backends[i]);
    }
}
```

## 跨平台兼容性

### 编译时检测

```cpp
// 检测可用后端
#if defined(GGML_USE_CUDA)
    #define HAS_CUDA_BACKEND 1
#endif

#if defined(GGML_USE_METAL)
    #define HAS_METAL_BACKEND 1
#endif

#if defined(GGML_USE_VULKAN)
    #define HAS_VULKAN_BACKEND 1
#endif
```

### 运行时检测

```cpp
// 检测硬件支持
bool check_cuda_available() {
#if defined(GGML_USE_CUDA)
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

bool check_metal_available() {
#if defined(GGML_USE_METAL)
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
#else
    return false;
#endif
}
```

## 性能对比

```
推理速度对比 (7B 模型, tokens/s):

CPU (AVX-512):    5-10
Metal (M1):       30-50
CUDA (RTX 3090):  100-150
CUDA (A100):      200-300
```

## 练习

1. 阅读 `ggml/include/ggml-backend.h`，理解后端接口设计
2. 比较不同后端的内核实现差异
3. 尝试添加新的后端支持

## 总结

完成本教程后，你应该能够：

1. 理解 CUDA 编程模型和优化技术
2. 处理不同 GPU 架构的兼容性
3. 理解量化技术及其实现
4. 阅读和优化核心 CUDA 内核
5. 处理 CPU SIMD 指令集兼容性
6. 为项目添加新模型支持
7. 理解多后端架构

继续学习建议：
- 阅读项目源码，深入理解实现细节
- 尝试优化现有内核
- 参与社区讨论和贡献

祝学习顺利！
