# ROCm/HIP 后端详解

## 概述

ROCm（Radeon Open Compute）是 AMD 开发的 GPU 计算平台，HIP（Heterogeneous-Compute Interface for Portability）是 ROCm 的编程接口。HIP 允许开发者将 CUDA 代码移植到 AMD GPU，同时保持代码的可移植性。

## 1. ROCm 架构基础

### 1.1 ROCm 与 CUDA 对比

| 特性 | CUDA | ROCm/HIP |
|------|------|----------|
| **厂商** | NVIDIA | AMD |
| **编程语言** | CUDA C++ | HIP C++ |
| **指令集** | SASS | GCN/RDNA ISA |
| **内存模型** | 基本相同 | 基本相同 |
| **线程层次** | Grid/Block/Thread | Grid/Block/Thread |
| **工具链** | nvcc/cuBLAS/cuDNN | hipcc/rocBLAS/MIOpen |

### 1.2 HIP 与 CUDA 语法映射

```
CUDA → HIP 映射:
┌─────────────────────────────────────────────────────────────┐
│  CUDA                        HIP                             │
│  __global__                  →  __global__ (相同)            │
│  __device__                  →  __device__ (相同)            │
│  __host__                    →  __host__ (相同)              │
│  __shared__                  →  __shared__ (相同)            │
│  cudaMalloc()                →  hipMalloc()                  │
│  cudaFree()                  →  hipFree()                    │
│  cudaMemcpy()                →  hipMemcpy()                  │
│  cudaStream_t                →  hipStream_t                  │
│  cudaEvent_t                 →  hipEvent_t                   │
│  threadIdx.x                 →  threadIdx.x (相同)           │
│  blockIdx.x                  →  blockIdx.x (相同)            │
│  blockDim.x                  →  blockDim.x (相同)            │
│  gridDim.x                   →  gridDim.x (相同)             │
│  __syncthreads()             →  __syncthreads() (相同)       │
│  atomicAdd()                 →  atomicAdd() (相同)           │
└─────────────────────────────────────────────────────────────┘
```

## 2. HIP 编程模型

### 2.1 HIP 内核函数

```cpp
// 文件：hip_kernels.cpp

#include <hip/hip_runtime.h>

// 向量加法内核
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 矩阵乘法内核（使用共享内存）
__global__ void matmul(const float *A, const float *B, float *C, 
                      int M, int N, int K) {
    constexpr int TILE_SIZE = 16;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int k = 0; k < K / TILE_SIZE; k++) {
        // 加载到共享内存
        if (row < M && k * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = 
                A[row * K + k * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && k * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = 
                B[(k * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // 计算
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // 存储结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 2.2 HIP 内存管理

```cpp
// HIP 内存类型
enum hipMemoryType {
    hipMemoryTypeHost = 0,      // 主机内存
    hipMemoryTypeDevice = 1,    // 设备内存
    hipMemoryTypeUnified = 2,   // 统一内存
};

// HIP 内存拷贝
hipError_t hipMemcpy(
    void* dst,
    const void* src,
    size_t sizeBytes,
    hipMemcpyKind kind
);

// 内存拷贝类型:
// - hipMemcpyHostToHost
// - hipMemcpyHostToDevice
// - hipMemcpyDeviceToHost
// - hipMemcpyDeviceToDevice
// - hipMemcpyDefault (自动检测)
```

## 3. ROCm 后端实现

### 3.1 GGML HIP 后端

```cpp
// 文件：ggml-hip.cpp

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>

struct ggml_hip_context {
    int device_id;
    hipDevice_t device;
    hipStream_t stream;
    hipblasHandle_t hipblas_handle;
    rocblas_handle rocblas_handle;
    
    // 内存池
    std::vector<void*> device_buffers;
};

// 初始化 HIP 上下文
struct ggml_hip_context * ggml_hip_init(int device_id) {
    struct ggml_hip_context * ctx = calloc(1, sizeof(*ctx));
    
    ctx->device_id = device_id;
    
    // 1. 设置设备
    hipError_t err = hipSetDevice(device_id);
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to set HIP device\n");
        return NULL;
    }
    
    // 2. 获取设备
    err = hipGetDevice(&ctx->device);
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to get HIP device\n");
        return NULL;
    }
    
    // 3. 创建流
    err = hipStreamCreate(&ctx->stream);
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to create HIP stream\n");
        return NULL;
    }
    
    // 4. 创建 BLAS 句柄
    hipblasCreate(&ctx->hipblas_handle);
    hipblasSetStream(ctx->hipblas_handle, ctx->stream);
    
    rocblas_create_handle(&ctx->rocblas_handle);
    rocblas_set_stream(ctx->rocblas_handle, ctx->stream);
    
    return ctx;
}

// HIP 矩阵乘法
void ggml_hip_gemm(
    struct ggml_hip_context * ctx,
    hipblasOperation_t trans_a,
    hipblasOperation_t trans_b,
    int m, int n, int k,
    const float * alpha,
    const float * A, int lda,
    const float * B, int ldb,
    const float * beta,
    float * C, int ldc) {
    
    hipblasSgemm(ctx->hipblas_handle,
                trans_a, trans_b,
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
}
```

### 3.2 量化内核 HIP 实现

```cpp
// HIP Q4_0 量化内核
__global__ void dequantize_q4_0_hip(
    const block_q4_0 * vx,
    float * y,
    int k) {
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= k / QK4_0) {
        return;
    }
    
    const float d = vx[i].d;
    const uint8_t * qs = vx[i].qs;
    
    // 反量化
    for (int j = 0; j < QK4_0 / 2; j++) {
        const float v0 = (qs[j] & 0x0F) * d - 8.0f * d;
        const float v1 = (qs[j] >> 4) * d - 8.0f * d;
        
        y[i * QK4_0 + j * 2 + 0] = v0;
        y[i * QK4_0 + j * 2 + 1] = v1;
    }
}

// HIP 量化矩阵乘法
__global__ void matmul_q4_0_hip(
    const block_q4_0 * vx,
    const float * y,
    float * dst,
    int ncols_x,
    int nrows_x,
    int ncols_y) {
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= nrows_x || col >= ncols_y) {
        return;
    }
    
    const int blocks_per_row = ncols_x / QK4_0;
    const block_q4_0 * x = vx + row * blocks_per_row;
    
    float sum = 0.0f;
    
    for (int i = 0; i < blocks_per_row; i++) {
        const float d = x[i].d;
        const uint8_t * qs = x[i].qs;
        
        for (int j = 0; j < QK4_0 / 2; j++) {
            const float v0 = (qs[j] & 0x0F) * d - 8.0f * d;
            const float v1 = (qs[j] >> 4) * d - 8.0f * d;
            
            sum += v0 * y[i * QK4_0 + j * 2 + 0];
            sum += v1 * y[i * QK4_0 + j * 2 + 1];
        }
    }
    
    dst[row * ncols_y + col] = sum;
}
```

## 4. CUDA 到 HIP 移植

### 4.1 自动移植工具

```bash
# 使用 hipify-perl 自动转换
hipify-perl my_cuda_kernel.cu > my_hip_kernel.cpp

# 使用 hipify-clang（更准确）
hipify-clang my_cuda_kernel.cu \
    -o my_hip_kernel.cpp \
    -- \
    -I/opt/rocm/include \
    -x cuda

# 手动检查和修复
# 1. 检查 hipify 输出
# 2. 手动修复复杂宏
# 3. 测试编译
```

### 4.2 手动移植示例

```cpp
// CUDA 版本
__global__ void cuda_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

// HIP 版本（基本相同）
__global__ void hip_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

// 主要区别在 API 调用
void launch_cuda_kernel(float *d_data, int n) {
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(...);
    cuda_kernel<<<blocks, threads>>>(d_data, n);
    cudaDeviceSynchronize();
}

void launch_hip_kernel(float *d_data, int n) {
    hipMalloc(&d_data, n * sizeof(float));
    hipMemcpy(...);
    hip_kernel<<<blocks, threads>>>(d_data, n);
    hipDeviceSynchronize();
}
```

### 4.3 条件编译支持

```cpp
// 同时支持 CUDA 和 HIP
#ifdef __HIP_PLATFORM_AMD__
    #include <hip/hip_runtime.h>
    #define GGML_BACKEND_NAME "HIP"
    #define GGML_MALLOC hipMalloc
    #define GGML_FREE hipFree
    #define GGML_MEMCPY hipMemcpy
    #define GGML_MEMCPY_H2D hipMemcpyHostToDevice
    #define GGML_MEMCPY_D2H hipMemcpyDeviceToHost
    #define GGML_SYNCHRONIZE hipDeviceSynchronize
    #define GGML_GET_ERROR hipGetLastError
    #define GGML_GET_ERROR_STRING hipGetErrorString
#else
    #include <cuda_runtime.h>
    #define GGML_BACKEND_NAME "CUDA"
    #define GGML_MALLOC cudaMalloc
    #define GGML_FREE cudaFree
    #define GGML_MEMCPY cudaMemcpy
    #define GGML_MEMCPY_H2D cudaMemcpyHostToDevice
    #define GGML_MEMCPY_D2H cudaMemcpyDeviceToHost
    #define GGML_SYNCHRONIZE cudaDeviceSynchronize
    #define GGML_GET_ERROR cudaGetLastError
    #define GGML_GET_ERROR_STRING cudaGetErrorString
#endif

// 通用内核启动
void launch_kernel(void *d_data, int n) {
    my_kernel<<<blocks, threads>>>(d_data, n);
    
    auto err = GGML_GET_ERROR();
    if (err != 0) {
        fprintf(stderr, "%s kernel launch error: %s\n",
                GGML_BACKEND_NAME, GGML_GET_ERROR_STRING(err));
    }
    
    GGML_SYNCHRONIZE();
}
```

## 5. ROCm 性能优化

### 5.1 AMD GPU 架构特性

```cpp
// 针对 AMD GPU 优化
__global__ void amd_optimized_kernel(float *data, int n) {
    // 使用 LDS (Local Data Share，相当于共享内存)
    __shared__ float lds_data[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // 优化内存访问模式
    if (idx < n) {
        // 合并访问
        lds_data[tid] = data[idx];
        __syncthreads();
        
        // LDS 操作（更快）
        float value = lds_data[tid] * 2.0f;
        
        // 写回
        data[idx] = value;
    }
}

// 使用 AMD 矩阵核心（Matrix Core）
#ifdef __HIP_PLATFORM_AMD__
#include <hip/amd_detail/amd_hip_matrix_intrinsic.h>

__global__ void matrix_core_kernel(float *C, const float *A, const float *B,
                                   int M, int N, int K) {
    // 使用 AMD Matrix Core 指令
    // 类似 NVIDIA Tensor Core
}
#endif
```

### 5.2 内存带宽优化

```cpp
// 优化内存带宽使用
__global__ void bandwidth_optimized_kernel(
    const float * __restrict__ input,
    float * __restrict__ output,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 使用 restrict 关键字告诉编译器指针不重叠
        // 帮助编译器优化
        
        float value = input[idx];
        
        // 融合多个操作
        value = fmaf(value, 2.0f, 1.0f);  // value * 2.0 + 1.0
        
        // 减少全局内存访问
        output[idx] = value;
    }
}
```

## 6. ROCm 工具链

### 6.1 性能分析工具

```bash
# rocprof - ROCm 性能分析器
rocprof --stats ./program

# roctracer - ROCm 跟踪工具
roctracer --hip-activity --kernel-trace ./program

# rocminfo - 查看 ROCm 设备信息
rocminfo

# rocm-smi - 系统管理接口
rocm-smi --showalluse
```

### 6.2 调试工具

```bash
# rocgdb - ROCm GDB
rocgdb ./program

# 使用 KFD 调试
export HSA_ENABLE_SDMA=0
./program
```

## 7. 实际案例

### 7.1 Flash Attention HIP 实现

```cpp
// Flash Attention HIP 内核
__global__ void flash_attention_hip(
    const half *Q, const half *K, const half *V,
    float *O,
    int B, int H, int T, int D,
    float scale) {
    
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B || h >= H || i >= T || d >= D) {
        return;
    }
    
    // 加载 Q
    half q = Q[((b * H + h) * T + i) * D + d];
    
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc = 0.0f;
    
    // 遍历所有 K
    for (int j = 0; j < T; j++) {
        half k = K[((b * H + h) * T + j) * D + d];
        
        // 计算 QK^T
        float qk = (float)q * (float)k * scale;
        
        // 在线 Softmax
        float new_max = fmaxf(max_val, qk);
        float new_sum = sum_exp * expf(max_val - new_max) + expf(qk - new_max);
        
        // 加载 V 并累加
        half v = V[((b * H + h) * T + j) * D + d];
        acc = (acc * sum_exp * expf(max_val - new_max) + 
               expf(qk - new_max) * (float)v) / new_sum;
        
        max_val = new_max;
        sum_exp = new_sum;
    }
    
    // 存储结果
    O[((b * H + h) * T + i) * D + d] = acc;
}
```

## 练习

1. 将 CUDA 向量加法移植到 HIP
2. 实现 Q4_0 量化的 HIP 内核
3. 使用 rocprof 分析 HIP 内核性能
4. 编写同时支持 CUDA 和 HIP 的代码

## 参考资料

- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP API Reference](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/api.html)
- [GGML HIP Source](https://github.com/ggerganov/llama.cpp/blob/master/ggml-hip.cpp)
