# Metal 后端详解

## 概述

Metal 是 Apple 的图形和计算 API，用于在 Apple Silicon（M1/M2/M3 系列）和 AMD GPU 上实现高性能计算。Prima.cpp 通过 Metal 后端支持 macOS 和 iOS 设备。

## 1. Metal 架构基础

### 1.1 Metal 与 CUDA 对比

| 特性 | CUDA | Metal |
|------|------|-------|
| **平台** | NVIDIA GPU | Apple Silicon/AMD GPU |
| **编程语言** | CUDA C++ | Metal Shading Language (MSL) |
| **内存模型** | 全局/共享/常量/寄存器 | 全局/线程组/常量/私有 |
| **线程层次** | Grid/Block/Thread | Grid/Threadgroup/Thread |
| **编译方式** | nvcc | clang + metallib |

### 1.2 线程层次映射

```
CUDA → Metal 映射:
┌─────────────────────────────────────────────────────────────┐
│  CUDA                    Metal                               │
│  Grid                    →  MTLGrid                          │
│  Block                   →  MTLThreadgroup                   │
│  Thread                  →  MTLThread                         │
│                                                                │
│  threadIdx.x/y/z         →  thread_position_in_threadgroup    │
│  blockIdx.x/y/z          →  threadgroup_position_in_grid      │
│  blockDim.x/y/z          →  threads_per_threadgroup           │
│  gridDim.x/y/z           →  grid_size_in_threadgroups         │
└─────────────────────────────────────────────────────────────┘
```

## 2. Metal 编程模型

### 2.1 Metal 内核函数

```metal
// 文件：metal_kernels.metal

#include <metal_stdlib>
using namespace metal;

// 向量加法内核
kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = a[id] + b[id];
}

// 矩阵乘法内核（使用线程组内存）
kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    // 线程组大小
    constexpr uint TILE_SIZE = 16;
    
    // 共享内存
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    // 分块计算
    for (uint k = 0; k < K / TILE_SIZE; k++) {
        // 加载到共享内存
        As[lid.y][lid.x] = A[gid.y * K + k * TILE_SIZE + lid.x];
        Bs[lid.y][lid.x] = B[(k * TILE_SIZE + lid.y) * N + gid.x];
        
        // 同步
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 计算
        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += As[lid.y][i] * Bs[i][lid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // 写回结果
    C[gid.y * N + gid.x] = sum;
}
```

### 2.2 内存类型映射

```cpp
// CUDA → Metal 内存映射
┌─────────────────────────────────────────────────────────────┐
│  CUDA                    Metal                               │
│  __global__              →  device                           │
│  __shared__              →  threadgroup                      │
│  __constant__            →  constant                         │
│  __local__               →  thread                           │
│                                                                │
│  cudaMalloc()            →  device.newBuffer()               │
│  cudaMemcpyAsync()       →  blitEncoder.copy()               │
│  __syncthreads()         →  threadgroup_barrier()            │
└─────────────────────────────────────────────────────────────┘
```

## 3. Metal 后端实现

### 3.1 GGML Metal 后端

```cpp
// 文件：ggml-metal.m

#import <Metal/Metal.h>

struct ggml_metal_context {
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> library;
    
    // 缓冲区
    NSMutableArray<id<MTLBuffer>> *buffers;
    
    // 内核函数
    id<MTLFunction> vector_add_fn;
    id<MTLFunction> matmul_fn;
    id<MTLFunction> gemm_fn;
};

// 初始化 Metal 上下文
struct ggml_metal_context * ggml_metal_init(void) {
    struct ggml_metal_context * ctx = calloc(1, sizeof(*ctx));
    
    // 1. 获取默认设备
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "Metal is not supported on this device\n");
        return NULL;
    }
    
    // 2. 创建命令队列
    ctx->command_queue = [ctx->device newCommandQueue];
    if (!ctx->command_queue) {
        fprintf(stderr, "Failed to create command queue\n");
        return NULL;
    }
    
    // 3. 编译着色器
    NSError *error = nil;
    NSString *source_path = [[NSBundle mainBundle] 
                             pathForResource:@"ggml-metal" 
                             ofType:@"metal"];
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    ctx->library = [ctx->device newLibraryWithFile:source_path
                                           options:options
                                             error:&error];
    
    if (error) {
        fprintf(stderr, "Failed to compile Metal shaders: %s\n", 
                [error.localizedDescription UTF8String]);
        return NULL;
    }
    
    // 4. 获取内核函数
    ctx->vector_add_fn = [ctx->library newFunctionWithName:@"vector_add"];
    ctx->matmul_fn = [ctx->library newFunctionWithName:@"matmul"];
    
    return ctx;
}

// 执行 Metal 内核
void ggml_metal_compute(
    struct ggml_metal_context * ctx,
    struct ggml_tensor * tensor) {
    
    // 1. 创建命令缓冲区和编码器
    id<MTLCommandBuffer> command_buffer = 
        [ctx->command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = 
        [command_buffer computeCommandEncoder];
    
    // 2. 设置管线状态
    MTLComputePipelineState *pipeline_state = 
        [ctx->device newComputePipelineStateWithFunction:ctx->matmul_fn
                                                   error:nil];
    [encoder setComputePipelineState:pipeline_state];
    
    // 3. 设置缓冲区参数
    [encoder setBuffer:tensor->buffer offset:0 atIndex:0];
    // ... 设置其他参数
    
    // 4. 计算网格大小
    MTLSize grid_size;
    grid_size.width = tensor->ne[0];
    grid_size.height = tensor->ne[1];
    grid_size.depth = tensor->ne[2];
    
    // 5. 分派计算
    [encoder dispatchThreads:grid_size 
     threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    
    // 6. 结束编码并执行
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}
```

### 3.2 量化内核实现

```metal
// 文件：ggml-metal-quant.metal

#include <metal_stdlib>
using namespace metal;

// Q4_0 反量化
inline float dequantize_q4_0(thread const uint8_t &qs, int shift) {
    return (shift == 0 ? (qs & 0x0F) : (qs >> 4)) - 8;
}

// Q4_0 矩阵乘法内核
kernel void matmul_q4_0(
    device const void * vx [[buffer(0)]],
    device const void * vy [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int & ncols_x [[buffer(3)]],
    constant int & nrows_x [[buffer(4)]],
    constant int & ncols_y [[buffer(5)]],
    constant int & nrows_y [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= nrows_y || gid.y >= ncols_x) {
        return;
    }
    
    // Q4_0 块大小
    constexpr int QK4_0 = 32;
    constexpr int QR4_0 = 2;
    
    const int blocks_per_row = ncols_x / QK4_0;
    
    // 获取权重块
    const block_q4_0 * x = (const block_q4_0 *) vx;
    x += gid.y * blocks_per_row;
    
    // 获取激活值
    const float * y = (const float *) vy;
    y += gid.x * ncols_y;
    
    float sum = 0.0f;
    
    // 逐块计算
    for (int i = 0; i < blocks_per_row; i++) {
        const float d = x[i].d;
        
        // 反量化并计算点积
        for (int j = 0; j < QK4_0 / QR4_0; j++) {
            const float v0 = dequantize_q4_0(x[i].qs[j], 0) * d;
            const float v1 = dequantize_q4_0(x[i].qs[j], 4) * d;
            
            sum += v0 * y[i * QK4_0 + j * QR4_0 + 0];
            sum += v1 * y[i * QK4_0 + j * QR4_0 + 1];
        }
    }
    
    // 写回结果
    dst[gid.y * nrows_y + gid.x] = sum;
}
```

## 4. Metal 性能优化

### 4.1 线程组分块优化

```metal
// 优化的线程组分块
kernel void optimized_matmul(
    device const float * A [[buffer(0)]],
    device const float * B [[buffer(1)]],
    device float * C [[buffer(2)]],
    constant int & M [[buffer(3)]],
    constant int & N [[buffer(4)]],
    constant int & K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint gidx [[threadgroup_index_in_grid]]
) {
    // 优化的块大小
    constexpr uint BM = 32;
    constexpr uint BN = 32;
    constexpr uint BK = 8;
    
    // 共享内存
    threadgroup float As[BM][BK];
    threadgroup float Bs[BK][BN];
    
    // 计算全局线程 ID
    uint row = gid.y * BM + lid.y;
    uint col = gid.x * BN + lid.x;
    
    float sum = 0.0f;
    
    // 分块循环
    for (uint k = 0; k < K / BK; k++) {
        // 加载 A 块（优化内存访问模式）
        if (row < M && k * BK + lid.x < K) {
            As[lid.y][lid.x] = A[row * K + k * BK + lid.x];
        } else {
            As[lid.y][lid.x] = 0.0f;
        }
        
        // 加载 B 块
        if (k * BK + lid.y < K && col < N) {
            Bs[lid.y][lid.x] = B[(k * BK + lid.y) * N + col];
        } else {
            Bs[lid.y][lid.x] = 0.0f;
        }
        
        // 同步
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 计算
        for (uint i = 0; i < BK; i++) {
            sum += As[lid.y][i] * Bs[i][lid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 4.2 内存访问优化

```metal
// 优化的内存访问模式
kernel void optimized_memory_access(
    device const float * input [[buffer(0)]],
    device float * output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    // 使用 SIMD 组函数优化
    constexpr uint SIMD_SIZE = 32;
    uint simd_id = gid % SIMD_SIZE;
    uint simd_group_id = gid / SIMD_SIZE;
    
    // 合并内存访问
    float value = input[gid];
    
    // SIMD 组内归约
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        value += simd_shuffle_down(value, offset);
    }
    
    // 写回结果
    if (simd_id == 0) {
        output[simd_group_id] = value;
    }
}
```

### 4.3 使用 AMX 加速（Apple Silicon）

```metal
// 使用 Apple 矩阵扩展 (AMX)
#include <metal_amx>

kernel void amx_matmul(
    device const half * A [[buffer(0)]],
    device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // 使用 AMX 矩阵乘累加
    constexpr uint M = 16;
    constexpr uint N = 16;
    constexpr uint K = 16;
    
    // 加载矩阵块
    simdgroup_half8x8 a_block;
    simdgroup_half8x8 b_block;
    simdgroup_float8x8 c_block;
    
    // 执行矩阵乘
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 使用 AMX 指令
    for (uint k = 0; k < K / 8; k++) {
        // 加载 A 块
        a_block[0] = A[gid.y * K + k * 8 + 0];
        // ... 加载其他元素
        
        // 加载 B 块
        b_block[0] = B[(k * 8 + 0) * N + gid.x];
        // ... 加载其他元素
        
        // 矩阵乘累加
        c_block = simdgroup_multiply_accumulate(a_block, b_block, c_block);
    }
    
    // 存储结果
    C[gid.y * N + gid.x] = c_block[0];
}
```

## 5. Metal 后端调试

### 5.1 Xcode GPU 调试

```cpp
// 启用 Metal 调试
void enable_metal_debug() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    
    // 启用调试
    [device setSupportsDynamicLibrary:YES];
    
    // 设置调试标签
    MTLDebugSettings *debug_settings = 
        [[MTLDebugSettings alloc] init];
    [debug_settings setDefaultLabelGenerator:
     [MTLDefaultLabelGenerator new]];
}

// 添加调试标记
void add_debug_marker(id<MTLCommandBuffer> command_buffer, 
                     const char * label) {
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
        NSLog(@"Completed: %s", label);
    }];
}
```

### 5.2 性能分析

```cpp
// Metal 性能分析
void profile_metal_kernel(struct ggml_metal_context * ctx) {
    // 1. 创建命令缓冲区
    id<MTLCommandBuffer> command_buffer = 
        [ctx->command_queue commandBuffer];
    
    // 2. 开始 GPU 时间戳
    MTLCounterSampleBufferDescriptor *descriptor = 
        [[MTLCounterSampleBufferDescriptor alloc] init];
    descriptor.counterSet = [ctx->device 
                             newCounterSetWithName:MTLCounterSetTimestamp];
    
    // 3. 采样时间戳
    id<MTLCounterSampleBuffer> sample_buffer = 
        [ctx->device newCounterSampleBufferWithDescriptor:descriptor
                                            sampleCount:2
                                                  error:nil];
    
    // 4. 记录开始时间
    [command_buffer encodeSampleBuffer:sample_buffer
                         sampleIndex:0];
    
    // 5. 执行内核
    // ...
    
    // 6. 记录结束时间
    [command_buffer encodeSampleBuffer:sample_buffer
                         sampleIndex:1];
    
    // 7. 读取结果
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
        const MTLCounterResult *results = 
            (const MTLCounterResult *)[sample_buffer resolveCounterRange];
        
        uint64_t start = results[0].timestamp;
        uint64_t end = results[1].timestamp;
        
        NSLog(@"Kernel execution time: %llu ns", end - start);
    }];
    
    [command_buffer commit];
}
```

## 6. 实际案例

### 6.1 Flash Attention Metal 实现

```metal
// 文件：flash_attention.metal

kernel void flash_attention(
    device const half * Q [[buffer(0)]],
    device const half * K [[buffer(1)]],
    device const half * V [[buffer(2)]],
    device float * O [[buffer(3)]],
    constant int & B [[buffer(4)]],
    constant int & H [[buffer(5)]],
    constant int & T [[buffer(6)]],
    constant int & D [[buffer(7)]],
    constant float & scale [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // 每个线程处理一个 head 和一个 query 位置
    uint b = gid.z / H;  // batch
    uint h = gid.z % H;  // head
    uint i = gid.y;      // query position
    uint o_idx = gid.x;  // output dimension
    
    // 加载 Q
    half q = Q[((b * H + h) * T + i) * D + o_idx];
    
    // 在线 Softmax 状态
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc = 0.0f;
    
    // 遍历所有 key
    for (uint j = 0; j < T; j++) {
        // 加载 K
        half k = K[((b * H + h) * T + j) * D + o_idx];
        
        // 计算 QK^T
        float qk = (float)q * (float)k * scale;
        
        // 在线 Softmax
        float new_max = fmax(max_val, qk);
        float new_sum = sum_exp * exp(max_val - new_max) + exp(qk - new_max);
        
        // 更新累加器
        half v = V[((b * H + h) * T + j) * D + o_idx];
        acc = (acc * sum_exp * exp(max_val - new_max) + exp(qk - new_max) * (float)v) / new_sum;
        
        max_val = new_max;
        sum_exp = new_sum;
    }
    
    // 存储结果
    O[((b * H + h) * T + i) * D + o_idx] = acc;
}
```

### 6.2 量化推理优化

```metal
// 文件：quantized_inference.metal

// INT4 量化矩阵乘法
kernel void matmul_int4(
    device const uint8_t * W_q [[buffer(0)]],
    device const float * W_s [[buffer(1)]],
    device const float * x [[buffer(2)]],
    device float * y [[buffer(3)]],
    constant int & M [[buffer(4)]],
    constant int & N [[buffer(5)]],
    constant int & K [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= M || gid.y >= N) return;
    
    float sum = 0.0f;
    
    // 每 8 个元素一组（打包在 4 字节中）
    for (int k = 0; k < K / 8; k++) {
        // 加载量化权重（4 字节包含 8 个 INT4 值）
        uint8_t packed = W_q[gid.x * (K / 8) + k];
        
        // 加载 scale
        float scale = W_s[gid.x * (K / 32) + k / 4];
        
        // 加载激活值
        float x_vals[8];
        for (int i = 0; i < 8; i++) {
            x_vals[i] = x[(k * 8 + i) * N + gid.y];
        }
        
        // 解包并计算
        for (int i = 0; i < 8; i++) {
            int4 value = (packed >> (i * 4)) & 0x0F;
            sum += (value - 8) * scale * x_vals[i];
        }
    }
    
    // 存储结果
    y[gid.x * N + gid.y] = sum;
}
```

## 7. 最佳实践

### 7.1 内存管理

```cpp
// 优化 Metal 内存管理
void optimize_metal_memory(id<MTLDevice> device) {
    // 1. 使用共享内存（统一内存架构）
    MTLResourceOptions options = MTLResourceStorageModeShared;
    id<MTLBuffer> buffer = [device newBufferWithLength:size
                                               options:options];
    
    // 2. 避免不必要的拷贝
    // 使用 buffer 的 offset 参数
    
    // 3. 重用缓冲区
    // 使用缓冲区池
    
    // 4. 使用 purgeable 内存
    MTLResourceOptions purgeable_options = 
        MTLResourceStorageModePrivate | MTLResourceCPUCacheModeDefaultCache;
}
```

### 7.2 批处理优化

```cpp
// Metal 批处理优化
void batch_optimization(struct ggml_metal_context * ctx,
                       struct ggml_tensor ** tensors,
                       int num_tensors) {
    // 1. 合并多个计算到单个命令缓冲区
    id<MTLCommandBuffer> command_buffer = 
        [ctx->command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = 
        [command_buffer computeCommandEncoder];
    
    // 2. 批处理所有张量
    for (int i = 0; i < num_tensors; i++) {
        // 设置参数
        [encoder setBuffer:tensors[i]->buffer 
                    offset:0 
                   atIndex:0];
        
        // 分派计算
        [encoder dispatchThreads:grid_size 
         threadsPerThreadgroup:threadgroup_size];
    }
    
    // 3. 单次提交
    [encoder endEncoding];
    [command_buffer commit];
}
```

## 练习

1. 实现一个简单的 Metal 向量加法内核
2. 将 CUDA 矩阵乘法转换为 Metal 版本
3. 实现 Q4_0 量化的 Metal 内核
4. 使用 Xcode GPU Frame Capture 调试 Metal 内核

## 参考资料

- [Metal Documentation](https://developer.apple.com/metal/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [GGML Metal Source](https://github.com/ggerganov/llama.cpp/blob/master/ggml-metal.m)
