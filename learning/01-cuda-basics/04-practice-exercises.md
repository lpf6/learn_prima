# 1.4 实践练习

本节提供一系列练习，帮助你巩固 CUDA 基础知识。

## 练习 1：向量加法

### 任务
实现一个 CUDA 内核，将两个向量相加。

### 参考代码

```cpp
// vector_add.cu

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add_kernel(const float *a, const float *b, float *c, int n) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查边界
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void vector_add(const float *a, const float *b, float *c, int n) {
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);

    // 分配设备内存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 复制数据到设备
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 配置内核参数
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    // 启动内核
    vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

    // 复制结果回主机
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // 释放内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

### 思考题
1. 为什么需要边界检查？
2. 如果 n 不是 block_size 的整数倍会发生什么？

## 练习 2：矩阵转置

### 任务
使用共享内存优化矩阵转置。

### 参考代码

```cpp
// matrix_transpose.cu

#define TILE_SIZE 32

__global__ void matrix_transpose_naive(const float *input, float *output,
                                        int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        output[col * height + row] = input[row * width + col];
    }
}

__global__ void matrix_transpose_shared(const float *input, float *output,
                                         int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank conflict

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 加载数据到共享内存
    if (col < width && row < height) {
        tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }
    __syncthreads();

    // 计算转置后的位置
    col = blockIdx.y * TILE_SIZE + threadIdx.x;
    row = blockIdx.x * TILE_SIZE + threadIdx.y;

    // 写回全局内存
    if (col < height && row < width) {
        output[row * height + col] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### 思考题
1. 为什么共享内存声明为 `[TILE_SIZE][TILE_SIZE + 1]`？
2. 比较朴素实现和共享内存实现的性能差异。

## 练习 3：并行归约

### 任务
实现并行归约求和。

### 参考代码

```cpp
// reduce.cu

__global__ void reduce_kernel_v1(const float *input, float *output, int n) {
    __shared__ float partial_sum[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 加载数据到共享内存
    partial_sum[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }

    // 写回结果
    if (tid == 0) {
        output[blockIdx.x] = partial_sum[0];
    }
}

// 使用 Warp 归约优化
__global__ void reduce_kernel_v2(const float *input, float *output, int n) {
    __shared__ float partial_sum[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    partial_sum[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Warp 内归约
    float val = partial_sum[tid];
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    }

    // 只有每个 Warp 的第一个线程有正确结果
    if ((tid & 31) == 0) {
        partial_sum[tid / 32] = val;
    }
    __syncthreads();

    // 最后一个 Warp 归约
    if (tid < blockDim.x / 32) {
        val = partial_sum[tid];
        for (int mask = 4; mask > 0; mask >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, mask, 32);
        }
        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}
```

### 思考题
1. 为什么 Warp 归约比共享内存归约更快？
2. 如何处理任意大小的输入？

## 练习 4：阅读 Prima.cpp 代码

### 任务
阅读并理解以下文件：

1. **ggml/src/ggml-cuda/common.cuh**
   - 找到 `warp_reduce_sum` 函数
   - 理解 `ggml_cuda_dp4a` 函数的作用

2. **ggml/src/ggml-cuda/norm.cu**
   - 理解 RMSNorm 的实现
   - 找到线程索引计算
   - 找到同步点

3. **ggml/src/ggml-cuda/softmax.cu**
   - 理解 Softmax 的数值稳定性处理
   - 找到 Warp 归约的使用

### 参考答案框架

```cpp
// common.cuh 中的 dp4a 函数
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
    // dp4a = dot product of 4 bytes, accumulate
    // 将两个 32 位整数各分成 4 个 8 位整数，计算点积并累加

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    // AMD GPU 实现
#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    return __dp4a(a, b, c);  // 硬件指令
#else
    // 软件实现
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}
```

## 练习 5：调试 CUDA 代码

### 常见错误检查

```cpp
// CUDA 错误检查宏
#define CUDA_CHECK(call)                                        \
    do {                                                        \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                            \
        }                                                       \
    } while (0)

// 内核错误检查
#define CUDA_KERNEL_CHECK()                                     \
    do {                                                        \
        cudaError_t err = cudaGetLastError();                   \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                            \
        }                                                       \
        CUDA_CHECK(cudaDeviceSynchronize());                    \
    } while (0)
```

### 使用 cuda-memcheck

```bash
# 编译时添加 -G 标志生成调试信息
nvcc -G -o my_program my_program.cu

# 运行内存检查
cuda-memcheck ./my_program

# 更详细的检查
compute-sanitizer ./my_program
```

## 练习 6：性能测量

### 使用 CUDA 事件计时

```cpp
void measure_performance() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 开始计时
    cudaEventRecord(start);

    // 执行内核
    my_kernel<<<grid, block>>>(...);

    // 结束计时
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel time: %.3f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
```

## 总结

完成以上练习后，你应该能够：

1. 编写基本的 CUDA 内核
2. 理解线程索引计算
3. 使用共享内存优化性能
4. 使用 Warp 级原语
5. 阅读和理解 Prima.cpp 中的 CUDA 代码

## 下一步

完成本阶段后，请继续学习 [第二阶段：GPU 架构](../02-gpu-architecture/README.md)。
