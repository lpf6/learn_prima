# 1.3 同步机制

## 为什么需要同步？

在并行计算中，多个线程可能需要：

1. **等待数据**：确保其他线程完成数据写入
2. **协调执行**：确保某些操作按顺序执行
3. **避免竞争**：防止多个线程同时修改同一数据

## CUDA 同步层次

```
┌─────────────────────────────────────────────────────┐
│              cudaDeviceSynchronize()                 │
│              (主机等待设备完成所有工作)               │
├─────────────────────────────────────────────────────┤
│              cudaStreamSynchronize()                 │
│              (主机等待特定流完成)                     │
├─────────────────────────────────────────────────────┤
│              cudaEventSynchronize()                  │
│              (主机等待特定事件)                       │
├─────────────────────────────────────────────────────┤
│              __syncthreads()                         │
│              (Block 内所有线程同步)                   │
├─────────────────────────────────────────────────────┤
│              __syncwarp()                            │
│              (Warp 内线程同步)                        │
└─────────────────────────────────────────────────────┘
```

## Block 级同步：__syncthreads()

### 基本用法

```cpp
__global__ void example_kernel(float *data, int n) {
    __shared__ float shared_data[256];

    int idx = threadIdx.x;

    // 阶段1：加载数据
    shared_data[idx] = data[idx];

    // 同步：确保所有线程完成加载
    __syncthreads();

    // 阶段2：处理数据
    float result = shared_data[n - 1 - idx];

    // 同步：确保所有线程完成处理
    __syncthreads();

    // 阶段3：写回结果
    data[idx] = result;
}
```

### 注意事项

1. **所有线程必须执行**：如果某些线程跳过 `__syncthreads()`，会导致死锁
2. **条件分支中使用要小心**：

```cpp
// 危险！可能导致死锁
if (threadIdx.x < 128) {
    __syncthreads();  // 只有部分线程执行
}

// 正确做法
if (threadIdx.x < 128) {
    // 做一些工作
}
__syncthreads();  // 所有线程都执行
```

## Warp 级同步

### __syncwarp()

```cpp
__global__ void warp_sync_example() {
    int lane_id = threadIdx.x & 31;  // 0-31

    // Warp 级同步
    __syncwarp();

    // 只同步特定线程
    unsigned mask = 0x0000FFFF;  // 只同步前16个线程
    __syncwarp(mask);
}
```

### Warp 级原语

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

// Warp 归约求和（float 版本）
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

// Warp 归约求最大值
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}
```

### Shuffle 指令详解

```cpp
// __shfl_xor_sync：XOR 方式交换数据
// mask: 掩码，指定参与同步的线程
// var: 要交换的变量
// laneMask: XOR 掩码
// width: 通常是 32

int result = __shfl_xor_sync(mask, var, laneMask, width);

// 示例：
// 线程 0 (lane_id=0) 与线程 1 (lane_id=0^1=1) 交换
// 线程 2 (lane_id=2) 与线程 3 (lane_id=2^1=3) 交换
```

## 原子操作

### 基本原子操作

```cpp
// 原子加
int atomicAdd(int *address, int val);

// 原子减
int atomicSub(int *address, int val);

// 原子交换
int atomicExch(int *address, int val);

// 原子比较并交换
int atomicCAS(int *address, int compare, int val);

// 原子最小/最大
int atomicMin(int *address, int val);
int atomicMax(int *address, int val);
```

### 使用示例

```cpp
__global__ void histogram_kernel(int *histogram, int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int bin = data[idx];
        // 原子操作：多个线程可能同时更新同一个 bin
        atomicAdd(&histogram[bin], 1);
    }
}
```

### 原子操作的性能考虑

```cpp
// 高竞争情况：原子操作会成为瓶颈
// 解决方案：使用局部聚合

__global__ void optimized_histogram(int *histogram, int *data, int n) {
    __shared__ int local_hist[256];

    // 初始化局部直方图
    if (threadIdx.x < 256) {
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    // 更新局部直方图
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&local_hist[data[idx]], 1);
    }
    __syncthreads();

    // 合并到全局直方图
    if (threadIdx.x < 256) {
        atomicAdd(&histogram[threadIdx.x], local_hist[threadIdx.x]);
    }
}
```

## 内存屏障

### 显式内存屏障

```cpp
// 线程栅栏：确保内存操作对 Block 内所有线程可见
__threadfence_block();

// 设备栅栏：确保内存操作对设备上所有线程可见
__threadfence();

// 系统栅栏：确保内存操作对主机和设备都可见
__threadfence_system();
```

### 使用场景

```cpp
__global__ void producer_consumer(float *data, int *flag, int n) {
    int idx = threadIdx.x;

    if (idx == 0) {
        // 生产者：写入数据
        for (int i = 0; i < n; i++) {
            data[i] = compute(i);
        }
        // 确保数据写入完成
        __threadfence();
        // 设置标志
        *flag = 1;
    }

    if (idx == 1) {
        // 消费者：等待数据
        while (*flag == 0) {
            // 自旋等待
        }
        // 现在可以安全读取数据
        float val = data[0];
    }
}
```

## Prima.cpp 中的同步示例

### CUDA 流管理

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

struct ggml_backend_cuda_context {
    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };

    cudaStream_t stream(int device, int stream_idx) {
        if (streams[device][stream_idx] == nullptr) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream_idx],
                                                  cudaStreamNonBlocking));
        }
        return streams[device][stream_idx];
    }
};
```

### 事件同步

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

struct ggml_tensor_extra_gpu {
    void * data_device[GGML_CUDA_MAX_DEVICES];
    cudaEvent_t events[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS];
};

// 使用事件进行跨流同步
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event, 0);
```

## 同步最佳实践

### 1. 最小化同步次数

```cpp
// 不好：频繁同步
for (int i = 0; i < n; i++) {
    compute_step_1(data, i);
    __syncthreads();
    compute_step_2(data, i);
    __syncthreads();
}

// 更好：批量处理
compute_all_step_1(data, n);
__syncthreads();
compute_all_step_2(data, n);
```

### 2. 使用 Warp 级同步替代 Block 级同步

```cpp
// 如果只需要 Warp 内同步，使用 __syncwarp 或 shuffle
// 比 __syncthreads 更快
```

### 3. 避免死锁

```cpp
// 死锁示例
if (threadIdx.x < 16) {
    __syncthreads();  // 只有部分线程到达
    // 永远等待其他线程...
}

// 正确做法：确保所有线程都执行同步点
if (threadIdx.x < 16) {
    // 工作
}
__syncthreads();  // 所有线程都到达
```

## 练习

1. 实现一个使用共享内存的并行归约
2. 使用原子操作实现一个简单的计数器
3. 阅读 `ggml/src/ggml-cuda/softmax.cu`，理解其中的同步使用

## 下一步

完成本节后，请继续学习 [实践练习](04-practice-exercises.md)。
