# CUDA 流和事件

## 概述

CUDA 流（Stream）和事件（Event）是实现并发执行和异步操作的关键机制。它们允许在 GPU 上并行执行多个内核，以及与主机端操作重叠。

## 1. CUDA 流基础

### 1.1 什么是 CUDA 流

CUDA 流是一个执行队列，其中的任务（内核启动、内存拷贝等）按顺序执行，但不同流中的任务可以并发执行。

```
默认流（NULL stream）：
任务 1 → 任务 2 → 任务 3 （串行执行）
  ↓       ↓       ↓
GPU 时间轴

自定义流（并发执行）：
流 1: 任 1 → 任 2 → 任 3
流 2: 任 1 → 任 2 → 任 3  （可与流 1 并发）
流 3: 任 1 → 任 2 → 任 3  （可与其他流并发）
     ↑     ↑     ↑
     GPU 时间轴（并发）
```

### 1.2 创建和销毁流

```cpp
// 创建流
cudaStream_t stream;
cudaStreamCreate(&stream);

// 创建带标志的流
cudaStream_t stream_with_flags;
cudaStreamCreateWithFlags(&stream_with_flags, cudaStreamNonBlocking);

// 使用流启动内核
my_kernel<<<grid_size, block_size, 0, stream>>>(args);

// 等待流完成
cudaStreamSynchronize(stream);

// 销毁流
cudaStreamDestroy(stream);
```

### 1.3 流的属性

```cpp
// 查询流状态
cudaError_t err = cudaStreamQuery(stream);
if (err == cudaSuccess) {
    // 流已完成
} else if (err == cudaErrorNotReady) {
    // 流仍在执行
} else {
    // 错误
}

// 检查流是否为空闲
bool is_idle = (cudaStreamQuery(stream) == cudaSuccess);
```

## 2. 实际应用示例

### 2.1 数据传输与计算重叠

```cpp
// 使用多个流实现 H2D、计算、D2H 重叠
void overlapped_computation(float *h_data, float *d_data, int n) {
    const int num_streams = 4;
    const int chunk_size = n / num_streams;
    
    cudaStream_t streams[num_streams];
    
    // 创建流
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // 分块处理
    for (int i = 0; i < num_streams; i++) {
        float *h_chunk = h_data + i * chunk_size;
        float *d_chunk = d_data + i * chunk_size;
        
        // 异步内存拷贝
        cudaMemcpyAsync(d_chunk, h_chunk, 
                       chunk_size * sizeof(float),
                       cudaMemcpyHostToDevice, 
                       streams[i]);
        
        // 异步内核执行
        process_kernel<<<grid_size, block_size, 0, streams[i]>>>(
            d_chunk, chunk_size);
        
        // 异步内存拷贝回主机
        cudaMemcpyAsync(h_chunk, d_chunk, 
                       chunk_size * sizeof(float),
                       cudaMemcpyDeviceToHost, 
                       streams[i]);
    }
    
    // 等待所有流完成
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}
```

### 2.2 流优先级

```cpp
// 查询流优先级范围
int priority_low, priority_high;
cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);

// 创建不同优先级的流
cudaStream_t high_priority_stream, low_priority_stream;
cudaStreamCreateWithPriority(&high_priority_stream, 
                            cudaStreamNonBlocking, 
                            priority_high);
cudaStreamCreateWithPriority(&low_priority_stream, 
                            cudaStreamNonBlocking, 
                            priority_low);

// 高优先级流中的任务会优先执行
process_critical_kernel<<<grid, block, 0, high_priority_stream>>>(args);
process_normal_kernel<<<grid, block, 0, low_priority_stream>>>(args);
```

## 3. CUDA 事件

### 3.1 事件基础

CUDA 事件用于测量时间、同步流、以及标记流中的特定点。

```cpp
// 创建事件
cudaEvent_t start_event, stop_event;
cudaEventCreate(&start_event);
cudaEventCreate(&stop_event);

// 记录事件
cudaEventRecord(start_event);
my_kernel<<<grid_size, block_size>>>(args);
cudaEventRecord(stop_event);

// 等待事件完成
cudaEventSynchronize(stop_event);

// 计算时间
float elapsed_time;
cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

// 销毁事件
cudaEventDestroy(start_event);
cudaEventDestroy(stop_event);
```

### 3.2 事件在流同步中的应用

```cpp
// 使用事件实现更细粒度的同步
void complex_synchronization_example() {
    cudaStream_t stream1, stream2;
    cudaEvent_t event1, event2;
    
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    
    // 在 stream1 中执行任务 A
    task_A_kernel<<<grid, block, 0, stream1>>>(args);
    cudaEventRecord(event1, stream1);  // 记录任务 A 完成点
    
    // 在 stream2 中执行任务 B，依赖于事件 1
    cudaStreamWaitEvent(stream2, event1, 0);  // 等待 event1
    task_B_kernel<<<grid, block, 0, stream2>>>(args);
    
    // 在 stream1 中执行任务 C，依赖于任务 B
    cudaEventRecord(event2, stream2);  // 记录任务 B 完成点
    cudaStreamWaitEvent(stream1, event2, 0);  // 等待 event2
    task_C_kernel<<<grid, block, 0, stream1>>>(args);
    
    // 等待所有任务完成
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // 清理
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}
```

## 4. 性能优化技巧

### 4.1 合理选择流的数量

```cpp
// 根据硬件特性选择流的数量
int get_optimal_stream_count() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // 基于计算能力和内存带宽
    int compute_capability = prop.major * 10 + prop.minor;
    int max_concurrent_streams = prop.maxThreadsPerMultiProcessor;
    
    // 一般选择 2-8 个流，过多可能导致调度开销
    return min(8, max_concurrent_streams / 4);
}
```

### 4.2 内存拷贝优化

```cpp
// 使用 pinned memory 提高内存拷贝性能
void optimized_transfer() {
    float *h_pinned_data, *d_data;
    int size = 1024 * 1024 * sizeof(float);
    
    // 分配 pinned memory
    cudaMallocHost(&h_pinned_data, size);
    cudaMalloc(&d_data, size);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 异步拷贝性能更好
    cudaMemcpyAsync(d_data, h_pinned_data, size, 
                   cudaMemcpyHostToDevice, stream);
    
    // 清理
    cudaStreamDestroy(stream);
    cudaFreeHost(h_pinned_data);
    cudaFree(d_data);
}
```

## 5. 最佳实践

### 5.1 流的生命周期管理

```cpp
// RAII 风格的流管理
class CudaStreamManager {
private:
    cudaStream_t stream_;
    
public:
    CudaStreamManager() {
        cudaStreamCreate(&stream_);
    }
    
    ~CudaStreamManager() {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
    
    cudaStream_t get() const { return stream_; }
    
    void synchronize() {
        cudaStreamSynchronize(stream_);
    }
};

// 使用示例
{
    CudaStreamManager stream_mgr;
    my_kernel<<<grid, block, 0, stream_mgr.get()>>>(args);
    // 析构函数自动同步和销毁
}
```

### 5.2 错误处理

```cpp
// 带错误检查的流操作
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

void safe_stream_operations() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    my_kernel<<<grid, block, 0, stream>>>(args);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
```

## 6. 实际案例：批处理优化

```cpp
// 使用流优化批处理推理
class BatchProcessor {
private:
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    int num_streams_;
    
public:
    BatchProcessor(int num_streams) : num_streams_(num_streams) {
        streams_.resize(num_streams);
        events_.resize(num_streams);
        
        for (int i = 0; i < num_streams; i++) {
            cudaStreamCreate(&streams_[i]);
            cudaEventCreate(&events_[i]);
        }
    }
    
    ~BatchProcessor() {
        for (int i = 0; i < num_streams_; i++) {
            cudaStreamSynchronize(streams_[i]);
            cudaStreamDestroy(streams_[i]);
            cudaEventDestroy(events_[i]);
        }
    }
    
    void process_batch(float *input, float *output, int batch_size) {
        int samples_per_stream = batch_size / num_streams_;
        
        for (int i = 0; i < num_streams_; i++) {
            float *input_ptr = input + i * samples_per_stream;
            float *output_ptr = output + i * samples_per_stream;
            
            // 异步处理每个子批次
            inference_kernel<<<grid, block, 0, streams_[i]>>>(
                input_ptr, output_ptr, samples_per_stream);
            
            // 记录事件以便同步
            cudaEventRecord(events_[i], streams_[i]);
        }
        
        // 等待所有流完成
        for (int i = 0; i < num_streams_; i++) {
            cudaEventSynchronize(events_[i]);
        }
    }
};
```

## 练习

1. 实现一个使用多个流的矩阵乘法，实现计算与内存传输的重叠
2. 设计一个事件驱动的流水线系统，处理连续的数据流
3. 比较使用不同数量流时的性能差异
4. 实现一个流池管理器，支持动态创建和复用流

## 参考资料

- [CUDA Streams Best Practices](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-streams-best-practices-common-pitfalls/)
- [CUDA Event Programming Guide](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
- [Asynchronous Data Transfers](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/)
