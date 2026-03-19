# CUDA 性能分析工具

## 概述

CUDA 提供了强大的性能分析工具，帮助开发者识别性能瓶颈、优化内核性能。本章介绍主要的性能分析工具和使用方法。

## 1. Nsight Compute (NCU)

Nsight Compute 是 NVIDIA 提供的专业 CUDA 内核性能分析器，可以详细分析单个内核的性能特征。

### 1.1 基本使用

```bash
# 基本分析
ncu --target-processes all ./your_program

# 分析特定内核
ncu --kernel-name "my_kernel" ./your_program

# 采集特定指标
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed, dram__throughput.avg.bytes_per_second ./your_program

# 保存结果到文件
ncu --export my_profile ./your_program
```

### 1.2 关键性能指标

```cpp
// 常见性能指标含义

/* 
 * SM Throughput (sm__throughput)
 * - 衡量 SM 利用率
 * - 理想值接近 100%
 */

/*
 * Memory Throughput (dram__throughput)
 * - 衡量显存带宽使用率
 * - 计算带宽绑定还是计算绑定
 */

/*
 * Instruction Mix
 * - 计算指令 vs 内存指令比例
 * - 优化重点方向
 */

/*
 * Warp Execution Efficiency
 * - 活跃线程 vs 总线程比例
 * - 指示分支发散问题
 */
```

### 1.3 实际分析示例

```cpp
// 示例内核：低效版本
__global__ void inefficient_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 未对齐访问
        data[idx * 3] *= 2.0f;  // Strided access
    }
}

// 高效版本
__global__ void efficient_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 对齐访问
        data[idx] *= 2.0f;  // Coalesced access
    }
}

// 使用 Nsight Compute 分析
/*
$ ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.bytes_per_second --kernel-name "inefficient_kernel" ./test

结果分析：
- SM Throughput: 60% (理想应接近 100%)
- DRAM Throughput: 低效访问导致带宽利用率低
- 需要优化内存访问模式
*/
```

### 1.4 高级分析

```bash
# 分析所有可用指标
ncu --set full ./your_program

# 分析特定阶段
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./your_program

# 对比两个版本
ncu --comparison-folder baseline --comparison-folder optimized ./your_program
```

## 2. Nsight Systems (NSYS)

Nsight Systems 提供应用程序的全景视图，显示 CPU 和 GPU 的活动、内存传输和同步操作。

### 2.1 基本使用

```bash
# 基本分析
nsys profile --trace=cuda,nvtx ./your_program

# 限制分析时间
nsys profile --duration=10 --trace=cuda,nvtx ./your_program

# 生成报告
nsys profile --export=sqlite,none --output=report ./your_program

# 分析特定时间段
nsys profile --start-time=5000 --duration=2000 --trace=cuda,nvtx ./your_program
```

### 2.2 分析报告解读

```
Nsight Systems 报告包含：

┌─────────────────────────────────────────────────────────────┐
│ CPU Timeline                                                │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Host Code → Kernel Launch → Memory Copy → Sync          │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ GPU Timeline                                                │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Stream 0: [Kernel A]─────[Kernel B]                     │ │
│ │ Stream 1:      [Kernel C]─────────[Kernel D]            │ │
│ │ Stream 2: [Memory]─[Kernel E]                           │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Memory Transfers                                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ H2D: 100MB in 10ms → Bandwidth: 10GB/s                │ │
│ │ D2H: 50MB in 5ms  → Bandwidth: 10GB/s                 │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 性能问题识别

```cpp
// 使用 NVTX 标记区域以便分析
#include <nvtx3/nvtx3.hpp>

void profiled_function() {
    nvtxRangePush("Data Preprocessing");
    preprocess_data();
    nvtxRangePop();  // 结束标记
    
    nvtxRangePush("Kernel Execution");
    my_kernel<<<grid, block>>>(args);
    cudaDeviceSynchronize();
    nvtxRangePop();
    
    nvtxRangePush("Postprocessing");
    postprocess_results();
    nvtxRangePop();
}

// 分析命令
// nsys profile --trace=cuda,nvtx ./program_with_nvtx
```

## 3. 内存检查工具

### 3.1 cuda-memcheck

cuda-memcheck 用于检测内存错误。

```bash
# 基本内存检查
cuda-memcheck ./your_program

# 详细检查
cuda-memcheck --tool memcheck --leak-check full ./your_program

# 检查特定错误
cuda-memcheck --tool racecheck ./your_program  # 检查竞态条件
cuda-memcheck --tool initcheck ./your_program  # 检查未初始化内存
```

### 3.2 Compute Sanitizer

Compute Sanitizer 是新一代内存调试工具。

```bash
# 基本使用
compute-sanitizer ./your_program

# 检查内存错误
compute-sanitizer --tool memcheck ./your_program

# 检查竞赛条件
compute-sanitizer --tool racecheck ./your_program

# 检查非法内存访问
compute-sanitizer --tool illegalmemacc ./your_program

# 生成详细报告
compute-sanitizer --tool memcheck --print-level info ./your_program
```

## 4. 编程接口性能分析

### 4.1 在代码中集成性能分析

```cpp
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

class Profiler {
private:
    cudaEvent_t start_, stop_;
    
public:
    Profiler() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~Profiler() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        cudaEventRecord(start_);
    }
    
    float stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start_, stop_);
        return elapsed_time;
    }
};

// 使用示例
void benchmark_kernel() {
    Profiler profiler;
    
    profiler.start();
    my_kernel<<<grid, block>>>(args);
    float time_ms = profiler.stop();
    
    printf("Kernel execution time: %.2f ms\n", time_ms);
}
```

### 4.2 自定义性能指标

```cpp
// 收集自定义指标
struct PerformanceMetrics {
    float execution_time;
    size_t memory_transferred;
    float bandwidth;
    int kernel_calls;
};

class MetricsCollector {
private:
    PerformanceMetrics metrics_{};
    cudaEvent_t start_, stop_;
    
public:
    MetricsCollector() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~MetricsCollector() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start_kernel(size_t bytes_transferred = 0) {
        cudaEventRecord(start_);
        metrics_.memory_transferred += bytes_transferred;
        metrics_.kernel_calls++;
    }
    
    void stop_kernel() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start_, stop_);
        metrics_.execution_time += elapsed_ms;
        
        if (elapsed_ms > 0) {
            float bandwidth_gb = (metrics_.memory_transferred / 1e9) / (elapsed_ms / 1000.0);
            metrics_.bandwidth = bandwidth_gb;
        }
    }
    
    PerformanceMetrics get_metrics() const {
        return metrics_;
    }
    
    void print_summary() const {
        printf("Performance Summary:\n");
        printf("  Execution Time: %.2f ms\n", metrics_.execution_time);
        printf("  Memory Transferred: %.2f GB\n", metrics_.memory_transferred / 1e9);
        printf("  Avg Bandwidth: %.2f GB/s\n", metrics_.bandwidth);
        printf("  Kernel Calls: %d\n", metrics_.kernel_calls);
    }
};
```

## 5. 性能优化策略

### 5.1 识别性能瓶颈

```cpp
// 性能瓶颈分类
enum class BottleneckType {
    COMPUTE_BOUND,    // 计算瓶颈
    MEMORY_BOUND,     // 内存瓶颈
    BANDWIDTH_BOUND,  // 带宽瓶颈
    LATENCY_BOUND   // 延迟瓶颈
};

BottleneckType analyze_bottleneck(float compute_intensity, float bandwidth_utilization) {
    /*
     * 计算强度 = FLOPs / Byte
     * 如果计算强度高且带宽利用率低 → 计算瓶颈
     * 如果计算强度低且带宽利用率高 → 内存瓶颈
     */
    
    const float arith_intensity_threshold = 1.0;  // FLOPs/Byte
    const float bandwidth_threshold = 0.7;        // 70% 带宽利用率
    
    if (compute_intensity > arith_intensity_threshold && 
        bandwidth_utilization < bandwidth_threshold) {
        return BottleneckType::COMPUTE_BOUND;
    } else if (compute_intensity < arith_intensity_threshold &&
               bandwidth_utilization > bandwidth_threshold) {
        return BottleneckType::MEMORY_BOUND;
    } else if (bandwidth_utilization < bandwidth_threshold) {
        return BottleneckType::BANDWIDTH_BOUND;
    } else {
        return BottleneckType::LATENCY_BOUND;
    }
}
```

### 5.2 优化建议

```cpp
// 根据瓶颈类型提供建议
void provide_optimization_advice(BottleneckType bottleneck) {
    switch (bottleneck) {
        case BottleneckType::COMPUTE_BOUND:
            printf("Optimization advice for compute-bound kernel:\n");
            printf("- Increase arithmetic intensity\n");
            printf("- Use more computational instructions\n");
            printf("- Consider using Tensor Cores if applicable\n");
            printf("- Optimize instruction-level parallelism\n");
            break;
            
        case BottleneckType::MEMORY_BOUND:
            printf("Optimization advice for memory-bound kernel:\n");
            printf("- Improve memory access patterns\n");
            printf("- Use shared memory for frequently accessed data\n");
            printf("- Increase coalescing of memory accesses\n");
            printf("- Consider using texture memory\n");
            break;
            
        case BottleneckType::BANDWIDTH_BOUND:
            printf("Optimization advice for bandwidth-bound kernel:\n");
            printf("- Reduce memory traffic\n");
            printf("- Use compression techniques\n");
            printf("- Increase data reuse\n");
            printf("- Consider using different data layout\n");
            break;
            
        case BottleneckType::LATENCY_BOUND:
            printf("Optimization advice for latency-bound kernel:\n");
            printf("- Increase occupancy\n");
            printf("- Hide latency with more warps\n");
            printf("- Overlap computation with memory access\n");
            printf("- Use asynchronous memory operations\n");
            break;
    }
}
```

## 6. 实际案例分析

### 6.1 矩阵乘法性能分析

```cpp
// 分析不同版本的矩阵乘法
__global__ void naive_matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void optimized_matmul(float *A, float *B, float *C, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < N; t += 16) {
        // 加载到共享内存
        if (row < N && t + tx < N)
            As[ty][tx] = A[row * N + t + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && t + ty < N)
            Bs[ty][tx] = B[(t + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
        
        __syncthreads();
        
        // 计算
        for (int k = 0; k < 16; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N)
        C[row * N + col] = sum;
}

/*
Nsight Compute 对比分析：
Naive version:
- SM Throughput: 20%
- DRAM Read BW: 80 GB/s
- DRAM Write BW: 40 GB/s
- Efficiency: Low due to poor memory coalescing

Optimized version:
- SM Throughput: 85%
- DRAM Read BW: 300 GB/s (using shared memory)
- DRAM Write BW: 20 GB/s
- Efficiency: High due to memory optimization
*/
```

## 7. 性能分析最佳实践

### 7.1 基准测试设置

```cpp
// 设置准确的基准测试
class AccurateBenchmark {
private:
    cudaEvent_t warmup_start_, warmup_stop_;
    cudaEvent_t bench_start_, bench_stop_;
    
public:
    AccurateBenchmark() {
        cudaEventCreate(&warmup_start_);
        cudaEventCreate(&warmup_stop_);
        cudaEventCreate(&bench_start_);
        cudaEventCreate(&bench_stop_);
    }
    
    ~AccurateBenchmark() {
        cudaEventDestroy(warmup_start_);
        cudaEventDestroy(warmup_stop_);
        cudaEventDestroy(bench_start_);
        cudaEventDestroy(bench_stop_);
    }
    
    float benchmark(std::function<void()> kernel_func, int warmup_runs = 3, int bench_runs = 10) {
        // 预热
        for (int i = 0; i < warmup_runs; i++) {
            kernel_func();
        }
        cudaDeviceSynchronize();
        
        // 正式测试
        cudaEventRecord(bench_start_);
        for (int i = 0; i < bench_runs; i++) {
            kernel_func();
        }
        cudaEventRecord(bench_stop_);
        cudaEventSynchronize(bench_stop_);
        
        float total_time;
        cudaEventElapsedTime(&total_time, bench_start_, bench_stop_);
        
        return total_time / bench_runs;  // 平均时间
    }
};
```

### 7.2 多维度性能分析

```cpp
// 分析不同参数下的性能
void parameter_sweep_analysis() {
    std::vector<int> block_sizes = {32, 64, 128, 256, 512};
    std::vector<int> grid_sizes = {128, 256, 512, 1024};
    
    printf("BlockSize\tGridSize\tTime(ms)\tBandwidth(GB/s)\n");
    
    for (int bs : block_sizes) {
        for (int gs : grid_sizes) {
            // 设置网格和块尺寸
            dim3 block_dim(bs);
            dim3 grid_dim(gs);
            
            // 运行内核并测量性能
            AccurateBenchmark bench;
            float time_ms = bench.benchmark([&]() {
                my_kernel<<<grid_dim, block_dim>>>(args);
            });
            
            // 计算带宽
            size_t bytes_processed = calculate_bytes_processed();
            float bandwidth = (bytes_processed / 1e9) / (time_ms / 1000.0);
            
            printf("%d\t\t%d\t\t%.2f\t\t%.2f\n", bs, gs, time_ms, bandwidth);
        }
    }
}
```

## 练习

1. 使用 Nsight Compute 分析一个现有的 CUDA 内核，识别性能瓶颈
2. 实现一个自定义的性能分析器，收集多个指标
3. 使用 Nsight Systems 分析 CPU-GPU 交互模式
4. 编写一个参数扫描程序，找出最优的网格和块尺寸

## 参考资料

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CUDA Performance Optimization](https://developer.download.nvidia.com/assets/cuda/files/reliable_gpu_computing/CUDA_Profiler_Tools.pdf)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices/index.html)
