# CUDA 调试技术

## 概述

CUDA 调试是开发高性能 GPU 程序的关键环节。本章介绍 CUDA 调试工具、常见错误类型和调试最佳实践。

## 1. CUDA 错误检查基础

### 1.1 错误检查宏

```cpp
// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// 内核错误检查
#define CHECK_KERNEL_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel execution error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// 使用示例
__global__ void my_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

void safe_kernel_launch() {
    float *d_data;
    int n = 1024;
    
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
    
    // 启动内核
    my_kernel<<<128, 256>>>(d_data, n);
    CHECK_KERNEL_ERROR();  // 检查内核错误
    
    CUDA_CHECK(cudaFree(d_data));
}
```

### 1.2 常见错误类型

```cpp
// 1. 内存越界
__global__ void out_of_bounds_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 错误：没有边界检查
    data[idx] *= 2.0f;  // 可能越界
}

// 正确版本
__global__ void safe_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {  // 边界检查
        data[idx] *= 2.0f;
    }
}

// 2. 未检查内存分配
void unsafe_malloc() {
    float *d_data;
    size_t size = 1024 * 1024 * 1024 * 100;  // 100GB，超出显存
    cudaMalloc(&d_data, size);  // 失败但没检查
    // 使用 d_data 会导致错误
}

// 3. 同步错误
void unsafe_async() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    float *h_data, *d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    
    // 异步拷贝后没有同步就使用
    cudaMemcpyAsync(d_data, h_data, 1024 * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
    
    // 错误：没有等待异步拷贝完成
    my_kernel<<<128, 256, 0, stream>>>(d_data, 1024);
    
    // 应该：cudaStreamSynchronize(stream);
}
```

## 2. cuda-memcheck 使用

### 2.1 基本使用

```bash
# 基本内存检查
cuda-memcheck ./your_program

# 详细内存检查
cuda-memcheck --tool memcheck --leak-check full ./your_program

# 输出到文件
cuda-memcheck --log-file memcheck.log ./your_program

# 只检查特定内核
cuda-memcheck --kernel-name "my_kernel" ./your_program
```

### 2.2 竞态条件检测

```cpp
// 示例：存在竞态条件的内核
__global__ void race_condition_kernel(float *data, int n) {
    int idx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 错误：多个线程同时写入同一位置
    if (tid < n) {
        int target_idx = tid % 10;  // 多个线程映射到同一索引
        data[target_idx] += 1.0f;  // 竞态条件
    }
}

// 使用 racecheck 检测
// cuda-memcheck --tool racecheck ./program

// 输出示例：
// ========= Invalid __shared__ atomic operation
// =========     at 0x1234 in race_condition_kernel
// =========     Address 0x5678 is shared by different threads
```

### 2.3 未初始化内存检测

```cpp
// 示例：使用未初始化内存
__global__ void uninitialized_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float shared_data[256];
    
    // 错误：使用未初始化的共享内存
    float value = shared_data[idx] * 2.0f;  // 未初始化
    data[idx] = value;
}

// 使用 initcheck 检测
// cuda-memcheck --tool initcheck ./program
```

## 3. Compute Sanitizer

Compute Sanitizer 是新一代调试工具，提供更强大的检测能力。

### 3.1 基本使用

```bash
# 基本使用
compute-sanitizer ./your_program

# 内存检查
compute-sanitizer --tool memcheck ./your_program

# 竞态条件检查
compute-sanitizer --tool racecheck ./your_program

# 非法内存访问检查
compute-sanitizer --tool illegalmemacc ./your_program

# 生成详细报告
compute-sanitizer --tool memcheck --print-level info ./your_program
```

### 3.2 实际调试示例

```cpp
// 示例程序：包含多个错误
__global__ void buggy_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 错误 1：越界访问
    if (idx <= n) {  // 应该是 idx < n
        output[idx] = input[idx] * 2.0f;
    }
    
    // 错误 2：共享内存竞态条件
    __shared__ float shared_data[256];
    shared_data[threadIdx.x] = input[idx];
    __syncthreads();
    
    // 错误 3：未初始化内存使用
    float temp;  // 未初始化
    if (threadIdx.x % 2 == 0) {
        temp = shared_data[threadIdx.x];
    }
    output[idx] += temp;  // 可能使用未初始化的值
}

// 调试步骤：
// 1. 运行 compute-sanitizer
// 2. 分析错误报告
// 3. 逐个修复错误
// 4. 重新运行验证
```

## 4. GDB 调试 CUDA 程序

### 4.1 基本设置

```bash
# 编译时添加调试信息
nvcc -G -g -o program program.cu

# 使用 cuda-gdb 调试
cuda-gdb ./program

# 或者
gdb ./program
(gdb) cuda set device 0
(gdb) run
```

### 4.2 常用调试命令

```bash
# CUDA GDB 常用命令

# 设置断点
(gdb) break my_kernel  # 在内核入口设置断点
(gdb) break file.cu:123  # 在特定行设置断点

# 查看线程信息
(gdb) cuda thread  # 显示当前线程
(gdb) cuda threads all  # 显示所有线程

# 切换线程
(gdb) cuda thread 0  # 切换到线程 0
(gdb) cuda block 1  # 切换到块 1

# 查看变量
(gdb) print variable_name
(gdb) print *array@10  # 打印数组前 10 个元素

# 单步执行
(gdb) step  # 单步执行（进入函数）
(gdb) next  # 单步执行（不进入函数）
(gdb) continue  # 继续执行

# 查看内核启动
(gdb) cuda kernels  # 显示所有内核
(gdb) cuda launch  # 显示内核启动信息
```

### 4.3 实际调试示例

```cpp
// 示例程序：调试内核
__global__ void debug_kernel(float *data, int n, int factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 设置断点在此行
    if (idx < n) {
        float value = data[idx];
        
        // 检查条件
        if (value > 100.0f) {
            value = 100.0f;
        }
        
        // 计算结果
        data[idx] = value * factor;
    }
}

// 调试会话示例：
/*
$ cuda-gdb ./program

(gdb) break debug_kernel
Breakpoint 1 at 0x1234: file kernel.cu, line 10.

(gdb) run
Starting program: ./program

Breakpoint 1, debug_kernel<<<...>>> (data=0x5678, n=1024, factor=2) at kernel.cu:10

(gdb) cuda threads
  Thread 1 (block (0,0,0), thread (0,0,0))
* Thread 2 (block (0,0,0), thread (1,0,0))
  Thread 3 (block (0,0,0), thread (2,0,0))
  ...

(gdb) cuda thread 2
[Switching focus to CUDA thread 2, block (0,0,0), thread (1,0,0)]

(gdb) print idx
$1 = 1

(gdb) print data[idx]
$2 = 50.5

(gdb) step
11        if (value > 100.0f) {

(gdb) print value
$3 = 50.5

(gdb) continue
Continuing.
*/
```

## 5. 调试技巧和最佳实践

### 5.1 调试友好的代码设计

```cpp
// 1. 添加调试输出
__global__ void debuggable_kernel(float *data, int n, bool debug_mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (debug_mode && idx < 10) {
        printf("Thread %d: data[%d] = %f\n", idx, idx, data[idx]);
    }
    
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

// 2. 使用断言
__global__ void safe_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 运行时检查
    assert(idx >= 0 && idx < n);
    assert(data != nullptr);
    
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

// 3. 分阶段调试
void debug_step_by_step() {
    float *d_data;
    int n = 1024;
    
    // 阶段 1：内存分配
    cudaMalloc(&d_data, n * sizeof(float));
    cudaDeviceSynchronize();
    
    // 阶段 2：数据拷贝
    float *h_data = new float[n];
    // ... 初始化 h_data
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    // 阶段 3：内核执行
    my_kernel<<<128, 256>>>(d_data, n);
    cudaDeviceSynchronize();
    
    // 阶段 4：结果拷贝
    cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 清理
    delete[] h_data;
    cudaFree(d_data);
}
```

### 5.2 调试辅助函数

```cpp
// 调试辅助类
class CudaDebugger {
private:
    static bool debug_mode_;
    
public:
    static void enable_debug() {
        debug_mode_ = true;
    }
    
    static void disable_debug() {
        debug_mode_ = false;
    }
    
    static bool is_debug_enabled() {
        return debug_mode_;
    }
    
    // 打印内核配置
    static void print_kernel_config(const char* kernel_name,
                                   dim3 grid, dim3 block) {
        if (!debug_mode_) return;
        
        printf("Kernel: %s\n", kernel_name);
        printf("  Grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
        printf("  Block: (%d, %d, %d)\n", block.x, block.y, block.z);
        printf("  Total threads: %d\n", grid.x * grid.y * grid.z * block.x * block.y * block.z);
    }
    
    // 检查内核错误
    static void check_kernel_error(const char* kernel_name) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel '%s' launch error: %s\n",
                    kernel_name, cudaGetErrorString(err));
            exit(1);
        }
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel '%s' execution error: %s\n",
                    kernel_name, cudaGetErrorString(err));
            exit(1);
        }
    }
    
    // 打印内存信息
    static void print_memory_info() {
        if (!debug_mode_) return;
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        
        printf("GPU Memory Info:\n");
        printf("  Total: %.2f GB\n", total_mem / 1e9);
        printf("  Free:  %.2f GB\n", free_mem / 1e9);
        printf("  Used:  %.2f GB\n", (total_mem - free_mem) / 1e9);
    }
};

bool CudaDebugger::debug_mode_ = false;

// 使用示例
__global__ void my_kernel(float *data, int n) {
    CudaDebugger::print_kernel_config("my_kernel", gridDim, blockDim);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

void debug_kernel_launch() {
    CudaDebugger::enable_debug();
    CudaDebugger::print_memory_info();
    
    float *d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    
    my_kernel<<<128, 256>>>(d_data, 1024);
    CudaDebugger::check_kernel_error("my_kernel");
    
    cudaFree(d_data);
}
```

### 5.3 单元测试和验证

```cpp
// 简单的 CUDA 单元测试框架
template<typename T>
class CudaTest {
private:
    int passed_;
    int failed_;
    
public:
    CudaTest() : passed_(0), failed_(0) {}
    
    void assert_equals(T *d_result, T *expected, int n, const char* test_name) {
        T *h_result = new T[n];
        cudaMemcpy(h_result, d_result, n * sizeof(T), cudaMemcpyDeviceToHost);
        
        bool passed = true;
        for (int i = 0; i < n; i++) {
            if (h_result[i] != expected[i]) {
                printf("Test '%s' failed at index %d: got %f, expected %f\n",
                       test_name, i, h_result[i], expected[i]);
                passed = false;
                break;
            }
        }
        
        if (passed) {
            printf("Test '%s' PASSED\n", test_name);
            passed_++;
        } else {
            failed_++;
        }
        
        delete[] h_result;
    }
    
    void print_summary() {
        printf("\nTest Summary:\n");
        printf("  Passed: %d\n", passed_);
        printf("  Failed: %d\n", failed_);
        printf("  Total:  %d\n", passed_ + failed_);
    }
};

// 测试示例
void test_vector_addition() {
    CudaTest<float> test;
    
    // 准备测试数据
    int n = 1024;
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_expected = new float[n];
    
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
        h_expected[i] = h_a[i] + h_b[i];
    }
    
    // 分配设备内存
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_result, n * sizeof(float));
    
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // 运行内核
    vector_add<<<128, 256>>>(d_a, d_b, d_result, n);
    cudaDeviceSynchronize();
    
    // 验证结果
    test.assert_equals(d_result, h_expected, n, "vector_addition");
    
    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    delete[] h_a;
    delete[] h_b;
    delete[] h_expected;
    
    test.print_summary();
}
```

## 6. 常见错误和解决方案

### 6.1 错误分类和解决方案

```cpp
/*
 * 1. 内存越界错误
 * 症状：程序崩溃、结果错误
 * 检测：cuda-memcheck, compute-sanitizer
 * 解决：添加边界检查
 */

__global__ void fix_out_of_bounds(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {  // 添加边界检查
        data[idx] *= 2.0f;
    }
}

/*
 * 2. 共享内存竞态条件
 * 症状：结果不一致、随机错误
 * 检测：racecheck 工具
 * 解决：使用同步原语或原子操作
 */

__global__ void fix_race_condition(float *data, int n) {
    __shared__ float shared_data[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 使用原子操作
    if (idx < n) {
        atomicAdd(&shared_data[0], data[idx]);
    }
    
    __syncthreads();  // 确保所有线程完成
    
    // 或者使用归约算法
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
}

/*
 * 3. 未初始化内存
 * 症状：结果随机、不稳定
 * 检测：initcheck 工具
 * 解决：初始化所有内存
 */

__global__ void fix_uninitialized(float *data, int n) {
    __shared__ float shared_data[256];
    
    // 初始化共享内存
    shared_data[threadIdx.x] = 0.0f;
    __syncthreads();
    
    // 现在可以安全使用
    if (threadIdx.x < n) {
        shared_data[threadIdx.x] = data[threadIdx.x];
    }
}

/*
 * 4. 同步错误
 * 症状：死锁、结果错误
 * 检测：仔细审查代码
 * 解决：正确使用同步原语
 */

__global__ void fix_sync_error(float *data, int n) {
    __shared__ float shared_data[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        shared_data[threadIdx.x] = data[idx];
    }
    
    __syncthreads();  // 确保所有线程完成加载
    
    // 现在可以安全使用共享数据
    if (threadIdx.x < n) {
        data[idx] = shared_data[threadIdx.x] * 2.0f;
    }
}
```

## 7. 调试检查清单

```
调试检查清单：

□ 1. 内存分配检查
  - 检查 cudaMalloc 返回值
  - 验证分配大小是否正确
  - 检查显存是否充足

□ 2. 内核启动检查
  - 验证网格和块尺寸
  - 检查内核参数
  - 确认内核启动后无错误

□ 3. 内存访问检查
  - 添加边界检查
  - 验证内存对齐
  - 检查共享内存使用

□ 4. 同步检查
  - 正确使用 __syncthreads()
  - 验证流同步
  - 检查事件同步

□ 5. 错误处理检查
  - 所有 CUDA API 调用都检查错误
  - 内核启动后检查错误
  - 使用 cuda-memcheck 验证

□ 6. 性能检查
  - 使用 Nsight Compute 分析
  - 检查内存访问模式
  - 验证占用率

□ 7. 边界条件检查
  - 测试空输入
  - 测试最大尺寸
  - 测试非对齐尺寸
```

## 练习

1. 编写一个包含故意错误的 CUDA 内核，使用 cuda-memcheck 检测并修复
2. 实现一个调试辅助类，简化 CUDA 程序的调试
3. 使用 cuda-gdb 调试一个多线程 CUDA 程序
4. 为现有的 CUDA 内核编写单元测试

## 参考资料

- [CUDA-MEMCHECK User Guide](https://docs.nvidia.com/cuda/cuda-memcheck/)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/)
- [CUDA-GDB Documentation](https://docs.nvidia.com/cuda/cuda-gdb/)
- [CUDA Best Practices Guide - Debugging](https://docs.nvidia.com/cuda/cuda-c-best-practices/index.html#debugging)
