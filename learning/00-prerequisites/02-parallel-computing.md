# 并行计算基础

## 概述

并行计算是高性能计算的核心。本章介绍并行计算的基本概念、定律和编程模型，为后续学习 CUDA 和多线程编程打下基础。

## 1. 并行计算概述

### 1.1 什么是并行计算

并行计算是同时使用多个计算资源解决一个计算问题的方法：

```
串行计算：
任务 1 → 任务 2 → 任务 3 → 任务 4
总时间：4T

并行计算（4 个处理器）：
任务 1 ┐
任务 2 ├─ 同时执行
任务 3 ┘
任务 4 ┐
       └─ 同时执行
总时间：2T
```

### 1.2 并行的层次

```
┌─────────────────────────────────────────────────────────────┐
│  指令级并行 (ILP)                                           │
│  ├── 流水线 (Pipelining)                                    │
│  ├── 超标量 (Superscalar)                                   │
│  └── 乱序执行 (Out-of-order)                                │
├─────────────────────────────────────────────────────────────┤
│  数据级并行 (DLP)                                           │
│  ├── SIMD (单指令多数据)                                    │
│  ├── GPU (图形处理器)                                       │
│  └── 向量处理器                                             │
├─────────────────────────────────────────────────────────────┤
│  任务级并行 (TLP)                                           │
│  ├── 多线程 (Multithreading)                                │
│  ├── 多进程 (Multiprocessing)                               │
│  └── 分布式计算 (Distributed)                               │
└─────────────────────────────────────────────────────────────┘
```

## 2. 并行计算定律

### 2.1 Amdahl 定律

Amdahl 定律描述了并行程序的理论加速比：

```
加速比 = 1 / (S + P/N)

其中:
- S: 串行部分比例 (0-1)
- P: 并行部分比例 (P = 1 - S)
- N: 处理器数量

示例:
假设程序 90% 可并行 (P=0.9, S=0.1)

N=10:  加速比 = 1 / (0.1 + 0.9/10) = 5.3x
N=100: 加速比 = 1 / (0.1 + 0.9/100) = 9.2x
N=∞:   加速比 = 1 / 0.1 = 10x (理论上限)
```

**重要结论**：
- 即使无限增加处理器，加速比也受限于串行部分
- 优化时应优先减少串行部分

### 2.2 Gustafson 定律

Gustafson 定律从不同角度看问题：

```
加速比 = S + P × N

当问题规模随处理器数量增加时：
- 串行部分保持不变
- 并行部分线性增长
- 加速比可以接近线性
```

### 2.3 实际应用

```cpp
// 示例：矩阵乘法
void matrix_multiply_serial(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {           // 串行
        for (int j = 0; j < n; j++) {       // 可并行
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {   // 可并行
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

// 并行版本（伪代码）
void matrix_multiply_parallel(float* A, float* B, float* C, int n, int num_threads) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // ...
        }
    }
}
```

## 3. 并行编程模型

### 3.1 数据并行 vs 任务并行

**数据并行（Data Parallelism）**：
```
相同操作应用于不同数据

处理器 1: [数据块 1] → 操作 A
处理器 2: [数据块 2] → 操作 A
处理器 3: [数据块 3] → 操作 A

典型应用：SIMD、GPU、向量处理
```

**任务并行（Task Parallelism）**：
```
不同操作同时执行

处理器 1: 任务 A → 任务 B → 任务 C
处理器 2: 任务 X → 任务 Y → 任务 Z

典型应用：多线程、多进程、流水线
```

### 3.2 SIMD（单指令多数据）

```
CPU:
标量操作:
  ADD R1, R2, R3  // R1 = R2 + R3

SIMD 操作 (AVX-512, 512-bit):
  VADDPS ZMM1, ZMM2, ZMM3
  // 同时计算 16 个 float32 加法
  // ZMM2[0:15] + ZMM3[0:15] → ZMM1[0:15]

GPU:
SIMT (单指令多线程):
  32 个线程同时执行相同指令
  每个线程处理不同数据
```

### 3.3 共享内存 vs 分布式内存

**共享内存模型**：
```
┌─────────────────────────────────────┐
│           共享内存                   │
│  ┌──────┐ ┌──────┐ ┌──────┐        │
│  │ 线程 1│ │ 线程 2│ │ 线程 3│        │
│  └──────┘ └──────┘ └──────┘        │
└─────────────────────────────────────┘

优点：通信快，编程简单
缺点：扩展性受限
典型：OpenMP、pthread、CUDA 线程块
```

**分布式内存模型**：
```
┌─────────┐   ┌─────────┐   ┌─────────┐
│ 节点 1   │   │ 节点 2   │   │ 节点 3   │
│ 本地内存 │   │ 本地内存 │   │ 本地内存 │
└─────────┘   └─────────┘   └─────────┘
      ↕             ↕             ↕
   网络通信 (MPI、NCCL)
   
优点：扩展性好
缺点：通信开销大，编程复杂
典型：MPI、NCCL
```

## 4. 并行算法设计

### 4.1 分解策略

**1. 数据分解**：
```cpp
// 将数据分成块，每个处理器处理一块
void parallel_reduce(float* data, int n, int num_procs) {
    int chunk_size = n / num_procs;
    int start = proc_id * chunk_size;
    int end = start + chunk_size;
    
    // 每个处理器计算局部和
    float local_sum = 0;
    for (int i = start; i < end; i++) {
        local_sum += data[i];
    }
    
    // 归约所有局部和
    float global_sum = all_reduce_sum(local_sum);
}
```

**2. 任务分解**：
```cpp
// 将任务分成独立的子任务
void parallel_pipeline(Data* input, Result* output, int n) {
    #pragma omp parallel sections
    {
        #pragma omp section
        stage1(input, temp1, n);
        
        #pragma omp section
        stage2(temp1, temp2, n);
        
        #pragma omp section
        stage3(temp2, output, n);
    }
}
```

**3. 递归分解**：
```cpp
// 分治算法
void parallel_merge_sort(float* arr, int n) {
    if (n <= 1000) {  // 串行阈值
        serial_sort(arr, n);
        return;
    }
    
    int mid = n / 2;
    
    #pragma omp task
    parallel_merge_sort(arr, mid);
    
    #pragma omp task
    parallel_merge_sort(arr + mid, n - mid);
    
    #pragma omp taskwait
    merge(arr, mid, n - mid);
}
```

### 4.2 负载均衡

**静态负载均衡**：
```cpp
// 均匀分配工作
for (int i = thread_id; i < n; i += num_threads) {
    process(data[i]);
}
```

**动态负载均衡**：
```cpp
// 工作队列
std::queue<Task*> work_queue;
std::mutex queue_mutex;

void worker_thread() {
    while (true) {
        Task* task;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (work_queue.empty()) return;
            task = work_queue.front();
            work_queue.pop();
        }
        task->execute();
    }
}
```

### 4.3 同步原语

**1. Barrier（屏障）**：
```cpp
// 所有线程到达屏障后才继续
#pragma omp barrier

// CUDA 中
__syncthreads();  // 块内同步
```

**2. Lock（锁）**：
```cpp
std::mutex mtx;

void critical_section() {
    mtx.lock();
    // 临界区代码
    mtx.unlock();
}

// RAII 风格
std::lock_guard<std::mutex> lock(mtx);
// 自动释放
```

**3. Atomic（原子操作）**：
```cpp
// 原子加法
std::atomic<int> counter(0);
counter.fetch_add(1);

// CUDA 中
atomicAdd(&counter, 1);
```

## 5. 并行性能优化

### 5.1 减少同步开销

```cpp
// 不好：频繁同步
for (int i = 0; i < n; i++) {
    #pragma omp critical
    sum += data[i];  // 每次迭代都同步
}

// 好：局部累加，最后归约
float local_sum = 0;
for (int i = 0; i < n; i++) {
    local_sum += data[i];
}
#pragma omp atomic
sum += local_sum;  // 只同步一次
```

### 5.2 减少数据竞争

```cpp
// 不好：数据竞争
for (int i = 0; i < n; i++) {
    result[i % 4] += data[i];  // 多个线程访问同一位置
}

// 好：私有化
float private_result[4] = {0};
for (int i = 0; i < n; i++) {
    private_result[i % 4] += data[i];
}
// 最后合并
for (int j = 0; j < 4; j++) {
    #pragma omp atomic
    result[j] += private_result[j];
}
```

### 5.3 缓存友好性

```cpp
// 不好：缓存不友好（列访问）
for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
        sum += matrix[i][j];  // 跳跃访问
    }
}

// 好：缓存友好（行访问）
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        sum += matrix[i][j];  // 顺序访问
    }
}
```

## 6. CUDA 并行模型预览

### 6.1 线程层次

```
Grid（网格）
├── Block (0, 0)
│   ├── Thread (0, 0)
│   ├── Thread (1, 0)
│   └── ...
├── Block (1, 0)
│   ├── Thread (0, 0)
│   └── ...
└── ...

每个线程有唯一索引：
threadIdx.x, threadIdx.y, threadIdx.z
blockIdx.x, blockIdx.y, blockIdx.z
```

### 6.2 内存层次

```
┌─────────────────────────────────────────┐
│          Global Memory (显存)            │
│  ┌───────────────────────────────────┐  │
│  │    Shared Memory (每块共享)        │  │
│  │  ┌─────┐ ┌─────┐ ┌─────┐         │  │
│  │  │线程0│ │线程1│ │ ... │         │  │
│  │  │寄存器│ │寄存器│ │     │         │  │
│  │  └─────┘ └─────┘ └─────┘         │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## 7. 实际案例分析

### 7.1 并行矩阵乘法优化

```cpp
// 版本 1：朴素并行
__global__ void matmul_v1(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// 版本 2：使用共享内存
__global__ void matmul_v2(float* A, float* B, float* C, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // 加载到共享内存
        if (row < n && tile * BLOCK_SIZE + threadIdx.x < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + tile * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < n && tile * BLOCK_SIZE + threadIdx.y < n)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // 计算
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n)
        C[row * n + col] = sum;
}
```

## 练习

1. 使用 Amdahl 定律计算：如果一个程序 95% 可并行，使用 1000 个处理器的理论加速比是多少？

2. 实现一个并行向量加法，比较不同线程数下的性能

3. 分析矩阵乘法的缓存访问模式，说明为什么行优先访问更快

4. 设计一个并行归约算法，计算数组的和

## 参考资料

- [Introduction to Parallel Computing](https://hpc-tutorials.llnl.gov/parallel/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Parallel Programming Patterns](https://www.patternsforparallelprogramming.com/)
