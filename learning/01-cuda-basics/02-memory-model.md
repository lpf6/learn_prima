# 1.2 内存模型

## CUDA 内存层次结构

CUDA 提供多种内存类型，每种都有不同的性能特征：

```
┌─────────────────────────────────────────────────────┐
│                    Global Memory                     │
│              (大容量，高延迟，所有线程可见)            │
├─────────────────────────────────────────────────────┤
│                   Constant Memory                    │
│              (只读，缓存，广播优化)                   │
├─────────────────────────────────────────────────────┤
│                    Texture Memory                    │
│              (只读，缓存，插值硬件支持)               │
├─────────────────────────────────────────────────────┤
│              Shared Memory (每SM)                    │
│           (低延迟，用户管理，Block内共享)             │
├─────────────────────────────────────────────────────┤
│              Registers (每线程)                      │
│           (最快，私有，编译器管理)                    │
└─────────────────────────────────────────────────────┘
```

## 各类内存详解

### 1. Global Memory（全局内存）

**特点**：
- 容量最大（GB 级别）
- 延迟最高（数百个时钟周期）
- 所有线程可见
- 通过 PCIe 从主机传输

**使用场景**：存储主要数据

```cpp
// 全局内存访问示例
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // 从全局内存读取/写入
    }
}
```

### 2. Shared Memory（共享内存）

**特点**：
- 低延迟（约 28 个时钟周期）
- 容量有限（每 SM 48KB-228KB）
- Block 内所有线程共享
- 用户显式管理

**使用场景**：数据复用、线程协作

```cpp
// 共享内存使用示例
__global__ void matrix_mul_shared(float *A, float *B, float *C,
                                   int M, int N, int K) {
    // 声明共享内存（动态大小）
    extern __shared__ float shared_mem[];

    // 或静态大小
    __shared__ float tile_A[16][16];
    __shared__ float tile_B[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // 分块计算
    for (int t = 0; t < K / 16; t++) {
        // 加载到共享内存
        tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * 16 + threadIdx.x];
        tile_B[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];

        __syncthreads();  // 同步，确保数据加载完成

        // 计算
        for (int k = 0; k < 16; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();  // 同步，确保计算完成
    }

    C[row * N + col] = sum;
}
```

### 3. Registers（寄存器）

**特点**：
- 最快的存储
- 每线程私有
- 数量有限（每线程最多 255 个）
- 编译器自动分配

**使用建议**：
- 频繁访问的变量
- 中间计算结果
- 避免过多导致寄存器溢出

```cpp
__global__ void example() {
    // 这些变量通常存储在寄存器中
    int local_var = threadIdx.x;
    float temp = 0.0f;

    // 寄存器溢出警告：如果变量太多
    // 编译器会将溢出的变量存入本地内存（实际是全局内存）
}
```

### 4. Constant Memory（常量内存）

**特点**：
- 只读
- 有缓存
- 广播机制（同一 Warp 读取相同地址时高效）

**使用场景**：只读常量数据

```cpp
// 常量内存声明（全局作用域）
__constant__ float const_data[256];

// 主机端复制数据
cudaMemcpyToSymbol(const_data, host_data, size);
```

## 内存访问优化

### 合并访问（Coalesced Access）

**最佳实践**：相邻线程访问相邻地址

```cpp
// 好的模式：合并访问
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[idx];  // 线程 0 访问 data[0]，线程 1 访问 data[1]...

// 坏的模式：跨步访问
float val = data[idx * 2];  // 跨步访问，效率低
```

### 内存对齐

```cpp
// 好的模式：对齐访问
struct alignas(16) MyStruct {
    float a, b, c, d;
};

// 坏的模式：非对齐
struct BadStruct {
    char a;
    float b;  // 可能未对齐
};
```

## Prima.cpp 中的内存使用示例

让我们看一个实际例子：

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

// 内存池分配器
struct ggml_cuda_pool {
    virtual ~ggml_cuda_pool() = default;
    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

// RAII 风格的内存分配
template<typename T>
struct ggml_cuda_pool_alloc {
    ggml_cuda_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    T * alloc(size_t size) {
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    ~ggml_cuda_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }
};
```

### 设备信息结构

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh

struct ggml_cuda_device_info {
    int device_count;

    struct cuda_device_info {
        int     cc;                 // 计算能力
        int     nsm;                // SM 数量
        size_t  smpb;               // 每 Block 最大共享内存
        size_t  smpbo;              // 每 Block 最大共享内存（可选）
        bool    vmm;                // 虚拟内存支持
        size_t  vmm_granularity;    // 虚拟内存粒度
        size_t  total_vram;         // 总显存
    };

    cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};
};
```

## 内存带宽计算

### 理论带宽

```
理论带宽 = 内存频率 × 总线宽度 × 2 / 8

例如：RTX 3090
- 内存频率：19.5 Gbps
- 总线宽度：384-bit
- 理论带宽 = 19.5 × 384 × 2 / 8 = 936 GB/s
```

### 实际带宽测量

```cpp
// 简单的带宽测试
__global__ void copy_kernel(float *dst, float *src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// 带宽 = 数据量 / 时间
// 有效带宽 = (读数据量 + 写数据量) / 时间
```

## 练习

1. 编写一个使用共享内存的矩阵转置内核
2. 比较使用和不使用共享内存的性能差异
3. 阅读 `ggml/src/ggml-cuda/mmq.cu`，理解其中的共享内存使用

## 下一步

完成本节后，请继续学习 [同步机制](03-synchronization.md)。
