# 1.1 线程层次结构

## 核心概念

CUDA 使用三层线程层次结构来组织并行计算：

```
Grid（网格）
  └── Block（线程块）[可以有多个]
        └── Thread（线程）[每个块最多1024个线程]
```

### 关键术语

| 术语 | 说明 | 访问方式 |
|------|------|----------|
| **Grid** | 内核启动的所有线程集合 | `gridDim` |
| **Block** | 线程组，可以协作 | `blockIdx`, `blockDim` |
| **Thread** | 最小执行单元 | `threadIdx` |
| **Warp** | 32个线程的执行单元 | 隐式 |

## 线程索引计算

### 一维索引

```cpp
// 全局线程索引
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 检查边界
if (idx < n) {
    // 处理数据
}
```

### 二维索引

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;
```

### 三维索引

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;
int idz = blockIdx.z * blockDim.z + threadIdx.z;
```

## Warp 概念

**Warp 是 CUDA 执行的基本单位**，包含 32 个线程。同一 Warp 中的线程：

- 同时执行相同的指令（SIMT）
- 如果执行路径分歧，会串行执行（Warp Divergence）
- 可以通过 Shuffle 指令直接交换数据

### Warp 大小定义

在 prima.cpp 中，Warp 大小定义为常量：

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh
#define WARP_SIZE 32
```

## 实际代码示例

让我们看 prima.cpp 中的一个实际例子：

```cpp
// 文件：ggml/src/ggml-cuda/common.cuh
// Warp 级归约求和

static __device__ __forceinline__ int warp_reduce_sum(int x) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_AMPERE
    // Ampere 及以上架构使用硬件指令
    return __reduce_add_sync(0xffffffff, x);
#else
    // 其他架构使用 Shuffle 指令
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
#endif
}
```

### 代码解析

1. **`__device__`**：表示这是设备端函数，只能在 GPU 上执行
2. **`__forceinline__`**：强制内联，避免函数调用开销
3. **`__reduce_add_sync`**：Ampere 架构引入的硬件归约指令
4. **`__shfl_xor_sync`**：Warp 级数据交换指令

### Shuffle 操作图解

```
初始状态（mask = 16）：
线程:  0  1  2  3  ... 16 17 18 19 ... 31
值:    a  b  c  d  ... p  q  r  s  ...
       ↓  ↓  ↓  ↓      ↓  ↓  ↓  ↓
交换:  p  q  r  s  ... a  b  c  d  ...
结果: a+p b+q c+r d+s ...

mask = 8:
线程:  0  1  2  3  4  5  6  7  8  9 ...
交换:  8  9  10 11 12 13 14 15 0  1  ...
结果: 累加...

最终：所有 32 个线程得到相同的和
```

## Block 和 Grid 大小选择

### 常见配置

```cpp
// 典型的 Block 大小
const int BLOCK_SIZE = 256;  // 或 128, 512

// 计算 Grid 大小
int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

// 启动内核
my_kernel<<<grid_size, BLOCK_SIZE>>>(...);
```

### 选择原则

1. **Block 大小**：通常是 32 的倍数（Warp 大小）
2. **常用值**：128, 256, 512
3. **最大值**：1024（硬件限制）
4. **考虑因素**：
   - 寄存器使用量
   - 共享内存使用量
   - 占用率（Occupancy）

## 练习

1. 编写一个 CUDA 内核，将数组中每个元素乘以 2
2. 修改上面的内核，使用二维线程索引处理矩阵
3. 阅读 `ggml/src/ggml-cuda/norm.cu`，理解其中的线程索引计算

## 下一步

完成本节后，请继续学习 [内存模型](02-memory-model.md)。
