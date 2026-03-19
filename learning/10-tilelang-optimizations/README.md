# TileLang 优化实现详解

## 目录

本章节包含以下详细子章节：

1. [架构设计](subsections/01-architecture.md) - TileLang 编译器架构、前端设计、IR 结构
2. [编译器架构](subsections/01-compiler-architecture.md) - 编译流程、JIT 编译、缓存系统
3. [IR 设计](subsections/02-ir-design.md) - TensorIR、循环嵌套、内存抽象、并行原语
4. [优化 Pass](subsections/03-optimization-pass.md) - 自动并行化、向量化、张量化、内存优化
5. [代码生成](subsections/04-code-generation.md) - CUDA/ROCm/Metal/Ascend/CPU 代码生成
6. [运行时系统](subsections/05-runtime-system.md) - 内核启动器、内存管理、自动调优、性能分析
7. [优缺点分析](subsections/05-pros-cons-cuda-compat.md) - TileLang 优势、限制、CUDA 兼容性

---

## 概述

TileLang 是一个领域特定语言（DSL），旨在简化高性能 GPU/CPU/Accelerators 内核的开发。它采用 Pythonic 语法，底层基于 TVM 编译器基础设施，让开发者专注于生产力，同时不牺牲底层优化所需的性能。

### 为什么需要 TileLang？

传统的 CUDA 开发面临以下挑战：

1. **学习曲线陡峭**：需要深入理解 GPU 架构、内存层次、线程调度
2. **代码复杂度高**：手动优化需要大量模板代码和架构特定调整
3. **可移植性差**：针对不同 GPU 架构需要重写代码
4. **调试困难**：底层 CUDA 代码难以调试和验证

TileLang 通过高层抽象解决这些问题，同时保持与手写 CUDA 内核相当的性能。

## TileLang 与 CUDA 设计对比

### 编程模型对比

| 特性 | CUDA | TileLang |
|------|------|----------|
| **编程语言** | C++ 扩展 | Python DSL |
| **抽象层次** | 低层（线程级） | 高层（块/瓦片级） |
| **内存管理** | 手动管理 | 自动推断 |
| **并行模型** | Grid/Block/Thread | Tile/Block 自动映射 |
| **编译方式** | nvcc 直接编译 | TVM 中间表示 → 多后端 |
| **跨平台** | 仅 NVIDIA GPU | NVIDIA/AMD/Intel/Apple Metal/华为昇腾 |

### 代码复杂度对比

#### CUDA 实现 Flash Attention（简化版）

```cpp
// 传统 CUDA 需要 ~200+ 行代码
template<int D, int ncols>
__global__ void flash_attn_tile_f16(
    const char * __restrict__ Q,
    const char * __restrict__ K,
    const char * __restrict__ V,
    float * __restrict__ O,
    const int ncols_q,
    const int nrows_k,
    const float scale) {
    
    // 1. 手动计算线程索引
    const int col = blockIdx.x * ncols + threadIdx.x;
    const int row_q = blockIdx.y;
    
    // 2. 手动管理共享内存
    __shared__ half K_tile[TILE_SIZE][D];
    __shared__ half V_tile[TILE_SIZE][D];
    
    // 3. 手动加载数据
    // ... 大量加载代码
    
    // 4. 手动实现在线 Softmax
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    // ... 复杂的归约逻辑
    
    // 5. 手动处理边界条件
    // ... 边界检查代码
    
    // 6. 手动写回结果
    // ... 存储代码
}
```

#### TileLang 实现 Flash Attention

```python
# TileLang 只需要 ~30 行代码
import tilelang
from tilelang import language as T

@tilelang.jit
def flash_attn(batch, heads, seq_len, head_dim, block_M, block_N):
    # 声明输入输出缓冲区
    Q = T.Buffer([batch, heads, seq_len, head_dim], "float16")
    K = T.Buffer([batch, heads, seq_len, head_dim], "float16")
    V = T.Buffer([batch, heads, seq_len, head_dim], "float16")
    O = T.Buffer([batch, heads, seq_len, head_dim], "float16")
    
    # 使用块级编程
    for b, h, i in T.Parallel(batch, heads, seq_len // block_M):
        # 分配累加器（自动映射到寄存器/共享内存）
        acc = T.Fragment([block_M, head_dim], "float")
        m = T.Fragment([block_M], "float")
        l = T.Fragment([block_M], "float")
        
        # 初始化
        T.fill(acc, 0.0)
        T.fill(m, -float("inf"))
        T.fill(l, 0.0)
        
        # 分块计算
        for j in T.Serial(seq_len // block_N):
            # 加载 K, V 块（自动优化内存访问）
            k_block = T.load(K[b, h, j*block_N:(j+1)*block_N, :])
            v_block = T.load(V[b, h, j*block_N:(j+1)*block_N, :])
            
            # 计算 QK^T
            q_block = T.load(Q[b, h, i*block_M:(i+1)*block_M, :])
            qk = T.gemm(q_block, k_block, transpose_B=True)
            
            # 在线 Softmax（自动优化）
            m_new = T.max(m, T.max(qk, axis=1))
            p = T.exp(qk - m_new)
            l_new = l * T.exp(m - m_new) + T.sum(p, axis=1)
            
            # 更新累加器
            acc = acc * (l / l_new) + T.gemm(p, v_block)
            
            m, l = m_new, l_new
        
        # 写回结果
        T.store(O[b, h, i*block_M:(i+1)*block_M, :], acc / l)
    
    return O

# 编译并运行
kernel = flash_attn.compile(target="cuda")
output = kernel(q_tensor, k_tensor, v_tensor)
```

### 设计哲学对比

```
CUDA 设计哲学：
┌─────────────────────────────────────────────────────────────┐
│  程序员负责一切：                                             │
│  ├── 线程索引计算                                            │
│  ├── 内存层次管理（寄存器/共享内存/全局内存）                   │
│  ├── 数据加载/存储模式                                        │
│  ├── 同步点放置                                              │
│  ├── 架构特定优化（Tensor Core、Warp 级原语）                  │
│  └── 边界条件处理                                            │
└─────────────────────────────────────────────────────────────┘

TileLang 设计哲学：
┌─────────────────────────────────────────────────────────────┐
│  编译器负责优化，程序员关注算法：                               │
│  ├── 声明式描述计算逻辑                                       │
│  ├── 自动内存层次映射                                         │
│  ├── 自动并行化调度                                           │
│  ├── 自动生成架构特定代码                                      │
│  └── 自动边界处理                                             │
└─────────────────────────────────────────────────────────────┘
```

## TileLang 核心概念

### 1. Tile（瓦片）抽象

TileLang 的核心是 Tile 抽象，将数据分块处理：

```python
import tilelang
from tilelang import language as T

# 定义块大小
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32

@tilelang.jit
def matmul(M, N, K):
    # 输入矩阵
    A = T.Buffer([M, K], "float16")
    B = T.Buffer([K, N], "float16")
    C = T.Buffer([M, N], "float16")
    
    # 块级并行
    for i, j in T.Parallel(M // BLOCK_M, N // BLOCK_N):
        # 累加器（自动映射到寄存器）
        acc = T.Fragment([BLOCK_M, BLOCK_N], "float")
        T.fill(acc, 0.0)
        
        # 分块计算
        for k in T.Serial(K // BLOCK_K):
            # 加载块（自动优化内存访问模式）
            a_tile = T.load(A[i*BLOCK_M:(i+1)*BLOCK_M, k*BLOCK_K:(k+1)*BLOCK_K])
            b_tile = T.load(B[k*BLOCK_K:(k+1)*BLOCK_K, j*BLOCK_N:(j+1)*BLOCK_N])
            
            # 矩阵乘法（自动使用 Tensor Core）
            acc = T.gemm(a_tile, b_tile, acc)
        
        # 存储结果
        T.store(C[i*BLOCK_M:(i+1)*BLOCK_M, j*BLOCK_N:(j+1)*BLOCK_N], acc)
    
    return C
```

### 2. 内存层次自动映射

TileLang 自动将数据映射到最优的内存层次：

```python
# TileLang 自动决定：
# - 哪些数据放入寄存器
# - 哪些数据放入共享内存
# - 如何优化内存访问模式
# - 如何避免 Bank Conflict

@tilelang.jit
def optimized_kernel(...):
    # Fragment: 自动映射到寄存器
    reg_data = T.Fragment([64, 64], "float")
    
    # Buffer: 自动映射到共享内存
    shared_data = T.SharedBuffer([128, 128], "float16")
    
    # 编译器自动优化内存布局
```

### 3. 自动调度

TileLang 提供自动调度优化：

```python
from tilelang import auto_scheduler

# 定义搜索空间
config = auto_scheduler.TuningConfig(
    num_trials_per_task=1000,
    max_trials_global=10000,
)

# 自动搜索最优调度
with auto_scheduler.ApplyHistoryBest(config):
    kernel = matmul.compile(
        target="cuda",
        auto_tune=True,  # 启用自动调优
    )
```

### 4. 多后端支持

TileLang 支持多种硬件后端：

```python
# NVIDIA GPU
kernel_cuda = matmul.compile(target="cuda")

# AMD GPU
kernel_rocm = matmul.compile(target="rocm")

# Intel GPU
kernel_intel = matmul.compile(target="intel_gpu")

# Apple Metal
kernel_metal = matmul.compile(target="metal")

# 华为昇腾 NPU
kernel_ascend = matmul.compile(target="ascend")

# CPU (SIMD)
kernel_cpu = matmul.compile(target="llvm")
```

## TileLang 优化实现详解

### 1. 高性能矩阵乘法

```python
# 文件：examples/gemm.py

import tilelang
from tilelang import language as T

def make_gemm_kernel(M, N, K, block_M, block_N, block_K):
    @tilelang.jit
    def gemm(
        A: T.Buffer([M, K], "float16"),
        B: T.Buffer([K, N], "float16"),
        C: T.Buffer([M, N], "float16"),
    ):
        # 使用 2D 块并行
        for i, j in T.Parallel(M // block_M, N // block_N):
            # 分配累加器
            acc = T.Fragment([block_M, block_N], "float")
            T.fill(acc, 0.0)
            
            # 分块计算
            for k in T.Serial(K // block_K):
                # 加载 A 块
                a = T.load(A[i * block_M:(i + 1) * block_M, 
                            k * block_K:(k + 1) * block_K])
                
                # 加载 B 块
                b = T.load(B[k * block_K:(k + 1) * block_K,
                            j * block_N:(j + 1) * block_N])
                
                # 矩阵乘法累加
                acc = T.gemm(a, b, acc)
            
            # 存储结果
            T.store(C[i * block_M:(i + 1) * block_M,
                      j * block_N:(j + 1) * block_N], acc)
        
        return C
    
    return gemm

# 编译内核
gemm = make_gemm_kernel(1024, 1024, 1024, 64, 64, 32)
kernel = gemm.compile(target="cuda", auto_tune=True)

# 性能：在 H100 上达到理论峰值的 95%+
```

### 2. Flash Attention 实现

```python
# 文件：examples/flash_attention.py

import tilelang
from tilelang import language as T
import math

def make_flash_attn(batch, heads, seq_len, head_dim, block_M, block_N, causal=False):
    scale = 1.0 / math.sqrt(head_dim)
    
    @tilelang.jit
    def flash_attn(
        Q: T.Buffer([batch, heads, seq_len, head_dim], "float16"),
        K: T.Buffer([batch, heads, seq_len, head_dim], "float16"),
        V: T.Buffer([batch, heads, seq_len, head_dim], "float16"),
        O: T.Buffer([batch, heads, seq_len, head_dim], "float16"),
    ):
        # 批量和头并行
        for b, h, i in T.Parallel(batch, heads, seq_len // block_M):
            # 累加器和 Softmax 状态
            acc = T.Fragment([block_M, head_dim], "float")
            m_i = T.Fragment([block_M], "float")
            l_i = T.Fragment([block_M], "float")
            
            T.fill(acc, 0.0)
            T.fill(m_i, -float("inf"))
            T.fill(l_i, 0.0)
            
            # 加载 Q 块
            q = T.load(Q[b, h, i * block_M:(i + 1) * block_M, :])
            
            # 分块计算 Attention
            for j in T.Serial(seq_len // block_N):
                # Causal mask
                if causal and j * block_N > (i + 1) * block_M:
                    continue
                
                # 加载 K, V 块
                k = T.load(K[b, h, j * block_N:(j + 1) * block_N, :])
                v = T.load(V[b, h, j * block_N:(j + 1) * block_N, :])
                
                # 计算 QK^T
                qk = T.gemm(q, k, transpose_B=True) * scale
                
                # 在线 Softmax
                m_new = T.maximum(m_i, T.max(qk, axis=1))
                p = T.exp(qk - m_new[:, None])
                l_new = l_i * T.exp(m_i - m_new) + T.sum(p, axis=1)
                
                # 更新输出
                acc = acc * (l_i / l_new)[:, None] + T.gemm(p, v)
                
                m_i, l_i = m_new, l_new
            
            # 写回结果
            T.store(O[b, h, i * block_M:(i + 1) * block_M, :], 
                   acc / l_i[:, None])
        
        return O
    
    return flash_attn

# 编译并运行
flash_attn = make_flash_attn(1, 32, 4096, 128, 64, 64, causal=True)
kernel = flash_attn.compile(target="cuda")

# 性能：在 H100 上与 FlashAttention-2 性能相当
```

### 3. 量化矩阵乘法

```python
# 文件：examples/dequant_gemm.py

import tilelang
from tilelang import language as T

def make_dequant_gemm(M, N, K, block_M, block_N, block_K):
    """量化矩阵乘法：A 是 FP16，B 是 INT4 量化"""
    
    @tilelang.jit
    def dequant_gemm(
        A: T.Buffer([M, K], "float16"),
        B_q: T.Buffer([K, N // 8], "int8"),  # INT4 打包存储
        B_scale: T.Buffer([K, N // 32], "float16"),  # 每 32 个元素一个 scale
        C: T.Buffer([M, N], "float16"),
    ):
        for i, j in T.Parallel(M // block_M, N // block_N):
            acc = T.Fragment([block_M, block_N], "float")
            T.fill(acc, 0.0)
            
            for k in T.Serial(K // block_K):
                # 加载 A 块
                a = T.load(A[i * block_M:(i + 1) * block_M,
                            k * block_K:(k + 1) * block_K])
                
                # 加载量化 B 块
                b_q = T.load(B_q[k * block_K:(k + 1) * block_K,
                              j * block_N // 8:(j + 1) * block_N // 8])
                b_s = T.load(B_scale[k * block_K:(k + 1) * block_K,
                              j * block_N // 32:(j + 1) * block_N // 32])
                
                # 反量化并计算
                b = T.dequantize_int4(b_q, b_s)  # 内置反量化
                acc = T.gemm(a, b, acc)
            
            T.store(C[i * block_M:(i + 1) * block_M,
                      j * block_N:(j + 1) * block_N], acc)
        
        return C
    
    return dequant_gemm
```

### 4. MLA Decoding 优化

```python
# 文件：examples/mla_decode.py
# TileLang 实现的 MLA Decoding 只需要 ~80 行代码
# 在 H100 上达到与 FlashMLA 相当的性能

import tilelang
from tilelang import language as T
import math

def make_mla_decode(batch, heads, seq_len, qk_nope_head_dim, qk_rope_head_dim, v_head_dim):
    """
    Multi-Latent Attention (MLA) Decoding
    - qk_nope_head_dim: 非旋转部分维度
    - qk_rope_head_dim: 旋转部分维度
    - v_head_dim: 值头维度
    """
    
    @tilelang.jit
    def mla_decode(
        q_nope: T.Buffer([batch, heads, qk_nope_head_dim], "float16"),
        q_rope: T.Buffer([batch, heads, qk_rope_head_dim], "float16"),
        kv: T.Buffer([batch, seq_len, qk_nope_head_dim + v_head_dim], "float16"),
        k_rope: T.Buffer([batch, seq_len, qk_rope_head_dim], "float16"),
        output: T.Buffer([batch, heads, v_head_dim], "float16"),
    ):
        # 头并行
        for b, h in T.Parallel(batch, heads):
            # 分割 KV
            k_nope = kv[b, :, :qk_nope_head_dim]
            v = kv[b, :, qk_nope_head_dim:]
            
            # 计算 QK^T (非旋转部分)
            qk_nope = T.gemm(q_nope[b, h, :], k_nope, transpose_B=True)
            
            # 计算 QK^T (旋转部分)
            qk_rope = T.gemm(q_rope[b, h, :], k_rope[b, :, :], transpose_B=True)
            
            # 合并注意力分数
            qk = qk_nope + qk_rope
            qk = qk / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
            
            # Softmax
            m = T.max(qk)
            p = T.exp(qk - m)
            p = p / T.sum(p)
            
            # 计算 Output
            out = T.gemm(p, v)
            
            T.store(output[b, h, :], out)
        
        return output
    
    return mla_decode
```

## TileLang 高级特性

### 1. Tensor Core 自动使用

```python
# TileLang 自动检测并使用 Tensor Core

@tilelang.jit
def tensor_core_gemm(M, N, K):
    A = T.Buffer([M, K], "float16")
    B = T.Buffer([K, N], "float16")
    C = T.Buffer([M, N], "float16")
    
    for i, j in T.Parallel(M // 16, N // 16):
        # 16x16x16 是 Tensor Core 的原生大小
        acc = T.Fragment([16, 16], "float")
        
        for k in T.Serial(K // 16):
            a = T.load(A[i*16:(i+1)*16, k*16:(k+1)*16])
            b = T.load(B[k*16:(k+1)*16, j*16:(j+1)*16])
            # 自动使用 Tensor Core (HMMA)
            acc = T.gemm(a, b, acc)
        
        T.store(C[i*16:(i+1)*16, j*16:(j+1)*16], acc)
    
    return C

# 编译器自动生成 Tensor Core 指令
```

### 2. 异步内存拷贝

```python
# Ampere+ 架构的异步拷贝

@tilelang.jit
def async_copy_kernel(M, N):
    A = T.Buffer([M, N], "float16")
    B = T.Buffer([M, N], "float16")
    
    for i in T.Parallel(M // 64):
        # 异步加载到共享内存
        shared = T.SharedBuffer([64, N], "float16")
        T.copy_async(A[i*64:(i+1)*64, :], shared)
        
        # 计算其他任务...
        
        # 等待异步拷贝完成
        T.wait_async()
        
        # 使用数据
        # ...
    
    return B

# 自动使用 cp.async 指令
```

### 3. TMA (Tensor Memory Accelerator)

```python
# Hopper 架构的 TMA 支持

@tilelang.jit
def tma_gemm(M, N, K):
    A = T.Buffer([M, K], "float16")
    B = T.Buffer([K, N], "float16")
    C = T.Buffer([M, N], "float16")
    
    for i, j in T.Parallel(M // 128, N // 128):
        acc = T.Fragment([128, 128], "float")
        
        for k in T.Serial(K // 64):
            # 使用 TMA 异步加载
            a = T.tma_load(A, [i*128, k*64], [128, 64])
            b = T.tma_load(B, [k*64, j*128], [64, 128])
            
            acc = T.gemm(a, b, acc)
        
        T.store(C[i*128:(i+1)*128, j*128:(j+1)*128], acc)
    
    return C

# 自动使用 TMA 指令 (H100+)
```

### 4. Warp 级原语

```python
# TileLang 提供高层 Warp 级原语

@tilelang.jit
def warp_reduce_example(N):
    data = T.Buffer([N], "float")
    output = T.Buffer([1], "float")
    
    # Warp 级归约
    for i in T.Parallel(N // 32):
        warp_data = T.load(data[i*32:(i+1)*32])
        
        # Warp 归约求和
        sum_val = T.warp_reduce_sum(warp_data)
        
        # Warp 归约求最大值
        max_val = T.warp_reduce_max(warp_data)
    
    return output

# 自动生成 shuffle 指令
```

## 性能优化技巧

### 1. 块大小选择

```python
# 块大小对性能影响很大

# 好的选择：匹配 Tensor Core 大小
block_M, block_N, block_K = 64, 64, 32  # 适合大多数 GPU

# 更好的选择：根据架构调整
# H100: 可以使用更大的块
block_M, block_N, block_K = 128, 128, 64

# 自动调优找到最优块大小
from tilelang import auto_scheduler

config = auto_scheduler.TuningConfig(
    block_m_candidates=[32, 64, 128],
    block_n_candidates=[32, 64, 128],
    block_k_candidates=[16, 32, 64],
)
```

### 2. 内存布局优化

```python
# TileLang 支持多种内存布局

@tilelang.jit
def layout_optimized_kernel(M, N):
    # 行主序
    A_row = T.Buffer([M, N], "float16", layout="row_major")
    
    # 列主序
    B_col = T.Buffer([M, N], "float16", layout="column_major")
    
    # 优化布局（避免 Bank Conflict）
    C_opt = T.Buffer([M, N], "float16", layout="swizzled")
    
    # 编译器自动选择最优布局
```

### 3. 流水线并行

```python
# 软件流水线

@tilelang.jit
def pipelined_gemm(M, N, K):
    A = T.Buffer([M, K], "float16")
    B = T.Buffer([K, N], "float16")
    C = T.Buffer([M, N], "float16")
    
    for i, j in T.Parallel(M // 64, N // 64):
        acc = T.Fragment([64, 64], "float")
        
        # 软件流水线：预取下一块数据
        T.pipeline_prefetch(A, B, stages=2)
        
        for k in T.Serial(K // 32):
            a = T.load(A[i*64:(i+1)*64, k*32:(k+1)*32])
            b = T.load(B[k*32:(k+1)*32, j*64:(j+1)*64])
            acc = T.gemm(a, b, acc)
        
        T.store(C[i*64:(i+1)*64, j*64:(j+1)*64], acc)
    
    return C
```

## 调试和性能分析

### 1. 打印调试

```python
@tilelang.jit
def debug_kernel(M, N):
    A = T.Buffer([M, N], "float16")
    
    for i in T.Parallel(M):
        row = T.load(A[i, :])
        
        # 打印变量值
        T.print(f"Row {i}:", row)
        
        # 打印缓冲区内容
        T.print_buffer(A[i, :])
    
    return A
```

### 2. 内存布局可视化

```python
from tilelang.tools import plot_layout

@tilelang.jit
def kernel(M, N):
    A = T.Buffer([M, N], "float16")
    # ...
    return A

# 可视化内存布局
plot_layout(kernel, save_path="layout.png")
```

### 3. 性能分析

```python
import tilelang.profiler as profiler

# 编译内核
kernel = gemm.compile(target="cuda")

# 性能分析
with profiler.Profile() as prof:
    result = kernel(a, b, c)

# 打印性能指标
print(prof.summary())
# 输出：
# - 执行时间
# - 内存带宽
# - 计算吞吐量
# - GPU 利用率
```

## 与 Prima.cpp 集成

### 使用 TileLang 优化 Prima.cpp 内核

```python
# 将 TileLang 编译的内核集成到 Prima.cpp

import tilelang
from tilelang import language as T
import torch

# 定义 TileLang 内核
@tilelang.jit
def optimized_attention(batch, heads, seq_len, head_dim):
    # ... Flash Attention 实现
    pass

# 编译为 PyTorch 扩展
attention_kernel = optimized_attention.compile(
    target="cuda",
    output_format="torch",  # 输出 PyTorch 扩展
)

# 在 Prima.cpp 中使用
class OptimizedAttention(torch.nn.Module):
    def __init__(self, heads, head_dim):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.kernel = attention_kernel
    
    def forward(self, q, k, v):
        return self.kernel(q, k, v)
```

## 能力目标

学习完本章节后，你能够：

**能做什么：**
- 使用 TileLang 编写高性能 GPU 内核
- 理解 TileLang 与 CUDA 的设计差异
- 将 TileLang 内核集成到现有项目
- 使用自动调优优化内核性能
- 支持多种硬件后端

**还不能做什么：**
- 完全替代手写 CUDA 内核（某些极端优化场景）
- 深入理解 TVM 编译器内部实现
- 自定义编译器优化 Pass

**实际工作示例：**
- 用 30 行代码实现 Flash Attention
- 用 80 行代码实现 MLA Decoding
- 自动生成多后端代码
- 性能达到手写 CUDA 内核的 95%+

## 练习

1. 使用 TileLang 实现一个简单的向量加法内核
2. 实现一个矩阵乘法内核，并使用自动调优优化性能
3. 将 TileLang 实现的 Flash Attention 集成到 Prima.cpp
4. 比较相同算法在 TileLang 和 CUDA 中的代码复杂度

## 参考资料

- [TileLang GitHub](https://github.com/tile-ai/tilelang)
- [TileLang 文档](https://tilelang.readthedocs.io/)
- [TileLang Puzzles](https://tilelang.ai/puzzles) - 交互式学习
- [TVM 编译器文档](https://tvm.apache.org/docs/)
