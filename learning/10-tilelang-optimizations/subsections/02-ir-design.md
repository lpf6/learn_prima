# TileLang IR 设计

## 目录

1. [TensorIR 基础](#tensorir-基础)
2. [循环嵌套表示](#循环嵌套表示)
3. [内存抽象](#内存抽象)
4. [并行原语](#并行原语)
5. [张量操作](#张量操作)

---

## TensorIR 基础

TensorIR 是 TileLang 的核心中间表示，专门为张量计算设计。

### IR 层次结构

```
┌─────────────────────────────────────────────────────────────┐
│                    TensorIR 层次结构                          │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Module (模块)                           │    │
│  │  - 全局常量                                          │    │
│  │  - 函数集合                                          │    │
│  │  - 类型定义                                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Function (函数)                         │    │
│  │  - 参数列表                                          │    │
│  │  - 返回类型                                          │    │
│  │  - 函数体 (StmtBlock)                                │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              StmtBlock (语句块)                      │    │
│  │  - 变量声明                                          │    │
│  │  - 语句序列                                          │    │
│  │  - 作用域管理                                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Stmt (语句)                             │    │
│  │  - For (循环)                                        │    │
│  │  - If (条件)                                         │    │
│  │  - BufferStore (存储)                                │    │
│  │  - Evaluate (求值)                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Expr (表达式)                           │    │
│  │  - Var (变量)                                        │    │
│  │  - BufferLoad (加载)                                 │    │
│  │  - BinaryOp (二元操作)                               │    │
│  │  - Call (函数调用)                                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 核心数据结构

```python
# tilelang/ir/tensor_ir.py
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
from enum import Enum

class DataType(Enum):
    """数据类型枚举"""
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    BFLOAT16 = "bfloat16"

@dataclass
class Buffer:
    """
    缓冲区表示
    
    属性:
    - name: 缓冲区名称
    - shape: 形状
    - dtype: 数据类型
    - scope: 内存作用域 (global, shared, local)
    - layout: 内存布局 (row_major, column_major)
    """
    name: str
    shape: List[int]
    dtype: DataType
    scope: str = "global"
    layout: str = "row_major"
    
    def size(self) -> int:
        """计算缓冲区大小"""
        result = 1
        for dim in self.shape:
            result *= dim
        return result
    
    def bytes(self) -> int:
        """计算字节数"""
        dtype_size = {
            DataType.FLOAT16: 2,
            DataType.FLOAT32: 4,
            DataType.FLOAT64: 8,
            DataType.INT8: 1,
            DataType.INT16: 2,
            DataType.INT32: 4,
            DataType.INT64: 8,
            DataType.UINT8: 1,
            DataType.UINT16: 2,
            DataType.UINT32: 4,
            DataType.UINT64: 8,
            DataType.BFLOAT16: 2,
        }
        return self.size() * dtype_size[self.dtype]

@dataclass
class Var:
    """
    变量表示
    
    属性:
    - name: 变量名称
    - dtype: 数据类型
    """
    name: str
    dtype: DataType

@dataclass
class IterVar:
    """
    迭代变量
    
    用于表示循环迭代器
    
    属性:
    - var: 变量
    - dom_min: 域最小值
    - dom_max: 域最大值
    - iter_type: 迭代类型 (DataPar, ThreadBind, BlockBind)
    """
    var: Var
    dom_min: int
    dom_max: int
    iter_type: str = "DataPar"
```

---

## 循环嵌套表示

### For 循环

```python
@dataclass
class For:
    """
    For 循环语句
    
    属性:
    - loop_var: 循环变量
    - min_val: 最小值
    - max_val: 最大值
    - step: 步长
    - body: 循环体
    - kind: 循环类型 (serial, parallel, vectorized)
    - annotations: 注解 (unroll, vectorize, etc.)
    """
    loop_var: IterVar
    min_val: int
    max_val: int
    step: int = 1
    body: 'StmtBlock'
    kind: str = "serial"
    annotations: Dict[str, Any] = None
    
    def is_parallel(self) -> bool:
        """是否并行循环"""
        return self.kind in ("parallel", "vectorized")
    
    def is_vectorized(self) -> bool:
        """是否向量化循环"""
        return self.kind == "vectorized"
    
    def get_extent(self) -> int:
        """获取循环范围"""
        return (self.max_val - self.min_val) // self.step

class LoopNest:
    """
    循环嵌套
    
    管理多层嵌套循环
    """
    
    def __init__(self):
        self.loops: List[For] = []
        self.body: Optional[StmtBlock] = None
    
    def add_loop(self, loop: For):
        """添加循环"""
        self.loops.append(loop)
    
    def get_loop_vars(self) -> List[IterVar]:
        """获取所有循环变量"""
        return [loop.loop_var for loop in self.loops]
    
    def get_parallel_loops(self) -> List[For]:
        """获取并行循环"""
        return [loop for loop in self.loops if loop.is_parallel()]
    
    def get_serial_loops(self) -> List[For]:
        """获取串行循环"""
        return [loop for loop in self.loops if not loop.is_parallel()]
    
    def unroll(self, loop_idx: int, factor: int = 0):
        """
        展开循环
        
        参数:
        - loop_idx: 循环索引
        - factor: 展开因子 (0 表示完全展开)
        """
        loop = self.loops[loop_idx]
        
        if factor == 0:
            # 完全展开
            self.fully_unroll(loop_idx)
        else:
            # 部分展开
            self.partially_unroll(loop_idx, factor)
    
    def fully_unroll(self, loop_idx: int):
        """完全展开循环"""
        loop = self.loops[loop_idx]
        
        # 创建展开后的语句块
        unrolled_body = StmtBlock()
        
        for i in range(loop.min_val, loop.max_val, loop.step):
            # 替换循环变量
            substituted_body = self.substitute_var(
                loop.body, loop.loop_var, i
            )
            unrolled_body.extend(substituted_body)
        
        # 移除循环，替换为展开后的语句块
        self.loops.pop(loop_idx)
        self.body = unrolled_body
    
    def partially_unroll(self, loop_idx: int, factor: int):
        """部分展开循环"""
        loop = self.loops[loop_idx]
        
        # 计算展开后的步长
        new_step = loop.step * factor
        
        # 创建外层循环
        outer_loop = For(
            loop_var=loop.loop_var,
            min_val=loop.min_val,
            max_val=loop.max_val,
            step=new_step,
            body=StmtBlock(),
            kind=loop.kind,
        )
        
        # 创建内层展开的语句块
        for i in range(factor):
            inner_body = self.substitute_var(
                loop.body, loop.loop_var, 
                loop.loop_var + i * loop.step
            )
            outer_loop.body.extend(inner_body)
        
        # 替换原循环
        self.loops[loop_idx] = outer_loop
```

### 循环变换

```python
class LoopTransformation:
    """
    循环变换
    
    提供各种循环优化变换
    """
    
    @staticmethod
    def tile(loop_nest: LoopNest, tile_sizes: List[int]) -> LoopNest:
        """
        循环分块
        
        将循环分成小块以提高缓存利用率
        
        示例:
        原始循环:
        for i in range(M):
            for j in range(N):
                C[i, j] = A[i, j] + B[i, j]
        
        分块后:
        for i_outer in range(M // TILE_M):
            for j_outer in range(N // TILE_N):
                for i_inner in range(TILE_M):
                    for j_inner in range(TILE_N):
                        i = i_outer * TILE_M + i_inner
                        j = j_outer * TILE_N + j_inner
                        C[i, j] = A[i, j] + B[i, j]
        """
        pass
    
    @staticmethod
    def reorder(loop_nest: LoopNest, order: List[int]) -> LoopNest:
        """
        循环重排
        
        改变循环顺序以提高内存访问效率
        
        示例:
        原始顺序: [i, j, k]
        重排后: [k, i, j]
        """
        pass
    
    @staticmethod
    def fuse(loop_nest1: LoopNest, loop_nest2: LoopNest) -> LoopNest:
        """
        循环融合
        
        合并两个循环以提高指令级并行
        
        示例:
        原始:
        for i in range(N):
            A[i] = B[i] + 1
        for i in range(N):
            C[i] = A[i] * 2
        
        融合后:
        for i in range(N):
            A[i] = B[i] + 1
            C[i] = A[i] * 2
        """
        pass
    
    @staticmethod
    def split(loop_nest: LoopNest, loop_idx: int, factors: List[int]) -> List[LoopNest]:
        """
        循环拆分
        
        将一个循环拆分为多个循环
        
        示例:
        原始:
        for i in range(N):
            A[i] = B[i] + C[i]
        
        拆分后:
        for i in range(N // 2):
            A[i] = B[i] + C[i]
        for i in range(N // 2, N):
            A[i] = B[i] + C[i]
        """
        pass
    
    @staticmethod
    def interchange(loop_nest: LoopNest, idx1: int, idx2: int) -> LoopNest:
        """
        循环交换
        
        交换两个循环的位置
        
        示例:
        原始顺序: [i, j]
        交换后: [j, i]
        """
        pass
```

---

## 内存抽象

### Buffer 操作

```python
@dataclass
class BufferLoad:
    """
    缓冲区加载
    
    从缓冲区加载数据
    
    属性:
    - buffer: 缓冲区
    - indices: 索引表达式
    - dtype: 数据类型
    """
    buffer: Buffer
    indices: List['Expr']
    dtype: DataType
    
    def get_access_pattern(self) -> AccessPattern:
        """获取访问模式"""
        pass

@dataclass
class BufferStore:
    """
    缓冲区存储
    
    将数据存储到缓冲区
    
    属性:
    - buffer: 缓冲区
    - indices: 索引表达式
    - value: 值表达式
    - dtype: 数据类型
    """
    buffer: Buffer
    indices: List['Expr']
    value: 'Expr'
    dtype: DataType

class AccessPattern:
    """
    内存访问模式
    
    分析和优化内存访问
    """
    
    def __init__(self, load: BufferLoad, store: BufferStore):
        self.load = load
        self.store = store
    
    def is_coalesced(self) -> bool:
        """
        检查是否合并访问
        
        合并访问条件:
        - 相邻线程访问相邻地址
        - 访问对齐
        """
        pass
    
    def is_bank_conflict_free(self) -> bool:
        """
        检查是否无 Bank Conflict
        
        无冲突条件:
        - 访问模式不导致同一 Bank 同时访问
        """
        pass
    
    def get_spatial_locality(self) -> float:
        """
        获取空间局部性分数
        
        返回 0-1 的分数，越高越好
        """
        pass
    
    def get_temporal_locality(self) -> float:
        """
        获取时间局部性分数
        
        返回 0-1 的分数，越高越好
        """
        pass
```

### 内存作用域

```python
class MemoryScope(Enum):
    """内存作用域枚举"""
    GLOBAL = "global"      # 全局内存 (HBM)
    SHARED = "shared"      # 共享内存 (SRAM)
    LOCAL = "local"        # 本地内存 (寄存器)
    CONSTANT = "constant"  # 常量内存
    TEXTURE = "texture"    # 纹理内存

class MemoryHierarchy:
    """
    内存层次管理
    
    管理数据在不同内存层次间的移动
    """
    
    HIERARCHY = [
        (MemoryScope.GLOBAL, "HBM", "高容量，高延迟"),
        (MemoryScope.SHARED, "SRAM", "低容量，低延迟"),
        (MemoryScope.LOCAL, "Register", "最低容量，最低延迟"),
    ]
    
    @staticmethod
    def get_latency(scope: MemoryScope) -> int:
        """
        获取内存延迟 (时钟周期)
        
        返回:
        - GLOBAL: ~400 cycles
        - SHARED: ~28 cycles
        - LOCAL: ~1 cycle
        """
        latencies = {
            MemoryScope.GLOBAL: 400,
            MemoryScope.SHARED: 28,
            MemoryScope.LOCAL: 1,
            MemoryScope.CONSTANT: 1,  # 有缓存
            MemoryScope.TEXTURE: 100,
        }
        return latencies[scope]
    
    @staticmethod
    def get_bandwidth(scope: MemoryScope) -> int:
        """
        获取内存带宽 (GB/s)
        
        返回:
        - GLOBAL: ~2000 GB/s (HBM2e)
        - SHARED: ~19000 GB/s
        - LOCAL: ~50000 GB/s
        """
        bandwidths = {
            MemoryScope.GLOBAL: 2000,
            MemoryScope.SHARED: 19000,
            MemoryScope.LOCAL: 50000,
            MemoryScope.CONSTANT: 1000,
            MemoryScope.TEXTURE: 1000,
        }
        return bandwidths[scope]
```

---

## 并行原语

### 并行类型

```python
class ParallelKind(Enum):
    """并行类型枚举"""
    SERIAL = "serial"           # 串行执行
    PARALLEL = "parallel"       # 并行执行
    VECTORIZED = "vectorized"   # 向量化执行
    UNROLLED = "unrolled"       # 展开执行

@dataclass
class ParallelBinding:
    """
    并行绑定
    
    将循环迭代绑定到并行执行单元
    
    属性:
    - loop_var: 循环变量
    - threads: 线程绑定
    - blocks: 块绑定
    - vector_lane: 向量通道绑定
    """
    loop_var: IterVar
    threads: Optional[List[int]] = None
    blocks: Optional[List[int]] = None
    vector_lane: Optional[int] = None
    
    def get_launch_config(self) -> dict:
        """
        获取内核启动配置
        
        返回:
        {
            "grid": (grid_x, grid_y, grid_z),
            "block": (block_x, block_y, block_z),
        }
        """
        pass

class ParallelScheduler:
    """
    并行调度器
    
    决定如何将循环映射到并行执行单元
    """
    
    def schedule(self, loop_nest: LoopNest, target: str) -> List[ParallelBinding]:
        """
        调度循环嵌套
        
        策略:
        1. 分析循环依赖
        2. 识别可并行循环
        3. 分配并行资源
        4. 生成并行绑定
        """
        # 1. 分析循环依赖
        deps = self.analyze_dependencies(loop_nest)
        
        # 2. 识别可并行循环
        parallel_loops = self.identify_parallel_loops(loop_nest, deps)
        
        # 3. 分配并行资源
        bindings = self.allocate_parallel_resources(
            parallel_loops, target
        )
        
        return bindings
    
    def analyze_dependencies(self, loop_nest: LoopNest) -> DependencyGraph:
        """
        分析循环依赖
        
        依赖类型:
        - RAW (Read After Write): 真依赖
        - WAR (Write After Read): 反依赖
        - WAW (Write After Write): 输出依赖
        """
        pass
    
    def identify_parallel_loops(self, loop_nest: LoopNest, 
                                deps: DependencyGraph) -> List[For]:
        """
        识别可并行循环
        
        条件:
        - 无跨迭代的真依赖
        - 可通过私有化消除反依赖和输出依赖
        """
        pass
```

### 线程映射

```python
class ThreadMapper:
    """
    线程映射器
    
    将循环迭代映射到 GPU 线程
    """
    
    @staticmethod
    def map_to_block_idx(loop: For, dim: int = 0) -> ParallelBinding:
        """
        映射到 blockIdx
        
        用于粗粒度并行
        """
        return ParallelBinding(
            loop_var=loop.loop_var,
            blocks=[dim],
        )
    
    @staticmethod
    def map_to_threadIdx(loop: For, dim: int = 0) -> ParallelBinding:
        """
        映射到 threadIdx
        
        用于细粒度并行
        """
        return ParallelBinding(
            loop_var=loop.loop_var,
            threads=[dim],
        )
    
    @staticmethod
    def map_to_vector_lane(loop: For, width: int) -> ParallelBinding:
        """
        映射到向量通道
        
        用于向量化
        """
        return ParallelBinding(
            loop_var=loop.loop_var,
            vector_lane=width,
        )
    
    @staticmethod
    def get_optimal_mapping(loop_nest: LoopNest, target: str) -> List[ParallelBinding]:
        """
        获取最优映射
        
        策略:
        1. 外层循环映射到 blockIdx
        2. 中层循环映射到 threadIdx
        3. 内层循环向量化或展开
        """
        bindings = []
        
        loops = loop_nest.loops
        n_loops = len(loops)
        
        for i, loop in enumerate(loops):
            if i < n_loops - 2:
                # 外层循环 -> blockIdx
                bindings.append(ThreadMapper.map_to_block_idx(loop, i))
            elif i < n_loops - 1:
                # 中层循环 -> threadIdx
                bindings.append(ThreadMapper.map_to_threadIdx(loop, i - (n_loops - 2)))
            else:
                # 内层循环 -> 向量化
                bindings.append(ThreadMapper.map_to_vector_lane(loop, 4))
        
        return bindings
```

---

## 张量操作

### GEMM 操作

```python
@dataclass
class GEMM:
    """
    矩阵乘法操作
    
    C = alpha * A @ B + beta * C
    
    属性:
    - A: 矩阵 A
    - B: 矩阵 B
    - C: 矩阵 C
    - alpha: 缩放因子
    - beta: 缩放因子
    - transpose_A: 是否转置 A
    - transpose_B: 是否转置 B
    """
    A: Buffer
    B: Buffer
    C: Buffer
    alpha: float = 1.0
    beta: float = 0.0
    transpose_A: bool = False
    transpose_B: bool = False
    
    def get_mnk(self) -> Tuple[int, int, int]:
        """获取 M, N, K 维度"""
        if self.transpose_A:
            M, K = self.A.shape[1], self.A.shape[0]
        else:
            M, K = self.A.shape[0], self.A.shape[1]
        
        if self.transpose_B:
            K, N = self.B.shape[0], self.B.shape[1]
        else:
            K, N = self.B.shape[0], self.B.shape[1]
        
        return M, N, K
    
    def get_flops(self) -> int:
        """获取 FLOPs"""
        M, N, K = self.get_mnk()
        return 2 * M * N * K  # 每个元素 2 FLOPs (乘法 + 加法)
    
    def get_memory_access(self) -> int:
        """获取内存访问量 (字节)"""
        M, N, K = self.get_mnk()
        
        # A: M * K 个元素
        # B: K * N 个元素
        # C: M * N 个元素
        
        dtype_size = 2  # float16
        return dtype_size * (M * K + K * N + 2 * M * N)
    
    def get_arithmetic_intensity(self) -> float:
        """
        获取计算强度 (FLOPs/Byte)
        
        高计算强度意味着计算密集型
        低计算强度意味着内存密集型
        """
        return self.get_flops() / self.get_memory_access()

class GEMMScheduler:
    """
    GEMM 调度器
    
    优化 GEMM 操作的性能
    """
    
    def schedule(self, gemm: GEMM, target: str, arch: int) -> LoopNest:
        """
        调度 GEMM
        
        策略:
        1. 分块 (Tiling)
        2. Tensor Core 映射
        3. 流水线化
        4. 向量化
        """
        # 1. 创建基础循环嵌套
        loop_nest = self.create_basic_loop_nest(gemm)
        
        # 2. 分块
        tile_sizes = self.get_optimal_tile_sizes(target, arch)
        loop_nest = LoopTransformation.tile(loop_nest, tile_sizes)
        
        # 3. Tensor Core 映射
        if self.supports_tensor_core(target, arch):
            loop_nest = self.map_to_tensor_core(loop_nest, arch)
        
        # 4. 流水线化
        loop_nest = self.pipeline(loop_nest)
        
        return loop_nest
    
    def get_optimal_tile_sizes(self, target: str, arch: int) -> List[int]:
        """
        获取最优分块大小
        
        根据架构特性选择
        """
        if target == "cuda":
            if arch >= 90:  # Hopper
                return [128, 128, 64]
            elif arch >= 80:  # Ampere
                return [128, 128, 32]
            else:  # Volta/Turing
                return [64, 64, 32]
        elif target == "rocm":
            return [64, 64, 32]
        else:
            return [32, 32, 16]
```

### Flash Attention 操作

```python
@dataclass
class FlashAttention:
    """
    Flash Attention 操作
    
    高效的注意力计算
    
    属性:
    - Q: Query 矩阵
    - K: Key 矩阵
    - V: Value 矩阵
    - O: Output 矩阵
    - scale: 缩放因子
    - causal: 是否因果注意力
    """
    Q: Buffer
    K: Buffer
    V: Buffer
    O: Buffer
    scale: float
    causal: bool = False
    
    def get_dimensions(self) -> Tuple[int, int, int, int]:
        """
        获取维度
        
        返回: (batch, heads, seq_len, head_dim)
        """
        return self.Q.shape

class FlashAttentionScheduler:
    """
    Flash Attention 调度器
    """
    
    def schedule(self, fa: FlashAttention, target: str, arch: int) -> LoopNest:
        """
        调度 Flash Attention
        
        策略:
        1. 分块计算
        2. 在线 Softmax
        3. Tensor Core 映射
        """
        # 1. 创建分块循环
        loop_nest = self.create_tiled_loops(fa)
        
        # 2. 插入在线 Softmax
        loop_nest = self.insert_online_softmax(loop_nest)
        
        # 3. Tensor Core 映射
        if self.supports_tensor_core(target, arch):
            loop_nest = self.map_to_tensor_core(loop_nest, arch)
        
        return loop_nest
    
    def insert_online_softmax(self, loop_nest: LoopNest) -> LoopNest:
        """
        插入在线 Softmax
        
        在线 Softmax 算法:
        1. 维护最大值 m
        2. 维护指数和 l
        3. 增量更新输出
        """
        # 创建状态变量
        m = Var("m", DataType.FLOAT32)
        l = Var("l", DataType.FLOAT32)
        
        # 插入更新逻辑
        # ...
        
        return loop_nest
```

## 下一步

继续学习 [优化 Pass](03-optimization-pass.md)，了解 TileLang 的优化技术。
