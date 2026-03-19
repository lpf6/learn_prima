# TileLang 优化 Pass

## 目录

1. [自动并行化](#自动并行化)
2. [自动向量化](#自动向量化)
3. [自动张量化](#自动张量化)
4. [内存访问优化](#内存访问优化)
5. [循环优化](#循环优化)
6. [存储优化](#存储优化)

---

## 自动并行化

### 依赖分析

```python
# tilelang/optimization/parallelization.py
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple
from enum import Enum

class DependencyType(Enum):
    """依赖类型"""
    RAW = "RAW"    # Read After Write (真依赖)
    WAR = "WAR"    # Write After Read (反依赖)
    WAW = "WAW"    # Write After Write (输出依赖)

@dataclass
class Dependency:
    """
    依赖关系
    
    属性:
    - source: 源语句
    - sink: 目标语句
    - dep_type: 依赖类型
    - distance: 依赖距离向量
    """
    source: 'Stmt'
    sink: 'Stmt'
    dep_type: DependencyType
    distance: List[int]

class DependencyAnalyzer:
    """
    依赖分析器
    
    分析语句间的依赖关系
    """
    
    def analyze(self, loop_nest: LoopNest) -> List[Dependency]:
        """
        分析循环嵌套中的依赖
        
        算法:
        1. 收集所有内存访问
        2. 构建依赖图
        3. 计算依赖距离
        """
        dependencies = []
        
        # 1. 收集所有内存访问
        accesses = self.collect_memory_accesses(loop_nest)
        
        # 2. 分析每对访问
        for i, access1 in enumerate(accesses):
            for access2 in accesses[i+1:]:
                dep = self.check_dependency(access1, access2)
                if dep:
                    dependencies.append(dep)
        
        return dependencies
    
    def check_dependency(self, access1: MemoryAccess, 
                        access2: MemoryAccess) -> Optional[Dependency]:
        """
        检查两个访问之间是否存在依赖
        
        条件:
        - 访问同一内存位置
        - 至少有一个是写操作
        """
        # 检查是否访问同一缓冲区
        if access1.buffer != access2.buffer:
            return None
        
        # 检查索引是否可能相同
        if not self.indices_may_overlap(access1.indices, access2.indices):
            return None
        
        # 确定依赖类型
        if access1.is_write and access2.is_read:
            dep_type = DependencyType.RAW
        elif access1.is_read and access2.is_write:
            dep_type = DependencyType.WAR
        elif access1.is_write and access2.is_write:
            dep_type = DependencyType.WAW
        else:
            return None  # RAR 不是依赖
        
        # 计算依赖距离
        distance = self.compute_distance(access1, access2)
        
        return Dependency(
            source=access1.stmt,
            sink=access2.stmt,
            dep_type=dep_type,
            distance=distance
        )
    
    def compute_distance(self, access1: MemoryAccess, 
                        access2: MemoryAccess) -> List[int]:
        """
        计算依赖距离向量
        
        距离向量表示依赖跨越的循环迭代次数
        """
        # 使用 GCD 测试计算距离
        pass

class Parallelizer:
    """
    并行化器
    
    将串行循环转换为并行循环
    """
    
    def parallelize(self, loop_nest: LoopNest) -> LoopNest:
        """
        并行化循环嵌套
        
        策略:
        1. 分析依赖
        2. 识别可并行循环
        3. 应用并行化变换
        """
        # 1. 分析依赖
        deps = DependencyAnalyzer().analyze(loop_nest)
        
        # 2. 识别可并行循环
        parallelizable = self.identify_parallelizable_loops(loop_nest, deps)
        
        # 3. 应用并行化
        for loop_idx in parallelizable:
            loop_nest = self.apply_parallelization(loop_nest, loop_idx)
        
        return loop_nest
    
    def identify_parallelizable_loops(self, loop_nest: LoopNest, 
                                      deps: List[Dependency]) -> List[int]:
        """
        识别可并行循环
        
        条件:
        - 无跨迭代的真依赖
        - 可通过私有化消除反依赖和输出依赖
        """
        parallelizable = []
        
        for i, loop in enumerate(loop_nest.loops):
            # 检查是否有跨迭代的真依赖
            has_true_dep = False
            
            for dep in deps:
                if dep.dep_type == DependencyType.RAW:
                    # 检查依赖距离
                    if dep.distance[i] != 0:
                        has_true_dep = True
                        break
            
            if not has_true_dep:
                parallelizable.append(i)
        
        return parallelizable
    
    def apply_parallelization(self, loop_nest: LoopNest, 
                             loop_idx: int) -> LoopNest:
        """
        应用并行化
        
        变换:
        - 将循环类型改为 parallel
        - 添加线程绑定注解
        """
        loop = loop_nest.loops[loop_idx]
        loop.kind = "parallel"
        loop.annotations["parallel"] = True
        
        return loop_nest
```

---

## 自动向量化

### 向量化分析

```python
# tilelang/optimization/vectorization.py
class Vectorizer:
    """
    向量化器
    
    将标量操作转换为 SIMD 操作
    """
    
    def vectorize(self, loop_nest: LoopNest, target: str) -> LoopNest:
        """
        向量化循环嵌套
        
        策略:
        1. 识别可向量化循环
        2. 确定向量宽度
        3. 应用向量化变换
        """
        # 1. 识别可向量化循环
        vectorizable = self.identify_vectorizable_loops(loop_nest)
        
        # 2. 确定向量宽度
        vector_width = self.get_vector_width(target)
        
        # 3. 应用向量化
        for loop_idx in vectorizable:
            loop_nest = self.apply_vectorization(
                loop_nest, loop_idx, vector_width
            )
        
        return loop_nest
    
    def identify_vectorizable_loops(self, loop_nest: LoopNest) -> List[int]:
        """
        识别可向量化循环
        
        条件:
        - 循环迭代独立
        - 内存访问连续
        - 无函数调用
        """
        vectorizable = []
        
        for i, loop in enumerate(loop_nest.loops):
            # 检查迭代独立性
            if not self.is_iteration_independent(loop):
                continue
            
            # 检查内存访问连续性
            if not self.has_contiguous_access(loop):
                continue
            
            # 检查无函数调用
            if self.has_function_calls(loop):
                continue
            
            vectorizable.append(i)
        
        return vectorizable
    
    def get_vector_width(self, target: str) -> int:
        """
        获取向量宽度
        
        根据目标平台确定
        """
        widths = {
            "cuda": 4,      # float4
            "rocm": 4,      # float4
            "metal": 4,     # float4
            "llvm": 8,      # AVX-256
            "llvm-avx512": 16,  # AVX-512
        }
        return widths.get(target, 4)
    
    def apply_vectorization(self, loop_nest: LoopNest, 
                           loop_idx: int, width: int) -> LoopNest:
        """
        应用向量化
        
        变换:
        1. 将循环步长乘以向量宽度
        2. 将循环类型改为 vectorized
        3. 替换标量操作为向量操作
        """
        loop = loop_nest.loops[loop_idx]
        
        # 修改步长
        loop.step *= width
        
        # 修改类型
        loop.kind = "vectorized"
        loop.annotations["vector_width"] = width
        
        # 替换操作
        loop.body = self.replace_scalar_with_vector(loop.body, width)
        
        return loop_nest
```

---

## 自动张量化

### Tensor Core 映射

```python
# tilelang/optimization/tensorization.py
class Tensorizer:
    """
    张量化器
    
    将矩阵乘法映射到 Tensor Core
    """
    
    # Tensor Core 配置
    TENSOR_CORE_CONFIGS = {
        # (架构, 数据类型) -> (M, N, K, 指令)
        (70, "float16"): (16, 16, 16, "wmma_m16n16k16_f16"),
        (75, "float16"): (16, 16, 16, "wmma_m16n16k16_f16"),
        (80, "float16"): (16, 16, 16, "mma_m16n16k16_f16"),
        (80, "int8"): (16, 16, 16, "mma_m16n16k16_s8"),
        (89, "float16"): (16, 16, 16, "mma_m16n16k16_f16"),
        (90, "float16"): (64, 64, 16, "wgmma_m64n64k16_f16"),
        (90, "int8"): (64, 64, 16, "wgmma_m64n64k16_s8"),
    }
    
    def tensorize(self, loop_nest: LoopNest, 
                  target: str, arch: int) -> LoopNest:
        """
        张量化循环嵌套
        
        策略:
        1. 检测矩阵乘法模式
        2. 选择 Tensor Core 配置
        3. 应用张量化变换
        """
        # 1. 检测矩阵乘法模式
        patterns = self.detect_gemm_patterns(loop_nest)
        
        if not patterns:
            return loop_nest
        
        # 2. 选择配置
        config = self.select_config(target, arch, patterns[0].dtype)
        
        # 3. 应用张量化
        for pattern in patterns:
            loop_nest = self.apply_tensorization(
                loop_nest, pattern, config
            )
        
        return loop_nest
    
    def detect_gemm_patterns(self, loop_nest: LoopNest) -> List[GEMMPattern]:
        """
        检测矩阵乘法模式
        
        模式:
        C[i, j] += A[i, k] * B[k, j]
        """
        patterns = []
        
        # 遍历所有语句
        for stmt in loop_nest.get_statements():
            # 检查是否是累加赋值
            if not self.is_accumulate_assign(stmt):
                continue
            
            # 检查是否是矩阵乘法
            pattern = self.match_gemm_pattern(stmt)
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def select_config(self, target: str, arch: int, 
                     dtype: str) -> Tuple[int, int, int, str]:
        """
        选择 Tensor Core 配置
        """
        if target != "cuda":
            # 非 NVIDIA GPU，使用其他矩阵加速器
            return self.get_non_cuda_config(target, dtype)
        
        key = (arch, dtype)
        if key in self.TENSOR_CORE_CONFIGS:
            return self.TENSOR_CORE_CONFIGS[key]
        
        # 默认配置
        return (16, 16, 16, "wmma_m16n16k16_f16")
    
    def apply_tensorization(self, loop_nest: LoopNest, 
                           pattern: GEMMPattern,
                           config: Tuple[int, int, int, str]) -> LoopNest:
        """
        应用张量化
        
        变换:
        1. 分块循环以匹配 Tensor Core 形状
        2. 替换计算为 Tensor Core 指令
        3. 插入数据加载/存储指令
        """
        M, N, K, instruction = config
        
        # 1. 分块
        loop_nest = self.tile_for_tensor_core(loop_nest, pattern, M, N, K)
        
        # 2. 替换计算
        loop_nest = self.replace_with_tensor_core(
            loop_nest, pattern, instruction
        )
        
        return loop_nest

@dataclass
class GEMMPattern:
    """
    矩阵乘法模式
    
    属性:
    - A: 矩阵 A
    - B: 矩阵 B
    - C: 矩阵 C
    - i_loop: i 循环索引
    - j_loop: j 循环索引
    - k_loop: k 循环索引
    - dtype: 数据类型
    """
    A: Buffer
    B: Buffer
    C: Buffer
    i_loop: int
    j_loop: int
    k_loop: int
    dtype: str
```

---

## 内存访问优化

### 合并访问优化

```python
# tilelang/optimization/memory.py
class MemoryAccessOptimizer:
    """
    内存访问优化器
    
    优化内存访问模式以提高带宽利用率
    """
    
    def optimize(self, loop_nest: LoopNest, target: str) -> LoopNest:
        """
        优化内存访问
        
        策略:
        1. 合并访问优化
        2. 共享内存优化
        3. 异步拷贝优化
        """
        # 1. 合并访问优化
        loop_nest = self.optimize_coalesced_access(loop_nest)
        
        # 2. 共享内存优化
        loop_nest = self.optimize_shared_memory(loop_nest, target)
        
        # 3. 异步拷贝优化
        loop_nest = self.insert_async_copy(loop_nest, target)
        
        return loop_nest
    
    def optimize_coalesced_access(self, loop_nest: LoopNest) -> LoopNest:
        """
        优化合并内存访问
        
        合并访问条件:
        - 相邻线程访问相邻地址
        - 访问对齐
        """
        # 分析访问模式
        accesses = self.analyze_access_patterns(loop_nest)
        
        for access in accesses:
            if not access.is_coalesced():
                # 重排线程索引
                loop_nest = self.reorder_for_coalescing(
                    loop_nest, access
                )
        
        return loop_nest
    
    def optimize_shared_memory(self, loop_nest: LoopNest, 
                               target: str) -> LoopNest:
        """
        优化共享内存访问
        
        策略:
        1. 避免 Bank Conflict
        2. 使用 Padding
        3. 使用 Swizzling
        """
        # 检测 Bank Conflict
        conflicts = self.detect_bank_conflicts(loop_nest)
        
        for conflict in conflicts:
            # 方法1: Padding
            loop_nest = self.apply_padding(loop_nest, conflict)
            
            # 方法2: Swizzling (如果 Padding 不够)
            if conflict.severity > 0.5:
                loop_nest = self.apply_swizzling(loop_nest, conflict)
        
        return loop_nest
    
    def insert_async_copy(self, loop_nest: LoopNest, 
                         target: str) -> LoopNest:
        """
        插入异步拷贝
        
        策略:
        - Ampere+: 使用 cp.async
        - Hopper+: 使用 TMA
        """
        if target != "cuda":
            return loop_nest
        
        arch = self.get_cuda_arch()
        
        if arch >= 90:  # Hopper
            return self.insert_tma_copy(loop_nest)
        elif arch >= 80:  # Ampere
            return self.insert_cp_async(loop_nest)
        
        return loop_nest

class BankConflictAnalyzer:
    """
    Bank Conflict 分析器
    """
    
    NUM_BANKS = 32  # NVIDIA GPU 共享内存 Bank 数量
    
    def analyze(self, shared_buffer: Buffer, 
                access_pattern: AccessPattern) -> BankConflictReport:
        """
        分析 Bank Conflict
        
        返回:
        - 是否有冲突
        - 冲突严重程度
        - 冲突位置
        """
        conflicts = []
        
        for access in access_pattern.accesses:
            # 计算访问的 Bank
            bank = self.compute_bank(access.index, shared_buffer)
            
            # 检查是否有多个线程访问同一 Bank
            # ...
        
        return BankConflictReport(
            has_conflict=len(conflicts) > 0,
            severity=self.compute_severity(conflicts),
            conflicts=conflicts
        )
    
    def compute_bank(self, index: int, buffer: Buffer) -> int:
        """
        计算访问地址对应的 Bank
        
        Bank = (address / 4) % 32
        (假设每个 Bank 宽度为 4 字节)
        """
        return (index * buffer.dtype_size // 4) % self.NUM_BANKS
```

---

## 循环优化

### 循环展开

```python
# tilelang/optimization/loop.py
class LoopUnroller:
    """
    循环展开器
    
    展开循环以减少循环开销和增加指令级并行
    """
    
    def unroll(self, loop_nest: LoopNest, 
               max_unroll_factor: int = 8) -> LoopNest:
        """
        展开循环
        
        策略:
        1. 分析循环收益
        2. 决定展开因子
        3. 应用展开
        """
        for i, loop in enumerate(loop_nest.loops):
            # 计算展开收益
            benefit = self.compute_unroll_benefit(loop)
            
            if benefit > 0:
                # 决定展开因子
                factor = min(
                    loop.get_extent(),
                    max_unroll_factor,
                    self.compute_max_unroll_factor(loop)
                )
                
                # 应用展开
                if factor == loop.get_extent():
                    # 完全展开
                    loop_nest = self.fully_unroll(loop_nest, i)
                else:
                    # 部分展开
                    loop_nest = self.partially_unroll(loop_nest, i, factor)
        
        return loop_nest
    
    def compute_unroll_benefit(self, loop: For) -> float:
        """
        计算展开收益
        
        考虑因素:
        - 循环开销
        - 指令级并行
        - 寄存器压力
        """
        # 循环开销
        overhead = loop.get_extent() * LOOP_OVERHEAD
        
        # 指令级并行收益
        ilp_benefit = self.estimate_ilp_benefit(loop)
        
        # 寄存器压力成本
        reg_cost = self.estimate_register_pressure(loop)
        
        return ilp_benefit - overhead - reg_cost

class LoopFusion:
    """
    循环融合
    
    合并多个循环以提高数据局部性
    """
    
    def fuse(self, loops: List[LoopNest]) -> LoopNest:
        """
        融合多个循环
        
        条件:
        - 循环范围相同
        - 无依赖冲突
        """
        # 检查是否可以融合
        if not self.can_fuse(loops):
            return loops[0]  # 返回第一个循环
        
        # 创建融合后的循环
        fused = LoopNest()
        fused.loops = loops[0].loops.copy()
        
        # 合并循环体
        for loop in loops:
            fused.body.extend(loop.body)
        
        return fused
    
    def can_fuse(self, loops: List[LoopNest]) -> bool:
        """
        检查是否可以融合
        
        条件:
        1. 循环范围相同
        2. 无反依赖冲突
        """
        # 检查循环范围
        if not self.same_loop_bounds(loops):
            return False
        
        # 检查依赖
        if self.has_anti_dependency(loops):
            return False
        
        return True
```

---

## 存储优化

### 缓存优化

```python
# tilelang/optimization/storage.py
class StorageOptimizer:
    """
    存储优化器
    
    优化数据存储和访问
    """
    
    def optimize(self, loop_nest: LoopNest, target: str) -> LoopNest:
        """
        优化存储
        
        策略:
        1. 缓存读写
        2. 存储折叠
        3. 内存布局优化
        """
        # 1. 缓存读写
        loop_nest = self.cache_reads_writes(loop_nest, target)
        
        # 2. 存储折叠
        loop_nest = self.storage_folding(loop_nest)
        
        # 3. 内存布局优化
        loop_nest = self.optimize_memory_layout(loop_nest)
        
        return loop_nest
    
    def cache_reads_writes(self, loop_nest: LoopNest, 
                          target: str) -> LoopNest:
        """
        缓存读写
        
        将频繁访问的数据缓存到共享内存
        """
        # 分析访问频率
        access_freq = self.analyze_access_frequency(loop_nest)
        
        # 选择要缓存的数据
        for buffer, freq in access_freq.items():
            if freq > CACHE_THRESHOLD:
                # 插入缓存
                loop_nest = self.insert_cache(
                    loop_nest, buffer, target
                )
        
        return loop_nest
    
    def insert_cache(self, loop_nest: LoopNest, 
                    buffer: Buffer, target: str) -> LoopNest:
        """
        插入缓存
        
        在循环开始处加载数据到共享内存
        在循环结束处写回数据
        """
        # 创建共享内存缓冲区
        shared_buffer = Buffer(
            name=f"{buffer.name}_shared",
            shape=buffer.shape,
            dtype=buffer.dtype,
            scope="shared"
        )
        
        # 插入加载
        load_stmt = self.create_load_to_shared(buffer, shared_buffer)
        loop_nest.body.insert(0, load_stmt)
        
        # 插入存储
        store_stmt = self.create_store_from_shared(shared_buffer, buffer)
        loop_nest.body.append(store_stmt)
        
        # 替换访问
        loop_nest = self.replace_buffer_access(
            loop_nest, buffer, shared_buffer
        )
        
        return loop_nest

class MemoryLayoutOptimizer:
    """
    内存布局优化器
    
    优化数据的内存布局
    """
    
    def optimize_layout(self, buffer: Buffer, 
                       access_pattern: AccessPattern) -> Buffer:
        """
        优化内存布局
        
        策略:
        1. 行主序 vs 列主序
        2. Padding
        3. Swizzling
        """
        # 分析访问模式
        pattern = access_pattern.analyze()
        
        if pattern == "row_major":
            buffer.layout = "row_major"
        elif pattern == "column_major":
            buffer.layout = "column_major"
        else:
            buffer.layout = "swizzled"
        
        # 添加 Padding 以避免 Bank Conflict
        if access_pattern.has_bank_conflict():
            buffer = self.add_padding(buffer)
        
        return buffer
```

## 下一步

继续学习 [代码生成](04-code-generation.md)，了解 TileLang 如何生成目标平台代码。
