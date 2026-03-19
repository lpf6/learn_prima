# TileLang 运行时系统

## 目录

1. [内核启动器](#内核启动器)
2. [内存管理器](#内存管理器)
3. [自动调优框架](#自动调优框架)
4. [性能分析器](#性能分析器)

---

## 内核启动器

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    内核启动器架构                             │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              用户调用                                 │    │
│  │  kernel(a, b, c)                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              参数准备                                 │    │
│  │  - 类型检查                                          │    │
│  │  - 内存分配                                          │    │
│  │  - 数据传输                                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              内核缓存查找                             │    │
│  │  - 检查内存缓存                                      │    │
│  │  - 检查磁盘缓存                                      │    │
│  │  - JIT 编译 (如果需要)                               │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              内核启动                                 │    │
│  │  - 配置 Grid/Block                                   │    │
│  │  - 设置共享内存                                      │    │
│  │  - 启动内核                                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              结果处理                                 │    │
│  │  - 同步                                              │    │
│  │  - 数据传输                                          │    │
│  │  - 返回结果                                          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 实现

```python
# tilelang/runtime/launcher.py
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor

@dataclass
class LaunchConfig:
    """
    内核启动配置
    
    属性:
    - grid: Grid 维度 (grid_x, grid_y, grid_z)
    - block: Block 维度 (block_x, block_y, block_z)
    - shared_mem: 共享内存大小 (字节)
    - stream: CUDA 流
    """
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    shared_mem: int = 0
    stream: int = 0

class KernelLauncher:
    """
    内核启动器
    
    管理内核编译、缓存和执行
    """
    
    def __init__(self, cache_dir: str = None, backend: str = "nvrtc"):
        self.cache_dir = cache_dir or self.default_cache_dir()
        self.backend = backend
        
        # 缓存
        self.memory_cache: Dict[str, CompiledKernel] = {}
        self.disk_cache_enabled = True
        
        # JIT 编译器
        self.jit_compilers = {
            "nvrtc": NVRTCCompiler(),
            "nvcc": NVCCCompiler(),
            "hiprtc": HIPRTCCompiler(),
            "metal": MetalCompiler(),
        }
        
        # 线程池 (用于异步编译)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 锁
        self.cache_lock = threading.Lock()
    
    def launch(self, kernel_func: 'TileLangKernel', 
               *args, **kwargs) -> Any:
        """
        启动内核
        
        流程:
        1. 准备参数
        2. 获取或编译内核
        3. 启动内核
        4. 返回结果
        """
        # 1. 准备参数
        params = self.prepare_params(kernel_func, args, kwargs)
        
        # 2. 获取或编译内核
        kernel = self.get_or_compile_kernel(kernel_func, params)
        
        # 3. 启动内核
        return self.launch_kernel(kernel, params)
    
    def prepare_params(self, kernel_func: 'TileLangKernel',
                       args: tuple, kwargs: dict) -> Dict[str, Any]:
        """
        准备参数
        
        包括:
        - 类型检查
        - 内存分配
        - 数据传输
        """
        params = {}
        
        # 获取参数签名
        signature = kernel_func.signature
        
        # 匹配位置参数
        for i, (name, expected_type) in enumerate(signature.items()):
            if i < len(args):
                value = args[i]
            elif name in kwargs:
                value = kwargs[name]
            else:
                raise ValueError(f"Missing parameter: {name}")
            
            # 类型检查
            if not self.check_type(value, expected_type):
                raise TypeError(
                    f"Parameter {name}: expected {expected_type}, "
                    f"got {type(value)}"
                )
            
            # 内存分配 (如果需要)
            if self.is_device_pointer(value):
                params[name] = value
            else:
                # 分配设备内存并传输数据
                params[name] = self.to_device(value)
        
        return params
    
    def get_or_compile_kernel(self, kernel_func: 'TileLangKernel',
                              params: Dict[str, Any]) -> CompiledKernel:
        """
        获取或编译内核
        
        查找顺序:
        1. 内存缓存 (最快)
        2. 磁盘缓存 (较快)
        3. JIT 编译 (最慢)
        """
        # 生成缓存键
        cache_key = self.generate_cache_key(kernel_func, params)
        
        # 1. 检查内存缓存
        with self.cache_lock:
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key]
        
        # 2. 检查磁盘缓存
        if self.disk_cache_enabled:
            kernel = self.load_from_disk_cache(cache_key)
            if kernel:
                with self.cache_lock:
                    self.memory_cache[cache_key] = kernel
                return kernel
        
        # 3. JIT 编译
        kernel = self.jit_compile(kernel_func, params)
        
        # 缓存
        with self.cache_lock:
            self.memory_cache[cache_key] = kernel
        
        # 异步写入磁盘缓存
        if self.disk_cache_enabled:
            self.executor.submit(
                self.save_to_disk_cache, cache_key, kernel
            )
        
        return kernel
    
    def generate_cache_key(self, kernel_func: 'TileLangKernel',
                          params: Dict[str, Any]) -> str:
        """
        生成缓存键
        
        考虑:
        - 内核源码
        - 目标平台
        - 架构版本
        - 参数形状
        """
        key_parts = [
            kernel_func.source_code,
            kernel_func.target,
            str(kernel_func.arch),
        ]
        
        # 添加参数形状
        for name, value in params.items():
            if hasattr(value, 'shape'):
                key_parts.append(f"{name}:{value.shape}")
        
        # 计算哈希
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def jit_compile(self, kernel_func: 'TileLangKernel',
                    params: Dict[str, Any]) -> CompiledKernel:
        """
        JIT 编译内核
        
        步骤:
        1. 生成代码
        2. 编译为二进制
        3. 加载模块
        """
        # 选择编译器
        compiler = self.jit_compilers.get(self.backend)
        if not compiler:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        # 生成代码
        code = kernel_func.generate_code(params)
        
        # 编译
        binary = compiler.compile(code, kernel_func.arch)
        
        # 加载
        module = self.load_module(binary)
        
        return CompiledKernel(module, kernel_func.metadata)
    
    def launch_kernel(self, kernel: CompiledKernel,
                      params: Dict[str, Any]) -> Any:
        """
        启动内核
        
        步骤:
        1. 配置启动参数
        2. 设置共享内存
        3. 启动内核
        4. 同步
        """
        # 1. 配置启动参数
        config = kernel.get_launch_config(params)
        
        # 2. 设置共享内存
        if config.shared_mem > 0:
            self.set_shared_memory(config.shared_mem)
        
        # 3. 启动内核
        kernel.launch(
            grid=config.grid,
            block=config.block,
            args=list(params.values()),
            stream=config.stream
        )
        
        # 4. 同步
        if kernel.synchronous:
            self.synchronize()
        
        # 5. 返回结果
        return self.get_result(kernel, params)
```

### NVRTC 编译器

```python
# tilelang/runtime/compiler/nvrtc.py
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

class NVRTCCompiler:
    """
    NVRTC (NVIDIA Runtime Compiler) 编译器
    
    特点:
    - 运行时编译，无需 nvcc
    - 编译速度快
    - 支持大部分 CUDA 功能
    """
    
    def __init__(self):
        self.include_paths = self.get_include_paths()
        self.default_options = [
            "-arch=sm_80",
            "-std=c++17",
            "-use_fast_math",
        ]
    
    def compile(self, source: str, arch: int = 80) -> bytes:
        """
        编译 CUDA 源码
        
        参数:
        - source: CUDA C++ 源码
        - arch: 目标架构 (SM 版本)
        
        返回:
        - 编译后的二进制 (PTX 或 CUBIN)
        """
        # 构建编译选项
        options = self.default_options.copy()
        options[0] = f"-arch=sm_{arch}"
        
        # 添加包含路径
        for path in self.include_paths:
            options.append(f"-I{path}")
        
        # 编译
        from pycuda.compiler import compile
        try:
            binary = compile(source, options=options)
        except Exception as e:
            raise CompilationError(f"NVRTC compilation failed: {e}")
        
        return binary
    
    def get_include_paths(self) -> List[str]:
        """
        获取 CUDA 包含路径
        """
        import os
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        return [
            f"{cuda_home}/include",
        ]
```

---

## 内存管理器

### 设计

```python
# tilelang/runtime/memory.py
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
from collections import defaultdict

@dataclass
class MemoryBlock:
    """
    内存块
    
    属性:
    - ptr: 指针
    - size: 大小
    - dtype: 数据类型
    - shape: 形状
    - device: 设备 ID
    - in_use: 是否使用中
    """
    ptr: int
    size: int
    dtype: str
    shape: Tuple[int, ...]
    device: int
    in_use: bool = False

class MemoryManager:
    """
    内存管理器
    
    管理设备内存的分配、释放和复用
    """
    
    def __init__(self, device: str = "cuda", device_id: int = 0):
        self.device = device
        self.device_id = device_id
        
        # 内存池
        # 按大小分组
        self.free_blocks: Dict[int, List[MemoryBlock]] = defaultdict(list)
        self.used_blocks: Dict[int, MemoryBlock] = {}
        
        # 统计
        self.total_allocated = 0
        self.total_used = 0
        self.peak_used = 0
        
        # 锁
        self.lock = threading.Lock()
        
        # 初始化设备
        self.init_device()
    
    def allocate(self, shape: Tuple[int, ...], dtype: str) -> MemoryBlock:
        """
        分配内存
        
        策略:
        1. 检查内存池
        2. 如果找到合适大小，复用
        3. 否则，分配新内存
        """
        size = self.compute_size(shape, dtype)
        
        with self.lock:
            # 1. 检查内存池
            # 查找最接近的大小
            for block_size in sorted(self.free_blocks.keys()):
                if block_size >= size:
                    # 找到合适的块
                    block = self.free_blocks[block_size].pop()
                    if not self.free_blocks[block_size]:
                        del self.free_blocks[block_size]
                    
                    block.in_use = True
                    block.shape = shape
                    block.dtype = dtype
                    
                    self.used_blocks[block.ptr] = block
                    self.total_used += block.size
                    self.peak_used = max(self.peak_used, self.total_used)
                    
                    return block
            
            # 2. 分配新内存
            ptr = self.device_malloc(size)
            
            block = MemoryBlock(
                ptr=ptr,
                size=size,
                dtype=dtype,
                shape=shape,
                device=self.device_id,
                in_use=True
            )
            
            self.used_blocks[ptr] = block
            self.total_allocated += size
            self.total_used += size
            self.peak_used = max(self.peak_used, self.total_used)
            
            return block
    
    def free(self, ptr: int):
        """
        释放内存
        
        策略:
        - 不立即释放，而是放回内存池
        - 定期清理内存池
        """
        with self.lock:
            if ptr not in self.used_blocks:
                return
            
            block = self.used_blocks[ptr]
            block.in_use = False
            
            # 从使用列表移除
            del self.used_blocks[ptr]
            self.total_used -= block.size
            
            # 添加到空闲列表
            self.free_blocks[block.size].append(block)
            
            # 检查是否需要清理
            if self.should_cleanup():
                self.cleanup()
    
    def cleanup(self):
        """
        清理内存池
        
        策略:
        - 保留最近使用的块
        - 释放其他块
        """
        with self.lock:
            # 计算保留阈值
            keep_threshold = self.peak_used * 0.5
            
            current_free = sum(
                size * len(blocks) 
                for size, blocks in self.free_blocks.items()
            )
            
            if current_free <= keep_threshold:
                return
            
            # 释放多余的块
            for size in sorted(self.free_blocks.keys(), reverse=True):
                blocks = self.free_blocks[size]
                
                while blocks and current_free > keep_threshold:
                    block = blocks.pop()
                    self.device_free(block.ptr)
                    self.total_allocated -= block.size
                    current_free -= block.size
                
                if not blocks:
                    del self.free_blocks[size]
    
    def should_cleanup(self) -> bool:
        """
        判断是否需要清理
        
        条件:
        - 空闲内存超过峰值使用的 50%
        """
        current_free = sum(
            size * len(blocks) 
            for size, blocks in self.free_blocks.items()
        )
        return current_free > self.peak_used * 0.5
    
    def compute_size(self, shape: Tuple[int, ...], dtype: str) -> int:
        """
        计算内存大小
        """
        import numpy as np
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        
        dtype_size = np.dtype(dtype).itemsize
        return num_elements * dtype_size
    
    def device_malloc(self, size: int) -> int:
        """
        设备内存分配
        """
        if self.device == "cuda":
            import pycuda.driver as cuda
            ptr = cuda.mem_alloc(size)
            return int(ptr)
        elif self.device == "rocm":
            # ROCm 内存分配
            pass
        elif self.device == "metal":
            # Metal 内存分配
            pass
    
    def device_free(self, ptr: int):
        """
        设备内存释放
        """
        if self.device == "cuda":
            import pycuda.driver as cuda
            cuda.mem_free(ptr)
```

---

## 自动调优框架

### 设计

```python
# tilelang/runtime/autotune.py
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import time
import random
import json
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np

@dataclass
class TuningConfig:
    """
    调优配置
    
    属性:
    - search_space: 搜索空间
    - num_trials: 尝试次数
    - early_stopping: 早停次数
    - num_workers: 并行工作数
    - timeout: 超时时间
    """
    search_space: Dict[str, List[Any]]
    num_trials: int = 1000
    early_stopping: int = 100
    num_workers: int = 4
    timeout: int = 3600  # 1 hour

@dataclass
class TuningResult:
    """
    调优结果
    
    属性:
    - best_config: 最优配置
    - best_latency: 最优延迟
    - all_results: 所有结果
    - tuning_time: 调优时间
    """
    best_config: Dict[str, Any]
    best_latency: float
    all_results: List[Tuple[Dict[str, Any], float]]
    tuning_time: float

class AutoTuner:
    """
    自动调优器
    
    搜索最优调度配置
    """
    
    def __init__(self, kernel_func: 'TileLangKernel', 
                 config: TuningConfig):
        self.kernel_func = kernel_func
        self.config = config
        
        # 代价模型
        self.cost_model = XGBoostCostModel()
        
        # 历史记录
        self.history: List[Tuple[Dict[str, Any], float]] = []
        
        # 最优结果
        self.best_config = None
        self.best_latency = float('inf')
        
        # 早停计数
        self.no_improvement_count = 0
    
    def tune(self, args: tuple, kwargs: dict = None) -> TuningResult:
        """
        执行调优
        
        算法:
        1. 初始随机采样
        2. 训练代价模型
        3. 基于模型采样
        4. 迭代优化
        """
        start_time = time.time()
        kwargs = kwargs or {}
        
        # 阶段 1: 初始随机采样
        initial_trials = min(100, self.config.num_trials // 10)
        for _ in range(initial_trials):
            config = self.random_sample()
            latency = self.evaluate(config, args, kwargs)
            self.update(config, latency)
        
        # 阶段 2: 基于模型的优化
        remaining_trials = self.config.num_trials - initial_trials
        for i in range(remaining_trials):
            # 检查早停
            if self.no_improvement_count >= self.config.early_stopping:
                break
            
            # 检查超时
            if time.time() - start_time > self.config.timeout:
                break
            
            # 采样配置
            if random.random() < 0.1:
                # 10% 随机探索
                config = self.random_sample()
            else:
                # 90% 基于模型
                config = self.model_based_sample()
            
            # 评估
            latency = self.evaluate(config, args, kwargs)
            
            # 更新
            self.update(config, latency)
        
        tuning_time = time.time() - start_time
        
        return TuningResult(
            best_config=self.best_config,
            best_latency=self.best_latency,
            all_results=self.history,
            tuning_time=tuning_time
        )
    
    def random_sample(self) -> Dict[str, Any]:
        """
        随机采样配置
        """
        config = {}
        for key, values in self.config.search_space.items():
            config[key] = random.choice(values)
        return config
    
    def model_based_sample(self) -> Dict[str, Any]:
        """
        基于代价模型采样
        """
        # 生成候选配置
        candidates = []
        for _ in range(100):
            config = self.random_sample()
            candidates.append(config)
        
        # 使用代价模型预测
        predictions = []
        for config in candidates:
            pred = self.cost_model.predict(config)
            predictions.append((config, pred))
        
        # 选择预测最优的
        predictions.sort(key=lambda x: x[1])
        return predictions[0][0]
    
    def evaluate(self, config: Dict[str, Any], 
                 args: tuple, kwargs: dict) -> float:
        """
        评估配置
        
        返回平均延迟 (毫秒)
        """
        # 编译内核
        kernel = self.kernel_func.compile(**config)
        
        # 预热
        for _ in range(5):
            kernel(*args, **kwargs)
        
        # 测量
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            kernel(*args, **kwargs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
        
        # 返回中位数
        return np.median(latencies)
    
    def update(self, config: Dict[str, Any], latency: float):
        """
        更新状态
        """
        # 记录历史
        self.history.append((config, latency))
        
        # 更新最优
        if latency < self.best_latency:
            self.best_latency = latency
            self.best_config = config
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # 更新代价模型
        self.cost_model.update(config, latency)

class XGBoostCostModel:
    """
    基于 XGBoost 的代价模型
    
    预测配置的性能
    """
    
    def __init__(self):
        self.model = None
        self.features = []
        self.X = []
        self.y = []
    
    def extract_features(self, config: Dict[str, Any]) -> List[float]:
        """
        提取特征
        """
        features = []
        
        for key, value in config.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, bool):
                features.append(float(value))
            elif isinstance(value, str):
                # 哈希编码
                features.append(float(hash(value) % 1000) / 1000)
        
        return features
    
    def predict(self, config: Dict[str, Any]) -> float:
        """
        预测延迟
        """
        if self.model is None:
            return float('inf')
        
        features = self.extract_features(config)
        return self.model.predict([features])[0]
    
    def update(self, config: Dict[str, Any], latency: float):
        """
        更新模型
        """
        features = self.extract_features(config)
        
        self.X.append(features)
        self.y.append(latency)
        
        # 定期重新训练
        if len(self.X) % 50 == 0:
            self.train()
    
    def train(self):
        """
        训练模型
        """
        import xgboost as xgb
        
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
        )
        
        self.model.fit(self.X, self.y)
```

---

## 性能分析器

### 设计

```python
# tilelang/runtime/profiler.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import json

@dataclass
class ProfileResult:
    """
    性能分析结果
    
    属性:
    - kernel_name: 内核名称
    - latency_ms: 延迟 (毫秒)
    - throughput_tflops: 吞吐量 (TFLOPS)
    - bandwidth_gb_s: 带宽 (GB/s)
    - gpu_utilization: GPU 利用率
    - memory_usage: 内存使用
    """
    kernel_name: str
    latency_ms: float
    throughput_tflops: float
    bandwidth_gb_s: float
    gpu_utilization: float
    memory_usage: int

class Profiler:
    """
    性能分析器
    
    分析内核性能
    """
    
    def __init__(self):
        self.events: Dict[str, List[float]] = {}
        self.enabled = True
    
    def profile(self, kernel: CompiledKernel, 
                args: tuple, kwargs: dict = None,
                num_runs: int = 100) -> ProfileResult:
        """
        分析内核性能
        """
        kwargs = kwargs or {}
        
        # 预热
        for _ in range(10):
            kernel(*args, **kwargs)
        
        # 测量延迟
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            kernel(*args, **kwargs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        
        # 计算性能指标
        flops = self.estimate_flops(kernel, args, kwargs)
        bytes_moved = self.estimate_bytes_moved(kernel, args, kwargs)
        
        throughput_tflops = flops / (avg_latency / 1000) / 1e12
        bandwidth_gb_s = bytes_moved / (avg_latency / 1000) / 1e9
        
        # GPU 利用率
        gpu_utilization = self.get_gpu_utilization()
        
        # 内存使用
        memory_usage = self.get_memory_usage()
        
        return ProfileResult(
            kernel_name=kernel.name,
            latency_ms=avg_latency,
            throughput_tflops=throughput_tflops,
            bandwidth_gb_s=bandwidth_gb_s,
            gpu_utilization=gpu_utilization,
            memory_usage=memory_usage
        )
    
    def estimate_flops(self, kernel: CompiledKernel,
                       args: tuple, kwargs: dict) -> int:
        """
        估计 FLOPs
        """
        # 从内核元数据获取
        if hasattr(kernel, 'metadata') and 'flops' in kernel.metadata:
            return kernel.metadata['flops']
        
        # 否则，从参数推断
        # 例如，矩阵乘法: 2 * M * N * K
        pass
    
    def estimate_bytes_moved(self, kernel: CompiledKernel,
                             args: tuple, kwargs: dict) -> int:
        """
        估计内存移动量
        """
        # 从内核元数据获取
        if hasattr(kernel, 'metadata') and 'bytes_moved' in kernel.metadata:
            return kernel.metadata['bytes_moved']
        
        # 否则，从参数推断
        total_bytes = 0
        for arg in args:
            if hasattr(arg, 'nbytes'):
                total_bytes += arg.nbytes
        
        # 假设读写各一次
        return total_bytes * 2
    
    def get_gpu_utilization(self) -> float:
        """
        获取 GPU 利用率
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu / 100.0
        except:
            return 0.0
    
    def get_memory_usage(self) -> int:
        """
        获取内存使用
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used
        except:
            return 0
    
    def start_event(self, name: str):
        """
        开始计时事件
        """
        if not self.enabled:
            return
        
        if name not in self.events:
            self.events[name] = []
        
        self.events[name].append({
            'start': time.perf_counter(),
            'end': None
        })
    
    def end_event(self, name: str):
        """
        结束计时事件
        """
        if not self.enabled:
            return
        
        if name in self.events and self.events[name]:
            self.events[name][-1]['end'] = time.perf_counter()
    
    def get_event_time(self, name: str) -> float:
        """
        获取事件时间 (毫秒)
        """
        if name not in self.events:
            return 0.0
        
        times = []
        for event in self.events[name]:
            if event['end'] is not None:
                times.append((event['end'] - event['start']) * 1000)
        
        return sum(times) / len(times) if times else 0.0
    
    def summary(self) -> Dict[str, Any]:
        """
        生成摘要
        """
        summary = {}
        
        for name, events in self.events.items():
            times = [
                (e['end'] - e['start']) * 1000
                for e in events
                if e['end'] is not None
            ]
            
            if times:
                summary[name] = {
                    'count': len(times),
                    'total_ms': sum(times),
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                }
        
        return summary
    
    def export_chrome_trace(self, output_path: str):
        """
        导出 Chrome Trace 格式
        """
        events = []
        
        for name, event_list in self.events.items():
            for event in event_list:
                if event['end'] is None:
                    continue
                
                events.append({
                    'name': name,
                    'cat': 'kernel',
                    'ph': 'X',
                    'ts': event['start'] * 1e6,
                    'dur': (event['end'] - event['start']) * 1e6,
                    'pid': 1,
                    'tid': 1,
                })
        
        with open(output_path, 'w') as f:
            json.dump(events, f)
```

## 下一步

继续学习 [多后端支持](06-multi-backend.md)，了解 TileLang 如何支持多种硬件平台。
