# TileLang 枸架架构设计

## 目录

1. [编译器架构](01-compiler-architecture.md)
2. [IR 设计](02-ir-design.md)
3. [优化 Pass](03-optimization-pass.md)
4. [代码生成](04-code-generation.md)
5. [运行时系统](05-runtime-system.md)
6. [多后端支持](06-multi-backend.md)

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         用户代码 (Python DSL)                              │
│                    @tilelang.jit 装饰的函数                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      前端编译 (Frontend)                               │
│  - 解析 Python 代码                                            │
│  - 构建 AST (抽象语法树)                                          │
│  - 类型检查                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TensorIR 构建 (IR Builder)                            │
│  - 循环嵌套展开                                                │
│  - 内存访问分析                                                  │
│  - 并行化分析                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      优化 Pass (Optimizer)                               │
│  - 自动并行化                                                    │
│  - 自动向量化                                                    │
│  - 自动张量化                                                    │
│  - 内存访问优化                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    代码生成 (Codegen)                                │
│  - CUDA Codegen                                                   │
│  - ROCm/HIP Codegen                                                │
│  - Metal Codegen                                                   │
│  - Vulkan Codegen                                                  │
│  - Ascend Codegen                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    运行时系统 (Runtime)                               │
│  - 内核启动器                                                    │
│  - 内存管理器                                                    │
│  - 自动调优框架                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 娡块系统

```python
# tilelang/kernel/module.py
class TileLangModule:
    """
    TileLang 模块
    包含编译后的内核代码和元数据
    """
    
    def __init__(self, source: str, compiled_code: str, metadata: dict):
        self.source = source          # 原始 Python 源码
        self.compiled_code = compiled_code  # 编译后的代码
        self.metadata = metadata      # 元数据（参数、类型等）
        self.kernel_cache = {}        # 内核缓存
    
    def get_kernel(self, device: str, **kwargs):
        """
        获取或编译内核
        
        流程:
        1. 检查缓存
        2. 如果未缓存，则 JIT 编译
        3. 返回内核
        """
        cache_key = (device, tuple(sorted(kwargs.items())))
        
        if cache_key not in self.kernel_cache:
            # JIT 编译
            kernel = self.jit_compile(device, **kwargs)
            self.kernel_cache[cache_key] = kernel
        
        return self.kernel_cache[cache_key]
```

### 2. JIT 编译器

```python
# tilelang/compiler/jit.py
class JITCompiler:
    """
    JIT 编译器
    支持多种编译后端
    """
    
    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self.compilers = {
            "nvrtc": NVRTCCompiler(),    # NVIDIA 运行时编译
            "nvcc": NVCCCompiler(),      # NVIDIA 离线编译
            "hiprtc": HIPRTCCompiler(),  # AMD 运行时编译
            "metal": MetalCompiler(),    # Apple Metal 编译
        }
    
    def compile(self, ir_module: tvm.IRModule, target: str) -> CompiledKernel:
        """
        编译 IR 模块
        
        步骤:
        1. 选择编译后端
        2. 生成代码
        3. 编译为二进制
        4. 返回可执行内核
        """
        # 1. 选择编译后端
        compiler = self.select_compiler(target)
        
        # 2. 生成代码
        code = compiler.generate_code(ir_module)
        
        # 3. 编译为二进制
        binary = compiler.compile_to_binary(code)
        
        # 4. 返回可执行内核
        return CompiledKernel(binary, ir_module.metadata)
    
    def select_compiler(self, target: str) -> Compiler:
        """
        选择编译器
        
        策略:
        - CUDA: 优先使用 NVRTC (快速编译)
        - ROCm: 使用 HIPRTC
        - Metal: 使用 Metal 编译器
        """
        if target == "cuda":
            return self.compilers.get(self.backend, self.compilers["nvrtc"])
        elif target == "rocm":
            return self.compilers["hiprtc"]
        elif target == "metal":
            return self.compilers["metal"]
        else:
            raise ValueError(f"Unsupported target: {target}")
```

### 3. 缓存系统

```python
# tilelang/cache.py
class KernelCache:
    """
    内核缓存系统
    缓存编译后的内核以加速后续调用
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or self.default_cache_dir()
        self.memory_cache = {}
        self.disk_cache = {}
    
    def get(self, key: str) -> Optional[CompiledKernel]:
        """
        获取缓存的内核
        
        查找顺序:
        1. 内存缓存 (最快)
        2. 磁盘缓存 (较快)
        3. 返回 None (需要重新编译)
        """
        # 1. 检查内存缓存
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # 2. 检查磁盘缓存
        cache_file = os.path.join(self.cache_dir, f"{key}.bin")
        if os.path.exists(cache_file):
            kernel = self.load_from_disk(cache_file)
            self.memory_cache[key] = kernel
            return kernel
        
        return None
    
    def put(self, key: str, kernel: CompiledKernel):
        """
        缓存内核
        
        策略:
        1. 存入内存缓存
        2. 异步写入磁盘缓存
        """
        # 1. 存入内存缓存
        self.memory_cache[key] = kernel
        
        # 2. 异步写入磁盘
        threading.Thread(
            target=self.save_to_disk,
            args=(os.path.join(self.cache_dir, f"{key}.bin"), kernel)
        ).start()
```

## 与 TVM 的关系

TileLang 基于 Apache TVM 构建，但提供了更高层的抽象：

```
TVM 架构:
┌─────────────────────────────────────────────────────────────┐
│                    TVM Stack                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Relay IR (中间表示)                        │   │
│  │  - 计算图表示                                          │   │
│  │  - 类型系统                                            │   │
│  │  - 内存布局                                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              TensorIR (张量中间表示)                    │   │
│  │  - 张量操作                                            │   │
│  │  - 循环嵌套                                            │   │
│  │  - 并行原语                                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Lowered IR (底层中间表示)                  │   │
│  │  - 硬件原语                                            │   │
│  │  - 内存指令                                            │   │
│  │  - 同步指令                                            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

TileLang 扩展:
┌─────────────────────────────────────────────────────────────┐
│                    TileLang Extensions                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Tile 抽象层                               │   │
│  │  - 块级编程                                            │   │
│  │  - 自动内存映射                                        │   │
│  │  - 架构无关原语                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              自动调度器                                │   │
│  │  - 搜索空间定义                                        │   │
│  │  - 代价模型                                            │   │
│  │  - 自动调优                                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              架构特定优化                              │   │
│  │  - Tensor Core 映射                                    │   │
│  │  - 异步内存拷贝                                        │   │
│  │  - TMA 支持                                            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 设计决策

### 为什么选择 TVM？

1. **成熟的编译器基础设施**
   - 经过大规模生产验证
   - 丰富的优化 Pass
   - 活跃的社区支持

2. **多后端支持**
   - 原生支持 CUDA, ROCm, Metal, Vulkan
   - 易于扩展新后端

3. **自动调优能力**
   - 内置 AutoScheduler
   - 代价模型
   - 搜索空间

### 为什么选择 Python DSL？

1. **易用性**
   - Python 语法简单
   - 丰富的生态系统
   - 易于集成

2. **灵活性**
   - 动态类型
   - 元编程支持
   - 快速原型开发

3. **可移植性**
   - 无需编译器工具链
   - 跨平台兼容

## 架构特定优化

### NVIDIA GPU

```python
# tilelang/target/cuda.py
class CUDATarget:
    """
    CUDA 目标平台
    """
    
    ARCH_FEATURES = {
        # Volta (SM 70)
        70: {
            "tensor_core": True,
            "tensor_core_type": "wmma",
            "tensor_core_shape": (16, 16, 16),
            "async_copy": False,
            "shared_memory": 96 * 1024,
            "warp_size": 32,
            "max_threads_per_block": 1024,
        },
        # Turing (SM 75)
        75: {
            "tensor_core": True,
            "tensor_core_type": "wmma",
            "tensor_core_shape": (16, 16, 16),
            "async_copy": True,  # cp.async
            "shared_memory": 64 * 1024,
            "warp_size": 32,
            "max_threads_per_block": 1024,
        },
        # Ampere (SM 80)
        80: {
            "tensor_core": True,
            "tensor_core_type": "mma",
            "tensor_core_shape": (16, 16, 16),
            "async_copy": True,
            "shared_memory": 164 * 1024,
            "warp_size": 32,
            "max_threads_per_block": 1024,
        },
        # Ada Lovelace (SM 89)
        89: {
            "tensor_core": True,
            "tensor_core_type": "mma",
            "tensor_core_shape": (16, 16, 16),
            "async_copy": True,
            "shared_memory": 100 * 1024,
            "warp_size": 32,
            "max_threads_per_block": 1024,
        },
        # Hopper (SM 90)
        90: {
            "tensor_core": True,
            "tensor_core_type": "wgmma",
            "tensor_core_shape": (64, 64, 16),
            "async_copy": True,
            "tma": True,  # Tensor Memory Accelerator
            "shared_memory": 228 * 1024,
            "warp_size": 32,
            "max_threads_per_block": 1024,
        },
    }
    
    def get_optimal_schedule(self, arch: int, kernel_type: str) -> dict:
        """
        根据架构获取最优调度参数
        """
        features = self.ARCH_FEATURES[arch]
        
        if kernel_type == "gemm":
            return {
                "block_m": 128 if arch >= 90 else 64,
                "block_n": 128 if arch >= 90 else 64,
                "block_k": 32 if arch >= 90 else 16,
                "use_tensor_core": features["tensor_core"],
                "use_async_copy": features.get("async_copy", False),
                "use_tma": features.get("tma", False),
            }
        
        elif kernel_type == "flash_attention":
            return {
                "block_m": 128 if arch >= 90 else 64,
                "block_n": 64,
                "use_tensor_core": features["tensor_core"],
            }
```

### AMD GPU

```python
# tilelang/target/rocm.py
class ROCmTarget:
    """
    ROCm/HIP 目标平台
    """
    
    ARCH_FEATURES = {
        # CDNA (MI200, MI300)
        "gfx90a": {
            "matrix_core": True,
            "matrix_core_type": "mfma",
            "matrix_core_shape": (16, 16, 16),
            "async_copy": True,
            "shared_memory": 128 * 1024,
            "wavefront_size": 64,
        },
        # RDNA3 (RX 7000)
        "gfx1100": {
            "matrix_core": True,
            "matrix_core_type": "wmma",
            "matrix_core_shape": (16, 16, 16),
            "async_copy": True,
            "shared_memory": 64 * 1024,
            "wavefront_size": 32,
        },
    }
```

### Apple Silicon

```python
# tilelang/target/metal.py
class MetalTarget:
    """
    Apple Metal 目标平台
    """
    
    DEVICE_FEATURES = {
        "M1": {
            "simd_width": 32,
            "threadgroup_memory": 32 * 1024,
            "max_threads_per_group": 1024,
        },
        "M2": {
            "simd_width": 32,
            "threadgroup_memory": 64 * 1024,
            "max_threads_per_group": 1024,
        },
        "M3": {
            "simd_width": 32,
            "threadgroup_memory": 128 * 1024,
            "max_threads_per_group": 1024,
        },
    }
```

## 下一步

继续学习 [IR 设计](02-ir-design.md)，了解 TileLang 的中间表示设计。
