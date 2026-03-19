# TileLang 代码生成

## 目录

1. [CUDA 代码生成](#cuda-代码生成)
2. [ROCm/HIP 代码生成](#rocmhip-代码生成)
3. [Metal 代码生成](#metal-代码生成)
4. [Ascend 代码生成](#ascend-代码生成)
5. [CPU SIMD 代码生成](#cpu-simd-代码生成)

---

## CUDA 代码生成

### 代码生成流程

```
┌─────────────────────────────────────────────────────────────┐
│                    CUDA 代码生成流程                          │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              TensorIR                                 │    │
│  │  - 循环嵌套                                          │    │
│  │  - 内存访问                                          │    │
│  │  - 并行原语                                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Lowering (降级)                          │    │
│  │  - 循环展开                                          │    │
│  │  - 向量化                                            │    │
│  │  - 张量化                                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              CUDA C++ 代码生成                        │    │
│  │  - 内核签名                                          │    │
│  │  - 启动配置                                          │    │
│  │  - 内核体                                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              编译                                     │    │
│  │  - NVRTC (运行时编译)                                │    │
│  │  - nvcc (离线编译)                                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### CUDA Codegen 实现

```python
# tilelang/codegen/cuda.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CUDAKernelConfig:
    """
    CUDA 内核配置
    
    属性:
    - grid: Grid 维度
    - block: Block 维度
    - shared_mem: 共享内存大小
    - stream: CUDA 流
    """
    grid: tuple
    block: tuple
    shared_mem: int = 0
    stream: int = 0

class CUDACodegen:
    """
    CUDA 代码生成器
    """
    
    def __init__(self, arch: int = 80):
        self.arch = arch
        self.indent = 0
        self.code_lines = []
    
    def generate(self, ir_module: tvm.IRModule) -> str:
        """
        生成 CUDA 代码
        
        步骤:
        1. 生成头文件
        2. 生成内核函数
        3. 生成启动代码
        """
        # 1. 生成头文件
        self.emit_header()
        
        # 2. 生成内核函数
        for func in ir_module.functions:
            self.generate_kernel(func)
        
        return "\n".join(self.code_lines)
    
    def emit_header(self):
        """生成头文件"""
        self.emit("#include <cuda_fp16.h>")
        self.emit("#include <cuda_bf16.h>")
        self.emit("#include <mma.h>")
        self.emit("")
        self.emit("using namespace nvcuda::wmma;")
        self.emit("")
    
    def generate_kernel(self, func: tir.PrimFunc):
        """
        生成 CUDA 内核
        
        结构:
        __global__ void kernel_name(params...) {
            // 计算线程索引
            // 加载数据
            // 计算
            // 存储结果
        }
        """
        # 生成函数签名
        signature = self.generate_signature(func)
        self.emit(f"__global__ void {signature} {{")
        self.indent += 1
        
        # 生成线程索引计算
        self.generate_thread_indices(func)
        
        # 生成边界检查
        self.generate_boundary_check(func)
        
        # 生成函数体
        self.generate_body(func.body)
        
        self.indent -= 1
        self.emit("}")
        self.emit("")
    
    def generate_signature(self, func: tir.PrimFunc) -> str:
        """
        生成函数签名
        
        示例:
        kernel_name(
            half* __restrict__ A,
            half* __restrict__ B,
            half* __restrict__ C,
            int M, int N, int K
        )
        """
        params = []
        
        for param in func.params:
            if isinstance(param, tir.Buffer):
                dtype = self.dtype_to_cuda(param.dtype)
                params.append(f"{dtype}* __restrict__ {param.name}")
            else:
                params.append(f"int {param.name}")
        
        return f"{func.name}({', '.join(params)})"
    
    def generate_thread_indices(self, func: tir.PrimFunc):
        """
        生成线程索引计算
        
        示例:
        int blockIdx_x = blockIdx.x;
        int blockIdx_y = blockIdx.y;
        int threadIdx_x = threadIdx.x;
        int threadIdx_y = threadIdx.y;
        """
        self.emit("// 计算线程索引")
        self.emit("const int blockIdx_x = blockIdx.x;")
        self.emit("const int blockIdx_y = blockIdx.y;")
        self.emit("const int threadIdx_x = threadIdx.x;")
        self.emit("const int threadIdx_y = threadIdx.y;")
        self.emit("")
    
    def generate_body(self, stmt: tir.Stmt):
        """
        生成函数体
        
        处理不同类型的语句:
        - For: 循环
        - If: 条件
        - BufferStore: 存储
        - Evaluate: 求值
        """
        if isinstance(stmt, tir.For):
            self.generate_for(stmt)
        elif isinstance(stmt, tir.If):
            self.generate_if(stmt)
        elif isinstance(stmt, tir.BufferStore):
            self.generate_buffer_store(stmt)
        elif isinstance(stmt, tir.Evaluate):
            self.generate_evaluate(stmt)
        elif isinstance(stmt, tir.SeqStmt):
            for s in stmt.seq:
                self.generate_body(s)
    
    def generate_for(self, loop: tir.For):
        """
        生成循环
        
        根据循环类型生成不同的代码:
        - serial: 普通 for 循环
        - parallel: 映射到 blockIdx 或 threadIdx
        - vectorized: 向量化操作
        """
        if loop.kind == "parallel":
            # 并行循环：映射到线程索引
            self.emit(f"// Parallel loop: {loop.loop_var.name}")
            # 线程索引已经在前面计算
        elif loop.kind == "vectorized":
            # 向量化循环：生成向量操作
            self.generate_vectorized_for(loop)
        else:
            # 串行循环：生成 for 循环
            self.emit(f"for (int {loop.loop_var.name} = {loop.min_val}; "
                     f"{loop.loop_var.name} < {loop.max_val}; "
                     f"{loop.loop_var.name} += {loop.step}) {{")
            self.indent += 1
            self.generate_body(loop.body)
            self.indent -= 1
            self.emit("}")
    
    def generate_tensor_core_intrinsics(self, op: tir.Call) -> str:
        """
        生成 Tensor Core 内联函数
        
        根据架构生成不同的指令:
        - Volta/Turing: wmma
        - Ampere: mma.sync
        - Hopper: wgmma.mma_async
        """
        if self.arch >= 90:  # Hopper
            return self.generate_wgmma(op)
        elif self.arch >= 80:  # Ampere
            return self.generate_mma(op)
        else:  # Volta/Turing
            return self.generate_wmma(op)
    
    def generate_wmma(self, op: tir.Call) -> str:
        """
        生成 WMMA 指令 (Volta/Turing)
        
        示例:
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, A + offset, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag);
        wmma::store_matrix_sync(C + offset, c_frag, 16, wmma::mem_row_major);
        """
        code = []
        
        # 声明 fragment
        code.append(f"wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;")
        code.append(f"wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;")
        code.append(f"wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;")
        
        # 加载数据
        code.append(f"wmma::load_matrix_sync(a_frag, A + offset_a, 16);")
        code.append(f"wmma::load_matrix_sync(b_frag, B + offset_b, 16);")
        
        # 计算
        code.append(f"wmma::mma_sync(c_frag, a_frag, b_frag);")
        
        # 存储结果
        code.append(f"wmma::store_matrix_sync(C + offset_c, c_frag, 16, wmma::mem_row_major);")
        
        return "\n".join(code)
    
    def generate_mma(self, op: tir.Call) -> str:
        """
        生成 MMA 指令 (Ampere)
        
        示例:
        asm volatile(
            "mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
        """
        code = []
        
        code.append('asm volatile(')
        code.append('    "mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "')
        code.append('    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"')
        code.append('    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])')
        code.append('    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),')
        code.append('      "r"(b[0]), "r"(b[1]),')
        code.append('      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])')
        code.append(');')
        
        return "\n".join(code)
    
    def generate_wgmma(self, op: tir.Call) -> str:
        """
        生成 WGMMA 指令 (Hopper)
        
        示例:
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
            "{%0, ..., %63}, %4, %5, {%64, ..., %127};"
            : ...
            : "l"(desc_a), "l"(desc_b), ...
        );
        """
        code = []
        
        code.append('asm volatile(')
        code.append('    "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "')
        code.append('    "{%0, ..., %63}, %4, %5, {%64, ..., %127};"')
        code.append('    : ...')
        code.append('    : "l"(desc_a), "l"(desc_b), ...')
        code.append(');')
        
        return "\n".join(code)
    
    def emit(self, line: str):
        """输出代码行"""
        if line:
            self.code_lines.append("  " * self.indent + line)
        else:
            self.code_lines.append("")
```

---

## ROCm/HIP 代码生成

```python
# tilelang/codegen/rocm.py
class ROCmCodegen:
    """
    ROCm/HIP 代码生成器
    用于 AMD GPU
    """
    
    def __init__(self, arch: str = "gfx90a"):
        self.arch = arch
        self.indent = 0
        self.code_lines = []
    
    def generate(self, ir_module: tvm.IRModule) -> str:
        """
        生成 ROCm/HIP 代码
        """
        # 生成头文件
        self.emit("#include <hip/hip_runtime.h>")
        self.emit("#include <hip/hip_fp16.h>")
        self.emit("")
        
        # 生成内核函数
        for func in ir_module.functions:
            self.generate_kernel(func)
        
        return "\n".join(self.code_lines)
    
    def generate_mfma_intrinsics(self, op: tir.Call) -> str:
        """
        生成 MFMA (Matrix Fused Multiply-Add) 指令
        
        AMD CDNA 架构的矩阵加速指令
        """
        code = []
        
        # MFMA 指令格式
        # __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, cbsz, abid, blgp)
        
        code.append('// MFMA 16x16x16')
        code.append('__builtin_amdgcn_mfma_f32_16x16x16f16(')
        code.append('    a, b, c,')
        code.append('    0,  // cbsz')
        code.append('    0,  // abid')
        code.append('    0   // blgp')
        code.append(');')
        
        return "\n".join(code)
    
    def generate_wavefront_operations(self, op: tir.Call) -> str:
        """
        生成 Wavefront 操作
        
        AMD GPU 的 Wavefront 大小为 64
        """
        code = []
        
        # Wavefront 归约
        code.append('// Wavefront 归约')
        code.append('__builtin_amdgcn_ds_bpermute(')
        code.append('    index,')
        code.append('    value')
        code.append(');')
        
        return "\n".join(code)
```

---

## Metal 代码生成

```python
# tilelang/codegen/metal.py
class MetalCodegen:
    """
    Metal 代码生成器
    用于 Apple Silicon
    """
    
    def __init__(self, device: str = "M1"):
        self.device = device
        self.indent = 0
        self.code_lines = []
    
    def generate(self, ir_module: tvm.IRModule) -> str:
        """
        生成 Metal Shading Language 代码
        """
        # 生成头文件
        self.emit("#include <metal_stdlib>")
        self.emit("#include <simd/simd.h>")
        self.emit("using namespace metal;")
        self.emit("")
        
        # 生成内核函数
        for func in ir_module.functions:
            self.generate_kernel(func)
        
        return "\n".join(self.code_lines)
    
    def generate_kernel(self, func: tir.PrimFunc):
        """
        生成 Metal 内核
        
        结构:
        kernel void kernel_name(
            device float* A [[buffer(0)]],
            device float* B [[buffer(1)]],
            uint3 blockIdx [[threadgroup_position_in_grid]],
            uint3 threadIdx [[thread_position_in_threadgroup]]
        ) {
            // 内核体
        }
        """
        # 生成函数签名
        params = []
        buffer_idx = 0
        
        for param in func.params:
            if isinstance(param, tir.Buffer):
                dtype = self.dtype_to_metal(param.dtype)
                params.append(f"device {dtype}* {param.name} [[buffer({buffer_idx})]]")
                buffer_idx += 1
        
        # 添加线程索引参数
        params.append("uint3 blockIdx [[threadgroup_position_in_grid]]")
        params.append("uint3 threadIdx [[thread_position_in_threadgroup]]")
        
        self.emit(f"kernel void {func.name}({', '.join(params)}) {{")
        self.indent += 1
        
        # 生成函数体
        self.generate_body(func.body)
        
        self.indent -= 1
        self.emit("}")
        self.emit("")
    
    def generate_simd_operations(self, op: tir.Call) -> str:
        """
        生成 SIMD 操作
        
        Apple Silicon 的 SIMD 组操作
        """
        code = []
        
        # SIMD 组归约
        code.append('// SIMD 组归约')
        code.append('float result = simd_sum(value);')
        
        # SIMD 组矩阵乘法 (Apple M 系列芯片)
        code.append('// SIMD 矩阵乘法')
        code.append('// 使用 simdgroup_matrix')
        
        return "\n".join(code)
```

---

## Ascend 代码生成

```python
# tilelang/codegen/ascend.py
class AscendCodegen:
    """
    Ascend 代码生成器
    用于华为昇腾 NPU
    """
    
    def __init__(self, arch: str = "ascend910"):
        self.arch = arch
        self.indent = 0
        self.code_lines = []
    
    def generate(self, ir_module: tvm.IRModule) -> str:
        """
        生成 Ascend C 代码
        """
        # 生成头文件
        self.emit('#include "kernel_operator.h"')
        self.emit("")
        
        # 生成内核函数
        for func in ir_module.functions:
            self.generate_kernel(func)
        
        return "\n".join(self.code_lines)
    
    def generate_kernel(self, func: tir.PrimFunc):
        """
        生成 Ascend C 内核
        
        结构:
        class KernelName {
        public:
            __aicore__ void Process() {
                // 内核体
            }
        };
        
        extern "C" __global__ __aicore__ void kernel_name(GM_ADDR A, GM_ADDR B, GM_ADDR C) {
            KernelName op;
            op.Process();
        }
        """
        # 生成类定义
        self.emit(f"class {func.name.capitalize()} {{")
        self.emit("public:")
        self.indent += 1
        
        self.emit("__aicore__ void Process() {")
        self.indent += 1
        
        # 生成函数体
        self.generate_body(func.body)
        
        self.indent -= 1
        self.emit("}")
        
        self.indent -= 1
        self.emit("};")
        self.emit("")
        
        # 生成入口函数
        params = []
        for param in func.params:
            if isinstance(param, tir.Buffer):
                params.append(f"GM_ADDR {param.name}")
        
        self.emit(f'extern "C" __global__ __aicore__ void {func.name}({", ".join(params)}) {{')
        self.emit(f"    {func.name.capitalize()} op;")
        self.emit("    op.Process();")
        self.emit("}")
        self.emit("")
    
    def generate_cube_operations(self, op: tir.Call) -> str:
        """
        生成 Cube 操作
        
        华为昇腾的矩阵加速单元
        """
        code = []
        
        # Cube 矩阵乘法
        code.append('// Cube 矩阵乘法')
        code.append('AscendC::MatMul(')
        code.append('    aTensor,')
        code.append('    bTensor,')
        code.append('    cTensor')
        code.append(');')
        
        return "\n".join(code)
```

---

## CPU SIMD 代码生成

```python
# tilelang/codegen/llvm.py
class LLVMCodegen:
    """
    LLVM 代码生成器
    用于 CPU SIMD 优化
    """
    
    def __init__(self, target: str = "x86_64"):
        self.target = target
        self.indent = 0
        self.code_lines = []
    
    def generate(self, ir_module: tvm.IRModule) -> str:
        """
        生成 CPU 代码
        """
        # 生成头文件
        self.emit("#include <immintrin.h>  // AVX/AVX2/AVX-512")
        self.emit("#include <arm_neon.h>   // ARM NEON")
        self.emit("")
        
        # 生成函数
        for func in ir_module.functions:
            self.generate_function(func)
        
        return "\n".join(self.code_lines)
    
    def generate_simd_intrinsics(self, op: tir.Call, 
                                  target: str) -> str:
        """
        生成 SIMD 内联函数
        
        根据目标平台生成不同的 SIMD 指令
        """
        if target == "x86_64":
            return self.generate_x86_simd(op)
        elif target == "aarch64":
            return self.generate_arm_simd(op)
        else:
            return self.generate_scalar(op)
    
    def generate_x86_simd(self, op: tir.Call) -> str:
        """
        生成 x86 SIMD 指令
        
        支持: SSE, AVX, AVX2, AVX-512
        """
        code = []
        
        # 检测支持的指令集
        features = self.detect_cpu_features()
        
        if "avx512f" in features:
            # AVX-512
            code.append('// AVX-512')
            code.append('__m512 a_vec = _mm512_load_ps(a);')
            code.append('__m512 b_vec = _mm512_load_ps(b);')
            code.append('__m512 c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);')
            code.append('_mm512_store_ps(c, c_vec);')
        
        elif "avx2" in features:
            # AVX2
            code.append('// AVX2')
            code.append('__m256 a_vec = _mm256_load_ps(a);')
            code.append('__m256 b_vec = _mm256_load_ps(b);')
            code.append('__m256 c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);')
            code.append('_mm256_store_ps(c, c_vec);')
        
        elif "avx" in features:
            # AVX
            code.append('// AVX')
            code.append('__m256 a_vec = _mm256_load_ps(a);')
            code.append('__m256 b_vec = _mm256_load_ps(b);')
            code.append('__m256 c_vec = _mm256_add_ps(_mm256_mul_ps(a_vec, b_vec), c_vec);')
            code.append('_mm256_store_ps(c, c_vec);')
        
        else:
            # SSE
            code.append('// SSE')
            code.append('__m128 a_vec = _mm_load_ps(a);')
            code.append('__m128 b_vec = _mm_load_ps(b);')
            code.append('__m128 c_vec = _mm_add_ps(_mm_mul_ps(a_vec, b_vec), c_vec);')
            code.append('_mm_store_ps(c, c_vec);')
        
        return "\n".join(code)
    
    def generate_arm_simd(self, op: tir.Call) -> str:
        """
        生成 ARM SIMD 指令
        
        支持: NEON, SVE
        """
        code = []
        
        # NEON
        code.append('// ARM NEON')
        code.append('float32x4_t a_vec = vld1q_f32(a);')
        code.append('float32x4_t b_vec = vld1q_f32(b);')
        code.append('float32x4_t c_vec = vmlaq_f32(c_vec, a_vec, b_vec);')
        code.append('vst1q_f32(c, c_vec);')
        
        return "\n".join(code)
```

## 下一步

继续学习 [运行时系统](05-runtime-system.md)，了解 TileLang 的运行时支持。
