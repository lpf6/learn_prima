# 第五阶段：CPU SIMD 优化

## 概述

SIMD（单指令多数据）是 CPU 并行计算的重要技术。本阶段介绍 x86 和 ARM 架构的 SIMD 指令集及其在 Prima.cpp 中的应用。

## 能力目标

完成本阶段后，你将能够：

### 能做什么

| 能力 | 具体表现 | 相关代码 |
|------|----------|----------|
| **阅读 SIMD 代码** | 理解 AVX/NEON 内联汇编 | `sgemm.cpp`, `ggml-aarch64.c` |
| **理解条件编译** | 知道如何处理多平台兼容 | 各平台代码 |
| **修改 SIMD 优化** | 调整现有 SIMD 实现 | 性能优化 |
| **检测 CPU 特性** | 运行时检测 SIMD 支持 | 初始化代码 |

### 还不能做什么

- 从零编写高性能 SIMD 库
- 设计复杂的 SIMD 算法
- 处理极端边缘情况

### 实际工作示例

学完本阶段后，你可以：

1. **理解 SIMD 条件编译**
```cpp
// 你能理解这种多平台代码结构
#if defined(__AVX512F__)
    __m512 a = _mm512_loadu_ps(ptr);
    // AVX-512 实现
#elif defined(__AVX2__)
    __m256 a = _mm256_loadu_ps(ptr);
    // AVX2 实现
#elif defined(__ARM_NEON)
    float32x4_t a = vld1q_f32(ptr);
    // NEON 实现
#else
    // 标量实现
#endif
```

2. **添加新的 SIMD 优化**
```cpp
// 你能为新操作添加 SIMD 实现
void my_operation_simd(const float* a, const float* b, float* c, int n) {
#if defined(__AVX__)
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
#endif
}
```

3. **检测 CPU 特性**
```cpp
// 你能添加运行时检测
bool has_avx2() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1 << 5)) != 0;
}
```

4. **理解量化 SIMD 实现**
```cpp
// 你能理解这段 NEON 量化代码
void dequantize_q4_0_neon(const block_q4_0 * x, float * y, int n) {
    float32x4_t scale = vdupq_n_f32(d);
    uint8x8_t qs = vld1_u8(block->qs + j / 2);
    // 解包和转换...
}
```

## 章节目录

1. [x86 SIMD 指令集](01-x86-simd.md)
2. [ARM NEON 指令集](02-arm-neon.md)

## 预计学习时间

2 周

## 开始学习

请从 [01-x86-simd.md](01-x86-simd.md) 开始。
