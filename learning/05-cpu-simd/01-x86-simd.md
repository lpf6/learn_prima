# 5.1 x86 SIMD 指令集

## 概述

x86 架构提供了多种 SIMD（单指令多数据）指令集，可以大幅提升计算性能。本节介绍 SSE、AVX、AVX2 和 AVX-512 指令集及其在 Prima.cpp 中的应用。

## SIMD 指令集演进

```
SSE (1999)      128-bit    4 float / 2 double
AVX (2011)      256-bit    8 float / 4 double
AVX2 (2013)     256-bit    整数 SIMD + FMA
AVX-512 (2016)  512-bit    16 float / 8 double
```

## SSE 指令集

### 寄存器

```
XMM0 - XMM15 (128-bit)
```

### 基本操作

```cpp
#include <immintrin.h>

// 加载
__m128 a = _mm_loadu_ps(ptr);      // 加载 4 个 float（非对齐）
__m128 b = _mm_load_ps(ptr);        // 加载 4 个 float（对齐）

// 算术运算
__m128 c = _mm_add_ps(a, b);        // 加法
__m128 d = _mm_mul_ps(a, b);        // 乘法
__m128 e = _mm_sub_ps(a, b);        // 减法
__m128 f = _mm_div_ps(a, b);        // 除法

// 存储
_mm_storeu_ps(dst, c);              // 存储（非对齐）
_mm_store_ps(dst, c);               // 存储（对齐）
```

### 常用指令

```cpp
// 水平求和
float sum = _mm_cvtss_f32(_mm_add_ss(
    _mm_add_ps(_mm_movehl_ps(a, a), a),
    _mm_shuffle_ps(a, a, 1)));

// 最大值
__m128 max = _mm_max_ps(a, b);

// 最小值
__m128 min = _mm_min_ps(a, b);

// 平方根
__m128 sqrt = _mm_sqrt_ps(a);
```

## AVX 指令集

### 寄存器

```
YMM0 - YMM15 (256-bit)
低 128-bit 是 XMM
```

### 基本操作

```cpp
// 加载
__m256 a = _mm256_loadu_ps(ptr);    // 加载 8 个 float

// 算术运算
__m256 b = _mm256_add_ps(a, c);     // 加法
__m256 d = _mm256_mul_ps(a, c);     // 乘法

// 存储
_mm256_storeu_ps(dst, b);           // 存储 8 个 float
```

### FP16 转换

```cpp
// FP16 <-> FP32 转换（需要 F16C）
__m256 fp32 = _mm256_cvtph_ps(fp16);  // FP16 -> FP32
__m128i fp16 = _mm256_cvtps_ph(fp32, 0);  // FP32 -> FP16
```

## AVX2 指令集

### 整数 SIMD

```cpp
// 整数运算
__m256i a = _mm256_loadu_si256((__m256i*)ptr);
__m256i b = _mm256_add_epi32(a, c);   // 32-bit 整数加法
__m256i d = _mm256_mullo_epi32(a, c); // 32-bit 整数乘法（低位）
```

### FMA（融合乘加）

```cpp
// FMA: d = a * b + c
__m256 d = _mm256_fmadd_ps(a, b, c);

// FMS: d = a * b - c
__m256 e = _mm256_fmsub_ps(a, b, c);
```

## AVX-512 指令集

### 寄存器

```
ZMM0 - ZMM31 (512-bit)
低 256-bit 是 YMM，低 128-bit 是 XMM
```

### 基本操作

```cpp
// 加载
__m512 a = _mm512_loadu_ps(ptr);    // 加载 16 个 float

// 算术运算
__m512 b = _mm512_add_ps(a, c);
__m512 d = _mm512_mul_ps(a, c);

// 存储
_mm512_storeu_ps(dst, b);
```

### 掩码操作

```cpp
// 掩码加法
__mmask16 mask = 0xAAAA;  // 只处理偶数位置
__m512 c = _mm512_mask_add_ps(a, mask, b, a);

// 掩码加载
__m512 d = _mm512_maskz_loadu_ps(mask, ptr);
```

## Prima.cpp 中的应用

### 条件编译

```cpp
// 文件：ggml/src/ggml-aarch64.c

#if defined(__AVX512F__)
    // AVX-512 实现
    __m512 a = _mm512_loadu_ps(ptr);
    // ...
#elif defined(__AVX2__)
    // AVX2 实现
    __m256 a = _mm256_loadu_ps(ptr);
    // ...
#elif defined(__AVX__)
    // AVX 实现
    __m256 a = _mm256_loadu_ps(ptr);
    // ...
#elif defined(__SSE__)
    // SSE 实现
    __m128 a = _mm_loadu_ps(ptr);
    // ...
#else
    // 标量实现
    for (int i = 0; i < n; ++i) {
        // ...
    }
#endif
```

### FP16 加载优化

```cpp
// 文件：ggml/src/ggml-aarch64.c

#if defined(__AVX__) && defined(__F16C__)
#if defined(__AVX512F__)
// AVX-512 FP16 加载
#define GGML_F32Cx8x2_LOAD(x, y) \
    _mm512_cvtph_ps(_mm256_set_m128i( \
        _mm_loadu_si128((const __m128i *)(y)), \
        _mm_loadu_si128((const __m128i *)(x))))
#endif

// AVX FP16 加载
#define GGML_F32Cx8_LOAD(x) \
    _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#endif
```

### 向量点积

```cpp
// 使用 AVX 实现向量点积
float dot_product_avx(const float* a, const float* b, int n) {
    __m256 sum = _mm256_setzero_ps();

    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // 水平求和
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);

    return _mm_cvtss_f32(s);
}
```

### 矩阵乘法

```cpp
// 文件：ggml/src/llamafile/sgemm.cpp

// AVX-512 SGEMM 内核
void sgemm_kernel_avx512(
    const float* A, const float* B, float* C,
    int M, int N, int K) {

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m512 sum = _mm512_setzero_ps();

            for (int k = 0; k < K; k += 16) {
                __m512 a = _mm512_loadu_ps(A + i * K + k);
                __m512 b = _mm512_loadu_ps(B + k * N + j);
                sum = _mm512_fmadd_ps(a, b, sum);
            }

            C[i * N + j] = _mm512_reduce_add_ps(sum);
        }
    }
}
```

## 检测 CPU 支持

### 运行时检测

```cpp
#include <cpuid.h>

bool has_avx2() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);

    // 检查 OSXSAVE
    if (!(ecx & (1 << 27))) return false;

    // 检查 AVX
    if (!(ecx & (1 << 28))) return false;

    // 检查 AVX2
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1 << 5)) != 0;
}

bool has_avx512f() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1 << 16)) != 0;
}
```

### 编译时检测

```cpp
#if defined(__AVX512F__)
    // AVX-512 代码
#elif defined(__AVX2__)
    // AVX2 代码
#elif defined(__AVX__)
    // AVX 代码
#elif defined(__SSE__)
    // SSE 代码
#endif
```

## 性能对比

```
向量点积 (n=1024) 性能对比：

标量:     1000 ns
SSE:       300 ns  (3.3x)
AVX:       180 ns  (5.5x)
AVX2:      150 ns  (6.7x)
AVX-512:   100 ns  (10x)
```

## 练习

1. 编写使用不同 SIMD 指令集的向量加法函数
2. 阅读 `ggml/src/llamafile/sgemm.cpp`，理解 SGEMM 的 SIMD 实现
3. 阅读 `ggml/src/ggml-aarch64.c`，理解条件编译的使用

## 下一步

完成本节后，请继续学习 [ARM NEON](02-arm-neon.md)。
