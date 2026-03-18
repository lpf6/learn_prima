# 5.2 ARM NEON 指令集

## 概述

NEON 是 ARM 架构的 SIMD 指令集，广泛用于移动设备、嵌入式系统和 Apple Silicon。本节介绍 NEON 指令集及其在 Prima.cpp 中的应用。

## NEON 寄存器

```
Q0-Q15: 128-bit 寄存器
D0-D31: 64-bit 寄存器（Q 的低/高半部分）
```

## 数据类型

```cpp
// 128-bit 向量类型
float32x4_t   // 4 个 float32
float64x2_t   // 2 个 float64
int32x4_t     // 4 个 int32
int16x8_t     // 8 个 int16
int8x16_t     // 16 个 int8
uint8x16_t    // 16 个 uint8

// 64-bit 向量类型
float32x2_t   // 2 个 float32
int32x2_t     // 2 个 int32
int16x4_t     // 4 个 int16
int8x8_t      // 8 个 int8
```

## 基本操作

### 加载和存储

```cpp
#include <arm_neon.h>

// 加载
float32x4_t a = vld1q_f32(ptr);      // 加载 4 个 float32
int32x4_t b = vld1q_s32(ptr);        // 加载 4 个 int32
int8x16_t c = vld1q_s8(ptr);         // 加载 16 个 int8

// 存储
vst1q_f32(dst, a);                   // 存储 4 个 float32
vst1q_s32(dst, b);                   // 存储 4 个 int32
vst1q_s8(dst, c);                    // 存储 16 个 int8
```

### 算术运算

```cpp
float32x4_t a = vld1q_f32(ptr_a);
float32x4_t b = vld1q_f32(ptr_b);

// 加法
float32x4_t c = vaddq_f32(a, b);

// 减法
float32x4_t d = vsubq_f32(a, b);

// 乘法
float32x4_t e = vmulq_f32(a, b);

// 乘加: a * b + c
float32x4_t f = vmlaq_f32(c, a, b);

// 除法
float32x4_t g = vdivq_f32(a, b);  // ARMv8+
```

### 水平操作

```cpp
// 水平加法（相邻元素相加）
float32x4_t a = {1, 2, 3, 4};
float32x2_t b = vpadd_f32(vget_low_f32(a), vget_high_f32(a));
// b = {3, 7}  (1+2, 3+4)

// 水平求和
float sum = vaddvq_f32(a);  // ARMv8+
// sum = 10  (1+2+3+4)

// 最大值
float max = vmaxvq_f32(a);  // ARMv8+

// 最小值
float min = vminvq_f32(a);  // ARMv8+
```

## 高级操作

### 类型转换

```cpp
// int32 -> float32
int32x4_t a = vld1q_s32(ptr);
float32x4_t b = vcvtq_f32_s32(a);

// float32 -> int32
int32x4_t c = vcvtq_s32_f32(b);

// FP16 <-> FP32 (ARMv8.2+)
float16x4_t fp16 = vcvt_f16_f32(fp32);
float32x4_t fp32 = vcvt_f32_f16(fp16);
```

### 重排操作

```cpp
// 提取低/高 64-bit
float32x2_t low = vget_low_f32(a);
float32x2_t high = vget_high_f32(a);

// 组合
float32x4_t combined = vcombine_f32(low, high);

// 重排
float32x4_t reversed = vrev64q_f32(a);  // 反转每 64-bit 内的元素
```

### 点积

```cpp
// ARMv8.2+ 点积指令
float32x4_t dot_product(float32x4_t a, float32x4_t b, float32x4_t c) {
    return vmlaq_f32(c, a, b);  // a * b + c
}

// 手动实现点积
float dot_product_manual(const float* a, const float* b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    for (int i = 0; i < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, va, vb);
    }

    return vaddvq_f32(sum);
}
```

## Prima.cpp 中的应用

### 条件编译

```cpp
// 文件：ggml/src/llamafile/sgemm.cpp

#if defined(__ARM_NEON)
    // NEON 实现
    float32x4_t sum = vdupq_n_f32(0.0f);
    // ...
#endif
```

### FP16 处理

```cpp
// 文件：ggml/src/ggml-aarch64.c

// ARM64 FP16 加载
static inline float32x4_t load_fp16_to_fp32(const ggml_fp16_t * ptr) {
    float16x4_t fp16 = vld1_f16((const float16_t *)ptr);
    return vcvt_f32_f16(fp16);
}

// FP32 存储
static inline void store_fp32_to_fp16(ggml_fp16_t * ptr, float32x4_t fp32) {
    float16x4_t fp16 = vcvt_f16_f32(fp32);
    vst1_f16((float16_t *)ptr, fp16);
}
```

### 量化操作

```cpp
// Q4_0 反量化
void dequantize_q4_0_neon(const block_q4_0 * x, float * y, int n) {
    for (int i = 0; i < n; i += QK4_0) {
        const block_q4_0 * block = &x[i / QK4_0];

        float d = GGML_FP16_TO_FP32(block->d);
        float32x4_t scale = vdupq_n_f32(d);

        for (int j = 0; j < QK4_0; j += 8) {
            uint8x8_t qs = vld1_u8(block->qs + j / 2);

            uint8x8_t lo = vand_u8(qs, vdup_n_u8(0x0F));
            uint8x8_t hi = vshr_n_u8(qs, 4);

            int8x8_t lo_s = vreinterpret_s8_u8(vsub_u8(lo, vdup_n_u8(8)));
            int8x8_t hi_s = vreinterpret_s8_u8(vsub_u8(hi, vdup_n_u8(8)));

            float32x4_t lo_f = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(lo_s)))), scale);
            float32x4_t hi_f = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(hi_s)))), scale);

            vst1q_f32(y + i + j, lo_f);
            vst1q_f32(y + i + j + 4, hi_f);
        }
    }
}
```

### 矩阵乘法

```cpp
// 文件：ggml/src/llamafile/sgemm.cpp

// NEON SGEMM 内核
void sgemm_kernel_neon(
    const float* A, const float* B, float* C,
    int M, int N, int K) {

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; j += 4) {
            float32x4_t sum = vdupq_n_f32(0.0f);

            for (int k = 0; k < K; ++k) {
                float32x4_t b = vld1q_f32(B + k * N + j);
                float a = A[i * K + k];
                sum = vmlaq_n_f32(sum, b, a);
            }

            vst1q_f32(C + i * N + j, sum);
        }
    }
}
```

## SVE (Scalable Vector Extension)

### 概述

SVE 是 ARM 的新一代 SIMD 指令集，向量长度可变（128-bit 到 2048-bit）。

```cpp
#include <arm_sve.h>

// 可变长度向量
svbool_t pg = svptrue_b32();           // 谓词
svfloat32_t a = svld1_f32(pg, ptr);    // 加载
svfloat32_t b = svadd_f32_z(pg, a, c); // 加法
svst1_f32(pg, dst, b);                  // 存储

int vl = svcntw();  // 每向量多少个 float32
```

## 性能对比

```
向量点积 (n=1024) 性能对比：

标量:     800 ns
NEON:     200 ns  (4x)
```

## 检测 CPU 支持

### 编译时检测

```cpp
#if defined(__ARM_NEON)
    // NEON 代码
#endif

#if defined(__ARM_FEATURE_FMA)
    // FMA 可用
#endif

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    // FP16 向量运算可用
#endif
```

### 运行时检测

```cpp
#include <sys/auxv.h>
#include <asm/hwcap.h>

bool has_neon() {
    return (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
}

bool has_fp16() {
    return (getauxval(AT_HWCAP) & HWCAP_FPHP) != 0;
}
```

## Apple Silicon 特性

### M 系列芯片

```
M1/M2/M3:
- NEON 支持
- FP16 加速
- AMX (矩阵协处理器)
- 高内存带宽
```

### 优化建议

```cpp
// Apple Silicon 优化
#if defined(__APPLE__) && defined(__aarch64__)
    // 使用 FP16 加速
    float16x8_t fp16_data = vcvtq_f16_f32(fp32_data);
#endif
```

## 练习

1. 编写使用 NEON 的向量加法和点积函数
2. 阅读 `ggml/src/llamafile/sgemm.cpp`，理解 NEON SGEMM 的实现
3. 阅读 `ggml/src/ggml-aarch64.c`，理解 ARM64 专用优化

## 下一步

完成本阶段后，请继续学习 [第六阶段：LLM 架构](../06-llm-architecture/README.md)。
