# C++ 编程基础

## 概述

Prima.cpp 主要使用 C++ 编写，理解 C++ 编程对于阅读和修改代码至关重要。本章介绍项目中常用的 C++ 特性。

## 1. 模板编程

模板是 C++ 泛型编程的基础，在 CUDA 内核中广泛使用。

### 1.1 函数模板

```cpp
// 示例：通用向量加法
template<typename T>
__global__ void vector_add(const T *a, const T *b, T *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 使用
vector_add<float><<<blocks, threads>>>(d_a, d_b, d_c, n);
vector_add<double><<<blocks, threads>>>(d_a, d_b, d_c, n);
```

### 1.2 类模板

```cpp
// 示例：通用缓冲区
template<typename T, int SIZE>
class Buffer {
private:
    T data[SIZE];
    int count;
    
public:
    __device__ void push(T value) {
        if (count < SIZE) {
            data[count++] = value;
        }
    }
    
    __device__ T pop() {
        if (count > 0) {
            return data[--count];
        }
        return T();
    }
};

// 使用
Buffer<float, 256> float_buffer;
Buffer<int, 128> int_buffer;
```

### 1.3 模板特化

```cpp
// 通用版本
template<typename T>
__device__ float compute_score(T value) {
    return static_cast<float>(value);
}

// 特化版本（针对 int 优化）
template<>
__device__ float compute_score<int>(int value) {
    // 使用更快的整数运算
    return value * 1.0f;
}

// 部分特化
template<typename T, int N>
class Matrix {
    // 通用矩阵实现
};

template<typename T>
class Matrix<T, 4> {
    // 4x4 矩阵的特殊优化实现
};
```

### 1.4 SFINAE（替换失败不是错误）

```cpp
// 示例：根据类型特征选择实现
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
safe_divide(T a, T b) {
    return b != 0 ? a / b : 0;
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
safe_divide(T a, T b) {
    return b != 0 ? a / b : 0;
}
```

## 2. 内存管理

### 2.1 RAII（资源获取即初始化）

```cpp
// 示例：自动管理 CUDA 内存
class CUDABuffer {
private:
    void* device_ptr;
    size_t size;
    
public:
    CUDABuffer(size_t size) : size(size) {
        cudaMalloc(&device_ptr, size);
    }
    
    ~CUDABuffer() {
        cudaFree(device_ptr);
    }
    
    // 禁止拷贝
    CUDABuffer(const CUDABuffer&) = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;
    
    // 允许移动
    CUDABuffer(CUDABuffer&& other) noexcept 
        : device_ptr(other.device_ptr), size(other.size) {
        other.device_ptr = nullptr;
        other.size = 0;
    }
    
    void* get() const { return device_ptr; }
};

// 使用：无需手动释放
void kernel_wrapper() {
    CUDABuffer buffer(1024 * sizeof(float));
    // 使用 buffer.get()
    // 离开作用域时自动释放
}
```

### 2.2 智能指针

```cpp
#include <memory>

// unique_ptr：独占所有权
std::unique_ptr<float[]> data(new float[1024]);
data[0] = 1.0f;
// 自动释放

// shared_ptr：共享所有权
std::shared_ptr<Context> ctx = std::make_shared<Context>();
auto ctx2 = ctx;  // 引用计数 +1
// 最后一个 shared_ptr 销毁时释放

// weak_ptr：弱引用（不增加引用计数）
std::weak_ptr<Context> weak_ctx = ctx;
if (auto locked_ctx = weak_ctx.lock()) {
    // 使用 locked_ctx
}
```

### 2.3 自定义删除器

```cpp
// 示例：自动释放 CUDA 内存的 unique_ptr
auto cuda_deleter = [](float* ptr) {
    cudaFree(ptr);
};

std::unique_ptr<float[], decltype(cuda_deleter)> d_data(
    nullptr, cuda_deleter
);

cudaMalloc(&d_data.get(), size);
// 自动调用 cudaFree
```

## 3. 移动语义

### 3.1 右值引用

```cpp
// 左值：有名字的变量
int a = 5;
int& lref = a;

// 右值：临时值
int&& rref = 5;

// 移动构造函数
class LargeObject {
    int* data;
    size_t size;
    
public:
    // 拷贝构造函数（深拷贝）
    LargeObject(const LargeObject& other) 
        : size(other.size) {
        data = new int[size];
        std::copy(other.data, other.data + size, data);
    }
    
    // 移动构造函数（窃取资源）
    LargeObject(LargeObject&& other) noexcept
        : data(other.data), size(other.size) {
        other.data = nullptr;  // 置空，防止双重释放
        other.size = 0;
    }
    
    ~LargeObject() {
        delete[] data;
    }
};
```

### 3.2 std::move

```cpp
// 强制转换为右值，触发移动
std::vector<float> v1 = {1, 2, 3, 4, 5};
std::vector<float> v2 = std::move(v1);
// v1 现在为空，v2 拥有数据
```

## 4. Lambda 表达式

```cpp
// 基本语法
auto add = [](int a, int b) {
    return a + b;
};

// 捕获列表
int factor = 2;
auto scale = [factor](int x) {
    return x * factor;
};

// 引用捕获
std::vector<int> results;
auto collect = [&results](int x) {
    results.push_back(x);
};

// 在 CUDA 中的使用
auto kernel_launch = []() {
    some_kernel<<<blocks, threads>>>(args);
};
kernel_launch();
```

## 5. 类型推导

### 5.1 auto

```cpp
// 自动类型推导
auto x = 5;           // int
auto y = 3.14f;       // float
auto ptr = &x;        // int*

// 在循环中使用
std::vector<float> vec;
for (auto& val : vec) {
    val *= 2.0f;
}

// 在 CUDA 中
auto kernel = []<<<blocks, threads>>>() {
    // ...
};
```

### 5.2 decltype

```cpp
int x = 5;
decltype(x) y = 10;  // y 的类型是 int

// 在模板中很有用
template<typename T, typename U>
auto multiply(T a, U b) -> decltype(a * b) {
    return a * b;
}
```

## 6. 常量表达式

### 6.1 constexpr

```cpp
// 编译期计算
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

constexpr int fact5 = factorial(5);  // 120，编译期计算

// 在 CUDA 中
constexpr int BLOCK_SIZE = 256;
constexpr float PI = 3.141592653589793f;
```

### 6.2 consteval 和 constinit (C++20)

```cpp
// consteval：必须在编译期求值
consteval int compile_time_square(int x) {
    return x * x;
}

// constinit：必须在初始化时求值
constinit int global_counter = 0;
```

## 7. 概念和约束 (C++20)

```cpp
#include <concepts>

// 定义概念
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

// 使用概念约束模板
template<Numeric T>
T add(T a, T b) {
    return a + b;
}

// 等价于
template<typename T>
requires std::is_arithmetic_v<T>
T add(T a, T b) {
    return a + b;
}
```

## 8. 变参模板

```cpp
// 示例：通用日志函数
template<typename... Args>
void log(const char* format, Args... args) {
    printf("[LOG] ");
    printf(format, args...);
    printf("\n");
}

// 使用
log("Value: %d", 42);
log("Float: %f, String: %s", 3.14f, "hello");
```

## 9. 项目中的实际应用

### 9.1 通用量化类型

```cpp
// 定义量化类型特征
template<typename T>
struct quant_traits;

// Q4_0 特化
template<>
struct quant_traits<block_q4_0> {
    static constexpr int QK = QK4_0;
    static constexpr int BLCK = 2;
    static constexpr size_t type_size() {
        return sizeof(block_q4_0);
    }
};

// 通用反量化函数
template<typename T>
__device__ float dequantize(const T& block, int i) {
    return quant_traits<T>::dequantize(block, i);
}
```

### 9.2 类型安全的枚举

```cpp
enum class GGMLType : int32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // ...
};

// 类型安全，不能隐式转换为 int
GGMLType type = GGMLType::Q4_0;
```

## 练习

1. 实现一个模板化的 CUDA 内核，支持 float 和 double 类型
2. 使用 RAII 模式封装 CUDA 流（cudaStream_t）
3. 实现一个移动语义的矩阵类
4. 使用 Lambda 表达式重构现有的 CUDA 内核启动代码

## 参考资料

- [C++ Reference](https://en.cppreference.com/)
- [C++ Templates Guide](https://en.cppreference.com/w/cpp/language/templates)
- [Effective Modern C++](https://www.aristeia.com/Book7/effective_modern_cpp.html)
