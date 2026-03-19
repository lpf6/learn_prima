# 形成性评估模板

本目录包含第一阶段各章节的形成性评估内容，供学习者自我检测。

## 使用指南

### 对于学习者
- 每学完一节，完成对应的自我检测
- 不要查看答案，先独立完成
- 记录自己的得分，追踪进步
- 如果得分低于 80%，建议重新学习该节

### 对于教育者
- 可以将这些题目用于在线测验
- 可以根据学员情况调整难度
- 鼓励学员贡献新的题目

---

## 1.1 线程层次结构 - 自我检测

### 理解检查 ✓

完成本节后，检查你是否能够：

- [ ] 解释 Grid、Block、Thread 三层结构的关系
- [ ] 计算一维、二维、三维线程索引
- [ ] 说明 Warp 的概念和大小
- [ ] 理解 `__shfl_xor_sync` 指令的作用
- [ ] 解释为什么 Block 大小通常是 32 的倍数

### 小测验 📝

#### 题目 1：线程索引计算

如果一个 kernel 启动配置为 `kernel<<<4, 128>>>`，那么：

1. 总共有多少个线程？
2. 第 3 个 Block 中的第 5 个线程的全局索引是多少？
3. 如何计算任意线程的全局索引？

<details>
<summary>点击查看答案解析</summary>

**答案**：
1. 总线程数 = 4 × 128 = **512 个线程**
2. 第 3 个 Block（blockIdx.x=2）的第 5 个线程（threadIdx.x=4）：
   - 全局索引 = 2 × 128 + 4 = **260**
3. 通用公式：`global_idx = blockIdx.x * blockDim.x + threadIdx.x`

**解析**：
- Grid 中有 4 个 Block
- 每个 Block 有 128 个线程
- 全局索引计算公式是 CUDA 编程的基础
</details>

---

#### 题目 2：Warp 理解

关于 Warp 的说法，以下哪些是正确的？

A. 一个 Warp 包含 64 个线程
B. 同一 Warp 中的线程同时执行相同指令
C. Warp 分歧会降低性能
D. Shuffle 指令可以在 Warp 内交换数据

<details>
<summary>点击查看答案解析</summary>

**正确答案**：B, C, D

**解析**：
- A ❌：一个 Warp 包含 **32** 个线程，不是 64 个
- B ✅：这是 SIMT（单指令多线程）执行模型的核心
- C ✅：Warp 分歧导致线程串行执行，降低性能
- D ✅：`__shfl_xor_sync` 等指令允许 Warp 内高效数据交换
</details>

---

#### 题目 3：代码理解

阅读以下代码，回答问题：

```cpp
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

1. 为什么要检查 `if (idx < n)`？
2. 如果去掉这个检查会发生什么？
3. 如果 n=1000，blockDim.x=256，需要多少个 Block？

<details>
<summary>点击查看答案解析</summary>

**答案**：

1. **边界检查**：确保不访问数组越界。GPU 线程数可能多于元素数。

2. **后果**：
   - 访问非法内存地址
   - 可能导致程序崩溃
   - 可能破坏其他数据
   - cuda-memcheck 会报告错误

3. **Block 数量计算**：
   ```
   grid_size = (n + blockDim.x - 1) / blockDim.x
             = (1000 + 256 - 1) / 256
             = 1255 / 256
             = 4.90...
             = 4（向上取整）
   ```
   实际需要 **4 个 Block**（4×256=1024 个线程，覆盖 1000 个元素）

</details>

---

### 反思问题 💭

思考以下问题，可以在学习社区分享你的答案：

1. **对比思考**：CUDA 的线程模型和你之前了解的 CPU 多线程（如 pthread、OpenMP）有什么不同？

2. **应用场景**：为什么 GPU 要使用三层线程结构，而不是扁平化的线程模型？

3. **性能思考**：如果 Warp 中的线程执行不同的分支（Warp Divergence），为什么会影响性能？你能想到避免的方法吗？

4. **知识联系**：线程索引计算公式和并行计算基础中的什么概念相关？

---

## 1.2 内存模型 - 自我检测

### 理解检查 ✓

- [ ] 列出 CUDA 的 6 种内存类型
- [ ] 说明每种内存的速度和容量特征
- [ ] 解释共享内存的作用和使用场景
- [ ] 理解内存合并访问的重要性
- [ ] 说明常量内存和纹理内存的优化原理

### 小测验 📝

#### 题目 1：内存类型匹配

将内存类型与其特征进行匹配：

| 内存类型 | 特征 |
|---------|------|
| 1. 寄存器 | A. 所有线程可见，速度慢，容量大 |
| 2. 共享内存 | B. 单个线程私有，速度最快 |
| 3. 全局内存 | C. Block 内线程共享，速度很快 |
| 4. 常量内存 | D. 只读，缓存优化 |

<details>
<summary>点击查看答案解析</summary>

**答案**：
- 1 - B：寄存器是线程私有的，速度最快
- 2 - C：共享内存是 Block 内共享的
- 3 - A：全局内存是所有线程可见的
- 4 - D：常量内存是只读的，有专门缓存

**解析**：
理解内存层次是 CUDA 优化的关键。速度：寄存器 > 共享内存 > 全局内存
</details>

---

#### 题目 2：共享内存优化

以下代码使用共享内存优化矩阵转置，请填空：

```cpp
__global__ void transpose(float *in, float *out, int width) {
    __shared__ float tile[16][16];
    
    int x = blockIdx.x * 16 + threadIdx.x;
    int y = blockIdx.y * 16 + threadIdx.y;
    
    // 1. 从全局内存加载到共享内存
    tile[___][___] = in[y * width + x];
    
    __syncthreads();
    
    // 2. 从共享内存写入（转置）
    int out_x = blockIdx.y * 16 + threadIdx.y;
    int out_y = blockIdx.x * 16 + threadIdx.x;
    out[out_y * width + out_x] = tile[___][___];
}
```

<details>
<summary>点击查看答案解析</summary>

**答案**：
```cpp
// 1. 从全局内存加载到共享内存
tile[threadIdx.y][threadIdx.x] = in[y * width + x];

// 2. 从共享内存写入（转置）
out[out_y * width + out_x] = tile[threadIdx.x][threadIdx.y];
```

**解析**：
- 加载时使用 `threadIdx.y][threadIdx.x]`（行优先）
- 存储时交换索引为 `threadIdx.x][threadIdx.y]`（实现转置）
- `__syncthreads()` 确保所有线程完成加载后再读取
</details>

---

### 反思问题 💭

1. **性能思考**：为什么共享内存比全局内存快？硬件层面是如何实现的？

2. **设计决策**：在什么情况下你会选择使用共享内存？有什么代价吗？

3. **实际问题**：如果你遇到"shared memory usage exceeds limit"错误，如何解决？

4. **知识迁移**：CUDA 的内存层次和 CPU 的缓存层次（L1/L2/L3）有什么相似和不同？

---

## 1.3 同步机制 - 自我检测

### 理解检查 ✓

- [ ] 解释 `__syncthreads()` 的作用
- [ ] 说明为什么同步必须在条件分支内谨慎使用
- [ ] 理解原子操作的使用场景
- [ ] 区分同步和原子操作的差异
- [ ] 说明 Warp 级原语的优势

### 小测验 📝

#### 题目 1：同步错误识别

以下代码有什么问题？

```cpp
__global__ void buggy_sync(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (threadIdx.x < 128) {
        data[idx] = compute_value();
        __syncthreads();  // 问题在这里！
    } else {
        use_value(data[idx]);
    }
}
```

<details>
<summary>点击查看答案解析</summary>

**答案**：

**问题**：`__syncthreads()` 在条件分支内使用，导致死锁！

**原因**：
- 只有 threadIdx.x < 128 的线程执行 `__syncthreads()`
- threadIdx.x >= 128 的线程不执行同步
- 执行同步的线程会永远等待其他线程
- 结果：程序死锁

**正确做法**：
```cpp
__global__ void fixed_sync(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (threadIdx.x < 128) {
        data[idx] = compute_value();
    }
    
    __syncthreads();  // 所有线程都执行
    
    if (threadIdx.x < 128) {
        use_value(data[idx]);
    }
}
```

</details>

---

### 反思问题 💭

1. **调试经验**：你遇到过死锁问题吗？是如何发现和解决的？

2. **性能权衡**：频繁的同步会影响性能，如何平衡正确性和性能？

3. **扩展思考**：除了 `__syncthreads()`，还有哪些同步机制？它们适用什么场景？

---

## 使用建议

### 学习计划

| 章节 | 学习 | 练习 | 自测 | 总时间 |
|------|------|------|------|--------|
| 1.1 线程层次 | 2 小时 | 1 小时 | 30 分钟 | 3.5 小时 |
| 1.2 内存模型 | 3 小时 | 2 小时 | 30 分钟 | 5.5 小时 |
| 1.3 同步机制 | 2 小时 | 1 小时 | 30 分钟 | 3.5 小时 |
| 1.4 实践练习 | - | 4 小时 | - | 4 小时 |
| **总计** | **7 小时** | **8 小时** | **2 小时** | **17 小时** |

### 评分标准

- **90-100%**：优秀！可以进入下一阶段
- **80-89%**：良好，建议复习薄弱环节
- **70-79%**：合格，需要额外练习
- **<70%**：建议重新学习本阶段内容

### 下一步

完成所有自测后：
1. 记录你的得分
2. 在学习日志中反思
3. 加入学习社区讨论
4. 准备进入下一阶段学习

---

**祝你学习顺利！** 🚀
