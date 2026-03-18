# Prima.cpp 优化实现详解

## 概述

Prima.cpp 是 llama.cpp 的分布式实现，针对**异构低资源家庭集群**进行了深度优化。相比 llama.cpp，prima.cpp 实现了 **15 倍速度提升**，同时保持**内存压力低于 10%**。

## llama.cpp vs prima.cpp

### 架构对比

```
llama.cpp (单机推理):
┌─────────────────────────────────────┐
│           单设备                     │
│  ┌─────────────────────────────┐   │
│  │    模型层 (全部在本地)       │   │
│  │    Layer 0 ... Layer N      │   │
│  └─────────────────────────────┘   │
│           GPU / CPU                 │
└─────────────────────────────────────┘

prima.cpp (分布式推理):
┌─────────────────────────────────────────────────────────────┐
│                    分布式集群                                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Device1 │  │ Device2 │  │ Device3 │  │ Device4 │       │
│  │ Layers  │  │ Layers  │  │ Layers  │  │ Layers  │       │
│  │  0-7    │  │  8-15   │  │  16-23  │  │  24-31  │       │
│  │ Mac M1  │  │ Laptop  │  │ Desktop │  │ Phone   │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
│       ↖______________Ring Pipeline_______________↗         │
└─────────────────────────────────────────────────────────────┘
```

### 性能对比

| 模型 | llama.cpp | prima.cpp | 提升 |
|------|-----------|-----------|------|
| Llama 3-70B | 10120 ms | 674 ms | **15x** |
| QwQ-32B | 224 ms | 89 ms | **2.5x** |
| DeepSeek R1 70B | 10978 ms | 724 ms | **15x** |

## 核心优化技术

### 1. Piped-Ring 并行 + 预取

#### 问题：传统流水线并行的瓶颈

```
传统 Pipeline Parallelism:
Device 1: [Compute L0-L7]  [Wait]  [Compute L0-L7]  [Wait]
Device 2:      [Wait]  [Compute L8-L15]  [Wait]  [Compute L8-L15]
                        ↑
                   "Prefetch-Release" 效应：设备空闲等待
```

#### 解决方案：Piped-Ring 并行

```
Piped-Ring Parallelism:
Device 1: [L0-L7] → [L0-L7] → [L0-L7] → ...  (连续运行)
Device 2:      [L8-L15] → [L8-L15] → ...
Device 3:           [L16-L23] → [L16-L23] → ...
Device 4:                [L24-L31] → [L24-L31] → ...
          ↓         ↓         ↓         ↓
        Token 1   Token 2   Token 3   Token 4
```

#### 代码实现

```cpp
// 文件：src/llama.cpp (简化示例)

struct pipe_ring_context {
    int n_devices;
    int n_layers_per_device;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
};

void piped_ring_forward(pipe_ring_context & ctx, ...) {
    // 每个设备运行多个周期
    for (int cycle = 0; cycle < n_tokens; ++cycle) {
        for (int dev = 0; dev < ctx.n_devices; ++dev) {
            // 预取下一周期的权重
            prefetch_weights(dev, cycle + 1);
            
            // 计算当前层的输出
            compute_layers(dev, cycle);
            
            // 发送输出到下一个设备
            send_to_next_device(dev);
        }
    }
}
```

### 2. 异构感知负载分配

#### 设备能力评估

```cpp
// 文件：src/llama.cpp

struct device_capabilities {
    float compute_power;     // 计算能力 (GFLOPS)
    float memory_bandwidth;  // 内存带宽 (GB/s)
    float disk_speed;        // 磁盘速度 (GB/s)
    size_t memory_available; // 可用内存
    size_t vram_available;   // 可用显存
    int os_type;             // 操作系统类型
};

// 评估设备能力
device_capabilities evaluate_device(int device_id) {
    device_capabilities caps;
    
    // 测试计算能力
    caps.compute_power = benchmark_compute();
    
    // 测试内存带宽
    caps.memory_bandwidth = benchmark_memory();
    
    // 测试磁盘速度
    caps.disk_speed = benchmark_disk();
    
    // 获取可用内存
    caps.memory_available = get_available_memory();
    
    return caps;
}
```

#### 负载分配算法

```cpp
// 根据设备能力分配层数
std::vector<int> distribute_layers(
    const std::vector<device_capabilities> & devices,
    int total_layers) {
    
    std::vector<int> layers_per_device(devices.size());
    
    // 计算每个设备的权重
    float total_weight = 0;
    for (const auto & dev : devices) {
        float weight = dev.compute_power * 0.4f + 
                       dev.memory_bandwidth * 0.3f +
                       dev.disk_speed * 0.3f;
        total_weight += weight;
    }
    
    // 按权重分配层数
    for (size_t i = 0; i < devices.size(); ++i) {
        float weight = devices[i].compute_power * 0.4f + ...;
        layers_per_device[i] = (int)(total_layers * weight / total_weight);
    }
    
    return layers_per_device;
}
```

### 3. GPU & CPU 混合卸载

#### 问题：显存不足

```
70B 模型 Q4_K 量化后约 40GB
RTX 3090 只有 24GB 显存
→ 无法完全放入显存
```

#### 解决方案：混合卸载

```
┌─────────────────────────────────────┐
│           RTX 3090                  │
│  ┌─────────────────────────────┐   │
│  │  VRAM (24GB): Layer 0-15    │   │
│  │  快速计算                    │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │  RAM (部分): Layer 16-31    │   │
│  │  较慢计算                    │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

#### 代码实现

```cpp
// 文件：src/llama.cpp

struct offload_config {
    int n_layers_gpu;  // GPU 层数
    int n_layers_cpu;  // CPU 层数
    bool use_mmap;     // 使用 mmap
};

offload_config compute_offload_config(
    size_t vram_available,
    size_t ram_available,
    size_t model_size) {
    
    offload_config config;
    
    // 计算可以放入显存的层数
    size_t layer_size = model_size / total_layers;
    config.n_layers_gpu = vram_available / layer_size;
    
    // 剩余层放入内存
    config.n_layers_cpu = total_layers - config.n_layers_gpu;
    
    // 使用 mmap 延迟加载
    config.use_mmap = true;
    
    return config;
}
```

### 4. 自动设备选择

#### 问题：弱设备拖慢整体速度

```
集群配置：
- Device 1: Mac M1 (快)
- Device 2: Laptop (快)
- Device 3: 旧手机 (慢) ← 瓶颈
- Device 4: Desktop (快)

结果：整体速度受限于最慢的设备
```

#### 解决方案：自动排除弱设备

```cpp
// 文件：src/llama.cpp

std::vector<int> select_devices(
    const std::vector<device_capabilities> & devices) {
    
    std::vector<int> selected;
    float total_speed_without = 0;
    
    // 评估每个设备的贡献
    for (size_t i = 0; i < devices.size(); ++i) {
        // 计算不包含该设备时的速度
        float speed_without = estimate_speed_without(devices, i);
        
        // 如果排除该设备后速度更快，则排除
        if (speed_without > total_speed_without) {
            // 标记为排除
            continue;
        }
        
        selected.push_back(i);
        total_speed_without = speed_without;
    }
    
    return selected;
}
```

### 5. 内存压力控制

#### mmap 延迟加载

```cpp
// 文件：src/llama.cpp

// 使用 mmap 加载模型
void * model_data = mmap(NULL, model_size, 
                          PROT_READ, 
                          MAP_PRIVATE, 
                          fd, 0);

// OS 按需加载页面，内存压力低
// 当内存不足时，OS 自动释放页面缓存
```

#### 内存压力监控

```cpp
// 监控内存使用
void monitor_memory_pressure() {
    size_t used_memory = get_used_memory();
    size_t total_memory = get_total_memory();
    float pressure = (float)used_memory / total_memory;
    
    // 保持内存压力低于 10%
    if (pressure > 0.1f) {
        // 主动释放缓存
        release_page_cache();
    }
}
```

### 6. 推测解码 (Speculative Decoding)

#### 原理

```
标准解码：
[Target Model] → Token 1 → Token 2 → Token 3 → ...
                每次生成一个 token

推测解码：
[Draft Model]  → Token 1, 2, 3, 4 (快速生成)
[Target Model] → 验证 Token 1, 2, 3, 4
                如果正确，一次获得多个 token
```

#### 实现

```cpp
// 文件：src/llama.cpp

void speculative_decode(
    llama_model & target_model,
    llama_model & draft_model,
    const std::vector<int> & input_tokens) {
    
    // 1. 使用 draft model 快速生成多个 token
    std::vector<int> draft_tokens = draft_model.generate(input_tokens, n_draft);
    
    // 2. 使用 target model 验证
    std::vector<int> verified_tokens = target_model.verify(input_tokens, draft_tokens);
    
    // 3. 返回验证通过的 token
    // 通常可以一次通过多个 token
}
```

### 7. 动态批处理

#### 原理

```
单请求：
[Request 1] → Token 1 → Token 2 → Token 3 → ...

多请求批处理：
[Request 1] ─┐
[Request 2] ─┼→ [Batch] → Token 1,1  Token 1,2  ...
[Request 3] ─┘             Token 2,1  Token 2,2
                           Token 3,1  Token 3,2
```

#### 实现

```cpp
// 文件：src/llama.cpp

struct batch_request {
    int request_id;
    std::vector<int> input_tokens;
    std::vector<int> output_tokens;
};

void process_batch(std::vector<batch_request> & requests) {
    // 合并输入
    std::vector<int> batch_input;
    for (auto & req : requests) {
        batch_input.insert(batch_input.end(), 
                           req.input_tokens.begin(), 
                           req.input_tokens.end());
    }
    
    // 批量推理
    auto batch_output = model.forward(batch_input);
    
    // 分配输出
    int offset = 0;
    for (auto & req : requests) {
        int n_tokens = req.input_tokens.size();
        req.output_tokens.assign(
            batch_output.begin() + offset,
            batch_output.begin() + offset + n_tokens);
        offset += n_tokens;
    }
}
```

## 性能优化总结

### 优化技术对比

| 技术 | llama.cpp | prima.cpp | 效果 |
|------|-----------|-----------|------|
| 并行方式 | 单机 | 分布式 Piped-Ring | 15x 加速 |
| 内存管理 | 全量加载 | mmap + 延迟加载 | 内存压力 <10% |
| 设备利用 | 单设备 | 异构多设备 | 资源充分利用 |
| 负载均衡 | 无 | 异构感知分配 | 最优分配 |
| 弱设备处理 | 无 | 自动排除 | 避免瓶颈 |
| 显存不足 | OOM | GPU/CPU 混合卸载 | 支持大模型 |

### 适用场景

| 场景 | 推荐方案 |
|------|----------|
| 单设备、显存充足 | llama.cpp |
| 多设备、大模型 | prima.cpp |
| 显存不足 | prima.cpp (混合卸载) |
| 低内存设备 | prima.cpp (mmap) |
| 多用户并发 | prima.cpp (动态批处理) |

## 练习

1. 阅读 prima.cpp 的分布式调度代码
2. 理解 Piped-Ring 并行的实现
3. 尝试配置多设备推理

## 下一步

完成本教程后，你已经掌握了 prima.cpp 的核心优化技术，可以开始实际的项目维护和开发工作！
