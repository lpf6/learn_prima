# Prima.cpp 优化技术详解

本章节深入讲解prima.cpp特有的优化技术，这些优化在llama.cpp基础上进一步提升了多GPU、异构计算和大规模推理的性能。

## 目录
1. [Piped-Ring 并行计算](#1-piped-ring-并行计算)
2. [异构工作负载分配](#2-异构工作负载分配)
3. [GPU/CPU 混合卸载](#3-gpucpu-混合卸载)
4. [自动设备选择](#4-自动设备选择)
5. [内存压力控制](#5-内存压力控制)
6. [投机解码](#6-投机解码)
7. [动态批处理](#7-动态批处理)

---

## 1. Piped-Ring 并行计算

### 1.1 核心概念

Piped-Ring并行是prima.cpp针对多GPU场景设计的创新并行策略，结合了流水线并行和环形通信的优势：

- **流水线并行**：将模型按层分割到不同GPU，每个GPU处理连续的层
- **环形通信**：GPU间通过环形拓扑交换数据，减少通信瓶颈
- **流水线填充**：通过微批次填充流水线，提高GPU利用率

### 1.2 架构设计

```cpp
// prima.cpp/src/multi_gpu.h
struct prima_multi_gpu_context {
    int num_devices;                    // GPU数量
    int device_ids[MAX_GPUS];           // GPU ID列表
    int pipeline_stages[MAX_GPUS];      // 每个GPU的层数
    int total_layers;                   // 总层数
    
    // 流水线状态
    int micro_batch_size;               // 微批次大小
    int num_pipeline_stages;            // 流水线阶段数
    int current_stage;                  // 当前阶段
    
    // 环形通信
    cudaStream_t comm_streams[MAX_GPUS]; // 通信流
    void * send_buffers[MAX_GPUS];      // 发送缓冲区
    void * recv_buffers[MAX_GPUS];      // 接收缓冲区
    size_t buffer_size;                 // 缓冲区大小
    
    // 同步
    cudaEvent_t events[MAX_GPUS];       // 同步事件
    int forward_stage;                  // 前向传播阶段
    int backward_stage;                 // 反向传播阶段
};

// 环形拓扑配置
struct ring_topology {
    int rank;                           // 当前GPU rank
    int next_rank;                      // 下一个GPU rank
    int prev_rank;                      // 上一个GPU rank
    int world_size;                     // 总GPU数
};
```

### 1.3 流水线并行实现

```cpp
// prima.cpp/src/multi_gpu.c
void prima_pipeline_forward(
    struct prima_multi_gpu_context * ctx,
    const float * input,
    float * output,
    int batch_size,
    int seq_len
) {
    // 1. 分割输入到第一个GPU
    int first_device = ctx->device_ids[0];
    cudaSetDevice(first_device);
    
    // 分配中间缓冲区
    float * pipeline_buffers[ctx->num_devices];
    for (int i = 0; i < ctx->num_devices; i++) {
        cudaSetDevice(ctx->device_ids[i]);
        size_t buffer_size = batch_size * seq_len * HIDDEN_DIM * sizeof(float);
        cudaMalloc(&pipeline_buffers[i], buffer_size);
    }
    
    // 2. 流水线前向传播
    for (int stage = 0; stage < ctx->num_pipeline_stages; stage++) {
        // 当前处理的GPU
        int device_idx = stage % ctx->num_devices;
        int device_id = ctx->device_ids[device_idx];
        cudaSetDevice(device_id);
        
        // 确定输入和输出
        float * stage_input = (stage == 0) ? (float *)input : 
                              pipeline_buffers[(stage - 1) % ctx->num_devices];
        float * stage_output = pipeline_buffers[device_idx];
        
        // 处理当前阶段的层
        int layer_start = device_idx * ctx->pipeline_stages[device_idx];
        int layer_end = layer_start + ctx->pipeline_stages[device_idx];
        
        for (int layer = layer_start; layer < layer_end; layer++) {
            // 执行transformer层
            prima_transformer_layer_forward(
                ctx->model,
                layer,
                stage_input,
                stage_output,
                batch_size,
                seq_len
            );
            
            // 更新输入为当前输出
            stage_input = stage_output;
        }
        
        // 同步当前GPU
        cudaDeviceSynchronize();
    }
    
    // 3. 从最后一个GPU获取输出
    int last_device = ctx->device_ids[ctx->num_devices - 1];
    cudaSetDevice(last_device);
    cudaMemcpy(output, pipeline_buffers[ctx->num_devices - 1],
               batch_size * seq_len * HIDDEN_DIM * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    // 4. 清理缓冲区
    for (int i = 0; i < ctx->num_devices; i++) {
        cudaSetDevice(ctx->device_ids[i]);
        cudaFree(pipeline_buffers[i]);
    }
}
```

### 1.4 环形通信优化

```cpp
// prima.cpp/src/ring_comm.c
void prima_ring_allreduce(
    struct prima_multi_gpu_context * ctx,
    float * buffer,
    size_t count,
    int rank
) {
    const int world_size = ctx->num_devices;
    const size_t chunk_size = count / world_size;
    
    // 1. Reduce-scatter: 每个GPU累加部分数据
    for (int step = 0; step < world_size - 1; step++) {
        int send_to = (rank + 1) % world_size;
        int recv_from = (rank - 1 + world_size) % world_size;
        
        // 计算发送和接收的数据块
        int send_chunk = (rank - step + world_size) % world_size;
        int recv_chunk = (rank - step - 1 + world_size) % world_size;
        
        float * send_ptr = buffer + send_chunk * chunk_size;
        float * recv_ptr = buffer + recv_chunk * chunk_size;
        
        // 异步发送和接收
        cudaMemcpyAsync(ctx->send_buffers[rank], send_ptr,
                       chunk_size * sizeof(float),
                       cudaMemcpyDeviceToDevice, ctx->comm_streams[rank]);
        
        cudaMemcpyAsync(recv_ptr, ctx->recv_buffers[rank],
                       chunk_size * sizeof(float),
                       cudaMemcpyDeviceToDevice, ctx->comm_streams[rank]);
        
        // 累加接收的数据
        vector_add_kernel<<<(chunk_size + 255) / 256, 256>>>(
            recv_ptr, ctx->recv_buffers[rank], chunk_size
        );
        
        cudaStreamSynchronize(ctx->comm_streams[rank]);
    }
    
    // 2. All-gather: 广播完整数据
    for (int step = 0; step < world_size - 1; step++) {
        int send_to = (rank + 1) % world_size;
        int recv_from = (rank - 1 + world_size) % world_size;
        
        int send_chunk = (rank - step + world_size) % world_size;
        int recv_chunk = (rank - step - 1 + world_size) % world_size;
        
        float * send_ptr = buffer + send_chunk * chunk_size;
        float * recv_ptr = buffer + recv_chunk * chunk_size;
        
        cudaMemcpyAsync(recv_ptr, send_ptr,
                       chunk_size * sizeof(float),
                       cudaMemcpyDeviceToDevice, ctx->comm_streams[rank]);
        
        cudaStreamSynchronize(ctx->comm_streams[rank]);
    }
}

// 优化的点对点通信
void prima_ring_sendrecv(
    struct prima_multi_gpu_context * ctx,
    float * send_data,
    float * recv_data,
    size_t count,
    int dest,
    int src
) {
    int rank = ctx->device_ids[0]; // 当前rank
    
    // 使用P2P直接访问（如果支持）
    int can_access_peer = 0;
    cudaDeviceCanAccessPeer(&can_access_peer, rank, dest);
    
    if (can_access_peer) {
        // 直接P2P访问，避免拷贝
        cudaMemcpyPeerAsync(recv_data, rank, send_data, dest,
                          count * sizeof(float), ctx->comm_streams[rank]);
    } else {
        // 通过主机中转
        cudaMemcpyAsync(ctx->send_buffers[rank], send_data,
                       count * sizeof(float),
                       cudaMemcpyDeviceToHost, ctx->comm_streams[rank]);
        
        cudaMemcpyAsync(recv_data, ctx->recv_buffers[rank],
                       count * sizeof(float),
                       cudaMemcpyHostToDevice, ctx->comm_streams[rank]);
    }
    
    cudaStreamSynchronize(ctx->comm_streams[rank]);
}
```

### 1.5 流水线调度优化

```cpp
// prima.cpp/src/pipeline_scheduler.c
struct pipeline_schedule {
    int micro_batch_id;                 // 微批次ID
    int stage_id;                       // 阶段ID
    int device_id;                      // 设备ID
    cudaEvent_t start_event;            // 开始事件
    cudaEvent_t end_event;              // 结束事件
};

void prima_pipeline_schedule(
    struct prima_multi_gpu_context * ctx,
    float * input,
    float * output,
    int total_batch_size
) {
    const int num_micro_batches = (total_batch_size + ctx->micro_batch_size - 1) / 
                                   ctx->micro_batch_size;
    
    // 创建调度表
    struct pipeline_schedule schedule[num_micro_batches][ctx->num_devices];
    
    // 1. 流水线填充阶段
    for (int mb = 0; mb < num_micro_batches + ctx->num_devices - 1; mb++) {
        for (int dev = 0; dev < ctx->num_devices; dev++) {
            int micro_batch_idx = mb - dev;
            
            if (micro_batch_idx >= 0 && micro_batch_idx < num_micro_batches) {
                // 计算输入和输出位置
                float * mb_input = (dev == 0) ? 
                                   input + micro_batch_idx * ctx->micro_batch_size * HIDDEN_DIM :
                                   ctx->pipeline_buffers[(mb - 1) % ctx->num_devices];
                
                float * mb_output = ctx->pipeline_buffers[dev];
                
                // 记录调度信息
                schedule[mb][dev].micro_batch_id = micro_batch_idx;
                schedule[mb][dev].stage_id = dev;
                schedule[mb][dev].device_id = ctx->device_ids[dev];
                
                // 异步执行
                cudaSetDevice(ctx->device_ids[dev]);
                cudaEventRecord(schedule[mb][dev].start_event, 
                               ctx->comm_streams[dev]);
                
                prima_process_micro_batch(
                    ctx,
                    mb_input,
                    mb_output,
                    ctx->micro_batch_size,
                    dev
                );
                
                cudaEventRecord(schedule[mb][dev].end_event, 
                               ctx->comm_streams[dev]);
            }
        }
    }
    
    // 2. 收集输出
    for (int mb = 0; mb < num_micro_batches; mb++) {
        int final_dev = ctx->num_devices - 1;
        int output_idx = mb * ctx->micro_batch_size * HIDDEN_DIM;
        
        cudaSetDevice(ctx->device_ids[final_dev]);
        cudaMemcpyAsync(output + output_idx,
                       ctx->pipeline_buffers[final_dev],
                       ctx->micro_batch_size * HIDDEN_DIM * sizeof(float),
                       cudaMemcpyDeviceToHost,
                       ctx->comm_streams[final_dev]);
    }
    
    // 3. 同步所有设备
    for (int dev = 0; dev < ctx->num_devices; dev++) {
        cudaSetDevice(ctx->device_ids[dev]);
        cudaStreamSynchronize(ctx->comm_streams[dev]);
    }
}
```

### 1.6 性能优化技巧

**1. 计算通信重叠**
```cpp
// 在计算的同时进行通信
void prima_overlap_compute_comm(
    struct prima_multi_gpu_context * ctx,
    float * input,
    float * output
) {
    // 启动异步通信
    prima_ring_sendrecv_async(ctx, input, output, BUFFER_SIZE, next_rank, prev_rank);
    
    // 同时进行计算
    prima_compute_kernel<<<blocks, threads>>>(input, output);
    
    // 等待通信完成
    cudaStreamSynchronize(ctx->comm_streams[0]);
}
```

**2. 梯度累积**
```cpp
// 累积梯度减少通信频率
void prima_gradient_accumulation(
    struct prima_multi_gpu_context * ctx,
    float * gradients,
    int accumulation_steps
) {
    static float accumulated_gradients[BUFFER_SIZE] = {0};
    static int step_count = 0;
    
    // 累积梯度
    vector_add(accumulated_gradients, gradients, BUFFER_SIZE);
    step_count++;
    
    // 达到累积步数后同步
    if (step_count >= accumulation_steps) {
        prima_ring_allreduce(ctx, accumulated_gradients, BUFFER_SIZE, 0);
        
        // 清零
        memset(accumulated_gradients, 0, BUFFER_SIZE * sizeof(float));
        step_count = 0;
    }
}
```

---

## 2. 异构工作负载分配

### 2.1 核心概念

异构工作负载分配是指根据不同计算设备的特性（GPU、CPU、不同型号GPU）智能分配任务，最大化整体性能：

- **性能建模**：为每个设备建立性能模型
- **动态调度**：根据实时性能调整任务分配
- **负载均衡**：确保各设备负载均衡
- **故障恢复**：设备故障时自动重新分配

### 2.2 设备性能建模

```cpp
// prima.cpp/src/device_profiler.h
struct device_profile {
    int device_id;
    char device_name[256];
    
    // 硬件规格
    int compute_capability;
    int multi_processor_count;
    int max_threads_per_block;
    size_t total_memory;
    size_t shared_memory_per_block;
    
    // 性能指标
    float peak_flops;                    // 峰值FLOPS
    float memory_bandwidth;              // 内存带宽
    float compute_throughput;            // 计算吞吐量
    float memory_latency;                // 内存延迟
    
    // 历史性能数据
    float avg_compute_time;             // 平均计算时间
    float avg_comm_time;                // 平均通信时间
    float success_rate;                  // 成功率
    int total_tasks;                     // 总任务数
    int failed_tasks;                    // 失败任务数
    
    // 权重因子
    float compute_weight;                // 计算权重
    float memory_weight;                 // 内存权重
    float comm_weight;                   // 通信权重
};

struct workload_characteristics {
    size_t input_size;                   // 输入大小
    size_t output_size;                  // 输出大小
    size_t parameter_size;               // 参数大小
    float compute_intensity;             // 计算强度 (FLOPs/Byte)
    float memory_intensity;              // 内存强度 (Bytes/FLOP)
    float parallelism;                   // 并行度
    float dependency;                    // 依赖度
};
```

### 2.3 性能分析器实现

```cpp
// prima.cpp/src/device_profiler.c
void prima_profile_device(struct device_profile * profile) {
    cudaSetDevice(profile->device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, profile->device_id);
    
    // 获取硬件规格
    profile->compute_capability = prop.major * 10 + prop.minor;
    profile->multi_processor_count = prop.multiProcessorCount;
    profile->max_threads_per_block = prop.maxThreadsPerBlock;
    profile->total_memory = prop.totalGlobalMem;
    profile->shared_memory_per_block = prop.sharedMemPerBlock;
    
    // 计算峰值FLOPS
    float clock_rate = prop.clockRate / 1000.0f; // MHz to GHz
    float cores_per_sm = get_cores_per_sm(profile->compute_capability);
    profile->peak_flops = profile->multi_processor_count * 
                         cores_per_sm * clock_rate * 2; // FMA
    
    // 测量内存带宽
    profile->memory_bandwidth = measure_memory_bandwidth(profile->device_id);
    
    // 测量计算吞吐量
    profile->compute_throughput = measure_compute_throughput(profile->device_id);
    
    // 测量内存延迟
    profile->memory_latency = measure_memory_latency(profile->device_id);
    
    // 初始化历史数据
    profile->avg_compute_time = 0.0f;
    profile->avg_comm_time = 0.0f;
    profile->success_rate = 1.0f;
    profile->total_tasks = 0;
    profile->failed_tasks = 0;
    
    // 计算权重因子
    profile->compute_weight = profile->compute_throughput / profile->peak_flops;
    profile->memory_weight = profile->memory_bandwidth / (1000.0f * 1024 * 1024 * 1024); // GB/s
    profile->comm_weight = 1.0f / (profile->memory_latency + 0.001f);
}

float measure_memory_bandwidth(int device_id) {
    const size_t buffer_size = 256 * 1024 * 1024; // 256 MB
    float * d_buffer;
    
    cudaSetDevice(device_id);
    cudaMalloc(&d_buffer, buffer_size);
    
    // 预热
    cudaMemcpy(d_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToDevice);
    
    // 测量拷贝时间
    const int iterations = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cudaMemcpy(d_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    // 计算带宽
    float bandwidth = (buffer_size * iterations) / (elapsed_time / 1000.0f);
    
    cudaFree(d_buffer);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return bandwidth;
}

float measure_compute_throughput(int device_id) {
    const int N = 1024 * 1024;
    const int iterations = 100;
    
    float * d_a, * d_b, * d_c;
    cudaSetDevice(device_id);
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    // SAXPY kernel
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        saxpy_kernel<<<blocks, threads>>>(N, 2.0f, d_a, d_b, d_c);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    // 计算吞吐量 (GFLOPS)
    float flops = 2.0f * N * iterations; // 每次迭代2N次FLOP
    float throughput = flops / (elapsed_time / 1000.0f) / 1e9;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return throughput;
}
```

### 2.4 智能任务分配

```cpp
// prima.cpp/src/workload_scheduler.c
struct task_assignment {
    int device_id;
    float expected_time;
    float confidence;
};

struct task_assignment prima_assign_task(
    struct device_profile * profiles,
    int num_devices,
    struct workload_characteristics * workload
) {
    struct task_assignment best_assignment;
    float best_score = -INFINITY;
    
    for (int i = 0; i < num_devices; i++) {
        struct device_profile * profile = &profiles[i];
        
        // 计算匹配分数
        float compute_score = workload->compute_intensity * profile->compute_weight;
        float memory_score = workload->memory_intensity * profile->memory_weight;
        float comm_score = (1.0f - workload->parallelism) * profile->comm_weight;
        
        // 综合分数
        float total_score = compute_score + memory_score + comm_score;
        
        // 考虑历史成功率
        total_score *= profile->success_rate;
        
        // 考虑负载均衡
        float load_factor = 1.0f / (profile->total_tasks + 1);
        total_score *= load_factor;
        
        // 估算执行时间
        float estimated_time = estimate_execution_time(profile, workload);
        
        // 计算置信度
        float confidence = min(profile->total_tasks / 100.0f, 1.0f);
        
        if (total_score > best_score) {
            best_score = total_score;
            best_assignment.device_id = profile->device_id;
            best_assignment.expected_time = estimated_time;
            best_assignment.confidence = confidence;
        }
    }
    
    return best_assignment;
}

float estimate_execution_time(
    struct device_profile * profile,
    struct workload_characteristics * workload
) {
    // 计算时间 = 计算时间 + 内存时间 + 通信时间
    float compute_time = workload->compute_intensity / profile->compute_throughput;
    float memory_time = workload->memory_intensity / profile->memory_bandwidth;
    float comm_time = workload->dependency * profile->memory_latency;
    
    // 加权平均
    float total_time = compute_time + memory_time + comm_time;
    
    // 考虑历史数据
    if (profile->total_tasks > 0) {
        float historical_time = profile->avg_compute_time + profile->avg_comm_time;
        total_time = 0.7f * total_time + 0.3f * historical_time;
    }
    
    return total_time;
}
```

### 2.5 动态负载均衡

```cpp
// prima.cpp/src/dynamic_balancer.c
struct dynamic_balancer {
    struct device_profile * profiles;
    int num_devices;
    
    // 负载状态
    int current_load[MAX_GPUS];
    float load_factor[MAX_GPUS];
    
    // 迁移阈值
    float migration_threshold;
    float rebalance_interval;
    
    // 统计
    int migrations;
    int rebalances;
};

void prima_rebalance_workload(
    struct dynamic_balancer * balancer,
    struct task * tasks,
    int num_tasks
) {
    // 1. 计算当前负载
    for (int i = 0; i < balancer->num_devices; i++) {
        balancer->load_factor[i] = (float)balancer->current_load[i] / 
                                   balancer->profiles[i].total_tasks;
    }
    
    // 2. 识别过载和空闲设备
    int overloaded[MAX_GPUS], underloaded[MAX_GPUS];
    int num_overloaded = 0, num_underloaded = 0;
    
    float avg_load = 0.0f;
    for (int i = 0; i < balancer->num_devices; i++) {
        avg_load += balancer->load_factor[i];
    }
    avg_load /= balancer->num_devices;
    
    for (int i = 0; i < balancer->num_devices; i++) {
        if (balancer->load_factor[i] > avg_load * 1.2f) {
            overloaded[num_overloaded++] = i;
        } else if (balancer->load_factor[i] < avg_load * 0.8f) {
            underloaded[num_underloaded++] = i;
        }
    }
    
    // 3. 迁移任务
    for (int i = 0; i < num_overloaded && i < num_underloaded; i++) {
        int src_device = overloaded[i];
        int dst_device = underloaded[i];
        
        // 找到最适合迁移的任务
        struct task * task_to_migrate = find_best_migration_task(
            tasks, num_tasks, src_device, dst_device
        );
        
        if (task_to_migrate) {
            // 迁移任务
            migrate_task(task_to_migrate, src_device, dst_device);
            
            // 更新负载
            balancer->current_load[src_device]--;
            balancer->current_load[dst_device]++;
            balancer->migrations++;
        }
    }
    
    balancer->rebalances++;
}

struct task * find_best_migration_task(
    struct task * tasks,
    int num_tasks,
    int src_device,
    int dst_device
) {
    struct task * best_task = NULL;
    float best_benefit = -INFINITY;
    
    for (int i = 0; i < num_tasks; i++) {
        if (tasks[i].device_id == src_device) {
            // 计算迁移收益
            float src_load = balancer->load_factor[src_device];
            float dst_load = balancer->load_factor[dst_device];
            
            float benefit = (src_load - dst_load) * tasks[i].priority;
            
            if (benefit > best_benefit) {
                best_benefit = benefit;
                best_task = &tasks[i];
            }
        }
    }
    
    return best_task;
}
```

### 2.6 故障恢复机制

```cpp
// prima.cpp/src/fault_tolerance.c
struct fault_monitor {
    int device_status[MAX_GPUS];         // 设备状态: 0=正常, 1=降级, 2=故障
    int heartbeat[MAX_GPUS];             // 心跳计数
    time_t last_heartbeat[MAX_GPUS];     // 最后心跳时间
    int timeout_threshold;               // 超时阈值
};

void prima_monitor_devices(struct fault_monitor * monitor) {
    for (int i = 0; i < monitor->num_devices; i++) {
        time_t current_time;
        time(&current_time);
        
        // 检查心跳超时
        if (current_time - monitor->last_heartbeat[i] > monitor->timeout_threshold) {
            if (monitor->device_status[i] == 0) {
                // 设备降级
                monitor->device_status[i] = 1;
                log_warning("Device %d degraded, redistributing workload", i);
                
                // 重新分配工作负载
                redistribute_workload(i);
            } else if (monitor->device_status[i] == 1) {
                // 设备故障
                monitor->device_status[i] = 2;
                log_error("Device %d failed, removing from pool", i);
                
                // 从设备池移除
                remove_device(i);
            }
        }
    }
}

void redistribute_workload(int failed_device) {
    // 1. 找到该设备上的所有任务
    struct task * failed_tasks[MAX_TASKS];
    int num_failed_tasks = 0;
    
    for (int i = 0; i < total_tasks; i++) {
        if (tasks[i].device_id == failed_device) {
            failed_tasks[num_failed_tasks++] = &tasks[i];
        }
    }
    
    // 2. 重新分配任务
    for (int i = 0; i < num_failed_tasks; i++) {
        struct task_assignment assignment = prima_assign_task(
            profiles, num_devices, &failed_tasks[i]->workload
        );
        
        if (assignment.device_id != failed_device) {
            migrate_task(failed_tasks[i], failed_device, assignment.device_id);
        }
    }
}
```

---

## 3. GPU/CPU 混合卸载

### 3.1 核心概念

GPU/CPU混合卸载根据计算特性和资源可用性，智能地将不同层或操作分配到GPU或CPU执行：

- **层级卸载**：将部分层卸载到CPU
- **操作级卸载**：将特定操作（如RMSNorm）卸载到CPU
- **动态卸载**：根据内存压力动态调整
- **流水线卸载**：GPU和CPU流水线并行

### 3.2 卸载策略

```cpp
// prima.cpp/src/offload_strategy.h
enum offload_decision {
    OFFLOAD_GPU,                         // 卸载到GPU
    OFFLOAD_CPU,                         // 卸载到CPU
    OFFLOAD_HYBRID                       // 混合卸载
};

struct offload_config {
    // GPU配置
    int gpu_layers;                      // GPU层数
    size_t gpu_memory_budget;            // GPU内存预算
    float gpu_threshold;                 // GPU卸载阈值
    
    // CPU配置
    int cpu_threads;                     // CPU线程数
    size_t cpu_memory_budget;            // CPU内存预算
    float cpu_threshold;                 // CPU卸载阈值
    
    // 动态配置
    bool dynamic_offload;                // 动态卸载
    float memory_pressure_threshold;     // 内存压力阈值
    int rebalance_interval;              // 重新平衡间隔
};

struct layer_offload_info {
    int layer_id;
    enum offload_decision decision;
    float gpu_score;                     // GPU得分
    float cpu_score;                     // CPU得分
    size_t memory_size;                  // 内存占用
    float compute_cost;                  // 计算成本
};
```

### 3.3 智能卸载决策

```cpp
// prima.cpp/src/offload_manager.c
enum offload_decision prima_decide_offload(
    struct offload_config * config,
    int layer_id,
    size_t layer_size,
    float compute_intensity
) {
    // 1. 检查GPU内存
    size_t gpu_free_memory = get_gpu_free_memory();
    if (gpu_free_memory < layer_size * 1.2f) {
        // GPU内存不足，强制CPU
        return OFFLOAD_CPU;
    }
    
    // 2. 计算GPU和CPU得分
    float gpu_score = calculate_gpu_score(config, layer_size, compute_intensity);
    float cpu_score = calculate_cpu_score(config, layer_size, compute_intensity);
    
    // 3. 考虑内存压力
    float memory_pressure = 1.0f - (float)gpu_free_memory / config->gpu_memory_budget;
    if (memory_pressure > config->memory_pressure_threshold) {
        // 内存压力大，偏向CPU
        cpu_score *= (1.0f + memory_pressure);
    }
    
    // 4. 做出决策
    if (gpu_score > cpu_score * config->gpu_threshold) {
        return OFFLOAD_GPU;
    } else if (cpu_score > gpu_score * config->cpu_threshold) {
        return OFFLOAD_CPU;
    } else {
        return OFFLOAD_HYBRID;
    }
}

float calculate_gpu_score(
    struct offload_config * config,
    size_t layer_size,
    float compute_intensity
) {
    float score = 0.0f;
    
    // 计算强度得分
    score += compute_intensity * 2.0f;
    
    // 内存效率得分
    float memory_efficiency = 1.0f - (float)layer_size / config->gpu_memory_budget;
    score += memory_efficiency;
    
    // 批处理得分
    score += config->gpu_threshold;
    
    return score;
}

float calculate_cpu_score(
    struct offload_config * config,
    size_t layer_size,
    float compute_intensity
) {
    float score = 0.0f;
    
    // 内存容量得分
    float memory_capacity = (float)layer_size / config->cpu_memory_budget;
    score += memory_capacity;
    
    // 计算复杂度得分（CPU适合低计算强度）
    score += (1.0f - compute_intensity) * 2.0f;
    
    // 线程利用率得分
    float thread_utilization = (float)config->cpu_threads / get_cpu_cores();
    score += thread_utilization;
    
    return score;
}
```

### 3.4 混合执行实现

```cpp
// prima.cpp/src/hybrid_executor.c
struct hybrid_execution_context {
    // GPU上下文
    cudaStream_t gpu_stream;
    void * gpu_buffers[MAX_LAYERS];
    
    // CPU上下文
    int cpu_threads;
    void * cpu_buffers[MAX_LAYERS];
    pthread_t cpu_worker_threads[MAX_THREADS];
    
    // 同步
    cudaEvent_t gpu_events[MAX_LAYERS];
    pthread_mutex_t sync_mutex;
    pthread_cond_t sync_cond;
    
    // 状态
    int pending_layers;
    int completed_layers;
};

void prima_hybrid_forward(
    struct hybrid_execution_context * ctx,
    struct layer_offload_info * offload_info,
    int num_layers,
    float * input,
    float * output
) {
    float * current_input = input;
    
    for (int i = 0; i < num_layers; i++) {
        struct layer_offload_info * info = &offload_info[i];
        
        if (info->decision == OFFLOAD_GPU) {
            // GPU执行
            cudaSetDevice(0);
            prima_gpu_layer_forward(
                ctx->gpu_buffers[i],
                current_input,
                ctx->gpu_buffers[(i + 1) % num_layers],
                info->layer_id
            );
            
            cudaEventRecord(ctx->gpu_events[i], ctx->gpu_stream);
            current_input = ctx->gpu_buffers[(i + 1) % num_layers];
            
        } else if (info->decision == OFFLOAD_CPU) {
            // CPU执行
            // 等待GPU数据传输完成
            if (i > 0 && offload_info[i - 1].decision == OFFLOAD_GPU) {
                cudaEventSynchronize(ctx->gpu_events[i - 1]);
                cudaMemcpy(ctx->cpu_buffers[i], current_input,
                          BUFFER_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost);
            }
            
            prima_cpu_layer_forward(
                ctx->cpu_buffers[i],
                current_input,
                ctx->cpu_buffers[(i + 1) % num_layers],
                info->layer_id,
                ctx->cpu_threads
            );
            
            current_input = ctx->cpu_buffers[(i + 1) % num_layers];
            
        } else {
            // 混合执行：部分GPU部分CPU
            prima_hybrid_layer_forward(
                ctx,
                info,
                current_input,
                ctx->gpu_buffers[(i + 1) % num_layers],
                ctx->cpu_buffers[(i + 1) % num_layers]
            );
            
            current_input = ctx->gpu_buffers[(i + 1) % num_layers];
        }
    }
    
    // 最终输出
    if (offload_info[num_layers - 1].decision == OFFLOAD_CPU) {
        cudaMemcpy(output, current_input, BUFFER_SIZE * sizeof(float),
                  cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(output, current_input, BUFFER_SIZE * sizeof(float),
                  cudaMemcpyDeviceToDevice);
    }
}

void prima_hybrid_layer_forward(
    struct hybrid_execution_context * ctx,
    struct layer_offload_info * info,
    float * input,
    float * gpu_output,
    float * cpu_output
) {
    // 将层分割为GPU和CPU部分
    int gpu_split = info->layer_id * 0.7f; // 70% GPU, 30% CPU
    int cpu_split = info->layer_id - gpu_split;
    
    // GPU部分
    cudaSetDevice(0);
    prima_gpu_partial_forward(
        gpu_output,
        input,
        gpu_split,
        info->layer_id
    );
    
    // CPU部分（异步）
    pthread_t cpu_thread;
    struct cpu_task_args args = {
        .input = input,
        .output = cpu_output,
        .split = cpu_split,
        .layer_id = info->layer_id,
        .threads = ctx->cpu_threads
    };
    
    pthread_create(&cpu_thread, NULL, cpu_layer_worker, &args);
    
    // 等待CPU完成
    pthread_join(cpu_thread, NULL);
    
    // 合并结果
    merge_outputs(gpu_output, cpu_output, info->layer_id);
}
```

### 3.5 动态内存管理

```cpp
// prima.cpp/src/dynamic_memory.c
struct dynamic_memory_manager {
    // GPU内存池
    void * gpu_pool;
    size_t gpu_pool_size;
    size_t gpu_used;
    
    // CPU内存池
    void * cpu_pool;
    size_t cpu_pool_size;
    size_t cpu_used;
    
    // 分配记录
    struct allocation_record {
        void * ptr;
        size_t size;
        int device_id;
        bool active;
    } records[MAX_ALLOCATIONS];
    
    int num_records;
};

void * prima_dynamic_alloc(
    struct dynamic_memory_manager * manager,
    size_t size,
    int preferred_device
) {
    // 1. 尝试在首选设备分配
    if (preferred_device >= 0) {
        void * ptr = try_allocate(manager, size, preferred_device);
        if (ptr) {
            return ptr;
        }
    }
    
    // 2. 尝试在其他设备分配
    for (int device = 0; device < 2; device++) {
        if (device != preferred_device) {
            void * ptr = try_allocate(manager, size, device);
            if (ptr) {
                return ptr;
            }
        }
    }
    
    // 3. 触发内存回收
    trigger_memory_reclamation(manager);
    
    // 4. 再次尝试分配
    for (int device = 0; device < 2; device++) {
        void * ptr = try_allocate(manager, size, device);
        if (ptr) {
            return ptr;
        }
    }
    
    // 5. 分配失败
    log_error("Failed to allocate %zu bytes", size);
    return NULL;
}

void * try_allocate(
    struct dynamic_memory_manager * manager,
    size_t size,
    int device_id
) {
    if (device_id == 0) {
        // GPU分配
        if (manager->gpu_used + size <= manager->gpu_pool_size) {
            void * ptr = (char *)manager->gpu_pool + manager->gpu_used;
            manager->gpu_used += size;
            
            // 记录分配
            manager->records[manager->num_records++] = (struct allocation_record){
                .ptr = ptr,
                .size = size,
                .device_id = device_id,
                .active = true
            };
            
            return ptr;
        }
    } else {
        // CPU分配
        if (manager->cpu_used + size <= manager->cpu_pool_size) {
            void * ptr = (char *)manager->cpu_pool + manager->cpu_used;
            manager->cpu_used += size;
            
            // 记录分配
            manager->records[manager->num_records++] = (struct allocation_record){
                .ptr = ptr,
                .size = size,
                .device_id = device_id,
                .active = true
            };
            
            return ptr;
        }
    }
    
    return NULL;
}

void trigger_memory_reclamation(struct dynamic_memory_manager * manager) {
    // 1. 识别不活跃的分配
    for (int i = 0; i < manager->num_records; i++) {
        if (!manager->records[i].active) {
            // 释放内存
            if (manager->records[i].device_id == 0) {
                cudaFree(manager->records[i].ptr);
                manager->gpu_used -= manager->records[i].size;
            } else {
                free(manager->records[i].ptr);
                manager->cpu_used -= manager->records[i].size;
            }
            
            // 标记为已释放
            manager->records[i].ptr = NULL;
            manager->records[i].size = 0;
        }
    }
    
    // 2. 压缩记录
    int write_idx = 0;
    for (int i = 0; i < manager->num_records; i++) {
        if (manager->records[i].ptr != NULL) {
            manager->records[write_idx++] = manager->records[i];
        }
    }
    manager->num_records = write_idx;
}
```

---

## 4. 自动设备选择

### 4.1 核心概念

自动设备选择根据硬件特性、工作负载特征和性能历史，自动选择最优的计算设备：

- **设备发现**：自动检测可用设备
- **能力评估**：评估设备计算能力
- **性能预测**：预测任务执行性能
- **自适应选择**：根据实际性能调整选择

### 4.2 设备发现和评估

```cpp
// prima.cpp/src/device_discovery.c
struct device_info {
    int device_id;
    enum device_type type;              // DEVICE_GPU, DEVICE_CPU, DEVICE_ACCELERATOR
    char name[256];
    
    // GPU特有
    int compute_capability;
    int multi_processor_count;
    size_t total_memory;
    float clock_rate;
    
    // CPU特有
    int num_cores;
    int num_threads;
    float base_frequency;
    float turbo_frequency;
    
    // 性能指标
    float compute_score;
    float memory_score;
    float overall_score;
};

struct device_pool {
    struct device_info devices[MAX_DEVICES];
    int num_devices;
    
    // 性能历史
    float performance_history[MAX_DEVICES][MAX_HISTORY];
    int history_size[MAX_DEVICES];
};

void prima_discover_devices(struct device_pool * pool) {
    pool->num_devices = 0;
    
    // 1. 发现GPU设备
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    
    for (int i = 0; i < gpu_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        pool->devices[pool->num_devices] = (struct device_info){
            .device_id = i,
            .type = DEVICE_GPU,
            .compute_capability = prop.major * 10 + prop.minor,
            .multi_processor_count = prop.multiProcessorCount,
            .total_memory = prop.totalGlobalMem,
            .clock_rate = prop.clockRate / 1000.0f
        };
        
        strncpy(pool->devices[pool->num_devices].name, prop.name, 255);
        pool->num_devices++;
    }
    
    // 2. 发现CPU设备
    pool->devices[pool->num_devices] = (struct device_info){
        .device_id = -1,
        .type = DEVICE_CPU,
        .num_cores = get_cpu_cores(),
        .num_threads = get_cpu_threads(),
        .base_frequency = get_cpu_base_frequency(),
        .turbo_frequency = get_cpu_turbo_frequency()
    };
    
    strncpy(pool->devices[pool->num_devices].name, "CPU", 255);
    pool->num_devices++;
    
    // 3. 评估设备性能
    for (int i = 0; i < pool->num_devices; i++) {
        evaluate_device_performance(&pool->devices[i]);
    }
    
    // 4. 排序设备
    qsort(pool->devices, pool->num_devices, sizeof(struct device_info),
          compare_device_score);
}

void evaluate_device_performance(struct device_info * device) {
    if (device->type == DEVICE_GPU) {
        // GPU性能评估
        device->compute_score = evaluate_gpu_compute(device->device_id);
        device->memory_score = evaluate_gpu_memory(device->device_id);
    } else {
        // CPU性能评估
        device->compute_score = evaluate_cpu_compute();
        device->memory_score = evaluate_cpu_memory();
    }
    
    // 综合得分
    device->overall_score = 0.6f * device->compute_score + 
                           0.4f * device->memory_score;
}

float evaluate_gpu_compute(int device_id) {
    const int N = 1024 * 1024;
    const int iterations = 100;
    
    float * d_a, * d_b, * d_c;
    cudaSetDevice(device_id);
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    // GEMM基准测试
    const int M = 1024, K = 1024, N = 1024;
    const float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K, &alpha, d_b, N, d_a, K,
                   &beta, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    // 计算GFLOPS
    float flops = 2.0f * M * K * N * iterations;
    float gflops = flops / (elapsed_time / 1000.0f) / 1e9;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return gflops;
}
```

### 4.3 智能设备选择

```cpp
// prima.cpp/src/device_selector.c
struct device_selection {
    int device_id;
    float confidence;
    float estimated_time;
    struct selection_reason {
        char reason[256];
        float weight;
    } reasons[3];
    int num_reasons;
};

struct device_selection prima_select_device(
    struct device_pool * pool,
    struct workload_characteristics * workload
) {
    struct device_selection best_selection;
    float best_score = -INFINITY;
    
    for (int i = 0; i < pool->num_devices; i++) {
        struct device_info * device = &pool->devices[i];
        struct device_selection selection;
        
        // 计算匹配分数
        selection.device_id = device->device_id;
        selection.num_reasons = 0;
        
        // 1. 计算强度匹配
        float compute_match = match_compute_intensity(device, workload);
        add_selection_reason(&selection, "Compute intensity match", compute_match);
        
        // 2. 内存容量匹配
        float memory_match = match_memory_capacity(device, workload);
        add_selection_reason(&selection, "Memory capacity match", memory_match);
        
        // 3. 历史性能
        float history_score = get_historical_score(pool, device->device_id, workload);
        add_selection_reason(&selection, "Historical performance", history_score);
        
        // 综合分数
        float total_score = compute_match * 0.4f + 
                           memory_match * 0.3f + 
                           history_score * 0.3f;
        
        // 估算执行时间
        selection.estimated_time = estimate_device_time(device, workload);
        
        // 计算置信度
        selection.confidence = calculate_selection_confidence(device, workload);
        
        if (total_score > best_score) {
            best_score = total_score;
            best_selection = selection;
        }
    }
    
    return best_selection;
}

float match_compute_intensity(
    struct device_info * device,
    struct workload_characteristics * workload
) {
    float optimal_intensity;
    
    if (device->type == DEVICE_GPU) {
        // GPU适合高计算强度
        optimal_intensity = 10.0f; // FLOPs/Byte
    } else {
        // CPU适合低计算强度
        optimal_intensity = 1.0f;
    }
    
    // 计算匹配度
    float diff = fabs(workload->compute_intensity - optimal_intensity);
    float match = exp(-diff / optimal_intensity);
    
    return match;
}

float match_memory_capacity(
    struct device_info * device,
    struct workload_characteristics * workload
) {
    size_t required_memory = workload->input_size + 
                            workload->output_size + 
                            workload->parameter_size;
    
    size_t available_memory;
    if (device->type == DEVICE_GPU) {
        available_memory = device->total_memory * 0.8f; // 保留20%余量
    } else {
        available_memory = get_system_memory() * 0.8f;
    }
    
    // 计算匹配度
    float ratio = (float)required_memory / available_memory;
    float match;
    
    if (ratio <= 0.5f) {
        match = 1.0f; // 充足
    } else if (ratio <= 0.8f) {
        match = 0.8f; // 适中
    } else if (ratio <= 1.0f) {
        match = 0.5f; // 紧张
    } else {
        match = 0.0f; // 不足
    }
    
    return match;
}

float get_historical_score(
    struct device_pool * pool,
    int device_id,
    struct workload_characteristics * workload
) {
    struct device_info * device = &pool->devices[device_id];
    
    if (device->history_size[device_id] == 0) {
        // 没有历史数据，使用设备综合得分
        return device->overall_score;
    }
    
    // 计算平均历史性能
    float avg_performance = 0.0f;
    for (int i = 0; i < device->history_size[device_id]; i++) {
        avg_performance += pool->performance_history[device_id][i];
    }
    avg_performance /= device->history_size[device_id];
    
    return avg_performance;
}
```

### 4.4 自适应学习

```cpp
// prima.cpp/src/adaptive_learning.c
struct adaptive_learner {
    struct device_pool * pool;
    
    // 学习参数
    float learning_rate;
    float exploration_rate;
    int update_interval;
    
    // 统计
    int total_selections[MAX_DEVICES];
    int successful_selections[MAX_DEVICES];
    float cumulative_reward[MAX_DEVICES];
};

void prima_update_learning(
    struct adaptive_learner * learner,
    int device_id,
    float actual_time,
    float estimated_time
) {
    // 1. 计算奖励
    float error = fabs(actual_time - estimated_time) / estimated_time;
    float reward = exp(-error); // 误差越小，奖励越高
    
    // 2. 更新统计
    learner->total_selections[device_id]++;
    learner->cumulative_reward[device_id] += reward;
    
    if (error < 0.2f) {
        learner->successful_selections[device_id]++;
    }
    
    // 3. 更新性能历史
    struct device_info * device = &learner->pool->devices[device_id];
    if (device->history_size[device_id] >= MAX_HISTORY) {
        // 移除最旧的记录
        memmove(&learner->pool->performance_history[device_id][0],
                &learner->pool->performance_history[device_id][1],
                (MAX_HISTORY - 1) * sizeof(float));
        device->history_size[device_id]--;
    }
    
    learner->pool->performance_history[device_id][device->history_size[device_id]++] = reward;
    
    // 4. 定期更新设备得分
    if (learner->total_selections[device_id] % learner->update_interval == 0) {
        update_device_score(learner, device_id);
    }
}

void update_device_score(struct adaptive_learner * learner, int device_id) {
    struct device_info * device = &learner->pool->devices[device_id];
    
    // 计算平均奖励
    float avg_reward = learner->cumulative_reward[device_id] / 
                       learner->total_selections[device_id];
    
    // 计算成功率
    float success_rate = (float)learner->successful_selections[device_id] / 
                         learner->total_selections[device_id];
    
    // 更新设备得分
    float new_score = 0.7f * device->overall_score + 
                     0.3f * (avg_reward * success_rate);
    
    device->overall_score = new_score;
}

// 探索-利用策略
int prima_select_with_exploration(
    struct adaptive_learner * learner,
    struct workload_characteristics * workload
) {
    // epsilon-greedy策略
    float random = (float)rand() / RAND_MAX;
    
    if (random < learner->exploration_rate) {
        // 探索：随机选择设备
        return rand() % learner->pool->num_devices;
    } else {
        // 利用：选择最优设备
        struct device_selection selection = prima_select_device(
            learner->pool, workload
        );
        return selection.device_id;
    }
}
```

---

## 5. 内存压力控制

### 5.1 核心概念

内存压力控制通过监控和管理内存使用，防止内存溢出和性能下降：

- **内存监控**：实时监控内存使用情况
- **压力检测**：检测内存压力状态
- **自动调节**：根据压力自动调整策略
- **预防措施**：提前采取预防措施

### 5.2 内存监控系统

```cpp
// prima.cpp/src/memory_monitor.c
struct memory_stats {
    // GPU内存
    size_t gpu_total;
    size_t gpu_used;
    size_t gpu_free;
    float gpu_usage_percent;
    
    // CPU内存
    size_t cpu_total;
    size_t cpu_used;
    size_t cpu_free;
    float cpu_usage_percent;
    
    // 分配统计
    int num_allocations;
    size_t total_allocated;
    size_t peak_allocated;
    
    // 碎片化
    float fragmentation_ratio;
};

struct memory_pressure_state {
    enum pressure_level {
        PRESSURE_NONE,                     // 无压力
        PRESSURE_LOW,                      // 低压力
        PRESSURE_MEDIUM,                   // 中等压力
        PRESSURE_HIGH,                     // 高压力
        PRESSURE_CRITICAL                  // 严重压力
    } level;
    
    float pressure_score;                  // 压力分数 (0-1)
    time_t last_update;
    
    // 趋势
    float usage_trend;                     // 使用趋势
    float allocation_rate;                 // 分配速率
};

struct memory_monitor {
    struct memory_stats stats;
    struct memory_pressure_state pressure;
    
    // 配置
    float warning_threshold;              // 警告阈值
    float critical_threshold;              // 严重阈值
    int monitor_interval;                 // 监控间隔
    
    // 回调
    void (*pressure_callback)(enum pressure_level);
};

void prima_update_memory_stats(struct memory_monitor * monitor) {
    // 1. 更新GPU内存统计
    size_t gpu_free, gpu_total;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    
    monitor->stats.gpu_total = gpu_total;
    monitor->stats.gpu_free = gpu_free;
    monitor->stats.gpu_used = gpu_total - gpu_free;
    monitor->stats.gpu_usage_percent = (float)monitor->stats.gpu_used / gpu_total;
    
    // 2. 更新CPU内存统计
    monitor->stats.cpu_total = get_system_memory();
    monitor->stats.cpu_used = get_used_memory();
    monitor->stats.cpu_free = monitor->stats.cpu_total - monitor->stats.cpu_used;
    monitor->stats.cpu_usage_percent = (float)monitor->stats.cpu_used / monitor->stats.cpu_total;
    
    // 3. 更新分配统计
    monitor->stats.num_allocations = get_num_allocations();
    monitor->stats.total_allocated = get_total_allocated();
    monitor->stats.peak_allocated = max(monitor->stats.peak_allocated,
                                         monitor->stats.total_allocated);
    
    // 4. 计算碎片化
    monitor->stats.fragmentation_ratio = calculate_fragmentation();
    
    // 5. 更新压力状态
    update_pressure_state(monitor);
}

void update_pressure_state(struct memory_monitor * monitor) {
    // 计算压力分数
    float gpu_pressure = monitor->stats.gpu_usage_percent;
    float cpu_pressure = monitor->stats.cpu_usage_percent;
    float fragmentation_penalty = monitor->stats.fragmentation_ratio * 0.2f;
    
    monitor->pressure.pressure_score = max(gpu_pressure, cpu_pressure) + 
                                      fragmentation_penalty;
    
    // 确定压力级别
    if (monitor->pressure.pressure_score < monitor->warning_threshold) {
        monitor->pressure.level = PRESSURE_NONE;
    } else if (monitor->pressure.pressure_score < monitor->critical_threshold * 0.7f) {
        monitor->pressure.level = PRESSURE_LOW;
    } else if (monitor->pressure.pressure_score < monitor->critical_threshold * 0.9f) {
        monitor->pressure.level = PRESSURE_MEDIUM;
    } else if (monitor->pressure.pressure_score < monitor->critical_threshold) {
        monitor->pressure.level = PRESSURE_HIGH;
    } else {
        monitor->pressure.level = PRESSURE_CRITICAL;
    }
    
    // 计算趋势
    time_t current_time;
    time(&current_time);
    float time_diff = current_time - monitor->pressure.last_update;
    
    if (time_diff > 0) {
        float usage_diff = monitor->stats.gpu_usage_percent - 
                          monitor->pressure.usage_trend;
        monitor->pressure.usage_trend = usage_diff / time_diff;
    }
    
    monitor->pressure.last_update = current_time;
    
    // 触发回调
    if (monitor->pressure_callback) {
        monitor->pressure_callback(monitor->pressure.level);
    }
}
```

### 5.3 自动压力缓解

```cpp
// prima.cpp/src/pressure_relief.c
struct pressure_relief_strategy {
    enum relief_action {
        RELIEF_NONE,                      // 无操作
        RELIEF_CACHE_CLEAR,               // 清空缓存
        RELIEF_OFFLOAD_TO_CPU,            // 卸载到CPU
        RELIEF_REDUCE_BATCH,              // 减少批次
        RELIEF_QUANTIZE,                  // 量化
        RELIEF_EVICT_LAYERS               // 驱逐层
    } action;
    
    float priority;                       // 优先级
    float estimated_savings;              // 预计节省
    float performance_impact;             // 性能影响
};

void prima_relieve_pressure(
    struct memory_monitor * monitor,
    struct offload_config * config
) {
    enum pressure_level level = monitor->pressure.level;
    
    if (level == PRESSURE_NONE) {
        return;
    }
    
    // 1. 生成缓解策略
    struct pressure_relief_strategy strategies[MAX_STRATEGIES];
    int num_strategies = generate_relief_strategies(
        monitor, config, strategies
    );
    
    // 2. 选择最优策略
    struct pressure_relief_strategy best_strategy = select_best_strategy(
        strategies, num_strategies, level
    );
    
    // 3. 执行策略
    execute_relief_strategy(&best_strategy, config);
    
    // 4. 验证效果
    prima_update_memory_stats(monitor);
    
    if (monitor->pressure.level >= level) {
        // 策略无效，尝试下一个
        log_warning("Relief strategy ineffective, trying next strategy");
        // ... 递归尝试其他策略
    }
}

int generate_relief_strategies(
    struct memory_monitor * monitor,
    struct offload_config * config,
    struct pressure_relief_strategy * strategies
) {
    int num_strategies = 0;
    
    // 策略1: 清空KV Cache
    strategies[num_strategies++] = (struct pressure_relief_strategy){
        .action = RELIEF_CACHE_CLEAR,
        .priority = 0.9f,
        .estimated_savings = estimate_kv_cache_size(),
        .performance_impact = 0.3f
    };
    
    // 策略2: 卸载层到CPU
    strategies[num_strategies++] = (struct pressure_relief_strategy){
        .action = RELIEF_OFFLOAD_TO_CPU,
        .priority = 0.8f,
        .estimated_savings = estimate_layer_offload_savings(config),
        .performance_impact = 0.5f
    };
    
    // 策略3: 减少批次大小
    strategies[num_strategies++] = (struct pressure_relief_strategy){
        .action = RELIEF_REDUCE_BATCH,
        .priority = 0.7f,
        .estimated_savings = estimate_batch_reduction_savings(config),
        .performance_impact = 0.4f
    };
    
    // 策略4: 量化模型
    strategies[num_strategies++] = (struct pressure_relief_strategy){
        .action = RELIEF_QUANTIZE,
        .priority = 0.6f,
        .estimated_savings = estimate_quantization_savings(),
        .performance_impact = 0.2f
    };
    
    // 策略5: 驱逐不活跃层
    strategies[num_strategies++] = (struct pressure_relief_strategy){
        .action = RELIEF_EVICT_LAYERS,
        .priority = 0.5f,
        .estimated_savings = estimate_eviction_savings(),
        .performance_impact = 0.6f
    };
    
    return num_strategies;
}

struct pressure_relief_strategy select_best_strategy(
    struct pressure_relief_strategy * strategies,
    int num_strategies,
    enum pressure_level level
) {
    struct pressure_relief_strategy best_strategy;
    float best_score = -INFINITY;
    
    for (int i = 0; i < num_strategies; i++) {
        // 计算策略得分
        float score = strategies[i].priority * strategies[i].estimated_savings / 
                     (strategies[i].performance_impact + 0.1f);
        
        // 根据压力级别调整
        if (level >= PRESSURE_CRITICAL) {
            // 严重压力：优先考虑节省
            score *= 1.5f;
        } else if (level >= PRESSURE_HIGH) {
            // 高压力：平衡节省和性能
            score *= 1.2f;
        }
        
        if (score > best_score) {
            best_score = score;
            best_strategy = strategies[i];
        }
    }
    
    return best_strategy;
}

void execute_relief_strategy(
    struct pressure_relief_strategy * strategy,
    struct offload_config * config
) {
    switch (strategy->action) {
        case RELIEF_CACHE_CLEAR:
            clear_kv_cache();
            log_info("Cleared KV cache");
            break;
            
        case RELIEF_OFFLOAD_TO_CPU:
            offload_layers_to_cpu(config);
            log_info("Offloaded layers to CPU");
            break;
            
        case RELIEF_REDUCE_BATCH:
            reduce_batch_size(config);
            log_info("Reduced batch size");
            break;
            
        case RELIEF_QUANTIZE:
            quantize_model();
            log_info("Quantized model");
            break;
            
        case RELIEF_EVICT_LAYERS:
            evict_inactive_layers();
            log_info("Evicted inactive layers");
            break;
            
        default:
            break;
    }
}
```

### 5.4 预防性内存管理

```cpp
// prima.cpp/src/preventive_memory.c
struct preventive_memory_manager {
    struct memory_monitor * monitor;
    
    // 预测模型
    float usage_history[MAX_HISTORY];
    int history_size;
    
    // 预测参数
    int prediction_window;                // 预测窗口
    float warning_margin;                 // 警告余量
    
    // 预防措施
    bool auto_preemptive;                 // 自动预防
    int preemption_threshold;             // 预防阈值
};

void prima_preemptive_memory_management(
    struct preventive_memory_manager * manager
) {
    // 1. 预测未来内存使用
    float predicted_usage = predict_memory_usage(manager);
    
    // 2. 检查是否需要预防措施
    float current_usage = manager->monitor->stats.gpu_usage_percent;
    float projected_usage = current_usage + predicted_usage;
    
    if (projected_usage > manager->preemption_threshold) {
        // 需要采取预防措施
        log_warning("Predicted memory usage: %.2f%%, taking preemptive action",
                   projected_usage * 100);
        
        take_preemptive_actions(manager, projected_usage);
    }
}

float predict_memory_usage(
    struct preventive_memory_manager * manager
) {
    if (manager->history_size < 3) {
        return 0.0f; // 数据不足
    }
    
    // 简单线性回归
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
    
    for (int i = 0; i < manager->history_size; i++) {
        float x = i;
        float y = manager->usage_history[i];
        
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }
    
    float n = manager->history_size;
    float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    float intercept = (sum_y - slope * sum_x) / n;
    
    // 预测下一个时间点
    float predicted = slope * manager->history_size + intercept;
    
    return predicted;
}

void take_preemptive_actions(
    struct preventive_memory_manager * manager,
    float projected_usage
) {
    // 根据预测的内存使用，采取相应的预防措施
    
    if (projected_usage > 0.95f) {
        // 严重预测：立即采取措施
        emergency_memory_relief();
    } else if (projected_usage > 0.85f) {
        // 高预测：采取积极措施
        aggressive_memory_relief();
    } else if (projected_usage > 0.75f) {
        // 中等预测：采取温和措施
        moderate_memory_relief();
    } else {
        // 低预测：采取预防措施
        light_memory_relief();
    }
}

void emergency_memory_relief() {
    // 立即释放所有可释放的内存
    clear_all_caches();
    offload_all_possible_layers();
    reduce_batch_to_minimum();
    force_garbage_collection();
}

void aggressive_memory_relief() {
    // 释放大部分内存
    clear_kv_cache();
    offload_multiple_layers(3);
    reduce_batch_size(0.5f);
}

void moderate_memory_relief() {
    // 释放部分内存
    clear_old_cache_entries();
    offload_one_layer();
    reduce_batch_size(0.8f);
}

void light_memory_relief() {
    // 轻微释放内存
    defragment_memory();
    optimize_allocations();
}
```

---

## 6. 投机解码

### 6.1 核心概念

投机解码是一种通过并行生成多个候选token，然后验证正确性的加速技术：

- **候选生成**：小模型快速生成多个候选
- **并行验证**：大模型并行验证候选
- **接受机制**：接受正确的候选序列
- **回退机制**：验证失败时回退

### 6.2 投机解码架构

```cpp
// prima.cpp/src/speculative_decoding.h
struct spec_decode_config {
    // 小模型配置
    struct llama_model * draft_model;
    int draft_tokens_per_step;           // 每步生成的候选数
    
    // 大模型配置
    struct llama_model * target_model;
    
    // 验证配置
    int max_speculation_steps;            // 最大投机步数
    float acceptance_threshold;          // 接受阈值
    bool early_termination;              // 早期终止
    
    // 性能配置
    int parallel_verify_batch;           // 并行验证批次
    bool use_tree_attention;             // 使用树形注意力
};

struct spec_decode_state {
    // 当前状态
    llama_token current_token;
    llama_token context[MAX_CONTEXT];
    int context_length;
    
    // 候选序列
    llama_token draft_tokens[MAX_SPEC_STEPS][MAX_CANDIDATES];
    int draft_token_counts[MAX_SPEC_STEPS];
    float draft_probs[MAX_SPEC_STEPS][MAX_CANDIDATES];
    
    // 验证结果
    bool accepted[MAX_SPEC_STEPS];
    int accepted_count;
    int first_rejected;
    
    // 统计
    int total_steps;
    int total_accepted;
    float acceptance_rate;
};
```

### 6.3 候选生成

```cpp
// prima.cpp/src/draft_generator.c
void prima_generate_draft_tokens(
    struct spec_decode_config * config,
    struct spec_decode_state * state
) {
    struct llama_model * draft_model = config->draft_model;
    
    for (int step = 0; step < config->max_speculation_steps; step++) {
        // 1. 使用小模型生成候选
        state->draft_token_counts[step] = 0;
        
        // 生成多个候选
        for (int candidate = 0; candidate < config->draft_tokens_per_step; candidate++) {
            // 获取下一个token的概率分布
            float * logits = llama_get_logits(draft_model);
            
            // 采样top-k候选
            llama_token sampled_token = sample_top_k(
                logits, draft_model->vocab_size, TOP_K
            );
            
            // 记录候选
            state->draft_tokens[step][candidate] = sampled_token;
            state->draft_probs[step][candidate] = logits[sampled_token];
            state->draft_token_counts[step]++;
            
            // 更新上下文
            state->context[state->context_length++] = sampled_token;
        }
        
        // 2. 检查是否应该停止
        if (should_stop_drafting(state, step)) {
            break;
        }
    }
}

llama_token sample_top_k(float * logits, int vocab_size, int k) {
    // 找到top-k token
    struct token_score {
        llama_token token;
        float score;
    } top_k[TOP_K];
    
    // 初始化
    for (int i = 0; i < k; i++) {
        top_k[i].token = i;
        top_k[i].score = logits[i];
    }
    
    // 找到top-k
    for (int i = k; i < vocab_size; i++) {
        if (logits[i] > top_k[k-1].score) {
            // 插入排序
            int j = k - 1;
            while (j > 0 && logits[i] > top_k[j-1].score) {
                top_k[j] = top_k[j-1];
                j--;
            }
            top_k[j].token = i;
            top_k[j].score = logits[i];
        }
    }
    
    // 从top-k中采样
    float sum_exp = 0.0f;
    for (int i = 0; i < k; i++) {
        top_k[i].score = exp(top_k[i].score);
        sum_exp += top_k[i].score;
    }
    
    float r = (float)rand() / RAND_MAX * sum_exp;
    float cumulative = 0.0f;
    
    for (int i = 0; i < k; i++) {
        cumulative += top_k[i].score;
        if (r <= cumulative) {
            return top_k[i].token;
        }
    }
    
    return top_k[k-1].token;
}

bool should_stop_drafting(
    struct spec_decode_state * state,
    int step
) {
    // 检查特殊token
    for (int i = 0; i < state->draft_token_counts[step]; i++) {
        llama_token token = state->draft_tokens[step][i];
        if (is_special_token(token)) {
            return true;
        }
    }
    
    // 检查上下文长度
    if (state->context_length >= MAX_CONTEXT) {
        return true;
    }
    
    // 检查概率阈值
    float max_prob = 0.0f;
    for (int i = 0; i < state->draft_token_counts[step]; i++) {
        max_prob = max(max_prob, state->draft_probs[step][i]);
    }
    
    if (max_prob < PROB_THRESHOLD) {
        return true;
    }
    
    return false;
}
```

### 6.4 并行验证

```cpp
// prima.cpp/src/parallel_verifier.c
void prima_verify_draft_tokens(
    struct spec_decode_config * config,
    struct spec_decode_state * state
) {
    struct llama_model * target_model = config->target_model;
    
    // 1. 构建验证批次
    llama_batch verify_batch = build_verify_batch(state, config);
    
    // 2. 并行验证所有候选
    llama_decode(target_model, verify_batch);
    
    // 3. 获取验证结果
    float * target_logits = llama_get_logits(target_model);
    
    // 4. 逐个验证候选
    state->accepted_count = 0;
    state->first_rejected = -1;
    
    for (int step = 0; step < config->max_speculation_steps; step++) {
        bool step_accepted = false;
        
        for (int candidate = 0; candidate < state->draft_token_counts[step]; candidate++) {
            llama_token draft_token = state->draft_tokens[step][candidate];
            float draft_prob = state->draft_probs[step][candidate];
            
            // 获取目标模型对该token的概率
            float target_prob = target_logits[draft_token];
            
            // 比较概率
            if (target_prob >= draft_prob * config->acceptance_threshold) {
                // 接受该候选
                state->accepted[step] = true;
                state->accepted_count++;
                step_accepted = true;
                
                // 更新上下文
                state->context[state->context_length++] = draft_token;
                
                break;
            }
        }
        
        if (!step_accepted) {
            // 该步骤没有候选被接受
            state->first_rejected = step;
            break;
        }
    }
    
    // 5. 更新统计
    state->total_steps++;
    state->total_accepted += state->accepted_count;
    state->acceptance_rate = (float)state->total_accepted / 
                            (state->total_steps * config->draft_tokens_per_step);
}

llama_batch build_verify_batch(
    struct spec_decode_state * state,
    struct spec_decode_config * config
) {
    llama_batch batch;
    batch.n_tokens = 0;
    
    // 添加原始上下文
    for (int i = 0; i < state->context_length; i++) {
        batch.token[batch.n_tokens] = state->context[i];
        batch.n_tokens++;
    }
    
    // 添加所有候选token（树形结构）
    for (int step = 0; step < config->max_speculation_steps; step++) {
        for (int candidate = 0; candidate < state->draft_token_counts[step]; candidate++) {
            batch.token[batch.n_tokens] = state->draft_tokens[step][candidate];
            
            // 设置父节点（用于树形注意力）
            if (step == 0) {
                batch.pos[batch.n_tokens] = state->context_length;
            } else {
                batch.pos[batch.n_tokens] = state->context_length + step;
            }
            
            batch.n_tokens++;
        }
    }
    
    return batch;
}
```

### 6.5 接受和回退机制

```cpp
// prima.cpp/src/acceptance_handler.c
void prima_handle_verification_result(
    struct spec_decode_config * config,
    struct spec_decode_state * state
) {
    if (state->first_rejected == -1) {
        // 所有候选都被接受
        handle_all_accepted(state);
    } else {
        // 有候选被拒绝
        handle_partial_rejection(config, state);
    }
}

void handle_all_accepted(struct spec_decode_state * state) {
    // 所有候选都被接受，继续投机解码
    log_debug("All %d draft tokens accepted", state->accepted_count);
    
    // 更新当前token
    state->current_token = state->context[state->context_length - 1];
}

void handle_partial_rejection(
    struct spec_decode_config * config,
    struct spec_decode_state * state
) {
    int first_rejected = state->first_rejected;
    
    // 1. 回退到第一个被拒绝的位置
    state->context_length -= (config->max_speculation_steps - first_rejected);
    
    // 2. 使用目标模型重新生成
    llama_token correct_token = generate_with_target_model(
        config->target_model,
        state->context,
        state->context_length
    );
    
    // 3. 更新上下文
    state->context[state->context_length++] = correct_token;
    state->current_token = correct_token;
    
    log_debug("Rejected at step %d, regenerated token: %d",
             first_rejected, correct_token);
}

llama_token generate_with_target_model(
    struct llama_model * model,
    llama_token * context,
    int context_length
) {
    // 构建批次
    llama_batch batch;
    batch.n_tokens = context_length;
    
    for (int i = 0; i < context_length; i++) {
        batch.token[i] = context[i];
        batch.pos[i] = i;
    }
    
    // 执行推理
    llama_decode(model, batch);
    
    // 获取logits并采样
    float * logits = llama_get_logits(model);
    llama_token token = sample_token(logits, model->vocab_size);
    
    return token;
}

llama_token sample_token(float * logits, int vocab_size) {
    // 使用温度采样
    float temperature = 0.8f;
    
    // 应用温度
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }
    
    // Softmax
    float max_logit = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        max_logit = max(max_logit, logits[i]);
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = exp(logits[i] - max_logit);
        sum_exp += logits[i];
    }
    
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= sum_exp;
    }
    
    // 采样
    float r = (float)rand() / RAND_MAX;
    float cumulative = 0.0f;
    
    for (int i = 0; i < vocab_size; i++) {
        cumulative += logits[i];
        if (r <= cumulative) {
            return i;
        }
    }
    
    return vocab_size - 1;
}
```

### 6.6 性能优化

```cpp
// prima.cpp/src/spec_decode_optimization.c
void prima_optimized_speculative_decode(
    struct spec_decode_config * config,
    llama_token * output,
    int max_tokens
) {
    struct spec_decode_state state;
    init_spec_decode_state(&state);
    
    int output_pos = 0;
    
    while (output_pos < max_tokens) {
        // 1. 生成候选
        auto_timer_start("draft_generation");
        prima_generate_draft_tokens(config, &state);
        auto_timer_stop("draft_generation");
        
        // 2. 并行验证
        auto_timer_start("parallel_verification");
        prima_verify_draft_tokens(config, &state);
        auto_timer_stop("parallel_verification");
        
        // 3. 处理验证结果
        prima_handle_verification_result(config, &state);
        
        // 4. 输出接受的token
        for (int i = 0; i < state.accepted_count; i++) {
            output[output_pos++] = state.context[state.context_length - state.accepted_count + i];
            
            if (output_pos >= max_tokens) {
                break;
            }
        }
        
        // 5. 检查是否应该停止
        if (should_stop_generation(&state)) {
            break;
        }
        
        // 6. 动态调整参数
        adjust_speculation_parameters(config, &state);
    }
    
    // 输出统计
    log_info("Speculative decoding stats:");
    log_info("  Total steps: %d", state.total_steps);
    log_info("  Total accepted: %d", state.total_accepted);
    log_info("  Acceptance rate: %.2f%%", state.acceptance_rate * 100);
}

void adjust_speculation_parameters(
    struct spec_decode_config * config,
    struct spec_decode_state * state
) {
    // 根据接受率动态调整参数
    
    if (state.acceptance_rate > 0.8f) {
        // 高接受率，增加投机步数
        config->max_speculation_steps = min(
            config->max_speculation_steps + 1,
            MAX_SPEC_STEPS
        );
    } else if (state.acceptance_rate < 0.5f) {
        // 低接受率，减少投机步数
        config->max_speculation_steps = max(
            config->max_speculation_steps - 1,
            MIN_SPEC_STEPS
        );
    }
    
    // 调整候选数量
    if (state.acceptance_rate > 0.7f) {
        config->draft_tokens_per_step = min(
            config->draft_tokens_per_step + 1,
            MAX_CANDIDATES
        );
    } else if (state.acceptance_rate < 0.4f) {
        config->draft_tokens_per_step = max(
            config->draft_tokens_per_step - 1,
            MIN_CANDIDATES
        );
    }
}
```

---

## 7. 动态批处理

### 7.1 核心概念

动态批处理根据请求的到达时间和资源可用性，动态地组合请求进行批处理：

- **请求队列**：管理待处理的请求
- **批次构建**：智能组合请求形成批次
- **动态调度**：根据资源情况调整批次大小
- **优先级管理**：处理不同优先级的请求

### 7.2 请求队列管理

```cpp
// prima.cpp/src/dynamic_batch.h
struct inference_request {
    int request_id;
    int priority;                        // 优先级
    time_t arrival_time;                 // 到达时间
    time_t deadline;                     // 截止时间
    
    // 请求内容
    llama_token * input_tokens;
    int input_length;
    int max_output_tokens;
    
    // 状态
    enum request_status {
        STATUS_PENDING,                   // 等待中
        STATUS_BATCHED,                  // 已批处理
        STATUS_PROCESSING,                // 处理中
        STATUS_COMPLETED,                 // 已完成
        STATUS_FAILED                    // 失败
    } status;
    
    // 结果
    llama_token * output_tokens;
    int output_length;
    float latency;
};

struct request_queue {
    struct inference_request requests[MAX_REQUESTS];
    int num_requests;
    
    // 优先级队列
    struct inference_request * priority_queue[MAX_REQUESTS];
    int queue_size;
    
    // 统计
    int total_requests;
    int completed_requests;
    float avg_latency;
    float max_latency;
};
```

### 7.3 批次构建策略

```cpp
// prima.cpp/src/batch_builder.c
struct batch_builder {
    struct request_queue * queue;
    
    // 批次配置
    int max_batch_size;                  // 最大批次大小
    int max_sequence_length;             // 最大序列长度
    float max_batch_latency;             // 最大批次延迟
    
    // 构建策略
    enum build_strategy {
        STRATEGY_FIFO,                   // 先进先出
        STRATEGY_PRIORITY,               // 优先级
        STRATEGY_DEADLINE,               // 截止时间
        STRATEGY_ADAPTIVE                // 自适应
    } strategy;
    
    // 当前批次
    struct inference_request * current_batch[MAX_BATCH_SIZE];
    int batch_size;
    time_t batch_start_time;
};

struct inference_request ** prima_build_batch(
    struct batch_builder * builder
) {
    // 1. 清空当前批次
    builder->batch_size = 0;
    time(&builder->batch_start_time);
    
    // 2. 根据策略选择请求
    switch (builder->strategy) {
        case STRATEGY_FIFO:
            build_fifo_batch(builder);
            break;
        case STRATEGY_PRIORITY:
            build_priority_batch(builder);
            break;
        case STRATEGY_DEADLINE:
            build_deadline_batch(builder);
            break;
        case STRATEGY_ADAPTIVE:
            build_adaptive_batch(builder);
            break;
    }
    
    // 3. 验证批次
    validate_batch(builder);
    
    return builder->current_batch;
}

void build_adaptive_batch(struct batch_builder * builder) {
    struct request_queue * queue = builder->queue;
    
    // 1. 计算当前系统状态
    float system_load = calculate_system_load();
    float avg_latency = queue->avg_latency;
    int pending_requests = count_pending_requests(queue);
    
    // 2. 动态调整批次大小
    int target_batch_size;
    if (system_load > 0.8f) {
        // 高负载：小批次
        target_batch_size = builder->max_batch_size * 0.5f;
    } else if (system_load > 0.5f) {
        // 中等负载：中等批次
        target_batch_size = builder->max_batch_size * 0.75f;
    } else {
        // 低负载：大批次
        target_batch_size = builder->max_batch_size;
    }
    
    // 3. 选择请求
    int selected = 0;
    for (int i = 0; i < queue->num_requests && selected < target_batch_size; i++) {
        struct inference_request * request = &queue->requests[i];
        
        if (request->status == STATUS_PENDING) {
            // 检查约束
            if (can_add_to_batch(builder, request)) {
                builder->current_batch[builder->batch_size++] = request;
                request->status = STATUS_BATCHED;
                selected++;
            }
        }
    }
    
    // 4. 考虑截止时间
    consider_deadlines(builder);
}

bool can_add_to_batch(
    struct batch_builder * builder,
    struct inference_request * request
) {
    // 检查批次大小
    if (builder->batch_size >= builder->max_batch_size) {
        return false;
    }
    
    // 检查序列长度
    if (request->input_length > builder->max_sequence_length) {
        return false;
    }
    
    // 检查批次延迟
    time_t current_time;
    time(&current_time);
    float batch_age = difftime(current_time, builder->batch_start_time);
    
    if (batch_age > builder->max_batch_latency) {
        return false;
    }
    
    // 检查内存
    if (!check_memory_available(builder, request)) {
        return false;
    }
    
    return true;
}

void consider_deadlines(struct batch_builder * builder) {
    time_t current_time;
    time(&current_time);
    
    // 检查是否有紧急请求
    for (int i = 0; i < builder->queue->num_requests; i++) {
        struct inference_request * request = &builder->queue->requests[i];
        
        if (request->status == STATUS_PENDING) {
            float time_to_deadline = difftime(request->deadline, current_time);
            
            if (time_to_deadline < URGENT_THRESHOLD) {
                // 紧急请求，优先处理
                if (builder->batch_size < builder->max_batch_size) {
                    builder->current_batch[builder->batch_size++] = request;
                    request->status = STATUS_BATCHED;
                }
            }
        }
    }
}
```

### 7.4 动态调度

```cpp
// prima.cpp/src/dynamic_scheduler.c
struct dynamic_scheduler {
    struct batch_builder * builder;
    
    // 调度配置
    int scheduling_interval;              // 调度间隔
    float max_queue_wait_time;            // 最大队列等待时间
    
    // 性能监控
    float throughput;                     // 吞吐量
    float avg_batch_time;                 // 平均批次时间
    float resource_utilization;          // 资源利用率
    
    // 自适应参数
    float target_latency;                 // 目标延迟
    float target_throughput;              // 目标吞吐量
};

void prima_dynamic_schedule(struct dynamic_scheduler * scheduler) {
    while (true) {
        // 1. 构建批次
        struct inference_request ** batch = prima_build_batch(scheduler->builder);
        
        if (scheduler->builder->batch_size == 0) {
            // 没有请求，等待
            usleep(scheduler->scheduling_interval * 1000);
            continue;
        }
        
        // 2. 执行批次
        auto_timer_start("batch_execution");
        execute_batch(batch, scheduler->builder->batch_size);
        auto_timer_stop("batch_execution");
        
        // 3. 更新统计
        update_scheduler_stats(scheduler);
        
        // 4. 自适应调整
        adapt_scheduler_parameters(scheduler);
        
        // 5. 检查是否需要重新调度
        if (should_reschedule(scheduler)) {
            reschedule_pending_requests(scheduler);
        }
    }
}

void execute_batch(
    struct inference_request ** batch,
    int batch_size
) {
    // 1. 构建llama_batch
    llama_batch llama_batch;
    build_llama_batch(&llama_batch, batch, batch_size);
    
    // 2. 执行推理
    llama_decode(model, llama_batch);
    
    // 3. 分发结果
    distribute_results(batch, batch_size, &llama_batch);
}

void build_llama_batch(
    llama_batch * batch,
    struct inference_request ** requests,
    int num_requests
) {
    batch->n_tokens = 0;
    
    // 填充批次
    for (int i = 0; i < num_requests; i++) {
        struct inference_request * request = requests[i];
        
        for (int j = 0; j < request->input_length; j++) {
            batch->token[batch->n_tokens] = request->input_tokens[j];
            batch->pos[batch->n_tokens] = j;
            batch->n_tokens++;
        }
    }
}

void distribute_results(
    struct inference_request ** batch,
    int batch_size,
    llama_batch * llama_batch
) {
    int token_offset = 0;
    
    for (int i = 0; i < batch_size; i++) {
        struct inference_request * request = batch[i];
        
        // 复制输出token
        int output_length = min(request->max_output_tokens,
                               llama_batch->n_tokens - token_offset);
        
        for (int j = 0; j < output_length; j++) {
            request->output_tokens[j] = llama_batch->token[token_offset + j];
        }
        
        request->output_length = output_length;
        request->status = STATUS_COMPLETED;
        
        // 计算延迟
        time_t completion_time;
        time(&completion_time);
        request->latency = difftime(completion_time, request->arrival_time);
        
        token_offset += request->input_length;
    }
}
```

### 7.5 性能监控和优化

```cpp
// prima.cpp/src/batch_monitor.c
struct batch_monitor {
    // 性能指标
    float batch_throughput;               // 批次吞吐量
    float token_throughput;               // token吞吐量
    float avg_batch_latency;              // 平均批次延迟
    float p99_batch_latency;              // P99批次延迟
    
    // 资源使用
    float gpu_utilization;                // GPU利用率
    float memory_usage;                   // 内存使用率
    
    // 队列状态
    int queue_length;                     // 队列长度
    float avg_wait_time;                  // 平均等待时间
    
    // 历史数据
    float throughput_history[MAX_HISTORY];
    float latency_history[MAX_HISTORY];
    int history_size;
};

void prima_monitor_batch_performance(
    struct batch_monitor * monitor,
    struct dynamic_scheduler * scheduler
) {
    // 1. 收集性能指标
    monitor->batch_throughput = calculate_batch_throughput(scheduler);
    monitor->token_throughput = calculate_token_throughput(scheduler);
    monitor->avg_batch_latency = calculate_avg_latency(scheduler);
    monitor->p99_batch_latency = calculate_p99_latency(scheduler);
    
    // 2. 收集资源使用
    monitor->gpu_utilization = get_gpu_utilization();
    monitor->memory_usage = get_memory_usage();
    
    // 3. 收集队列状态
    monitor->queue_length = count_pending_requests(scheduler->builder->queue);
    monitor->avg_wait_time = calculate_avg_wait_time(scheduler->builder->queue);
    
    // 4. 更新历史数据
    update_history(monitor);
    
    // 5. 检测异常
    detect_anomalies(monitor);
    
    // 6. 生成报告
    generate_performance_report(monitor);
}

void detect_anomalies(struct batch_monitor * monitor) {
    // 检测吞吐量下降
    if (monitor->history_size >= 10) {
        float recent_avg = calculate_recent_average(monitor->throughput_history, 5);
        float historical_avg = calculate_historical_average(monitor->throughput_history, 10);
        
        if (recent_avg < historical_avg * 0.8f) {
            log_warning("Throughput degradation detected: %.2f -> %.2f",
                       historical_avg, recent_avg);
            
            // 触发优化
            trigger_throughput_optimization();
        }
    }
    
    // 检测延迟增加
    if (monitor->history_size >= 10) {
        float recent_avg = calculate_recent_average(monitor->latency_history, 5);
        float historical_avg = calculate_historical_average(monitor->latency_history, 10);
        
        if (recent_avg > historical_avg * 1.2f) {
            log_warning("Latency increase detected: %.2f -> %.2f",
                       historical_avg, recent_avg);
            
            // 触发优化
            trigger_latency_optimization();
        }
    }
    
    // 检测资源瓶颈
    if (monitor->gpu_utilization > 0.95f) {
        log_warning("GPU utilization near capacity: %.2f%%",
                   monitor->gpu_utilization * 100);
        
        trigger_resource_optimization();
    }
    
    if (monitor->memory_usage > 0.9f) {
        log_warning("Memory usage near capacity: %.2f%%",
                   monitor->memory_usage * 100);
        
        trigger_memory_optimization();
    }
}
```

---

## 总结

Prima.cpp的优化技术在llama.cpp基础上，针对多GPU、异构计算和大规模推理场景进行了深度优化：

1. **Piped-Ring并行**：结合流水线并行和环形通信，最大化多GPU利用率
2. **异构工作负载分配**：智能分配任务到不同设备，实现负载均衡
3. **GPU/CPU混合卸载**：根据计算特性和资源可用性动态卸载
4. **自动设备选择**：基于性能预测和历史数据自动选择最优设备
5. **内存压力控制**：实时监控内存使用，自动缓解压力
6. **投机解码**：通过候选生成和并行验证加速推理
7. **动态批处理**：智能组合请求，平衡吞吐量和延迟

这些优化技术相互配合，使prima.cpp能够在各种硬件配置下实现高性能推理。

## 能力目标

学习完本章节后，你能够：

**能做什么：**
- 理解prima.cpp的优化技术原理和实现
- 分析多GPU场景下的性能瓶颈
- 设计和实现异构计算系统
- 优化大规模推理系统的性能
- 调试和解决多设备协同问题

**还不能做什么：**
- 从零开始设计全新的并行算法
- 深度优化底层硬件通信协议
- 处理极端大规模（100+ GPU）的部署场景

**实际工作示例：**
- 理解Piped-Ring并行的实现细节
- 分析和优化GPU/CPU混合卸载策略
- 调试投机解码的接受率问题
- 优化动态批处理的吞吐量和延迟平衡
