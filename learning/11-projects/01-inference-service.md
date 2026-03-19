# 项目 1：构建推理服务

## 项目概述

构建一个生产级的 LLM 推理服务，支持 HTTP API、并发请求处理和性能监控。

## 项目目标

完成本项目后，你将能够：

- 设计和实现推理服务 API
- 处理并发请求和批处理
- 实现性能监控和日志
- 部署和运维推理服务

## 需求分析

### 功能需求

1. **API 接口**
   - POST `/v1/completions` - 文本生成
   - POST `/v1/chat/completions` - 对话生成
   - GET `/health` - 健康检查

2. **请求处理**
   - 支持并发请求
   - 动态批处理（Dynamic Batching）
   - 请求队列管理

3. **性能优化**
   - KV Cache 管理
   - 连续批处理（Continuous Batching）
   - 流式输出

4. **监控和日志**
   - 请求延迟统计
   - Token 生成速度
   - GPU 利用率监控
   - 错误日志记录

### 非功能需求

- 延迟：P99 < 500ms（首 token）
- 吞吐量：> 100 tokens/s
- 可用性：> 99.9%
- 并发：支持 100+ 并发连接

## 架构设计

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │  Client  │  │  Client  │  │  Client  │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
└───────┼─────────────┼─────────────┼─────────────────────────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │     Load Balancer         │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │    API Gateway            │
        │  - Authentication         │
        │  - Rate Limiting          │
        │  - Request Routing        │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Request Scheduler       │
        │  - Queue Management       │
        │  - Dynamic Batching       │
        │  - Priority Scheduling    │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Inference Engine        │
        │  - Model Loading          │
        │  - KV Cache Management    │
        │  - Batch Execution        │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │    Backend (GPU)          │
        │  - CUDA/Metal/Vulkan      │
        │  - Tensor Parallel        │
        └───────────────────────────┘
```

### 组件设计

**1. Web 服务器层**

```cpp
// 使用 FastHTTP 或 Crow
#include <crow.h>

class InferenceServer {
private:
    crow::SimpleApp app;
    ModelManager *model_mgr;
    RequestScheduler *scheduler;
    
public:
    void setup_routes() {
        CROW_ROUTE(app, "/v1/completions")
            .methods("POST"_method)
            ([&](const crow::request& req) {
                auto body = crow::json::load(req.body);
                return handle_completion(body);
            });
        
        CROW_ROUTE(app, "/health")
            .methods("GET"_method)
            ([]() {
                return crow::response(200, "OK");
            });
    }
    
    void run() {
        app.port(8080).multithreaded().run();
    }
};
```

**2. 请求调度器**

```cpp
class RequestScheduler {
private:
    std::queue<Request> pending_queue;
    std::vector<Request> batch;
    size_t max_batch_size;
    std::mutex queue_mutex;
    
public:
    void enqueue(Request req) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        pending_queue.push(req);
    }
    
    std::vector<Request> get_next_batch() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        batch.clear();
        
        while (!pending_queue.empty() && 
               batch.size() < max_batch_size) {
            batch.push_back(pending_queue.front());
            pending_queue.pop();
        }
        
        return batch;
    }
};
```

**3. 模型管理器**

```cpp
class ModelManager {
private:
    ggml_backend_t backend;
    struct ggml_model *model;
    KVCacheManager *kv_cache;
    
public:
    void load_model(const std::string &path) {
        // 加载模型到 GPU
        model = ggml_model_load(path.c_str(), backend);
        kv_cache = new KVCacheManager(backend);
    }
    
    std::vector<float> infer(
        const std::vector<float>& input) {
        // 分配 KV Cache
        kv_cache->allocate(input.size());
        
        // 执行推理
        auto output = ggml_model_forward(model, input, kv_cache);
        
        return output;
    }
};
```

**4. 性能监控器**

```cpp
class PerformanceMonitor {
private:
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> total_tokens{0};
    std::vector<double> latencies;
    std::mutex stats_mutex;
    
public:
    void record_request(double latency_ms, int tokens) {
        total_requests++;
        total_tokens += tokens;
        
        std::lock_guard<std::mutex> lock(stats_mutex);
        latencies.push_back(latency_ms);
    }
    
    void print_stats() {
        printf("Total Requests: %lu\n", total_requests.load());
        printf("Total Tokens: %lu\n", total_tokens.load());
        
        // 计算 P50, P90, P99 延迟
        std::sort(latencies.begin(), latencies.end());
        printf("P50 Latency: %.2f ms\n", 
               latencies[latencies.size() * 0.5]);
        printf("P99 Latency: %.2f ms\n", 
               latencies[latencies.size() * 0.99]);
    }
};
```

## 实现步骤

### 步骤 1：搭建基础框架

```bash
# 创建项目结构
mkdir inference-service
cd inference-service

# 初始化 CMake
cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.14)
project(inference_service)

set(CMAKE_CXX_STANDARD 17)

find_package(crow REQUIRED)
find_package(ggml REQUIRED)

add_executable(server main.cpp)
target_link_libraries(server crow::crow ggml::ggml)
EOF
```

### 步骤 2：实现 Web 服务器

```cpp
// main.cpp
#include <crow.h>
#include "inference_engine.h"

InferenceEngine engine;

int main() {
    // 初始化模型
    engine.load_model("models/llama-7b");
    
    crow::SimpleApp app;
    
    // 健康检查
    CROW_ROUTE(app, "/health")
        ([]() {
            return crow::response(200, "OK");
        });
    
    // 文本生成
    CROW_ROUTE(app, "/v1/completions")
        .methods("POST"_method)
        ([&](const crow::request& req) {
            auto body = crow::json::load(req.body);
            auto prompt = body["prompt"].s();
            auto max_tokens = body["max_tokens"].i();
            
            auto result = engine.generate(prompt, max_tokens);
            
            crow::json::wvalue response;
            response["text"] = result.text;
            response["tokens"] = result.tokens;
            
            return crow::response(200, response);
        });
    
    // 启动服务器
    app.port(8080).multithreaded().run();
    
    return 0;
}
```

### 步骤 3：实现推理引擎

```cpp
// inference_engine.h
#pragma once

#include <string>
#include <vector>

struct GenerationResult {
    std::string text;
    int tokens;
    double latency_ms;
};

class InferenceEngine {
private:
    struct ggml_model *model;
    ggml_backend_t backend;
    
public:
    void load_model(const std::string &path);
    GenerationResult generate(
        const std::string &prompt, 
        int max_tokens);
};
```

```cpp
// inference_engine.cpp
#include "inference_engine.h"

void InferenceEngine::load_model(const std::string &path) {
    backend = ggml_backend_cuda_init(0);
    model = ggml_model_load(path.c_str(), backend);
}

GenerationResult InferenceEngine::generate(
    const std::string &prompt,
    int max_tokens) {
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Tokenize
    std::vector<int> tokens = tokenize(prompt);
    
    // 推理
    std::vector<int> output_tokens;
    for (int i = 0; i < max_tokens; i++) {
        auto logits = ggml_model_forward(model, tokens);
        int next_token = sample(logits);
        output_tokens.push_back(next_token);
        
        if (next_token == EOS_TOKEN) break;
        
        tokens.push_back(next_token);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration<double, std::milli>(
        end - start).count();
    
    GenerationResult result;
    result.text = detokenize(output_tokens);
    result.tokens = output_tokens.size();
    result.latency_ms = latency;
    
    return result;
}
```

### 步骤 4：添加动态批处理

```cpp
// batch_scheduler.h
#pragma once

#include <queue>
#include <mutex>

struct Request {
    std::string prompt;
    int max_tokens;
    std::promise<GenerationResult> promise;
};

class BatchScheduler {
private:
    std::queue<Request> queue;
    std::mutex mutex;
    size_t max_batch_size;
    
public:
    BatchScheduler(size_t batch_size) 
        : max_batch_size(batch_size) {}
    
    void submit(Request req) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(std::move(req));
    }
    
    std::vector<Request> get_batch() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<Request> batch;
        
        while (!queue.empty() && 
               batch.size() < max_batch_size) {
            batch.push_back(std::move(queue.front()));
            queue.pop();
        }
        
        return batch;
    }
};
```

### 步骤 5：添加监控

```cpp
// metrics.h
#pragma once

#include <atomic>
#include <vector>

class Metrics {
private:
    std::atomic<uint64_t> requests{0};
    std::atomic<uint64_t> tokens_generated{0};
    std::vector<double> latencies;
    std::mutex mutex;
    
public:
    void record(double latency_ms, int tokens) {
        requests++;
        tokens_generated += tokens;
        
        std::lock_guard<std::mutex> lock(mutex);
        latencies.push_back(latency_ms);
    }
    
    void export_prometheus() {
        printf("# HELP inference_requests_total Total requests\n");
        printf("# TYPE inference_requests_total counter\n");
        printf("inference_requests_total %lu\n", 
               requests.load());
        
        printf("# HELP inference_tokens_total Total tokens\n");
        printf("# TYPE inference_tokens_total counter\n");
        printf("inference_tokens_total %lu\n", 
               tokens_generated.load());
    }
};
```

## 测试验证

### 功能测试

```bash
# 健康检查
curl http://localhost:8080/health

# 文本生成
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 100
  }'
```

### 性能测试

```bash
# 使用 ab (Apache Bench)
ab -n 1000 -c 10 \
   -p request.json \
   -T application/json \
   http://localhost:8080/v1/completions

# 使用 wrk
wrk -t12 -c400 -d30s \
   -s post.lua \
   http://localhost:8080/v1/completions
```

## 部署指南

### Docker 部署

```dockerfile
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

WORKDIR /app

COPY server /app/
COPY models /app/models/

EXPOSE 8080

CMD ["./server"]
```

### Kubernetes 部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference
  template:
    spec:
      containers:
      - name: server
        image: inference-service:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: inference-service
spec:
  selector:
    app: inference
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

## 评估标准

| 标准 | 权重 | 要求 |
|------|------|------|
| **功能完整性** | 30% | 所有 API 正常工作 |
| **性能表现** | 25% | P99 < 500ms |
| **代码质量** | 20% | 代码规范、可维护 |
| **监控完善** | 15% | 完整的指标监控 |
| **文档完整** | 10% | 清晰的部署文档 |

## 扩展挑战

完成基础功能后，可以尝试：

1. **流式输出**：实现 Server-Sent Events (SSE)
2. **模型热加载**：支持不重启切换模型
3. **多模型支持**：同时加载多个模型
4. **分布式部署**：多节点负载均衡
5. **自动扩缩容**：基于负载自动扩展

## 参考资料

- [Crow Documentation](https://crowcpp.org/)
- [vLLM Architecture](https://vllm.ai/)
- [TGI Documentation](https://huggingface.co/docs/text-generation-inference)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
