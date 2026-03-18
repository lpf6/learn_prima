# LLaMA.cpp 实现与优化详解

## 概述

LLaMA.cpp 是一个高效的 LLM 推理框架，以其**极低的资源占用**和**出色的跨平台支持**著称。Prima.cpp 基于 llama.cpp 构建，继承了其核心优化技术。

## 核心设计理念

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLaMA.cpp 设计原则                           │
├─────────────────────────────────────────────────────────────────┤
│  1. 纯 C/C++ 实现，无外部依赖                                    │
│  2. 内存优先：最小化内存占用                                     │
│  3. 量化优先：支持多种量化格式                                   │
│  4. 跨平台：CPU/GPU/Mobile 全覆盖                               │
│  5. 简单易用：单文件可编译                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 一、内存映射 (mmap) 模型加载

### 1.1 问题：传统加载方式

```cpp
// 传统方式：一次性加载整个模型到内存
void * model_data = malloc(model_size);  // 70B 模型需要 40GB+ 内存
fread(model_data, model_size, 1, file);
// 问题：
// 1. 内存占用大：需要完整内存
// 2. 启动慢：需要等待全部加载
// 3. 不灵活：无法处理比内存大的模型
```

### 1.2 解决方案：mmap 延迟加载

```cpp
// 文件：src/llama.cpp

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

struct llama_model_loader {
    int fd;                    // 文件描述符
    void * base_addr;          // mmap 基地址
    size_t file_size;          // 文件大小
    struct gguf_context * ctx; // GGUF 上下文
};

// 使用 mmap 映射模型文件
struct llama_model_loader * llama_model_loader_init(const char * path) {
    struct llama_model_loader * loader = calloc(1, sizeof(*loader));
    
    // 1. 打开文件
    loader->fd = open(path, O_RDONLY);
    if (loader->fd == -1) {
        return NULL;
    }
    
    // 2. 获取文件大小
    struct stat st;
    fstat(loader->fd, &st);
    loader->file_size = st.st_size;
    
    // 3. mmap 映射
    loader->base_addr = mmap(
        NULL,                    // 让系统选择地址
        loader->file_size,       // 文件大小
        PROT_READ,               // 只读
        MAP_PRIVATE,             // 私有映射（写时复制）
        loader->fd,
        0                        // 偏移量
    );
    
    if (loader->base_addr == MAP_FAILED) {
        close(loader->fd);
        free(loader);
        return NULL;
    }
    
    // 4. 解析 GGUF 头部
    loader->ctx = gguf_init_from_file(path, &(struct gguf_init_params) {
        .no_alloc = true,  // 不分配内存，使用 mmap
    });
    
    return loader;
}
```

### 1.3 mmap 优势详解

```
┌─────────────────────────────────────────────────────────────────┐
│                    mmap vs 传统加载                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  传统加载:                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  磁盘文件 (40GB)                                         │   │
│  │       ↓ fread() - 需要读取全部数据                        │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  内存 (40GB) - 必须有足够内存                      │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │  启动时间: ~30秒                                         │   │
│  │  内存需求: 40GB                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  mmap 加载:                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  磁盘文件 (40GB)                                         │   │
│  │       ↓ mmap() - 立即返回                                │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  虚拟地址空间 (映射)                               │   │   │
│  │  │  ┌─────┬─────┬─────┬─────┬─────┐               │   │   │
│  │  │  │ P0  │ P1  │ P2  │ P3  │ ... │               │   │   │
│  │  │  │未加载│已加载│未加载│已加载│     │               │   │   │
│  │  │  └─────┴─────┴─────┴─────┴─────┘               │   │   │
│  │  │       ↓         ↓         ↓                     │   │   │
│  │  │  ┌─────────────────────────────────────┐       │   │   │
│  │  │  │  物理内存 (只加载访问的页面)          │       │   │   │
│  │  │  │  实际使用: ~2GB                      │       │   │   │
│  │  │  └─────────────────────────────────────┘       │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │  启动时间: <1秒                                          │   │
│  │  内存需求: 按需分配                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 页面缓存机制

```cpp
// mmap 页面缓存工作原理

// 1. 首次访问某层权重
float * layer_0_weight = (float *)((char *)loader->base_addr + layer_0_offset);
float val = layer_0_weight[0];  // 触发缺页中断

// 2. 缺页中断处理
// OS 检测到访问未加载的页面
// 从磁盘加载对应页面到物理内存
// 更新页表映射
// 继续执行

// 3. 后续访问同一页面
float val2 = layer_0_weight[1];  // 直接访问，无缺页中断

// 4. 内存不足时
// OS 自动释放最近未使用的页面
// 下次访问时重新加载
```

### 1.5 GGUF 文件格式详解

```cpp
// 文件：gguf-py/gguf/constants.py

/*
GGUF 文件结构:
┌────────────────────────────────────────────────────────────────┐
│ Header (24 bytes)                                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ magic:     "GGUF" (4 bytes)                              │ │
│  │ version:   3 (4 bytes)                                   │ │
│  │ tensor_count:  N (8 bytes)                               │ │
│  │ metadata_kv_count: M (8 bytes)                           │ │
│  └──────────────────────────────────────────────────────────┘ │
├────────────────────────────────────────────────────────────────┤
│ Metadata KV (M 个键值对)                                        │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ key_length: uint32                                       │ │
│  │ key: string (key_length bytes)                           │ │
│  │ type: uint32 (0-8)                                       │ │
│  │ value: varies by type                                    │ │
│  │   - UINT8, INT8, UINT16, INT16, UINT32, INT32           │ │
│  │   - FLOAT32, BOOL, STRING, ARRAY, OBJECT                │ │
│  └──────────────────────────────────────────────────────────┘ │
│  示例:                                                         │
│  - "general.name": "LLaMA-2-7B"                               │
│  - "llama.embedding_length": 4096                             │
│  - "llama.block_count": 32                                    │
│  - "llama.attention.head_count": 32                           │
│  - "llama.attention.head_count_kv": 32                        │
│  - "llama.attention.layer_norm_rms_epsilon": 0.00001          │
│  - "llama.rope.dimension_count": 128                          │
│  - "llama.rope.freq_base": 10000.0                            │
├────────────────────────────────────────────────────────────────┤
│ Tensor Info (N 个张量信息)                                      │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ name_length: uint32                                      │ │
│  │ name: string (name_length bytes)                         │ │
│  │ n_dims: uint32                                           │ │
│  │ dims: int64[n_dims]                                      │ │
│  │ type: ggml_type (uint32)                                 │ │
│  │ offset: uint64 (从文件开始的偏移)                         │ │
│  └──────────────────────────────────────────────────────────┘ │
│  示例:                                                         │
│  - "token_embd.weight": [4096, 32000], Q4_K, offset=...       │
│  - "blk.0.attn_q.weight": [4096, 4096], Q4_K, offset=...      │
│  - "blk.0.attn_k.weight": [4096, 1024], Q4_K, offset=...      │
├────────────────────────────────────────────────────────────────┤
│ Alignment Padding                                              │
│  (对齐到 32 字节边界，优化内存访问)                              │
├────────────────────────────────────────────────────────────────┤
│ Tensor Data (张量数据)                                          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ 权重数据，按 tensor info 中的 offset 访问                 │ │
│  │ 支持多种量化格式:                                          │ │
│  │   - F32, F16: 浮点                                       │ │
│  │   - Q4_0, Q4_1, Q5_0, Q5_1, Q8_0: 基础量化               │ │
│  │   - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K: K-quant                │ │
│  │   - IQ1_S, IQ2_XXS, IQ3_XXS, IQ4_NL: IQ 系列             │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
*/

// 读取 GGUF 实现
struct gguf_context * gguf_init_from_file(const char * path, struct gguf_init_params params) {
    struct gguf_context * ctx = calloc(1, sizeof(*ctx));
    
    // 1. 读取 Header
    uint32_t magic;
    fread(&magic, sizeof(magic), 1, file);
    GGML_ASSERT(magic == GGUF_MAGIC);
    
    uint32_t version;
    fread(&version, sizeof(version), 1, file);
    GGML_ASSERT(version >= 2);
    
    uint64_t tensor_count;
    fread(&tensor_count, sizeof(tensor_count), 1, file);
    ctx->header.n_tensors = tensor_count;
    
    uint64_t metadata_kv_count;
    fread(&metadata_kv_count, sizeof(metadata_kv_count), 1, file);
    ctx->header.n_kv = metadata_kv_count;
    
    // 2. 读取 Metadata KV
    ctx->kv = calloc(ctx->header.n_kv, sizeof(*ctx->kv));
    for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
        // 读取 key
        uint64_t key_len;
        fread(&key_len, sizeof(key_len), 1, file);
        ctx->kv[i].key = calloc(key_len + 1, 1);
        fread(ctx->kv[i].key, key_len, 1, file);
        
        // 读取 type
        uint32_t type;
        fread(&type, sizeof(type), 1, file);
        ctx->kv[i].type = type;
        
        // 读取 value
        gguf_read_value(file, type, &ctx->kv[i].value);
    }
    
    // 3. 读取 Tensor Info
    ctx->infos = calloc(ctx->header.n_tensors, sizeof(*ctx->infos));
    for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
        // 读取 name
        uint64_t name_len;
        fread(&name_len, sizeof(name_len), 1, file);
        ctx->infos[i].name = calloc(name_len + 1, 1);
        fread(ctx->infos[i].name, name_len, 1, file);
        
        // 读取 dims
        uint32_t n_dims;
        fread(&n_dims, sizeof(n_dims), 1, file);
        ctx->infos[i].n_dims = n_dims;
        
        for (uint32_t j = 0; j < n_dims; ++j) {
            fread(&ctx->infos[i].dims[j], sizeof(int64_t), 1, file);
        }
        
        // 读取 type
        uint32_t type;
        fread(&type, sizeof(type), 1, file);
        ctx->infos[i].type = type;
        
        // 读取 offset
        fread(&ctx->infos[i].offset, sizeof(uint64_t), 1, file);
    }
    
    return ctx;
}
```

### 1.6 张量访问实现

```cpp
// 文件：src/llama.cpp

// 通过 mmap 访问张量
struct ggml_tensor * llama_get_tensor(
    struct llama_model_loader * loader,
    const char * name,
    enum ggml_type type) {
    
    // 1. 查找张量信息
    int tensor_idx = -1;
    for (int i = 0; i < loader->ctx->header.n_tensors; ++i) {
        if (strcmp(loader->ctx->infos[i].name, name) == 0) {
            tensor_idx = i;
            break;
        }
    }
    
    if (tensor_idx == -1) {
        return NULL;
    }
    
    // 2. 获取张量信息
    struct gguf_tensor_info * info = &loader->ctx->infos[tensor_idx];
    
    // 3. 计算张量大小
    size_t tensor_size = ggml_type_size(info->type);
    for (uint32_t i = 0; i < info->n_dims; ++i) {
        tensor_size *= info->dims[i];
    }
    
    // 4. 创建 ggml_tensor
    struct ggml_tensor * tensor = ggml_new_tensor(
        ctx,
        info->type,
        info->n_dims,
        info->dims
    );
    
    // 5. 设置数据指针（指向 mmap 区域）
    tensor->data = (void *)((char *)loader->base_addr + info->offset);
    
    // 6. 首次访问时触发缺页中断，加载到内存
    // 这是自动的，无需额外代码
    
    return tensor;
}
```

### 1.7 内存压力监控

```cpp
// 文件：src/llama.cpp

// 监控内存使用
size_t llama_get_used_memory(struct llama_context * ctx) {
    size_t used = 0;
    
    // 1. 模型权重（mmap）
    // 通过 /proc/self/smaps 或 task_info 获取
    
    // 2. KV Cache
    used += ctx->kv_self.size * ggml_type_size(GGML_TYPE_F16) * 2;
    
    // 3. 中间结果
    used += ctx->buf_compute.size;
    
    // 4. 临时缓冲区
    used += ctx->buf_scratch[0].size + ctx->buf_scratch[1].size;
    
    return used;
}

// 检查内存压力
bool llama_check_memory_pressure(struct llama_context * ctx) {
    size_t used = llama_get_used_memory(ctx);
    size_t total = get_total_system_memory();
    
    float pressure = (float)used / total;
    
    // 保持内存压力低于 50%
    if (pressure > 0.5f) {
        // 可以选择释放一些缓存
        llama_kv_cache_clear(ctx);
        return false;
    }
    
    return true;
}
```

---

## 二、KV Cache 优化

### 2.1 问题：重复计算

```
自回归生成过程详解:

Token 1 生成:
┌─────────────────────────────────────────────────────────────┐
│  输入: "The"                                                │
│  计算: Q1, K1, V1                                           │
│  Attention: softmax(Q1 × K1^T) × V1                         │
│  输出: Token 2 = "quick"                                     │
└─────────────────────────────────────────────────────────────┘

Token 2 生成:
┌─────────────────────────────────────────────────────────────┐
│  输入: "The", "quick"                                       │
│  计算: Q1, K1, V1 (重复!), Q2, K2, V2                       │
│  Attention: softmax([Q1,Q2] × [K1,K2]^T) × [V1,V2]          │
│  输出: Token 3 = "brown"                                     │
│                                                             │
│  问题: K1, V1 已经计算过，为什么要重复计算？                   │
└─────────────────────────────────────────────────────────────┘

Token 3 生成:
┌─────────────────────────────────────────────────────────────┐
│  输入: "The", "quick", "brown"                              │
│  计算: K1, V1 (重复!), K2, V2 (重复!), K3, V3               │
│  Attention: softmax([Q1,Q2,Q3] × [K1,K2,K3]^T) × [V1,V2,V3] │
│  输出: Token 4 = "fox"                                       │
│                                                             │
│  问题: 重复计算越来越多！                                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 解决方案：KV Cache

```cpp
// 文件：src/llama.cpp

struct llama_kv_cache {
    // 主要存储
    struct ggml_tensor * k;  // K cache: [n_layer, n_ctx, n_head_kv, head_dim]
    struct ggml_tensor * v;  // V cache: [n_layer, n_ctx, n_head_kv, head_dim]
    
    // 管理信息
    int n;                   // 当前缓存的 token 数
    int head;                // 循环缓冲区头部（用于旋转）
    int size;                // 缓冲区大小（最大 token 数）
    int used;                // 已使用的槽位数
    
    // 优化信息
    bool has_shift;          // 是否需要移位（处理删除的 token）
    bool do_defrag;          // 是否需要碎片整理
    int * cells;             // 槽位管理
    
    // 量化支持
    struct ggml_tensor * k_l;  // 量化后的 K
    struct ggml_tensor * v_l;  // 量化后的 V
    enum ggml_type type;       // 缓存类型 (F16, Q8_0, Q4_0)
};

// 初始化 KV Cache
bool llama_kv_cache_init(
    struct llama_kv_cache & cache,
    struct llama_context & lctx,
    enum ggml_type type,
    int n_ctx,
    int n_layer,
    int n_head_kv,
    int head_dim) {
    
    const int64_t n_elements = (int64_t)n_layer * n_ctx * n_head_kv * head_dim;
    
    // 计算内存大小
    const size_t element_size = ggml_type_size(type);
    const size_t size = n_elements * element_size;
    
    // 分配内存
    cache.k = ggml_new_tensor_4d(lctx.ctx, type, head_dim, n_head_kv, n_ctx, n_layer);
    cache.v = ggml_new_tensor_4d(lctx.ctx, type, head_dim, n_head_kv, n_ctx, n_layer);
    
    // 初始化管理信息
    cache.size = n_ctx;
    cache.n = 0;
    cache.head = 0;
    cache.used = 0;
    cache.type = type;
    
    // 分配槽位管理数组
    cache.cells = calloc(n_ctx, sizeof(*cache.cells));
    
    return true;
}
```

### 2.3 KV Cache 更新实现

```cpp
// 文件：src/llama.cpp

// 更新 KV Cache
bool llama_kv_cache_update(
    struct llama_context & lctx,
    int n_tokens,
    const llama_pos * pos,
    const struct ggml_tensor * k_cur,
    const struct ggml_tensor * v_cur) {
    
    auto & kv = lctx.kv_self;
    const auto & model = lctx.model;
    const int n_layer = model.hparams.n_layer;
    
    // 1. 检查是否有足够空间
    if (kv.n + n_tokens > kv.size) {
        // 需要驱逐旧的 token
        llama_kv_cache_evict(kv, n_tokens);
    }
    
    // 2. 为每个层更新缓存
    for (int il = 0; il < n_layer; ++il) {
        // 获取当前层的 K, V
        struct ggml_tensor * k_layer = ggml_view_3d(
            lctx.ctx, kv.k,
            model.hparams.head_dim,
            model.hparams.n_head_kv,
            n_tokens,
            ggml_row_size(kv.k->type, model.hparams.head_dim),
            ggml_row_size(kv.k->type, model.hparams.head_dim * model.hparams.n_head_kv),
            il * ggml_nbytes(kv.k) / n_layer
        );
        
        struct ggml_tensor * v_layer = ggml_view_3d(
            lctx.ctx, kv.v,
            model.hparams.head_dim,
            model.hparams.n_head_kv,
            n_tokens,
            ggml_row_size(kv.v->type, model.hparams.head_dim),
            ggml_row_size(kv.v->type, model.hparams.head_dim * model.hparams.n_head_kv),
            il * ggml_nbytes(kv.v) / n_layer
        );
        
        // 3. 复制新的 K, V 到缓存
        // 注意：这里使用异步复制，与计算重叠
        ggml_cpy(lctx.ctx, k_cur, k_layer);
        ggml_cpy(lctx.ctx, v_cur, v_layer);
    }
    
    // 4. 更新管理信息
    for (int i = 0; i < n_tokens; ++i) {
        kv.cells[kv.n + i].pos = pos[i];
        kv.cells[kv.n + i].seq_id = lctx.current_seq_id;
    }
    
    kv.n += n_tokens;
    kv.used += n_tokens;
    
    return true;
}
```

### 2.4 KV Cache 内存布局详解

```
KV Cache 内存布局:

维度顺序: [n_layer, n_ctx, n_head_kv, head_dim]

Layer 0:
┌─────────────────────────────────────────────────────────────────┐
│ K Cache                                                         │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ Head 0                                                      ││
│ │ ┌─────────────────────────────────────────────────────────┐││
│ │ │ Token 0: [d0, d1, d2, ..., d127]  (head_dim=128)        │││
│ │ │ Token 1: [d0, d1, d2, ..., d127]                        │││
│ │ │ ...                                                      │││
│ │ │ Token N: [d0, d1, d2, ..., d127]                        │││
│ │ └─────────────────────────────────────────────────────────┘││
│ │ Head 1: ...                                                 ││
│ │ ...                                                         ││
│ │ Head 31: ...                                                ││
│ └─────────────────────────────────────────────────────────────┘│
│ V Cache (同样结构)                                              │
└─────────────────────────────────────────────────────────────────┘

Layer 1:
┌─────────────────────────────────────────────────────────────────┐
│ ...                                                             │
└─────────────────────────────────────────────────────────────────┘

...

Layer 31:
┌─────────────────────────────────────────────────────────────────┐
│ ...                                                             │
└─────────────────────────────────────────────────────────────────┘

内存占用计算:
───────────────────────────────────────────────────────────────────
FP32:
  KV Cache = 2 × n_layer × n_ctx × n_head_kv × head_dim × 4 bytes
           = 2 × 32 × 4096 × 32 × 128 × 4
           = 16 GB

FP16:
  KV Cache = 2 × 32 × 4096 × 32 × 128 × 2
           = 8 GB

Q8_0:
  KV Cache = 2 × 32 × 4096 × 32 × 128 × 1
           = 4 GB

Q4_0:
  KV Cache = 2 × 32 × 4096 × 32 × 128 × 0.5
           = 2 GB
───────────────────────────────────────────────────────────────────
```

### 2.5 KV Cache 量化实现

```cpp
// 文件：src/llama.cpp

// KV Cache 量化
void llama_kv_cache_quantize(
    struct llama_kv_cache & kv,
    enum ggml_type type) {
    
    if (kv.type == type) {
        return;  // 已经是目标类型
    }
    
    // 1. 分配新的量化缓存
    struct ggml_tensor * k_q = ggml_new_tensor_4d(
        ctx, type,
        kv.k->ne[0], kv.k->ne[1], kv.k->ne[2], kv.k->ne[3]
    );
    
    struct ggml_tensor * v_q = ggml_new_tensor_4d(
        ctx, type,
        kv.v->ne[0], kv.v->ne[1], kv.v->ne[2], kv.v->ne[3]
    );
    
    // 2. 量化
    // K 量化
    ggml_quantize_chunk(
        type,
        (const float *)kv.k->data,
        k_q->data,
        0,  // offset
        ggml_nelements(kv.k),
        NULL,  // work
        0      // work size
    );
    
    // V 量化
    ggml_quantize_chunk(
        type,
        (const float *)kv.v->data,
        v_q->data,
        0,
        ggml_nelements(kv.v),
        NULL,
        0
    );
    
    // 3. 替换缓存
    ggml_free(kv.k);
    ggml_free(kv.v);
    kv.k = k_q;
    kv.v = v_q;
    kv.type = type;
}
```

### 2.6 KV Cache 碎片整理

```cpp
// 文件：src/llama.cpp

// KV Cache 碎片整理
void llama_kv_cache_defrag(struct llama_kv_cache & kv) {
    if (!kv.do_defrag) {
        return;
    }
    
    // 1. 分配临时缓冲区
    size_t buf_size = ggml_nbytes(kv.k) + ggml_nbytes(kv.v);
    void * buf = malloc(buf_size);
    
    // 2. 紧凑复制
    int new_n = 0;
    for (int i = 0; i < kv.size; ++i) {
        if (kv.cells[i].pos >= 0) {
            // 复制 K
            memcpy(
                (char *)buf + new_n * ggml_row_size(kv.k->type, kv.k->ne[0]),
                (char *)kv.k->data + i * ggml_row_size(kv.k->type, kv.k->ne[0]),
                ggml_row_size(kv.k->type, kv.k->ne[0])
            );
            
            // 复制 V
            // ...
            
            kv.cells[new_n] = kv.cells[i];
            new_n++;
        }
    }
    
    // 3. 复制回原缓存
    memcpy(kv.k->data, buf, ggml_nbytes(kv.k));
    memcpy(kv.v->data, (char *)buf + ggml_nbytes(kv.k), ggml_nbytes(kv.v));
    
    // 4. 更新管理信息
    kv.n = new_n;
    kv.used = new_n;
    kv.do_defrag = false;
    
    free(buf);
}
```

---

## 三、批处理推理

### 3.1 问题：逐个生成效率低

```cpp
// 单个 token 生成的问题
for (int i = 0; i < n_tokens; ++i) {
    // 每次只处理一个 token
    float * logits = llama_eval(ctx, &token, 1, n_past);
    token = sample_token(logits);
    n_past++;
}

// 问题分析:
// 1. GPU 内核启动开销: 每次启动需要 ~0.1ms
// 2. GPU 利用率低: 单个 token 无法充分利用并行性
// 3. 内存带宽浪费: 每次都要加载权重

// 性能分析:
// 假设:
//   - 内核启动开销: 0.1ms
//   - 单 token 计算: 0.5ms
//   - 总时间: 0.6ms/token
//   - 512 tokens: 307ms
//
// 批处理:
//   - 内核启动开销: 0.1ms (一次)
//   - 512 tokens 计算: 5ms
//   - 总时间: 5.1ms
//   - 加速比: 60x
```

### 3.2 批处理数据结构

```cpp
// 文件：src/llama.cpp

struct llama_batch {
    int32_t n_tokens;        // 批次中的 token 数
    
    llama_token * token;     // token IDs [n_tokens]
    int32_t * pos;           // 每个 token 的位置 [n_tokens]
    int32_t * n_seq_id;      // 每个 token 的序列 ID 数 [n_tokens]
    llama_seq_id ** seq_id;  // 序列 IDs [n_tokens][n_seq_id]
    int8_t * logits;         // 是否需要输出 logits [n_tokens]
    
    // 便捷构造函数
    static llama_batch get_one(llama_token token, int32_t pos) {
        return {
            /*n_tokens*/ 1,
            /*token*/    &token,
            /*pos*/      &pos,
            /*n_seq_id*/ nullptr,
            /*seq_id*/   nullptr,
            /*logits*/   nullptr,
        };
    }
    
    // 批处理构造
    static llama_batch get(
        int32_t n_tokens,
        llama_token * token,
        int32_t * pos,
        int32_t * n_seq_id,
        llama_seq_id ** seq_id,
        int8_t * logits) {
        return {
            n_tokens,
            token,
            pos,
            n_seq_id,
            seq_id,
            logits,
        };
    }
};
```

### 3.3 批处理推理实现

```cpp
// 文件：src/llama.cpp

int llama_decode(
    struct llama_context & lctx,
    struct llama_batch batch) {
    
    const auto & model = lctx.model;
    const auto & hparams = model.hparams;
    
    const int n_tokens = batch.n_tokens;
    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    
    // 1. 准备输入嵌入
    struct ggml_tensor * cur = ggml_new_tensor_2d(
        lctx.ctx, GGML_TYPE_F32, n_embd, n_tokens
    );
    
    // 获取每个 token 的嵌入
    for (int i = 0; i < n_tokens; ++i) {
        ggml_tensor * embd = ggml_get_rows(
            lctx.ctx,
            model.tok_embd,
            ggml_new_i32(lctx.ctx, batch.token[i])
        );
        
        // 复制到 cur
        ggml_cpy(lctx.ctx, embd, ggml_view_1d(lctx.ctx, cur, n_embd, i * n_embd * 4));
    }
    
    // 2. 添加位置编码
    cur = ggml_add(lctx.ctx, cur, ggml_get_rows(lctx.ctx, model.pos_embd, ...));
    
    // 3. 逐层处理
    for (int il = 0; il < n_layer; ++il) {
        // 3.1 Attention
        struct ggml_tensor * attn_out = build_attention_batch(
            lctx, il, cur, batch
        );
        
        // 3.2 残差连接
        cur = ggml_add(lctx.ctx, cur, attn_out);
        
        // 3.3 FFN
        struct ggml_tensor * ffn_out = build_ffn_batch(
            lctx, il, cur
        );
        
        // 3.4 残差连接
        cur = ggml_add(lctx.ctx, cur, ffn_out);
    }
    
    // 4. 最终归一化
    cur = ggml_rms_norm(lctx.ctx, cur, hparams.f_norm_eps);
    cur = ggml_mul(lctx.ctx, cur, model.output_norm);
    
    // 5. 输出投影
    cur = ggml_mul_mat(lctx.ctx, model.output, cur);
    
    // 6. 构建计算图并执行
    ggml_build_forward_expand(&lctx.gf, cur);
    ggml_graph_compute(lctx.ctx, &lctx.gf);
    
    // 7. 提取 logits
    for (int i = 0; i < n_tokens; ++i) {
        if (batch.logits[i]) {
            memcpy(
                lctx.logits + i * model.hparams.n_vocab,
                (float *)cur->data + i * model.hparams.n_vocab,
                model.hparams.n_vocab * sizeof(float)
            );
        }
    }
    
    return 0;
}
```

### 3.4 批处理 Attention 实现

```cpp
// 文件：src/llama.cpp

struct ggml_tensor * build_attention_batch(
    struct llama_context & lctx,
    int il,
    struct ggml_tensor * cur,
    struct llama_batch batch) {
    
    const auto & model = lctx.model;
    const auto & hparams = model.hparams;
    const int n_tokens = batch.n_tokens;
    const int n_head = hparams.n_head;
    const int n_head_kv = hparams.n_head_kv;
    const int head_dim = hparams.n_embd / n_head;
    
    // 1. Attention Norm
    struct ggml_tensor * attn_norm = ggml_rms_norm(lctx.ctx, cur, hparams.f_norm_eps);
    attn_norm = ggml_mul(lctx.ctx, attn_norm, model.layers[il].attn_norm);
    
    // 2. Q, K, V 投影
    struct ggml_tensor * Q = ggml_mul_mat(lctx.ctx, model.layers[il].wq, attn_norm);
    struct ggml_tensor * K = ggml_mul_mat(lctx.ctx, model.layers[il].wk, attn_norm);
    struct ggml_tensor * V = ggml_mul_mat(lctx.ctx, model.layers[il].wv, attn_norm);
    
    // 3. Reshape 为多头形式
    // Q: [n_embd, n_tokens] -> [head_dim, n_head, n_tokens]
    Q = ggml_reshape_3d(lctx.ctx, Q, head_dim, n_head, n_tokens);
    // K: [n_embd, n_tokens] -> [head_dim, n_head_kv, n_tokens]
    K = ggml_reshape_3d(lctx.ctx, K, head_dim, n_head_kv, n_tokens);
    // V: [n_embd, n_tokens] -> [head_dim, n_head_kv, n_tokens]
    V = ggml_reshape_3d(lctx.ctx, V, head_dim, n_head_kv, n_tokens);
    
    // 4. RoPE 位置编码
    Q = ggml_rope(lctx.ctx, Q, batch.pos, head_dim, hparams.rope_type, 0);
    K = ggml_rope(lctx.ctx, K, batch.pos, head_dim, hparams.rope_type, 0);
    
    // 5. 更新 KV Cache
    llama_kv_cache_update(lctx, n_tokens, batch.pos, K, V);
    
    // 6. 获取完整的 K, V (包括缓存的)
    struct ggml_tensor * K_full = ggml_view_3d(
        lctx.ctx, lctx.kv_self.k,
        head_dim, lctx.kv_self.n, n_head_kv,
        ggml_row_size(lctx.kv_self.k->type, head_dim),
        ggml_row_size(lctx.kv_self.k->type, head_dim * lctx.kv_self.n),
        il * ggml_nbytes(lctx.kv_self.k) / n_layer
    );
    
    struct ggml_tensor * V_full = ggml_view_3d(
        lctx.ctx, lctx.kv_self.v,
        head_dim, lctx.kv_self.n, n_head_kv,
        ggml_row_size(lctx.kv_self.v->type, head_dim),
        ggml_row_size(lctx.kv_self.v->type, head_dim * lctx.kv_self.n),
        il * ggml_nbytes(lctx.kv_self.v) / n_layer
    );
    
    // 7. Attention 计算
    // QK^T: [head_dim, n_head, n_tokens] × [head_dim, n_kv, n_head_kv]^T
    //     = [n_kv, n_head, n_tokens]
    struct ggml_tensor * QK = ggml_mul_mat(lctx.ctx, K_full, Q);
    
    // Scale
    QK = ggml_scale(lctx.ctx, QK, 1.0f / sqrtf(head_dim));
    
    // 8. Causal Mask
    // 对于批处理中的每个 token，只能看到之前的 token
    QK = ggml_diag_mask_inf(lctx.ctx, QK, n_tokens);
    
    // 9. Softmax
    QK = ggml_soft_max(lctx.ctx, QK);
    
    // 10. Attention × V
    // [n_kv, n_head, n_tokens] × [head_dim, n_kv, n_head_kv]
    // = [head_dim, n_head, n_tokens]
    struct ggml_tensor * O = ggml_mul_mat(lctx.ctx, V_full, QK);
    
    // 11. 合并多头
    O = ggml_reshape_2d(lctx.ctx, O, hparams.n_embd, n_tokens);
    
    // 12. Output 投影
    return ggml_mul_mat(lctx.ctx, model.layers[il].wo, O);
}
```

### 3.5 批处理应用场景

```
批处理应用场景:

1. Prompt 处理
   ┌─────────────────────────────────────────────────────────────┐
   │ 输入: "Translate the following to French: Hello, world!"   │
   │ 批量处理整个 prompt (10-20 tokens)                          │
   │ 加速比: 10-20x                                              │
   └─────────────────────────────────────────────────────────────┘

2. 多序列并行生成
   ┌─────────────────────────────────────────────────────────────┐
   │ 序列 1: "Once upon a time" → "there was a princess..."      │
   │ 序列 2: "In the future" → "robots will help humans..."      │
   │ 序列 3: "The scientist" → "discovered a new element..."     │
   │ 同时处理 3 个序列的下一个 token                              │
   │ 吞吐量提升: 3x                                               │
   └─────────────────────────────────────────────────────────────┘

3. Speculative Decoding 验证
   ┌─────────────────────────────────────────────────────────────┐
   │ Draft Model 生成: Token 1, 2, 3, 4                          │
   │ Target Model 验证: 批量处理 4 个 token                       │
   │ 如果全部正确: 一次获得 4 个 token                            │
   │ 加速比: 最高 4x                                              │
   └─────────────────────────────────────────────────────────────┘
```

---

## 四、量化推理优化

### 4.1 量化矩阵乘法流程

```
量化矩阵乘法完整流程:

┌─────────────────────────────────────────────────────────────────┐
│ 阶段 1: 加载量化权重                                             │
├─────────────────────────────────────────────────────────────────┤
│ 磁盘:                                                            │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ block_q4_0: { d: fp16, qs: [16 bytes] }                    │ │
│ │ 每个 block: 32 个元素 → 18 bytes (压缩比 4.5x)              │ │
│ │                                                             │ │
│ │ 7B 模型 Q4_0:                                               │ │
│ │   FP16: 14 GB                                               │ │
│ │   Q4_0: 3.5 GB                                              │ │
│ └─────────────────────────────────────────────────────────────┘ │
│       ↓ mmap (无需解压)                                          │
├─────────────────────────────────────────────────────────────────┤
│ 阶段 2: GPU 内存                                                 │
├─────────────────────────────────────────────────────────────────┤
│ 显存:                                                            │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 量化权重直接加载到显存                                       │ │
│ │ 无需 CPU 解压，节省内存带宽                                  │ │
│ │                                                             │ │
│ │ RTX 3090 (24GB):                                            │ │
│ │   FP16 70B: 需要 140GB → 无法加载                           │ │
│ │   Q4_0 70B: 需要 35GB → 可以加载 (部分卸载到 CPU)            │ │
│ └─────────────────────────────────────────────────────────────┘ │
│       ↓ CUDA 内核                                                │
├─────────────────────────────────────────────────────────────────┤
│ 阶段 3: 内核计算                                                 │
├─────────────────────────────────────────────────────────────────┤
│ CUDA 内核:                                                       │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 1. 加载量化块到寄存器/共享内存                               │ │
│ │    - 一次加载 32 个元素 (18 bytes)                          │ │
│ │    - 比 FP16 (64 bytes) 节省 3.5x 带宽                      │ │
│ │                                                             │ │
│ │ 2. 反量化到 FP16/F32                                        │ │
│ │    - scale = __half2float(block.d)                         │ │
│ │    - value = (qs[i] - 8) * scale  // Q4_0                  │ │
│ │                                                             │ │
│ │ 3. 执行矩阵乘法                                              │ │
│ │    - 使用 dp4a 指令加速 INT8 点积                           │ │
│ │    - 或使用 Tensor Core                                     │ │
│ │                                                             │ │
│ │ 4. 累加结果                                                  │ │
│ │    - 输出为 FP32                                            │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 量化矩阵乘法内核详解

```cpp
// 文件：ggml/src/ggml-cuda/mmq.cu

/*
Q4_0 矩阵乘法内核详解

矩阵维度:
  A (权重): [K, N] 量化
  B (激活): [K, M] FP16
  C (输出): [N, M] FP32

分块策略:
  - 将矩阵分成小块，利用共享内存复用
  - 每个线程块处理一部分输出
*/

// Q4_0 反量化
static __device__ __forceinline__ void dequantize_q4_0(
    const block_q4_0 * x,
    const int k,
    float * v) {
    
    const float d = __half2float(x[k].d);
    
    // 每个 block_q4_0 包含 32 个元素
    // 存储在 16 个字节中 (每个字节存储 2 个 4-bit 值)
    const uint8_t * qs = x[k].qs;
    
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        // 解包 4-bit 值
        const uint8_t vi = qs[i];
        
        // 低 4-bit
        const int8_t v0 = (vi & 0x0F) - 8;  // [-8, 7]
        v[i * 2] = v0 * d;
        
        // 高 4-bit
        const int8_t v1 = (vi >> 4) - 8;
        v[i * 2 + 1] = v1 * d;
    }
}

// 使用 dp4a 的优化版本
static __device__ __forceinline__ float vec_dot_q4_0_q8_0(
    const void * __restrict__ vbq,
    const block_q8_0 * __restrict__ bq8_0,
    const int kbx,
    const int kby) {
    
    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq;
    
    // 获取缩放因子
    const float d = __half2float(bq4_0[kbx].d) * __half2float(bq8_0[kby].d);
    
    // 使用 dp4a 指令计算点积
    int sumi = 0;
    
    #pragma unroll
    for (int i = 0; i < QK4_0 / 8; ++i) {
        // 加载 8 个 4-bit 值 (打包在 4 个字节中)
        const uint32_t v0 = *((const uint32_t *)bq4_0[kbx].qs + i);
        
        // 解包为 2 个 int32 (每个包含 4 个 int8)
        int v0_lo, v0_hi;
        
        // 使用 SIMD 指令解包
        // v0_lo = [q0, q1, q2, q3] (每个 int8)
        // v0_hi = [q4, q5, q6, q7]
        
        // 加载对应的 q8_0 值
        const int v1_lo = *((const int *)bq8_0[kby].qs + i * 2);
        const int v1_hi = *((const int *)bq8_0[kby].qs + i * 2 + 1);
        
        // 使用 dp4a 计算 4 字节点积
        sumi = ggml_cuda_dp4a(v0_lo, v1_lo, sumi);
        sumi = ggml_cuda_dp4a(v0_hi, v1_hi, sumi);
    }
    
    return d * sumi;
}

// 完整的矩阵乘法内核
template<int MMQ_M, int MMQ_N, int MMQ_K>
__global__ void mul_mat_q4_0(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y) {
    
    // 1. 计算输出位置
    const int row = blockIdx.y * MMQ_M + threadIdx.y;
    const int col = blockIdx.x * MMQ_N + threadIdx.x;
    
    if (row >= nrows_x || col >= ncols_y) return;
    
    // 2. 共享内存缓存
    __shared__ float tile_x[MMQ_M][MMQ_K];
    __shared__ float tile_y[MMQ_K][MMQ_N];
    
    // 3. 加载并反量化权重块
    const block_q4_0 * x = (const block_q4_0 *) vx;
    const int blocks_per_row = ncols_x / QK4_0;
    
    float sum = 0.0f;
    
    // 4. 分块计算
    for (int k0 = 0; k0 < ncols_x; k0 += MMQ_K) {
        // 加载权重块到共享内存
        const int k = k0 / QK4_0 + threadIdx.x;
        if (k < blocks_per_row) {
            dequantize_q4_0(x + row * blocks_per_row + k, 0, tile_x[threadIdx.y]);
        }
        
        // 加载激活值到共享内存
        // ...
        
        __syncthreads();
        
        // 计算点积
        for (int k = 0; k < MMQ_K; ++k) {
            sum += tile_x[threadIdx.y][k] * tile_y[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // 5. 写回结果
    dst[row * ncols_y + col] = sum;
}
```

### 4.3 内存带宽优化分析

```
内存带宽优化分析:

假设: RTX 3090, 内存带宽 936 GB/s

FP16 矩阵乘法:
───────────────────────────────────────────────────────────────────
权重加载: 14 GB (7B 模型)
激活加载: 假设 batch=1, seq_len=1, 约 4 KB
总数据量: ~14 GB
理论时间: 14 GB / 936 GB/s = 15 ms
实际时间: ~20 ms (考虑计算时间)
───────────────────────────────────────────────────────────────────

Q4_0 矩阵乘法:
───────────────────────────────────────────────────────────────────
权重加载: 3.5 GB (7B 模型, Q4_0)
激活加载: 4 KB
总数据量: ~3.5 GB
理论时间: 3.5 GB / 936 GB/s = 3.7 ms
实际时间: ~5 ms (考虑反量化开销)
───────────────────────────────────────────────────────────────────

加速比: 20 ms / 5 ms = 4x

关键因素:
1. 内存带宽: 量化减少 4x 数据传输
2. 反量化开销: 需要额外计算，但通常比内存访问快
3. dp4a 加速: INT8 点积比 FP16 快 2x
```

---

由于内容很长，我将继续创建更多详细章节。让我继续补充剩余内容...
