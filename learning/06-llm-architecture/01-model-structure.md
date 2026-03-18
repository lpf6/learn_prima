# 6.1 模型结构理解

## 概述

本节介绍 LLM 的基本结构，以及如何在 Prima.cpp 中实现模型推理。

## Transformer 架构

### 基本结构

```
Transformer Block
├── Input
├── Self-Attention
│   ├── LayerNorm/RMSNorm
│   ├── Q, K, V 投影
│   ├── Attention 计算
│   └── Output 投影
├── Feed-Forward Network
│   ├── LayerNorm/RMSNorm
│   ├── Gate 投影 (SwiGLU)
│   ├── Up 投影
│   ├── Down 投影
│   └── 激活函数
└── Output
```

### 数据流

```
输入 Token IDs
     │
     ▼
┌─────────────┐
│  Embedding  │  查表获取词嵌入
└─────────────┘
     │
     ▼
┌─────────────┐
│   Layer 1   │
│  Attention  │
│     FFN     │
└─────────────┘
     │
     ▼
    ...
     │
     ▼
┌─────────────┐
│   Norm      │
│   Output    │  投影到词表大小
└─────────────┘
     │
     ▼
  Logits (下一个 token 的概率分布)
```

## LLaMA 架构详解

### 模型参数

```cpp
// 文件：src/llama.cpp

struct llama_hparams {
    uint32_t n_vocab;       // 词表大小
    uint32_t n_embd;        // 隐藏层维度
    uint32_t n_layer;       // 层数
    uint32_t n_head;        // 注意力头数
    uint32_t n_head_kv;     // KV 头数（GQA）
    uint32_t n_rot;         // RoPE 维度
    uint32_t n_ff;          // FFN 中间维度

    float f_norm_eps;       // LayerNorm epsilon
    float rope_freq_base;   // RoPE 基础频率
};
```

### 层结构

```cpp
// 文件：src/llama.cpp

struct llama_layer {
    // Attention 权重
    struct ggml_tensor * wq;      // Q 投影 [n_embd, n_embd]
    struct ggml_tensor * wk;      // K 投影 [n_embd, n_embd_kv]
    struct ggml_tensor * wv;      // V 投影 [n_embd, n_embd_kv]
    struct ggml_tensor * wo;      // Output 投影 [n_embd, n_embd]

    // FFN 权重
    struct ggml_tensor * ffn_gate;  // Gate [n_embd, n_ff]
    struct ggml_tensor * ffn_up;    // Up [n_embd, n_ff]
    struct ggml_tensor * ffn_down;  // Down [n_ff, n_embd]

    // Normalization
    struct ggml_tensor * attn_norm;  // Attention 前的 norm
    struct ggml_tensor * ffn_norm;   // FFN 前的 norm
};
```

## GGML 计算图

### GGML 基本概念

```cpp
// GGML 张量
struct ggml_tensor {
    enum ggml_type type;        // 数据类型
    int n_dims;                  // 维度数
    int64_t ne[GGML_MAX_DIMS];  // 每维大小
    size_t nb[GGML_MAX_DIMS];   // 每维步长
    void * data;                 // 数据指针
    struct ggml_tensor * src[GGML_MAX_SRC];  // 源张量
    enum ggml_op op;            // 操作类型
    int32_t op_params[GGML_MAX_OP_PARAMS];  // 操作参数
};

// GGML 计算图
struct ggml_cgraph {
    int n_nodes;
    struct ggml_tensor * nodes[GGML_MAX_NODES];  // 计算节点
    int n_leafs;
    struct ggml_tensor * leafs[GGML_MAX_LEAFS];  // 叶子节点
};
```

### GGML 操作类型

```cpp
enum ggml_op {
    GGML_OP_NONE,

    // 基本运算
    GGML_OP_ADD,
    GGML_OP_MUL,
    GGML_OP_DIV,

    // 矩阵运算
    GGML_OP_MAT_MUL,
    GGML_OP_MUL_MAT,

    // 归一化
    GGML_OP_NORM,
    GGML_OP_RMS_NORM,

    // 激活函数
    GGML_OP_SILU,
    GGML_OP_GELU,
    GGML_OP_RELU,

    // Softmax
    GGML_OP_SOFT_MAX,

    // RoPE
    GGML_OP_ROPE,

    // 其他
    GGML_OP_PERMUTE,
    GGML_OP_RESHAPE,
    GGML_OP_VIEW,
    GGML_OP_CPY,
    // ...
};
```

## 计算图构建示例

### Attention 层

```cpp
// 文件：src/llama.cpp（简化示例）

struct ggml_tensor * build_llama_attention(
    ggml_context * ctx,
    const llama_layer & layer,
    struct ggml_tensor * cur,
    const llama_hparams & hparams) {

    const int n_head = hparams.n_head;
    const int n_head_kv = hparams.n_head_kv;
    const int head_dim = hparams.n_embd / n_head;

    // 1. Attention Norm
    struct ggml_tensor * attn_norm = ggml_rms_norm(ctx, cur, hparams.f_norm_eps);
    attn_norm = ggml_mul(ctx, attn_norm, layer.attn_norm);

    // 2. Q, K, V 投影
    struct ggml_tensor * Q = ggml_mul_mat(ctx, layer.wq, attn_norm);
    struct ggml_tensor * K = ggml_mul_mat(ctx, layer.wk, attn_norm);
    struct ggml_tensor * V = ggml_mul_mat(ctx, layer.wv, attn_norm);

    // 3. Reshape 为多头形式
    Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, n_tokens);
    K = ggml_reshape_3d(ctx, K, head_dim, n_head_kv, n_tokens);
    V = ggml_reshape_3d(ctx, V, head_dim, n_head_kv, n_tokens);

    // 4. RoPE 位置编码
    Q = ggml_rope(ctx, Q, pos, head_dim, hparams.rope_freq_base);
    K = ggml_rope(ctx, K, pos, head_dim, hparams.rope_freq_base);

    // 5. Attention 计算
    struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
    KQ = ggml_scale(ctx, KQ, 1.0f / sqrtf(head_dim));
    KQ = ggml_diag_mask_inf(ctx, KQ, n_past);
    KQ = ggml_soft_max(ctx, KQ);

    // 6. KQV = KQ * V
    struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);

    // 7. Output 投影
    return ggml_mul_mat(ctx, layer.wo, KQV);
}
```

### FFN 层

```cpp
struct ggml_tensor * build_llama_ffn(
    ggml_context * ctx,
    const llama_layer & layer,
    struct ggml_tensor * cur,
    const llama_hparams & hparams) {

    // 1. FFN Norm
    struct ggml_tensor * ffn_norm = ggml_rms_norm(ctx, cur, hparams.f_norm_eps);
    ffn_norm = ggml_mul(ctx, ffn_norm, layer.ffn_norm);

    // 2. SwiGLU FFN
    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, ffn_norm);
    gate = ggml_silu(ctx, gate);

    struct ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up, ffn_norm);

    struct ggml_tensor * ffn_out = ggml_mul(ctx, gate, up);

    // 3. Down 投影
    return ggml_mul_mat(ctx, layer.ffn_down, ffn_out);
}
```

## GGUF 模型格式

### 文件结构

```
GGUF 文件格式：
┌────────────────────────────────┐
│ Header                         │
│  - magic: "GGUF"               │
│  - version: 3                  │
│  - tensor_count: N             │
│  - metadata_kv_count: M        │
├────────────────────────────────┤
│ Metadata KV                    │
│  - "general.name": "LLaMA"     │
│  - "llama.embedding_length": 4096
│  - "llama.block_count": 32     │
│  - ...                         │
├────────────────────────────────┤
│ Tensor Info                    │
│  - name: "token_embd.weight"   │
│  - n_dims: 2                   │
│  - dims: [4096, 32000]         │
│  - type: Q4_K                  │
│  - offset: ...                 │
├────────────────────────────────┤
│ Tensor Data                    │
│  - 权重数据                     │
└────────────────────────────────┘
```

### 加载模型

```cpp
// 文件：src/llama.cpp

bool llama_model_load(const char * path, llama_model & model) {
    // 1. 初始化 GGUF 上下文
    struct gguf_context * ctx_gguf = gguf_init_from_file(path, ...);

    // 2. 读取元数据
    model.hparams.n_embd = gguf_get_val_u32(ctx_gguf, "llama.embedding_length");
    model.hparams.n_layer = gguf_get_val_u32(ctx_gguf, "llama.block_count");
    // ...

    // 3. 加载张量
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ...;
        model.tensors[name] = tensor;
    }

    return true;
}
```

## 不同架构对比

| 架构 | Norm 位置 | FFN 类型 | 注意力类型 |
|------|-----------|----------|------------|
| LLaMA | Pre-norm | SwiGLU | GQA |
| Mistral | Pre-norm | SwiGLU | GQA + Sliding Window |
| Qwen | Pre-norm | SwiGLU | MHA |
| GPT-2 | Post-norm | GELU | MHA |
| BERT | Post-norm | GELU | MHA |

## 练习

1. 阅读 `src/llama.cpp`，理解完整的计算图构建流程
2. 比较不同模型架构的计算图差异
3. 尝试修改计算图，添加自定义操作

## 下一步

完成本节后，请继续学习 [添加新模型](02-add-new-model.md)。
