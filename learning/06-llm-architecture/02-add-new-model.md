# 6.2 添加新模型

## 概述

本节详细介绍如何为 Prima.cpp 添加新的模型架构支持。

## 添加新模型的步骤

```
步骤 1: 转换模型到 GGUF 格式
├── 定义 Model 类
├── 定义张量名称映射
└── 实现转换脚本

步骤 2: 定义 C++ 架构
├── 添加架构枚举
├── 定义张量布局
├── 加载模型参数
└── 创建张量

步骤 3: 构建计算图
├── 实现 build 函数
└── 注册构建函数
```

## 步骤 1: 转换模型到 GGUF

### 1.1 定义 Model 类

```python
# 文件：convert_hf_to_gguf.py

@Model.register("MyModelForCausalLM")
class MyModel(Model):
    model_arch = gguf.MODEL_ARCH.MY_MODEL

    def set_gguf_parameters(self):
        self.gguf.add_uint32("my_model.embedding_length", self.hparams["hidden_size"])
        self.gguf.add_uint32("my_model.block_count", self.hparams["num_hidden_layers"])
        self.gguf.add_uint32("my_model.attention.head_count", self.hparams["num_attention_heads"])
        # ...

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def write_tensors(self):
        for name, data in self.get_tensors():
            new_name = self.map_tensor_name(name)
            self.write_tensor(new_name, data)
```

### 1.2 定义张量名称

```python
# 文件：gguf-py/gguf/constants.py

class MODEL_ARCH(IntEnum):
    MY_MODEL = auto()

MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
    MODEL_ARCH.MY_MODEL: "my-model",
}

MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.MY_MODEL: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_DOWN,
    ],
}
```

### 1.3 映射张量名称

```python
# 文件：gguf-py/gguf/tensor_mapping.py

block_mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
    MODEL_TENSOR.ATTN_Q: (
        "model.layers.{bid}.self_attn.q_proj",
        "my_model.layers.{bid}.attention.q_proj",
    ),
    MODEL_TENSOR.ATTN_K: (
        "model.layers.{bid}.self_attn.k_proj",
        "my_model.layers.{bid}.attention.k_proj",
    ),
    # ...
}
```

## 步骤 2: 定义 C++ 架构

### 2.1 添加架构枚举

```cpp
// 文件：src/llama.cpp

enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_MISTRAL,
    LLM_ARCH_MY_MODEL,
    // ...
};

static const std::map<llm_arch, std::string> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,       "llama"       },
    { LLM_ARCH_MISTRAL,     "mistral"     },
    { LLM_ARCH_MY_MODEL,    "my-model"    },
};
```

### 2.2 定义张量布局

```cpp
// 文件：src/llama.cpp

static const std::map<llm_arch, std::map<llm_tensor, std::string>> LLM_TENSOR_NAMES = {
    {
        LLM_ARCH_MY_MODEL,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{bid}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{bid}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{bid}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{bid}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{bid}.attn_out" },
            { LLM_TENSOR_FFN_NORM,        "blk.{bid}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{bid}.ffn_gate" },
            { LLM_TENSOR_FFN_UP,          "blk.{bid}.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{bid}.ffn_down" },
        },
    },
};
```

### 2.3 加载模型参数

```cpp
// 文件：src/llama.cpp

static void llm_load_hparams(llama_model & model, ...) {
    model.hparams.n_embd = gguf_get_val_u32(ctx, "my_model.embedding_length");
    model.hparams.n_layer = gguf_get_val_u32(ctx, "my_model.block_count");
    model.hparams.n_head = gguf_get_val_u32(ctx, "my_model.attention.head_count");
    // ...
}
```

### 2.4 创建张量

```cpp
// 文件：src/llama.cpp

static bool llm_load_tensors(llama_model & model, ...) {
    model.layers.resize(model.hparams.n_layer);

    for (int i = 0; i < model.hparams.n_layer; ++i) {
        auto & layer = model.layers[i];

        layer.wq = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd});
        layer.wk = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_kv});
        layer.wv = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_kv});
        layer.wo = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.attn_norm = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.ffn_gate = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff});
        layer.ffn_up = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff});
        layer.ffn_down = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});

        layer.ffn_norm = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
    }

    return true;
}
```

## 步骤 3: 构建计算图

### 3.1 实现构建函数

```cpp
// 文件：src/llama.cpp

struct ggml_cgraph * build_my_model(llama_context & lctx) {
    const auto & model = lctx.model;
    const auto & hparams = model.hparams;

    struct ggml_context * ctx = ggml_init({...});
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    struct ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, lctx.embd);

    for (int il = 0; il < hparams.n_layer; ++il) {
        struct ggml_tensor * inpL = cur;

        // Attention
        cur = ggml_rms_norm(ctx, inpL, hparams.f_norm_eps);
        cur = ggml_mul(ctx, cur, model.layers[il].attn_norm);

        struct ggml_tensor * Q = ggml_mul_mat(ctx, model.layers[il].wq, cur);
        struct ggml_tensor * K = ggml_mul_mat(ctx, model.layers[il].wk, cur);
        struct ggml_tensor * V = ggml_mul_mat(ctx, model.layers[il].wv, cur);

        // ... Attention 计算 ...

        cur = ggml_mul_mat(ctx, model.layers[il].wo, KQV);
        cur = ggml_add(ctx, inpL, cur);

        // FFN
        struct ggml_tensor * inpFF = cur;

        cur = ggml_rms_norm(ctx, inpFF, hparams.f_norm_eps);
        cur = ggml_mul(ctx, cur, model.layers[il].ffn_norm);

        struct ggml_tensor * gate = ggml_mul_mat(ctx, model.layers[il].ffn_gate, cur);
        gate = ggml_silu(ctx, gate);

        struct ggml_tensor * up = ggml_mul_mat(ctx, model.layers[il].ffn_up, cur);

        cur = ggml_mul(ctx, gate, up);
        cur = ggml_mul_mat(ctx, model.layers[il].ffn_down, cur);

        cur = ggml_add(ctx, inpFF, cur);
    }

    cur = ggml_rms_norm(ctx, cur, hparams.f_norm_eps);
    cur = ggml_mul(ctx, cur, model.output_norm);
    cur = ggml_mul_mat(ctx, model.output, cur);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
```

### 3.2 注册构建函数

```cpp
// 文件：src/llama.cpp

struct ggml_cgraph * llama_build_graph(llama_context & lctx) {
    switch (lctx.model.arch) {
        case LLM_ARCH_LLAMA:
            return build_llama(lctx);
        case LLM_ARCH_MY_MODEL:
            return build_my_model(lctx);
        default:
            GGML_ASSERT(false && "unknown architecture");
    }
}
```

## 测试新模型

### 编译和转换

```bash
# 1. 转换模型
python convert_hf_to_gguf.py /path/to/my-model --outfile my-model-f16.gguf

# 2. 量化（可选）
./build/bin/llama-quantize my-model-f16.gguf my-model-q4_k.gguf Q4_K

# 3. 测试推理
./build/bin/llama-cli -m my-model-q4_k.gguf -p "Hello, world!" -n 128
```

## 常见问题

### 1. 张量名称不匹配

```bash
# 使用调试模式查看原始张量名称
python convert_hf_to_gguf.py /path/to/model --dry-run
```

### 2. 架构特定参数

确保所有必要参数都已正确加载。

### 3. 计算图调试

```bash
# 使用 eval-callback 示例调试计算图
./build/bin/llama-eval-callback -m model.gguf
```

## 练习

1. 选择一个 HuggingFace 模型，尝试为其添加支持
2. 阅读 `docs/development/HOWTO-add-model.md`，理解官方文档
3. 比较不同架构的实现差异

## 下一步

完成本阶段后，请继续学习 [第七阶段：后端兼容性](../07-backend-compatibility/README.md)。
