# 第六阶段：LLM 架构

## 概述

本阶段深入 LLM 的架构设计，以及如何在 Prima.cpp 中添加新的模型支持。

## 能力目标

完成本阶段后，你将能够：

### 能做什么

| 能力 | 具体表现 | 相关代码 |
|------|----------|----------|
| **添加新模型** | 完整实现新模型架构支持 | `llama.cpp`, `convert_hf_to_gguf.py` |
| **理解计算图** | 阅读 GGML 计算图构建代码 | `llama.cpp` |
| **修改模型结构** | 调整现有模型的实现 | 维护工作 |
| **调试模型问题** | 定位模型加载和推理错误 | 调试排错 |

### 还不能做什么

- 设计全新的模型架构
- 实现复杂的模型变体
- 处理极端边缘情况

### 实际工作示例

学完本阶段后，你可以：

1. **添加新模型支持**
```python
# 你能实现新模型的转换脚本
@Model.register("MyModelForCausalLM")
class MyModel(Model):
    model_arch = gguf.MODEL_ARCH.MY_MODEL
    
    def set_gguf_parameters(self):
        self.gguf.add_uint32("my_model.embedding_length", ...)
        # ...
```

```cpp
// 你能实现 C++ 端的模型支持
struct ggml_cgraph * build_my_model(llama_context & lctx) {
    // 构建计算图
    // ...
}
```

2. **理解计算图构建**
```cpp
// 你能理解这种计算图构建代码
struct ggml_tensor * cur = ggml_rms_norm(ctx, inpL, eps);
cur = ggml_mul(ctx, cur, layer.attn_norm);

struct ggml_tensor * Q = ggml_mul_mat(ctx, layer.wq, cur);
struct ggml_tensor * K = ggml_mul_mat(ctx, layer.wk, cur);
struct ggml_tensor * V = ggml_mul_mat(ctx, layer.wv, cur);
// ...
```

3. **调试模型问题**
```bash
# 你能使用这些工具调试
./llama-cli -m model.gguf --verbose-prompt
./llama-eval-callback -m model.gguf
```

4. **修改现有模型**
```cpp
// 你能修改现有模型的实现
// 例如：添加新的注意力变体
struct ggml_tensor * build_attention_variant(...) {
    // 自定义注意力实现
}
```

## 章节目录

1. [模型结构理解](01-model-structure.md)
2. [添加新模型](02-add-new-model.md)

## 预计学习时间

2-3 周

## 开始学习

请从 [01-model-structure.md](01-model-structure.md) 开始。
