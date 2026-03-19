# 深度学习基础

## 概述

本章介绍深度学习的基础知识，特别是与 LLM（大语言模型）推理相关的内容。理解这些概念对于学习后续的模型架构和优化技术至关重要。

## 1. 神经网络基础

### 1.1 感知机

```
输入       权重       输出
x1 ──────→ w1 ─┐
               ├→ Σ → 激活函数 → y
x2 ──────→ w2 ─┘

计算：y = f(w1*x1 + w2*x2 + b)
其中：
- x: 输入
- w: 权重
- b: 偏置
- f: 激活函数
```

### 1.2 多层感知机（MLP）

```
输入层      隐藏层      输出层
  ○           ○           ○
  ○           ○           ○
  ○ ─────→   ○ ─────→   ○
  ○           ○
  ○           ○
  
前向传播：
h1 = f(W1 * x + b1)
h2 = f(W2 * h1 + b2)
y = W3 * h2 + b3
```

### 1.3 常见激活函数

```python
# Sigmoid
σ(x) = 1 / (1 + exp(-x))
# 范围：(0, 1)
# 问题：梯度消失

# ReLU (Rectified Linear Unit)
ReLU(x) = max(0, x)
# 范围：[0, ∞)
# 优点：计算简单，梯度不消失

# GELU (Gaussian Error Linear Unit)
GELU(x) = x * Φ(x)
# Transformer 常用激活函数

# SiLU / Swish
SiLU(x) = x * σ(x)
# SwiGLU 中使用
```

## 2. Transformer 架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Transformer                            │
├─────────────────────────────────────────────────────────────┤
│  输入 → Embedding → Positional Encoding                     │
│                      ↓                                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Decoder Block × N                      │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │ 1. Self-Attention                            │  │    │
│  │  │    - Q, K, V 投影                            │  │    │
│  │  │    - Attention(Q, K, V) = softmax(QK^T/√d)V │  │    │
│  │  │    - 多头注意力 (Multi-Head)                 │  │    │
│  │  ├──────────────────────────────────────────────┤  │    │
│  │  │ 2. Add & Norm                                │  │    │
│  │  │    - 残差连接                                │  │    │
│  │  │    - LayerNorm                               │  │    │
│  │  ├──────────────────────────────────────────────┤  │    │
│  │  │ 3. Feed-Forward Network (FFN)                │  │    │
│  │  │    - MLP: Linear → Activation → Linear       │  │    │
│  │  │    - SwiGLU 变体                             │  │    │
│  │  ├──────────────────────────────────────────────┤  │    │
│  │  │ 4. Add & Norm                                │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────┘    │
│                      ↓                                      │
│                   输出 → Logits                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 自注意力机制（Self-Attention）

```python
# 注意力计算
def attention(Q, K, V):
    # Q, K, V: [batch, seq_len, head_dim]
    
    # 1. 计算注意力分数
    scores = Q @ K.transpose(-2, -1)  # [batch, seq_len, seq_len]
    
    # 2. 缩放
    scores = scores / sqrt(head_dim)
    
    # 3. Softmax
    attn_weights = softmax(scores, dim=-1)
    
    # 4. 加权求和
    output = attn_weights @ V
    
    return output

# 数学公式：
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### 2.3 多头注意力（Multi-Head Attention）

```python
# 多头注意力
def multi_head_attention(Q, K, V, num_heads):
    # 分割成多个头
    Q_heads = split_heads(Q, num_heads)  # [batch, heads, seq_len, head_dim]
    K_heads = split_heads(K, num_heads)
    V_heads = split_heads(V, num_heads)
    
    # 每个头独立计算注意力
    outputs = []
    for i in range(num_heads):
        out = attention(Q_heads[i], K_heads[i], V_heads[i])
        outputs.append(out)
    
    # 拼接所有头的输出
    output = concat_heads(outputs)
    
    return output

# 优势：
# - 同时关注不同位置的信息
# - 学习不同的表示子空间
```

### 2.4 位置编码（Positional Encoding）

```python
# 绝对位置编码（Transformer）
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

# 旋转位置编码（RoPE，LLaMA 使用）
def apply_rope(x, position_ids):
    # x: [batch, seq_len, head_dim]
    # 将位置信息编码到 Q, K 中
    for i in range(0, head_dim, 2):
        theta = position_ids / (10000 ** (i / head_dim))
        x[..., i] = x[..., i] * cos(theta) - x[..., i+1] * sin(theta)
        x[..., i+1] = x[..., i] * sin(theta) + x[..., i+1] * cos(theta)
    return x
```

## 3. LLM 架构详解

### 3.1 LLaMA 架构

```
LLaMA Model
├── Embedding (词嵌入)
│   └── Token → Vector [hidden_size]
├── Layers × num_layers (32/40/80...)
│   ├── Attention
│   │   ├── RMSNorm (预归一化)
│   │   ├── Q, K, V 投影
│   │   ├── RoPE (旋转位置编码)
│   │   ├── Attention 计算
│   │   └── Output 投影
│   └── FFN (SwiGLU)
│       ├── RMSNorm
│       ├── Gate 投影
│       ├── Up 投影
│       ├── Down 投影
│       └── 残差连接
└── Final Norm
    └── RMSNorm → Output 投影 → Logits
```

### 3.2 关键组件详解

**1. RMSNorm（Root Mean Square Layer Normalization）**

```python
# LayerNorm
def layer_norm(x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return (x - mean) / sqrt(var + eps) * weight + bias

# RMSNorm (LLaMA 使用)
def rms_norm(x):
    # 只计算均方根，不减去均值
    rms = sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight
```

**2. SwiGLU 激活函数**

```python
# 标准 FFN
def ffn(x):
    return W2 * gelu(W1 * x)

# SwiGLU (LLaMA 使用)
def swiglu_ffn(x):
    gate = W_gate * x
    up = W_up * x
    # SiLU 激活
    gate = silu(gate)  # silu(x) = x * sigmoid(x)
    # 门控机制
    return W_down * (gate * up)
```

**3. Grouped-Query Attention (GQA)**

```python
# Multi-Query Attention (MQA)
# - 所有头共享一组 K, V
# - 推理快，质量略降

# Grouped-Query Attention (GQA)
# - 将查询头分组，每组共享 K, V
# - 平衡质量和速度

def gqa_attention(Q, K, V, num_q_heads, num_kv_heads):
    # Q: [batch, seq_len, num_q_heads * head_dim]
    # K, V: [batch, seq_len, num_kv_heads * head_dim]
    
    # 将 Q 分组，每组对应一个 K, V
    groups = num_q_heads // num_kv_heads
    
    # 重复 K, V 以匹配 Q 的数量
    K_repeated = repeat_kv(K, groups)
    V_repeated = repeat_kv(V, groups)
    
    # 计算注意力
    output = attention(Q, K_repeated, V_repeated)
    
    return output
```

## 4. 推理过程

### 4.1 自回归生成

```
生成过程：
1. 输入: "The"
   ↓
   模型预测: "quick" (概率最高)
   
2. 输入: "The quick"
   ↓
   模型预测: "brown"
   
3. 输入: "The quick brown"
   ↓
   模型预测: "fox"
   
...
重复直到生成 EOS (End of Sequence) token
```

### 4.2 KV Cache

```
问题：每次重复计算之前的 K, V

解决方案：缓存 K, V
第 1 步：计算 K1, V1 并缓存
第 2 步：计算 K2, V2，使用缓存的 K1, V1
第 3 步：计算 K3, V3，使用缓存的 K1, V1, K2, V2

内存占用：
KV Cache = 2 × num_layers × seq_len × num_heads × head_dim × precision

示例（LLaMA-7B, seq_len=4096）:
FP16: 2 × 32 × 4096 × 32 × 128 × 2 bytes ≈ 4 GB
```

### 4.3 批处理（Batching）

```python
# 无批处理（慢）
for request in requests:
    output = model.generate(request)

# 批处理（快）
batch = combine_requests(requests)
outputs = model.generate(batch)
results = split_outputs(outputs)

# 关键：填充到相同长度
batch_input = pad_to_longest(requests)
```

## 5. 量化基础

### 5.1 量化的动机

```
精度对比：
- FP32: 32-bit 浮点，4 bytes/参数
- FP16: 16-bit 浮点，2 bytes/参数
- INT8:  8-bit 整数，1 bytes/参数
- INT4:  4-bit 整数，0.5 bytes/参数

7B 模型大小：
FP16: 14 GB
Q4_0: 3.5 GB (压缩 4x)
```

### 5.2 线性量化

```python
# 对称线性量化
def quantize(x, bits=4):
    # 找到范围
    max_val = abs(x).max()
    
    # 量化
    scale = max_val / (2 ** (bits - 1) - 1)
    x_int = round(x / scale)
    
    # 存储
    return x_int.clamp(-2**(bits-1), 2**(bits-1) - 1), scale

# 反量化
def dequantize(x_int, scale):
    return x_int * scale
```

### 5.3 块量化

```python
# 将权重分成块，每块独立量化
def block_quantize(weight, block_size=32):
    # 分成块
    blocks = weight.reshape(-1, block_size)
    
    # 每块计算 scale
    scales = blocks.abs().max(dim=1).values
    
    # 每块量化
    quantized = []
    for i, block in enumerate(blocks):
        q_block = quantize(block, bits=4)
        quantized.append(q_block)
    
    return quantized, scales
```

## 6. 性能优化概念

### 6.1 计算瓶颈分析

```
LLM 推理瓶颈：
┌─────────────────────────────────────────────────────────────┐
│ Prefill 阶段（处理 prompt）                                  │
│ - 计算密集型                                                 │
│ - 并行处理所有 token                                          │
│ - 瓶颈：计算能力（TFLOPS）                                   │
├─────────────────────────────────────────────────────────────┤
│ Decoding 阶段（生成 token）                                  │
│ - 内存密集型                                                 │
│ - 每次生成一个 token                                          │
│ - 瓶颈：内存带宽（GB/s）                                     │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 优化技术概览

```
1. 算子融合
   - 将多个算子合并为一个 CUDA 内核
   - 减少内存访问
   - 示例：Q + K + V 投影合并

2. 内存优化
   - PagedAttention（vLLM）
   - 内存池
   - 量化 KV Cache

3. 并行策略
   - Tensor Parallelism（张量并行）
   - Pipeline Parallelism（流水线并行）
   - Data Parallelism（数据并行）

4. 投机解码
   - 小模型生成候选
   - 大模型验证
   - 加速 2-4x
```

## 7. 实际代码示例

### 7.1 Transformer 前向传播（简化版）

```python
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLUFFN(hidden_size)
    
    def forward(self, x, mask=None, cache=None):
        # 自注意力
        h = self.attn_norm(x)
        h = self.attention(h, mask=mask, cache=cache)
        x = x + h  # 残差连接
        
        # FFN
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h  # 残差连接
        
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, tokens, cache=None):
        # 嵌入
        x = self.tok_emb(tokens)
        
        # 逐层处理
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            x = layer(x, cache=layer_cache)
        
        # 输出
        x = self.norm(x)
        logits = self.out(x)
        
        return logits
```

### 7.2 注意力计算（带 KV Cache）

```python
def attention_with_cache(q, k, v, cache=None):
    """
    q, k, v: [batch, heads, seq_len, head_dim]
    cache: {'k': ..., 'v': ...} 或 None
    """
    if cache is not None:
        # 拼接缓存的 K, V
        k = torch.cat([cache['k'], k], dim=2)
        v = torch.cat([cache['v'], v], dim=2)
        
        # 更新缓存
        cache['k'] = k
        cache['v'] = v
    
    # 计算注意力
    scores = q @ k.transpose(-2, -1) / sqrt(q.size(-1))
    
    # Causal mask（解码阶段）
    if q.size(2) == 1:  # 单个 token 生成
        mask = None
    else:
        mask = torch.triu(
            torch.ones(q.size(2), k.size(2)),
            diagonal=1
        ) * float('-inf')
    
    if mask is not None:
        scores = scores + mask
    
    attn_weights = softmax(scores, dim=-1)
    output = attn_weights @ v
    
    return output
```

## 练习

1. 手动实现一个简单的 Transformer 层（不含注意力）

2. 计算 LLaMA-7B 在 seq_len=4096 时的 KV Cache 大小（FP16 和 Q8_0）

3. 实现 RoPE 位置编码的前向传播

4. 分析为什么 SwiGLU 比 ReLU 效果更好

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LLaMA Paper](https://arxiv.org/abs/2302.13971)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [HuggingFace Course](https://huggingface.co/learn/nlp-course)
