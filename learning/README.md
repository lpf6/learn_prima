# Prima.cpp 学习教程

本教程面向希望深入理解 prima.cpp 源码、参与项目维护和开发、处理多平台兼容性问题的开发者。

## 学习目标

完成本教程后，你将能够：

1. 理解 CUDA 编程模型和优化技术
2. 处理不同 GPU 架构（SM 75, 80, 86, 89, 90）的兼容性
3. 理解并实现各种量化算法
4. 阅读和优化核心 CUDA 内核
5. 处理 CPU SIMD 指令集兼容性（x86_64 AVX/AVX2/AVX-512，ARM NEON）
6. 为项目添加新的模型架构支持
7. 使用性能分析工具优化内核
8. 调试 CUDA 程序和性能问题
9. 实现多后端兼容性和分布式推理

## 各阶段能力目标

| 阶段 | 学完后能做什么 | 实际工作示例 |
|------|----------------|--------------|
| **1. CUDA 基础** | 阅读 CUDA 内核代码、理解基础优化、调试简单问题 | 理解 `warp_reduce_sum`、修改内核参数 |
| **2. GPU 架构** | 理解架构差异、修改架构支持、诊断架构错误 | 添加新架构编译支持、解决 "no device code" 错误 |
| **3. 量化技术** | 选择量化类型、使用量化工具、阅读量化代码 | 量化模型、估算内存占用、理解反量化实现 |
| **4. CUDA 内核** | 阅读复杂内核、修改内核参数、添加简单算子 | 理解 MMQ、Flash Attention，使用 Nsight 分析 |
| **5. CPU SIMD** | 阅读 SIMD 代码、理解条件编译、修改 SIMD 优化 | 添加 SIMD 实现、检测 CPU 特性 |
| **6. LLM 架构** | 添加新模型、理解计算图、调试模型问题 | 实现新模型转换脚本、构建计算图 |
| **7. 后端兼容** | 理解后端架构、处理后端问题、跨平台调试 | 诊断后端问题、处理后端选择 |
| **8. llama.cpp 优化** | 理解 mmap、KV Cache、批处理、Flash Attention | 优化推理性能、配置 GPU 卸载 |
| **9. prima.cpp 优化** | 理解分布式推理、Piped-Ring 并行、异构调度 | 配置多设备推理、优化负载分配 |
| **9. prima.cpp 优化** | 理解分布式推理、Piped-Ring 并行、异构调度 | 配置多设备推理、优化负载分配 |

## 教程结构

```
learning/
├── README.md                    # 本文件 - 总览
├── KNOWLEDGE_MAP.md             # 知识脉络图
├── 00-prerequisites/            # 第零阶段：前置知识
│   ├── README.md
│   ├── 01-cpp-fundamentals.md   # C++ 编程基础
│   ├── 02-parallel-computing.md # 并行计算基础
│   └── 03-dl-basics.md          # 深度学习基础
├── 01-cuda-basics/              # 第一阶段：CUDA 基础
│   ├── README.md
│   ├── 01-thread-hierarchy.md
│   ├── 02-memory-model.md
│   ├── 03-synchronization.md
│   ├── 04-practice-exercises.md
│   ├── 05-cuda-streams.md       # CUDA 流和事件
│   ├── 06-performance-tools.md  # 性能分析工具
│   └── 07-debugging.md          # CUDA 调试技术
├── 02-gpu-architecture/         # 第二阶段：GPU 架构
│   ├── README.md
│   ├── 01-compute-capability.md
│   ├── 02-arch-specific-features.md
│   └── 03-code-adaptation.md
├── 03-quantization/             # 第三阶段：量化技术
│   ├── README.md
│   ├── 01-quant-basics.md
│   ├── 02-quant-types.md
│   └── 03-quant-implementation.md
├── 04-cuda-kernels/             # 第四阶段：核心 CUDA 内核
│   ├── README.md
│   ├── 01-mmq.md
│   ├── 02-flash-attention.md
│   └── 03-other-kernels.md
├── 05-cpu-simd/                 # 第五阶段：CPU SIMD 优化
│   ├── README.md
│   ├── 01-x86-simd.md
│   └── 02-arm-neon.md
├── 06-llm-architecture/         # 第六阶段：LLM 架构
│   ├── README.md
│   ├── 01-model-structure.md
│   └── 02-add-new-model.md
├── 07-backend-compatibility/    # 第七阶段：后端兼容性
│   ├── README.md
│   └── 01-multi-backend.md
├── 08-llama-optimizations/      # 第八阶段：LLaMA.cpp 优化实现
│   └── README.md                # mmap、KV Cache、批处理、Flash Attention
├── 09-prima-optimizations/      # 第九阶段：Prima.cpp 优化实现
│   └── README.md                # 分布式推理优化详解
├── 10-tilelang-optimizations/   # 第十阶段：TileLang 优化
│   └── README.md                # DSL、编译器架构、多后端支持
├── 11-projects/                 # 实践项目模块
│   ├── README.md
│   ├── 01-inference-service.md  # 构建推理服务
│   ├── 02-custom-kernel.md      # 自定义内核开发
│   ├── 03-model-quantization.md # 模型量化实践
│   └── 04-performance-benchmark.md # 性能基准测试
├── 12-tools-ecosystem/          # 工具与生态模块
│   ├── README.md
│   ├── 01-build-tools.md        # 构建工具
│   ├── 02-debugging-tools.md    # 调试工具
│   ├── 03-profiling-tools.md    # 性能分析工具
│   └── 04-community-resources.md # 社区资源
└── 13-troubleshooting/          # 故障排查模块
    ├── README.md
    ├── 01-common-errors.md      # 常见错误
    ├── 02-debugging-strategies.md # 调试策略
    ├── 03-performance-debugging.md # 性能调试
    └── 04-faq.md                # 常见问题 FAQ
```

## 学习路线图

```
Week 0-1:  第零阶段 - 前置知识（可选，根据需要）
           ├── C++ 编程基础
           ├── 并行计算基础
           └── 深度学习基础

Week 1-2:  第一阶段 - CUDA 基础
           ├── 线程层次结构
           ├── 内存模型
           ├── 同步机制
           ├── CUDA 流和事件
           ├── 性能分析工具
           └── CUDA 调试技术

Week 3-4:  第二阶段 - GPU 架构
           ├── 计算能力版本
           ├── 架构特定特性
           └── 代码适配技术

Week 5-6:  第三阶段 - 量化技术
           ├── 量化基础概念
           ├── 量化类型详解
           └── CUDA 量化实现

Week 7-10: 第四阶段 - 核心 CUDA 内核
           ├── 矩阵乘法 (MMQ)
           ├── Flash Attention
           └── 其他核心算子

Week 11-12: 第五阶段 - CPU SIMD 优化
            ├── x86 SIMD (AVX/AVX2/AVX-512)
            └── ARM NEON

Week 13-15: 第六阶段 - LLM 架构
            ├── 模型结构理解
            ├── 添加新模型
            └── 计算图构建

Week 16:   第七阶段 - 后端兼容性
           └── 多后端适配

Week 17-18: 第八阶段 - LLaMA.cpp 优化
            ├── mmap 内存映射
            ├── KV Cache 优化
            ├── 批处理技术
            └── Flash Attention

Week 19-20: 第九阶段 - Prima.cpp 优化
            ├── 分布式推理
            ├── Piped-Ring 并行
            └── 异构调度

Week 21-22: 第十阶段 - TileLang 优化
            ├── DSL 编程
            ├── 编译器架构
            └── 多后端支持

Week 23+:  实践项目与故障排查
           ├── 实践项目模块
           ├── 工具与生态
           └── 故障排查
```

## 项目代码结构

在学习过程中，你需要参考以下核心代码：

```
prima.cpp/
├── ggml/
│   ├── src/
│   │   ├── ggml-cuda/        # CUDA 内核实现
│   │   │   ├── common.cuh    # 公共工具函数
│   │   │   ├── mmq.cu        # 矩阵乘法
│   │   │   ├── fattn.cu      # Flash Attention
│   │   │   ├── quantize.cu   # 量化
│   │   │   └── ...
│   │   ├── ggml-aarch64.c    # ARM64 优化
│   │   ├── ggml-quants.c     # 量化实现
│   │   └── llamafile/
│   │       └── sgemm.cpp     # SIMD 优化 SGEMM
│   └── include/
│       ├── ggml.h            # 核心张量库
│       └── ggml-cuda.h       # CUDA 接口
├── src/
│   └── llama.cpp             # 模型实现
└── convert_hf_to_gguf.py     # 模型转换
```

## 如何使用本教程

### 📚 学习流程

1. **确定你的路径**：查看 [个性化学习路径](learning-paths.md)，找到适合你的方案
2. **按顺序学习**：每个阶段都建立在前一阶段的基础上
3. **边学边练**：每个章节都有实践练习
4. **自我检测**：完成章节后做 [形成性评估](01-cuda-basics/formative-assessments.md)
5. **阅读源码**：教程会引用项目中的实际代码
6. **动手修改**：尝试修改代码并观察效果
7. **实践项目**：完成 [综合项目](11-projects/00-capstone-project.md) 巩固所学

### 🎯 快速开始

**新手**：从 [第零阶段](00-prerequisites/README.md) 开始
**有经验**：直接 [第一阶段](01-cuda-basics/README.md)
**专家**：查看 [学习路径](learning-paths.md) 选择快速通道

### 📊 学习支持

- **学习策略**：[高效学习方法](learning-strategies.md)（费曼技巧、康奈尔笔记、番茄工作法）
- **学习路径**：[个性化方案](learning-paths.md)（学生/工程师/转行者）
- **自我检测**：[形成性评估](01-cuda-basics/formative-assessments.md)（每章测验）
- **实践项目**：[综合项目](11-projects/00-capstone-project.md)（端到端实战）
- **故障排查**：[常见问题](13-troubleshooting/README.md)（错误解决）

## 知识脉络图

为了帮助你建立完整的知识体系，请参考 [知识脉络图](KNOWLEDGE_MAP.md)，其中包含：

- 核心知识架构图
- 知识点依赖关系
- CUDA 与 CPU SIMD 的对应关系
- 学习路径建议
- 常见问题与知识点关联

## 新增内容（2024 版）

本教程于 2024 年进行了全面升级，新增以下内容：

### 🎓 教育增强

- ✅ **形成性评估**：每章自测题，及时检验学习效果
- ✅ **个性化路径**：5 种身份、3 种速度的学习方案
- ✅ **学习策略**：元认知技能、时间管理、刻意练习方法
- ✅ **综合项目**：端到端的推理引擎实战项目

### 📚 内容完善

- ✅ **CUDA 基础扩展**：增加 CUDA 流、性能工具、调试技术
- ✅ **后端详解**：Metal、Vulkan、ROCm 后端详细说明
- ✅ **实践模块**：4 个完整实践项目 + 评估标准
- ✅ **工具生态**：构建、调试、性能分析工具全涵盖

### 🎯 学习支持

- ✅ **学习社区**：Discord、GitHub、微信群
- ✅ **进度追踪**：学习日志、里程碑检查
- ✅ **职业指导**：技能映射、面试准备、薪资参考

## 开始学习

### 第一步：选择你的路径

访问 [个性化学习路径](learning-paths.md) 找到适合你的方案

### 第二步：开始第一阶段

从 [CUDA 基础](01-cuda-basics/README.md) 开始学习

### 第三步：自我检测

完成章节后做 [形成性评估](01-cuda-basics/formative-assessments.md)

### 第四步：实践项目

用 [综合项目](11-projects/00-capstone-project.md) 巩固所学

---

**祝你学习顺利！** 🚀

有任何问题，欢迎在学习社区讨论！
