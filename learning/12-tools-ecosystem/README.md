# 工具与生态模块

## 概述

本模块介绍 Prima.cpp 开发中使用的各种工具、构建系统和社区资源。

## 章节目录

### 1. 构建工具

- CMake 配置详解
- 编译选项和标志
- 依赖管理
- 跨平台构建

### 2. 调试工具

- GDB 和 CUDA-GDB
- 内存检测工具
- 日志和追踪
- IDE 集成

### 3. 性能分析工具

- Nsight Compute/Systems
- rocprof (ROCm)
- perf (Linux)
- 火焰图生成

### 4. 社区资源

- 文档导航
- 社区贡献指南
- Issue 提交规范
- 学习资源推荐

## 工具链总览

```
开发工具链:
┌─────────────────────────────────────────────────────────────┐
│  开发阶段              工具                                   │
├─────────────────────────────────────────────────────────────┤
│  代码编辑              VS Code, CLion, Vim                   │
│  构建系统              CMake, Make, Ninja                    │
│  编译器                nvcc, hipcc, clang                    │
│  调试工具              GDB, cuda-gdb, compute-sanitizer      │
│  性能分析              Nsight, rocprof, perf                 │
│  版本控制              Git, GitHub                           │
│  持续集成              GitHub Actions, Jenkins               │
│  文档工具              Doxygen, Sphinx, Markdown             │
└─────────────────────────────────────────────────────────────┘
```

## 开始学习

请从以下章节开始：

1. [构建工具](01-build-tools.md)
2. [调试工具](02-debugging-tools.md)
3. [性能分析工具](03-profiling-tools.md)
4. [社区资源](04-community-resources.md)

## 工具安装指南

### Ubuntu/Debian

```bash
# CUDA Toolkit
wget https://developer.nvidia.com/cuda-downloads
# 按照官网指引安装

# CMake
sudo apt install cmake

# GDB
sudo apt install gdb

# perf
sudo apt install linux-tools-generic
```

### macOS

```bash
# Xcode Command Line Tools
xcode-select --install

# Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# CMake
brew install cmake

# Vulkan SDK
brew install vulkan-sdk
```

### Windows

```powershell
# CUDA Toolkit
# 从 NVIDIA 官网下载安装

# Visual Studio
# 安装 "Desktop development with C++"

# CMake
choco install cmake

# Vulkan SDK
choco install vulkan-sdk
```

## 参考资料

- [CMake Documentation](https://cmake.org/documentation/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [GCC Documentation](https://gcc.gnu.org/onlinedocs/)
- [GDB Documentation](https://sourceware.org/gdb/current/onlinedocs/)
