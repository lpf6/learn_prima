# Vulkan 后端详解

## 概述

Vulkan 是 Khronos Group 开发的跨平台图形和计算 API，提供底层硬件访问和高性能计算能力。Vulkan 后端使 Prima.cpp 能够在各种 GPU（NVIDIA、AMD、Intel、移动 GPU）上运行。

## 1. Vulkan 架构基础

### 1.1 Vulkan 与 CUDA 对比

| 特性 | CUDA | Vulkan |
|------|------|--------|
| **平台** | NVIDIA GPU | 跨平台（NVIDIA/AMD/Intel/Mobile） |
| **API 层级** | 高级计算 API | 底层硬件 API |
| **编程语言** | CUDA C++ | GLSL/HLSL/SPIR-V |
| **内存管理** | 自动/手动 | 完全手动 |
| **命令提交** | 隐式 | 显式命令缓冲 |
| **同步** | 隐式同步 | 显式同步 |

### 1.2 Vulkan 计算管线

```
Vulkan 计算管线流程:
┌─────────────────────────────────────────────────────────────┐
│  1. 创建实例 (Instance)                                      │
│     ↓                                                        │
│  2. 选择物理设备 (Physical Device)                           │
│     ↓                                                        │
│  3. 创建逻辑设备 (Logical Device)                            │
│     ↓                                                        │
│  4. 创建命令池 (Command Pool)                                │
│     ↓                                                        │
│  5. 创建命令缓冲 (Command Buffer)                            │
│     ↓                                                        │
│  6. 创建缓冲区 (Buffer) 和图像 (Image)                       │
│     ↓                                                        │
│  7. 创建描述符集 (Descriptor Set)                            │
│     ↓                                                        │
│  8. 创建管线 (Pipeline)                                      │
│     ↓                                                        │
│  9. 记录命令 (Record Commands)                               │
│     ↓                                                        │
│  10. 提交到队列 (Submit to Queue)                            │
│     ↓                                                        │
│  11. 等待完成 (Wait for Completion)                          │
└─────────────────────────────────────────────────────────────┘
```

## 2. Vulkan 计算着色器

### 2.1 GLSL 计算着色器基础

```glsl
// 文件：compute_shader.glsl

#version 450

// 指定工作组大小
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// 绑定存储缓冲
layout(std430, binding = 0) buffer InputBuffer {
    float input_data[];
} input_buffer;

layout(std430, binding = 1) buffer OutputBuffer {
    float output_data[];
} output_buffer;

// Uniform 参数
layout(binding = 2) uniform Params {
    int width;
    int height;
    float scale;
} params;

// 主函数
void main() {
    // 计算全局 ID
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    
    // 边界检查
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    uint idx = y * params.width + x;
    
    // 计算
    output_buffer.output_data[idx] = 
        input_buffer.input_data[idx] * params.scale;
}
```

### 2.2 共享内存优化

```glsl
// 使用共享内存的矩阵乘法
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, binding = 0) buffer MatrixA {
    float data[];
} matrix_a;

layout(std430, binding = 1) buffer MatrixB {
    float data[];
} matrix_b;

layout(std430, binding = 2) buffer MatrixC {
    float data[];
} matrix_c;

layout(binding = 3) uniform Dimensions {
    uint M, N, K;
} dims;

// 共享内存
shared float shared_a[16][16];
shared float shared_b[16][16];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint local_x = gl_LocalInvocationID.x;
    uint local_y = gl_LocalInvocationID.y;
    
    float sum = 0.0;
    
    // 分块计算
    for (uint k = 0; k < dims.K / 16; k++) {
        // 加载到共享内存
        shared_a[local_y][local_x] = matrix_a.data[
            row * dims.K + k * 16 + local_x];
        shared_b[local_y][local_x] = matrix_b.data[
            (k * 16 + local_y) * dims.N + col];
        
        // 同步
        barrier();
        memoryBarrierShared();
        
        // 计算
        for (uint i = 0; i < 16; i++) {
            sum += shared_a[local_y][i] * shared_b[i][local_x];
        }
        
        // 同步
        barrier();
        memoryBarrierShared();
    }
    
    // 存储结果
    matrix_c.data[row * dims.N + col] = sum;
}
```

### 2.3 量化计算着色器

```glsl
// Q4_0 量化矩阵乘法
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

struct BlockQ4_0 {
    float d;
    uint qs;  // 打包的 4-bit 值
};

layout(std430, binding = 0) buffer WeightsBuffer {
    BlockQ4_0 weights[];
} weights;

layout(std430, binding = 1) buffer InputBuffer {
    float input[];
} input;

layout(std430, binding = 2) buffer OutputBuffer {
    float output[];
} output;

layout(binding = 3) uniform Shape {
    uint rows, cols;
} shape;

// 反量化函数
float dequantize_q4_0(uint qs, int shift) {
    return ((shift == 0) ? (qs & 0x0F) : (qs >> 4)) - 8;
}

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    if (row >= shape.rows || col >= shape.cols) {
        return;
    }
    
    uint block_idx = row * (shape.cols / 32) + col / 32;
    
    float sum = 0.0;
    
    // 逐块计算
    for (uint i = 0; i < shape.cols / 32; i++) {
        BlockQ4_0 block = weights.weights[row * (shape.cols / 32) + i];
        
        // 反量化并计算
        for (uint j = 0; j < 16; j++) {
            float v0 = dequantize_q4_0(block.qs, 0) * block.d;
            float v1 = dequantize_q4_0(block.qs, 4) * block.d;
            
            sum += v0 * input.input[i * 32 + j * 2];
            sum += v1 * input.input[i * 32 + j * 2 + 1];
        }
    }
    
    output.output[row * shape.cols + col] = sum;
}
```

## 3. Vulkan 后端实现

### 3.1 Vulkan 初始化

```cpp
// 文件：ggml-vulkan.cpp

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

struct ggml_vk_context {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorSet descriptor_set;
};

// Vulkan 验证层
const std::vector<const char*> validation_layers = {
    "VK_LAYER_KHRONOS_validation"
};

// 设备扩展
const std::vector<const char*> device_extensions = {
    VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME,
};

// 初始化 Vulkan
VkResult ggml_vk_init(struct ggml_vk_context * ctx) {
    VkResult err;
    
    // 1. 创建实例
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Prima.cpp";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "GGML Vulkan";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;
    
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    
    // 启用验证层（调试模式）
#ifdef DEBUG
    create_info.enabledLayerCount = validation_layers.size();
    create_info.ppEnabledLayerNames = validation_layers.data();
#endif
    
    err = vkCreateInstance(&create_info, nullptr, &ctx->instance);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "Failed to create Vulkan instance\n");
        return err;
    }
    
    // 2. 选择物理设备
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, nullptr);
    
    if (device_count == 0) {
        fprintf(stderr, "No Vulkan devices found\n");
        return VK_ERROR_INCOMPATIBLE_DRIVER;
    }
    
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, devices.data());
    
    // 选择第一个支持计算的设备
    ctx->physical_device = devices[0];
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        
        // 优先选择独立显卡
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            ctx->physical_device = device;
            break;
        }
    }
    
    // 3. 创建逻辑设备
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        ctx->physical_device, &queue_family_count, nullptr);
    
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        ctx->physical_device, &queue_family_count, queue_families.data());
    
    // 查找计算队列
    int queue_family_index = -1;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_family_index = i;
            break;
        }
    }
    
    if (queue_family_index == -1) {
        fprintf(stderr, "No compute queue found\n");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_family_index;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    
    VkDeviceCreateInfo device_create_info = {};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.enabledExtensionCount = device_extensions.size();
    device_create_info.ppEnabledExtensionNames = device_extensions.data();
    
    err = vkCreateDevice(ctx->physical_device, &device_create_info, 
                        nullptr, &ctx->device);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "Failed to create Vulkan device\n");
        return err;
    }
    
    // 4. 获取计算队列
    vkGetDeviceQueue(ctx->device, queue_family_index, 0, &ctx->compute_queue);
    
    // 5. 创建命令池
    VkCommandPoolCreateInfo pool_create_info = {};
    pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_create_info.queueFamilyIndex = queue_family_index;
    
    err = vkCreateCommandPool(ctx->device, &pool_create_info, 
                             nullptr, &ctx->command_pool);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "Failed to create command pool\n");
        return err;
    }
    
    return VK_SUCCESS;
}
```

### 3.2 创建计算管线

```cpp
// 创建计算管线
VkResult ggml_vk_create_pipeline(
    struct ggml_vk_context * ctx,
    const char * shader_code,
    size_t shader_size) {
    
    VkResult err;
    
    // 1. 编译着色器
    VkShaderModuleCreateInfo shader_create_info = {};
    shader_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_create_info.codeSize = shader_size;
    shader_create_info.pCode = 
        reinterpret_cast<const uint32_t*>(shader_code);
    
    VkShaderModule shader_module;
    err = vkCreateShaderModule(ctx->device, &shader_create_info, 
                               nullptr, &shader_module);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "Failed to create shader module\n");
        return err;
    }
    
    // 2. 创建描述符集布局
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    VkDescriptorSetLayoutCreateInfo layout_create_info = {};
    layout_create_info.sType = 
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_create_info.bindingCount = 1;
    layout_create_info.pBindings = &binding;
    
    err = vkCreateDescriptorSetLayout(ctx->device, &layout_create_info,
                                      nullptr, &ctx->descriptor_set_layout);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "Failed to create descriptor set layout\n");
        return err;
    }
    
    // 3. 创建管线布局
    VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
    pipeline_layout_create_info.sType = 
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_create_info.setLayoutCount = 1;
    pipeline_layout_create_info.pSetLayouts = &ctx->descriptor_set_layout;
    
    err = vkCreatePipelineLayout(ctx->device, &pipeline_layout_create_info,
                                 nullptr, &ctx->pipeline_layout);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "Failed to create pipeline layout\n");
        return err;
    }
    
    // 4. 创建计算管线
    VkPipelineShaderStageCreateInfo stage_create_info = {};
    stage_create_info.sType = 
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_create_info.module = shader_module;
    stage_create_info.pName = "main";
    
    VkComputePipelineCreateInfo pipeline_create_info = {};
    pipeline_create_info.sType = 
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create_info.stage = stage_create_info;
    pipeline_create_info.layout = ctx->pipeline_layout;
    
    err = vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1,
                                   &pipeline_create_info, nullptr,
                                   &ctx->pipeline);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "Failed to create compute pipeline\n");
        return err;
    }
    
    // 清理
    vkDestroyShaderModule(ctx->device, shader_module, nullptr);
    
    return VK_SUCCESS;
}
```

### 3.3 缓冲区管理

```cpp
// 创建 Vulkan 缓冲区
VkResult ggml_vk_create_buffer(
    struct ggml_vk_context * ctx,
    VkBuffer * buffer,
    VkDeviceMemory * memory,
    size_t size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties) {
    
    VkResult err;
    
    // 1. 创建缓冲区
    VkBufferCreateInfo buffer_create_info = {};
    buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_create_info.size = size;
    buffer_create_info.usage = usage;
    buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    err = vkCreateBuffer(ctx->device, &buffer_create_info, 
                        nullptr, buffer);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "Failed to create buffer\n");
        return err;
    }
    
    // 2. 分配内存
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(ctx->device, *buffer, &mem_requirements);
    
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    
    // 查找内存类型
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &mem_properties);
    
    uint32_t memory_type_index = 0;
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((mem_requirements.memoryTypeBits & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) 
            == properties) {
            memory_type_index = i;
            break;
        }
    }
    
    alloc_info.memoryTypeIndex = memory_type_index;
    
    err = vkAllocateMemory(ctx->device, &alloc_info, nullptr, memory);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "Failed to allocate memory\n");
        return err;
    }
    
    // 3. 绑定内存到缓冲区
    err = vkBindBufferMemory(ctx->device, *buffer, *memory, 0);
    if (err != VK_SUCCESS) {
        fprintf(stderr, "Failed to bind buffer memory\n");
        return err;
    }
    
    return VK_SUCCESS;
}

// 上传数据到设备
void ggml_vk_upload_data(
    struct ggml_vk_context * ctx,
    VkBuffer device_buffer,
    void * host_data,
    size_t size) {
    
    // 创建临时 staging buffer
    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    
    ggml_vk_create_buffer(ctx, &staging_buffer, &staging_memory,
                         size,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    // 映射并拷贝数据
    void * mapped_data;
    vkMapMemory(ctx->device, staging_memory, 0, size, 0, &mapped_data);
    memcpy(mapped_data, host_data, size);
    vkUnmapMemory(ctx->device, staging_memory);
    
    // 创建命令缓冲
    VkCommandBufferAllocateInfo cmd_alloc_info = {};
    cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc_info.commandPool = ctx->command_pool;
    cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc_info.commandBufferCount = 1;
    
    vkAllocateCommandBuffers(ctx->device, &cmd_alloc_info, &ctx->command_buffer);
    
    // 记录命令
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(ctx->command_buffer, &begin_info);
    
    // 执行拷贝
    vkCmdCopyBuffer(ctx->command_buffer, staging_buffer, device_buffer,
                   0, nullptr, 1);
    
    vkEndCommandBuffer(ctx->command_buffer);
    
    // 提交命令
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &ctx->command_buffer;
    
    vkQueueSubmit(ctx->compute_queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->compute_queue);
    
    // 清理 staging buffer
    vkFreeMemory(ctx->device, staging_memory, nullptr);
    vkDestroyBuffer(ctx->device, staging_buffer, nullptr);
}
```

## 4. Vulkan 性能优化

### 4.1 异步计算

```cpp
// 使用多个队列实现异步计算
struct ggml_vk_async_context {
    VkQueue compute_queue;
    VkQueue transfer_queue;
    VkCommandPool compute_pool;
    VkCommandPool transfer_pool;
    VkFence compute_fence;
    VkFence transfer_fence;
};

void ggml_vk_async_compute(
    struct ggml_vk_async_context * ctx,
    VkBuffer input_buffer,
    VkBuffer output_buffer,
    uint32_t workgroup_count) {
    
    // 1. 异步传输数据
    VkCommandBuffer transfer_cmd;
    // ... 分配和记录传输命令
    
    VkSubmitInfo transfer_submit = {};
    transfer_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    transfer_submit.commandBufferCount = 1;
    transfer_submit.pCommandBuffers = &transfer_cmd;
    
    vkQueueSubmit(ctx->transfer_queue, 1, &transfer_submit, 
                 ctx->transfer_fence);
    
    // 2. 等待传输完成信号
    VkSemaphore transfer_complete;
    // ... 创建信号量
    
    // 3. 计算命令（等待传输完成）
    VkCommandBuffer compute_cmd;
    // ... 分配和记录计算命令
    
    VkSubmitInfo compute_submit = {};
    compute_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    compute_submit.waitSemaphoreCount = 1;
    compute_submit.pWaitSemaphores = &transfer_complete;
    compute_submit.pWaitDstStageMask = 
        &(VkPipelineStageFlags){VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    compute_submit.commandBufferCount = 1;
    compute_submit.pCommandBuffers = &compute_cmd;
    
    vkQueueSubmit(ctx->compute_queue, 1, &compute_submit, 
                 ctx->compute_fence);
}
```

### 4.2 内存优化

```cpp
// 使用专用视频内存
VkResult ggml_vk_create_device_buffer(
    struct ggml_vk_context * ctx,
    VkBuffer * buffer,
    VkDeviceMemory * memory,
    size_t size) {
    
    return ggml_vk_create_buffer(
        ctx, buffer, memory, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT  // 专用显存
    );
}

// 使用主机可见内存（用于频繁访问）
VkResult ggml_vk_create_host_buffer(
    struct ggml_vk_context * ctx,
    VkBuffer * buffer,
    VkDeviceMemory * memory,
    size_t size) {
    
    return ggml_vk_create_buffer(
        ctx, buffer, memory, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
}
```

### 4.3 批处理优化

```cpp
// 批处理多个计算任务
void ggml_vk_batch_compute(
    struct ggml_vk_context * ctx,
    struct ggml_tensor ** tensors,
    int num_tensors) {
    
    // 1. 分配命令缓冲
    VkCommandBuffer cmd_buffers[num_tensors];
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = ctx->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = num_tensors;
    
    vkAllocateCommandBuffers(ctx->device, &alloc_info, cmd_buffers);
    
    // 2. 记录所有命令
    for (int i = 0; i < num_tensors; i++) {
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        
        vkBeginCommandBuffer(cmd_buffers[i], &begin_info);
        
        // 绑定管线
        vkCmdBindPipeline(cmd_buffers[i], 
                         VK_PIPELINE_BIND_POINT_COMPUTE,
                         ctx->pipeline);
        
        // 绑定描述符集
        vkCmdBindDescriptorSets(cmd_buffers[i],
                               VK_PIPELINE_BIND_POINT_COMPUTE,
                               ctx->pipeline_layout,
                               0, 1, &tensors[i]->descriptor_set,
                               0, nullptr);
        
        // 分派计算
        vkCmdDispatch(cmd_buffers[i],
                     tensors[i]->grid_size.x,
                     tensors[i]->grid_size.y,
                     tensors[i]->grid_size.z);
        
        vkEndCommandBuffer(cmd_buffers[i]);
    }
    
    // 3. 批量提交
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = num_tensors;
    submit_info.pCommandBuffers = cmd_buffers;
    
    vkQueueSubmit(ctx->compute_queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->compute_queue);
}
```

## 5. Vulkan 调试

### 5.1 调试扩展

```cpp
// 启用调试扩展
VkDebugUtilsMessengerCreateInfoEXT debug_create_info = {};
debug_create_info.sType = 
    VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
debug_create_info.messageSeverity = 
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
debug_create_info.messageType = 
    VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
debug_create_info.pfnUserCallback = debug_callback;

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        fprintf(stderr, "Vulkan Validation: %s\n", pCallbackData->pMessage);
    }
    
    return VK_FALSE;
}
```

### 5.2 性能分析

```cpp
// 使用 Vulkan 时间戳进行性能分析
void profile_vulkan_compute(
    struct ggml_vk_context * ctx,
    VkCommandBuffer cmd_buffer) {
    
    // 1. 创建查询池
    VkQueryPoolCreateInfo query_pool_create_info = {};
    query_pool_create_info.sType = 
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_pool_create_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_pool_create_info.queryCount = 2;
    
    VkQueryPool query_pool;
    vkCreateQueryPool(ctx->device, &query_pool_create_info,
                     nullptr, &query_pool);
    
    // 2. 记录时间戳
    vkCmdResetQueryPool(cmd_buffer, query_pool, 0, 2);
    vkCmdWriteTimestamp(cmd_buffer,
                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       query_pool, 0);
    
    // ... 计算命令 ...
    
    vkCmdWriteTimestamp(cmd_buffer,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       query_pool, 1);
    
    // 3. 获取结果
    uint64_t timestamps[2];
    vkGetQueryPoolResults(ctx->device, query_pool, 0, 2,
                         sizeof(timestamps), timestamps,
                         sizeof(uint64_t),
                         VK_QUERY_RESULT_64_BIT);
    
    // 4. 转换为时间
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx->physical_device, &props);
    
    double elapsed_ns = (timestamps[1] - timestamps[0]) * 
                       props.limits.timestampPeriod;
    
    printf("Compute time: %.2f ms\n", elapsed_ns / 1e6);
    
    // 清理
    vkDestroyQueryPool(ctx->device, query_pool, nullptr);
}
```

## 6. 实际案例

### 6.1 Flash Attention Vulkan 实现

```glsl
// 文件：flash_attention.glsl

#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, binding = 0) buffer QBuffer {
    half Q[];
} q_buffer;

layout(std430, binding = 1) buffer KBuffer {
    half K[];
} k_buffer;

layout(std430, binding = 2) buffer VBuffer {
    half V[];
} v_buffer;

layout(std430, binding = 3) buffer OBuffer {
    float O[];
} o_buffer;

layout(binding = 4) uniform AttentionParams {
    int B, H, T, D;
    float scale;
} params;

shared float shared_q[16];
shared float shared_k[16];
shared float shared_v[16];

void main() {
    uint b = gl_GlobalInvocationID.z / params.H;
    uint h = gl_GlobalInvocationID.z % params.H;
    uint i = gl_GlobalInvocationID.y;  // query position
    uint d = gl_GlobalInvocationID.x;  // dimension
    
    float max_val = -1.0e30;
    float sum_exp = 0.0;
    float acc = 0.0;
    
    // 加载 Q
    shared_q[gl_LocalInvocationID.x] = float(q_buffer.Q[
        ((b * params.H + h) * params.T + i) * params.D + d]);
    barrier();
    
    // 遍历所有 K
    for (int j = 0; j < params.T; j++) {
        // 加载 K
        shared_k[gl_LocalInvocationID.x] = float(k_buffer.K[
            ((b * params.H + h) * params.T + j) * params.D + d]);
        barrier();
        
        // 计算注意力分数
        float qk = shared_q[gl_LocalInvocationID.x] * 
                  shared_k[gl_LocalInvocationID.x] * params.scale;
        
        // 在线 Softmax
        float new_max = max(max_val, qk);
        float new_sum = sum_exp * exp(max_val - new_max) + exp(qk - new_max);
        
        // 加载 V 并累加
        shared_v[gl_LocalInvocationID.x] = float(v_buffer.V[
            ((b * params.H + h) * params.T + j) * params.D + d]);
        barrier();
        
        acc = (acc * sum_exp * exp(max_val - new_max) + 
               exp(qk - new_max) * shared_v[gl_LocalInvocationID.x]) / new_sum;
        
        max_val = new_max;
        sum_exp = new_sum;
        
        barrier();
    }
    
    // 存储结果
    o_buffer.O[((b * params.H + h) * params.T + i) * params.D + d] = acc;
}
```

## 练习

1. 实现一个简单的 Vulkan 向量加法计算着色器
2. 将 CUDA 矩阵乘法转换为 Vulkan 版本
3. 实现 Q4_0 量化的 Vulkan 计算着色器
4. 使用 Vulkan 时间戳进行性能分析

## 参考资料

- [Vulkan Specification](https://www.khronos.org/registry/vulkan/specs/)
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [Sascha Willems Vulkan Examples](https://github.com/SaschaWillems/Vulkan)
- [GGML Vulkan Source](https://github.com/ggerganov/llama.cpp/blob/master/ggml-vulkan.cpp)
