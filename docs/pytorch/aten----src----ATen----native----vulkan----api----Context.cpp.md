# `.\pytorch\aten\src\ATen\native\vulkan\api\Context.cpp`

```
#include <ATen/native/vulkan/api/Context.h>

#include <cstring>
#include <memory>
#include <sstream>

#ifndef VULKAN_DESCRIPTOR_POOL_SIZE
#define VULKAN_DESCRIPTOR_POOL_SIZE 1024u
#endif

#ifndef VULKAN_QUERY_POOL_SIZE
#define VULKAN_QUERY_POOL_SIZE 4096u
#endif

namespace at {
namespace native {
namespace vulkan {
namespace api {

// Vulkan Context构造函数，接受适配器索引和配置对象作为参数
Context::Context(size_t adapter_i, const ContextConfig& config)
    : config_(config), // 初始化成员变量config_
      // 重要的句柄初始化
      adapter_p_(runtime()->get_adapter_p(adapter_i)), // 获取适配器指针
      device_(adapter_p_->device_handle()), // 获取设备句柄
      queue_(adapter_p_->request_queue()), // 请求队列

      // 资源池初始化
      command_pool_(device_, queue_.family_index, config_.cmdPoolConfig), // 命令池初始化
      descriptor_pool_(device_, config_.descriptorPoolConfig), // 描述符池初始化
      fences_(device_), // 设备的fences初始化

// 诊断信息
#ifdef USE_VULKAN_GPU_DIAGNOSTICS
      querypool_(config_.queryPoolConfig, adapter_p_), // 查询池初始化
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */

      // 命令缓冲提交
      cmd_mutex_{}, // 命令缓冲互斥锁初始化
      cmd_(VK_NULL_HANDLE, 0u), // 命令缓冲句柄和标志初始化
      submit_count_{0u}, // 提交计数初始化

      // 内存管理
      buffer_clearlist_mutex_{}, // 缓冲清理列表互斥锁初始化
      buffers_to_clear_{}, // 待清理缓冲列表初始化
      image_clearlist_mutex_{}, // 图像清理列表互斥锁初始化
      images_to_clear_{} { // 待清理图像列表初始化
}

// Vulkan Context析构函数
Context::~Context() {
  try {
    flush(); // 刷新操作
    // 通知设备队列不再使用该上下文
    adapter_p_->return_queue(queue_);
  } catch (...) {
  }
}

// 获取描述符集合的方法，接受着色器信息和本地工作组大小作为参数
DescriptorSet Context::get_descriptor_set(
    const ShaderInfo& shader_descriptor,
    const utils::uvec3& local_workgroup_size) {
  // 获取着色器布局
  VkDescriptorSetLayout shader_layout =
      shader_layout_cache().retrieve(shader_descriptor.kernel_layout);

  // 获取管线布局
  VkPipelineLayout pipeline_layout =
      pipeline_layout_cache().retrieve(shader_layout);

  // 获取管线
  VkPipeline pipeline = pipeline_cache().retrieve(
      {pipeline_layout_cache().retrieve(shader_layout),
       shader_cache().retrieve(shader_descriptor),
       local_workgroup_size});

  // 绑定管线和管线布局到命令缓冲
  cmd_.bind_pipeline(pipeline, pipeline_layout, local_workgroup_size);

  // 返回描述符池中的描述符集合
  return descriptor_pool().get_descriptor_set(
      shader_layout, shader_descriptor.kernel_layout);
}

// 注册着色器调度的方法，接受描述符集合、管线障碍和着色器信息作为参数
void Context::register_shader_dispatch(
    const DescriptorSet& descriptors,
    PipelineBarrier& pipeline_barrier,
    const ShaderInfo& shader_descriptor,
    const utils::uvec3& global_workgroup_size) {
  // 根据输出瓦片大小调整全局工作组大小
  const utils::uvec3 effective_global_wg = {
      utils::div_up(
          global_workgroup_size.data[0u],
          shader_descriptor.out_tile_size.data[0u]),
      utils::div_up(
          global_workgroup_size.data[1u],
          shader_descriptor.out_tile_size.data[1u]),
      utils::div_up(
          global_workgroup_size.data[2u],
          shader_descriptor.out_tile_size.data[2u]),
  };

  // 绑定描述符到命令缓冲
  cmd_.bind_descriptors(descriptors.get_bind_handle());
  // 插入管线障碍到命令缓冲
  cmd_.insert_barrier(pipeline_barrier);
  // 分派计算任务到命令缓冲
  cmd_.dispatch(effective_global_wg);
}

// 将命令提交给GPU的方法，接受栅栏句柄和最终使用标志作为参数
void Context::submit_cmd_to_gpu(VkFence fence_handle, const bool final_use) {
  if (cmd_) {
    cmd_.end(); // 结束命令缓冲


注：以上是对给定的C++ Vulkan代码的逐行注释，详细解释了每行代码的作用和初始化过程。
    # 使用 adapter_p_ 的 submit_cmd 方法提交命令到指定队列 queue_，使用 cmd_.get_submit_handle(final_use) 获取命令的提交句柄，同时传入 fence_handle 作为参数
    adapter_p_->submit_cmd(
        queue_, cmd_.get_submit_handle(final_use), fence_handle);
    
    # 将 submit_count_ 设置为无符号整数 0，表示提交计数归零
    submit_count_ = 0u;
}

void Context::flush() {
  // 等待队列空闲，确保之前提交的命令执行完成
  VK_CHECK(vkQueueWaitIdle(queue()));

  // 清空命令池和描述符池中的资源
  command_pool_.flush();
  descriptor_pool_.flush();

  // 如果存在当前命令缓冲区，使其无效
  if (cmd_) {
    cmd_.invalidate();
  }

  // 清空缓冲区和图像列表，使用互斥锁保护
  std::lock_guard<std::mutex> bufferlist_lock(buffer_clearlist_mutex_);
  std::lock_guard<std::mutex> imagelist_lock(image_clearlist_mutex_);
  buffers_to_clear_.clear();
  images_to_clear_.clear();
}

bool available() {
  // 检查是否有可用的 Vulkan 上下文
  return context();
}

Context* context() {
  // 返回静态的 Vulkan 上下文对象，如果不存在则创建
  static const std::unique_ptr<Context> context([]() -> Context* {
    try {
      // 设置提交命令的频率
      const uint32_t submit_frequency = 16u;

      // 定义命令池的配置
      const CommandPoolConfig cmd_config{
          32u, // cmdPoolInitialSize
          8u,  // cmdPoolBatchSize
      };

      // 定义描述符池的配置
      const DescriptorPoolConfig descriptor_pool_config{
          VULKAN_DESCRIPTOR_POOL_SIZE,      // descriptorPoolMaxSets
          VULKAN_DESCRIPTOR_POOL_SIZE,      // descriptorUniformBufferCount
          VULKAN_DESCRIPTOR_POOL_SIZE,      // descriptorStorageBufferCount
          VULKAN_DESCRIPTOR_POOL_SIZE,      // descriptorCombinedSamplerCount
          VULKAN_DESCRIPTOR_POOL_SIZE,      // descriptorStorageImageCount
          32u,                              // descriptorPileSizes
      };

      // 定义查询池的配置
      const QueryPoolConfig query_pool_config{
          VULKAN_QUERY_POOL_SIZE,  // maxQueryCount
          256u,                     // initialReserveSize
      };

      // 定义 Vulkan 上下文的总体配置
      const ContextConfig config{
          submit_frequency,       // cmdSubmitFrequency
          cmd_config,             // cmdPoolConfig
          descriptor_pool_config, // descriptorPoolConfig
          query_pool_config,      // queryPoolConfig
      };

      // 使用默认的适配器和配置创建 Vulkan 上下文对象
      return new Context(runtime()->default_adapter_i(), config);
    } catch (...) {
    }

    return nullptr;
  }());

  return context.get();
}

//
// UniformParamsBuffer
//

namespace {

void memcpy_to_buffer(const VulkanBuffer& src, VulkanBuffer& dst) {
  // 将源缓冲区的内容复制到目标缓冲区，使用内存映射来访问和写入
  MemoryMap dst_mapping(dst, MemoryAccessType::WRITE);

  MemoryMap src_mapping(src, MemoryAccessType::READ);
  src_mapping.invalidate();

  void* dst_ptr = dst_mapping.template data<void>();
  void* src_ptr = src_mapping.template data<void>();

  // 使用 memcpy 安全地复制内存内容
  // @lint-ignore CLANGTIDY facebook-security-vulnerable-memcpy
  memcpy(dst_ptr, src_ptr, src.mem_size());
}

} // namespace

// 复制构造函数，通过复制另一个 UniformParamsBuffer 对象来初始化
UniformParamsBuffer::UniformParamsBuffer(const UniformParamsBuffer& other)
    : context_p_(other.context_p_), vulkan_buffer_{} {
  if (other.vulkan_buffer_) {
    // 创建一个新的 Vulkan 缓冲区并将数据从另一个缓冲区复制过来
    vulkan_buffer_ = context_p_->adapter_ptr()->vma().create_uniform_buffer(
        other.vulkan_buffer_.mem_size());

    memcpy_to_buffer(other.vulkan_buffer_, vulkan_buffer_);
  }
}

// 赋值运算符重载，通过复制另一个 UniformParamsBuffer 对象来赋值
UniformParamsBuffer& UniformParamsBuffer::operator=(
    const UniformParamsBuffer& other) {
  if (&other != this) {
    // 更新 Vulkan 上下文指针
    context_p_ = other.context_p_;

    // 移动 vulkan_buffer_ 到另一个 VulkanBuffer 以进行清理
    if (vulkan_buffer_) {
      VulkanBuffer temp_buffer(std::move(vulkan_buffer_));
      context_p_->register_buffer_cleanup(temp_buffer);
    }
    // vulkan_buffer_ 现在应为空
    // ...
    // 检查另一个对象是否有 Vulkan 缓冲区
    if (other.vulkan_buffer_) {
      // 在当前对象的上下文中，使用适配器指针的 Vulkan 内存分配器创建一个统一缓冲区
      vulkan_buffer_ = context_p_->adapter_ptr()->vma().create_uniform_buffer(
          other.vulkan_buffer_.mem_size());

      // 将另一个对象的 Vulkan 缓冲区内容复制到当前对象的 Vulkan 缓冲区中
      memcpy_to_buffer(other.vulkan_buffer_, vulkan_buffer_);
    }
  }

  // 返回当前对象的引用
  return *this;
}
} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
```