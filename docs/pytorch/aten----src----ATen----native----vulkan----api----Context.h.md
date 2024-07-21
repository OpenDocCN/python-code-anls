# `.\pytorch\aten\src\ATen\native\vulkan\api\Context.h`

```
  // 预处理指令，用于仅在定义了宏 USE_VULKAN_API 时编译此部分代码
#pragma once

// 忽略 lint 工具检测到的名为 facebook-hte-BadMemberName 的错误
// lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#ifdef USE_VULKAN_API

// 包含 Vulkan API 头文件
#include <ATen/native/vulkan/api/vk_api.h>

// 包含 Vulkan API 的各个组件头文件
#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Pipeline.h>
#include <ATen/native/vulkan/api/QueryPool.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Runtime.h>
#include <ATen/native/vulkan/api/Shader.h>
#include <ATen/native/vulkan/api/Utils.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

// Vulkan 上下文配置结构体
struct ContextConfig final {
  uint32_t cmdSubmitFrequency;          // 命令提交频率
  CommandPoolConfig cmdPoolConfig;      // 命令池配置
  DescriptorPoolConfig descriptorPoolConfig;  // 描述符池配置
  QueryPoolConfig queryPoolConfig;      // 查询池配置
};

//
// Vulkan 上下文类持有所有与 PyTorch 中 Vulkan 使用相关的 Vulkan 状态。
// 一个上下文与一个 Adapter 相关联，为多 GPU 支持的先决条件。
// PyTorch 中的所有 Vulkan 张量都与一个上下文相关联，以明确张量 <-> 设备的关联关系。
// 上下文当前是一个全局对象，但如果我们将其显式地提供给用户，它实际上并不需要是全局的。
//

class Context final {
 public:
  // 显式构造函数，初始化上下文对象
  explicit Context(size_t adapter_i, const ContextConfig&);

  // 禁止拷贝构造和赋值操作
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  // 禁止移动构造和移动赋值操作
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  // 析构函数，释放资源
  ~Context();

 private:
  // 配置信息
  ContextConfig config_;
  // 重要的句柄
  Adapter* adapter_p_;        // 适配器指针
  VkDevice device_;           // Vulkan 设备句柄
  Adapter::Queue queue_;      // 适配器队列
  // 资源池
  CommandPool command_pool_;  // 命令池
  DescriptorPool descriptor_pool_;  // 描述符池
  FencePool fences_;          // 围栏池
  // 诊断信息
  // TODO: 删除 USE_VULKAN_GPU_DIAGNOSTICS
  bool enable_op_profiling_{false};  // 是否启用操作分析
#ifdef USE_VULKAN_GPU_DIAGNOSTICS
  QueryPool querypool_;       // 查询池
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */
  // 命令缓冲提交
  std::mutex cmd_mutex_;      // 命令锁
  CommandBuffer cmd_;         // 命令缓冲
  uint32_t submit_count_;     // 提交计数
  // 内存管理
  std::mutex buffer_clearlist_mutex_;   // 缓冲清理列表锁
  std::vector<VulkanBuffer> buffers_to_clear_;  // 待清理的 Vulkan 缓冲列表
  std::mutex image_clearlist_mutex_;    // 图像清理列表锁
  std::vector<VulkanImage> images_to_clear_;  // 待清理的 Vulkan 图像列表

 public:
  // 适配器访问函数

  // 返回适配器指针
  inline Adapter* adapter_ptr() {
    return adapter_p_;
  }

  // 启用操作分析
  inline void enable_op_profiling() {
    enable_op_profiling_ = true;
  }

  // 禁用操作分析
  inline void disable_op_profiling() {
    enable_op_profiling_ = false;
  }

  // 检查是否启用操作分析
  inline bool op_profiling_enabled() {
    return enable_op_profiling_;
  }

  // 返回 Vulkan 设备句柄
  inline VkDevice device() {
    return device_;
  }

  // 返回 Vulkan 队列句柄
  inline VkQueue queue() {
    return queue_.handle;
  }

  // 设备缓存

  // 返回着色器布局缓存引用
  inline ShaderLayoutCache& shader_layout_cache() {
    return adapter_ptr()->shader_layout_cache();
  }

  // 返回着色器缓存引用
  inline ShaderCache& shader_cache() {
    return adapter_ptr()->shader_cache();
  }

  // 返回管线布局缓存引用
  inline PipelineLayoutCache& pipeline_layout_cache() {
    // 返回当前适配器指针所指向的管线布局缓存
    return adapter_ptr()->pipeline_layout_cache();
    }
    
    // 返回当前适配器指针所指向的计算管线缓存
    inline ComputePipelineCache& pipeline_cache() {
      return adapter_ptr()->compute_pipeline_cache();
    }
    
    // 资源池
    
    // 返回描述符池对象的引用
    inline DescriptorPool& descriptor_pool() {
      return descriptor_pool_;
    }
    
    // 返回围栏池对象的引用
    inline FencePool& fences() {
      return fences_;
    }
    
    // 诊断
#ifdef USE_VULKAN_GPU_DIAGNOSTICS
  // 如果定义了 USE_VULKAN_GPU_DIAGNOSTICS 宏，则返回查询池对象的引用
  inline QueryPool& querypool() {
    return querypool_;
  }

  // 如果定义了 USE_VULKAN_GPU_DIAGNOSTICS 宏，则重置查询池对象
  inline void reset_querypool() {
    // 设置命令以准备执行重置操作
    set_cmd();
    // 调用查询池对象的重置方法
    querypool_.reset(cmd_);
  }
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */

// 内存管理部分

// 注册 VulkanBuffer 实例以便在清理列表中管理
void register_buffer_cleanup(VulkanBuffer& buffer) {
  std::lock_guard<std::mutex> bufferlist_lock(buffer_clearlist_mutex_);
  // 将传入的 VulkanBuffer 实例移动到清理列表中
  buffers_to_clear_.emplace_back(std::move(buffer));
}

// 注册 VulkanImage 实例以便在清理列表中管理
void register_image_cleanup(VulkanImage& image) {
  std::lock_guard<std::mutex> imagelist_lock(image_clearlist_mutex_);
  // 将传入的 VulkanImage 实例移动到清理列表中
  images_to_clear_.emplace_back(std::move(image));
}

// GPU RPC 部分

// 返回 GPU 命令锁的独占性互斥量
inline std::unique_lock<std::mutex> dispatch_lock() {
  return std::unique_lock<std::mutex>(cmd_mutex_);
}

// 设置 GPU 命令，如果可重用则设置为可重用状态
inline void set_cmd(bool reusable = false) {
  if (!cmd_) {
    // 获取新的命令对象并开始执行
    cmd_ = command_pool_.get_new_cmd(reusable);
    cmd_.begin();
  }
}

// 获取描述符集合对象
DescriptorSet get_descriptor_set(const ShaderInfo&, const utils::uvec3&);

// 注册着色器调度任务
void register_shader_dispatch(
    const DescriptorSet&,
    PipelineBarrier&,
    const ShaderInfo&,
    const utils::uvec3&);

// 提交复制任务模板
template <class S, class D>
bool submit_copy(
    PipelineBarrier&,
    const S&,
    const D&,
    const api::utils::uvec3&,
    const api::utils::uvec3&,
    const api::utils::uvec3&,
    VkFence fence_handle);

// 提交计算任务到 GPU
template <typename... Arguments>
bool submit_compute_job(
    const ShaderInfo&,
    PipelineBarrier&,
    const utils::uvec3&,
    const utils::uvec3&,
    VkFence fence_handle,
    Arguments&&...);

// 提交命令到 GPU
void submit_cmd_to_gpu(
    VkFence fence_handle = VK_NULL_HANDLE,
    const bool final_use = false);

// 刷新操作
void flush();
};

// UniformParamsBuffer 类定义开始

class UniformParamsBuffer final {
 private:
  Context* context_p_;        // 上下文指针
  size_t nbytes_;             // 数据字节数
  VulkanBuffer vulkan_buffer_;  // Vulkan 缓冲对象

 public:
  // 默认构造函数
  UniformParamsBuffer() : context_p_{nullptr}, vulkan_buffer_{} {}

  // 带参数的构造函数，用于创建 UniformParamsBuffer 实例
  template <typename Block>
  UniformParamsBuffer(Context* context_p, const Block& block)
      : context_p_(context_p),
        nbytes_(sizeof(block)),
        vulkan_buffer_(
            context_p_->adapter_ptr()->vma().create_params_buffer(block)) {}

  // 拷贝构造函数
  UniformParamsBuffer(const UniformParamsBuffer&);

  // 拷贝赋值运算符重载
  UniformParamsBuffer& operator=(const UniformParamsBuffer&);

  // 移动构造函数
  UniformParamsBuffer(UniformParamsBuffer&&) = default;

  // 移动赋值运算符重载
  UniformParamsBuffer& operator=(UniformParamsBuffer&&) = default;

  // 析构函数
  ~UniformParamsBuffer() {
    if (vulkan_buffer_) {
      // 注册 VulkanBuffer 实例以进行清理
      context_p_->register_buffer_cleanup(vulkan_buffer_);
    }
  }

  // 返回 Vulkan 缓冲对象的引用
  VulkanBuffer& buffer() {
    return vulkan_buffer_;
  }

  // 更新 UniformParamsBuffer 数据
  template <typename Block>
  void update(const Block& block) {
    if (sizeof(block) != nbytes_) {
      // 如果数据块大小与预期不符，抛出异常
      VK_THROW(
          "Attempted to update UniformParamsBuffer with data of different size");
    }
    // 使用数据块填充 UniformParamsBuffer
    {
      MemoryMap mapping(vulkan_buffer_, MemoryAccessType::WRITE);
      Block* data_ptr = mapping.template data<Block>();

      *data_ptr = block;
    }
  }
};
/*
  定义了一个名为 StorageBuffer 的类，用于管理 Vulkan 缓冲区的存储，
  包括上下文指针、数据类型、元素数量、字节数和 Vulkan 缓冲区对象。

  构造函数初始化了对象的成员变量，创建了 Vulkan 缓冲区对象。

  析构函数负责在对象生命周期结束时清理 Vulkan 缓冲区。

  提供了几个内联函数用于访问类的私有成员变量，如数据类型、Vulkan 缓冲区对象、
  元素数量和字节数。

  构造函数和赋值运算符被标记为删除，禁止对象的复制构造和赋值操作。
*/
class StorageBuffer final {
 private:
  Context* context_p_;
  ScalarType dtype_;
  size_t numel_;
  size_t nbytes_;
  VulkanBuffer vulkan_buffer_;

 public:
  StorageBuffer(
      Context* context_p,
      const ScalarType dtype,
      const size_t numel,
      const bool gpuonly = false)
      : context_p_(context_p),
        dtype_(dtype),
        numel_(numel),
        nbytes_(element_size(dtype_) * numel_),
        vulkan_buffer_(context_p_->adapter_ptr()->vma().create_storage_buffer(
            nbytes_,
            gpuonly)) {}

  StorageBuffer(const StorageBuffer&) = delete;
  StorageBuffer& operator=(const StorageBuffer&) = delete;

  StorageBuffer(StorageBuffer&&) = default;
  StorageBuffer& operator=(StorageBuffer&&) = default;

  ~StorageBuffer() {
    context_p_->register_buffer_cleanup(vulkan_buffer_);
  }

  inline ScalarType dtype() {
    return dtype_;
  }

  inline VulkanBuffer& buffer() {
    return vulkan_buffer_;
  }

  inline size_t numel() {
    return numel_;
  }

  inline size_t nbytes() {
    return nbytes_;
  }
};

/*
  声明了一个全局函数 available()，用于检查某种资源是否可用。
*/

bool available();

/*
  使用此函数获取全局运行时上下文对象 Context*，在函数内部作为静态局部变量声明。
*/

Context* context();

namespace detail {

/*
  内联函数 arg_is_empty 用于检查 VulkanBuffer 对象是否没有分配内存，
  如果没有分配则将 any_is_empty 置为 true。
*/

inline void arg_is_empty(bool& any_is_empty, const VulkanBuffer& buffer) {
  // bool(buffer) will evaluate to false if no memory has been allocated
  any_is_empty = any_is_empty || !buffer;
}

/*
  内联函数 arg_is_empty 用于检查 VulkanImage 对象是否没有分配内存，
  如果没有分配则将 any_is_empty 置为 true。
*/

inline void arg_is_empty(bool& any_is_empty, const VulkanImage& image) {
  // bool(image) will evaluate to false if no memory has been allocated
  any_is_empty = any_is_empty || !image;
}

/*
  模板函数 any_arg_is_empty 用于检查可变参数列表中的 VulkanBuffer 或 VulkanImage
  对象是否有任何一个未分配内存的情况，并返回布尔值。
*/

template <typename... Arguments>
inline bool any_arg_is_empty(Arguments&&... arguments) {
  bool any_is_empty = false;
  VK_UNUSED const int _[]{
      0,
      (arg_is_empty(any_is_empty, std::forward<Arguments>(arguments)), 0)...,
  };

  return any_is_empty;
}

/*
  模板函数 bind 用于绑定 DescriptorSet 对象中的 VulkanBuffer 或 VulkanImage 对象，
  根据索引序列 Indices 绑定相应的参数 arguments。
*/

template <size_t... Indices, typename... Arguments>
inline void bind(
    DescriptorSet& descriptor_set,
    const std::index_sequence<Indices...>&,
    Arguments&&... arguments) {
  VK_UNUSED const int _[]{
      0,
      (descriptor_set.bind(Indices, std::forward<Arguments>(arguments)), 0)...,
  };
}

} // namespace detail

/*
  模板函数 record_copy 用于记录将 VulkanBuffer 对象之间的数据复制操作，
  但被标记为删除，禁止其使用。
*/

template <class S, class D>
inline void record_copy(
    CommandBuffer& cmd,
    const S& source,
    const D& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) = delete;

/*
  特化模板函数 record_copy 用于实现 VulkanBuffer 对象之间的数据复制操作，
  使用 CommandBuffer 对象记录复制指令。
*/

template <>
inline void record_copy<VulkanBuffer, VulkanBuffer>(
    CommandBuffer& cmd,
    const VulkanBuffer& source,
    const VulkanBuffer& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  cmd.copy_buffer_to_buffer(
      source, destination, copy_range, src_offset, dst_offset);
}

/*
  以下是一个未完成的模板特化声明。
*/

template <>
/*
  在当前命令缓冲区中记录GPU数据拷贝操作。如果submit_*_job调用次数超过配置的频率，
  或者提供了一个fence，则将命令缓冲区提交到GPU执行。返回一个布尔值，指示函数调用是否导致GPU队列提交。
 */
template <class S, class D>
inline bool Context::submit_copy(
    PipelineBarrier& pipeline_barrier,
    const S& source,
    const D& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset,
    VkFence fence_handle) {
  // 如果任一提供的参数没有关联的内存，则早期退出，因为没有需要执行的工作。
  // 然而，如果传递了一个fence并且提交计数submit_count_ > 0，则仍然需要提交当前命令缓冲区以便信号fence。
  if (!source || !destination) {
    if (fence_handle != VK_NULL_HANDLE && submit_count_ > 0) {
      submit_cmd_to_gpu(fence_handle);
      return true;
    }
    return false;
  }

  // 将记录操作序列化到共享命令缓冲区中。暂时不使用互斥锁进行初始化，因为在某些情况下它将由外部管理。
  std::unique_lock<std::mutex> cmd_lock;
  // 参考submit_compute_job中的注释以获取解释。
  if (fence_handle == VK_NULL_HANDLE) {
    cmd_lock = std::unique_lock<std::mutex>(cmd_mutex_);
  }

  // 设置命令缓冲区
  set_cmd();

#ifdef USE_VULKAN_GPU_DIAGNOSTICS
  uint32_t log_idx = UINT32_MAX;
  // 如果启用了操作性能分析，则创建一个标签并开始记录着色器的性能。
  if (enable_op_profiling_) {
    std::string label = "cmd_copy";
    log_idx = querypool_.shader_profile_begin(
        cmd_, label, create_extent3d({0, 0, 0}), create_extent3d({0, 0, 0}));
  }
#endif
#ifdef USE_VULKAN_GPU_DIAGNOSTICS
  // 如果定义了 USE_VULKAN_GPU_DIAGNOSTICS 宏，则执行以下代码块
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */

  // 在当前命令缓冲中插入流水线屏障
  cmd_.insert_barrier(pipeline_barrier);

  // 记录从源到目标的复制操作
  record_copy(cmd_, source, destination, copy_range, src_offset, dst_offset);

#ifdef USE_VULKAN_GPU_DIAGNOSTICS
  // 如果启用操作性能分析
  if (enable_op_profiling_) {
    // 结束着色器的性能分析
    querypool_.shader_profile_end(cmd_, log_idx);
  }
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */

  // 递增提交计数
  submit_count_++;
  // 如果存在 fence_handle 或者提交计数达到配置的 cmdSubmitFrequency
  if (fence_handle != VK_NULL_HANDLE ||
      submit_count_ >= config_.cmdSubmitFrequency) {
    // 向 GPU 提交命令缓冲
    submit_cmd_to_gpu(fence_handle);
    // 返回提交成功
    return true;
  }
  // 返回提交失败
  return false;
}

/*
  记录一个计算着色器分派到当前命令缓冲中。如果提交任务的调用次数超过配置的频率，
  或者提供了 fence，则将命令缓冲提交到 GPU 执行。返回一个布尔值，指示函数调用是否导致 GPU 队列提交。
 */
template <typename... Arguments>
inline bool Context::submit_compute_job(
    const ShaderInfo& shader,
    PipelineBarrier& pipeline_barrier,
    const utils::uvec3& global_work_group,
    const utils::uvec3& local_work_group_size,
    VkFence fence_handle,
    Arguments&&... arguments) {
  // 如果任何一个提供的参数没有关联的内存，则提前退出，因为没有工作需要执行。
  // 但是，如果传递了 fence，并且命令缓冲不为空，则必须提交当前命令缓冲，以便可以信号化 fence。
  if (detail::any_arg_is_empty(arguments...)) {
    if (fence_handle != VK_NULL_HANDLE && submit_count_ > 0) {
      // 提交命令缓冲到 GPU
      submit_cmd_to_gpu(fence_handle);
      // 返回提交成功
      return true;
    }
    // 返回提交失败
    return false;
  }

  // 序列化到共享命令缓冲中的记录。在某些情况下不要初始化互斥锁，
  // 因为它将由外部管理。
  std::unique_lock<std::mutex> cmd_lock;
  // 如果传递了 fence，则假设主机意图与 GPU 同步，意味着将立即调用 fence.wait() 和 flush()。
  // 因此，在这种情况下假设互斥锁是外部管理的，并且调用线程在调用函数之前已经锁定了互斥锁，
  // 并且在调用 flush() 后将手动释放互斥锁。这将阻止在我们刷新 Context 之前记录更多的分派。
  if (fence_handle == VK_NULL_HANDLE) {
    cmd_lock = std::unique_lock<std::mutex>(cmd_mutex_);
  }

  // 设置命令
  set_cmd();

#ifdef USE_VULKAN_GPU_DIAGNOSTICS
  uint32_t log_idx = UINT32_MAX;
  // 如果启用操作性能分析
  if (enable_op_profiling_) {
    // 开始着色器的性能分析
    log_idx = querypool_.shader_profile_begin(
        cmd_,
        shader.kernel_name,
        create_extent3d(global_work_group),
        create_extent3d(local_work_group_size));
  }
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */
// 结束条件编译指令，用于控制是否包含 Vulkan GPU 诊断功能相关代码段

  // 将与模板参数无关的代码分离出来，以减少代码膨胀。
  DescriptorSet descriptor_set =
      get_descriptor_set(shader, local_work_group_size);
  // 获取描述符集合，根据着色器和局部工作组大小

  detail::bind(
      descriptor_set,
      std::index_sequence_for<Arguments...>{},
      std::forward<Arguments>(arguments)...);
  // 绑定描述符集合和参数列表，使用完美转发的方式

  // 将与模板参数无关的代码分离出来，以减少代码膨胀。
  register_shader_dispatch(
      descriptor_set, pipeline_barrier, shader, global_work_group);
  // 注册着色器分发，传入描述符集合、管线障碍、着色器和全局工作组

#ifdef USE_VULKAN_GPU_DIAGNOSTICS
  if (enable_op_profiling_) {
    querypool_.shader_profile_end(cmd_, log_idx);
  }
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */
  // 如果启用操作分析，调用查询池的着色器分析结束函数

  submit_count_++;
  // 递增提交计数器

  if (fence_handle != VK_NULL_HANDLE ||
      submit_count_ >= config_.cmdSubmitFrequency) {
    submit_cmd_to_gpu(fence_handle);
    return true;
  }
  // 如果存在信号量句柄或者提交计数达到配置的提交频率，则向 GPU 提交命令并返回 true

  return false;
}
// 函数结束，返回 false

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
// 结束条件编译指令，用于控制是否包含 Vulkan API 相关代码段
```