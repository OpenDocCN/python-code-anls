# `.\pytorch\aten\src\ATen\native\vulkan\api\Tensor.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName
// 忽略 lint 工具中的特定警告，允许使用不符合命名规范的成员名

#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则包含以下头文件内容

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Types.h>
// 引入 Vulkan 相关的头文件

namespace at {
namespace native {
namespace vulkan {

struct LastAccess {
  api::PipelineStageFlags stage; // 记录管线阶段的标志
  api::MemoryAccessFlags access; // 记录内存访问权限的标志

  LastAccess()
      : stage{api::PipelineStage::NO_STAGE},
        access{api::MemoryAccessType::NONE} {} // 默认构造函数，初始化为无效值

  LastAccess(
      api::PipelineStageFlags stage_flags,
      api::MemoryAccessFlags access_flags)
      : stage{stage_flags}, access{access_flags} {} // 带参数的构造函数，初始化为指定的阶段和权限
};

class vTensorStorage final {
 public:
  // 禁止空的 vTensorStorage 构造
  vTensorStorage() = default;

  vTensorStorage(
      api::Context* context,
      const api::StorageType storage_type,
      const api::GPUMemoryLayout gpu_memory_layout,
      const std::vector<int64_t>& sizes,
      const api::ScalarType dtype,
      const bool allocate_memory = true);
      // 构造函数，初始化 vTensorStorage 对象，可能会分配内存

  vTensorStorage(const vTensorStorage&) = delete;
  vTensorStorage& operator=(const vTensorStorage&) = delete;
  // 禁用拷贝构造函数和赋值运算符

  vTensorStorage(vTensorStorage&&) = default;
  vTensorStorage operator=(vTensorStorage&&) = delete;
  // 允许移动构造函数，禁用移动赋值运算符

  ~vTensorStorage();
  // 析构函数，释放资源

  friend class vTensor;

 private:
  api::Context* context_{}; // Vulkan 上下文对象指针

  api::StorageType storage_type_; // 存储类型

  api::utils::uvec3 extents_{}; // 资源尺寸
  int64_t buffer_length_{}; // 缓冲区长度

  mutable api::VulkanImage image_; // 可变的 Vulkan 图像对象
  mutable api::VulkanBuffer buffer_; // 可变的 Vulkan 缓冲区对象

  LastAccess last_access_; // 记录最后访问的阶段和权限，用于插入内存屏障

 private:
  void flush();
  // 注册底层内存以便清理

  void transition(
      api::PipelineBarrier&,
      const api::PipelineStageFlags,
      const api::MemoryAccessFlags);
  // 插入内存屏障操作

  void verify() const;
  // 验证函数，用于检查对象状态的一致性

 public:
  inline VkFormat texture_format() {
    return image_.format();
  }
  // 内联函数，返回 Vulkan 图像的格式

  void discard_and_reallocate(
      const std::vector<int64_t>& gpu_sizes,
      const api::GPUMemoryLayout gpu_memory_layout,
      const api::ScalarType dtype);
  // 丢弃当前资源并重新分配新资源的函数
};
// vTensor 类的定义，表示一个 Vulkan 张量类
class vTensor final {
 public:
  // 禁止空的 vTensor 构造
  vTensor() = default;

  // 默认构造函数，用给定的上下文、尺寸、数据类型创建 vTensor
  vTensor(
      api::Context* context,
      const std::vector<int64_t>& sizes,
      const api::ScalarType dtype,
      const api::StorageType storage_type = api::StorageType::TEXTURE_3D,
      const api::GPUMemoryLayout memory_layout =
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
      const bool allocate_memory = true);

  // 用于量化 vTensor 的默认构造函数，带有量化参数
  vTensor(
      api::Context* const context,
      const std::vector<int64_t>& sizes,
      double q_scale,
      int64_t q_zero_point,
      const api::ScalarType dtype,
      const api::StorageType storage_type = api::StorageType::TEXTURE_3D,
      const api::GPUMemoryLayout memory_layout =
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);

  // 拷贝构造函数和赋值运算符，用于复制 vTensor 对象
  vTensor(const vTensor& other) = default;
  vTensor& operator=(const vTensor& other) = default;

  // 移动构造函数和移动赋值运算符，支持移动语义
  vTensor(vTensor&& other) = default;
  vTensor& operator=(vTensor&& other) = default;

  // 用于传递缓冲区大小和步幅数据给着色器的结构体
  struct BufferMetadata {
    api::utils::uvec4 sizes;  // 缓冲区尺寸
    api::utils::uvec4 strides;  // 缓冲区步幅
    uint32_t ndim;  // 维度数量
    return view_->storage_type_;  // 返回视图的存储类型
  }

  // 返回常规图像视图的 Vulkan 图像对象
  inline api::VulkanImage& image() const& {
    return view_->image_;
  }

  // 返回带有屏障和管线阶段标志的 Vulkan 图像对象
  api::VulkanImage& image(api::PipelineBarrier&, const api::PipelineStageFlags)
      const&;

  // 返回带有屏障、管线阶段标志和内存访问标志的 Vulkan 图像对象
  api::VulkanImage& image(
      api::PipelineBarrier&,
      const api::PipelineStageFlags,
      const api::MemoryAccessFlags) &;

  // 返回常规缓冲区视图的 Vulkan 缓冲区对象
  inline api::VulkanBuffer& buffer() const& {
    return view_->buffer_;
  }

  // 返回带有屏障和管线阶段标志的 Vulkan 缓冲区对象
  api::VulkanBuffer& buffer(
      api::PipelineBarrier&,
      const api::PipelineStageFlags) const&;

  // 返回带有屏障、管线阶段标志和内存访问标志的 Vulkan 缓冲区对象
  api::VulkanBuffer& buffer(
      api::PipelineBarrier&,
      const api::PipelineStageFlags,
      const api::MemoryAccessFlags) &;

  /*
    元数据
  */

  // 返回视图的尺寸
  inline const api::utils::uvec3& extents() const {
    return view_->extents_;
  }

  /*
   * 从 TensorOptions 成员中提取 api::ScalarType
   */
  inline api::ScalarType dtype() const {
    return dtype_;
  }

  /*
   * 获取与纹理图像格式对应的 api::ScalarType
   */
  inline api::ScalarType texture_dtype() const {
    return api::element_scalartype(view_->texture_format());
  }

  // 返回 GPU 内存布局类型
  inline api::GPUMemoryLayout gpu_memory_layout() const {
    return memory_layout_;
  }

  // 返回 GPU 内存布局类型的整数表示
  inline uint32_t gpu_memory_layout_as_uint() const {
    return static_cast<uint32_t>(memory_layout_);
  }

  // 返回尺寸向量的常引用
  inline const std::vector<int64_t>& sizes() const {
    return sizes_;
  }

  // 返回步幅向量的常引用
  inline const std::vector<int64_t>& strides() const {
    return strides_;
  }

  // 返回 GPU 尺寸向量的常引用
  inline const std::vector<int64_t>& gpu_sizes() const {
  // 返回 gpu_sizes_ 成员变量的引用
  return gpu_sizes_;
}

inline const std::vector<int64_t>& gpu_strides() const {
  // 返回 gpu_strides_ 成员变量的引用
  return gpu_strides_;
}

inline const api::utils::uvec3& virtual_extents() const {
  // 返回 virtual_extents_ 成员变量的引用
  return virtual_extents_;
}

/*
 * 获取包含 GPU 缓冲区大小和步幅信息的统一缓冲区
 */
api::VulkanBuffer& buffer_metadata();

/*
 * 获取包含在计算着色器中使用的张量大小的统一缓冲区对象。注意，第一次调用此函数时将创建 UBO。
 */
std::shared_ptr<api::UniformParamsBuffer> cpu_sizes_ubo();

/*
 * 获取包含在计算着色器中使用的张量 GPU 大小的统一缓冲区对象。注意，第一次调用此函数时将创建 UBO。
 */
std::shared_ptr<api::UniformParamsBuffer> gpu_sizes_ubo();

/*
 * 获取包含在计算着色器中使用的图像范围的统一缓冲区对象。注意，第一次调用此函数时将创建 UBO。
 */
std::shared_ptr<api::UniformParamsBuffer> extents_ubo();

/*
 * 根据原始大小和步幅构造一个 BufferMetadata 结构体，用于传递给着色器。
 */
BufferMetadata get_cpu_buffer_metadata() const;

inline void set_is_quantized() {
  // 将 is_quantized_ 设为 true
  is_quantized_ = true;
}

inline bool is_quantized() const {
  // 返回 is_quantized_ 成员变量的值
  return is_quantized_;
}

inline void set_scale(const double q_scale) {
  // 设置 q_scale_ 成员变量的值
  q_scale_ = q_scale;
}

inline double get_scale() const {
  // 返回 q_scale_ 成员变量的值
  return q_scale_;
}

inline float get_scale_float() const {
  // 将 q_scale_ 转换为 float 类型并返回其值
  return api::utils::safe_downcast<float>(q_scale_);
}

inline void set_zero_point(const int64_t q_zero_point) {
  // 设置 q_zero_point_ 成员变量的值
  q_zero_point_ = q_zero_point;
}

inline int64_t get_zero_point() const {
  // 返回 q_zero_point_ 成员变量的值
  return q_zero_point_;
}

inline int32_t get_zero_point_int32() const {
  // 将 q_zero_point_ 转换为 int32_t 类型并返回其值
  return api::utils::safe_downcast<int32_t>(q_zero_point_);
}

inline size_t numel() const {
  // 计算 sizes_ 中元素的数量并返回
  return api::utils::multiply_integers(sizes());
}

inline size_t nbytes() const {
  // 计算存储大小（字节数）并返回，使用 element_size(dtype()) 和 numel() 计算
  return api::element_size(dtype()) * numel();
}

/*
 * 返回 gpu_sizes_ 中元素的数量，而不是 sizes_ 中的数量
 */
inline size_t gpu_numel() const {
  return api::utils::multiply_integers(gpu_sizes_);
}

/*
 * 返回 gpu_sizes_ 中元素的存储大小（字节数），而不是 sizes_ 中的存储大小
 */
inline VkDeviceSize gpu_nbytes() const {
  // 返回元素大小乘以 GPU 中元素个数，用于计算总共占用的内存大小
  return api::element_size(dtype()) * gpu_numel();
}

/*
 * 返回底层资源的 VmaAllocationCreateInfo
 */
VmaAllocationCreateInfo get_allocation_create_info() const;

/*
 * 返回底层资源的 VkMemoryRequirements
 */
VkMemoryRequirements get_memory_requirements() const;

/*
 * 将底层资源绑定到给定的内存分配上
 */
void bind_allocation(const api::MemoryAllocation& allocation);

private:
/*
 * 更新 vTensor 的大小元数据为新的大小。不应直接使用，应使用 realllocate() 或 virtual_resize()。
 */
void update_size_metadata(const std::vector<int64_t>& new_sizes);

public:
/*
 * 丢弃底层的 VkImage 或 VkBuffer，并根据新的张量大小重新分配
 */
void reallocate(const std::vector<int64_t>& new_sizes);

/*
 * 通过修改在计算着色器中使用的大小元数据，执行 vTensor 的虚拟调整大小。这使得着色器能够将底层资源视为不同的大小。
 */
void virtual_resize(const std::vector<int64_t>& new_sizes);
};

void add_buffer_barrier(
    api::PipelineBarrier&,                   // 定义了一个名为 add_buffer_barrier 的函数，用于添加缓冲区屏障
    const api::VulkanBuffer&,                // 函数参数：Vulkan 缓冲区的引用
    const api::PipelineStageFlags,           // 函数参数：管线阶段标志，指定屏障的管线阶段
    const api::MemoryAccessFlags,            // 函数参数：内存访问标志，指定屏障的内存访问类型
    const api::PipelineStageFlags,           // 函数参数：管线阶段标志，指定屏障的目标管线阶段
    const api::MemoryAccessFlags);           // 函数参数：内存访问标志，指定屏障的目标内存访问类型

} // namespace vulkan                             // 结束 vulkan 命名空间的声明
} // namespace native                             // 结束 native 命名空间的声明
} // namespace at                                 // 结束 at 命名空间的声明

#endif /* USE_VULKAN_API */                       // 结束预处理指令，检查是否使用 Vulkan API
```