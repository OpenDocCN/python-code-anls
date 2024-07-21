# `.\pytorch\aten\src\ATen\native\vulkan\api\Tensor.cpp`

```py
namespace at {
namespace native {
namespace vulkan {

namespace {

/*
 * 计算连续张量的步长。参考了TensorImpl.h中的empty_tensor_restride函数。
 */
std::vector<int64_t> calc_contiguous_strides(
    const std::vector<int64_t>& sizes) {
  // 获取张量的维度数
  int64_t ndim = static_cast<int64_t>(sizes.size());
  // 创建一个存放步长的数组
  std::vector<int64_t> strides(ndim);

  // 初始化运行乘积为1
  int64_t running_product = 1;
  // 如果维度数大于等于1
  if (ndim >= 1) {
    // 最后一个维度的步长为1
    strides.at(ndim - 1) = running_product;
    // 从倒数第二个维度开始向前计算步长
    for (int i = static_cast<int>(sizes.size()) - 2; i >= 0; --i) {
      // 更新当前维度的步长为前一维度尺寸乘以之前的运行乘积
      running_product *= sizes.at(i + 1);
      strides.at(i) = running_product;
    }
  }

  // 返回计算得到的步长数组
  return strides;
}

/*
 * 计算ChannelsLast格式张量的步长。
 */
std::vector<int64_t> calc_channels_last_strides(
    const std::vector<int64_t>& sizes) {
  // 创建一个存放步长的数组
  std::vector<int64_t> strides(sizes.size());

  // 根据维度数选择不同的计算步长方式
  switch (sizes.size()) {
    // 对于四维张量
    case 4:
      // 计算每个维度的步长
      strides.at(1) = 1;
      strides.at(3) = sizes.at(1);
      strides.at(2) = strides.at(3) * sizes.at(3);
      strides.at(0) = strides.at(2) * sizes.at(2);
      return strides;
    // 对于三维张量
    case 3:
      // 计算每个维度的步长
      strides.at(0) = 1;
      strides.at(2) = sizes.at(0);
      strides.at(1) = strides.at(2) * sizes.at(2);
      return strides;
    // 其他维度数抛出异常
    default:
      VK_THROW("ChannelsLast format only available for 3 <= ndim <= 4!");
  }

  // 返回计算得到的步长数组
  return strides;
}

/*
 * 根据尺寸、内存布局和存储类型计算张量的步长。
 */
std::vector<int64_t> calc_strides(
    const std::vector<int64_t>& sizes,
    const api::GPUMemoryLayout memory_layout,
    const api::StorageType storage_type) {
  // 根据存储类型选择不同的计算步长方式
  switch (storage_type) {
    // 若为BUFFER存储类型
    case api::StorageType::BUFFER:
      // 根据内存布局选择计算步长方式
      switch (memory_layout) {
        // 若为TENSOR_WIDTH_PACKED布局
        case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
          // 调用计算连续步长的函数
          return calc_contiguous_strides(sizes);
          break;
        // 若为TENSOR_CHANNELS_PACKED布局
        case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
          // 调用计算ChannelsLast格式步长的函数
          return calc_channels_last_strides(sizes);
          break;
        // 其他内存布局抛出异常
        default:
          VK_THROW("Invalid memory format used to create vTensor!");
      }
      break;
    // 若为TEXTURE_3D或TEXTURE_2D存储类型
    case api::StorageType::TEXTURE_3D:
    case api::StorageType::TEXTURE_2D:
      // 返回一个空步长数组，因为纹理存储类型的步长无效
      return std::vector<int64_t>(sizes.size());
    // 其他存储类型抛出异常
    default:
      VK_THROW("Invalid storage type used to create vTensor!");
  }
}

/*
 * 在GPU上存储时，一个维度将对齐到下一个4的倍数，以利用vec4数据类型。这个函数根据所需的内存格式和存储类型调整一个维度，
 * 并返回一个描述GPU上存储张量数据的尺寸数组。
 */
std::vector<int64_t> calc_gpu_sizes(
    const std::vector<int64_t>& sizes,
    const api::GPUMemoryLayout memory_layout,
    // 检查存储类型是否已知，如果不是则抛出异常
    const api::StorageType storage_type) {
  // 使用 VK_CHECK_COND 宏检查存储类型是否不是 UNKNOWN
  VK_CHECK_COND(storage_type != api::StorageType::UNKNOWN);

  // 创建一个整数向量来存储 GPU 大小
  std::vector<int64_t> gpu_sizes;
  // 如果存储类型是 BUFFER
  if (storage_type == api::StorageType::BUFFER) {
    // 调整 GPU 大小向量大小与输入 sizes 向量一致
    gpu_sizes.resize(sizes.size());
    // 将 sizes 中的数据复制到 gpu_sizes 中
    for (size_t i = 0; i < sizes.size(); i++) {
      gpu_sizes.at(i) = sizes.at(i);
    }
  }
  // 如果是 TEXTURE 存储类型
  // 对于 TEXTURE 存储，张量通常使用 3D 图像纹理存储。
  // 批次沿深度维度堆叠。为了表示图像纹理的物理三维性（使用连接的批次），GPU 大小
  // 在使用纹理存储时将固定为 4 维。
  else {
    // 使用 VK_CHECK_COND 宏检查 sizes 的维度在 0 到 4 之间
    VK_CHECK_COND(
        sizes.size() >= 0 && sizes.size() <= 4,
        "Texture storage only valid for 0 <= ndim <= 4, received: ",
        sizes.size());

    // 将 GPU 大小向量调整为长度为 4
    gpu_sizes.resize(4);
    // 按照顺序将 sizes 中的值赋给 gpu_sizes 中的对应维度
    gpu_sizes.at(0) = api::utils::val_at(-4, sizes);
    gpu_sizes.at(1) = api::utils::val_at(-3, sizes);
    gpu_sizes.at(2) = api::utils::val_at(-2, sizes);
    gpu_sizes.at(3) = api::utils::val_at(-1, sizes);
  }

  // 获取 GPU 大小向量的维度数
  size_t ndim = gpu_sizes.size();
  // 根据内存布局类型进行处理
  switch (memory_layout) {
    // 如果是 TENSOR_WIDTH_PACKED 内存布局
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      // 如果维度数大于等于 1
      if (ndim >= 1) {
        // 对最后一个维度的大小进行对齐处理，使用 INT64_C(4)
        gpu_sizes.at(ndim - 1) =
            api::utils::align_up(api::utils::val_at(-1, sizes), INT64_C(4));
      }
      break;

    // 如果是 TENSOR_HEIGHT_PACKED 内存布局
    case api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED:
      // 如果维度数大于等于 2
      if (ndim >= 2) {
        // 对倒数第二个维度的大小进行对齐处理，使用 INT64_C(4)
        gpu_sizes.at(ndim - 2) =
            api::utils::align_up(api::utils::val_at(-2, sizes), INT64_C(4));
      }
      break;

    // 如果是 TENSOR_CHANNELS_PACKED 内存布局
    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      // 如果维度数大于等于 3
      if (ndim >= 3) {
        // 对倒数第三个维度的大小进行对齐处理，使用 INT64_C(4)
        gpu_sizes.at(ndim - 3) =
            api::utils::align_up(api::utils::val_at(-3, sizes), INT64_C(4));
      }
      break;
  }

  // 返回处理后的 GPU 大小向量
  return gpu_sizes;
}

/*
 * Creates a uvec3 denoting the extents of the image texture that will be
 * created to store a tensor of a given size.
 */
// 定义一个函数 create_image_extents，返回一个 uvec3，表示用于存储给定大小张量的图像纹理的范围
api::utils::uvec3 create_image_extents(
    const std::vector<int64_t>& gpu_sizes,   // GPU 尺寸向量
    const api::StorageType storage_type,      // 存储类型
    const api::GPUMemoryLayout memory_layout) // GPU 内存布局
{
  size_t ndim = gpu_sizes.size();  // 获取尺寸向量的维度数

  if (storage_type == api::StorageType::BUFFER) {
    // 如果存储类型为缓冲区，则图像范围不适用
    return {0u, 0u, 0u};
  } else {
    // 验证维度在 1 到 4 之间
    VK_CHECK_COND(
        ndim >= 1 && ndim <= 4,
        "Texture storage only valid for 1 <= ndim <= 4!");

    using namespace api::utils;
    uint32_t width = safe_downcast<uint32_t>(val_at(-1, gpu_sizes));    // 安全地转换宽度
    uint32_t height = safe_downcast<uint32_t>(val_at(-2, gpu_sizes));   // 安全地转换高度
    uint32_t channels = safe_downcast<uint32_t>(val_at(-3, gpu_sizes)); // 安全地转换通道数
    uint32_t batch = safe_downcast<uint32_t>(val_at(-4, gpu_sizes));    // 安全地转换批次数

    switch (memory_layout) {
      case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
        // 对于宽度打包的内存布局，要求宽度必须是 4 的倍数
        VK_CHECK_COND(width % 4 == 0, "Channels must be divisible by 4!");
        width /= 4;   // 将宽度除以 4
        break;
      case api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED:
        // 对于高度打包的内存布局，要求高度必须是 4 的倍数
        VK_CHECK_COND(height % 4 == 0, "Channels must be divisible by 4!");
        height /= 4;  // 将高度除以 4
        break;
      case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
        // 对于通道打包的内存布局，要求通道数必须是 4 的倍数
        VK_CHECK_COND(channels % 4 == 0, "Channels must be divisible by 4!");
        channels /= 4; // 将通道数除以 4
        break;
      default:
        VK_THROW("Invalid memory format used!");  // 非法的内存格式
    }

    return {width, height, batch * channels}; // 返回计算后的宽度、高度和通道数
  }
}

api::UniformParamsBuffer make_metadata_uniform(
    api::Context* const context,                    // 上下文对象指针
    const std::vector<int64_t>& sizes,              // 尺寸向量
    const std::vector<int64_t>& strides,            // 步幅向量
    const api::StorageType storage_type)            // 存储类型
{
  if (storage_type != api::StorageType::BUFFER) {
    return api::UniformParamsBuffer(); // 如果不是缓冲区存储类型，返回空的 UniformParamsBuffer
  }

  // 创建缓冲区元数据对象
  vTensor::BufferMetadata metadata{
      api::utils::make_whcn_uvec4(sizes),              // 创建 whcn_uvec4 对象
      api::utils::make_whcn_uvec4(strides),            // 创建 whcn_uvec4 对象
      api::utils::safe_downcast<uint32_t>(sizes.size()),   // 安全地转换尺寸向量的大小为 uint32_t
      api::utils::safe_downcast<uint32_t>(api::utils::multiply_integers(sizes)), // 安全地转换尺寸向量的乘积为 uint32_t
  };

  return api::UniformParamsBuffer(context, metadata); // 返回创建的 UniformParamsBuffer 对象
}

} // namespace

//
// vTensor
//

vTensor::vTensor(
    api::Context* const context,                        // 上下文对象指针
    const std::vector<int64_t>& sizes,                  // 尺寸向量
    const api::ScalarType dtype,                        // 标量类型
    const api::StorageType storage_type,                // 存储类型
    const api::GPUMemoryLayout memory_layout,           // GPU 内存布局
    const bool allocate_memory)                         // 是否分配内存的标志
    // 设置成员变量 `dtype_`，用给定的数据类型初始化
    // 设置成员变量 `memory_layout_`，用给定的内存布局初始化
    // 计算尺寸和步长
    // 使用 `sizes` 容器的开始和结束迭代器初始化 `sizes_` 容器
    // 调用 `calc_strides` 函数计算步长，存储在 `strides_` 中
    // 调用 `calc_gpu_sizes` 函数计算 GPU 尺寸，存储在 `gpu_sizes_` 中
    // 使用 `gpu_sizes_` 和 `memory_layout_` 调用 `calc_strides` 函数计算 GPU 步长，存储在 `gpu_strides_` 中
    // 使用 GPU 尺寸、存储类型和内存布局创建虚拟扩展对象 `virtual_extents_`
    // 初始化元数据 uniform buffer `metadata_uniform_`
    // 初始化 CPU 尺寸 uniform buffer `cpu_sizes_uniform_` 为 nullptr
    // 初始化 GPU 尺寸 uniform buffer `gpu_sizes_uniform_` 为 nullptr
    // 初始化 extents uniform buffer `extents_uniform_` 为 nullptr
    // 使用给定的参数创建 `vTensorStorage` 对象 `view_`，并与 `context`、存储类型、内存布局、GPU 尺寸、数据类型和分配标志关联
// 构造函数：初始化 vTensor 对象
vTensor::vTensor(
    api::Context* const context,                           // 上下文指针
    const std::vector<int64_t>& sizes,                     // 尺寸向量
    double q_scale,                                        // 量化比例
    int64_t q_zero_point,                                  // 量化零点
    const api::ScalarType dtype,                           // 标量类型
    const api::StorageType storage_type,                   // 存储类型
    const api::GPUMemoryLayout memory_layout)              // GPU 内存布局
    : dtype_(dtype),                                       // 初始化成员变量：数据类型
      memory_layout_(memory_layout),                       // 初始化成员变量：内存布局
      sizes_(sizes.begin(), sizes.end()),                  // 初始化成员变量：尺寸
      strides_{calc_strides(sizes, memory_layout_, storage_type)},  // 初始化成员变量：步长
      gpu_sizes_{calc_gpu_sizes(sizes, memory_layout_, storage_type)},  // 初始化成员变量：GPU 尺寸
      gpu_strides_{calc_strides(gpu_sizes_, memory_layout_, storage_type)},  // 初始化成员变量：GPU 步长
      virtual_extents_(                                   // 初始化成员变量：虚拟尺寸
          create_image_extents(gpu_sizes_, storage_type, memory_layout)),  // 创建虚拟尺寸
      metadata_uniform_(),                                // 初始化成员变量：元数据统一缓冲区
      cpu_sizes_uniform_(nullptr),                         // 初始化成员变量：CPU 尺寸统一缓冲区
      gpu_sizes_uniform_(nullptr),                         // 初始化成员变量：GPU 尺寸统一缓冲区
      extents_uniform_(nullptr),                           // 初始化成员变量：尺寸统一缓冲区
      is_quantized_{true},                                // 初始化成员变量：量化标志
      q_scale_{q_scale},                                  // 初始化成员变量：量化比例
      q_zero_point_{q_zero_point},                        // 初始化成员变量：量化零点
      view_(std::make_shared<vTensorStorage>(             // 初始化成员变量：视图对象
          context,                                         // 上下文指针
          storage_type,                                    // 存储类型
          memory_layout_,                                  // 内存布局
          gpu_sizes_,                                      // GPU 尺寸
          dtype_)) {}                                     // 数据类型

// 返回图像对象的引用，执行图像读取内存访问权限转换
api::VulkanImage& vTensor::image(
    api::PipelineBarrier& pipeline_barrier,               // 管道屏障对象引用
    const api::PipelineStageFlags stage) const& {         // 管道阶段标志
  view_->transition(pipeline_barrier, stage, api::MemoryAccessType::READ);  // 执行图像对象内存访问权限转换到读
  return view_->image_;                                  // 返回图像对象的引用
}

// 返回图像对象的引用，执行图像读取和指定内存访问权限转换
api::VulkanImage& vTensor::image(
    api::PipelineBarrier& pipeline_barrier,               // 管道屏障对象引用
    const api::PipelineStageFlags stage,                  // 管道阶段标志
    const api::MemoryAccessFlags access) & {             // 内存访问标志
  view_->transition(pipeline_barrier, stage, access);    // 执行图像对象内存访问权限转换
  return view_->image_;                                  // 返回图像对象的引用
}

// 返回缓冲区对象的引用，执行缓冲区读取内存访问权限转换
api::VulkanBuffer& vTensor::buffer(
    api::PipelineBarrier& pipeline_barrier,               // 管道屏障对象引用
    const api::PipelineStageFlags stage) const& {         // 管道阶段标志
  view_->transition(pipeline_barrier, stage, api::MemoryAccessType::READ);  // 执行缓冲区对象内存访问权限转换到读
  return view_->buffer_;                                 // 返回缓冲区对象的引用
}

// 返回缓冲区对象的引用，执行缓冲区读取和指定内存访问权限转换
api::VulkanBuffer& vTensor::buffer(
    api::PipelineBarrier& pipeline_barrier,               // 管道屏障对象引用
    const api::PipelineStageFlags stage,                  // 管道阶段标志
    const api::MemoryAccessFlags access) & {             // 内存访问标志
  view_->transition(pipeline_barrier, stage, access);    // 执行缓冲区对象内存访问权限转换
  return view_->buffer_;                                 // 返回缓冲区对象的引用
}

// 返回元数据缓冲区对象的引用，如不存在则创建
api::VulkanBuffer& vTensor::buffer_metadata() {
  if (!metadata_uniform_.buffer()) {                     // 如果元数据缓冲区对象不存在
    metadata_uniform_ = make_metadata_uniform(           // 创建元数据统一缓冲区对象
        view_->context_, gpu_sizes_, gpu_strides_, storage_type());
  }
  return metadata_uniform_.buffer();                     // 返回元数据缓冲区对象的引用
}

// 返回 CPU 尺寸统一缓冲区对象的共享指针，如不存在则创建
std::shared_ptr<api::UniformParamsBuffer> vTensor::cpu_sizes_ubo() {
  if (!cpu_sizes_uniform_) {                             // 如果 CPU 尺寸统一缓冲区对象不存在
    cpu_sizes_uniform_.reset(new api::UniformParamsBuffer(  // 创建 CPU 尺寸统一缓冲区对象
        view_->context_, api::utils::make_whcn_ivec4(sizes_)));  // 使用尺寸向量创建缓冲区对象
  }
  return cpu_sizes_uniform_;                             // 返回 CPU 尺寸统一缓冲区对象的共享指针
}

// 返回 GPU 尺寸统一缓冲区对象的共享指针，如不存在则创建
std::shared_ptr<api::UniformParamsBuffer> vTensor::gpu_sizes_ubo() {
  if (!gpu_sizes_uniform_) {                             // 如果 GPU 尺寸统一缓冲区对象不存在
    gpu_sizes_uniform_.reset(new api::UniformParamsBuffer(  // 创建 GPU 尺寸统一缓冲区对象
        view_->context_, api::utils::make_whcn_ivec4(gpu_sizes_)));  // 使用 GPU 尺寸向量创建缓冲区对象
  }
  return gpu_sizes_uniform_;                             // 返回 GPU 尺寸统一缓冲区对象的共享指针
}
// 返回扩展参数缓冲区的共享指针，如果未初始化则进行初始化
std::shared_ptr<api::UniformParamsBuffer> vTensor::extents_ubo() {
  // 如果扩展参数缓冲区未初始化
  if (!extents_uniform_) {
    // 使用视图上下文和视图尺寸数据创建新的统一参数缓冲区对象
    extents_uniform_.reset(new api::UniformParamsBuffer(
        view_->context_,
        api::utils::uvec4(
            {view_->extents_.data[0],
             view_->extents_.data[1],
             view_->extents_.data[2],
             1u})));
  }
  // 返回扩展参数缓冲区对象
  return extents_uniform_;
}

// 返回 CPU 缓冲区的元数据
vTensor::BufferMetadata vTensor::get_cpu_buffer_metadata() const {
  return {
      // 使用 sizes_ 创建宽度、高度、通道数和数量的 uvec4 对象
      api::utils::make_whcn_uvec4(sizes_),
      // 使用 strides_ 创建宽度、高度、通道数和数量的 uvec4 对象
      api::utils::make_whcn_uvec4(strides_),
      // 安全地将 sizes_ 的大小转换为 uint32_t
      api::utils::safe_downcast<uint32_t>(sizes_.size()),
      // 安全地将 sizes_ 中所有元素相乘后的结果转换为 uint32_t
      api::utils::safe_downcast<uint32_t>(
          api::utils::multiply_integers(sizes_)),
  };
}

// 根据存储类型返回内存分配创建信息
VmaAllocationCreateInfo vTensor::get_allocation_create_info() const {
  switch (storage_type()) {
    // 如果存储类型为 BUFFER，则返回视图上的缓冲区的内存分配创建信息
    case api::StorageType::BUFFER:
      return view_->buffer_.allocation_create_info();
    // 如果存储类型为 TEXTURE_2D 或 TEXTURE_3D，则返回视图上的图像的内存分配创建信息
    case api::StorageType::TEXTURE_2D:
    case api::StorageType::TEXTURE_3D:
      return view_->image_.allocation_create_info();
    // 对于未知的存储类型，返回一个空的分配创建信息对象
    case api::StorageType::UNKNOWN:
      break;
  }
  // 默认情况下返回一个空的分配创建信息对象
  return {};
}

// 根据存储类型返回内存需求信息
VkMemoryRequirements vTensor::get_memory_requirements() const {
  switch (storage_type()) {
    // 如果存储类型为 BUFFER，则返回视图上的缓冲区的内存需求信息
    case api::StorageType::BUFFER:
      return view_->buffer_.get_memory_requirements();
    // 如果存储类型为 TEXTURE_2D 或 TEXTURE_3D，则返回视图上的图像的内存需求信息
    case api::StorageType::TEXTURE_2D:
    case api::StorageType::TEXTURE_3D:
      return view_->image_.get_memory_requirements();
    // 对于未知的存储类型，返回一个空的内存需求信息对象
    case api::StorageType::UNKNOWN:
      break;
  }
  // 默认情况下返回一个空的内存需求信息对象
  return {};
}

// 将内存分配与给定的内存分配对象绑定
void vTensor::bind_allocation(const api::MemoryAllocation& allocation) {
  switch (storage_type()) {
    // 如果存储类型为 BUFFER，则将内存分配绑定到视图上的缓冲区
    case api::StorageType::BUFFER:
      view_->buffer_.bind_allocation(allocation);
      break;
    // 如果存储类型为 TEXTURE_2D 或 TEXTURE_3D，则将内存分配绑定到视图上的图像
    case api::StorageType::TEXTURE_2D:
    case api::StorageType::TEXTURE_3D:
      view_->image_.bind_allocation(allocation);
      break;
    // 对于未知的存储类型，不执行任何操作
    case api::StorageType::UNKNOWN:
      break;
  }
}

// 更新尺寸元数据，基于新的尺寸向量
void vTensor::update_size_metadata(const std::vector<int64_t>& new_sizes) {
  // 更新 sizes_ 成员变量为新的尺寸向量
  sizes_ = new_sizes;
  // 计算 GPU 尺寸向量，基于给定的内存布局和存储类型
  gpu_sizes_ = calc_gpu_sizes(sizes_, memory_layout_, storage_type());
  // 创建虚拟尺寸范围，基于 GPU 尺寸向量、存储类型和内存布局
  virtual_extents_ =
      create_image_extents(gpu_sizes_, storage_type(), memory_layout_);

  // 如果存在 CPU 尺寸统一参数缓冲区，则更新其尺寸
  if (cpu_sizes_uniform_) {
    cpu_sizes_uniform_->update(api::utils::make_whcn_ivec4(sizes_));
  }

  // 如果存在 GPU 尺寸统一参数缓冲区，则更新其尺寸
  if (gpu_sizes_uniform_) {
    gpu_sizes_uniform_->update(api::utils::make_whcn_ivec4(gpu_sizes_));
  }

  // 如果存在扩展参数统一参数缓冲区，则更新其内容
  if (extents_uniform_) {
    extents_uniform_->update(api::utils::uvec4(
        {virtual_extents_.data[0],
         virtual_extents_.data[1],
         virtual_extents_.data[2],
         1u}));
  }
}

// 重新分配视图的内存，基于新的尺寸向量
void vTensor::reallocate(const std::vector<int64_t>& new_sizes) {
  // 更新尺寸元数据为新的尺寸向量
  update_size_metadata(new_sizes);
  // 丢弃当前内存并重新分配，基于新的 GPU 尺寸向量、内存布局和数据类型
  view_->discard_and_reallocate(
      calc_gpu_sizes(new_sizes, memory_layout_, storage_type()),
      memory_layout_,
      dtype_);
}

// 虚拟调整尺寸，基于新的尺寸向量
void vTensor::virtual_resize(const std::vector<int64_t>& new_sizes) {
  // 更新尺寸元数据为新的尺寸向量
  update_size_metadata(new_sizes);
  // 如果存储类型为 BUFFER，则创建新的图像尺寸范围
  if (storage_type() == api::StorageType::BUFFER) {
    // 创建新的图像尺寸范围，基于 GPU 尺寸向量、存储类型和内存布局
    virtual_extents_ =
        create_image_extents(gpu_sizes_, storage_type(), memory_layout_);

  // 更新扩展参数统一参数缓冲区的内容，基于虚拟尺寸范围数据
  extents_uniform_->update(api::utils::uvec4(
      {virtual_extents_.data[0],
       virtual_extents_.data[1],
       virtual_extents_.data[2],
       1u}));
  }
}
    # 检查当前 GPU 占用的内存是否大于视图缓冲区的内存大小
    if (gpu_nbytes() > view_->buffer_.mem_size()) {
      # 若是，则抛出异常，指示无法使用 virtual_resize 来调整 vTensor 的大小，
      # 因为需要更大的缓冲区。应该使用 reallocate() 方法来重新分配内存。
      VK_THROW(
          "Cannot virtual_resize a vTensor with sizes that require a larger "
          "buffer! reallocate() should be used instead.");
    }
  } else {
    # 若 GPU 占用内存未超过视图缓冲区大小，则检查是否可以有效调整大小
    bool valid_resize = true;
    # 检查虚拟张量在每个维度上的大小是否超过了视图张量对应维度的大小
    if (virtual_extents_.data[0] > view_->extents_.data[0]) {
      valid_resize = false;
    }
    if (virtual_extents_.data[1] > view_->extents_.data[1]) {
      valid_resize = false;
    }
    if (virtual_extents_.data[2] > view_->extents_.data[2]) {
      valid_resize = false;
    }

    # 若存在无法有效调整大小的情况，则抛出异常，指示无法使用 virtual_resize 来调整 vTensor 的大小，
    # 因为需要更大的图像纹理。应该使用 reallocate() 方法来重新分配内存。
    if (!valid_resize) {
      VK_THROW(
          "Cannot virtual_resize a vTensor with sizes that require a larger "
          "image texture! reallocate() should be used instead.");
    }
  }
}

//
// vTensorStorage
//

// 分配一个 Vulkan 图像
api::VulkanImage allocate_image(
    api::Context* const context_ptr,
    api::utils::uvec3& extents,
    const api::StorageType storage_type,
    const VkFormat image_format,
    const bool allocate_memory) {
  // 获取上下文中的适配器指针
  api::Adapter* adapter_ptr = context_ptr->adapter_ptr();

  // 定义图像采样器的属性
  api::ImageSampler::Properties sampler_props{
      VK_FILTER_NEAREST,
      VK_SAMPLER_MIPMAP_MODE_NEAREST,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
  };

  // 初始化图像类型和图像视图类型为 3D
  VkImageType image_type = VK_IMAGE_TYPE_3D;
  VkImageViewType image_view_type = VK_IMAGE_VIEW_TYPE_3D;

  // 根据存储类型选择对应的图像类型和图像视图类型
  switch (storage_type) {
    case api::StorageType::TEXTURE_3D:
      image_type = VK_IMAGE_TYPE_3D;
      image_view_type = VK_IMAGE_VIEW_TYPE_3D;
      break;
    case api::StorageType::TEXTURE_2D:
      image_type = VK_IMAGE_TYPE_2D;
      image_view_type = VK_IMAGE_VIEW_TYPE_2D;
      break;
    default:
      // 默认情况下返回一个空的 VulkanImage
      return api::VulkanImage();
  }

  // 检索适配器中的采样器
  VkSampler sampler = adapter_ptr->sampler_cache().retrieve(sampler_props);

  // 创建图像对象并返回
  return adapter_ptr->vma().create_image(
      api::create_extent3d(extents),
      image_format,
      image_type,
      image_view_type,
      sampler_props,
      sampler,
      /*allow_transfer = */ true,
      /*allocate_memory = */ allocate_memory);
}

// 分配一个 Vulkan 缓冲区
api::VulkanBuffer allocate_buffer(
    api::Context* const context_ptr,
    const int64_t numel,
    const api::StorageType storage_type,
    const api::ScalarType dtype,
    const bool allocate_memory) {
  // 获取上下文中的适配器指针
  api::Adapter* adapter_ptr = context_ptr->adapter_ptr();

  // 根据存储类型选择是否分配缓冲区，仅支持 BUFFER 类型
  switch (storage_type) {
    case api::StorageType::BUFFER:
      break;
    default:
      // 如果不使用 BUFFER 存储类型，则返回一个空的 VulkanBuffer
      return api::VulkanBuffer();
  }

  // 创建存储缓冲区并返回
  return adapter_ptr->vma().create_storage_buffer(
      api::element_size(dtype) * numel, /*gpu_only = */ true, allocate_memory);
}

// vTensorStorage 类的构造函数
vTensorStorage::vTensorStorage(
    api::Context* const context,
    const api::StorageType storage_type,
    const api::GPUMemoryLayout gpu_memory_layout,
    const std::vector<int64_t>& gpu_sizes,
    const api::ScalarType dtype,
    const bool allocate_memory)
    : context_(context),
      storage_type_{storage_type},
      extents_(
          create_image_extents(gpu_sizes, storage_type, gpu_memory_layout)), // 创建图像的大小
      buffer_length_{api::utils::multiply_integers(gpu_sizes)}, // 计算缓冲区的长度
      image_(allocate_image( // 分配图像
          context_,
          extents_,
          storage_type_,
          api::to_vkformat(dtype),
          allocate_memory)),
      buffer_(allocate_buffer( // 分配缓冲区
          context_,
          buffer_length_,
          storage_type_,
          dtype,
          allocate_memory)),
      last_access_{} {} // 最后访问时间为空

// vTensorStorage 类的析构函数
vTensorStorage::~vTensorStorage() {
  flush(); // 清理资源
}

// 刷新函数，根据图像或缓冲区的存在注册清理操作
void vTensorStorage::flush() {
  if (image_) {
    context_->register_image_cleanup(image_);
  } else if (buffer_) {
    context_->register_buffer_cleanup(buffer_);
  }
  last_access_ = {}; // 最后访问时间置空
}
// 执行张量存储过渡操作，更新管线屏障信息
void vTensorStorage::transition(
    api::PipelineBarrier& pipeline_barrier,               // 管线屏障对象的引用，用于更新屏障信息
    const api::PipelineStageFlags cur_stage,              // 当前管线阶段的标志位
    const api::MemoryAccessFlags cur_access) {            // 当前内存访问标志位
  // 获取上一次访问的阶段和访问标志
  api::PipelineStageFlags prev_stage = last_access_.stage; // 上一次访问的管线阶段
  api::MemoryAccessFlags prev_access = last_access_.access; // 上一次访问的内存访问标志

  // 检查是否上一次是写操作
  const bool prev_written = (prev_access & api::MemoryAccessType::WRITE) != 0;

  // 当前图像布局的变量声明和初始化
  VkImageLayout cur_layout = VK_IMAGE_LAYOUT_UNDEFINED;   // 当前图像布局，默认未定义
  VkImageLayout new_layout = VK_IMAGE_LAYOUT_UNDEFINED;   // 新图像布局，默认未定义
  bool layout_changed = false;                            // 标志位，指示布局是否发生了变化
  if (image_) {
    cur_layout = image_.layout();                         // 获取当前图像的布局
    new_layout = api::vk_layout(cur_stage, cur_access);   // 根据当前阶段和访问标志获取新的布局

    layout_changed = cur_layout != new_layout;            // 检查当前布局是否与新布局不同
  }

  // 如果之前有写操作或者布局发生了变化
  if (prev_written || layout_changed) {
    // 获取源管线阶段，如果未定义则设置为顶部管线阶段
    VkPipelineStageFlags src_stage = api::vk_stage(prev_stage);
    if (0u == src_stage) {
      src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
    // 获取目标管线阶段，如果未定义则设置为底部管线阶段
    VkPipelineStageFlags dst_stage = api::vk_stage(cur_stage);
    if (0u == dst_stage) {
      dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    // 更新管线屏障的源和目标管线阶段
    pipeline_barrier.stage.src |= src_stage;
    pipeline_barrier.stage.dst |= dst_stage;

    // 如果存在图像对象
    if (image_) {
      // 向管线屏障中添加图像屏障信息
      pipeline_barrier.images.emplace_back(
          api::vk_access(prev_stage, prev_access),       // 上一次访问的访问标志
          api::vk_access(cur_stage, cur_access),         // 当前访问的访问标志
          cur_layout,                                    // 当前图像布局
          new_layout,                                    // 新图像布局
          image_);                                       // 图像对象引用

      // 设置图像对象的新布局
      image_.set_layout(new_layout);
    } else if (buffer_) { // 如果存在缓冲区对象
      // 向管线屏障中添加缓冲区屏障信息
      pipeline_barrier.buffers.emplace_back(
          api::vk_access(prev_stage, prev_access),       // 上一次访问的访问标志
          api::vk_access(cur_stage, cur_access),         // 当前访问的访问标志
          buffer_);                                      // 缓冲区对象引用
    }
  }

  // 更新最后访问的管线阶段和访问标志
  last_access_.stage = cur_stage;                        // 更新最后访问的管线阶段
  last_access_.access = cur_access;                      // 更新最后访问的内存访问标志
}

// 添加缓冲区屏障操作
void add_buffer_barrier(
    api::PipelineBarrier& pipeline_barrier,               // 管线屏障对象的引用，用于更新屏障信息
    const api::VulkanBuffer& buffer,                      // Vulkan缓冲区对象的引用
    const api::PipelineStageFlags prev_stage,             // 上一个管线阶段的标志位
    const api::MemoryAccessFlags prev_access,             // 上一个内存访问标志位
    const api::PipelineStageFlags cur_stage,              // 当前管线阶段的标志位
    const api::MemoryAccessFlags cur_access) {            // 当前内存访问标志位
  // 检查是否有读请求和之前是否有写操作，确定是否属于RAW操作
  const bool read_requested = (cur_access & api::MemoryAccessType::READ) != 0;    // 是否有读请求
  const bool prev_written = (prev_access & api::MemoryAccessType::WRITE) != 0;    // 之前是否有写操作

  const bool is_RAW = read_requested && prev_written;    // 是否为RAW操作

  // 如果属于RAW操作
  if (is_RAW) {
    // 获取源管线阶段，如果未定义则设置为顶部管线阶段
    VkPipelineStageFlags src_stage = api::vk_stage(prev_stage);
    if (0u == src_stage) {
      src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
    // 获取目标管线阶段，如果未定义则设置为底部管线阶段
    VkPipelineStageFlags dst_stage = api::vk_stage(cur_stage);
    if (0u == dst_stage) {
      dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    // 更新管线屏障的源和目标管线阶段
    pipeline_barrier.stage.src |= src_stage;
    pipeline_barrier.stage.dst |= dst_stage;

    // 向管线屏障中添加缓冲区屏障信息
    pipeline_barrier.buffers.emplace_back(
        api::vk_access(prev_stage, prev_access),           // 上一次访问的访问标志
        api::vk_access(cur_stage, cur_access),             // 当前访问的访问标志
        buffer);                                          // 缓冲区对象引用
  }
}
    // 获取 image_ 对象是否拥有内存的所有权
    const bool image_owns_memory = image_.owns_memory();
    // 获取 buffer_ 对象是否拥有内存的所有权
    const bool buffer_owns_memory = buffer_.owns_memory();
    
    // 刷新操作（清空缓冲区或提交未完成的操作）
    flush();
    
    // 根据 GPU 大小、存储类型、内存布局创建图像的尺寸信息
    extents_ = create_image_extents(gpu_sizes, storage_type_, gpu_memory_layout);
    // 分配图像内存，使用指定的上下文、尺寸信息、存储类型、数据类型，并指定是否转移内存所有权
    image_ = allocate_image(
        context_,
        extents_,
        storage_type_,
        api::to_vkformat(dtype),
        image_owns_memory);
    
    // 计算缓冲区长度，即 GPU 大小数组中元素的乘积
    buffer_length_ = api::utils::multiply_integers(gpu_sizes);
    // 分配缓冲区内存，使用指定的上下文、缓冲区长度、存储类型、数据类型，并指定是否转移内存所有权
    buffer_ = allocate_buffer(
        context_,
        buffer_length_,
        storage_type_,
        dtype,
        buffer_owns_memory);
}

} // namespace vulkan
} // namespace native
} // namespace at
```