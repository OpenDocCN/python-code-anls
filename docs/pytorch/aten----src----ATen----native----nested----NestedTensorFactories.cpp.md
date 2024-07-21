# `.\pytorch\aten\src\ATen\native\nested\NestedTensorFactories.cpp`

```
// 包含 ATen 库的头文件，提供张量操作的功能
#include <ATen/ATen.h>
// 包含 NestedTensorImpl 类的头文件，提供嵌套张量的实现
#include <ATen/NestedTensorImpl.h>
// 包含 NestedTensorFactories 头文件，提供嵌套张量的工厂函数
#include <ATen/native/nested/NestedTensorFactories.h>
// 包含 NestedTensorUtils 头文件，提供嵌套张量的实用函数
#include <ATen/native/nested/NestedTensorUtils.h>

// 进入 ATen 命名空间
namespace at {
namespace native {

// 静态函数，用于验证空参数并返回张量选项
static TensorOptions verify_empty_parameters(
    const at::Tensor& self,                            // 输入张量
    std::optional<ScalarType> dtype,                   // 可选的数据类型
    std::optional<Layout> layout,                      // 可选的布局
    std::optional<Device> device,                      // 可选的设备
    std::optional<bool> pin_memory,                    // 可选的内存固定
    std::optional<c10::MemoryFormat> optional_memory_format) {  // 可选的内存格式
  // 创建张量选项对象并初始化
  TensorOptions options_ = TensorOptions()
                               .dtype(dtype)
                               .layout(layout)
                               .device(device)
                               .pinned_memory(pin_memory)
                               .memory_format(optional_memory_format);

  // 合并输入张量的选项和新创建的选项
  TensorOptions options = self.options().merge_in(options_);

  // 获取或者设置内存格式，默认为 Preserve
  auto memory_format =
      options_.memory_format_opt().value_or(MemoryFormat::Preserve);

  // 检查内存格式是否为 Preserve 或者 Contiguous，否则报错
  TORCH_CHECK(
      memory_format == MemoryFormat::Preserve || memory_format == MemoryFormat::Contiguous,
      "empty_like_nested only supports memory format Preserve or Contiguous, but got ",
      memory_format,
      " instead.");

  // 检查布局是否为 kStrided，如果是，则内存格式选项必须为空
  TORCH_CHECK(
      !(options.layout() != kStrided && optional_memory_format.has_value()),
      "memory format option is only supported by strided tensors");

  // 返回验证后的张量选项
  return options;
}

// 创建一个类似于输入张量结构的空张量
Tensor empty_like_nested(
    const Tensor& self,                                   // 输入张量
    std::optional<ScalarType> dtype,                      // 可选的数据类型
    std::optional<Layout> layout,                         // 可选的布局
    std::optional<Device> device,                         // 可选的设备
    std::optional<bool> pin_memory,                       // 可选的内存固定
    std::optional<c10::MemoryFormat> optional_memory_format) {  // 可选的内存格式
  // 验证并获取空参数的张量选项
  auto options = verify_empty_parameters(
      self, dtype, layout, device, pin_memory, optional_memory_format);

  // 获取嵌套张量的实现
  auto self_nt = get_nested_tensor_impl(self);

  // 获取内存格式选项，默认为 Preserve
  auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Preserve);

  // 如果内存格式为 Contiguous，则创建新的缓冲区张量
  if (memory_format == MemoryFormat::Contiguous) {
    // 克隆嵌套大小
    auto nested_size = self_nt->get_nested_sizes().clone();
    // 计算嵌套大小的元素个数
    int64_t buffer_size = get_numel_from_nested_size_tensor(nested_size);
    // 使用选项创建新的空缓冲区张量
    Tensor new_buffer = at::empty({buffer_size}, options);
    // 使用新缓冲区包装张量并返回
    auto tensor = wrap_buffer(new_buffer, nested_size);
    return tensor;
  }

  // 如果内存格式为 Preserve，则使用不安全存储的张量创建新的缓冲区张量
  // 此处的路径是 Preserve
  TORCH_CHECK(
      memory_format == MemoryFormat::Preserve,
      "memory format option is only supported by strided tensors");

  // 使用不安全存储的张量创建新的缓冲区张量
  Tensor new_buffer =
      at::empty_like(self_nt->get_unsafe_storage_as_tensor(), options);

  // 克隆嵌套大小、步长和偏移量
  auto nested_size = self_nt->get_nested_sizes().clone();
  auto nested_strides = self_nt->get_nested_strides().clone();
  auto offsets = self_nt->get_storage_offsets().clone();

  // 使用新缓冲区和相关参数包装张量并返回
  auto tensor = wrap_buffer(new_buffer, nested_size, nested_strides, offsets);
  return tensor;
}

// 接受可能未设置设备索引的设备参数（即设为 -1）
// 确保设备具有索引，如果当前设备是 CPU 或已经有索引，则返回当前设备；否则，获取当前设备的实际设备并返回
static inline Device ensure_has_index(Device device) {
  if (device.is_cpu() || device.has_index()) {
    return device;
  }
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}

// 将输入张量转换为指定选项的副本，支持可选的数据类型、布局、设备、钉住内存、非阻塞操作和内存格式选项
Tensor _to_copy_nested(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // 检查是否允许转换到不同布局，如果不允许，则报错
  TORCH_CHECK(
      !layout.has_value() || self.layout() == layout.value(),
      "to(options) doesn't support converting to a different layout, "
      "but got self.layout being ",
      self.layout(),
      " and options.layout set as ",
      layout.value());
  // 根据指定的选项构建张量选项
  auto options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  // 如果选项指定了设备，则确保设备具有索引
  if (options.has_device()) {
    options = options.device(ensure_has_index(options.device()));
  }
  
  // 内存格式选项单独处理，由于存在 MemoryFormat::Preserve 逻辑
  options = self.options().merge_in(options).memory_format(c10::nullopt);
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);

  // 根据条件确定是否需要在 CPU 上钉住输出张量
  bool pin_out =
      (non_blocking && self.is_cuda() && options.device().is_cpu() &&
       (options.layout() == c10::kStrided));

  // 创建并返回与输入张量类似的空张量，根据给定的选项和内存格式
  Tensor r;
  r = at::empty_like(self, dtype, layout, device, pin_out, memory_format);
  get_nested_tensor_impl(r)->get_buffer().copy_(
      get_nested_tensor_impl(self)->get_buffer(), non_blocking);
  return r;
}

// 在原地复制源张量到目标张量，支持非阻塞操作
Tensor& copy_nested_(Tensor& self, const Tensor& src, bool non_blocking) {
  const auto* nt_self = get_nested_tensor_impl(self);
  const auto* nt_src = get_nested_tensor_impl(src);
  
  // 检查两个嵌套张量的尺寸是否相同，若不同则报错
  TORCH_CHECK(
      at::equal(
          nt_self->get_nested_sizes(), nt_src->get_nested_sizes()),
      "copy_ only supports tensors that are the same size for Nested implementations");
  
  // 在原地执行复制操作
  nt_self->get_buffer().copy_(nt_src->get_buffer(), non_blocking);
  return self;
}

// 克隆嵌套张量，支持指定的内存格式选项
Tensor clone_nested(
    const Tensor& self,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format = optional_memory_format.value_or(c10::MemoryFormat::Preserve);
  auto self_ptr = get_nested_tensor_impl(self);

  // 如果内存格式是 Preserve 或者是 Contiguous 且输入张量已经是连续的，则执行下面的操作
  if (memory_format == c10::MemoryFormat::Preserve ||
  (memory_format == c10::MemoryFormat::Contiguous && self.is_contiguous())) {
    const Tensor& buffer = self_ptr->get_unsafe_storage_as_tensor(),
        sizemat = self_ptr->get_nested_sizes(),
        stridemat = self_ptr->get_nested_strides();
    const auto& offsets = self_ptr->get_storage_offsets();
    // TODO: The size and the stride do not necessarily need to be cloned,
    //       but it is more conservative.
    // 如果内存格式是保留（Preserve），则执行以下代码块
    return wrap_buffer(buffer.clone(), sizemat.clone(), stridemat.clone(), offsets.clone());
  }
  // 如果内存格式是连续的（Contiguous），且 self 是非连续的情况
  else if (memory_format == c10::MemoryFormat::Contiguous) {
    // 获取 self 的底层存储作为 Tensor
    const Tensor& self_buffer = self_ptr->get_unsafe_storage_as_tensor(),
        // 获取 self 的嵌套大小
        sizemat = self_ptr->get_nested_sizes();
    // 创建一个与 self 元素数量相同的空 Tensor，使用 self_buffer 的选项
    Tensor output_buffer = at::empty(self.numel(), self_buffer.options());
    // 使用 output_buffer 和 sizemat 创建一个包装后的 Tensor
    Tensor output = wrap_buffer(output_buffer, sizemat);
    // 对 self 和 output 进行解绑定
    std::vector<Tensor> self_unbind = self.unbind(),
        output_unbind = output.unbind();
    // 对 self 的每个元素进行复制到 output 的对应位置
    for (const int64_t i: c10::irange(self_ptr->size(0))) {
      output_unbind[i].copy_(self_unbind[i]);
    }
    // 返回处理后的 output
    return output;
  } else {
    // 如果内存格式既不是保留也不是连续，则抛出错误
    TORCH_CHECK(
        false,
        "Nested tensor clone supports Preserve and Contiguous memory formats, called clone with memory format: ",
        memory_format);
  }
# 将嵌套张量按照指定维度解绑定为张量列表
std::vector<at::Tensor> NestedTensor_unbind(
    const at::Tensor& self,  // 输入的嵌套张量
    int64_t dim) {           // 解绑定的维度
  // 检查维度是否为0，嵌套张量只能在第0维度上解绑定
  TORCH_CHECK(
      dim == 0,
      "NestedTensor can only be unbound along dimension 0 ",
      "got dimension ",
      dim,
      " instead.");
  // 获取嵌套张量的实现指针
  auto self_ptr = get_nested_tensor_impl(self);
  // 获取嵌套张量的张量数目
  int64_t ntensors = self_ptr->size(0);
  // 创建存储结果张量的向量
  std::vector<at::Tensor> result_tensors(ntensors);
  // 如果张量数目为0，直接返回空结果向量
  if (ntensors == 0) {
    return result_tensors;
  }
  // 获取嵌套张量底层存储的视图
  auto buffer = self.values();
  // 获取嵌套张量各个子张量的大小和步长信息
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
      strides = NestedTensor_get_strides(self_ptr);
  // 获取嵌套张量底层存储的偏移量指针
  int64_t *offsets_ptr = self_ptr->get_storage_offsets().data_ptr<int64_t>();
  // 对于每一个子张量的索引，将其存储为结果向量中的视图张量
  for (const int64_t i: c10::irange(ntensors)){
    result_tensors[i] = buffer.as_strided(sizes[i], strides[i], offsets_ptr[i]);
  }
  // 返回解绑定后的结果张量列表
  return result_tensors;
}

# 在嵌套符号整数张量上进行缩小操作
Tensor narrow_nested_symint(const at::Tensor& self, int64_t dim, SymInt start, SymInt length) {
  // 检查维度是否为0，嵌套张量只支持dim=0的缩小操作
  TORCH_CHECK(dim == 0, "narrow(): only dim=0 supported for nested tensors, but got: ", dim);
  // 检查长度是否为非负数
  TORCH_SYM_CHECK(length.sym_ge(0), "narrow(): length must be non-negative");
  // 获取当前维度的符号整数大小
  auto cur_size = self.sym_size(dim);
  // 检查起始位置是否在合理范围内
  TORCH_CHECK_INDEX(
      ((-cur_size).sym_le(start).sym_and(start.sym_le(cur_size))).expect_true(__FILE__, __LINE__),
      "start out of range (expected to be in range of [", -cur_size, ", ", cur_size, "], but got ",
      start, ")");
  // 如果起始位置小于0，则调整为非负数
  if (start < 0) {
    start = start + cur_size;
  }
  // 检查起始位置加长度是否超出维度大小
  TORCH_SYM_CHECK(start.sym_le(cur_size - length),
      "start (", start, ") + length (", length, ") exceeds dimension size (", cur_size, ").");
  // 获取嵌套张量实现指针
  auto *nt_impl = get_nested_tensor_impl(self);
  // 检查嵌套张量是否是连续的
  TORCH_CHECK(self.is_contiguous(), "narrow(): only contiguous nested tensors supported");
  // 获取嵌套张量底层存储作为张量的视图
  auto buffer = nt_impl->get_unsafe_storage_as_tensor();
  // 获取嵌套张量的嵌套尺寸和嵌套步长
  auto nested_sizes = nt_impl->get_nested_sizes();
  auto nested_strides = nt_impl->get_nested_strides();
  auto storage_offsets = nt_impl->get_storage_offsets();
  auto storage_offsets_ptr = storage_offsets.data_ptr<int64_t>();

  // 将嵌套尺寸、嵌套步长和存储偏移缩小为指定的范围
  nested_sizes = nested_sizes.narrow(0, start.guard_int(__FILE__, __LINE__), length.guard_int(__FILE__, __LINE__));
  nested_strides = nested_strides.narrow(0, start.guard_int(__FILE__, __LINE__), length.guard_int(__FILE__, __LINE__));
  storage_offsets = storage_offsets.narrow(0, start.guard_int(__FILE__, __LINE__), length.guard_int(__FILE__, __LINE__));

  // 使用缩小后的信息创建嵌套张量的视图
  return at::detail::make_tensor<NestedTensorImpl>(
      c10::TensorImpl::VIEW,
      buffer.narrow(0, storage_offsets_ptr[start.guard_int(__FILE__, __LINE__)], buffer.numel() - storage_offsets_ptr[start.guard_int(__FILE__, __LINE__)]),
      nested_sizes,
      nested_strides,
      storage_offsets);
}
// 从输入的 Tensor 中获取 NestedTensorImpl 指针
auto* nt_impl = get_nested_tensor_impl(self);

// 从 NestedTensorImpl 中获取底层存储的 Tensor
const at::Tensor& buffer = nt_impl->get_unsafe_storage_as_tensor();

// 从 NestedTensorImpl 中获取嵌套大小信息
const auto& nested_sizes = nt_impl->get_nested_sizes();

// 从 NestedTensorImpl 中获取嵌套步长信息
const auto& nested_strides = nt_impl->get_nested_strides();

// 从 NestedTensorImpl 中获取存储偏移量信息
const auto& storage_offsets = nt_impl->get_storage_offsets();

// 使用获取到的信息构造一个新的 NestedTensorImpl，并返回新的 Tensor
return at::detail::make_tensor<NestedTensorImpl>(
    c10::TensorImpl::VIEW,
    std::move(buffer),
    std::move(nested_sizes),
    std::move(nested_strides),
    std::move(storage_offsets));
```