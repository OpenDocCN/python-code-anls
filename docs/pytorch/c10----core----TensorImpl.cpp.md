# `.\pytorch\c10\core\TensorImpl.cpp`

```
#include <c10/core/TensorImpl.h>

#include <c10/core/Contiguity.h>
#include <c10/core/CopyBytes.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#include <utility>

// 定义一个布尔类型的命令行标志，用于控制在张量缩小大小时是否保留内存
C10_DEFINE_bool(
    caffe2_keep_on_shrink,
    true,
    "If set, keeps memory when a tensor is shrinking its size.");

// 定义一个整数类型的命令行标志，用于设置在张量缩小大小时要保留的最大内存
C10_DEFINE_int64(
    caffe2_max_keep_on_shrink_memory,
    LLONG_MAX,
    "The maximum memory in bytes to keep on shrink, if the difference between "
    "tensor sizes is bigger than this then tensor will be reset.");

// 命名空间 c10 的开始
namespace c10 {

// TensorImpl 类中的一个静态成员，定义错误消息常量指针
const char* const TensorImpl::err_msg_tensor_metadata_change_not_allowed =
    "is not allowed on a Tensor created from .data or .detach().\n"
    "If your intent is to change the metadata of a Tensor (such as sizes / strides / storage / storage_offset)\n"
    "without autograd tracking the change, remove the .data / .detach() call and wrap the change in a `with torch.no_grad():` block.\n"
    "For example, change:\n"
    "    x.data.set_(y)\n"
    "to:\n"
    "    with torch.no_grad():\n"
    "        x.set_(y)";

// TensorImpl 类中的 mutable_grad 方法实现，用于获取可变的梯度张量
at::Tensor& TensorImpl::mutable_grad() {
  // 如果 autograd_meta_ 为空，则使用工厂函数创建一个 AutogradMeta 对象
  if (!autograd_meta_)
    autograd_meta_ = impl::GetAutogradMetaFactory()->make();
  // 返回可变的梯度张量引用
  return autograd_meta_->mutable_grad();
}

// TensorImpl 类中的 grad 方法实现，用于获取梯度张量的常量引用
const at::Tensor& TensorImpl::grad() const {
  // 如果 autograd_meta_ 为空，返回未定义的张量对象，用于表示没有梯度
  // 这段注释解释了为什么返回一个常量引用的 Tensor
  if (!autograd_meta_)
    return impl::GetAutogradMetaFactory()->undefined_tensor();
  // 返回梯度张量的常量引用
  return autograd_meta_->grad();
}

// TensorImpl 类中的 _fw_grad 方法实现，用于前向传播的梯度计算
const at::Tensor& TensorImpl::_fw_grad(
    uint64_t level,
    const at::TensorBase& self) const {
  // 同样的逻辑，如果 autograd_meta_ 为空，返回未定义的张量对象
  if (!autograd_meta_)
    return impl::GetAutogradMetaFactory()->undefined_tensor();
  // 返回前向传播梯度计算的结果
  return autograd_meta_->fw_grad(level, self);
}

// TensorImpl 类中的 _set_fw_grad 方法实现，用于设置前向传播的梯度
void TensorImpl::_set_fw_grad(
    const at::TensorBase& new_grad,
    const at::TensorBase& self,
    uint64_t level,
    bool is_inplace_op) {
  // 如果 autograd_meta_ 为空，使用工厂函数创建一个 AutogradMeta 对象
  if (!autograd_meta_)
    autograd_meta_ = impl::GetAutogradMetaFactory()->make();
  // 设置前向传播梯度
  autograd_meta_->set_fw_grad(new_grad, self, level, is_inplace_op);
}

// TensorImpl 类的析构函数，默认实现
TensorImpl::~TensorImpl() = default;

// TensorImpl 类的构造函数，接受 Storage、DispatchKeySet 和数据类型作为参数
TensorImpl::TensorImpl(
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type)
    // 使用 std::forward 抑制静态分析器的误报
    // 构造函数的初始化列表
    # 使用给定的存储、键集、数据类型和设备创建 TensorImpl 对象的构造函数
    : TensorImpl(
          std::forward<Storage>(storage),
          key_set,
          data_type,
          storage.device()) {}
// [Note: Python key removal]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 在大多数 TensorImpl 的构造函数中，你会看到从传入的 DispatchKeySet 中移除 Python 和 PythonTLSSnapshot 键。为什么？
//
// INVARIANT: 只有当张量的 PyObject 具有非平凡的 __torch_dispatch__ 实现时，才设置 Python 和 PythonTLSSnapshot dispatch 键。
//
// 当创建一个新的 TensorImpl 时，是没有 PyObject 的（只有在张量第一次传递到 Python 时才会懒惰地初始化）。
// 因此，我们会违反不变式。
//
// 实际上，很快之后，TensorImpl 将通过 Tensor._make_subclass 初始化其 PyObject；在此时，Python 和 PythonTLSSnapshot dispatch 键将被设置，一切都很好。
// 关键是延迟直到这一点再设置 dispatch 键。

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 构造函数：初始化 TensorImpl 对象
TensorImpl::TensorImpl(
    ImplType type,                       // 类型
    Storage&& storage,                   // 存储
    DispatchKeySet key_set,              // DispatchKeySet
    const caffe2::TypeMeta data_type)    // 数据类型
    : storage_(std::move(storage)),      // 移动存储

      numel_(0),                         // 元素数量
      data_type_(data_type),             // 数据类型
      device_opt_(storage_.device()),    // 设备选项
      key_set_(key_set - c10::python_ks) // 设置 key_set（参见 [Note: Python key removal]）
{
  init_bitfields();                     // 初始化位字段

  // 推理张量没有版本计数器
  if (!is_inference()) {
    version_counter_ = VariableVersion(/*version=*/0);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 构造函数：使用指定 key_set 初始化 TensorImpl 对象
TensorImpl::TensorImpl(
    DispatchKeySet key_set,              // DispatchKeySet
    const caffe2::TypeMeta data_type,    // 数据类型
    std::optional<c10::Device> device_opt) // 可选设备
    : TensorImpl({}, key_set, data_type, device_opt) {}  // 调用另一个构造函数初始化

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 构造函数：使用存储、key_set、数据类型和可选设备初始化 TensorImpl 对象
TensorImpl::TensorImpl(
    Storage&& storage,                   // 存储
    DispatchKeySet key_set,              // DispatchKeySet
    const caffe2::TypeMeta data_type,    // 数据类型
    std::optional<c10::Device> device_opt) // 可选设备
    : storage_(std::move(storage)),      // 移动存储

      numel_(0),                         // 元素数量
      data_type_(data_type),             // 数据类型
      device_opt_(device_opt)            // 设备选项
{
  init_bitfields();                     // 初始化位字段

  if (!key_set.empty()) {
    TORCH_INTERNAL_ASSERT(
        data_type == ScalarType::Undefined || device_opt_.has_value());
    // UndefinedTensorImpl 是单例，因此我们跳过记录它
    C10_LOG_API_USAGE_ONCE("tensor.create");
  }

  // XXX: 如果在此处更新 keyset 逻辑，也请更新 _change_backend_component_keys
  bool inference_mode = c10::InferenceMode::is_enabled();

  // TODO: 在调用处更明确地指定完整的 key set，以免在这里反复计算
  auto k = key_set.highestBackendKey();

  key_set = key_set | getAutocastRelatedKeySetFromBackend(k);

  // 参见 [Note: Python key removal]
  key_set = key_set - c10::python_ks;

  // 推理张量不具有自动求导相关的键
  if (inference_mode) {
    // 详见 Note [Expected TLS state in InferenceMode]，解释为何我们排除 Autograd & ADInplaceOrView 键。通常 key_set 只包含 backend
    // 从 key_set 中减去 c10::autograd_dispatch_keyset_with_ADInplaceOrView 中的键，更新 key_set_
    key_set_ = key_set - c10::autograd_dispatch_keyset_with_ADInplaceOrView;
  } else {
    // TODO: 理想情况下，只有在张量需要梯度时才添加 AutogradBackend 键。
    //       参见注释 [Dream: skip VariableType kernel when requires_grad=false]
    // 从 getAutogradRelatedKeySetFromBackend(k) 获取与 Autograd 相关的键集合，与 key_set_ 合并更新 key_set_
    key_set_ = key_set | getAutogradRelatedKeySetFromBackend(k);
  }

  // 推断张量没有版本计数器。
  if (!is_inference()) {
    // 初始化版本计数器为 0
    version_counter_ = VariableVersion(/*version=*/0);
  }
  // 我们也想检查非 CPU 设备是否具有索引，但是一些 Caffe2 运算符使用默认设备创建存储。
}

// 在 TensorImpl 类中实现 _change_backend_component_keys 方法
void TensorImpl::_change_backend_component_keys(c10::Device device) {
  // 将设备类型转换为对应的后端组件
  BackendComponent new_backend = toBackendComponent(device.type());
  // 获取当前关键集合中最高级别的后端组件
  BackendComponent old_backend = key_set_.highestBackendKey();

  // 根据 TensorImpl::TensorImpl 中的逻辑，更新与设备相关的后端组件键

  // TODO: Autocoast 应该是每个后端功能的一个功能键，一旦更改完成，这个键的交换将不再必要。
  auto key_set =
      key_set_ - c10::getAutocastRelatedKeySetFromBackend(old_backend);
  key_set = key_set | c10::getAutocastRelatedKeySetFromBackend(new_backend);

  // 参见注释 [从 DispatchKeySet 中删除键仅影响功能键]
  key_set = key_set.remove_backend(old_backend);
  key_set_ = key_set | DispatchKeySet(new_backend);
}

// 在 TensorImpl 类中实现 HandleResize 方法
void TensorImpl::HandleResize() {
  // 如果需要，释放数据。下一个 mutable_data() 调用将创建数据存储。
  bool reset_tensor = false;
  if (reserved_) {
    // 如果张量被保留，则只有当 nbytes() 小于新大小时才申请其内存
    reset_tensor =
        storage_.nbytes() < (storage_offset_ + numel_) * data_type_.itemsize();
  } else {
    reset_tensor = storage_.nbytes() <
            (storage_offset_ + numel_) * data_type_.itemsize() ||
        !FLAGS_caffe2_keep_on_shrink ||
        storage_.nbytes() - (storage_offset_ + numel_) * data_type_.itemsize() >
            static_cast<size_t>(FLAGS_caffe2_max_keep_on_shrink_memory);
  }

  if (reset_tensor && storage_initialized()) {
    // 如果需要重置张量并且存储已初始化，则释放内存
    FreeMemory();
  }
}

// 在 TensorImpl 类中实现 compute_contiguous 方法
bool TensorImpl::compute_contiguous(identity<bool>) const {
  if (is_sparse()) {
    // 如果张量是稀疏的，则不是连续的
    return false;
  }
  // 计算是否连续
  return _compute_contiguous<int64_t>(
      sizes_and_strides_.sizes_arrayref(),
      sizes_and_strides_.strides_arrayref(),
      numel_);
}

// 在 TensorImpl 类中实现 compute_channels_last_contiguous_2d 方法
bool TensorImpl::compute_channels_last_contiguous_2d(identity<bool>) const {
  if (is_sparse()) {
    // 如果张量是稀疏的，则不是通道最后的二维连续的
    return false;
  }
  // 计算是否通道最后的二维连续
  return _compute_channels_last_contiguous_2d<int64_t>(
      sizes_and_strides_.sizes_arrayref(),
      sizes_and_strides_.strides_arrayref());
}

// 在 TensorImpl 类中实现 compute_channels_last_contiguous_3d 方法
bool TensorImpl::compute_channels_last_contiguous_3d(identity<bool>) const {
  if (is_sparse()) {
    // 如果张量是稀疏的，则不是通道最后的三维连续的
    return false;
  }
  // 计算是否通道最后的三维连续
  return _compute_channels_last_contiguous_3d<int64_t>(
      sizes_and_strides_.sizes_arrayref(),
      sizes_and_strides_.strides_arrayref());
}

// 在 TensorImpl 类中实现 compute_strides_like_channels_last_2d 方法
bool TensorImpl::compute_strides_like_channels_last_2d(identity<bool>) const {
  if (is_sparse()) {
    // 如果张量是稀疏的，则不是像通道最后二维那样的步幅
    return false;
  }
  // 计算是否像通道最后二维那样的步幅
  return is_channels_last_strides_2d<int64_t>(
      sizes_and_strides_.sizes_arrayref(),
      sizes_and_strides_.strides_arrayref());
}

// 在 TensorImpl 类中实现 compute_strides_like_channels_last_3d 方法
bool TensorImpl::compute_strides_like_channels_last_3d(identity<bool>) const {
  if (is_sparse()) {
    // 如果张量是稀疏的，则不是像通道最后三维那样的步幅
    return false;
  }
  // 计算是否像通道最后三维那样的步幅
  return is_channels_last_strides_3d<int64_t>(
      sizes_and_strides_.sizes_arrayref(),
      sizes_and_strides_.strides_arrayref());
}

// 在 TensorImpl 类中实现 compute_non_overlapping_and_dense 方法
bool TensorImpl::compute_non_overlapping_and_dense(identity<bool>) const {
  if (is_sparse()) {
    // 如果张量是稀疏的，则不是非重叠且稠密的
    return false;
  }
  // 计算是否非重叠且稠密
  return true;  // Assuming compute_non_overlapping_and_dense always returns true for non-sparse tensors
}
    # 返回 false，表示函数结束且未成功执行
    return false;
  }
  # 调用 _compute_non_overlapping_and_dense 函数，并传入 sizes_and_strides_.sizes_arrayref() 和 sizes_and_strides_.strides_arrayref() 作为参数
  # 返回该函数的计算结果
  return _compute_non_overlapping_and_dense<int64_t>(
      sizes_and_strides_.sizes_arrayref(),
      sizes_and_strides_.strides_arrayref());
}

void TensorImpl::release_resources() {
  // 重置自动微分元数据
  autograd_meta_.reset();
  // 如果存在存储空间，将其重置为空
  if (storage_) {
    storage_ = {};
  }
  // 可能销毁 Python 对象槽中的 Python 对象
  pyobj_slot_.maybe_destroy_pyobj();
}

#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
bool TensorImpl::has_storage() const {
  // 检查是否存在存储空间
  return storage_;
}
#endif

void TensorImpl::throw_cannot_call_with_symbolic(const char* meth) const {
  // 抛出异常，指示无法在具有符号尺寸/步幅的张量上调用特定方法
  TORCH_CHECK_ALWAYS_SHOW_CPP_STACKTRACE(
      false, "Cannot call ", meth, "() on tensor with symbolic sizes/strides");
}

void TensorImpl::throw_storage_access_error() const {
  if (extra_meta_ && extra_meta_->custom_storage_error_msg_) {
    // 检查是否有自定义存储访问错误消息，若有则抛出异常
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    TORCH_CHECK(false, *extra_meta_->custom_storage_error_msg_);
  }
  // 抛出未实现异常，指示无法访问张量的存储
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "Cannot access storage of ", tensorimpl_type_name());
}

void TensorImpl::throw_data_ptr_access_error() const {
  if (extra_meta_ && extra_meta_->custom_data_ptr_error_msg_) {
    // 检查是否有自定义数据指针访问错误消息，若有则抛出异常
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    TORCH_CHECK(false, *extra_meta_->custom_data_ptr_error_msg_);
  }
  // 抛出异常，指示无法访问没有存储的张量的数据指针
  TORCH_CHECK(
      false, "Cannot access data pointer of Tensor that doesn't have storage");
}

bool TensorImpl::is_contiguous_custom(at::MemoryFormat memory_format) const {
  // 如果符合自定义步幅策略，则使用 Python 对象解释器检查是否连续
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomStrides))) {
    return pyobj_slot_.load_pyobj_interpreter()->is_contiguous(
        this, memory_format);
  }
  // 否则使用默认方法检查是否连续
  return is_contiguous_default(memory_format);
}

bool TensorImpl::is_strides_like_custom(at::MemoryFormat memory_format) const {
  // 如果符合自定义步幅策略，则使用 Python 对象解释器检查是否类似指定的步幅格式
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomStrides))) {
    return pyobj_slot_.load_pyobj_interpreter()->is_strides_like(
        this, memory_format);
  }
  // 否则使用默认方法检查是否类似指定的步幅格式
  return is_strides_like_default(memory_format);
}

bool TensorImpl::is_non_overlapping_and_dense_custom() const {
  // 如果符合自定义步幅策略，则使用 Python 对象解释器检查是否非重叠且密集
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomStrides))) {
    return pyobj_slot_.load_pyobj_interpreter()->is_non_overlapping_and_dense(
        this);
  }
  // 否则使用默认方法检查是否非重叠且密集
  return is_non_overlapping_and_dense_default();
}

IntArrayRef TensorImpl::sizes_custom() const {
  // 如果符合自定义尺寸策略或者具有符号尺寸/步幅，则使用 Python 对象解释器获取尺寸
  if (C10_UNLIKELY(
          matches_python_custom(SizesStridesPolicy::CustomSizes) ||
          has_symbolic_sizes_strides_)) {
    return pyobj_slot_.load_pyobj_interpreter()->sizes(this);
  }
  // 否则使用默认方法获取尺寸
  return sizes_default();
}

c10::SymIntArrayRef TensorImpl::sym_sizes_custom() const {
  // 如果符合自定义尺寸策略，则使用 Python 对象解释器获取符号化的尺寸
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomSizes))) {
    return pyobj_slot_.load_pyobj_interpreter()->sym_sizes(this);
  }
  // 否则使用默认方法获取符号化的尺寸
  return sym_sizes_default();
}

c10::SymInt TensorImpl::sym_numel_custom() const {
  // 如果符合自定义尺寸策略，则使用 Python 对象解释器获取符号化的元素数量
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomSizes))) {
    return pyobj_slot_.load_pyobj_interpreter()->sym_numel(this);
  }
  // 否则使用默认方法获取符号化的元素数量
  return sym_numel_default();
}

c10::SymIntArrayRef TensorImpl::sym_strides_custom() const {
  // 如果符合自定义步幅策略，则使用 Python 对象解释器获取符号化的步幅
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomStrides))) {
    // 调用 pyobj_slot_ 的 load_pyobj_interpreter 方法，返回其结果，并传递给 sym_strides 方法
    return pyobj_slot_.load_pyobj_interpreter()->sym_strides(this);
  }
  // 如果未能调用上述方法，则调用默认的 sym_strides_default 方法
  return sym_strides_default();
}

c10::Device TensorImpl::device_custom() const {
  // 如果 python_custom_device_ 为真，则调用 Python 解释器获取自定义设备信息
  if (C10_UNLIKELY(python_custom_device_)) {
    return pyobj_slot_.load_pyobj_interpreter()->device(this);
  }
  // 否则返回默认设备信息
  return device_default();
}

IntArrayRef TensorImpl::strides_custom() const {
  // 如果符合自定义大小和步幅策略或者具有符号大小和步幅，则调用 Python 解释器获取步幅信息
  if (C10_UNLIKELY(
          matches_python_custom(SizesStridesPolicy::CustomStrides) ||
          has_symbolic_sizes_strides_)) {
    return pyobj_slot_.load_pyobj_interpreter()->strides(this);
  }
  // 否则返回默认步幅信息
  return strides_default();
}

int64_t TensorImpl::dim_custom() const {
  // 如果符合自定义大小策略，则调用 Python 解释器获取维度信息
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomSizes))) {
    return pyobj_slot_.load_pyobj_interpreter()->dim(this);
  }
  // 否则返回默认维度信息
  return dim_default();
}

int64_t TensorImpl::numel_custom() const {
  // 如果符合自定义大小策略，则调用 Python 解释器获取元素数量信息
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomSizes))) {
    return pyobj_slot_.load_pyobj_interpreter()->numel(this);
  }
  // 否则返回默认元素数量信息
  return numel_default();
}

c10::Layout TensorImpl::layout_custom() const {
  // 如果 python_custom_layout_ 为真，则调用 Python 解释器获取自定义布局信息
  if (C10_UNLIKELY(python_custom_layout_)) {
    return pyobj_slot_.load_pyobj_interpreter()->layout(this);
  }
  // 否则抛出错误信息，当前类型的张量没有布局信息
  // TODO: fix this
  TORCH_CHECK(
      0, "Tensors of type ", tensorimpl_type_name(), " do not have layout")
  // return layout_default();
}

int64_t TensorImpl::storage_offset_custom() const {
  // 如果符合自定义大小策略，则调用 Python 解释器获取符号存储偏移信息
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomSizes))) {
    // TODO: fix this
    return pyobj_slot_.load_pyobj_interpreter()
        ->sym_storage_offset(this)
        .guard_int(__FILE__, __LINE__);
  }
  // 否则返回默认存储偏移信息
  return storage_offset_default();
}

c10::SymInt TensorImpl::sym_storage_offset_custom() const {
  // 如果符合自定义大小策略，则调用 Python 解释器获取符号存储偏移信息
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomSizes))) {
    return pyobj_slot_.load_pyobj_interpreter()->sym_storage_offset(this);
  }
  // 否则返回默认符号存储偏移信息
  return sym_storage_offset_default();
}

static void deletePlacementDeleteContext(void* ptr) {
  delete static_cast<PlacementDeleteContext*>(ptr);
}

at::DataPtr PlacementDeleteContext::makeDataPtr(
    at::DataPtr&& data_ptr,
    PlacementDtor placement_dtor,
    size_t size,
    at::Device device) {
  auto* ptr = data_ptr.get();
  // 创建包含自定义删除上下文的数据指针
  return {
      ptr,
      new PlacementDeleteContext(std::move(data_ptr), placement_dtor, size),
      &deletePlacementDeleteContext,
      device};
}

AutogradMetaInterface::~AutogradMetaInterface() = default;

// 在推断模式外部将 requires_grad 设置为 true 是禁止的。
// 理想情况下，它在推断模式内部也应该是非法的。
// 但是在 C++ 构造函数中无法直接分配一个具有 requires_grad = true 的张量，
// 因此 set_requires_grad 在 C++ 前端广泛使用。
// 在推断模式内部禁止这些设置器代码会迫使用户在其代码中删除这些设置，这不是理想的做法。
// 设置张量是否需要梯度，进行必要的检查
void TensorImpl::set_requires_grad(bool requires_grad) {
  // 检查是否同时设置了需要梯度、处于推断模式且未启用推断模式
  TORCH_CHECK(
      !(requires_grad && is_inference() && !c10::InferenceMode::is_enabled()),
      "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed.");
  // 如果不需要梯度且没有自动求导元数据，则直接返回
  if (!requires_grad && !autograd_meta_)
    return;
  // 如果没有自动求导元数据，则创建一个默认的自动求导元数据
  if (!autograd_meta_)
    autograd_meta_ = impl::GetAutogradMetaFactory()->make();
  // 注意：原则上，将 requires_grad 设置为 false 可能会导致 AutogradMeta 变为默认构造状态，
  // 在这种情况下，我们可以应用空指针 AutogradMeta 优化（参见 autograd_meta_ 文档）。
  // 但目前我们没有这样做。请注意，无条件将 AutogradMeta 设置为 false 是不安全的，
  // 因为其他字段可能包含非平凡信息；例如，我们可能为变量设置了字符串名称，或者可能已为其注册了钩子。
  autograd_meta_->set_requires_grad(requires_grad, this);
}

// 查询张量是否需要梯度
bool TensorImpl::requires_grad() const {
  // 如果没有自动求导元数据，则不需要梯度
  if (!autograd_meta_)
    return false;
  // 返回自动求导元数据中的 requires_grad 值
  return autograd_meta_->requires_grad();
}

// 设置张量的自动求导元数据
void TensorImpl::set_autograd_meta(
    std::unique_ptr<c10::AutogradMetaInterface> autograd_meta) {
  // 注意：autograd_meta 可能为空！这表示它是默认构造的
  autograd_meta_ = std::move(autograd_meta);
}

// 获取张量的自动求导元数据指针
c10::AutogradMetaInterface* TensorImpl::autograd_meta() const {
  // 注意：可能返回空指针！
  return autograd_meta_.get();
}

// 浅拷贝并分离张量的核心操作，用于变量版本控制和允许张量元数据更改的情况
template <typename VariableVersion>
c10::intrusive_ptr<TensorImpl> TensorImpl::shallow_copy_and_detach_core(
    VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  c10::intrusive_ptr<TensorImpl> r;
  // 获取当前 Torch 分发模式的堆栈长度
  const auto mode_stack_len = c10::impl::TorchDispatchModeTLS::stack_len();
  // TODO: 是否需要在 Python 分发键设置后排除？
  if (mode_stack_len > 0 &&
      !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
    // 获取当前 Torch 分发模式状态的 Python 解释器并分离张量
    const auto& cur_torch_dispatch_mode_state =
        c10::impl::TorchDispatchModeTLS::get_stack_at(mode_stack_len - 1);
    r = cur_torch_dispatch_mode_state->pyinterpreter()->detach(this);
  } else if (
      key_set_.has(DispatchKey::Python) &&
      !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
    // 如果具有 Python 分发键，则获取 Python 对象槽的 Python 解释器并分离张量
    r = (pyobj_slot_.load_pyobj_interpreter())->detach(this);
  }
  // 如果成功分离张量，则设置版本计数器和允许张量元数据更改标志
  if (r) {
    r->set_version_counter(std::forward<VariableVersion>(version_counter));
    r->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  // 返回当前对象的引用，结束函数执行
  return r;
}
// 否则，复制 TensorImpl 而不是 PyObject。因为解释器已经停止运行，没有人可以调用我们
auto impl = c10::make_intrusive<TensorImpl>(
    // 不需要填充 Storage；copy_tensor_metadata 函数会为我们完成
    key_set_,
    data_type_,
    device_opt_);
copy_tensor_metadata(
    /*src_impl=*/this,
    /*dest_impl=*/impl.get(),
    /*version_counter=*/std::forward<VariableVersion>(version_counter),
    /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
// 返回新创建的 TensorImpl 对象
return impl;
}

c10::intrusive_ptr<TensorImpl> TensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  // 返回调用深拷贝和分离核心函数处理后的结果
  return shallow_copy_and_detach_core(
      version_counter, allow_tensor_metadata_change);
}

c10::intrusive_ptr<TensorImpl> TensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  // 返回调用深拷贝和分离核心函数处理后的结果，移动语义
  return shallow_copy_and_detach_core(
      std::move(version_counter), allow_tensor_metadata_change);
}

// 此函数复制源张量的所有元数据，但排除以下内容：
// - key_set_
// - storage_
// - storage_access_should_throw_
// - sizes_strides_policy_
// - version_counter_
// - allow_tensor_metadata_change_
// 思路是，如果我们有一个“包装张量”（如在功能化中），
// 所有上述内容都是包装器希望自定义的属性，而其余所有内容应在包装器和内部张量之间进行镜像。
void TensorImpl::copy_generic_tensor_metadata(
    const TensorImpl* src_impl,
    TensorImpl* dest_impl) {
  // 复制大小和步幅信息
  dest_impl->sizes_and_strides_ = src_impl->sizes_and_strides_;
  dest_impl->has_symbolic_sizes_strides_ =
      src_impl->has_symbolic_sizes_strides_;

  // 复制存储偏移量、数据类型、设备信息等
  dest_impl->storage_offset_ = src_impl->storage_offset_;
  dest_impl->data_type_ = src_impl->data_type_;
  dest_impl->device_opt_ = src_impl->device_opt_;
  dest_impl->is_contiguous_ = src_impl->is_contiguous_;
  dest_impl->is_channels_last_contiguous_ =
      src_impl->is_channels_last_contiguous_;
  dest_impl->is_channels_last_3d_contiguous_ =
      src_impl->is_channels_last_3d_contiguous_;
  dest_impl->is_channels_last_ = src_impl->is_channels_last_;
  dest_impl->is_channels_last_3d_ = src_impl->is_channels_last_3d_;
  dest_impl->is_non_overlapping_and_dense_ =
      src_impl->is_non_overlapping_and_dense_;
  dest_impl->is_wrapped_number_ = src_impl->is_wrapped_number_;
  dest_impl->reserved_ = src_impl->reserved_;
  dest_impl->numel_ = src_impl->numel_;
  
  // 如果源张量有额外元数据，则进行克隆
  if (src_impl->extra_meta_ != nullptr) {
    dest_impl->extra_meta_ = src_impl->extra_meta_->clone();
  } else if (dest_impl->extra_meta_ != nullptr) {
    // 清除目标张量的额外元数据，因为浅复制来自目标实现是一个真实的张量实现，
    // 它可能会带有额外的元数据。这些信息将污染新的目标实现的元数据信息。
    dest_impl->extra_meta_.reset(nullptr);
  }

  // 刷新大小和步幅策略
  dest_impl->refresh_sizes_strides_policy();
  dest_impl->refresh_layout_policy();
  dest_impl->refresh_device_policy();
}

void TensorImpl::copy_tensor_metadata_except_version_counter(
    const TensorImpl* src_impl,
    TensorImpl* dest_impl,
    bool allow_tensor_metadata_change) {
  // 首先调用通用的复制函数，复制张量的元数据
  copy_generic_tensor_metadata(src_impl, dest_impl);
  // 然后复制其余部分（参见copy_generic_tensor_metadata中的注释，
  // 列出它没有直接复制的元数据）
  // 将源张量的存储指针赋值给目标张量的存储指针
  dest_impl->storage_ = src_impl->storage_;
  // 复制张量元数据不会改变 PyObject（也许应该改变），这意味着我们必须保留
  // 原始 Python 键集（因为它与 PyObject 是张量子类或非张量子类相关联的）
  dest_impl->key_set_ = (src_impl->key_set_ - c10::python_ks) |
      (dest_impl->key_set_ & c10::python_ks);
  // 设置是否允许修改张量元数据的标志
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  // 复制源张量是否应抛出存储访问异常的设置
  dest_impl->storage_access_should_throw_ =
      src_impl->storage_access_should_throw_;
// 复制张量的元数据到目标张量的实现函数
void TensorImpl::copy_tensor_metadata(
    const TensorImpl* src_impl,
    TensorImpl* dest_impl,
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) {
  // 调用函数复制张量的元数据，除了版本计数器
  copy_tensor_metadata_except_version_counter(
      src_impl, dest_impl, allow_tensor_metadata_change);
  
  // 如果目标张量不是推断张量，则设置版本计数器
  // TODO: 在理想的最终状态下，在推断张量上设置禁用的版本计数器是可以的，因为它是一个空操作。
  // 这需要调用站点上的重构。
  if (!dest_impl->is_inference()) {
    dest_impl->set_version_counter(version_counter);
  }
}

// 移动语义版本的复制张量元数据到目标张量的实现函数
void TensorImpl::copy_tensor_metadata(
    const TensorImpl* src_impl,
    TensorImpl* dest_impl,
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) {
  // 调用函数复制张量的元数据，除了版本计数器
  copy_tensor_metadata_except_version_counter(
      src_impl, dest_impl, allow_tensor_metadata_change);
  
  // 如果目标张量不是推断张量，则设置移动语义版本的版本计数器
  if (!dest_impl->is_inference()) {
    dest_impl->set_version_counter(std::move(version_counter));
  }
}

// 扩展张量的大小（适用于 Legacy Caffe2 操作）
void TensorImpl::Extend(int64_t num, float growthPct) {
  // 检查张量维度和步长数量至少为 1
  TORCH_CHECK(sizes_and_strides_.size() >= 1u);
  // 检查 num 参数必须为非负数
  TORCH_CHECK(num >= 0, "`num` must be non-negative for Extend");
  // 检查当前张量是否是连续的
  TORCH_CHECK(
      is_contiguous_,
      "Right now Extend is only supported for contiguous Tensor.");
  // 检查是否具有符号形状大小和步长
  TORCH_CHECK(
      !has_symbolic_sizes_strides_,
      "Extend() called on tensor with symbolic shape")

  // 使用 SmallVector 存储张量的大小和步长
  using SizesVector = SmallVector<int64_t, 5>;
  IntArrayRef sizes_and_strides = sizes_and_strides_.sizes_arrayref();
  SizesVector newDims(sizes_and_strides.begin(), sizes_and_strides.end());
  newDims[0] += num;

  // 如果存储数据为空，则调整大小并返回
  if (!storage_.data()) {
    Resize(newDims);
    return;
  }

  // 计算新的元素数量并检查是否可以容纳
  const auto newNumel = c10::multiply_integers(newDims.begin(), newDims.end());
  if (newNumel * data_type_.itemsize() <= storage_.nbytes()) {
    sizes_and_strides_.set_sizes(newDims);
    numel_ = newNumel;
    return;
  }

  // 计算新的容量大小并重新分配内存
  SizesVector newCapacity(sizes_and_strides.begin(), sizes_and_strides.end());
  newCapacity[0] = std::max(
      newDims[0],
      static_cast<int64_t>(std::ceil(
          static_cast<float>(sizes_and_strides_.size_at_unchecked(0)) *
          (1 + growthPct / 100))));
  auto oldData = std::move(storage_.mutable_data_ptr());
  auto oldSize = numel_;
  Resize(std::move(newCapacity));

  // 执行数据复制操作
  auto* newData = raw_mutable_data(data_type_);
  if (data_type_.copy()) {
    // 检查是否为 CPU 设备，非 POD 类型仅在 CPU 上工作
    TORCH_CHECK(
        device_type() == DeviceType::CPU, "non-POD types work only on CPU");
    // 使用数据类型的复制函数进行复制
    data_type_.copy()(oldData.get(), newData, oldSize);
  } else {
    // 否则，使用当前线程本地流进行复制，同时使用传入的设备 ID
    //
    // TODO: 可能需要更多的强制执行来避免意外切换到同步复制，如果当前设置的设备错误。
    //
    // 具体来说，可能需要在这里显式切换到不同的上下文设备，以避免依赖用户适当同步的问题。
    CopyBytes(
        oldSize * itemsize(),  // 计算要复制的字节数，乘以元素大小
        oldData.get(),         // 获取旧数据的指针
        device(),              // 获取当前设备信息
        newData,               // 新数据的指针
        device(),              // 获取当前设备信息
        true);                 // 使用非阻塞模式进行复制
  }
  reserved_ = true;            // 将 reserved_ 标记设为 true，表示已经保留
  sizes_and_strides_.set_sizes(newDims);  // 更新尺寸和步幅信息为新的维度
  numel_ = newNumel;           // 更新元素数量为新的 numel
}

void TensorImpl::ReserveSpace(int64_t outer_dim) {
  // 检查张量是否是连续的
  TORCH_CHECK(
      is_contiguous_,
      "Right now ReserveSpace is only supported for contiguous Tensor.");
  // 检查张量是否有符号形状和步幅
  TORCH_CHECK(
      !has_symbolic_sizes_strides_,
      "ReserveSpace() called on tensor with symbolic shape")

  // 检查存储是否唯一，不能在共享存储上调用ReserveSpace
  TORCH_CHECK(storage_.unique(), "Can't call ReserveSpace on shared storage.");
  
  // 创建一个新的容量数组，复制现有的大小和步幅
  IntArrayRef sizes_and_strides = sizes_and_strides_.sizes_arrayref();
  SmallVector<int64_t, 5> newCapacity(
      sizes_and_strides.begin(), sizes_and_strides.end());
  newCapacity[0] = outer_dim;
  auto newNumel = c10::multiply_integers(newCapacity);

  // 如果新的元素个数乘以数据类型的字节大小小于等于存储的总字节数，直接返回
  if (newNumel * data_type_.itemsize() <= storage_.nbytes()) {
    return;
  }

  // 释放旧数据
  storage_.mutable_data_ptr().clear();

  // 保存旧的大小信息
  auto oldSize = numel_;
  SmallVector<int64_t, 5> oldDims(
      sizes_and_strides.begin(), sizes_and_strides.end());

  // 调整张量的大小为新的容量，不复制数据
  Resize(std::move(newCapacity));

  // 分配新的内存，但不复制数据
  raw_mutable_data(data_type_);

  // 恢复旧的大小信息和步幅
  sizes_and_strides_.set_sizes(oldDims);
  numel_ = oldSize;
  reserved_ = true;
}

void TensorImpl::Reshape(const std::vector<int64_t>& dims) {
  // 检查张量是否是连续的
  TORCH_CHECK(
      is_contiguous_,
      "Right now Reshape is only supported for contiguous Tensor.");
  // 检查张量是否有符号形状和步幅
  TORCH_CHECK(
      !has_symbolic_sizes_strides_,
      "Reshape() called on tensor with symbolic shape")

  // 计算新形状的元素总数
  int64_t new_size = 1;
  for (auto d : dims) {
    TORCH_CHECK(d >= 0);
    new_size *= d;
  }

  // 检查新的元素总数与旧的是否相等，否则提示使用Resize而不是Reshape
  TORCH_CHECK(
      new_size == numel_,
      "New size and old size are not equal. You cannot use Reshape, "
      "but should use Resize."
      // TODO(jiayq): remove the following warning after pending diffs
      // stabilize.
      " The old caffe2 mixes Reshape and Resize but this behavior has "
      "been changed. If you find this error, most likely you will need "
      "to change corresponding code from Reshape to Resize.");

  // 设置新的大小和步幅
  sizes_and_strides_.set_sizes(dims);

  // 重新设置张量的步幅以保持连续存储格式
  empty_tensor_restride(MemoryFormat::Contiguous);
}

void TensorImpl::FreeMemory() {
  // 如果存储的使用计数不为1，或者存储不可调整大小，或者没有分配器，则创建一个新的存储
  if (storage_.use_count() != 1 || !storage_.resizable() ||
      !storage_.allocator()) {
    storage_ = Storage::create_legacy(storage_.device());
  } else {
    // 否则，重置存储
    storage_.reset_legacy();
  }
  // 重置存储偏移为0
  storage_offset_ = 0;
}
// 在当前实现中，假设设备类型相同，因为在非模板化代码中这是内在的条件。可能需要添加断言来验证这一点，尽管这可能会稍微影响性能。
void TensorImpl::ShareData(const TensorImpl& src) {
  TORCH_CHECK(
      src.numel_ == numel_,
      "Size mismatch - did you call reshape before sharing the data?");
  // 如果源张量尚未调用 mutable_data()，那么 ShareData() 操作没有实际意义，因为我们不知道要共享什么数据。
  if (!src.dtype_initialized()) {
    // 如果数据类型未初始化，则输出警告信息，建议调用 mutable_data<T> 初始化数据类型。
    C10_LOG_EVERY_MS(WARNING, 1000)
        << "Source tensor don't have a data type (did you call mutable_data<T> on the tensor?)";
  }
  TORCH_CHECK(
      src.storage_initialized(),
      "Source tensor has no content and has size > 0");
  // 最后进行数据共享操作。
  /* 由于我们在需要改变数据类型或字节大小时创建新的 Storage，这仍然保持了原始的语义 */
  storage_ = src.storage();
  data_type_ = src.dtype();
  device_opt_ = src.device_opt();
  storage_offset_ = src.storage_offset();
}

// 与原始外部指针共享数据
void TensorImpl::ShareExternalPointer(
    DataPtr&& data_ptr,
    const caffe2::TypeMeta data_type,
    size_t size_bytes) {
  TORCH_CHECK(
      data_type != ScalarType::Undefined,
      "To share with a raw external pointer you need to pass in an "
      "initialized data_type(TypeMeta).");
  TORCH_CHECK(
      !has_symbolic_sizes_strides_,
      "ShareExternalPointer() called on tensor with symbolic shape");
  // 如果 size_bytes 为零，则计算出总字节数
  if (!size_bytes) {
    size_bytes = numel_ * data_type.itemsize();
  }
  // 如果当前存储唯一，直接共享外部指针；否则创建一个新的 Storage 对象
  if (storage_.unique()) {
    storage_.UniqueStorageShareExternalPointer(std::move(data_ptr), size_bytes);
    data_type_ = data_type;
    device_opt_ = storage_.device();
    storage_offset_ = 0;
  } else {
    // 创建一个新的 Storage 对象，用于存储外部指针的数据
    storage_ = Storage(
        Storage::use_byte_size_t(),
        size_bytes,
        std::move(data_ptr),
        /*allocator=*/nullptr,
        /*resizable=*/false);
    data_type_ = data_type;
    device_opt_ = storage_.device();
    storage_offset_ = 0;
  }
}

// 克隆符号整数数组
static void clone_symvec(SymIntArrayRef src, SymDimVector& dst) {
  // 清空目标向量，准备进行克隆操作
  dst.clear();
  // 预留足够的空间以容纳源向量的大小
  dst.reserve(src.size());
  // 遍历源向量，对每个元素进行克隆并添加到目标向量中
  for (const auto& i : src) {
    dst.emplace_back(i.clone());
  }
}

// 注意：此函数不检查存储中的大小/步幅/偏移是否符合边界条件，有时我们会临时违反不变条件，先设置大小/步幅，然后更新存储
void TensorImpl::set_sizes_and_strides(
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides,
    ...
  // 使用 std::optional 包装的 c10::SymInt 对象作为参数，表示存储偏移量的可选值
  // 通过 asIntArrayRefSlowOpt 函数尝试将 sizes 转换为整数数组引用
  auto int_sizes = asIntArrayRefSlowOpt(sizes);
  // 通过 asIntArrayRefSlowOpt 函数尝试将 strides 转换为整数数组引用
  auto int_strides = asIntArrayRefSlowOpt(strides);

  // 检查 sizes 和 strides 是否成功转换为整数数组引用，并且满足一些条件
  if (int_sizes && int_strides &&
      // 注意：此处保证 storage_offset 是正值
      (!storage_offset.has_value() || !storage_offset->is_heap_allocated()) &&
      !has_symbolic_sizes_strides_) {
    // 设置 tensor 的 sizes 和 strides
    set_sizes_and_strides(*int_sizes, *int_strides);
    // 如果 storage_offset 有值，则设置存储偏移量
    if (storage_offset.has_value())
      set_storage_offset(storage_offset->as_int_unchecked());
    return;
  }

  // 如果不允许修改张量的元数据，则抛出异常
  TORCH_CHECK(
      allow_tensor_metadata_change(),
      "set_sizes_and_strides ",
      err_msg_tensor_metadata_change_not_allowed);

  // 标记张量具有符号形状的 sizes 和 strides
  has_symbolic_sizes_strides_ = true;
  // 更新 sizes 和 strides 的策略
  refresh_sizes_strides_policy();

  // 如果 extra_meta_ 为空，则创建并初始化 extra_meta_ 对象及其成员
  if (!extra_meta_) {
    extra_meta_ = std::make_unique<ExtraMeta>();
    extra_meta_->symbolic_shape_meta_ =
        std::make_unique<c10::SymbolicShapeMeta>();
    // 根据张量是否稀疏，设置符号形状的 strides 是否有效
    extra_meta_->symbolic_shape_meta_->strides_valid_ = !is_sparse();
    // 如果 storage_offset 没有值，则设置 storage_offset_ 的初始值
    if (!storage_offset.has_value()) {
      extra_meta_->symbolic_shape_meta_->storage_offset_ = storage_offset_;
    }
  }

  // 获取符号形状的元数据的引用
  auto& sym_shape_meta{symbolic_shape_meta()};
  // 克隆 sizes 到 sym_shape_meta 的 sizes_
  clone_symvec(sizes, sym_shape_meta.sizes_);
  // 克隆 strides 到 sym_shape_meta 的 strides_
  clone_symvec(strides, sym_shape_meta.strides_);
  // 如果 storage_offset 有值，则克隆 storage_offset 到 sym_shape_meta 的 storage_offset_
  if (storage_offset.has_value())
    sym_shape_meta.storage_offset_ = storage_offset->clone();

  // 刷新张量的元素数量
  refresh_numel();
  // 刷新张量的连续性
  refresh_contiguous();
}

void TensorImpl::generic_set_sizes_contiguous(SymIntArrayRef sizes) {
  auto int_sizes = asIntArrayRefSlowOpt(sizes);  // 尝试将sizes转换为IntArrayRef类型
  if (int_sizes.has_value()) {  // 如果成功转换，使用具体的整数尺寸设置函数
    set_sizes_contiguous(*int_sizes);
    return;
  }

  TORCH_CHECK(
      allow_tensor_metadata_change(),  // 检查是否允许张量元数据更改
      "generic_set_sizes_contiguous ",
      err_msg_tensor_metadata_change_not_allowed);

  has_symbolic_sizes_strides_ = true;  // 设置标志，表示存在符号化尺寸和步幅
  refresh_sizes_strides_policy();  // 刷新尺寸和步幅的策略
  auto& extra_meta{get_extra_meta()};  // 获取额外的元数据引用
  if (extra_meta.symbolic_shape_meta_ == nullptr) {  // 如果符号化形状元数据为空，则创建新的符号化形状元数据
    extra_meta_->symbolic_shape_meta_ =
        std::make_unique<c10::SymbolicShapeMeta>();
    extra_meta_->symbolic_shape_meta_->strides_valid_ = !is_sparse();  // 设置步幅是否有效的标志
  }

  clone_symvec(sizes, symbolic_shape_meta().sizes_);  // 克隆符号化向量到符号化形状元数据的尺寸
  refresh_numel();  // 刷新张量元素数目
  empty_tensor_restride_symint(
      MemoryFormat::Contiguous); // 调用空张量重新布局符号整数版本，使用连续内存格式
}

void TensorImpl::empty_tensor_restride_symint(MemoryFormat memory_format) {
  TORCH_INTERNAL_ASSERT(has_symbolic_sizes_strides_);  // 内部断言，确保存在符号化尺寸和步幅
  auto& sym_shape_meta{symbolic_shape_meta()};  // 获取符号化形状元数据的引用
  switch (memory_format) {
    case MemoryFormat::Contiguous: {
      // TODO: figure out if the non-symint version can also devirtualize;
      // the last time we tried it was probably a narrowing problem
      const auto dim_ = sym_shape_meta.dim();  // 获取符号化形状元数据的维度
      sym_shape_meta.strides_.resize(dim_);  // 调整步幅的大小为维度大小
      if (dim_ > 0) {
        const auto last_idx = dim_ - 1;  // 最后一个索引
        sym_shape_meta.strides_[last_idx] = c10::SymInt(1);  // 最后一个维度的步幅为1
        for (auto i = last_idx - 1; i >= 0; --i) {
          sym_shape_meta.strides_[i] = sym_shape_meta.strides_[i + 1] *
              sym_shape_meta.sizes_[i + 1].max(1);  // 计算每个维度的步幅
        }
      }
      break;
    }
    case MemoryFormat::ChannelsLast: {
      TORCH_CHECK(
          dim() == 4, "required rank 4 tensor to use channels_last format");  // 检查张量的维度是否为4
      clone_symvec(
          get_channels_last_strides_2d(sym_sizes()), sym_shape_meta.strides_);  // 克隆通道最后2D步幅
      break;
    }
    case MemoryFormat::ChannelsLast3d: {
      TORCH_CHECK(
          dim() == 5, "required rank 5 tensor to use channels_last_3d format");  // 检查张量的维度是否为5
      clone_symvec(
          get_channels_last_strides_3d(sym_sizes()), sym_shape_meta.strides_);  // 克隆通道最后3D步幅
      break;
    }
    case MemoryFormat::Preserve:
      TORCH_CHECK(false, "unsupported memory format ", memory_format);  // 不支持的内存格式，抛出错误
      // Cleaning warning messages, no need to break as TORCH_CHECK(false)
      // terminates flow.
      // break;
    case MemoryFormat::NumOptions:
      TORCH_INTERNAL_ASSERT(false, "invalid memory format ", memory_format);  // 无效的内存格式，内部断言错误
  }
  // 重新计算连续标志，因为当前的NHWC/NCHW标志不是互斥的，参见＃24090
  refresh_contiguous();  // 刷新连续标志
  // hard code some known true settings, for unbacked case
  // TODO: avoid chundering into the guards for computing these
  switch (memory_format) {
    case MemoryFormat::Contiguous: {
      sym_shape_meta.assume_contiguous();  // 假定为连续的张量
      sym_shape_meta.assume_non_overlapping_and_dense();  // 假定为非重叠且稠密的张量
      break;
    }
    case MemoryFormat::ChannelsLast: {
      // 假设数据是按照通道在最后的顺序连续存储
      sym_shape_meta.assume_channels_last_contiguous();
      // 假设数据是按照通道在最后的顺序存储
      sym_shape_meta.assume_channels_last();
      // 假设数据是非重叠且稠密存储
      sym_shape_meta.assume_non_overlapping_and_dense();
      break;
    }
    case MemoryFormat::ChannelsLast3d: {
      // 假设数据是按照三维通道在最后的顺序连续存储
      sym_shape_meta.assume_channels_last_3d_contiguous();
      // 假设数据是按照三维通道在最后的顺序存储
      sym_shape_meta.assume_channels_last_3d();
      // 假设数据是非重叠且稠密存储
      sym_shape_meta.assume_non_overlapping_and_dense();
      break;
    }
    default:
      break;
  }
}

namespace impl {

namespace {
AutogradMetaFactory* meta_factory = nullptr;
} // namespace

void SetAutogradMetaFactory(AutogradMetaFactory* factory) {
  meta_factory = factory;
}
AutogradMetaFactory* GetAutogradMetaFactory() {
  // 检查 meta_factory 是否为空指针，若为空则抛出错误信息
  TORCH_CHECK(
      meta_factory,
      "Support for autograd has not been loaded; have you linked against libtorch.so?")
  // 返回当前设置的 AutogradMetaFactory 指针
  return meta_factory;
}

} // namespace impl

} // namespace c10
```