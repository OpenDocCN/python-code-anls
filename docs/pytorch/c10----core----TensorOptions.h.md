# `.\pytorch\c10\core\TensorOptions.h`

```py
#pragma once

#include <c10/core/Backend.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <cstdint>
#include <iosfwd>
#include <string>
#include <type_traits>
#include <utility>

namespace c10 {

// 计算并返回分发键（DispatchKey），根据输入的dtype、layout和device的可选值
DispatchKey computeDispatchKey(
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device);

// 如果给定dtype为null，则返回默认的ScalarType
inline ScalarType dtype_or_default(std::optional<ScalarType> dtype) {
  return value_or_else(dtype, [] { return get_default_dtype_as_scalartype(); });
}

// 如果给定dtype为null，则返回默认的caffe2::TypeMeta
inline caffe2::TypeMeta dtype_or_default(
    std::optional<caffe2::TypeMeta> dtype) {
  return value_or_else(dtype, [] { return get_default_dtype(); });
}

// 如果给定layout为null，则返回默认的布局（kStrided）
inline Layout layout_or_default(std::optional<Layout> layout) {
  return layout.value_or(kStrided);
}

// 如果给定device为null，则返回默认的设备（kCPU）
inline Device device_or_default(std::optional<Device> device) {
  return value_or_else(device, [] { return Device(kCPU); });
}

// 如果给定pinned_memory为null，则返回false；否则返回其值
inline bool pinned_memory_or_default(std::optional<bool> pinned_memory) {
  return pinned_memory.value_or(false);
}

/// 一个用于封装Tensor构造轴的类。TensorOptions被设计为支持Python风格API的工厂函数构造选项，
/// 例如，
///
///     torch.zeros(2, 3, dtype=torch.int32)
///
/// 因为C++本身不支持关键字参数，必须有另一种方式来指定类似关键字的参数。
/// TensorOptions是一个构建器类，可用于构建这种关键字参数的“字典”：支持TensorOptions约定的函数通常将其作为最后一个可选参数。
///
/// 警告：在PyTorch中，存在`torch::`变体的工厂函数，例如，torch::zeros对应at::zeros。
/// 这些函数返回Variables（而标准的ATen函数返回普通Tensors）。如果混淆使用这些函数，你会很沮丧。
///
/// 你应该优先使用构造函数，并在其上链式调用设置方法，而不是直接使用此类的构造函数。
///
///     at::device(at::kCUDA).dtype(kInt)
///     at::dtype(at::kInt)
///
/// 此外，任何需要TensorOptions的地方，你都可以直接传递at::kCUDA / at::kInt，并且它会隐式转换为TensorOptions。
///
/// 这里是一些推荐的创建具有特定属性的2x2零张量的方法。即使它们没有明确提到TensorOptions类，它们都*隐式*使用了TensorOptions：
///
///     at::zeros({2,2}, at::kCUDA);
///     at::zeros({2,2}, at::kLong);
///     at::zeros({2,2}, at::device(at::kCUDA).dtype(at::kLong()));
///
class TensorOptions {
  // 该类包含了有关如何构造张量的选项，使得在使用工厂函数时可以像Python风格API那样指定构造选项。
};

} // namespace c10
///     at::zeros({2,2}, at::device({at::kCUDA, 1})); // 在设备 1 上创建一个大小为 2x2 的零张量
///     at::zeros({2,2}, at::requires_grad());          // 创建一个大小为 2x2 的零张量，并要求梯度跟踪
///

/// NOTE [ TensorOptions Constructors ]
///
/// TensorOptions 是一个类似于字典的结构，包含如下条目：{requires_grad, device, dtype, layout}，
/// 其中每个条目都可以是可选的（即可不指定）。在许多地方，包括 C++ 内部和 API 中，用于指定张量的属性，
/// 例如张量工厂方法如 `at::empty({10}, options)`，张量转换如 `tensor.to(...)` 等。
///
/// 为了提供与 Python 一致的简单 API，例如可以使用 `torch.device`、`torch.dtype` 或 `torch.layout`
/// 创建张量，我们希望 TensorOptions 可以隐式地从 `ScalarType dtype`、`Layout layout` 和 `Device device`
/// 转换而来。因此，我们为这三种类型分别定义了隐式构造函数。
///
/// 对于 `ScalarType` 和 `Layout`，这足够了，因为它们是简单的枚举类。但是，`Device` 是一个普通类，
/// 具有两个隐式构造函数 `Device(DeviceType, DeviceIndex = -1)` 和 `Device(std::string)`，为了与
/// Python API 保持一致，其中字符串被视为等同于 `torch.device` 对象（例如，"cuda:1" 可以传递到任何需要
/// `torch.device("cuda:1")` 的地方）。为了支持 `at::empty({10}, {kCUDA, 1})` 和 `tensor.to(kCUDA)`
/// 这样的语法，我们需要确保 `TensorOptions` 可以隐式地从 `Device` 可以构造的任何参数转换而来。因此，
/// 我们有以下构造函数：
///
///    /* implicit */ TensorOptions(T&& device) : TensorOptions() {
///      this->set_device(device);
///    }
///
///    template <typename... Args,
///             typename = std::enable_if_t<std::is_constructible<Device,
///             Args&&...>::value>>
///    /* implicit */ TensorOptions(Args&&... args)
///     : TensorOptions(Device(std::forward<Args>(args)...)) {}
///
///
/// 但是这会导致问题。考虑这种情况：`TensorOptions({kCUDA, 1})`。编译器会因为在复制构造函数和
/// `Device` 构造函数之间的模糊性而抱怨，因为 `{kCUDA, 1}` 可以转换为 `TensorOption` 和 `Device`。
///
/// 为了解决这个问题，我们使 `Device` 构造函数具有模板化。由于重载分辨是在模板解析之前进行的，所以我们的
/// 问题得到了解决。
///

DispatchKey computeDispatchKey(
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device);

struct C10_API TensorOptions {
  TensorOptions()
      : requires_grad_(false),
        pinned_memory_(false),
        has_device_(false),
        has_dtype_(false),
        has_layout_(false),
        has_requires_grad_(false),
        has_pinned_memory_(false),
        has_memory_format_(false) {}

  /// Constructs a `TensorOptions` object with the given layout.
  /* implicit */ TensorOptions(Layout layout) : TensorOptions() {
  /// 设置对象的布局选项
  this->set_layout(layout);
}

/// 使用给定设备构造一个 `TensorOptions` 对象。
/// 参见 NOTE [ TensorOptions Constructors ]，解释为什么这里使用模板。
template <
    typename T,
    typename = std::enable_if_t<std::is_same_v<std::decay_t<T>, Device>>>
/* implicit */ TensorOptions(T&& device) : TensorOptions() {
  // 设置设备选项
  this->set_device(std::forward<T>(device));
}

/// 从允许在 `Device` 构造函数中使用的参数构造一个 `TensorOptions` 对象。
///
/// 参见 NOTE [ TensorOptions Constructors ]。
///
/// 注意：理想情况下，我们只允许隐式构造函数。但是很难检测到它们。因此我们有这个函数，允许显式构造函数。
template <
    typename... Args,
    typename = std::enable_if_t<std::is_constructible_v<Device, Args&&...>>>
/* implicit */ TensorOptions(Args&&... args)
    : TensorOptions(Device(std::forward<Args>(args)...)) {}

/// 使用给定的数据类型构造一个 `TensorOptions` 对象。
/* implicit */ TensorOptions(caffe2::TypeMeta dtype) : TensorOptions() {
  // 设置数据类型选项
  this->set_dtype(dtype);
}

/// 为了支持 ScalarType 而存在的传统构造函数
/* implicit */ TensorOptions(ScalarType dtype) : TensorOptions() {
  // 设置数据类型选项
  this->set_dtype(dtype);
}

/// 使用给定的内存格式构造一个 `TensorOptions` 对象。
/* implicit */ TensorOptions(MemoryFormat memory_format) : TensorOptions() {
  // 设置内存格式选项
  set_memory_format(memory_format);
}

/// 返回一个将 `device` 设置为给定值（如果 `device` 是 `nullopt` 则清除）的 `TensorOptions` 的副本。
C10_NODISCARD TensorOptions
device(std::optional<Device> device) const noexcept {
  // 复制当前选项
  TensorOptions r = *this;
  // 设置设备选项
  r.set_device(device);
  return r;
}

/// 返回一个将 `device` 设置为给定值的 `TensorOptions` 的副本。
/// （这个重载确保为 Device 的可变模板 std::optional 构造函数正常工作。）
template <typename... Args>
C10_NODISCARD TensorOptions device(Args&&... args) const noexcept {
  return device(
      std::optional<Device>(std::in_place, std::forward<Args>(args)...));
}

/// 返回一个将设备设置为 CUDA，并将设备索引设置为给定值的 `TensorOptions` 的副本。
///
/// TODO: 这个函数鼓励不良行为（假设 CUDA 是唯一重要的设备）。应删除或重命名。
C10_NODISCARD TensorOptions
device_index(c10::DeviceIndex device_index) const noexcept {
  return device(Device::Type::CUDA, device_index);
}

/// 返回一个将 `dtype` 设置为给定值的 `TensorOptions` 的副本。
C10_NODISCARD TensorOptions
dtype(std::optional<caffe2::TypeMeta> dtype) const noexcept {
  // 复制当前选项
  TensorOptions r = *this;
  // 设置数据类型选项
  r.set_dtype(dtype);
  return r;
}

// 为了支持 ScalarType 而存在的传统函数
C10_NODISCARD TensorOptions
dtype(std::optional<ScalarType> dtype) const noexcept {
  // 复制当前选项
  TensorOptions r = *this;
  // 设置数据类型选项
  r.set_dtype(dtype);
  // 返回当前 TensorOptions 对象
  return r;
}

// 由于已经确定了数据类型 dtype...
template <typename T>
TensorOptions& dtype() {
  // 设置当前 TensorOptions 的数据类型为 T 对应的 caffe2::TypeMeta
  dtype_ = caffe2::TypeMeta::Make<T>();
  // 标记已经设置了数据类型
  has_dtype_ = true;
  // 返回当前 TensorOptions 对象的引用
  return *this;
}

/// 设置 `TensorOptions` 的布局。
C10_NODISCARD TensorOptions
layout(std::optional<Layout> layout) const noexcept {
  // 创建一个新的 TensorOptions 对象作为返回值
  TensorOptions r = *this;
  // 设置新对象的布局
  r.set_layout(layout);
  // 返回设置了布局的新 TensorOptions 对象
  return r;
}

/// 设置 `TensorOptions` 的 `requires_grad` 属性。
C10_NODISCARD TensorOptions
requires_grad(std::optional<bool> requires_grad) const noexcept {
  // 创建一个新的 TensorOptions 对象作为返回值
  TensorOptions r = *this;
  // 设置新对象的 `requires_grad` 属性
  r.set_requires_grad(requires_grad);
  // 返回设置了 `requires_grad` 的新 TensorOptions 对象
  return r;
}

/// 设置 `TensorOptions` 的 `pinned_memory` 属性。
C10_NODISCARD TensorOptions
pinned_memory(std::optional<bool> pinned_memory) const noexcept {
  // 创建一个新的 TensorOptions 对象作为返回值
  TensorOptions r = *this;
  // 设置新对象的 `pinned_memory` 属性
  r.set_pinned_memory(pinned_memory);
  // 返回设置了 `pinned_memory` 的新 TensorOptions 对象
  return r;
}

/// 设置 `TensorOptions` 的 `memory_format` 属性。
C10_NODISCARD TensorOptions
memory_format(std::optional<MemoryFormat> memory_format) const noexcept {
  // 创建一个新的 TensorOptions 对象作为返回值
  TensorOptions r = *this;
  // 设置新对象的 `memory_format` 属性
  r.set_memory_format(memory_format);
  // 返回设置了 `memory_format` 的新 TensorOptions 对象
  return r;
}
  // 返回是否设置了 requires_grad 属性
  return has_requires_grad_;
}

/// 返回 `TensorOptions` 的 `requires_grad` 属性，如果未指定则返回 `c10::nullopt`。
std::optional<bool> requires_grad_opt() const noexcept {
  // 如果设置了 requires_grad 属性，则返回其值作为 optional
  return has_requires_grad_ ? c10::make_optional(requires_grad_)
                            : c10::nullopt;
}

/// 返回 `TensorOptions` 的 `pinned_memory` 属性。
bool pinned_memory() const noexcept {
  // 返回是否已设置 `pinned_memory` 的值
  return pinned_memory_or_default(pinned_memory_opt());
}

/// 返回是否设置了 `pinned_memory`。
bool has_pinned_memory() const noexcept {
  // 返回是否设置了 `pinned_memory_` 标志
  return has_pinned_memory_;
}

/// 返回布局是否为稀疏布局
bool is_sparse() const {
  // 检查布局是否为稀疏布局
  return layout_ == c10::Layout::Sparse;
}

/// 返回布局是否为稀疏 CSR（压缩稀疏行格式），已弃用，使用 `is_sparse_compressed()` 代替
bool is_sparse_csr() const {
  // 检查布局是否为稀疏 CSR 格式
  return layout_ == c10::Layout::SparseCsr;
}

/// 返回布局是否为压缩稀疏格式（包括 CSR、CSC、BSR、BSC）
bool is_sparse_compressed() const {
  // 检查布局是否为稀疏 CSR、CSC、BSR 或 BSC 格式之一
  return layout_ == c10::Layout::SparseCsr ||
         layout_ == c10::Layout::SparseCsc ||
         layout_ == c10::Layout::SparseBsr || layout_ == c10::Layout::SparseBsc;
}

// 用于与旧版 tensor.type() 比较的兼容性
bool type_equal(const TensorOptions& other) const {
  // 比较计算的 dispatch key 和数据类型是否相同
  return computeDispatchKey() == other.computeDispatchKey() &&
         typeMetaToScalarType(dtype_) == typeMetaToScalarType(other.dtype());
}

/// 返回 `TensorOptions` 的 `pinned_memory` 属性，如果未指定则返回 `c10::nullopt`。
std::optional<bool> pinned_memory_opt() const noexcept {
  // 如果设置了 `pinned_memory_`，则返回其值作为 optional
  return has_pinned_memory_ ? c10::make_optional(pinned_memory_)
                            : c10::nullopt;
}

/// 返回是否设置了 `memory_layout`。
bool has_memory_format() const noexcept {
  // 返回是否设置了 `memory_format_` 标志
  return has_memory_format_;
}

// 注意：memory_format() getter 故意未定义，因为其默认行为在不同函数中可能有所不同。

/// 返回 `TensorOptions` 的 `memory_layout` 属性，如果未指定则返回 `c10::nullopt`。
std::optional<MemoryFormat> memory_format_opt() const noexcept {
  // 如果设置了 `memory_format_`，则返回其值作为 optional
  return has_memory_format_ ? c10::make_optional(memory_format_)
                            : c10::nullopt;
}

// 解析当前构造轴指定的 ATen 后端。
// TODO: 废弃此函数
Backend backend() const {
  /// 返回基于计算的调度键的后端类型
  return at::dispatchKeyToBackend(computeDispatchKey());
}

/// 返回两个 TensorOptions 的右偏合并。这将用 options 的指定选项覆盖 self 中的设置。
///
/// 注意：此合并操作不考虑设备的合并。
/// 例如，如果 self 的设备是{kCUDA, 1}，然后调用 merge_in(kCUDA)，最终将得到 kCUDA！
/// Tensor.new_empty 等函数通过设备守卫确保了正确的设备选择。
///
TensorOptions merge_in(TensorOptions options) const noexcept {
  TensorOptions merged = *this;
  if (options.has_device())
    merged.set_device(options.device_opt());
  if (options.has_dtype())
    merged.set_dtype(options.dtype_opt());
  if (options.has_layout())
    merged.set_layout(options.layout_opt());
  // 注意：requires grad 是右偏的；不是逻辑上的 AND/OR！
  if (options.has_requires_grad())
    merged.set_requires_grad(options.requires_grad_opt());
  if (options.has_pinned_memory())
    merged.set_pinned_memory(options.pinned_memory_opt());
  if (options.has_memory_format())
    merged.set_memory_format(options.memory_format_opt());
  return merged;
}

// TODO 在 TensorOptions 理性化之后移除
TensorOptions merge_memory_format(
    std::optional<MemoryFormat> optional_memory_format) const noexcept {
  TensorOptions merged = *this;
  if (optional_memory_format.has_value()) {
    merged.set_memory_format(*optional_memory_format);
  }
  return merged;
}

// INVARIANT: computeDispatchKey 仅返回 dispatch keys 的子集，其中 dispatchKeyToBackend 是单射的，
// 如果定义的话（大部分情况下，这意味着此函数不会返回 Autograd key）
DispatchKey computeDispatchKey() const {
  return c10::computeDispatchKey(
      optTypeMetaToScalarType(dtype_opt()), layout_opt(), device_opt());
}

private:
// 这些方法当前是私有的，因为我不确定是否明智地公开它们。
// 这些方法是为了在构造函数和函数 API 实现中使用。
//
// 如果你确实非常需要它们，可以将它们设为公开，但请先检查是否可以通过函数 API 来实现你的需求。
// 同样地，这些方法不可链接，因为如果你需要链接，你可能更应该使用函数 API。（将它们设为链接可能是可以的，
// 因为这些函数都明确地带有 ref-qualifier，即结尾的 &，使得在临时对象上调用它们是非法的。）

/// 可变地设置 `TensorOptions` 的设备。
void set_device(std::optional<Device> device) & noexcept {
  if (device) {
    device_ = *device;
    has_device_ = true;
  } else {
    has_device_ = false;
  }
}

/// Mutably set the dtype of `TensorOptions`.
void set_dtype(std::optional<caffe2::TypeMeta> dtype) & noexcept {
  // 如果 dtype 不为空
  if (dtype) {
    // 将 dtype 的值赋给 dtype_
    dtype_ = *dtype;
    // 设置 has_dtype_ 为 true
    has_dtype_ = true;
  } else {
    // 如果 dtype 为空，则设置 has_dtype_ 为 false
    has_dtype_ = false;
  }
}

// legacy function to support ScalarType
void set_dtype(std::optional<ScalarType> dtype) & noexcept {
  // 如果 dtype 不为空
  if (dtype) {
    // 将 dtype 转换为对应的 TypeMeta 类型，然后赋给 dtype_
    dtype_ = scalarTypeToTypeMeta(*dtype);
    // 设置 has_dtype_ 为 true
    has_dtype_ = true;
  } else {
    // 如果 dtype 为空，则设置 has_dtype_ 为 false
    has_dtype_ = false;
  }
}

/// Mutably set the layout of `TensorOptions`.
void set_layout(std::optional<Layout> layout) & noexcept {
  // 如果 layout 不为空
  if (layout) {
    // 将 layout 的值赋给 layout_
    layout_ = *layout;
    // 设置 has_layout_ 为 true
    has_layout_ = true;
  } else {
    // 如果 layout 为空，则设置 has_layout_ 为 false
    has_layout_ = false;
  }
}

/// Mutably set the `requires_grad` property of `TensorOptions`.
void set_requires_grad(std::optional<bool> requires_grad) & noexcept {
  // 如果 requires_grad 不为空
  if (requires_grad) {
    // 将 requires_grad 的值赋给 requires_grad_
    requires_grad_ = *requires_grad;
    // 设置 has_requires_grad_ 为 true
    has_requires_grad_ = true;
  } else {
    // 如果 requires_grad 为空，则设置 has_requires_grad_ 为 false
    has_requires_grad_ = false;
  }
}

/// Mutably set the `pinned_memory` property of `TensorOptions`.
void set_pinned_memory(std::optional<bool> pinned_memory) & noexcept {
  // 如果 pinned_memory 不为空
  if (pinned_memory) {
    // 将 pinned_memory 的值赋给 pinned_memory_
    pinned_memory_ = *pinned_memory;
    // 设置 has_pinned_memory_ 为 true
    has_pinned_memory_ = true;
  } else {
    // 如果 pinned_memory 为空，则设置 has_pinned_memory_ 为 false
    has_pinned_memory_ = false;
  }
}

/// Mutably set the `memory_Format` property of `TensorOptions`.
void set_memory_format(std::optional<MemoryFormat> memory_format) & noexcept {
  // 如果 memory_format 不为空
  if (memory_format) {
    // 将 memory_format 的值赋给 memory_format_
    memory_format_ = *memory_format;
    // 设置 has_memory_format_ 为 true
    has_memory_format_ = true;
  } else {
    // 如果 memory_format 为空，则设置 has_memory_format_ 为 false
    has_memory_format_ = false;
  }
}

// WARNING: If you edit TensorOptions to add more options, you
// may need to adjust the implementation of Tensor::options.
// The criteria for whether or not Tensor::options must be adjusted
// is whether or not the new option you added should preserved
// by functions such as empty_like(); if it should be preserved,
// you must adjust options().
//
// TODO: MemoryFormat is not implemented in this way

// NB: We didn't use std::optional here, because then we can't pack
// the has_***_ boolean fields.

Device device_ = at::kCPU; // 16-bit
caffe2::TypeMeta dtype_ = caffe2::TypeMeta::Make<float>(); // 16-bit
Layout layout_ = at::kStrided; // 8-bit
MemoryFormat memory_format_ = MemoryFormat::Contiguous; // 8-bit

// Bitmask required here to get this to fit inside 32 bits (or even 64 bits,
// for that matter)

bool requires_grad_ : 1;
bool pinned_memory_ : 1;

bool has_device_ : 1;
bool has_dtype_ : 1;
bool has_layout_ : 1;
bool has_requires_grad_ : 1;
bool has_pinned_memory_ : 1;
bool has_memory_format_ : 1;
};

// 我们应该努力将其适应一个机器大小的字；但大于两个字是过多的。(在32位架构中，我们在存储张量选项时需要三个机器大小的字。)
static_assert(
    sizeof(TensorOptions) <= sizeof(int64_t) * 2,
    "TensorOptions must fit in 128-bits");

/// 返回一个设置了指定 `dtype` 的 `TensorOptions` 对象的便捷函数。
inline TensorOptions dtype(caffe2::TypeMeta dtype) {
  return TensorOptions().dtype(dtype);
}

// 为了支持 ScalarType 而提供的遗留函数
inline TensorOptions dtype(ScalarType dtype) {
  return TensorOptions().dtype(scalarTypeToTypeMeta(dtype));
}

/// 返回一个设置了指定 `layout` 的 `TensorOptions` 对象的便捷函数。
inline TensorOptions layout(Layout layout) {
  return TensorOptions().layout(layout);
}

/// 返回一个设置了指定 `device` 的 `TensorOptions` 对象的便捷函数。
inline TensorOptions device(Device device) {
  return TensorOptions().device(device);
}

/// 返回一个设置了 `device` 为 CUDA，并设置了指定 `device_index` 的 `TensorOptions` 对象的便捷函数。
inline TensorOptions device_index(c10::DeviceIndex device_index) {
  return TensorOptions().device_index(device_index);
}

/// 返回一个设置了 `requires_grad` 的 `TensorOptions` 对象的便捷函数。
inline TensorOptions requires_grad(bool requires_grad = true) {
  return TensorOptions().requires_grad(requires_grad);
}

/// 返回一个设置了 `memory_format` 的 `TensorOptions` 对象的便捷函数。
inline TensorOptions memory_format(MemoryFormat memory_format) {
  return TensorOptions().memory_format(memory_format);
}

C10_API std::ostream& operator<<(
    std::ostream& stream,
    const TensorOptions& options);

template <typename T>
inline TensorOptions dtype() {
  return dtype(caffe2::TypeMeta::Make<T>());
}

/// 将 `TensorOptions` 对象转换为字符串的便捷函数。
inline std::string toString(const TensorOptions& options) {
  std::ostringstream stream;
  stream << options;
  return stream.str();
}

// 这是一个集中确定张量适当的 DispatchKey 的位置。
inline DispatchKey computeDispatchKey(
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device) {
  const auto layout_ = layout_or_default(layout);
  const auto device_ = device_or_default(device);
  switch (layout_) {
    case Layout::Jagged:
    case Layout::Strided: {
      const auto dtype_ = dtype_or_default(dtype);
      switch (device_.type()) {
#define DO_CASE(device, _)                   \
  case c10::DeviceType::device: {            \
    if (isQIntType(dtype_)) {                \
      return DispatchKey::Quantized##device; \
    }                                        \
    return DispatchKey::device;              \
  }
        C10_FORALL_BACKEND_DEVICE_TYPES(DO_CASE, unused)


注释：


    // 返回设备的分发键值，表明函数执行的设备类型
    return DispatchKey::device;              \
  }
        // 对于所有后端设备类型，执行宏定义的操作
        C10_FORALL_BACKEND_DEVICE_TYPES(DO_CASE, unused)
inline Layout dispatchKeyToLayout(DispatchKey dispatch_key) {
  // 根据 DispatchKey 转换为对应的布局类型

  switch (dispatch_key) {
#define DO_CASE(bc, _) case DispatchKey::Sparse##bc:
    // 展开所有稀疏组件的情况，根据 DispatchKey 返回 Sparse 布局
    C10_FORALL_BACKEND_COMPONENTS(DO_CASE, unused)
#undef DO_CASE
    // 如果无法映射 DispatchKey 到唯一的布局类型，则报错
    TORCH_CHECK(
        false, "Cannot map DispatchKey ", dispatch_key, " to a unique layout.");
    // 返回 MkldnnCPU 对应的 Mkldnn 布局
    case DispatchKey::MkldnnCPU:
      return Layout::Mkldnn;
    // 默认情况下返回 Strided 布局
    default:
      return Layout::Strided;
  }
}

inline c10::DeviceType dispatchKeyToDeviceType(DispatchKey dispatch_key) {
  // 根据 DispatchKey 转换为对应的设备类型

  switch (dispatch_key) {
    // 创建一个空的列表，用于存储实数
    reals = []
    
    // 创建一个空的集合，用于存储非实数
    non_reals = set()
    
    // 遍历给定列表中的每个元素
    for num in nums:
        // 如果元素是实数
        if isinstance(num, (int, float)):
            // 将实数添加到 reals 列表中
            reals.append(num)
        // 如果元素不是实数
        else:
            // 将非实数添加到 non_reals 集合中
            non_reals.add(num)
    
    // 返回两个结果：实数列表 reals 和非实数集合 non_reals
    return reals, non_reals
#define DO_CASE(suffix, prefix)     \  // 定义宏，根据后缀和前缀生成对应的 case 语句
  case DispatchKey::prefix##suffix: \  // 当前 case 语句处理 DispatchKey 中特定的键组合
    return c10::DeviceType::suffix;   // 返回对应的 c10 设备类型

#define DO_CASES(_, prefix) C10_FORALL_BACKEND_DEVICE_TYPES(DO_CASE, prefix)  // 展开宏，对所有后端设备类型生成 case 语句
    C10_FORALL_FUNCTIONALITY_KEYS(DO_CASES)  // 展开宏，对所有功能键生成对应的 case 语句
#undef DO_CASES   // 取消定义展开后的 DO_CASES 宏
#undef DO_CASE    // 取消定义展开后的 DO_CASE 宏

    case DispatchKey::MkldnnCPU:    // 处理特定的 DispatchKey：MkldnnCPU
      return c10::DeviceType::CPU;  // 返回对应的 CPU 设备类型
    case DispatchKey::Vulkan:       // 处理特定的 DispatchKey：Vulkan
      return c10::DeviceType::Vulkan;  // 返回对应的 Vulkan 设备类型

    case DispatchKey::MAIA:         // 处理特定的 DispatchKey：MAIA
      return c10::DeviceType::MAIA;  // 返回对应的 MAIA 设备类型
    default:                        // 如果没有匹配到任何特定的 DispatchKey
      TORCH_CHECK(                  // 使用 TORCH_CHECK 进行断言检查
          false,                    // 断言条件为假
          "DispatchKey ",           // 断言失败时输出的消息前缀
          dispatch_key,             // 输出断言失败时的具体 DispatchKey 值
          " doesn't correspond to a device");  // 输出断言失败时的具体消息
  }
}

inline TensorOptions dispatchKeyToTensorOptions(DispatchKey dispatch_key) {
  return TensorOptions()             // 创建一个 TensorOptions 对象
      .layout(dispatchKeyToLayout(dispatch_key))  // 设置 TensorOptions 的布局属性
      .device(dispatchKeyToDeviceType(dispatch_key));  // 设置 TensorOptions 的设备类型属性
}

namespace detail {
inline bool backend_supports_empty_operator(const TensorOptions& options) {
  // Quantized backends don't support at::empty().
  // They have separate operators like at::empty_quantized() that take in
  // extra information about how to quantize the tensor.
  return !isQIntType(typeMetaToScalarType(options.dtype()));  // 检查是否是量化整型数据类型，如果是则返回 false，否则返回 true
}

} // namespace detail

} // namespace c10
```