# `.\pytorch\aten\src\ATen\TensorIndexing.h`

```py
#pragma once

#include <ATen/ExpandUtils.h> // 引入 ATen 库中的扩展工具
#include <ATen/ScalarOps.h> // 引入 ATen 库中的标量操作
#include <ATen/core/Tensor.h> // 引入 ATen 库中的 Tensor 类
#include <ATen/core/TensorBody.h> // 引入 ATen 库中的 TensorBody 类
#include <c10/core/SymInt.h> // 引入 c10 库中的 SymInt 类
#include <c10/util/Optional.h> // 引入 c10 库中的 Optional 类
#include <c10/util/irange.h> // 引入 c10 库中的 irange 函数

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h> // 如果未定义 AT_PER_OPERATOR_HEADERS，则引入 ATen 库中的通用函数
#include <ATen/NativeFunctions.h> // 如果未定义 AT_PER_OPERATOR_HEADERS，则引入 ATen 库中的本地函数
#else
#include <ATen/ops/alias.h> // 如果定义了 AT_PER_OPERATOR_HEADERS，则引入 ATen 库中的别名操作
#include <ATen/ops/empty.h> // 如果定义了 AT_PER_OPERATOR_HEADERS，则引入 ATen 库中的创建空张量操作
#include <ATen/ops/scalar_tensor.h> // 如果定义了 AT_PER_OPERATOR_HEADERS，则引入 ATen 库中的标量张量操作
#include <ATen/ops/zeros.h> // 如果定义了 AT_PER_OPERATOR_HEADERS，则引入 ATen 库中的创建全零张量操作
#endif

#include <ATen/core/List.h> // 引入 ATen 库中的 List 类

#include <utility> // 引入 C++ 标准库中的 utility 头文件，用于 std::move 和其他工具

namespace at::indexing {

constexpr int64_t INDEX_MIN = c10::SymInt::min_representable_int(); // 定义最小的索引值为 SymInt 类型的最小表示整数
constexpr int64_t INDEX_MAX = -(INDEX_MIN + 1); // 定义最大的索引值为 SymInt 类型的最大表示整数减一

enum class TensorIndexType { None, Ellipsis, SymInt, Boolean, Slice, Tensor }; // 定义 TensorIndexType 枚举，表示不同的张量索引类型

constexpr c10::nullopt_t None = c10::nullopt; // 定义 None 常量为 c10::nullopt

struct TORCH_API EllipsisIndexType final {
  EllipsisIndexType() = default; // 默认构造函数
};
TORCH_API extern const EllipsisIndexType Ellipsis; // 外部声明 Ellipsis 常量

struct TORCH_API Slice final {
 public:
  Slice(
      std::optional<c10::SymInt> start_index = c10::nullopt, // 构造函数，可选参数为起始索引
      std::optional<c10::SymInt> stop_index = c10::nullopt, // 可选参数为结束索引
      std::optional<c10::SymInt> step_index = c10::nullopt) { // 可选参数为步长索引
    if (!step_index.has_value()) { // 如果步长索引未提供
      step_ = c10::SymInt(1); // 设置步长为默认值 1
    } else {
      step_ = std::move(step_index).value(); // 否则使用提供的步长值
    }

    TORCH_CHECK_VALUE(step_ != 0, "slice step cannot be zero"); // 检查步长不能为零

    if (!start_index.has_value()) { // 如果起始索引未提供
      start_ = c10::SymInt(step_ < 0 ? INDEX_MAX : 0); // 根据步长确定默认的起始索引值
    } else {
      start_ = std::move(start_index).value(); // 否则使用提供的起始索引值
    }

    if (!stop_index.has_value()) { // 如果结束索引未提供
      stop_ = c10::SymInt(step_ < 0 ? INDEX_MIN : INDEX_MAX); // 根据步长确定默认的结束索引值
    } else {
      stop_ = std::move(stop_index).value(); // 否则使用提供的结束索引值
    }
  }

  inline c10::SymInt start() const { // 返回起始索引值的访问函数
    return start_;
  }

  inline c10::SymInt stop() const { // 返回结束索引值的访问函数
    return stop_;
  }

  inline c10::SymInt step() const { // 返回步长值的访问函数
    return step_;
  }

 private:
  c10::SymInt start_; // 起始索引值
  c10::SymInt stop_; // 结束索引值
  c10::SymInt step_; // 步长值
};

TORCH_API std::ostream& operator<<(std::ostream& stream, const Slice& slice); // 重载流操作符，用于输出 Slice 对象

// `at::indexing::TensorIndex` is used for converting C++ tensor indices such as
// `{None, "...", Ellipsis, 0, true, Slice(1, None, 2), torch::tensor({1, 2})}`
// into its equivalent `std::vector<TensorIndex>`, so that further tensor
// indexing operations can be performed using the supplied indices.
//
// There is one-to-one correspondence between Python and C++ tensor index types:
// Python                  | C++
// -----------------------------------------------------
// `None`                  | `at::indexing::None`
// `Ellipsis`              | `at::indexing::Ellipsis`
// `...`                   | `"..."`
// `123`                   | `123`
// `True` / `False`        | `true` / `false`
// `:`                     | `Slice()` / `Slice(None, None)`
// `::`                    | `Slice()` / `Slice(None, None, None)`
// `1:`                    | `Slice(1, None)`
// `1::`                   | `Slice(1, None, None)`
// `:3`                    | `Slice(None, 3)`
/// `:3:` 表示一个切片，从起始位置开始取到索引 3 之前的元素
/// `::2` 表示一个切片，从起始位置开始，每隔一个元素取值
/// `1:3` 表示一个切片，从索引 1 开始取到索引 3 之前的元素
/// `1::2` 表示一个切片，从索引 1 开始，每隔一个元素取值
/// `:3:2` 表示一个切片，从起始位置开始取到索引 3 之前的元素，每隔一个元素取值
/// `1:3:2` 表示一个切片，从索引 1 开始取到索引 3 之前的元素，每隔一个元素取值
/// `torch.tensor([1, 2])` 表示创建一个包含元素 [1, 2] 的张量
struct TORCH_API TensorIndex final {
  // Case 1: `at::indexing::None`
  /// 构造函数，用于表示 None 类型的索引
  TensorIndex(c10::nullopt_t) : type_(TensorIndexType::None) {}

  // Case 2: "..." / `at::indexing::Ellipsis`
  /// 构造函数，用于表示省略号（Ellipsis）类型的索引
  TensorIndex(at::indexing::EllipsisIndexType)
      : type_(TensorIndexType::Ellipsis) {}
  /// 构造函数，接受字符串 "..."，用于表示省略号（Ellipsis）类型的索引
  TensorIndex(const char* str) : TensorIndex(at::indexing::Ellipsis) {
    TORCH_CHECK_VALUE(
        strcmp(str, "...") == 0,
        "Expected \"...\" to represent an ellipsis index, but got \"",
        str,
        "\"");
  }

  // Case 3: (Sym) Integer value
  /// 构造函数，用于表示符号整数类型的索引
  TensorIndex(SymInt integer)
      : integer_(std::move(integer)), type_(TensorIndexType::SymInt) {}
  /// 构造函数，用于表示整数类型的索引
  TensorIndex(int64_t integer) : TensorIndex(SymInt(integer)) {}
  /// 构造函数，用于表示整数类型的索引
  TensorIndex(int integer) : TensorIndex(SymInt(integer)) {}

  // Case 4: Boolean value
  /// 模板构造函数，用于表示布尔类型的索引
  template <class T, class = std::enable_if_t<std::is_same_v<bool, T>>>
  TensorIndex(T boolean) : boolean_(boolean), type_(TensorIndexType::Boolean) {}

  // Case 5: Slice represented in `at::indexing::Slice` form
  /// 构造函数，用于表示切片类型的索引
  TensorIndex(Slice slice)
      : slice_(std::move(slice)), type_(TensorIndexType::Slice) {}

  // Case 6: Tensor value
  /// 构造函数，用于表示张量类型的索引
  TensorIndex(Tensor tensor)
      : tensor_(std::move(tensor)), type_(TensorIndexType::Tensor) {}

  /// 返回是否为 None 类型的索引
  inline bool is_none() const {
    return type_ == TensorIndexType::None;
  }

  /// 返回是否为 Ellipsis 类型的索引
  inline bool is_ellipsis() const {
    return type_ == TensorIndexType::Ellipsis;
  }

  /// 返回是否为整数类型的索引
  inline bool is_integer() const {
    return type_ == TensorIndexType::SymInt;
  }

  /// 返回整数值
  inline SymInt integer() const {
    return integer_;
  }

  /// 返回是否为布尔类型的索引
  inline bool is_boolean() const {
    return type_ == TensorIndexType::Boolean;
  }

  /// 返回布尔值
  inline bool boolean() const {
    return boolean_;
  }

  /// 返回是否为切片类型的索引
  inline bool is_slice() const {
    return type_ == TensorIndexType::Slice;
  }

  /// 返回切片对象
  inline const Slice& slice() const {
    return slice_;
  }

  /// 返回是否为张量类型的索引
  inline bool is_tensor() const {
    return type_ == TensorIndexType::Tensor;
  }

  /// 返回张量对象
  inline const Tensor& tensor() const {
    return tensor_;
  }

 private:
  SymInt integer_ = 0; /// 整数值
  bool boolean_ = false; /// 布尔值
  Slice slice_; /// 切片对象
  Tensor tensor_; /// 张量对象
  TensorIndexType type_; /// 索引类型
};

/// 输出流操作符重载，用于打印 TensorIndex 对象
TORCH_API std::ostream& operator<<(
    std::ostream& stream,
    const TensorIndex& tensor_index);

/// 输出流操作符重载，用于打印 TensorIndex 向量
TORCH_API std::ostream& operator<<(
    std::ostream& stream,
    const std::vector<TensorIndex>& tensor_indices);

namespace impl {
/// 应用切片操作到张量的具体实现
inline Tensor applySlice(
    const Tensor& self,
    int64_t dim,
    c10::SymInt start,
    c10::SymInt stop,
    c10::SymInt step,
    bool disable_slice_optimization,
    const at::Device& self_device,
    // TODO: implement negative step
    // 检查步长是否为正数，若不是，则抛出异常信息
    TORCH_CHECK_VALUE(step > 0, "step must be greater than zero");
    
    // See NOTE [nested tensor size for indexing]
    // 如果提供了self_sizes，则执行以下优化；注意，如果在追踪过程中，这个优化将被跳过，
    // 因为追踪可能依赖于`self`张量的形状，并且我们仍然希望记录切片操作。
    if (self_sizes.has_value()) {
        // 确定在当前设备上self的长度
        SymInt length = (self_device == at::kCPU || self_device == at::kCUDA)
            ? (*self_sizes)[dim]
            : self.sym_size(dim);
    
        // 如果禁用了切片优化或者以下条件都满足：
        //   1. start的值在大小不可知的情况下等于0
        //   2. length的值在大小不可知的情况下等于stop
        //   3. 步长等于1
        // 则直接返回self，表示可以进行切片优化
        if (!disable_slice_optimization &&
            TORCH_GUARD_SIZE_OBLIVIOUS(start.sym_eq(0)) &&
            TORCH_GUARD_SIZE_OBLIVIOUS(length.sym_eq(stop)) && step == 1) {
          return self;
        }
    }
    
    // 否则，调用slice_symint方法对self进行切片操作，并返回切片后的结果
    return self.slice_symint(
        dim, std::move(start), std::move(stop), std::move(step));
}

// 在给定维度上应用索引操作，返回选择的张量
inline Tensor applySelect(
    const Tensor& self,                    // 输入张量
    int64_t dim,                           // 索引的维度
    SymInt index,                          // 符号化整数索引
    int64_t real_dim,                      // 实际维度
    const at::Device& /*self_device*/,     // 张量所在设备（未使用）
    const std::optional<SymIntArrayRef>& self_sizes) {  // 可选的张量大小信息

  // 查看嵌套张量索引大小的注意事项
  if (self_sizes.has_value()) {
    auto maybe_index = index.maybe_as_int();
    if (maybe_index.has_value()) {
      // 检查索引是否有效，特别处理0维张量的情况
      TORCH_CHECK_INDEX(
          !(maybe_index.value() == 0 && dim == 0 && self_sizes->empty()),
          "invalid index of a 0-dim tensor. ",
          "Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number");
    }

    auto size = (*self_sizes)[dim];
    // 注意：`size >= -index` 与 `size > -1 - index` 不等效，当索引为 INT64_MIN 时
    // 因为 INT64_MIN 的负数没有定义，但在实际中等同于 self 的长度
    // 对于所有负数 int64_t 值，索引包装在有效范围内，如 x[INT64_MIN] 等同于 x[INT64_MAX]
    TORCH_CHECK_INDEX(
        size > -1 - index && size > index,
        "index ",
        index,
        " is out of bounds for dimension ",
        real_dim,
        " with size ",
        size);
  }

  // 如果索引是负数，则不标准化它，因为这会将索引固定在当前张量大小上
  // aten::select 也可以使用负索引
  return self.select_symint(dim, std::move(index));
}

// 将布尔值转换为用于索引的张量（适用于 CPU 或 CUDA 设备）
inline Tensor boolToIndexingTensorCPUOrCUDA(const Tensor& self, bool value) {
  // 布尔值添加大小为 1 的维度。true 索引此维度如同 0:，false 索引为空。
  if (value) {
    return at::empty({1}, self.options().dtype(kLong)).fill_(0.);
  } else {
    return at::empty({0}, self.options().dtype(kLong));
  }
}

// 将布尔值转换为用于索引的张量（非本地设备类型）
inline Tensor boolToIndexingTensorNonNativeDeviceType(
    const Tensor& self,
    bool value) {
  // 布尔值添加大小为 1 的维度。true 索引此维度如同 0:，false 索引为空。
  if (value) {
    return at::zeros({1}, self.options().dtype(kLong));
  } else {
    return at::empty({0}, self.options().dtype(kLong));
  }
}

// 将布尔值转换为用于索引的张量，根据设备类型选择合适的转换方法
inline Tensor boolToIndexingTensor(
    const Tensor& self,
    bool value,
    const at::Device& self_device) {
  if (self_device == at::kCPU || self_device == at::kCUDA) {
    return boolToIndexingTensorCPUOrCUDA(self, value);
  } else {
    return boolToIndexingTensorNonNativeDeviceType(self, value);
  }
}

// 将标量转换为张量（非本地设备类型）
inline Tensor scalarToTensorNonNativeDeviceType(
    const Scalar& v,
    const TensorOptions& options) {
  return at::scalar_tensor(v, options);
}

// 记录张量索引
inline void recordTensorIndex(
    const Tensor& tensor,
    std::vector<Tensor>& outIndices,
    int64_t* dim_ptr) {
  // TODO: 检查标量类型
  outIndices.resize(*dim_ptr + 1);
  outIndices[*dim_ptr] = tensor;
  (*dim_ptr)++;
};

// 类型转换索引列表的函数（未使用输入张量）
inline c10::List<::std::optional<Tensor>> typeConvertIndices(
    const Tensor& /*self*/,
    // 定义一个函数，接受一个右值引用的 vector<Tensor> 参数，并返回一个 c10::List 包含 std::optional<Tensor> 的对象
    std::vector<Tensor>&& indices) {
      // 创建一个 c10::List，其中元素类型为 std::optional<Tensor>，用来存放转换后的索引
      c10::List<::std::optional<Tensor>> converted_inds;
      // 预留空间以容纳与 indices 相同数量的元素
      converted_inds.reserve(indices.size());
      // 遍历传入的 indices（右值引用），将每个元素 std::move 到 converted_inds 中
      for (auto&& i : std::move(indices)) {
        converted_inds.push_back(std::move(i));
      }
      // 返回转换后的索引列表
      return converted_inds;
    }
// NOTE: Why do we mirror instead of replace the `count_specified_dimensions`
// function in torch/csrc/autograd/python_variable_indexing.cpp? It's because
// `count_specified_dimensions` is on the hot path of Python tensor multi-dim
// indexing (i.e. it's called by `applySlicing` which is called by
// `THPVariable_getitem` / `THPVariable_setitem` when handling indexing of more
// than one dimension). If we were to merge the Python/C++
// `count_specified_dimensions` function, on the Python side we would have to
// construct a `std::vector` container to be consumed by the C++
// `count_specified_dimensions` function, which adds 100s of nanoseconds
// overhead and is undesirable.
inline int64_t count_specified_dimensions(
    const ArrayRef<TensorIndex>& indices) {
  // Count the number of indexed dimensions (everything but ellipsis and None)
  int64_t count = 0;
  for (auto& obj : indices) {
    if (obj.is_tensor()) {
      auto& tensor = obj.tensor();
      // If the tensor type is byte (kByte) or boolean (kBool), count all dimensions
      if (tensor.scalar_type() == kByte || tensor.scalar_type() == kBool) {
        count += tensor.dim();
      } else {
        count++;  // Otherwise, count as one dimension
      }
    } else if (!obj.is_none() && !obj.is_ellipsis() && !obj.is_boolean()) {
      count++;  // Count other valid indices (excluding None, ellipsis, and boolean)
    }
  }
  return count;  // Return the total count of specified dimensions
}
} // namespace impl

// NOTE: Many functions below are only for consumption from Python indexing
// implementation, they include:
//
// - `Tensor scalarToTensor(...)`
// - `IntArrayRef slicePrefix1sSize(...)`
// - `void copy_to(...)`
// - `Tensor handleDimInMultiDimIndexing(...)`
// - `Tensor dispatch_index(...)`
// - `Tensor dispatch_index_put_(...)`
// - `Tensor get_item(...)`
// - `void set_item(...)`
//
// The rest of the functions are in `at::indexing::impl` namespace, signifying
// that they shouldn't be used from Python indexing implementation.
inline Tensor scalarToTensor(
    const Scalar& v,
    const TensorOptions& options,
    const at::Device& self_device) {
  if (self_device == at::kCPU && !v.isSymbolic()) {
    // Return a tensor on CPU if device is CPU and scalar is not symbolic
    return at::detail::scalar_tensor_static(
        v, options.dtype_opt()->toScalarType(), self_device);
  } else {
    // Otherwise, use the non-native device type scalarToTensor implementation
    return impl::scalarToTensorNonNativeDeviceType(v, options);
  }
}

// To match numpy semantics:
// As a special case for backwards compatibility,
// strip away unit dimensions from the left of 'src'
inline SymIntArrayRef slicePrefix1sSize(const SymIntArrayRef& sizes) {
  size_t first_non1_src = sizes.size();
  for (const auto i : c10::irange(sizes.size())) {
    // Check for non-unit dimensions to determine the starting index
    if (!sizes[i].has_hint() || sizes[i] != 1) {
      first_non1_src = i;
      break;
    }
  }

  // Return a view into 'sizes' starting from the first non-unit dimension
  return sizes.slice(first_non1_src);
}

// Copy 'src' tensor data into 'dst' tensor, assuming sizes match
inline void copy_to(const Tensor& dst, const Tensor& src) {
  if (dst.sym_sizes().equals(src.sym_sizes())) {
    // Perform a direct data copy when symbolic sizes of 'dst' and 'src' tensors match
    // This shortcut avoids generating hard-coded constant sizes during tracing.

    // Perform the actual data copy
    dst.copy_(src);
  }
}
    // 如果 `src` 和 `dst` 的形状不同，这不是一个完美的解决方案：
    // 常量可能仍然会出现。用户可以通过 `dst[index..] = src.reshape(..)` 来解决这种情况。
    dst.copy_(src);
    return;
  } else if (src.dim() == 0 && src.device().type() == at::kCPU) {
    // 如果 `src` 是零维并且位于 CPU 设备上，用 `src` 来填充 `dst`。
    dst.fill_(src);
    return;
  }
  // 获取 `src` 的符号大小的前缀切片视图。
  auto src_view = src.view_symint(slicePrefix1sSize(src.sym_sizes()));
  // 尝试在原地扩展 `dst`，使其能容纳 `src_view` 的形状，用于 "setitem" 操作。
  c10::MaybeOwned<Tensor> b_src = expand_inplace(dst, src_view, "setitem");
  // 将扩展后的 `b_src` 复制到 `dst`。
  dst.copy_(*b_src);
}

// 处理多维索引中的维度操作，根据不同类型的索引进行处理并返回相应的 Tensor 结果
inline Tensor handleDimInMultiDimIndexing(
    const Tensor& prev_dim_result,                         // 前一维度操作结果的 Tensor
    const Tensor& original_tensor,                         // 原始的 Tensor
    const TensorIndex& index,                              // 当前的索引对象
    int64_t* dim_ptr,                                      // 当前处理的维度指针
    int64_t* specified_dims_ptr,                           // 指定维度的指针
    int64_t real_dim,                                      // 实际维度数
    std::vector<Tensor>& outIndices,                       // 输出索引的向量
    bool disable_slice_optimization,                       // 是否禁用切片优化标志
    const at::Device& original_tensor_device,               // 原始 Tensor 的设备信息
    const std::optional<SymIntArrayRef>& prev_dim_result_sizes) {  // 可选的前一维度结果大小信息
  if (index.is_integer()) {
    return impl::applySelect(
        prev_dim_result,                                  // 应用选择操作到前一维度结果
        *dim_ptr,                                         // 当前维度指针
        index.integer(),                                  // 整数索引值
        real_dim,                                         // 实际维度数
        original_tensor_device,                           // 原始 Tensor 的设备信息
        prev_dim_result_sizes);                           // 可选的前一维度结果大小信息
  } else if (index.is_slice()) {
    Tensor result = impl::applySlice(
        prev_dim_result,                                  // 应用切片操作到前一维度结果
        *dim_ptr,                                         // 当前维度指针
        index.slice().start(),                           // 切片的起始位置
        index.slice().stop(),                            // 切片的结束位置
        index.slice().step(),                            // 切片的步长
        /*disable_slice_optimization=*/disable_slice_optimization,  // 是否禁用切片优化
        original_tensor_device,                           // 原始 Tensor 的设备信息
        prev_dim_result_sizes);                           // 可选的前一维度结果大小信息
    (*dim_ptr)++;                                         // 增加维度指针
    return result;                                        // 返回切片结果
  } else if (index.is_ellipsis()) {
    (*dim_ptr) += original_tensor.dim() - (*specified_dims_ptr);  // 计算省略号对应的维度增量
    return prev_dim_result;                                // 返回前一维度结果
  } else if (index.is_none()) {
    Tensor result = prev_dim_result.unsqueeze(*dim_ptr);  // 在指定维度上进行unsqueeze操作
    (*dim_ptr)++;                                         // 增加维度指针
    return result;                                        // 返回unsqueeze后的结果
  } else if (index.is_boolean()) {
    Tensor result = prev_dim_result.unsqueeze(*dim_ptr);  // 在指定维度上进行unsqueeze操作
    impl::recordTensorIndex(
        impl::boolToIndexingTensor(
            result, index.boolean(), original_tensor_device),  // 将布尔索引转换为索引 Tensor
        outIndices,                                        // 记录索引到输出向量
        dim_ptr);
    return result;                                        // 返回unsqueeze后的结果
  } else if (index.is_tensor()) {
    Tensor result = prev_dim_result;                      // 默认使用前一维度结果
    const Tensor& tensor = index.tensor();                // 获取索引 Tensor
    auto scalar_type = tensor.scalar_type();              // 获取 Tensor 的标量类型
    if (tensor.dim() == 0 &&
        at::isIntegralType(scalar_type, /*includeBool=*/true)) {
      if (scalar_type != at::kByte && scalar_type != at::kBool) {
        result = impl::applySelect(
            result,                                       // 应用选择操作到前一维度结果
            *dim_ptr,                                      // 当前维度指针
            tensor.item<int64_t>(),                        // 获取整数值索引
            real_dim,                                      // 实际维度数
            original_tensor_device,                        // 原始 Tensor 的设备信息
            prev_dim_result_sizes);                        // 可选的前一维度结果大小信息
      } else {
        result = result.unsqueeze(*dim_ptr);              // 在指定维度上进行unsqueeze操作
        if (scalar_type == at::kBool) {
          impl::recordTensorIndex(
              impl::boolToIndexingTensor(
                  result, tensor.item<bool>() != 0, original_tensor_device),  // 将布尔值转换为索引 Tensor
              outIndices,                                  // 记录索引到输出向量
              dim_ptr);
        } else {
          impl::recordTensorIndex(
              impl::boolToIndexingTensor(
                  result, tensor.item<uint8_t>() != 0, original_tensor_device),  // 将字节值转换为索引 Tensor
              outIndices,                                  // 记录索引到输出向量
              dim_ptr);
        }
      }
    } else {
      impl::recordTensorIndex(tensor, outIndices, dim_ptr);  // 记录索引 Tensor
    }
    return result;                                        // 返回处理后的结果
  } else {
    TORCH_INTERNAL_ASSERT(false, "Invalid TensorIndex type");  // 如果索引类型无效则抛出断言错误
  }
}

namespace impl {
// This mirrors `applySlicing` in
// torch/csrc/autograd/python_variable_indexing.cpp
inline Tensor applySlicing(
    const Tensor& self,                                // 输入的张量引用
    const ArrayRef<TensorIndex>& indices,              // 索引数组引用
    std::vector<Tensor>& outIndices,                   // 输出索引张量的向量
    bool disable_slice_optimization,                   // 禁用切片优化的标志
    const at::Device& self_device,                     // 输入张量的设备
    const std::optional<SymIntArrayRef>& self_sizes) { // 可选的输入张量尺寸引用
  int64_t dim = 0;                                     // 初始化维度为0
  int64_t specified_dims = impl::count_specified_dimensions(indices); // 计算指定的维度数量

  // See NOTE [nested tensor size for indexing]
  if (self_sizes.has_value()) {                        // 如果有输入张量尺寸的值
    TORCH_CHECK_INDEX(
        specified_dims <= (int64_t)self_sizes->size(), // 检查指定的维度数量不超过张量的维度数
        "too many indices for tensor of dimension ",
        (int)self_sizes->size());
  }

  Tensor result = self;                                // 结果张量初始化为输入张量
  for (const auto i : c10::irange(indices.size())) {   // 遍历索引数组中的每个索引
    auto& obj = indices[i];                            // 获取当前索引对象的引用
    // See NOTE [nested tensor size for indexing]
    std::optional<SymIntArrayRef> result_sizes = result.is_nested()
        ? std::optional<SymIntArrayRef>(c10::nullopt)  // 如果结果是嵌套张量，则尺寸为空
        : std::optional<SymIntArrayRef>(result.sym_sizes()); // 否则获取结果的符号尺寸
    result = handleDimInMultiDimIndexing(
        /*prev_dim_result=*/result,                    // 前一维度的结果张量
        /*original_tensor=*/self,                      // 原始输入张量
        /*index=*/obj,                                 // 当前索引对象
        /*dim_ptr=*/&dim,                              // 维度指针
        /*specified_dims_ptr=*/&specified_dims,        // 指定维度数量的指针
        /*real_dim=*/static_cast<int64_t>(i),          // 实际维度作为整数
        /*outIndices=*/outIndices,                     // 输出索引张量的向量
        /*disable_slice_optimization=*/disable_slice_optimization, // 禁用切片优化标志
        /*original_tensor_device=*/self_device,        // 原始输入张量的设备
        /*prev_dim_result_sizes=*/result_sizes);       // 前一维度结果的尺寸
  }
  return result;                                       // 返回处理后的结果张量
}
} // namespace impl

inline Tensor dispatch_index(
    const Tensor& self,                               // 输入张量引用
    std::vector<Tensor>&& indices) {                  // 移动语义的索引张量向量
  return self.index(impl::typeConvertIndices(self, std::move(indices))); // 调用索引转换函数并索引
}

inline Tensor dispatch_index_put_(
    Tensor& self,                                     // 输入输出张量引用
    std::vector<Tensor>&& indices,                    // 移动语义的索引张量向量
    const Tensor& value) {                            // 值张量引用
  return self.index_put_(                             // 执行索引放置操作
      impl::typeConvertIndices(self, std::move(indices)), // 索引转换函数调用
      value);                                         // 值张量
}

// NOTE [ Setting `disable_slice_optimization` when calling C++ tensor indexing
// functions from Python ]
//
// Question: When should we set `disable_slice_optimization` to `true` when
// calling C++ tensor indexing functions from Python indexing code?
//
// Answer: What "slice optimization" means: when we have a slicing expression
// like `x[0:5, 0]`, where the sliced tensor was of size 5 in dimension 0, we
// would skip dispatching the actual slice call as an optimization. However,
// here are the cases where we DON'T want this optimization:
//
// 1. When we are doing 1-D slicing (e.g. `tensor[:]`).
//    Reason: we always return a shallow copy for expressions such as
//    `tensor[:]` / `tensor[...]` / `tensor[:, :]`. (Note that for `tensor[:,
//    :]`, we return an alias of `tensor` by doing the following:
//    ```
//    Tensor sliced = impl::applySlicing(self, indices, tensorIndices,
//    disable_slice_optimization, self_device, self_sizes); if
//    (tensorIndices.empty()) {
//      if (sliced.is_same(self)) {
//        // ensure we return a shallow copy for things like x[...]
//        sliced = at::alias(sliced);
//      }
//      return sliced;
//    }
// 定义函数 `get_item`，用于从张量中获取指定索引处的子张量
// `disable_slice_optimization` 参数用于控制是否禁用切片优化
inline Tensor get_item(
    const Tensor& self,                          // 输入张量 self
    const ArrayRef<TensorIndex>& indices,        // 索引数组
    bool disable_slice_optimization = false) {   // 是否禁用切片优化，默认为 false
  at::Device self_device = self.device();        // 获取输入张量的设备信息

  // NOTE [nested tensor size for indexing]
  // 嵌套张量目前没有确定的大小，因此暂时表示为 null
  std::optional<SymIntArrayRef> self_sizes = self.is_nested()
      ? std::optional<SymIntArrayRef>(c10::nullopt)
      : std::optional<SymIntArrayRef>(self.sym_sizes());

  // 处理简单类型的索引：整数、切片、None、ellipsis、布尔类型
  if (indices.size() == 1) {                     // 如果索引数组长度为 1
    const TensorIndex& index = indices[0];       // 获取第一个索引

    if (index.is_integer()) {                    // 如果索引为整数
      // 应用选择操作，返回指定位置的子张量
      return impl::applySelect(
          self, 0, index.integer(), 0, self_device, self_sizes);
    } else if (index.is_slice()) {               // 如果索引为切片
      // 应用切片操作，返回切片后的子张量
      return impl::applySlice(
          self,
          0,
          index.slice().start(),
          index.slice().stop(),
          index.slice().step(),
          /*disable_slice_optimization=*/true,
          self_device,
          self_sizes);
    } else if (index.is_none()) {                // 如果索引为 None
      // 在零维度处添加一个维度，返回处理后的张量
      return self.unsqueeze(0);
    } else if (index.is_ellipsis()) {            // 如果索引为 ellipsis
      // 返回原张量的别名（浅拷贝）
      return at::alias(self);
    } else if (index.is_boolean()) {             // 如果索引为布尔类型
      // 将布尔值转换为索引张量，返回处理后的结果
      Tensor result = self.unsqueeze(0);
      return dispatch_index(
          result,
          std::vector<Tensor>{impl::boolToIndexingTensor(
              result, index.boolean(), self_device)});
    }
  }

  std::vector<Tensor> tensorIndices;             // 定义张量索引的数组
  // 应用切片操作，返回切片后的子张量
  Tensor sliced = impl::applySlicing(
      self,
      indices,
      tensorIndices,
      disable_slice_optimization,
      self_device,
      self_sizes);

  if (tensorIndices.empty()) {                   // 如果张量索引为空
    if (sliced.is_same(self)) {
      // 确保对于像 x[...] 这样的情况返回浅拷贝
      sliced = at::alias(sliced);
    }
    return sliced;                               // 返回切片后的子张量
  }

  // 使用张量索引进行高级索引，返回处理后的结果
  return dispatch_index(sliced, std::move(tensorIndices));
}

// 定义函数 `set_item`，用于在张量中设置指定索引处的子张量的值
// `disable_slice_optimization` 参数用于控制是否禁用切片优化
inline void set_item(
    const Tensor& self,                          // 输入张量 self
    const ArrayRef<TensorIndex>& indices,        // 索引数组
    const Tensor& value,                         // 要设置的子张量的值
    bool disable_slice_optimization = false) {   // 是否禁用切片优化，默认为 false
  at::Device self_device = self.device();        // 获取输入张量的设备信息

  SymIntArrayRef self_sizes = self.sym_sizes();  // 获取输入张量的符号化大小

  // 处理简单类型的索引：整数、切片、ellipsis、布尔类型
  if (indices.size() == 1) {                     // 如果索引数组长度为 1
    const TensorIndex& index = indices[0];       // 获取第一个索引

    // 此处省略了处理整数、切片、ellipsis 和布尔类型索引的代码
    // ...
    // 如果索引是布尔类型且为假，不进行任何操作（实际上应该检查大小，但我们没有真正大小为零的形状）
    if (index.is_boolean() && !index.boolean()) {
        // 返回，结束函数
        return;
    } else if (index.is_ellipsis()) {
        // 如果索引是省略号，将值复制到当前对象中
        copy_to(self, value);
        // 返回，结束函数
        return;
    } else if (index.is_none() || (index.is_boolean() && index.boolean())) {
        // 如果索引是None或者是布尔类型且为真，将值复制到展开为一维后的对象中
        copy_to(self.unsqueeze(0), value);
        // 返回，结束函数
        return;
    } else if (index.is_integer()) {
        // 如果索引是整数，利用applySelect函数获取指定位置的子张量，然后将值复制进去
        copy_to(
            impl::applySelect(
                self, 0, index.integer(), 0, self_device, self_sizes),
            value);
        // 返回，结束函数
        return;
    } else if (index.is_slice()) {
        // 如果索引是切片类型，利用applySlice函数对当前对象进行切片操作，然后将值复制进去
        copy_to(
            impl::applySlice(
                self,
                0,
                index.slice().start(),
                index.slice().stop(),
                index.slice().step(),
                /*disable_slice_optimization=*/disable_slice_optimization,
                self_device,
                self_sizes),
            value);
        // 返回，结束函数
        return;
    }
  }

  // 创建一个空的Tensor数组来存储张量索引
  std::vector<Tensor> tensorIndices;
  // 使用applySlicing函数对当前对象进行切片操作，返回切片后的张量
  Tensor sliced = impl::applySlicing(
      self,
      indices,
      tensorIndices,
      disable_slice_optimization,
      self_device,
      self_sizes);
  // 如果tensorIndices为空，则直接将切片后的张量sliced复制为值
  if (tensorIndices.empty()) {
    copy_to(sliced, value);
    // 返回，结束函数
    return;
  }

  // 获取值的符号化大小
  SymIntArrayRef valueSizes = value.sym_sizes();
  // 获取切片前缀1的大小
  SymIntArrayRef slicedValueSizes = slicePrefix1sSize(valueSizes);
  Tensor valuesSliced;
  // 如果值的大小与切片后的大小不相等，则对值进行视图变换成切片后的大小
  if (!valueSizes.equals(slicedValueSizes)) {
    valuesSliced = value.view_symint(slicedValueSizes);
  } else {
    // 否则直接使用原始值
    valuesSliced = value;
  }
  // 调度index_put_函数来处理切片后的张量sliced、张量索引和值valuesSliced
  dispatch_index_put_(sliced, std::move(tensorIndices), valuesSliced);
  // 返回，结束函数
  return;
}

} // namespace at::indexing
```