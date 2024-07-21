# `.\pytorch\torch\csrc\api\include\torch\detail\TensorDataContainer.h`

```
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>

#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/tensor.h>
#endif

#include <initializer_list>

namespace torch {

namespace detail {

// 枚举类型，表示 TensorDataContainer 的数据类型：标量、初始化列表或张量
enum class TensorDataContainerType { Scalar, InitList, Tensor };

// 前向声明，用于流操作符重载
struct TensorDataContainer;

// 流操作符重载，用于将 TensorDataContainer 输出到流中
inline std::ostream& operator<<(
    std::ostream& stream,
    const TensorDataContainer& tensor_data_container);

// 计算所需的数据类型，根据输入的标量类型确定返回的张量数据类型
inline c10::ScalarType compute_desired_dtype(c10::ScalarType scalar_type) {
  if (scalar_type == at::kInt || scalar_type == at::kLong) {
    // 如果是整数类型，则返回 kLong 类型，与 Python 的行为保持一致
    return at::kLong;
  } else if (scalar_type == at::kFloat || scalar_type == at::kDouble) {
    // 如果是浮点数类型，则返回默认的数据类型，与 Python 的行为保持一致
    return at::typeMetaToScalarType(at::get_default_dtype());
  } else {
    // 其他类型直接返回输入的标量类型
    return scalar_type;
  }
}

// TensorDataContainer 类用于支持以下数据容器类型转换为等效的 Tensor：
//
// 1. 任意嵌套的初始化列表（如 `{{1, 2}, {3, 4}}`）。
// 2. 支持的张量数据类型的 `at::ArrayRef`。
// 3. 支持的张量数据类型的 `std::vector`。
//
// 在任何时候，TensorDataContainer 对象表示以下之一：
//
// 1. 具有值 `scalar()` 和类型 `scalar_type()` 的标量。
// 2. 以 `std::initializer_list<TensorDataContainer>` 形式表示的 Tensor，具有
//    值 `init_list()`，Tensor 标量类型 `scalar_type()`，和 Tensor 大小 `sizes()`。
// 3. 以 `at::Tensor` 形式表示的 Tensor，具有值 `tensor()`，标量类型 `scalar_type()`，
//    和 Tensor 大小 `sizes()`。
//
// 所有这些基础设施主要是为了成功支持将任意嵌套的初始化列表转换为等效的 Tensor。考虑以下示例：
//
// `torch::tensor({{1}, {2}})`
//
// 这将调用 `torch::tensor` 函数：
//
// `at::Tensor tensor(detail::TensorDataContainer tensor_data_container, const
// at::TensorOptions& options = {})`
//
// 编译器首先尝试将 `{{1}, {2}}` 转换为 `TensorDataContainer` 类型：
//
// `TensorDataContainer({{1}, {2}})`
//
// 这将匹配到 `TensorDataContainer(std::initializer_list<TensorDataContainer>)`
// 构造函数，并尝试将 `{1}` 和 `{2}` 转换为 `TensorDataContainer`，从而调用以下内容：
//
// `TensorDataContainer({1})`（同样的调用路径也适用于 `{2}`，我们这里只关注 `{1}`）
//
// 在这一点上，理论上有两种可能的方法使 `{1}` 匹配到 `TensorDataContainer` 的其中一个构造函数：
//
// 1. 它可以是一个标量值的列表初始化，因此匹配 `TensorDataContainer(int value)`。
// 2. 它可以转换为 `std::initializer_list<TensorDataContainer>`，因此匹配
//    `TensorDataContainer(std::initializer_list<TensorDataContainer>)`。
//
// 编译器如何决定选择哪一个？根据 `https://en.cppreference.com/w/cpp/language/list_initialization`，
// 用大括号初始化列表总是更倾向于选择接受 `std::initializer_list` 的构造函数。因此我们愉快地选择构造函数 #2，
// 并且它调用以下内容：
//
// `TensorDataContainer(1)`
//
// 现在它匹配到 `TensorDataContainer(int value)`，将 `1` 存储为标量值。一切正常。
struct TensorDataContainer {
  // 注意：对于具有零尺寸维度的张量（例如 `torch::tensor({{}, {}})`），最内层的空大括号初始化列表 `{}` 匹配
  // 最内层的 `TensorDataContainer` 的默认构造函数。
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  TensorDataContainer()
      : sizes_({0}),
        // 注意：在 Python 中，具有零尺寸维度的张量的 dtype（例如 `torch.tensor([[], []])`）取决于
        // `torch.get_default_dtype()` 的值，对于 C++ 等效部分我们也应该一样。
        scalar_type_(at::typeMetaToScalarType(at::get_default_dtype())),
        type_(TensorDataContainerType::InitList) {}

#define TENSOR(T, S)                            \
  TensorDataContainer(T value)                  \
      : sizes_(),                               \
        scalar_type_(at::k##S),                 \
        type_(TensorDataContainerType::Scalar), \
        scalar_(value) {}
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AT_FORALL_COMPLEX_TYPES(TENSOR)
#undef TENSOR

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  TensorDataContainer(std::initializer_list<TensorDataContainer> init_list)
      : sizes_(),
        scalar_type_(init_list.begin()->scalar_type()),
        type_(TensorDataContainerType::InitList),
        init_list_(init_list) {
    const TensorDataContainer& first_elem = *(init_list.begin());
    // 对初始化列表中的每个元素进行遍历
    for (const auto& elem : init_list) {
      // 检查当前元素的大小是否与第一个元素的大小相同
      TORCH_CHECK(
          elem.sizes() == first_elem.sizes(),
          "Expected all sub-lists to have sizes: ",
          first_elem.sizes(),
          " (e.g. ",
          first_elem,
          "), ",
          "but got sub-list ",
          elem,
          " with sizes: ",
          elem.sizes());
      // 检查当前元素的标量类型是否与第一个元素的标量类型相同
      TORCH_CHECK(
          elem.scalar_type() == first_elem.scalar_type(),
          "Expected all elements of the tensor to have the same scalar type: ",
          first_elem.scalar_type(),
          ", but got element of scalar type: ",
          elem.scalar_type());
    }
    // 预留空间以存储尺寸信息，包括子列表的数量和第一个元素的尺寸
    sizes_.reserve(first_elem.sizes().size() + 1);
    // 将初始化列表的大小添加到尺寸信息中
    sizes_.push_back(init_list.size());
    // 将第一个元素的尺寸添加到尺寸信息的末尾
    sizes_.insert(
        sizes_.end(), first_elem.sizes().begin(), first_elem.sizes().end());
  }
#define TENSOR(T, S)                                                          \
  // 定义一个构造函数，接受 at::ArrayRef<T> 类型的值并初始化 TensorDataContainer 对象
  TensorDataContainer(at::ArrayRef<T> values)                                 \
      : sizes_({(int64_t)values.size()}),                                     \
        scalar_type_(at::k##S),                                               \
        type_(TensorDataContainerType::Tensor) {                              \
    // 进入 AutoDispatchBelowAutograd 模式，确保在自动求导以下运行
    at::AutoDispatchBelowAutograd mode;                                       \
    // 如果标量类型是布尔类型
    if (scalar_type_ == at::kBool) {                                          \
      // 创建 CPU 上的张量，使用给定的值
      tensor_ = at::tensor(values, at::TensorOptions().device(at::kCPU));     \
    } else {                                                                  \
      // 创建 CPU 上的张量，使用给定的值和标量类型
      tensor_ = at::tensor(values, at::dtype(scalar_type_).device(at::kCPU)); \
    }                                                                         \
  }
  // 对所有标量类型（包括 Bool、Half 和 BFloat16）应用 TENSOR 宏定义
  // 禁止 lint 检查指定类型的成员初始化
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
  // 对所有复数类型应用 TENSOR 宏定义
  AT_FORALL_COMPLEX_TYPES(TENSOR)
#undef TENSOR

// 注意：我们需要显式处理 `std::vector`，而不是依赖隐式转换到 `at::ArrayRef`，否则可能会出现以下错误
// 在调用 `torch::tensor(std::vector<int>({1, 2}))` 时可能会抛出以下错误：
// ```
// error: no matching function for call to 'tensor(const std::vector<int>&)'
// no known conversion for argument 1 from 'const std::vector<int>' to
// 'torch::detail::TensorDataContainer'
// ```
//
// 注意：目前不支持 `torch::tensor(std::vector<bool>)`，因为无法从 std::vector<bool> 位字段构造 ArrayRef<bool>
#define TENSOR(T, S)                                \
  // 定义一个构造函数，接受 std::vector<T> 类型的值并初始化 TensorDataContainer 对象
  TensorDataContainer(const std::vector<T>& values) \
      : TensorDataContainer(at::ArrayRef<T>(values)) {}
  // 对所有标量类型（不包括 Half 和 BFloat16）应用 TENSOR 宏定义
  // 禁止 lint 检查指定类型的成员初始化
  AT_FORALL_SCALAR_TYPES_AND2(Half, BFloat16, TENSOR)
  // 对所有复数类型应用 TENSOR 宏定义
  AT_FORALL_COMPLEX_TYPES(TENSOR)
#undef TENSOR

// 检查对象是否是标量类型
bool is_scalar() const {
  return type_ == TensorDataContainerType::Scalar;
}

// 获取标量对象的引用
const c10::Scalar& scalar() const {
  // 断言对象确实是标量类型，否则抛出错误信息
  TORCH_CHECK(
      is_scalar(),
      "Can only call `scalar()` on a TensorDataContainer that has `is_scalar() == true`");
  return scalar_;
}

// 检查对象是否是初始化列表类型
bool is_init_list() const {
  return type_ == TensorDataContainerType::InitList;
}

// 获取初始化列表对象的引用
const std::initializer_list<TensorDataContainer>& init_list() const {
  // 断言对象确实是初始化列表类型，否则抛出错误信息
  TORCH_CHECK(
      is_init_list(),
      "Can only call `init_list()` on a TensorDataContainer that has `is_init_list() == true`");
  return init_list_;
}

// 检查对象是否是张量类型
bool is_tensor() const {
  return type_ == TensorDataContainerType::Tensor;
}

// 获取张量对象的引用
const at::Tensor& tensor() const {
  // 断言对象确实是张量类型，否则抛出错误信息
  TORCH_CHECK(
      is_tensor(),
      "Can only call `tensor()` on a TensorDataContainer that has `is_tensor() == true`");
    // 返回私有成员 tensor_
    return tensor_;
  }

  // 返回私有成员 sizes_
  const std::vector<int64_t>& sizes() const {
    return sizes_;
  }

  // 返回私有成员 scalar_type_
  const c10::ScalarType& scalar_type() const {
    return scalar_type_;
  }

  // 将 TensorDataContainer 转换为 Tensor 类型
  at::Tensor convert_to_tensor(at::TensorOptions options) const {
    // 如果选项中未指定数据类型，则根据 scalar_type_ 计算所需的数据类型
    if (!options.has_dtype()) {
      options = options.dtype(compute_desired_dtype(scalar_type_));
    }

    // 如果是标量类型
    if (is_scalar()) {
      // 自动选择自动微分下的分发模式，返回一个标量 tensor
      at::AutoDispatchBelowAutograd mode;
      return at::scalar_tensor(scalar_, options);
    } else if (is_init_list()) {
      // 注意：在此处我们明确选择首先在 CPU 上初始化张量，
      // 填充张量的每个元素，然后将张量移动到目标设备。
      // 对于 CUDA 设备，这种方法只涉及 1 次 CUDA 核心启动，
      // 比首先在 CUDA 上初始化张量，然后填充每个元素（涉及 `N` 次 CUDA 核心启动，
      // 其中 `N` 是张量中的元素数量）要快得多。
      at::Tensor tensor = ([&]() {
        at::AutoDispatchBelowAutograd mode;
        return at::empty(sizes_, options.device(at::kCPU));
      })();
      fill_tensor(tensor);  // 填充张量的每个元素
      return tensor.to(options.device());  // 将张量移动到目标设备
    } else if (is_tensor()) {
      auto output = tensor_.to(options);
      // 检查复杂数张量是否可以转换为非复杂数的数据类型，否则抛出错误
      TORCH_CHECK(
          !tensor_.is_complex() || output.is_complex(),
          "can not do torch::tensor(complex, dtype=non-complex) because complex can not be casted to real number without loss of information");
      return output;
    } else {
      // 如果类型无效，则抛出内部断言错误
      TORCH_INTERNAL_ASSERT(false, "Invalid TensorDataContainer type");
    }
  }

  // 递归地在流中美化打印 TensorDataContainer 的内容
  void pretty_print_recursive(std::ostream& stream) const {
    if (is_scalar()) {
      // 对标量类型进行打印
      AT_DISPATCH_ALL_TYPES_AND3(
          at::kBool,
          at::kHalf,
          at::kBFloat16,
          scalar_type_,
          "TensorDataContainer_pretty_print_scalar",
          [&] { stream << scalar_.to<scalar_t>(); });
    } else if (is_init_list()) {
      // 对初始化列表类型进行打印
      stream << "{";
      for (const TensorDataContainer* it = init_list_.begin();
           it != init_list_.end();
           it++) {
        stream << *it;
        if (std::next(it) != init_list_.end())
          stream << ", ";
      }
      stream << "}";
    } else if (is_tensor()) {
      // 对张量类型进行打印
      stream << "{";
      for (const auto i : c10::irange(tensor_.sizes()[0])) {
        AT_DISPATCH_ALL_TYPES_AND3(
            at::kBool,
            at::kHalf,
            at::kBFloat16,
            scalar_type_,
            "TensorDataContainer_pretty_print_tensor_item",
            [&] { stream << tensor_[i].item<scalar_t>(); });
        if (i != tensor_.sizes()[0] - 1)
          stream << ", ";
      }
      stream << "}";
    } else {
      // 如果类型无效，则抛出内部断言错误
      TORCH_INTERNAL_ASSERT(false, "Invalid TensorDataContainer type");
    }
  }

 private:
  // 填充给定的张量 tensor 的每个元素
  void fill_tensor(at::Tensor& tensor) const {
    if (is_scalar()) {
      // 如果数据容器是标量类型
      TORCH_INTERNAL_ASSERT(
          tensor.dim() == 0,
          "Expected a 0-dim Tensor, but got Tensor with dimensions: ",
          tensor.dim());
      // 确保传入的 tensor 是零维的 Tensor
      at::NoGradGuard guard;
      // 创建一个 NoGradGuard 对象，用于临时禁用梯度追踪
      tensor.fill_(scalar_);
      // 使用标量值填充该 Tensor
    } else if (is_init_list()) {
      // 如果数据容器是初始化列表类型
      TORCH_INTERNAL_ASSERT(
          tensor.sizes()[0] == (int64_t)init_list_.size(),
          "Expected a Tensor with size ",
          init_list_.size(),
          " in its first dimension, but got Tensor with size ",
          tensor.sizes()[0],
          " in its first dimension");
      // 确保传入的 tensor 的第一个维度大小与初始化列表的大小相符
      size_t index = 0;
      // 初始化索引变量
      for (const auto& elem : init_list_) {
        // 遍历初始化列表中的每个元素
        at::Tensor slice = tensor[index];
        // 获取 tensor 中的一个切片
        elem.fill_tensor(slice);
        // 使用 elem 对象填充这个切片
        index++;
        // 更新索引
      }
    } else if (is_tensor()) {
      // 如果数据容器是 Tensor 类型，但不应调用 fill_tensor
      TORCH_INTERNAL_ASSERT(
          false,
          "TensorDataContainer is already a Tensor type, `fill_tensor` should not be called");
      // 报错，因为 TensorDataContainer 已经是 Tensor 类型，不应该调用 fill_tensor 方法
    } else {
      // 如果数据容器类型无效
      TORCH_INTERNAL_ASSERT(false, "Invalid TensorDataContainer type");
      // 报错，因为数据容器类型无效
    }
  }

  std::vector<int64_t> sizes_;
  // 存储 Tensor 尺寸的向量
  c10::ScalarType scalar_type_;
  // 存储标量类型的枚举值
  TensorDataContainerType type_;
  // 存储数据容器的类型
  c10::Scalar scalar_;
  // 存储标量值的对象
  std::initializer_list<TensorDataContainer> init_list_;
  // 存储初始化列表的对象
  at::Tensor tensor_;
  // 存储 Tensor 对象
};

// 结束了 `torch` 命名空间的定义

inline std::ostream& operator<<(
    std::ostream& stream,
    const TensorDataContainer& tensor_data_container) {
  // 调用 `pretty_print_recursive` 方法打印 `TensorDataContainer` 对象的内容到流中
  tensor_data_container.pretty_print_recursive(stream);
  // 返回输出流对象
  return stream;
}

// 结束了 `detail` 命名空间的定义

// 结束了 `torch` 命名空间的定义
```