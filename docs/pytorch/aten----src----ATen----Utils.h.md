# `.\pytorch\aten\src\ATen\Utils.h`

```py
#pragma once

#include <ATen/EmptyTensor.h>
#include <ATen/Formatting.h>
#include <ATen/core/ATenGeneral.h>
#include <ATen/core/Generator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <sstream>
#include <typeinfo>

// 定义宏，禁止类型的复制和赋值操作
#define AT_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;         \
  void operator=(const TypeName&) = delete

namespace at {

// 在 Torch API 中声明一个函数，如果启用了地址安全性分析工具（ASAN），则会崩溃
TORCH_API int _crash_if_asan(int);

// 将 TensorList（即 ArrayRef<Tensor>）转换为 TensorImpl* 的 vector
// 注意：此函数仅用于传统的 TH 绑定，并且仅在 cat 函数中使用。
// 一旦 cat 函数完全迁移到 ATen，此函数可以删除！
static inline std::vector<TensorImpl*> checked_dense_tensor_list_unwrap(
    ArrayRef<Tensor> tensors,        // 输入的 Tensor 列表
    const char* name,                // 参数名称
    int pos,                         // 参数在参数列表中的位置
    c10::DeviceType device_type,     // 期望的设备类型
    ScalarType scalar_type) {        // 期望的标量类型
  std::vector<TensorImpl*> unwrapped; // 存储解包后的 TensorImpl 指针的 vector
  unwrapped.reserve(tensors.size());  // 预留空间以存储与输入 Tensor 数量相同的元素
  for (const auto i : c10::irange(tensors.size())) { // 遍历输入的 Tensor 列表
    const auto& expr = tensors[i];   // 获取当前 Tensor
    // 检查当前 Tensor 的布局是否为 Strided
    if (expr.layout() != Layout::Strided) {
      AT_ERROR(
          "Expected dense tensor but got ",
          expr.layout(),
          " for sequence element ",
          i,
          " in sequence argument at position #",
          pos,
          " '",
          name,
          "'");
    }
    // 检查当前 Tensor 的设备类型是否符合期望
    if (expr.device().type() != device_type) {
      AT_ERROR(
          "Expected object of device type ",
          device_type,
          " but got device type ",
          expr.device().type(),
          " for sequence element ",
          i,
          " in sequence argument at position #",
          pos,
          " '",
          name,
          "'");
    }
    // 检查当前 Tensor 的标量类型是否符合期望
    if (expr.scalar_type() != scalar_type) {
      AT_ERROR(
          "Expected object of scalar type ",
          scalar_type,
          " but got scalar type ",
          expr.scalar_type(),
          " for sequence element ",
          i,
          " in sequence argument at position #",
          pos,
          " '",
          name,
          "'");
    }
    // 将当前 Tensor 的 TensorImpl 指针添加到 unwrapped vector 中
    unwrapped.emplace_back(expr.unsafeGetTensorImpl());
  }
  // 返回存储 TensorImpl 指针的 vector
  return unwrapped;
}

// 检查 intlist 的长度是否为 N，并返回其作为 std::array 的结果
template <size_t N>
std::array<int64_t, N> check_intlist(
    ArrayRef<int64_t> list,     // 输入的 int 列表
    const char* name,           // 参数名称
    int pos) {                  // 参数在参数列表中的位置
  if (list.empty()) {
    // TODO: 这个分支是否必要？以前我们使用 nullptr 与非 nullptr 在 IntList 中处理 stride 时有所不同，作为一种模拟可选的方式。
    list = {};
  }
  auto res = std::array<int64_t, N>(); // 创建一个存储 N 个 int64_t 的数组
  if (list.size() == 1 && N > 1) {
    // 如果输入列表只有一个元素且 N 大于 1，则将数组填充为相同的值
    res.fill(list[0]);
    return res;
  }
  if (list.size() != N) {
    // 如果输入列表长度不等于 N，则抛出错误
    // 注意：此处未完全显示所有错误检查的代码
    # 抛出一个错误消息，指示预期得到的是一个包含 N 个整数的列表，但实际得到了 list 的大小（元素个数）个整数
    AT_ERROR(
        "Expected a list of ",
        N,
        " ints but got ",
        list.size(),
        " for argument #",
        pos,
        " '",
        name,
        "'");
  }
  # 将列表 list 的前 N 个元素复制到目标列表 res 中
  std::copy_n(list.begin(), N, res.begin());
  # 返回复制后的结果列表 res
  return res;
}

// 使用at命名空间中的detail命名空间下的check_size_nonnegative函数
using at::detail::check_size_nonnegative;

// 进入detail命名空间
namespace detail {

// 声明模板函数，返回一个在CPU上的Tensor对象，接受一个值数组和Tensor选项作为参数
template <typename T>
TORCH_API Tensor tensor_cpu(ArrayRef<T> values, const TensorOptions& options);

// 声明模板函数，返回一个在后端的Tensor对象，接受一个值数组和Tensor选项作为参数
template <typename T>
TORCH_API Tensor tensor_backend(ArrayRef<T> values, const TensorOptions& options);

// 声明模板函数，返回一个在CPU上的复杂数Tensor对象，接受一个值数组和Tensor选项作为参数
template <typename T>
TORCH_API Tensor tensor_complex_cpu(ArrayRef<T> values, const TensorOptions& options);

// 声明模板函数，返回一个在后端的复杂数Tensor对象，接受一个值数组和Tensor选项作为参数
template <typename T>
TORCH_API Tensor tensor_complex_backend(ArrayRef<T> values, const TensorOptions& options);

} // namespace detail

} // namespace at


这段代码是C++的命名空间和模板声明示例，按照要求给每一行都添加了注释，解释了它们的作用和所属关系。
```