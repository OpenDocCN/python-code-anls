# `.\pytorch\aten\src\ATen\native\DistributionTemplates.h`

```
#pragma once

#include <ATen/core/Tensor.h>  // 引入 ATen 核心张量库头文件
#include <ATen/Dispatch.h>  // 引入 ATen 分发机制头文件
#include <ATen/Dispatch_v2.h>  // 引入 ATen 分发机制 v2 版本头文件
#include <ATen/Generator.h>  // 引入 ATen 随机数生成器头文件
#include <ATen/ExpandUtils.h>  // 引入 ATen 扩展工具头文件
#include <ATen/Tensor.h>  // 引入 ATen 张量头文件
#include <ATen/MemoryOverlap.h>  // 引入 ATen 内存重叠检测头文件
#include <ATen/NamedTensorUtils.h>  // 引入 ATen 命名张量工具头文件
#include <ATen/native/Resize.h>  // 引入 ATen 本地重置头文件
#include <ATen/native/TensorIterator.h>  // 引入 ATen 本地张量迭代器头文件
#include <c10/util/Optional.h>  // 引入 c10 可选类型头文件
#include <limits>  // 引入数值上限头文件
#include <cmath>  // 引入数学计算头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>  // 引入 ATen 函数头文件
#else
#include <ATen/ops/empty_like.h>  // 引入 ATen 空张量操作头文件
#include <ATen/ops/empty.h>  // 引入 ATen 空张量操作头文件
#include <ATen/ops/full.h>  // 引入 ATen 满张量操作头文件
#include <ATen/ops/view_as_real.h>  // 引入 ATen 视为实数操作头文件
#endif

namespace at::native::templates {

// ==================================================== Random ========================================================

// `update_from` 的目的是找到最接近的有效 int64_t 数字，可以用作实际的 `from` 值。
// 当前实现的 `random_` 使用 uint64_t 运算，并将结果转换为目标数据类型（scalar_t）。
// 这种转换可能会导致生成的数字大于或等于 `to` 值。例如：
//
//    auto actual = torch::empty({3, 3}, torch::half);
//    actual.random_(0, 65504);
//
// 如果随机的 uint64_t 运算产生 65503 作为随机值，转换为 torch::half 后变成 65504，
// 违反了随机值必须小于 `to` 的要求。为了解决这个问题，`update_from` 和 `update_to`
// 将 `from` 向右移动，`to` 向左移动到下一个最接近的值，使得转换为目标数据类型后不会超出 [from, to) 范围。
// 对于 `to` = 65504，它向左移动 (1 << (log2(to) - 11 + 1)) = 32，变成 65472，这是 torch::half 数据类型的前一个可用数字。
template<typename scalar_t>
int64_t update_from(int64_t from) {
  static_assert(
    std::is_floating_point<scalar_t>::value ||
    std::is_same<scalar_t, at::Half>::value ||
    std::is_same<scalar_t, at::BFloat16>::value, "scalar_t must be floating-point type");
  const auto from_plus_1 = static_cast<int64_t>(static_cast<scalar_t>(from + 1));
  if (from_plus_1 < from) {
    int64_t from_ = std::abs(from + 1);
    int n = 0;
    while (from_ >>= 1) ++n;
    // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
    from = from_plus_1 + (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
  }
  return from;
}

// `update_to` 的目的是找到最接近的有效 int64_t 数字，可以用作实际的 `to` 值。
// 当前实现的 `random_` 使用 uint64_t 运算，并将结果转换为目标数据类型（scalar_t）。
// 这种转换可能会导致生成的数字大于或等于 `to` 值。为了解决这个问题，`update_from` 和 `update_to`
// 将 `from` 向右移动，`to` 向左移动到下一个最接近的值，使得转换为目标数据类型后不会超出 [from, to) 范围。
// 对于 `to` = 65504，它向左移动 (1 << (log2(to) - 11 + 1)) = 32，变成 65472，这是 torch::half 数据类型的前一个可用数字。
template<typename scalar_t>
int64_t update_to(int64_t to) {
  static_assert(
    std::is_floating_point<scalar_t>::value ||
    std::is_same<scalar_t, at::Half>::value ||
    std::is_same<scalar_t, at::BFloat16>::value, "scalar_t must be floating-point type");
  const auto to_minus_1 = static_cast<int64_t>(static_cast<scalar_t>(to - 1));
  if (to_minus_1 >= to) {
    int64_t to_ = std::abs(to - 1);
    int n = 0;
    while (to_ >>= 1) ++n;
    // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
    to = to_minus_1 - (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
  }
  return to;
}
// 宏定义，用于检查张量是否为空，如果为空则直接返回该张量
#define CHECK_EMPTY_AND_RETURN(tensor) \
  if (tensor.numel() == 0) {  \  // 检查张量的元素数量是否为0
    return tensor;  \  // 如果是空张量则立即返回
  }

// 模板函数，实现随机操作的具体逻辑
template<template<typename> class random_kernel, typename RNG>
at::Tensor& random_impl(at::Tensor& self, std::optional<Generator> generator) {
  CHECK_EMPTY_AND_RETURN(self);  // 检查并返回空张量
  auto iter = at::TensorIterator::borrowing_nullary_op(self);  // 创建一个张量迭代器
  random_kernel<RNG>()(iter, generator);  // 调用随机核函数对张量进行操作
  return self;  // 返回处理后的张量
}

// 宏定义，用于检查变量是否超出指定范围，如果超出则抛出错误
#define CHECK_OUT_OF_BOUNDS(var, name, min, max, dtype) \
  TORCH_CHECK(var >= min && var <= max, name , " is out of bounds for ", dtype); \

// 宏定义，用于警告变量是否超出推荐范围
#define WARN_OUT_OF_BOUNDS(var, name, digits, dtype) \
  if (var < -(1LL << digits) || var > (1LL << digits)) { \  // 如果超出范围则输出警告信息
    TORCH_WARN(name , " is out of bounds [-(2^", digits, "), 2^", digits, "]. ", \
      "Due to precision limitations ", dtype, " can support discrete uniform distribution only within this range. ", \
      "This warning will become an error in version 1.7 release, please fix the code in advance"); \
  }

// 静态函数，检查起始点和结束点是否在指定范围内
static void check_from_to_in_range(int64_t from, int64_t to_inc, caffe2::TypeMeta dtype) {
  const auto scalar_type = typeMetaToScalarType(dtype);  // 获取数据类型
  if (isFloatingType(scalar_type)) {  // 如果是浮点数类型
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "check_random_fp_bounds", [&] {
      const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());  // 获取最小值
      const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());  // 获取最大值
      CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);  // 检查起始点是否在范围内
      CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max, dtype);  // 检查结束点是否在范围内

      constexpr auto digits = std::numeric_limits<scalar_t>::digits;  // 获取精度
      WARN_OUT_OF_BOUNDS(from, "from", digits, dtype);  // 警告起始点是否超出推荐范围
      WARN_OUT_OF_BOUNDS(to_inc, "to - 1", digits, dtype);  // 警告结束点是否超出推荐范围
    });
  } else if (scalar_type == kUInt64) {  // 如果是无符号64位整数类型
    // 当你在int64_t和uint64_t之间进行比较时，通常的算术转换会将int64_t值提升为unsigned。
    // 但是这种转换会导致溢出：如果int64_t是-1，则在uint64_t中会提升为0xFFFFFFFFFFFFFFFF。这绝不是正确的操作。
    CHECK_OUT_OF_BOUNDS(from, "from", 0, INT64_MAX, dtype);  // 检查起始点是否在范围内
    CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", 0, INT64_MAX, dtype);  // 检查结束点是否在范围内
  } else if (isIntegralType(scalar_type, /*includeBool=*/true)) {  // 如果是整数类型（包括布尔类型）
    AT_DISPATCH_V2(scalar_type, "check_random_integral_bounds", AT_WRAP([&]() {
      const auto min = static_cast<int64_t>(std::numeric_limits<scalar_t>::lowest());  // 获取最小值
      const auto max = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());  // 获取最大值
      CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);  // 检查起始点是否在范围内
      CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max, dtype);  // 检查结束点是否在范围内
    }), AT_EXPAND(AT_INTEGRAL_TYPES), kUInt16, kUInt32, kBool);  // 扩展处理整数类型
  } else {
    TORCH_CHECK(false, "check_random_bounds handles only integral, floating-point and boolean types");  // 抛出错误，仅处理整数、浮点数和布尔类型
  }
}
template<template<typename> class random_from_to_kernel, typename RNG>
// 函数模板，实现从区间[from, to)或[std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max()]中生成随机数并更新到张量self
at::Tensor& random_from_to_impl(at::Tensor& self, int64_t from, std::optional<int64_t> to_opt, std::optional<Generator> generator) {
  uint64_t range = 0;
  // 借用一个空操作的张量迭代器
  auto iter = at::TensorIterator::borrowing_nullary_op(self);
  if (to_opt.has_value()) {
    // 如果to_opt有值，生成区间[from, to)
    int64_t to = *to_opt;
    // 检查from是否小于to
    TORCH_CHECK(from < to, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);
    // 如果张量是浮点类型
    if (isFloatingType(iter.dtype())) {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "random_update_from_to", [&] {
        // 更新from和to以匹配张量的数据类型
        from = update_from<scalar_t>(from);
        to = update_to<scalar_t>(to);
        // 再次检查from是否小于to
        TORCH_CHECK(from < to, "random_ expects 'from' casted to dtype to be less than 'to' casted to dtype, but got from=", from, " >= to=", to);
      });
    }
    // 检查from和to是否在合理范围内
    check_from_to_in_range(from, to - 1, self.dtype());
    // 检查张量是否为空并立即返回
    CHECK_EMPTY_AND_RETURN(self);
    // 计算范围
    range = static_cast<uint64_t>(to) - static_cast<uint64_t>(from);
    // 使用指定的随机数生成器生成随机数
    random_from_to_kernel<RNG>()(iter, range, from, generator);
  } else if (from != std::numeric_limits<int64_t>::lowest()) {
    // 如果to_opt没有值，生成区间[from, std::numeric_limits<int64_t>::max()]
    int64_t to_inc = 0;
    // 如果张量是浮点类型
    if (isFloatingType(iter.dtype())) {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "random_from_to_range_calc", [&] {
        // 计算标量类型的最大值
        constexpr int64_t scalar_t_max = static_cast<int64_t>(1) << std::numeric_limits<scalar_t>::digits;
        // 计算to_inc
        to_inc = scalar_t_max > std::numeric_limits<int64_t>::max() ? std::numeric_limits<int64_t>::max() : static_cast<int64_t>(scalar_t_max);
        // 更新from以匹配张量的数据类型，并检查from是否小于或等于to_inc
        from = update_from<scalar_t>(from);
        TORCH_CHECK(from < to_inc, "random_ expects 'from' casted to dtype to be less than or equal to 'to_inc' casted to dtype, but got from=", from, " > to_inc=", to_inc);
      });
    } else if (isIntegralType(iter.dtype(), /*includeBool=*/true)) {
      AT_DISPATCH_V2(self.scalar_type(), "random_from_to_range_calc", AT_WRAP([&] {
        // 对于整数类型，计算to_inc
        if constexpr (std::is_same_v<scalar_t, bool>) {
          to_inc = static_cast<int64_t>(true);
        } else {
          to_inc = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
        }
      }), AT_EXPAND(AT_INTEGRAL_TYPES_V2), kBool);
    } else {
      // 如果张量类型既不是浮点也不是整数，抛出错误
      TORCH_CHECK(false, "random_from_to_impl handles only integral, floating-point and boolean types");
    }
    // 检查from和to_inc是否在合理范围内
    check_from_to_in_range(from, to_inc, self.dtype());
    // 检查张量是否为空并立即返回
    CHECK_EMPTY_AND_RETURN(self);
    // 计算范围
    range = static_cast<uint64_t>(to_inc) - static_cast<uint64_t>(from) + 1;
    // 使用指定的随机数生成器生成随机数
    random_from_to_kernel<RNG>()(iter, range, from, generator);
  } else {
    // 如果from等于std::numeric_limits<int64_t>::lowest()，生成区间[std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max()]
    // 范围为2^64
    // 检查张量是否为空并立即返回
    CHECK_EMPTY_AND_RETURN(self);
    // 使用指定的随机数生成器生成随机数
    random_from_to_kernel<RNG>()(iter, generator);
  }
  // 返回更新后的张量self
  return self;
}
#define CHECK_NORMAL_TENSOR_STD(std) \  // 定义宏：检查张量 std 的标准差是否合法
  do { \  // 开始 do-while 循环
    TORCH_CHECK( \  // 使用 Torch 检查
      !std.is_complex(), \  // 确保标准差不是复数
      "normal expects standard deviation to be non-complex"); \  // 报错信息：normal 要求标准差必须是非复数
    TORCH_CHECK( \  // 使用 Torch 检查
      std.numel() == 0 || std.is_meta() || std.min().ge(0).item<bool>(), \  // 确保标准差中的所有元素都 >= 0.0
      "normal expects all elements of std >= 0.0"); \  // 报错信息：normal 要求标准差的所有元素都 >= 0.0
  } while (0)  // 结束 do-while 循环，0 是一个无操作语句，用于结束宏的定义

#define CHECK_NORMAL_STD(std) \  // 定义宏：检查标准差 std 是否合法
  TORCH_CHECK(std >= 0.0, "normal expects std >= 0.0, but found std ", std);  // 使用 Torch 检查标准差 std 是否 >= 0.0，否则输出错误信息

template<template<typename> class normal_kernel, typename RNG>  // 模板函数定义，normal_impl_

Tensor& normal_impl_(Tensor& self, double mean, double std, std::optional<Generator> gen) {  // 实现正态分布函数，对给定张量 self 进行操作
  CHECK_NORMAL_STD(std);  // 调用宏，检查标准差 std 是否合法
  CHECK_EMPTY_AND_RETURN(self);  // 调用函数，检查张量 self 是否为空，如果为空则返回

  if (self.is_complex()) {  // 如果张量 self 是复数类型
    auto float_tensor = at::view_as_real(self);  // 将复数张量 self 视图转换为实部张量 float_tensor
    // 实部和虚部的正态分布方差为输入方差的一半
    normal_kernel<RNG>()(float_tensor, mean, std/(std::sqrt(2)), gen);  // 调用指定类型的正态分布生成器，对 float_tensor 进行正态分布采样
  } else {  // 如果张量 self 不是复数类型
    normal_kernel<RNG>()(self, mean, std, gen);  // 调用指定类型的正态分布生成器，对 self 进行正态分布采样
  }
  return self;  // 返回处理后的张量 self
}

template<template<typename> class normal_kernel, typename RNG>  // 模板函数定义，normal_out_impl，输出到指定张量

Tensor& normal_out_impl(Tensor& output, const Tensor& mean, double std, std::optional<Generator> gen) {  // 实现正态分布函数，将结果输出到指定张量 output
  CHECK_NORMAL_STD(std);  // 调用宏，检查标准差 std 是否合法
  auto std_tensor = at::empty_like(output, MemoryFormat::Contiguous);  // 根据输出张量 output 的形状创建与之相同的空张量 std_tensor
  auto shape = at::infer_size(mean.sizes(), std_tensor.sizes());  // 推断输出形状
  at::native::resize_output(output, shape);  // 调整输出张量 output 的形状
  normal_impl_<normal_kernel, RNG>(output, 0, std, gen);  // 调用内部实现函数，对 output 进行正态分布采样
  output.add_(mean);  // 将均值 mean 加到输出张量 output 上
  return output;  // 返回处理后的输出张量 output
}

template<template<typename> class normal_kernel, typename RNG>  // 模板函数定义，normal_out_impl，输出到指定张量

Tensor& normal_out_impl(Tensor& output, double mean, const Tensor& std, std::optional<Generator> gen) {  // 实现正态分布函数，将结果输出到指定张量 output
  CHECK_NORMAL_TENSOR_STD(std);  // 调用宏，检查张量 std 的标准差是否合法
  auto mean_tensor = at::full({}, mean, output.options());  // 创建与输出张量 output 类型相同，且填充为指定值 mean 的张量 mean_tensor
  auto shape = at::infer_size(mean_tensor.sizes(), std.sizes());  // 推断输出形状
  at::native::resize_output(output, shape);  // 调整输出张量 output 的形状
  normal_impl_<normal_kernel, RNG>(output, 0, 1, gen);  // 调用内部实现函数，对 output 进行标准正态分布采样
  // CUDA 注意：addcmul_out 将要加的张量复制到输出中。
  // 此前的函数是 addcmul_out(output, mean_tensor, output, std, 1);
  // 第三个参数不是常量引用，因此输出中的样本被覆盖。
  // 结果计算为 mean_tensor + mean_tensor * std 而不是 mean_tensor + output * std
  output.mul_(std).add_(mean_tensor);  // 对输出张量 output 进行乘法和加法操作
  return output;  // 返回处理后的输出张量 output
}
// 对输出张量进行正态分布采样，并进行标准化处理
Tensor& normal_out_impl(Tensor& output, const Tensor& mean, const Tensor& std, std::optional<Generator> gen) {
  // 检查标准差张量是否符合正态分布的要求
  CHECK_NORMAL_TENSOR_STD(std);
  // 推断输出张量的形状，以匹配均值和标准差张量的形状
  auto shape = at::infer_size(mean.sizes(), std.sizes());
  // 根据推断的形状调整输出张量的大小
  at::native::resize_output(output, shape);
  // 使用指定的生成器进行正态分布采样，填充输出张量
  normal_impl_<normal_kernel, RNG>(output, 0, 1, gen);
  // 对输出张量进行标准化处理，替换内容为 mean + output * std
  output.mul_(std).add_(mean);
  // 返回处理后的输出张量
  return output;
}

// 根据给定的均值和标准差进行正态分布采样，返回新的张量
template<template<typename> class normal_kernel, typename RNG>
Tensor normal_impl(const Tensor& mean, double std, std::optional<Generator> gen) {
  // 检查标准差是否符合正态分布的要求
  CHECK_NORMAL_STD(std);
  // 根据输入张量的形状创建一个新的张量，格式为 Contiguous
  Tensor ret = at::empty_like(mean, MemoryFormat::Contiguous);
  // 使用 normal_out_impl 函数进行实际的正态分布采样
  normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
  // 返回正态分布采样结果的新张量
  return ret;
}

// 根据给定的均值和标准差张量进行正态分布采样，返回新的张量
template<template<typename> class normal_kernel, typename RNG>
Tensor normal_impl(double mean, const Tensor& std, std::optional<Generator> gen) {
  // 检查标准差张量是否符合正态分布的要求
  CHECK_NORMAL_TENSOR_STD(std);
  // 根据标准差张量的形状创建一个新的张量，格式为 Contiguous
  Tensor ret = at::empty_like(std, MemoryFormat::Contiguous);
  // 使用 normal_out_impl 函数进行实际的正态分布采样
  normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
  // 返回正态分布采样结果的新张量
  return ret;
}

// 根据给定的均值张量和标准差张量进行正态分布采样，返回新的张量
template<template<typename> class normal_kernel, typename RNG>
Tensor normal_impl(const Tensor& mean, const Tensor& std, std::optional<Generator> gen) {
  // 检查标准差张量是否符合正态分布的要求
  CHECK_NORMAL_TENSOR_STD(std);
  // 推断输出张量的形状，以匹配均值和标准差张量的形状
  auto shape = at::infer_size(mean.sizes(), std.sizes());
  // 根据推断的形状创建一个新的张量，使用均值张量的选项和格式为 Contiguous
  Tensor ret = at::empty(shape, mean.options(), MemoryFormat::Contiguous);
  // 使用 normal_out_impl 函数进行实际的正态分布采样
  normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
  // 返回正态分布采样结果的新张量
  return ret;
}
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "check_uniform_bounds", [&] {
      // 根据当前张量的数据类型确定相应的类型
      const auto dtype = self.dtype();
      // 获取当前数据类型的最小值，并转换为 double 类型
      const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
      // 获取当前数据类型的最大值，并转换为 double 类型
      const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
      // 检查 'from' 参数是否超出范围
      CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
      // 检查 'to' 参数是否超出范围
      CHECK_OUT_OF_BOUNDS(to, "to", min, max, dtype);
      // 检查 'from' 是否小于或等于 'to'，否则抛出错误
      TORCH_CHECK(from <= to, "uniform_ expects to return a [from, to) range, but found from=", from, " > to=", to);
      // 检查 (to - from) 是否在当前数据类型的最大范围内
      TORCH_CHECK((to - from) <= std::numeric_limits<scalar_t>::max(),
            "uniform_ expects to-from <= std::numeric_limits<", toString(self.scalar_type()),
            ">::max(), but found to=", to, " and from=", from,
            " which result in to-from to exceed the limit");
      // 将 'from' 限制在最小值和最大值之间
      from = std::min(std::max(from, min), max);
      // 将 'to' 限制在最小值和最大值之间
      to = std::max(std::min(to, max), min);
    });
    // 检查当前张量是否为空，若为空则直接返回
    CHECK_EMPTY_AND_RETURN(self);
    // 创建一个张量迭代器，使用当前张量进行空操作
    auto iter = at::TensorIterator::borrowing_nullary_op(self);
    // 调用 uniform_kernel 函数对迭代器进行处理，生成均匀分布的随机数
    uniform_kernel<RNG>()(iter, from, to, generator);
  }
  // 返回处理后的张量
  return self;
// 结束模板定义的右花括号，表示该模板的定义结束
}

// ================================================== LogNormal =======================================================

// 对 log_normal_impl_ 函数模板进行实现，使用指定的 RNG 引擎
template<template<typename> class log_normal_kernel, typename RNG>
// 修改传入的 self 引用 Tensor，按照对数正态分布生成数据
at::Tensor& log_normal_impl_(at::Tensor& self, double mean, double std, std::optional<Generator> gen) {
  // 检查标准差 std 必须大于 0
  TORCH_CHECK(std > 0.0, "log_normal_ expects std > 0.0, but found std=", std);
  // 检查并返回空张量的情况
  CHECK_EMPTY_AND_RETURN(self);
  // 借用 nullary 操作创建张量迭代器
  auto iter = TensorIterator::borrowing_nullary_op(self);
  // 调用 log_normal_kernel<RNG>() 模板生成数据
  log_normal_kernel<RNG>()(iter, mean, std, gen);
  // 返回生成的张量 self
  return self;
}

// =================================================== Geometric ======================================================

// 对 geometric_impl_ 函数模板进行实现，使用指定的 RNG 引擎
template<template<typename> class geometric_kernel, typename RNG>
// 修改传入的 self 引用 Tensor，按照几何分布生成数据
Tensor& geometric_impl_(Tensor& self, double p, std::optional<Generator> gen) {
  // 检查参数 p 必须在 (0, 1) 之间
  TORCH_CHECK(0 < p && p < 1, "geometric_ expects p to be in (0, 1), but got p=", p);
  // 检查并返回空张量的情况
  CHECK_EMPTY_AND_RETURN(self);
  // 借用 nullary 操作创建张量迭代器
  auto iter = TensorIterator::borrowing_nullary_op(self);
  // 调用 geometric_kernel<RNG>() 模板生成数据
  geometric_kernel<RNG>()(iter, p, gen);
  // 返回生成的张量 self
  return self;
}

// ================================================== Exponential =====================================================

// 对 exponential_impl_ 函数模板进行实现，使用指定的 RNG 引擎
template<template<typename> class exponential_kernel, typename RNG>
// 修改传入的 self 引用 Tensor，按照指数分布生成数据
Tensor& exponential_impl_(Tensor& self, double lambda, std::optional<Generator> gen) {
  // 检查参数 lambda 必须大于 0
  TORCH_CHECK(lambda > 0.0, "exponential_ expects lambda > 0.0, but found lambda=", lambda);
  // 检查并返回空张量的情况
  CHECK_EMPTY_AND_RETURN(self);
  // 借用 nullary 操作创建张量迭代器
  auto iter = TensorIterator::borrowing_nullary_op(self);
  // 调用 exponential_kernel<RNG>() 模板生成数据
  exponential_kernel<RNG>()(iter, lambda, gen);
  // 返回生成的张量 self
  return self;
}

// ==================================================== Cauchy ========================================================

// 对 cauchy_impl_ 函数模板进行实现，使用指定的 RNG 引擎
template<template<typename> class cauchy_kernel, typename RNG>
// 修改传入的 self 引用 Tensor，按照柯西分布生成数据
Tensor& cauchy_impl_(Tensor& self, double median, double sigma, std::optional<Generator> gen) {
  // TODO: instead of variable name 'sigma', use 'gamma' or 'scale'
  // 检查参数 sigma 必须大于 0
  TORCH_CHECK(sigma > 0.0, "cauchy_ expects sigma > 0.0, but found sigma=", sigma);
  // 检查张量的数据类型必须是浮点型
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "Cauchy distribution is a continuous probability distribution. dtype must be a floating point but you specified ", self.dtype());
  // 检查并返回空张量的情况
  CHECK_EMPTY_AND_RETURN(self);
  // 借用 nullary 操作创建张量迭代器
  auto iter = TensorIterator::borrowing_nullary_op(self);
  // 调用 cauchy_kernel<RNG>() 模板生成数据
  cauchy_kernel<RNG>()(iter, median, sigma, gen);
  // 返回生成的张量 self
  return self;
}

// ==================================================== Bernoulli =====================================================

// 对 bernoulli_impl_ 函数模板进行实现，使用指定的 RNG 引擎
template<template<typename> class bernoulli_tensor_kernel, typename RNG>
// 修改传入的 self 引用 Tensor，按照伯努利分布生成数据
Tensor& bernoulli_impl_(Tensor& self, const Tensor& p_, std::optional<Generator> gen) {
  // 检查并返回空张量的情况
  CHECK_EMPTY_AND_RETURN(self);
  // 临时禁用命名保护，确保不会有重叠命名
  NoNamesGuard guard;
  // 断言不会出现内部重叠
  at::assert_no_internal_overlap(self);
  // 调用 bernoulli_tensor_kernel<RNG>() 模板生成数据
  bernoulli_tensor_kernel<RNG>()(self, p_, gen);
  // 返回生成的张量 self
  return self;
}
// 修改自身的 Tensor 引用，并根据给定的概率 p 生成伯努利分布的随机数，如果提供了随机数生成器 gen 则使用该生成器
Tensor& bernoulli_impl_(Tensor& self, double p, std::optional<Generator> gen) {
  // 检查概率 p 是否在 [0, 1] 范围内，否则抛出错误信息
  TORCH_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
  // 检查并返回若 self 为空，则直接返回
  CHECK_EMPTY_AND_RETURN(self);
  // 断言 self 中不存在内部重叠
  at::assert_no_internal_overlap(self);
  // 调用 bernoulli_scalar_kernel<RNG>，生成伯努利分布的随机数填充到 self 中
  bernoulli_scalar_kernel<RNG>()(self, p, gen);
  // 返回修改后的 self 引用
  return self;
}

// 使用指定的模板 bernoulli_tensor_kernel 和 RNG，对 self 进行伯努利分布采样，并将结果保存到 result 中
template<template<typename> class bernoulli_tensor_kernel, typename RNG>
Tensor& bernoulli_out_impl(Tensor& result, const Tensor& self, std::optional<Generator> gen) {
  // result.resize_as_(self) 需要 self 和 result 具有相同的数据类型，因此我们使用 resize_ 来代替
  // TODO: 修复 resize_as_ 的问题，请参见 pytorch/pytorch#11665。
  result.resize_(self.sizes());
  // 调用 bernoulli_impl_，生成伯努利分布的随机数并存储到 result 中
  bernoulli_impl_<bernoulli_tensor_kernel, RNG>(result, self, gen);
  // 将 result 的命名信息从 self 传播过去
  namedinference::propagate_names(result, self);
  // 返回修改后的 result 引用
  return result;
}

#undef CHECK_OUT_OF_BOUNDS
#undef WARN_OUT_OF_BOUNDS
```