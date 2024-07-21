# `.\pytorch\torch\csrc\api\include\torch\nn\options\padding.h`

```
#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a `D`-dimensional ReflectionPad module.
template <size_t D>
struct TORCH_API ReflectionPadOptions {
  // 构造函数，接受一个大小为 D*2 的扩展数组作为参数，用于指定填充大小
  ReflectionPadOptions(ExpandingArray<D * 2> padding) : padding_(padding) {}

  /// The size of the padding.
  /// If it is `int`, uses the same padding in all boundaries.
  /// If it is a 2-`tuple` (for ReflectionPad1d), uses (padding_left,
  /// padding_right). If it is a 4-`tuple` (for ReflectionPad2d), uses
  /// (padding_left, padding_right, padding_top, padding_bottom). If it is a
  /// 6-`tuple` (for ReflectionPad3d), uses (padding_left, padding_right,
  /// padding_top, padding_bottom, padding_front, padding_back).

  TORCH_ARG(ExpandingArray<D * 2>, padding);
};

/// `ReflectionPadOptions` specialized for the `ReflectionPad1d` module.
///
/// Example:
/// ```
/// ReflectionPad1d model(ReflectionPad1dOptions({3, 1}));
/// ```
using ReflectionPad1dOptions = ReflectionPadOptions<1>;

/// `ReflectionPadOptions` specialized for the `ReflectionPad2d` module.
///
/// Example:
/// ```
/// ReflectionPad2d model(ReflectionPad2dOptions({1, 1, 2, 0}));
/// ```
using ReflectionPad2dOptions = ReflectionPadOptions<2>;

/// `ReflectionPadOptions` specialized for the `ReflectionPad3d` module.
///
/// Example:
/// ```
/// ReflectionPad3d model(ReflectionPad3dOptions({1, 1, 2, 0, 1, 1}));
/// ```
using ReflectionPad3dOptions = ReflectionPadOptions<3>;

// ============================================================================

/// Options for a `D`-dimensional ReplicationPad module.
template <size_t D>
struct TORCH_API ReplicationPadOptions {
  // 构造函数，接受一个大小为 D*2 的扩展数组作为参数，用于指定填充大小
  ReplicationPadOptions(ExpandingArray<D * 2> padding) : padding_(padding) {}

  /// The size of the padding.
  /// - If it is `int`, uses the same padding in all boundaries.
  /// - If it is a 2-`tuple` (for ReplicationPad1d), uses (padding_left,
  /// padding_right).
  /// - If it is a 4-`tuple` (for ReplicationPad2d), uses (padding_left,
  /// padding_right, padding_top, padding_bottom).
  /// - If it is a 6-`tuple` (for ReplicationPad3d), uses
  ///   (padding_left, padding_right, padding_top, padding_bottom,
  ///   padding_front, padding_back).
  TORCH_ARG(ExpandingArray<D * 2>, padding);
};

/// `ReplicationPadOptions` specialized for the `ReplicationPad1d` module.
///
/// Example:
/// ```
/// ReplicationPad1d model(ReplicationPad1dOptions({3, 1}));
/// ```
using ReplicationPad1dOptions = ReplicationPadOptions<1>;

/// `ReplicationPadOptions` specialized for the `ReplicationPad2d` module.
///
/// Example:
/// ```
/// ReplicationPad2d model(ReplicationPad2dOptions({1, 1, 2, 0}));
/// ```
using ReplicationPad2dOptions = ReplicationPadOptions<2>;

/// `ReplicationPadOptions` specialized for the `ReplicationPad3d` module.
///
/// Example:
/// ```
/// ReplicationPad3d model(ReplicationPad3dOptions({1, 1, 2, 0, 1, 1}));
/// ```
using ReplicationPad3dOptions = ReplicationPadOptions<3>;

} // namespace nn
} // namespace torch
/// 创建一个 `ReplicationPad3dOptions` 类型的别名，使用 `ReplicationPadOptions` 模板，指定维度为 3。
/// 这里初始化了一个具体的对象，其填充参数为 {1, 2, 1, 2, 1, 2}。
/// ```
using ReplicationPad3dOptions = ReplicationPadOptions<3>;

// ============================================================================

/// 模板定义：用于 `D` 维度的零填充选项。
template <size_t D>
struct TORCH_API ZeroPadOptions {
  /// 构造函数：接受一个 `ExpandingArray<D * 2>` 类型的填充参数。
  ZeroPadOptions(ExpandingArray<D * 2> padding) : padding_(padding) {}

  /// 填充的大小。
  /// - 如果是 `int`，在所有边界使用相同的填充。
  /// - 如果是 2 元组（用于 ZeroPad1d），使用 (padding_left, padding_right)。
  /// - 如果是 4 元组（用于 ZeroPad2d），使用 (padding_left, padding_right, padding_top, padding_bottom)。
  /// - 如果是 6 元组（用于 ZeroPad3d），使用
  ///   (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)。
  TORCH_ARG(ExpandingArray<D * 2>, padding);
};

/// `ZeroPadOptions` 的特化版本，适用于 `ZeroPad1d` 模块。
///
/// 示例：
/// ```
/// ConstantPad1d model(ConstantPad1dOptions({3, 1});
/// ```
using ZeroPad1dOptions = ZeroPadOptions<1>;

/// `ZeroPadOptions` 的特化版本，适用于 `ZeroPad2d` 模块。
///
/// 示例：
/// ```
/// ConstantPad2d model(ConstantPad2dOptions({1, 1, 2, 0});
/// ```
using ZeroPad2dOptions = ZeroPadOptions<2>;

/// `ZeroPadOptions` 的特化版本，适用于 `ZeroPad3d` 模块。
///
/// 示例：
/// ```
/// ConstantPad3d model(ConstantPad3dOptions({1, 2, 1, 2, 1, 2});
/// ```
using ZeroPad3dOptions = ZeroPadOptions<3>;

// ============================================================================

/// `D` 维度常量填充模块的选项。
template <size_t D>
struct TORCH_API ConstantPadOptions {
  /// 构造函数：接受一个 `ExpandingArray<D * 2>` 类型的填充参数和一个 `double` 类型的值。
  ConstantPadOptions(ExpandingArray<D * 2> padding, double value)
      : padding_(padding), value_(value) {}

  /// 填充的大小。
  /// - 如果是 `int`，在所有边界使用相同的填充。
  /// - 如果是 2 元组（用于 ConstantPad1d），使用 (padding_left, padding_right)。
  /// - 如果是 4 元组（用于 ConstantPad2d），使用 (padding_left, padding_right, padding_top, padding_bottom)。
  /// - 如果是 6 元组（用于 ConstantPad3d），使用
  ///   (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)。
  TORCH_ARG(ExpandingArray<D * 2>, padding);

  /// 常量填充的填充值。
  TORCH_ARG(double, value);
};

/// `ConstantPadOptions` 的特化版本，适用于 `ConstantPad1d` 模块。
///
/// 示例：
/// ```
/// ConstantPad1d model(ConstantPad1dOptions({3, 1}, 3.5));
/// ```
using ConstantPad1dOptions = ConstantPadOptions<1>;

/// `ConstantPadOptions` 的特化版本，适用于 `ConstantPad2d` 模块。
///
/// 示例：
/// ```
/// ConstantPad2d model(ConstantPad2dOptions({3, 0, 2, 1}, 3.5));
/// ```
using ConstantPad2dOptions = ConstantPadOptions<2>;

/// `ConstantPadOptions` 的特化版本，适用于 `ConstantPad3d` 模块。
///
/// 示例：
/// ```
/// ConstantPad3d model(ConstantPad3dOptions({1, 2, 1, 2, 1, 2}, 3.5));
/// ```
// 定义一个别名 ConstantPad3dOptions，代表 ConstantPadOptions 模板类的 3 维版本
using ConstantPad3dOptions = ConstantPadOptions<3>;

// ============================================================================

// 命名空间 functional 开始

namespace functional {

/// Options for `torch::nn::functional::pad`.
/// 对于 `torch::nn::functional::pad` 函数的选项。

/// Example:
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::pad(input, F::PadFuncOptions({1, 2, 2, 1, 1, 2}).mode(torch::kReplicate));
/// ```
struct TORCH_API PadFuncOptions {
  // 声明一个枚举类型 mode_t，可以是 enumtype 命名空间中的四个值之一
  // enumtype::kConstant, enumtype::kReflect, enumtype::kReplicate, enumtype::kCircular
  typedef std::variant<
      enumtype::kConstant,
      enumtype::kReflect,
      enumtype::kReplicate,
      enumtype::kCircular>
      mode_t;

  // 构造函数，接受一个整数向量 pad 作为参数
  PadFuncOptions(std::vector<int64_t> pad);

  /// m-elements tuple, where m/2 <= input dimensions and m is even.
  /// 表示 m 个元素的元组，其中 m/2 <= 输入维度，且 m 是偶数。

  // 定义 TORCH_ARG 宏，用于声明 pad 成员变量，类型为 std::vector<int64_t>
  TORCH_ARG(std::vector<int64_t>, pad);

  /// "constant", "reflect", "replicate" or "circular". Default: "constant"
  /// "constant", "reflect", "replicate" 或 "circular"。默认值为 "constant"

  // 定义 TORCH_ARG 宏，用于声明 mode 成员变量，类型为 mode_t，初始值为 torch::kConstant
  TORCH_ARG(mode_t, mode) = torch::kConstant;

  /// fill value for "constant" padding. Default: 0
  /// "constant" 填充的填充值。默认为 0

  // 定义 TORCH_ARG 宏，用于声明 value 成员变量，类型为 double，初始值为 0
  TORCH_ARG(double, value) = 0;
};

} // namespace functional

} // namespace nn
} // namespace torch
```