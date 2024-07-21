# `.\pytorch\torch\csrc\api\include\torch\nn\options\linear.h`

```py
#pragma once

// `#pragma once` 是预处理器指令，用于确保头文件只被包含一次，增加编译效率。


#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/types.h>

// 包含 Torch 库的头文件，提供了必要的类型和函数声明以及导出信息。


namespace torch {
namespace nn {

// 将代码置于 torch 和 nn 命名空间中，这样可以避免命名冲突并组织代码结构。


/// Options for the `Linear` module.
///
/// Example:
/// ```
/// Linear model(LinearOptions(5, 2).bias(false));
/// ```py
struct TORCH_API LinearOptions {
  LinearOptions(int64_t in_features, int64_t out_features);
  /// size of each input sample
  TORCH_ARG(int64_t, in_features);

  /// size of each output sample
  TORCH_ARG(int64_t, out_features);

  /// If set to false, the layer will not learn an additive bias. Default: true
  TORCH_ARG(bool, bias) = true;
};

// `LinearOptions` 结构体定义了 `Linear` 模块的选项，包括输入特征数量、输出特征数量以及是否包含偏置项的设置。


// ============================================================================

/// Options for the `Flatten` module.
///
/// Example:
/// ```
/// Flatten model(FlattenOptions().start_dim(2).end_dim(4));
/// ```py
struct TORCH_API FlattenOptions {
  /// first dim to flatten
  TORCH_ARG(int64_t, start_dim) = 1;
  /// last dim to flatten
  TORCH_ARG(int64_t, end_dim) = -1;
};

// `FlattenOptions` 结构体定义了 `Flatten` 模块的选项，包括要展平的起始和结束维度。


// ============================================================================

/// Options for the `Unflatten` module.
///
/// Note: If input tensor is named, use dimname and namedshape arguments.
///
/// Example:
/// ```
/// Unflatten unnamed_model(UnflattenOptions(0, {2, 2}));
/// Unflatten named_model(UnflattenOptions("B", {{"B1", 2}, {"B2", 2}}));
/// ```py
struct TORCH_API UnflattenOptions {
  typedef std::vector<std::pair<std::string, int64_t>> namedshape_t;

  UnflattenOptions(int64_t dim, std::vector<int64_t> sizes);
  UnflattenOptions(const char* dimname, namedshape_t namedshape);
  UnflattenOptions(std::string dimname, namedshape_t namedshape);

  /// dim to unflatten
  TORCH_ARG(int64_t, dim);
  /// name of dim to unflatten, for use with named tensors
  TORCH_ARG(std::string, dimname);
  /// new shape of unflattened dim
  TORCH_ARG(std::vector<int64_t>, sizes);
  /// new shape of unflattened dim with names, for use with named tensors
  TORCH_ARG(namedshape_t, namedshape);
};

// `UnflattenOptions` 结构体定义了 `Unflatten` 模块的选项，支持对指定维度进行展开，并且支持使用命名张量的维度名称和形状。


// ============================================================================

/// Options for the `Bilinear` module.
///
/// Example:
/// ```
/// Bilinear model(BilinearOptions(3, 2, 4).bias(false));
/// ```py
struct TORCH_API BilinearOptions {
  BilinearOptions(
      int64_t in1_features,
      int64_t in2_features,
      int64_t out_features);
  /// The number of features in input 1 (columns of the input1 matrix).
  TORCH_ARG(int64_t, in1_features);
  /// The number of features in input 2 (columns of the input2 matrix).
  TORCH_ARG(int64_t, in2_features);
  /// The number of output features to produce (columns of the output matrix).
  TORCH_ARG(int64_t, out_features);
  /// Whether to learn and add a bias after the bilinear transformation.
  TORCH_ARG(bool, bias) = true;
};

// `BilinearOptions` 结构体定义了 `Bilinear` 模块的选项，包括两个输入特征数量、输出特征数量以及是否包含偏置项的设置。


} // namespace nn
} // namespace torch

// 结束 torch 和 nn 命名空间的定义，确保所有定义的内容都在正确的命名空间下。
```