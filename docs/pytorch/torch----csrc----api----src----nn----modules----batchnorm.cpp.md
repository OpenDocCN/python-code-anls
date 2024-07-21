# `.\pytorch\torch\csrc\api\src\nn\modules\batchnorm.cpp`

```
# 引入 Torch 库中的批归一化函数定义文件
#include <torch/nn/functional/batchnorm.h>
# 引入 Torch 库中的批归一化模块定义文件
#include <torch/nn/modules/batchnorm.h>

# 引入 Torch 的 CUDA 功能
#include <torch/cuda.h>
# 引入 Torch 的类型定义
#include <torch/types.h>

# 引入 C10 库中的异常处理功能
#include <c10/util/Exception.h>

# 引入标准库中的一些常用类型和操作
#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

# Torch 命名空间
namespace torch {
# Torch 的神经网络命名空间
namespace nn {

# BatchNorm1dImpl 类的成员函数 _check_input_dim，用于检查输入张量的维度是否为 2 或 3
void BatchNorm1dImpl::_check_input_dim(const Tensor& input) {
  # 使用 TORCH_CHECK 宏检查输入张量的维度是否为 2 或 3，否则抛出错误信息
  TORCH_CHECK(
      input.dim() == 2 || input.dim() == 3,
      "expected 2D or 3D input (got ",
      input.dim(),
      "D input)");
}

# BatchNorm2dImpl 类的成员函数 _check_input_dim，用于检查输入张量的维度是否为 4
void BatchNorm2dImpl::_check_input_dim(const Tensor& input) {
  # 使用 TORCH_CHECK 宏检查输入张量的维度是否为 4，否则抛出错误信息
  TORCH_CHECK(
      input.dim() == 4, "expected 4D input (got ", input.dim(), "D input)");
}

# BatchNorm3dImpl 类的成员函数 _check_input_dim，用于检查输入张量的维度是否为 5
void BatchNorm3dImpl::_check_input_dim(const Tensor& input) {
  # 使用 TORCH_CHECK 宏检查输入张量的维度是否为 5，否则抛出错误信息
  TORCH_CHECK(
      input.dim() == 5, "expected 5D input (got ", input.dim(), "D input)");
}

# 实例化 BatchNormImplBase 模板类，分别传入维度 1、2、3 和对应的 BatchNorm 类型
template class BatchNormImplBase<1, BatchNorm1dImpl>;
template class BatchNormImplBase<2, BatchNorm2dImpl>;
template class BatchNormImplBase<3, BatchNorm3dImpl>;

} // namespace nn
} // namespace torch
```