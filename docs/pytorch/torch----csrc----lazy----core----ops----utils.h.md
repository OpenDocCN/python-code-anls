# `.\pytorch\torch\csrc\lazy\core\ops\utils.h`

```py
// 包含 vector 标准库头文件，用于定义和操作动态数组
#include <vector>

// 包含 Torch 懒执行模块的头文件，提供与张量操作相关的实用函数和结构
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/util.h>

// Torch 懒执行命名空间，用于组织和封装相关的函数和类
namespace torch {
namespace lazy {

// Torch API 函数声明：检查给定步长数组是否被支持
TORCH_API bool StrideIsSupported(c10::ArrayRef<int64_t> stride);

// Torch API 函数声明：根据步长数组获取其排列后的数组
TORCH_API std::vector<int64_t> GetArrayStridePermutation(
    c10::ArrayRef<int64_t> stride);

// Torch API 函数声明：创建对角线形状，基于给定形状、偏移量以及两个维度
TORCH_API Shape MakeDiagonalShape(
    const Shape& shape,
    int64_t offset,
    int64_t dim1,
    int64_t dim2);

// Torch API 函数声明：根据给定的排列数组创建排列后的形状
TORCH_API Shape MakePermuteShape(const Shape& source_shape, c10::ArrayRef<int64_t> permutation);

// Torch API 函数声明：创建选择形状，基于给定形状、维度、起始、结束和步长
TORCH_API Shape MakeSelectShape(
    const Shape& shape,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t stride);

// Torch API 函数声明：计算给定范围和步长的步幅
TORCH_API int64_t GetStride(int64_t start, int64_t end, int64_t stride);

// Torch API 函数声明：构建经过压缩维度的维度数组
TORCH_API std::vector<int64_t> BuildSqueezedDimensions(
    c10::ArrayRef<int64_t> dimensions,
    int64_t squeeze_dim);

// Torch API 函数声明：构建经过扩展维度的维度数组
TORCH_API std::vector<int64_t> BuildUnsqueezedDimensions(
    c10::ArrayRef<int64_t> dimensions,
    int64_t squeeze_dim);

} // namespace lazy
} // namespace torch
```