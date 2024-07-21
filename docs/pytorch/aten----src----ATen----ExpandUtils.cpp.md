# `.\pytorch\aten\src\ATen\ExpandUtils.cpp`

```
// 定义宏以便于在编译时仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 ATen 库中的扩展相关头文件
#include <ATen/ExpandUtils.h>
#include <ATen/ExpandBase.h>

// 引入 C++ 标准库中的范围工具
#include <c10/util/irange.h>

// ATen 命名空间
namespace at {
// ATen 内部命名空间
namespace internal {

// 慢速扩展路径函数，根据给定的尺寸扩展张量
TensorBase expand_slow_path(const TensorBase &self, IntArrayRef size) {
  // 利用 OptionalTensorRef 封装的可选引用，调用 expand 方法进行扩展
  return OptionalTensorRef(self)->expand(size);
}

} // namespace internal

// 匿名命名空间，定义模板函数用于推断尺寸
namespace {
// 注意：如果发生更改，请保持与 are_expandable 函数的同步
template <typename Container, typename ArrayType>
Container infer_size_impl(ArrayType a, ArrayType b) {
  // 使用 ptrdiff_t 确保进行有符号比较
  auto dimsA = static_cast<ptrdiff_t>(a.size());
  auto dimsB = static_cast<ptrdiff_t>(b.size());
  // 计算需要扩展的维度数，取较大值
  auto ndim = dimsA > dimsB ? dimsA : dimsB;
  // 创建用于存储扩展后尺寸的容器
  Container expandedSizes(ndim);

  // 从最高维度向最低维度遍历
  for (ptrdiff_t i = ndim - 1; i >= 0; --i) {
    ptrdiff_t offset = ndim - 1 - i;
    ptrdiff_t dimA = dimsA - 1 - offset;
    ptrdiff_t dimB = dimsB - 1 - offset;
    // 获取当前维度的大小，若索引无效则默认为1
    auto sizeA = (dimA >= 0) ? a[dimA] : 1;
    auto sizeB = (dimB >= 0) ? b[dimB] : 1;

    // 检查两个张量在非单一维度上的尺寸是否匹配或是否为单一维度
    TORCH_CHECK(
        sizeA == sizeB || sizeA == 1 || sizeB == 1,
        "The size of tensor a (", sizeA,
        ") must match the size of tensor b (", sizeB,
        ") at non-singleton dimension ", i);

    // 将尺寸为1的维度映射到另一个尺寸上（甚至是0）
    expandedSizes[i] = sizeA == 1 ? std::move(sizeB) : std::move(sizeA);
  }

  return expandedSizes;
}
}

// 推断尺寸的函数，返回 std::vector<int64_t> 类型的结果
std::vector<int64_t> infer_size(IntArrayRef a, IntArrayRef b) {
  return infer_size_impl<std::vector<int64_t>>(a, b);
}

// 推断符号整数尺寸的函数，返回 std::vector<SymInt> 类型的结果
std::vector<SymInt> infer_size_symint(SymIntArrayRef a, SymIntArrayRef b) {
  return infer_size_impl<std::vector<SymInt>>(a, b);
}

// 推断维度向量尺寸的函数，返回 DimVector 类型的结果
DimVector infer_size_dimvector(IntArrayRef a, IntArrayRef b) {
  return infer_size_impl<DimVector, IntArrayRef>(a, b);
}

// 推断符号维度向量尺寸的函数，返回 SymDimVector 类型的结果
SymDimVector infer_size_symdimvector(SymIntArrayRef a, SymIntArrayRef b) {
  return infer_size_impl<SymDimVector, SymIntArrayRef>(a, b);
}

// 模板函数，实现推断扩展几何属性
template<typename Container>
C10_ALWAYS_INLINE InferExpandGeometryResult<Container> inferExpandGeometryImpl(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes) {
  int64_t ndim = static_cast<int64_t>(sizes.size());
  int64_t tensor_dim = static_cast<int64_t>(tensor_sizes.size());

  // 若张量尺寸为0，则直接返回给定尺寸和维度数
  if (tensor_dim == 0) {
    return InferExpandGeometryResult<Container>(sizes, ndim);
  }

  // 创建用于存储结果的对象
  InferExpandGeometryResult<Container> result(ndim);
  auto& expandedSizes = result.sizes;
  auto& expandedStrides = result.strides;

  // 为张量创建新的几何结构
  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dim = tensor_dim - 1 - offset;
    // 获取当前维度的大小和步长，若索引无效则默认为1或由扩展后的尺寸和步长决定
    int64_t size = (dim >= 0) ? tensor_sizes[dim] : 1;
    int64_t stride = (dim >= 0) ? tensor_strides[dim]
                                : expandedSizes[i + 1] * expandedStrides[i + 1];
    int64_t targetSize = sizes[i];
    # 如果目标大小为 -1，则进行特殊处理
    if (targetSize == -1) {
      # 使用 TORCH_CHECK 确保维度 dim 大于等于 0
      TORCH_CHECK(
          dim >= 0,
          "The expanded size of the tensor (",
          targetSize,
          ") isn't allowed in a leading, non-existing dimension ",
          i);
      # 将 targetSize 设为当前 size
      targetSize = size;
    }
    # 如果当前 size 不等于目标大小 targetSize
    if (size != targetSize) {
      # 使用 TORCH_CHECK 确保 size 等于 1，目标大小 targetSize 与当前 size 在维度 i 上匹配
      TORCH_CHECK(
          size == 1,
          "The expanded size of the tensor (",
          targetSize,
          ") must match the existing size (",
          size,
          ") at non-singleton dimension ",
          i,
          ".  Target sizes: ",
          sizes,
          ".  Tensor sizes: ",
          tensor_sizes);
      # 将 size 更新为 targetSize，stride 更新为 0
      size = targetSize;
      stride = 0;
    }
    # 将当前维度的 size 和 stride 更新到 expandedSizes 和 expandedStrides 中
    expandedSizes[i] = size;
    expandedStrides[i] = stride;
  }
  # 返回处理后的结果
  return result;
}

// 此函数使用输入的张量尺寸和步长计算密集且非重叠的步长，保持与输入 `tensor_strides` 相同的布局排列顺序。
// 注意：
// 1. 此函数预期输入的 `tensor_strides` 和 `tensor_sizes` 是非密集或重叠的。
//    如果输入已经是密集且非重叠的，输出的步长将与 `tensor_strides` 相同。
//    然而，此函数不会检查输入是否密集或重叠，因此即使输入已经是密集且非重叠的，整个函数依然会执行，这可能导致速度变慢。
//
//    在调用此函数之前，请尽可能验证输入是否是非密集或重叠的。
//    如果输入来自张量，可以通过 `is_non_overlapping_and_dense()` 进行检查。
//
// 2. 此函数使用的步长传播规则与 TensorIterator 完全相同。更多细节请参考 https://github.com/pytorch/pytorch/pull/42922

std::vector<int64_t> infer_dense_strides(IntArrayRef tensor_sizes, IntArrayRef tensor_strides) {

  // 检查输入的尺寸和步长是否具有相同的大小
  TORCH_CHECK(tensor_sizes.size() == tensor_strides.size(),
    "Input sizes and strides should have same size but got ", tensor_sizes.size(), " and ", tensor_strides.size());

  // 获取张量的维度数
  size_t ndim = tensor_sizes.size();
  if (ndim == 0) {
    return {};  // 如果维度数为0，返回空向量
  }
  if (ndim == 1) {
    return {1};  // 如果维度数为1，返回包含1的向量
  }

  // 创建一个用于排列的索引向量，初始为 n-1, n-2, ..., 1, 0
  std::vector<int64_t> perm(ndim);
  std::iota(perm.rbegin(), perm.rend(), 0);

  // 下面的排序算法与 TensorIterator 完全相同，以确保步长传播的一致性

  // 如果 dim0 应该在 dim1 之前返回 -1
  // 如果 dim0 应该在 dim1 之后返回 1
  // 如果比较模糊返回 0
  auto should_swap = [&](size_t dim0, size_t dim1) {
    int64_t stride0 = tensor_strides[dim0];
    int64_t stride1 = tensor_strides[dim1];

    // 如果任一步长为0，则将其视为模糊比较，保持与 TensorIterator 相同的行为
    if (stride0 == 0 || stride1 == 0) {
      return 0;
    }
    if (stride0 < stride1) {
      return -1;
    }
    if (stride0 > stride1) {
      return 1;
    }
    // 对于相等的步长，尺寸较小的维度排在前面
  // 比较给定维度的张量尺寸，如果第一个维度大于第二个，返回1，否则返回0
  if (tensor_sizes[dim0] > tensor_sizes[dim1]) {
    return 1;
  }
  return 0;
};

// 根据输入张量的步幅和形状稳定地对`perm`中的索引进行插入排序，所有步幅为0的维度保持不动。
// 这与TensorIterator的行为相同。
// 例如，对于尺寸/步幅为(6, 5, 4, 3, 2)/(6, 0, 120, 0, 1)的张量，初始的`perm`是(4, 3, 2, 1, 0)，
// 排序后的`perm`将是(4, 3, 0, 1, 2)
for (const auto i : c10::irange(1, ndim)) {
  auto dim1 = i;
  for (const auto j : c10::irange(1, i + 1)) {
    auto dim0 = i - j;
    // 比较perm[dim0]和perm[dim1]，根据should_swap函数的返回值决定是否交换它们的位置
    int comparison = should_swap(perm[dim0], perm[dim1]);
    if (comparison > 0) {
      std::swap(perm[dim0], perm[dim1]);
      dim1 = dim0;
    }
    else if (comparison < 0) {
      break;
    }
  }
}

// 计算输出步幅，以保持输入张量的内存布局
std::vector<int64_t> out_strides(ndim);
int64_t curr_stride = 1;
for (const auto i : c10::irange(ndim)) {
  int64_t idx = perm[i];
  out_strides[idx] = curr_stride;
  // 注意：对于尺寸为0的情况，我们简单地将其视为1，这在这里并不重要，因为元素的总数为0。
  if (tensor_sizes[idx] > 1) {
    curr_stride *= tensor_sizes[idx];
  }
}
return out_strides;
}

} // namespace at
```