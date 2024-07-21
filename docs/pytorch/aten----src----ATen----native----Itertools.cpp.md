# `.\pytorch\aten\src\ATen\native\Itertools.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/cartesian_prod_native.h>
#include <ATen/ops/combinations_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/meshgrid.h>
#include <ATen/ops/stack.h>
#endif

#include <vector>

namespace {

using namespace at;

// 函数定义：生成一个三角形状掩码张量
// 参数 n: 生成张量的维度大小
// 参数 dims: 张量的维度数
// 参数 diagonal: 是否包括对角线上的元素
// 参数 opt: 张量的选项
Tensor _triu_mask(int64_t n, int64_t dims, bool diagonal, TensorOptions opt) {
  // 生成一个从 0 到 n-1 的整数张量
  Tensor range = at::arange(n, opt.dtype(kLong));
  // 使用 meshgrid 函数生成一组网格索引张量
  std::vector<Tensor> index_grids = at::meshgrid(std::vector<Tensor>(dims, range), "ij");
  // 生成一个全为 true 的张量，形状与索引张量相同
  Tensor mask = at::full(index_grids[0].sizes(), true, opt.dtype(kBool));
  // 根据 diagonal 参数决定生成上三角形状的掩码张量
  if(diagonal) {
    for(int64_t i = 0; i < dims - 1; i++) {
      mask *= index_grids[i] <= index_grids[i+1];
    }
  } else {
    for(int64_t i = 0; i < dims - 1; i++) {
      mask *= index_grids[i] < index_grids[i+1];
    }
  }
  // 返回生成的掩码张量
  return mask;
}

}  // namespace

namespace at::native {

// 函数定义：计算输入张量列表的笛卡尔积
// 参数 tensors: 输入的张量列表
Tensor cartesian_prod(TensorList tensors) {
  // 检查每个张量是否为一维向量
  for(const Tensor &t : tensors) {
    TORCH_CHECK(t.dim() == 1, "Expect a 1D vector, but got shape ", t.sizes());
  }
  // 如果输入只有一个张量，直接返回该张量
  if (tensors.size() == 1) {
    return tensors[0];
  }
  // 使用 meshgrid 函数生成张量列表的网格
  std::vector<Tensor> grids = at::meshgrid(tensors, "ij");
  // 将生成的网格张量展平
  for(Tensor &t : grids) {
    t = t.flatten();
  }
  // 将展平后的张量列表在新的维度上堆叠起来
  return at::stack(grids, 1);
}

// 函数定义：生成输入张量 self 的组合数
// 参数 self: 输入的一维向量张量
// 参数 r: 组合数中的元素个数
// 参数 with_replacement: 是否允许元素重复
Tensor combinations(const Tensor& self, int64_t r, bool with_replacement) {
  // 检查输入张量是否为一维向量
  TORCH_CHECK(self.dim() == 1, "Expect a 1D vector, but got shape ", self.sizes());
  // 检查 r 是否为非负数
  TORCH_CHECK(r >= 0, "Expect a non-negative number, but got ", r);
  // 如果 r 为 0，返回一个空张量
  if (r == 0) {
    return at::empty({0}, self.options());
  }
  // 获取输入张量中的元素个数
  int64_t num_elements = self.numel();
  // 使用 meshgrid 函数生成 self 的 r 维网格
  std::vector<Tensor> grids = at::meshgrid(std::vector<Tensor>(r, self), "ij");
  // 生成组合数的掩码张量
  Tensor mask = _triu_mask(num_elements, r, with_replacement, self.options());
  // 根据掩码张量选择有效的组合数
  for(Tensor &t : grids) {
    t = t.masked_select(mask);
  }
  // 将有效的组合数张量列表在新的维度上堆叠起来
  return at::stack(grids, 1);
}

}  // namespace at::native
```