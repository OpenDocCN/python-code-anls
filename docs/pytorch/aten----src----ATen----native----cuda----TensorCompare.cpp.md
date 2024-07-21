# `.\pytorch\aten\src\ATen\native\cuda\TensorCompare.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorCompare.h>

namespace at::native {

namespace {

// 在匿名命名空间中定义了一个函数，用于在 GPU 上执行 isin 操作的默认内核实现。
// 这个实现为了简单起见，实际上会生成元素和测试元素的交叉乘积，因此内存效率不高，但在 CUDA 上速度很快。
void isin_default_kernel_gpu(
    const Tensor& elements, const Tensor& test_elements, bool invert, const Tensor& out) {
  // 创建一个形状为 elements 维度加一的向量，每个维度都是 1，最后一个维度为 -1。
  std::vector<int64_t> bc_shape(elements.dim(), 1);
  bc_shape.push_back(-1);
  // 根据 invert 的值选择是执行不等于操作并全部为真（invert 为 true）还是等于操作并有任意为真（invert 为 false）。
  out.copy_(invert ? elements.unsqueeze(-1).ne(test_elements.view(bc_shape)).all(-1)
            : elements.unsqueeze(-1).eq(test_elements.view(bc_shape)).any(-1));
}

} // anonymous namespace

// 注册 CUDA 分发器，将 isin_default_stub 映射到 isin_default_kernel_gpu 函数。
REGISTER_CUDA_DISPATCH(isin_default_stub, &isin_default_kernel_gpu);

} // namespace at::native
```