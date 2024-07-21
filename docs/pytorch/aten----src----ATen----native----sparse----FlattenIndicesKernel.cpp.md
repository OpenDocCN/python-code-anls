# `.\pytorch\aten\src\ATen\native\sparse\FlattenIndicesKernel.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，用于仅仅在方法操作符上执行 Torch 断言

#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/FlattenIndicesCommon.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/AccumulateType.h>
// 引入相关头文件，用于稀疏张量操作、索引展平、CPU 循环、张量迭代器和累积类型

namespace at::native {

namespace {

template <typename func_t>
struct CPUKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    cpu_kernel(iter, f);
    // 调用 CPU 内核函数，传入迭代器和函数对象 f
  }
};

// 在匿名命名空间内定义的 CPUKernelLauncher 结构模板，用于启动 CPU 内核计算

Tensor flatten_indices_cpu_kernel(const Tensor& indices, IntArrayRef size) {
  return _flatten_indices<CPUKernelLauncher>(indices, size);
  // 调用 _flatten_indices 函数，传入 CPUKernelLauncher 结构模板和索引张量及尺寸，返回展平后的索引张量
}

}

// 匿名命名空间结束

REGISTER_ARCH_DISPATCH(flatten_indices_stub, DEFAULT, &flatten_indices_cpu_kernel);
REGISTER_AVX512_DISPATCH(flatten_indices_stub, &flatten_indices_cpu_kernel);
REGISTER_AVX2_DISPATCH(flatten_indices_stub, &flatten_indices_cpu_kernel);
REGISTER_VSX_DISPATCH(flatten_indices_stub, &flatten_indices_cpu_kernel);
REGISTER_ZVECTOR_DISPATCH(flatten_indices_stub, &flatten_indices_cpu_kernel);
// 注册不同架构下的展平索引函数的分发器

} // namespace at::native

// at::native 命名空间结束
```