# `.\pytorch\aten\src\ATen\native\cpu\FunctionOfAMatrixUtilsKernel.cpp`

```py
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/FunctionOfAMatrixUtils.h>

#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <c10/util/irange.h>

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

namespace at::native {

// 匿名命名空间，定义了一个私有函数
namespace {

// 定义了一个用于在 CPU 上计算线性组合的核心函数
void _compute_linear_combination_cpu_kernel(
  TensorIterator& iter,          // 张量迭代器，用于访问输入和输出张量
  int64_t in_stride,             // 输入张量的步长
  int64_t coeff_stride,          // 系数张量的步长
  int64_t num_summations         // 要执行的求和次数
) {
  // 根据迭代器的数据类型分发执行不同类型的计算
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
    iter.dtype(),
    "_compute_linear_combination_cpu", [&] {
      // 定义内部循环函数，处理数据数组和步长信息
      auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        auto* RESTRICT out_ptr = data[0];      // 输出张量的指针
        auto* RESTRICT in_ptr = data[1];       // 输入张量的指针
        auto* RESTRICT coeff_ptr = data[2];    // 系数张量的指针

        // 对于每个元素进行迭代
        for (const auto elem C10_UNUSED : c10::irange(n)) {
          auto* RESTRICT out_data = reinterpret_cast<scalar_t*>(out_ptr);      // 输出数据的类型转换
          auto* RESTRICT in_data = reinterpret_cast<scalar_t*>(in_ptr);        // 输入数据的类型转换
          using primitive_t = typename scalar_value_type<scalar_t>::type;
          auto* RESTRICT coeff_data = reinterpret_cast<primitive_t*>(coeff_ptr); // 系数数据的类型转换

          // 执行线性组合的求和操作
          for (const auto i : c10::irange(num_summations)) {
            *out_data += in_data[i * in_stride] * coeff_data[i * coeff_stride];
          }

          out_ptr += strides[0];     // 更新输出指针位置
          in_ptr += strides[1];      // 更新输入指针位置
          coeff_ptr += strides[2];   // 更新系数指针位置
        }
      };
      iter.for_each(loop);  // 使用迭代器执行定义的循环操作
  });
}

}

// 注册 CPU 版本的线性组合计算函数
REGISTER_DISPATCH(_compute_linear_combination_stub, &_compute_linear_combination_cpu_kernel);

} // namespace at::native
```