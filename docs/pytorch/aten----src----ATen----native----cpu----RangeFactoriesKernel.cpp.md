# `.\pytorch\aten\src\ATen\native\cpu\RangeFactoriesKernel.cpp`

```py
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/RangeFactories.h>
#include <cmath>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>

#include <ATen/AccumulateType.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/Loops.h>

#include <c10/core/Scalar.h>

namespace at::native {
namespace {

using namespace vec;

// 定义静态函数 arange_kernel，接收一个迭代器 iter 和三个标量参数
static void arange_kernel(TensorIterator& iter, const Scalar& scalar_start, const Scalar& scalar_steps, const Scalar& scalar_step) {
  // 使用模板函数，处理所有数据类型，生成函数名为 arange_cpu
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "arange_cpu", [&]() {
    // 定义累积类型 accscalar_t，将标量参数转换为这种类型
    using accscalar_t = at::acc_type<scalar_t, false>;
    auto start = scalar_start.to<accscalar_t>();  // 将 scalar_start 转换为 accscalar_t 类型
    auto steps = scalar_steps.to<accscalar_t>();  // 将 scalar_steps 转换为 accscalar_t 类型
    auto step = scalar_step.to<accscalar_t>();    // 将 scalar_step 转换为 accscalar_t 类型
    // 使用并行计算，按照 GRAIN_SIZE 分割，迭代从 0 到 steps
    at::parallel_for(0, steps, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
      int64_t idx(p_begin);  // 初始化 idx 为 p_begin
      TensorIterator it(iter);  // 创建迭代器 it
      // 使用 CPU 串行内核函数，处理迭代器 it
      cpu_serial_kernel_vec(
          it,
          [start, step, &idx]() -> scalar_t {
            return start + step * (idx++);  // 返回 start + step * idx，并递增 idx
          },
          [start, step, &idx]() -> Vectorized<scalar_t> {
            Vectorized<scalar_t> res;  // 创建向量化类型 res
            res = Vectorized<scalar_t>::arange(start + step * idx, step);  // 使用向量化方式生成范围
            idx += Vectorized<scalar_t>::size();  // 更新 idx
            return res;  // 返回向量化结果
          }, {p_begin, p_end});  // 迭代范围为 p_begin 到 p_end
    });
  });
}

// 定义静态函数 linspace_kernel，接收一个迭代器 iter 和三个标量参数
static void linspace_kernel(TensorIterator& iter, const Scalar& scalar_start, const Scalar& scalar_end, int64_t steps) {
  // 使用模板函数，处理所有数据类型和复数，生成函数名为 linspace_cpu
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, iter.dtype(), "linspace_cpu", [&]() {
    // step_t 应为 double 类型，用于所有整数类型
    using step_t = std::conditional_t<std::is_integral<scalar_t>::value, double, scalar_t>;
    const scalar_t start = scalar_start.to<scalar_t>();  // 将 scalar_start 转换为 scalar_t 类型
    const scalar_t end = scalar_end.to<scalar_t>();  // 将 scalar_end 转换为 scalar_t 类型
    // 将 `end` 和 `start` 强制转换为 step_t 类型，因为范围可能比 scalar_t 类型大
    const step_t step = (static_cast<step_t>(end) - static_cast<step_t>(start)) / (steps - 1);
    int64_t halfway = steps / 2;  // 计算步骤的一半
    // 使用并行计算，按照 GRAIN_SIZE 分割，迭代从 0 到 steps
    at::parallel_for(0, steps, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
      int64_t idx(p_begin);  // 初始化 idx 为 p_begin
      TensorIterator it(iter);  // 创建迭代器 it
      // 使用 CPU 串行内核函数，处理迭代器 it
      cpu_serial_kernel(
          it,
          [start, end, step, halfway, steps, &idx]() -> scalar_t {
            if (idx < halfway) {
              return start + step * (idx++);  // 如果 idx 小于 halfway，则返回 start + step * idx，并递增 idx
            } else {
              return end - step * (steps - (idx++) - 1);  // 否则返回 end - step * (steps - idx - 1)，并递增 idx
            }
          }, {p_begin, p_end});  // 迭代范围为 p_begin 到 p_end
    });
  });
}

} // anonymous namespace

// 注册 arange_stub 的分派函数为 arange_kernel
REGISTER_DISPATCH(arange_stub, &arange_kernel);
// 注册 linspace_stub 的分派函数为 linspace_kernel
REGISTER_DISPATCH(linspace_stub, &linspace_kernel);

} // namespace at::native
```