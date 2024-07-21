# `.\pytorch\aten\src\ATen\native\cpu\LinearAlgebraKernel.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebra.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/irange.h>

namespace at::native { namespace {

// 定义一个名为 addr_kernel 的函数，接受一个 Tensor 迭代器和两个标量 beta 和 alpha
void addr_kernel(TensorIterator &iter,
                 const Scalar& beta, const Scalar& alpha) {
  // 如果迭代器的数据类型为布尔类型
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    // 将 beta 和 alpha 转换为布尔类型
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    // 当 beta 为 false 时，self 中的值应被忽略，
    // self 中的 NaN 和 Inf 不应传播
    if (beta_val == false) {
      // 使用 CPU 内核处理，lambda 函数返回逻辑与操作的结果
      cpu_kernel(iter,
        [=](scalar_t /*self_val*/,
            scalar_t vec1_val,
            scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
          return alpha_val && vec1_val && vec2_val;
        }
      );
    } else {
      // 使用 CPU 内核处理，lambda 函数返回逻辑或操作的结果
      cpu_kernel(iter,
        [=](scalar_t self_val,
            scalar_t vec1_val,
            scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
          return (beta_val && self_val) || (alpha_val && vec1_val && vec2_val);
        }
      );
    }
    return;
  }

  // 对于除布尔类型外的所有类型，使用模板进行处理
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf,
    iter.dtype(), "addr_cpu", [&]() {
      using Vec = Vectorized<scalar_t>;

      auto beta_val = beta.to<scalar_t>();
      auto alpha_val = alpha.to<scalar_t>();

      auto beta_vec = Vec(beta_val);
      auto alpha_vec = Vec(alpha_val);

      const scalar_t zero_val(0);
      // 当 beta == 0 时，self 中的值应被忽略，
      // self 中的 NaN 和 Inf 不应传播
      if (beta_val == zero_val) {
        // 使用向量化 CPU 内核处理，lambda 函数返回数值乘积的结果
        cpu_kernel_vec(iter,
          [=](scalar_t /*self_val*/,
              scalar_t vec1_val,
              scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
            return alpha_val * vec1_val * vec2_val;
          },
          [=](Vec /*self_vec*/,
              Vec vec1_vec,
              Vec vec2_vec) __ubsan_ignore_undefined__ {
            return alpha_vec * vec1_vec * vec2_vec;
          }
        );
      } else {
        // 使用向量化 CPU 内核处理，lambda 函数返回加权和的结果
        cpu_kernel_vec(iter,
          [=](scalar_t self_val,
              scalar_t vec1_val,
              scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
            return beta_val * self_val + alpha_val * vec1_val * vec2_val;
          },
          [=](Vec self_vec,
              Vec vec1_vec,
              Vec vec2_vec) __ubsan_ignore_undefined__ {
            return beta_vec * self_vec + alpha_vec * vec1_vec * vec2_vec;
          }
        );
      }
    }
  );
}

} // anonymous namespace

// 注册 addr_kernel 函数作为 addr_stub 的调度函数
REGISTER_DISPATCH(addr_stub, &addr_kernel);
} // namespace at::native
```