# `.\pytorch\aten\src\ATen\native\cpu\PowKernel.cpp`

```py
// 定义编译时取消操作符的宏，这可能是为了避免在使用Torch时使用运算符的检查
#define TORCH_ASSERT_NO_OPERATORS

// 包含标准数学库头文件
#include <cmath>

// 包含 ATen 库的分发和并行处理头文件
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>

// 包含 ATen 库的向量化计算头文件
#include <ATen/cpu/vec/vec.h>

// 包含 ATen 库的张量迭代器头文件
#include <ATen/native/TensorIterator.h>

// 包含 ATen 库的指数计算头文件
#include <ATen/native/Pow.h>

// 包含 ATen 库的一元操作头文件
#include <ATen/native/UnaryOps.h>

// 包含 ATen 库的 CPU 循环头文件
#include <ATen/native/cpu/Loops.h>

// 包含 C10 库的标量类型头文件
#include <c10/core/Scalar.h>

// ATen 库的命名空间
namespace at::native {

// 内联的 CPU 能力命名空间
inline namespace CPU_CAPABILITY {

// 幂计算的张量与张量的核心函数
static void pow_tensor_tensor_kernel(TensorIteratorBase& iter) {
  // 获取迭代器中的公共数据类型
  const auto dtype = iter.common_dtype();

  // 如果数据类型为浮点型或复数类型
  if (isFloatingType(dtype) || isComplexType(dtype)) {
    // 使用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2 宏分发浮点型和复数类型
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, dtype, "pow", [&]() {
      // 使用 Vectorized 类处理向量化操作
      using Vec = Vectorized<scalar_t>;
      // 使用 CPU 内核进行向量化计算
      cpu_kernel_vec(iter,
        // 标量版本的计算：计算 base 的 exp 次方
        [=](scalar_t base, scalar_t exp) -> scalar_t {
          return std::pow(base, exp);
        },
        // 向量版本的计算：计算向量 base 的向量 exp 次方
        [&](Vec base, Vec exp) -> Vec {
          return base.pow(exp);
        }
      );
    });
  } else {
    // 如果数据类型为整数类型
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "pow", [&]() {
      // 使用 CPU 内核进行整数类型的计算
      cpu_kernel(iter,
        // 标量版本的计算：计算 base 的 exp 次方
        [=](scalar_t base, scalar_t exp) -> scalar_t {
          return native::powi(base, exp);
        }
      );
    });
  }
}

// 对于浮点型、双精度和复数类型，核心代码的源码是相似的，只是微小区别——即使输出数据类型为浮点型，也可以使用双精度的指数。
// 但是，复数类型的计算不允许混合使用标准和双精度，因为 std::pow 接受 complex64 或 complex128 输入，但不接受混合使用。
// 为了为浮点型、双精度和复数类型提供共同路径，使用模板参数 cast_scalar_t 来解决上述区别。这种方法也允许 BFloat16 使用这种共同路径。
// 目前，Half 不能使用这种共同路径，因为 AVX2 不支持其 sqrt 和 rsqrt。
template <typename scalar_t, typename cast_scalar_t, typename exp_scalar_t>
void pow_tensor_scalar_optimized_kernel(TensorIteratorBase& iter, const exp_scalar_t exp) {
  // 使用 Vectorized 类处理向量化操作
  using Vec = Vectorized<scalar_t>;
  // 处理 .5（sqrt）、-.5（rsqrt）和 -1（reciprocal）特殊情况在 pow_tensor_scalar_kernel 中进行处理
  if (exp == 2.0) {
    // 计算 base 的平方
    cpu_kernel_vec(iter,
        [](scalar_t base) -> scalar_t {
          return base * base;
        },
        [](Vec base) -> Vec { return base * base; }
    );
  } else if (exp == 3.0) {
    // 计算 base 的立方
    cpu_kernel_vec(iter,
        [](scalar_t base) -> scalar_t {
          return base * base * base;
        },
        [](Vec base) -> Vec { return base * base * base; }
    );
  } else if (exp == -2.0) {
    // 计算 base 的平方的倒数
    cpu_kernel_vec(iter,
        [](scalar_t base) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
          return static_cast<cast_scalar_t>(1.0) / (base * base);
        },
        [](Vec base) -> Vec { return (base * base).reciprocal(); }
    );
  } else {
    # 调用 cpu_kernel_vec 函数，传入参数 iter 和两个 lambda 函数作为参数
    cpu_kernel_vec(iter,
        # 第一个 lambda 函数：接收标量 base，返回 base 的 exp 次方结果
        [=](scalar_t base) -> scalar_t {
          return std::pow(base, static_cast<cast_scalar_t>(exp));
        },
        # 第二个 lambda 函数：接收 Vec 类型的 base，返回 base 的 exp 次方结果
        [=](Vec base) -> Vec {
          return base.pow(static_cast<cast_scalar_t>(exp));
        }
    );
  }
}

// 定义静态函数，实现张量与标量的幂运算
static void pow_tensor_scalar_kernel(
    TensorIteratorBase& iter,  // 张量迭代器对象的引用
    const Scalar& exp_scalar) {  // 幂指数标量的常量引用
  // 防止多次调用 iter.common_dtype()
  const auto dtype = iter.common_dtype();  // 获取张量迭代器中的公共数据类型

  // 如果数据类型为浮点型、双精度型、BFloat16 或复数类型，特化处理平方根、倒数平方根和倒数的情况
  if (dtype == ScalarType::Float || dtype == ScalarType::Double ||
      dtype == kBFloat16 || isComplexType(dtype)) {
    if (exp_scalar.equal(.5)) {  // 如果指数为0.5，调用平方根的核函数
      return sqrt_kernel(iter);
    } else if (exp_scalar.equal(-0.5)) {  // 如果指数为-0.5，调用倒数平方根的核函数
      return rsqrt_kernel(iter);
    } else if (exp_scalar.equal(-1.0)) {  // 如果指数为-1.0，调用倒数的核函数
      return reciprocal_kernel(iter);
    }
  }

  // 如果数据类型为浮点型或双精度型，使用优化后的幂运算核函数
  if (dtype == ScalarType::Float || dtype == ScalarType::Double) {
    AT_DISPATCH_FLOATING_TYPES(dtype, "pow", [&]() {
      pow_tensor_scalar_optimized_kernel<scalar_t, double>(
          iter, exp_scalar.to<double>());
    });
  } else if (isComplexType(dtype)) {  // 如果数据类型为复数类型，使用复数类型的优化幂运算核函数
    AT_DISPATCH_COMPLEX_TYPES(dtype, "pow", [&]() {
      pow_tensor_scalar_optimized_kernel<scalar_t, scalar_t>(
          iter, exp_scalar.to<c10::complex<double>>());
    });
  } else if (dtype == ScalarType::Half) {  // 如果数据类型为半精度浮点型
    [&]() {
      using scalar_t =
          decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);  // 定义半精度浮点数类型
      const auto exp = exp_scalar.to<scalar_t>();  // 将指数转换为半精度浮点数类型
      using Vec = Vectorized<scalar_t>;  // 使用矢量化操作进行计算
      cpu_kernel_vec(iter,
          [=](scalar_t base) -> scalar_t {  // 定义 CPU 核函数，计算幂运算
            return std::pow(base, exp);
          },
          [=](Vec base) -> Vec { return base.pow(exp); }  // 使用矢量化计算幂运算
      );
    }();
  } else if (dtype == ScalarType::BFloat16) {  // 如果数据类型为BFloat16
      AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, dtype, "pow", [&]() {
        pow_tensor_scalar_optimized_kernel<scalar_t, scalar_t>(
            iter, exp_scalar.to<scalar_t>());
      });
  } else {  // 其他整数类型的数据类型处理
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "pow", [&]() {
      const scalar_t exp = exp_scalar.to<scalar_t>();  // 将指数转换为整数类型
      cpu_kernel(iter, [=](scalar_t base) -> scalar_t {  // 定义 CPU 核函数，计算整数幂运算
        return native::powi(base, exp);
      });
    });
  }
}

} // 匿名命名空间结束

// 注册 AVX512 指令集的张量-张量幂运算的分发函数
ALSO_REGISTER_AVX512_DISPATCH(pow_tensor_tensor_stub, &CPU_CAPABILITY::pow_tensor_tensor_kernel);
// 注册 AVX512 指令集的张量-标量幂运算的分发函数
ALSO_REGISTER_AVX512_DISPATCH(pow_tensor_scalar_stub, &CPU_CAPABILITY::pow_tensor_scalar_kernel);

} // namespace at::native
```