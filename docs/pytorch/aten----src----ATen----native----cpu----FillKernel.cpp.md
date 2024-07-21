# `.\pytorch\aten\src\ATen\native\cpu\FillKernel.cpp`

```
// 定义 TORCH_ASSERT_NO_OPERATORS，禁用 ATen 库的运算符重载
#define TORCH_ASSERT_NO_OPERATORS
// 包含 ATen 库的分发和并行计算相关头文件
#include <ATen/Dispatch_v2.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
// 包含 ATen 库的张量迭代器相关头文件
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

// 包含 ATen 库的填充操作相关头文件
#include <ATen/native/Fill.h>
// 包含 C10 标量相关头文件
#include <c10/core/Scalar.h>

// ATen 库的本地命名空间 at::native
namespace at::native {
// 匿名命名空间，用于局部函数定义
namespace {

// 填充非本机类型的模板函数，scalar_t 是模板参数类型
template <typename scalar_t>
void fill_non_native_type(TensorIterator& iter, const Scalar& value_scalar) {
  // 将标量转换为 scalar_t 类型的值
  auto value = value_scalar.to<scalar_t>().x;
  // 使用 std::make_signed 获取值的带符号类型 H
  using H = typename std::make_signed<decltype(value)>::type;  // Signed type has more acceleration
  // 将 value 的表示保留，static_cast<H>(value) 在实现中定义
  H val = *reinterpret_cast<H*>(std::addressof(value));
  // 调用 CPU 内核向量化函数进行迭代
  cpu_kernel_vec</*check_dynamic_cast=*/false>(
      iter,
      [val]() -> H { return val; },   // lambda 函数返回值为 val
      [val]() { return Vectorized<H>(val); });  // lambda 函数返回 Vectorized<H> 对象
}

// 特化模板函数，处理 c10::complex<at::Half> 类型的填充操作
template <>
void fill_non_native_type<c10::complex<at::Half>>(TensorIterator& iter, const Scalar& value_scalar) {
  // 静态断言，确保 c10::complex<at::Half> 的大小为 32 位
  static_assert(sizeof(c10::complex<at::Half>) == sizeof(int32_t), "Size of ComplexHalf should be 32-bits");
  // 将标量转换为 c10::complex<float> 类型，再转换为 c10::complex<at::Half> 类型
  auto value = c10::complex<at::Half>(value_scalar.to<c10::complex<float>>());
  // 获取值的整数表示
  auto val = *reinterpret_cast<int32_t*>(std::addressof(value));
  // 调用 CPU 内核向量化函数进行迭代
  cpu_kernel_vec</*check_dynamic_cast=*/false>(
      iter,
      [val]() -> int32_t { return val; },   // lambda 函数返回值为 val
      [val]() { return Vectorized<int32_t>(val); });  // lambda 函数返回 Vectorized<int32_t> 对象
}

// 填充操作的核心函数，根据迭代器的数据类型选择不同的填充方法
void fill_kernel(TensorIterator& iter, const Scalar& value_scalar) {
  // 根据迭代器的数据类型选择不同的填充方法
  if (iter.dtype() == ScalarType::Half) {
    fill_non_native_type<at::Half>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::BFloat16) {
    fill_non_native_type<at::BFloat16>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::ComplexHalf) {
    fill_non_native_type<c10::complex<at::Half>>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::Float8_e4m3fn) {
    fill_non_native_type<at::Float8_e4m3fn>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::Float8_e5m2) {
    fill_non_native_type<at::Float8_e5m2>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::Float8_e4m3fnuz) {
    fill_non_native_type<at::Float8_e4m3fnuz>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::Float8_e5m2fnuz) {
    fill_non_native_type<at::Float8_e5m2fnuz>(iter, value_scalar);
  } else {
    // 使用 AT_DISPATCH_V2 宏处理迭代器的数据类型，根据数据类型调用相应的 CPU 内核函数
    AT_DISPATCH_V2(
      iter.dtype(), "fill_cpu", AT_WRAP([&]() {
        scalar_t value = value_scalar.to<scalar_t>();
        // 调用 CPU 内核向量化函数进行迭代
        cpu_kernel_vec(
            iter,
            [=]() -> scalar_t { return value; },   // lambda 函数返回值为 value
            [=]() { return Vectorized<scalar_t>(value); });  // lambda 函数返回 Vectorized<scalar_t> 对象
      }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kBool, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES)
    );
  }
}

} // namespace

// 使用 REGISTER_DISPATCH 宏注册 fill_stub 的调度函数
REGISTER_DISPATCH(fill_stub, &fill_kernel);

} // namespace at::native


这段代码是关于 ATen 库中填充操作的实现，使用了模板函数和向量化技术来处理不同数据类型的张量填充。
```