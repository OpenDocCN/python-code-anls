# `.\pytorch\aten\src\ATen\ScalarOps.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ScalarOps.h>

namespace at {
namespace {

// 模板函数：用指定的标量值填充给定的张量（原地操作）
template <typename scalar_t>
inline void fill_inplace(Tensor& self, const Scalar& value_scalar) {
  // 将标量值转换为指定类型的标量
  auto value = value_scalar.to<scalar_t>();
  // 获取张量的数据指针，并转换为指定类型的指针
  scalar_t* dptr = static_cast<scalar_t*>(self.data_ptr());
  // 将标量值赋给张量的数据指针所指向的位置
  *dptr = value;
}

} // namespace

namespace detail {

// 函数：使用标量值填充给定的张量
Tensor& scalar_fill(Tensor& self, const Scalar& value) {
  // 使用分派机制根据张量的类型执行相应操作，具体操作由 AT_WRAP 宏封装
  AT_DISPATCH_V2(
      self.scalar_type(), "fill_out", AT_WRAP([&]() {
        // 调用 fill_inplace 函数，根据张量的标量类型执行填充操作
        fill_inplace<scalar_t>(self, value);
      }), kComplexHalf, kHalf, kBool, kBFloat16, AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  // 返回填充后的张量
  return self;
}

// 函数：创建一个标量张量，使用指定的标量值、数据类型和设备（静态版本）
Tensor scalar_tensor_static(const Scalar& s, std::optional<ScalarType> dtype_opt, std::optional<Device> device_opt) {
  // 关闭追踪器的分派模式
  at::tracer::impl::NoTracerDispatchMode tracer_guard;
  // 进入自动分派模式下的计算
  at::AutoDispatchBelowAutograd mode;
  // 创建一个空的 CPU 张量，根据指定的参数
  Tensor result = at::detail::empty_cpu(
      {}, dtype_opt, c10::nullopt, device_opt, c10::nullopt, c10::nullopt);
  // 使用标量值填充结果张量
  scalar_fill(result, s);
  // 返回填充后的结果张量
  return result;
}

} // namespace detail
} // namespace at
```