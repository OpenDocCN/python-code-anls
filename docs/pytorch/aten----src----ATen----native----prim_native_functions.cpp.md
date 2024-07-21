# `.\pytorch\aten\src\ATen\native\prim_native_functions.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/is_nonzero_native.h>
#include <ATen/ops/_foobar_native.h>
#include <ATen/ops/_test_functorch_fallback_native.h>
#endif

// 定义命名空间 at::native
namespace at::native {

// 检查张量是否非零
bool is_nonzero(const Tensor& self) {
  // 获取张量中元素的数量
  auto n = self.numel();
  // 如果张量为空，则抛出异常
  TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
  // 如果张量元素超过一个，则抛出异常
  TORCH_CHECK(
      n < 2, "Boolean value of Tensor with more than one value is ambiguous");

  // 获取张量中的单个标量值
  Scalar localScalar = self.item();
  // 如果标量值为浮点数类型
  if (localScalar.isFloatingPoint()) {
    // 返回标量是否不等于零
    return localScalar.to<double>() != 0;
  } else if (localScalar.isComplex()) {  // 如果标量值为复数类型
    // 返回标量是否不等于复数 0
    return localScalar.to<c10::complex<double>>() !=
        c10::complex<double>(0.0, 0.0);
  } else if (localScalar.isIntegral(false)) {  // 如果标量值为整数类型
    // 返回标量是否不等于整数 0
    return localScalar.to<int64_t>() != 0;
  } else if (localScalar.isBoolean()) {  // 如果标量值为布尔类型
    // 返回布尔值
    return localScalar.to<bool>();
  }
  // 如果标量类型不符合预期，则抛出内部断言异常
  TORCH_INTERNAL_ASSERT(false, "Expected non-Tensor backend scalar");
}

// 辅助函数，用于测试 TestPythonDispatch.test_kwarg_only_and_positional_default
// 在 test/test_python_dispatch.py 中使用
Tensor foobar(const Tensor& self, bool arg1, bool arg2, bool arg3) {
  // 返回传入的张量本身
  return self;
}

// 辅助函数，用于测试 functorch 回退警告
Tensor _test_functorch_fallback(const Tensor& self, const Tensor& other) {
  // 返回输入张量的克隆
  return self.clone();
}

} // namespace at::native
```