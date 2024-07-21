# `.\pytorch\aten\src\ATen\native\cpu\ComplexKernel.cpp`

```
// 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS，用于控制是否仅支持方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含头文件，用于 ATen 的调度和张量工厂的实现
#include <ATen/Dispatch.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

// 进入 ATen 的 native 命名空间
namespace at::native {

// 匿名命名空间，用于定义私有函数或局部变量
namespace {

// 定义一个处理复数计算的核函数，接受一个张量迭代器作为参数
void complex_kernel(TensorIterator& iter) {
  // 模板函数，根据迭代器的数据类型和半精度类型调度相应的操作
  AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.input_dtype(), "complex_cpu", [&]() {
    // 调用 cpu_kernel 函数，使用 lambda 表达式对每对输入元素执行复数构造
    cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
      return c10::complex<scalar_t>(a, b);
    });
  });
}

// 定义一个处理极坐标转换的核函数，接受一个张量迭代器作为参数
void polar_kernel(TensorIterator& iter) {
  // 模板函数，根据迭代器的数据类型调度相应的操作
  AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "polar_cpu", [&]() {
    // 调用 cpu_kernel 函数，使用 lambda 表达式对每对输入元素执行极坐标转换
    cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
      return c10::complex<scalar_t>(a * std::cos(b), a * std::sin(b));
    });
  });
}

} // anonymous namespace

// 注册复数核函数的调度器
REGISTER_DISPATCH(complex_stub, &complex_kernel);

// 同时注册 AVX512 指令集支持的极坐标转换核函数的调度器
ALSO_REGISTER_AVX512_DISPATCH(polar_stub, &polar_kernel);

} // namespace at::native
```