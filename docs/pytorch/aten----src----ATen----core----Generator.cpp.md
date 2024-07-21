# `.\pytorch\aten\src\ATen\core\Generator.cpp`

```py
// 包含 ATen 库中的头文件，用于生成器（Generator）和张量（Tensor）的操作
#include <ATen/core/Generator.h>
#include <ATen/core/Tensor.h>
// 包含 C10 实用工具库中的异常处理头文件
#include <c10/util/Exception.h>

// 进入 ATen 命名空间
namespace at {

// 设置生成器状态的方法，接受一个张量作为新状态
void Generator::set_state(const at::Tensor& new_state) {
  // 检查新状态张量是否已定义，否则抛出异常
  TORCH_CHECK(new_state.defined(), "Undefined tensor is not allowed");
  // 调用生成器实现的方法设置状态，传入新状态张量的实现对象
  this->impl_->set_state(*new_state.unsafeGetTensorImpl());
}

// 获取生成器当前状态的方法，返回一个张量表示状态
at::Tensor Generator::get_state() const {
  // 调用生成器实现的方法获取当前状态的实现对象，并将其封装为张量
  return at::Tensor::wrap_tensor_impl(this->impl_->get_state());
}

// 安全地设置生成器状态的方法，接受另一个生成器作为新状态
void Generator::graphsafe_set_state(const Generator& new_state) {
  // 调用生成器实现的安全设置状态方法，传入新状态生成器的内部指针
  this->impl_->graphsafe_set_state(new_state.getIntrusivePtr());
}

// 安全地获取生成器状态的方法，返回一个新生成器对象表示状态
Generator Generator::graphsafe_get_state() const {
  // 调用生成器实现的安全获取状态方法，创建一个新的生成器对象
  return Generator(this->impl_->graphsafe_get_state());
}

} // 结束 ATen 命名空间
```