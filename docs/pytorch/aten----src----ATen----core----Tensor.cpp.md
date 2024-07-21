# `.\pytorch\aten\src\ATen\core\Tensor.cpp`

```
// 包含 ATen 库中的头文件，用于张量操作和格式化
#include <ATen/core/Tensor.h>
#include <ATen/core/Formatting.h>
#include <ATen/core/VariableHooksInterface.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/FunctionalTensorWrapper.h>

// 根据编译时的宏定义选择不同的操作符头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/MethodOperators.h>
#else
#include <ATen/ops/contiguous_ops.h>
#include <ATen/ops/fill_ops.h>
#include <ATen/ops/to_ops.h>
#include <ATen/ops/zero_ops.h>
#endif

// 标准输入输出流库
#include <iostream>

// ATen 命名空间
namespace at {

// 获取张量的基类对象
const TensorBase& get_tensor_base(const Tensor &t) {
  return t;
}

// 实现张量基类的 __dispatch_contiguous 方法
TensorBase TensorBase::__dispatch_contiguous(c10::MemoryFormat memory_format) const {
  // 使用 OptionalTensorRef 封装当前对象
  OptionalTensorRef self(*this);
  // 调用 ATen 的 contiguous 操作
  return at::_ops::contiguous::call(*self, memory_format);
}

// 实现张量基类的 fill_ 方法
const TensorBase& TensorBase::fill_(const c10::Scalar &fill_value) const {
  // 复制当前对象
  Tensor self(*this);
  // 调用 ATen 的 fill_ 操作
  at::_ops::fill__Scalar::call(self, fill_value);
  // 返回当前对象的引用
  return *this;
}

// 实现张量基类的 zero_ 方法
const TensorBase& TensorBase::zero_() const {
  // 复制当前对象
  Tensor self(*this);
  // 调用 ATen 的 zero_ 操作
  at::_ops::zero_::call(self);
  // 返回当前对象的引用
  return *this;
}

// 实现张量基类的 to 方法
TensorBase TensorBase::to(
    at::TensorOptions options,
    bool non_blocking,
    bool copy,
    std::optional<at::MemoryFormat> memory_format) const {
  // 复制当前对象
  Tensor self(*this);
  // 调用 ATen 的 to 操作
  return at::_ops::to_dtype_layout::call(
      self, optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(), options.device_opt(),
      options.pinned_memory_opt(), non_blocking, copy, memory_format);
}

// 强制检查张量对象的不变性
void TensorBase::enforce_invariants() {
  // 如果张量实现指针为空，抛出运行时异常
  if (impl_.get() == nullptr) {
    throw std::runtime_error("TensorImpl with nullptr is not supported");
  }
  // 检查标量类型是否支持
  scalar_type();
  // 如果张量已定义
  if (defined()) {
    // 断言张量类型已初始化
    TORCH_INTERNAL_ASSERT(
        impl_->dtype_initialized(),
        "Partially-initialized tensor not supported by Tensor");
    // 断言稀疏张量未实现
    TORCH_INTERNAL_ASSERT(
        !impl_->is_sparse(),
        "Sparse Tensors are supported by Tensor, but invariant checking isn't implemented.  Please file a bug.");
    // 断言张量存储已初始化
    TORCH_INTERNAL_ASSERT(
        !impl_->has_storage() || impl_->is_meta() || impl_->storage_initialized(),
        "Partially-initialized tensor not supported by Tensor");
  }
}

// 打印张量信息
void TensorBase::print() const {
  // 如果张量已定义
  if (defined()) {
    // 输出张量的字符串表示和大小信息
    std::cerr << "[" << toString() << " " << sizes() << "]" << '\n';
  } else {
    // 输出未定义张量的信息
    std::cerr << "[UndefinedTensor]" << '\n';
  }
}

// 获取张量的字符串表示
std::string TensorBase::toString() const {
  // 初始化基本字符串
  std::string base_str;
  // 如果标量类型未定义
  if (scalar_type() == ScalarType::Undefined) {
    base_str = "UndefinedType";
  } else {
    // 计算分发键并获取对应的字符串表示
    auto dispatchkey = options().computeDispatchKey();
    std::string dispatchkey_str;
    // 根据分发键设置不同的字符串表示
    if (dispatchkey == c10::DispatchKey::PrivateUse1) {
      dispatchkey_str = c10::get_privateuse1_backend();
    } else if (dispatchkey == c10::DispatchKey::AutocastPrivateUse1) {
      dispatchkey_str = "Autocast" + c10::get_privateuse1_backend();
    } else {
      dispatchkey_str = at::toString(dispatchkey);
    }
    // 拼接基本字符串
    base_str = dispatchkey_str + at::toString(scalar_type()) + "Type";
  }
  // 返回最终的字符串表示
  return base_str;
}

} // namespace at
// 获取当前 TensorBase 对象的变量数据，通过调用 impl::GetVariableHooks()->variable_data(*this) 实现
TensorBase TensorBase::variable_data() const {
  return impl::GetVariableHooks()->variable_data(*this);
}

// 获取当前 TensorBase 对象的张量数据，通过调用 impl::GetVariableHooks()->tensor_data(*this) 实现
TensorBase TensorBase::tensor_data() const {
  return impl::GetVariableHooks()->tensor_data(*this);
}

// 检查当前 TensorBase 对象是否为叶子节点，通过调用 impl::GetVariableHooks()->is_leaf(*this) 实现
bool TensorBase::is_leaf() const {
  return impl::GetVariableHooks()->is_leaf(*this);
}

// 获取当前 TensorBase 对象的输出编号，通过调用 impl::GetVariableHooks()->output_nr(*this) 实现
int64_t TensorBase::output_nr() const {
  return impl::GetVariableHooks()->output_nr(*this);
}

// 设置当前 TensorBase 对象的数据为 new_data，通过调用 impl::GetVariableHooks()->set_data(*this, new_data) 实现
void TensorBase::set_data(const TensorBase & new_data) const {
  impl::GetVariableHooks()->set_data(*this, new_data);
}

// 获取当前 TensorBase 对象的数据，通过调用 impl::GetVariableHooks()->data(*this) 实现
TensorBase TensorBase::data() const {
  return impl::GetVariableHooks()->data(*this);
}

// 获取当前 TensorBase 对象的版本号，通过调用 impl::GetVariableHooks()->_version(*this) 实现
int64_t TensorBase::_version() const {
  return impl::GetVariableHooks()->_version(*this);
}

// 保留当前 TensorBase 对象的梯度信息，通过调用 impl::GetVariableHooks()->retain_grad(*this) 实现
void TensorBase::retain_grad() const {
  impl::GetVariableHooks()->retain_grad(*this);
}

// 检查当前 TensorBase 对象是否保留梯度信息，通过调用 impl::GetVariableHooks()->retains_grad(*this) 实现
bool TensorBase::retains_grad() const {
  return impl::GetVariableHooks()->retains_grad(*this);
}

// 执行当前 Tensor 对象的反向传播，通过调用 impl::GetVariableHooks()->_backward(*this, inputs, gradient, keep_graph, create_graph) 实现
void Tensor::_backward(TensorList inputs,
        const std::optional<Tensor>& gradient,
        std::optional<bool> keep_graph,
        bool create_graph) const {
  return impl::GetVariableHooks()->_backward(*this, inputs, gradient, keep_graph, create_graph);
}

// 设置当前 TensorBase 对象是否需要梯度信息，通过调用 impl::GetVariableHooks()->requires_grad_(*this, _requires_grad) 实现
const TensorBase& TensorBase::requires_grad_(bool _requires_grad) const {
  impl::GetVariableHooks()->requires_grad_(*this, _requires_grad);
  return *this;
}

// 检查当前 TensorBase 对象是否为视图，通过调用 impl::GetVariableHooks()->is_view(*this) 实现
bool TensorBase::is_view() const {
  return impl::GetVariableHooks()->is_view(*this);
}

// 获取当前 TensorBase 对象的基础对象，通过调用 impl::GetVariableHooks()->base(*this) 实现
const TensorBase& TensorBase::_base() const {
  return impl::GetVariableHooks()->base(*this);
}

// 获取当前 TensorBase 对象的名称，通过调用 impl::GetVariableHooks()->name(*this) 实现
const std::string& TensorBase::name() const {
  return impl::GetVariableHooks()->name(*this);
}

// 获取当前 TensorBase 对象的梯度函数节点，通过调用 impl::GetVariableHooks()->grad_fn(*this) 实现
const std::shared_ptr<torch::autograd::Node>& TensorBase::grad_fn() const {
  return impl::GetVariableHooks()->grad_fn(*this);
}

// 移除当前 TensorBase 对象的指定位置的钩子，通过调用 impl::GetVariableHooks()->remove_hook(*this, pos) 实现
void TensorBase::remove_hook(unsigned pos) const {
  impl::GetVariableHooks()->remove_hook(*this, pos);
}

// 注册当前 TensorBase 对象的钩子函数，通过调用 impl::GetVariableHooks()->_register_hook(*this, std::move(hook)) 实现
unsigned TensorBase::_register_hook(std::function<TensorBase(const TensorBase&)> hook) const {
  return impl::GetVariableHooks()->_register_hook(*this, std::move(hook));
}
```