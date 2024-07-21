# `.\pytorch\aten\src\ATen\functorch\TensorWrapper.cpp`

```py
// 包含所需的头文件
#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <iostream>

// 命名空间 at::functorch 中的函数定义
namespace at::functorch {

// 将 tensor 的详细信息转储到输出流 ss 中
void dumpTensor(std::ostream& ss, const Tensor& tensor) {
  // 尝试获取 tensor 的包装器
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (!wrapped) {
    // 如果 tensor 没有包装器，尝试获取批处理实现
    auto* batched = maybeGetBatchedImpl(tensor);
    if (batched) {
      // 如果存在批处理实现，则输出批处理信息及其值的详细信息
      ss << "Batched[lvl=" << batched->level() << " dim=" << batched->bdim() << ", ";
      dumpTensor(ss, batched->value());
      ss << "]";
      return;
    }
    // 如果没有包装器或批处理实现，直接输出 tensor 的大小信息
    ss << "Tensor" << tensor.sizes();
    return;
  }
  // 如果存在包装器，则输出包装器信息及其值的详细信息
  ss << "Wrapper[";
  if (wrapped->level().has_value()) {
    ss << "lvl=" << wrapped->level().value() << ", ";
  } else {
    ss << "dead, ";
  }
  dumpTensor(ss, wrapped->value());
  ss << "]";
}

// 刷新 TensorWrapper 的元数据
void TensorWrapper::refreshMetadata() {
  // 更新大小、步长和存储偏移量
  set_sizes_and_strides(
      value_.sym_sizes(), value_.sym_strides(), value_.sym_storage_offset());

  // 刷新 numel（元素数）信息
  refresh_numel();
  // 刷新 contiguous（连续性）信息
  refresh_contiguous();
}

// 输出 tensor 的详细信息到标准输出流
void dumpTensorCout(const Tensor& tensor) {
  dumpTensor(std::cout, tensor);
  std::cout << '\n';
}

// 创建一个 TensorWrapper 的指针，使用给定的 tensor、level 和 life_handle
static c10::intrusive_ptr<TensorWrapper> makeTensorWrapperPtr(const Tensor& tensor, int64_t level, const std::shared_ptr<bool>& life_handle) {
  // 获取要传递给包装器的键集合，包括自动求导 CPU、CUDA 和 XLA 的键
  auto keys_to_propagate = kKeysToPropagateToWrapper | DispatchKeySet({
      DispatchKey::AutogradCPU, DispatchKey::AutogradCUDA, DispatchKey::AutogradXLA});
  auto key_set = getKeysToPropagateToWrapper(tensor, keys_to_propagate);
  // 添加 FuncTorchGradWrapper 键
  key_set = key_set.add(DispatchKey::FuncTorchGradWrapper);
  // 创建并返回 TensorWrapper 对象的指针
  return c10::make_intrusive<TensorWrapper>(key_set, tensor, level, life_handle);
}

// 安全创建 TensorWrapper 的函数，避免潜在的问题
// unsafeMakeTensorWrapper 不会检查 level 和 life_handle 是否来自同一个解释器
static Tensor unsafeMakeTensorWrapper(
    const Tensor& tensor,
    int64_t level,
    bool is_immutable,
    const std::shared_ptr<bool>& life_handle) {
  // 尝试获取 tensor 的包装器
  auto wrapped = maybeGetTensorWrapper(tensor);
  if (wrapped) {
    // 如果已存在包装器，则确保新包装的级别比现有的更高
    TORCH_INTERNAL_ASSERT(wrapped->level() < level);
  }

  // 获取要传递给包装器的键集合，包括自动求导 CPU、CUDA 和 XLA 的键
  auto keys_to_propagate = kKeysToPropagateToWrapper | DispatchKeySet({
      DispatchKey::AutogradCPU, DispatchKey::AutogradCUDA, DispatchKey::AutogradXLA});
  auto key_set = getKeysToPropagateToWrapper(tensor, keys_to_propagate);
  // 添加 FuncTorchGradWrapper 键
  key_set = key_set.add(DispatchKey::FuncTorchGradWrapper);
  
  // 使用详细函数创建 TensorWrapper，包括 level、life_handle 和是否不可变的标志
  auto result = at::detail::make_tensor<TensorWrapper>(
      key_set, tensor, level, life_handle, is_immutable);
  // 确保结果具有 FuncTorchGradWrapper 键
  TORCH_INTERNAL_ASSERT(result.key_set().has(DispatchKey::FuncTorchGradWrapper));

  // 如果 tensor 是包装数字，则将 result 设置为包装数字
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    result.unsafeGetTensorImpl()->set_wrapped_number(true);
  }

  return result;
}

// 结束命名空间 at::functorch
} // namespace at::functorch
}

// 创建一个包装了指定张量的 Tensor 对象，并设置生命周期级别和是否可变标志
Tensor makeTensorWrapper(const Tensor& tensor, int64_t level, bool is_immutable) {
  // 获取指定级别的生命周期句柄
  auto life_handle = getLifeHandleForLevel(level);
  // 调用不安全的创建 Tensor 包装器函数，返回创建的 Tensor 对象
  return unsafeMakeTensorWrapper(
      tensor,
      level,
      is_immutable,
      getLifeHandleForLevel(level));
}

// 创建一个包装了指定张量的 Tensor 对象，并使用解释器获取的级别和是否可变标志
Tensor makeTensorWrapper(const Tensor& tensor, const Interpreter& interpreter, bool is_immutable) {
  // 调用不安全的创建 Tensor 包装器函数，返回创建的 Tensor 对象
  return unsafeMakeTensorWrapper(
      tensor,
      interpreter.level(),
      is_immutable,
      interpreter.is_alive_ptr());
}

// 返回当前 TensorWrapper 对象的生存状态
bool TensorWrapper::is_alive() const {
  return *is_alive_;
}

// 创建当前 TensorWrapper 对象的浅拷贝，并分离数据，设置版本计数器和是否允许元数据更改标志
c10::intrusive_ptr<TensorImpl> TensorWrapper::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  // 使用当前值、级别和生存状态创建 TensorWrapper 的指针
  auto dest_impl = makeTensorWrapperPtr(value(), level_, is_alive_);
  dest_impl->set_version_counter(version_counter);

  // 设置是否允许张量元数据更改
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  return dest_impl;
}

// 创建当前 TensorWrapper 对象的浅拷贝，并分离数据，设置版本计数器和是否允许元数据更改标志
c10::intrusive_ptr<TensorImpl> TensorWrapper::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  // 使用当前值、级别和生存状态创建 TensorWrapper 的指针
  auto dest_impl = makeTensorWrapperPtr(value(), level_, is_alive_);
  dest_impl->set_version_counter(version_counter);

  // 设置是否允许张量元数据更改
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  return dest_impl;
}

// 不允许在 functorch 转换内部使用 `.data` 直接修改数据，抛出错误
void TensorWrapper::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  TORCH_CHECK(false, "mutating directly with `.data` inside functorch transform is not allowed.");
}

// 构造函数：使用指定的键集、张量值、级别、生存状态、是否可变标志和是否使用值的大小和步长，初始化 TensorWrapper 对象
TensorWrapper::TensorWrapper(
    c10::DispatchKeySet key_set,
    Tensor value,
    int64_t level,
    std::shared_ptr<bool> is_alive,
    bool is_immutable,
    bool use_value_sizes_strides)
  : TensorImpl(key_set, value.dtype(), value.device())
  , value_(std::move(value))
  , level_(level)
  , is_immutable_(is_immutable)
  , is_alive_(std::move(is_alive))
{
  // 内部断言，确保张量值已定义
  TORCH_INTERNAL_ASSERT(value_.defined());

  // TODO: 需要在突变时重置大小/步长
  TORCH_INTERNAL_ASSERT(use_value_sizes_strides);
  // 刷新元数据
  refreshMetadata();

  // 设置存储访问时应抛出异常
  set_storage_access_should_throw();
}

// 返回 TensorWrapper 对象的类型名称
const char* TensorWrapper::tensorimpl_type_name() const {
  return "TensorWrapper";
}

// 根据输入张量获取 TensorWrapper 对象，如果不是 FuncTorchGradWrapper 类型的张量则返回 nullptr
TensorWrapper* maybeGetTensorWrapper(const Tensor& tensor) {
  if (!tensor.key_set().has(DispatchKey::FuncTorchGradWrapper)) {
    return nullptr;
  }
  return (TensorWrapper*)(tensor.unsafeGetTensorImpl());
}

// 静态函数：处理死亡的 TensorWrapper 的后备方案，使用操作句柄和堆栈参数
static void dead_tensor_wrapper_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // 获取操作的参数数量
  auto args_size = op.schema().arguments().size();
  int64_t unwrapped_count = 0;
  // 如果张量是 TensorWrapper，且未死亡，则解包它
  auto unwrapIfDeadAndIncrement = [&](const Tensor& tensor) {
    auto* wrapped = maybeGetTensorWrapper(tensor);
    if (!wrapped) {
      return tensor;
    }

    // NOTE: 我们需要同时测试 is_alive 和 functorch 模式的调度键是否激活，因为某些操作可能会禁用键但未设置相关张量的死亡状态
    // 暂时省略具体操作
    // 定义一个包含两个调度键的常量集合，用于检查是否具有FuncTorchDynamicLayerFrontMode或FuncTorchDynamicLayerBackMode键
    constexpr auto functorch_mode_ks = DispatchKeySet(
        {DispatchKey::FuncTorchDynamicLayerFrontMode,
         DispatchKey::FuncTorchDynamicLayerBackMode});
    // 检查 wrapped 对象是否存活并且其键集合中包含任何functorch_mode_ks中的键
    if (wrapped->is_alive() && wrapped->key_set().has_any(functorch_mode_ks)) {
      // 如果条件满足，直接返回原始的 tensor
      return tensor;
    }
    // 如果条件不满足，则增加未包装计数器 unwrapped_count
    unwrapped_count++;
    // 返回 wrapped 对象的值
    return wrapped->value();
  };

  // 对栈中从 (stack->size() - args_size) 到 (stack->size() - 1) 的每个张量执行 unwrapIfDeadAndIncrement 操作
  foreachTensorInplace(*stack, stack->size() - args_size, stack->size(), unwrapIfDeadAndIncrement);
  // 断言 unwrapped_count 大于 0，否则抛出错误信息 "Should have at least one dead wrapper"
  TORCH_INTERNAL_ASSERT(unwrapped_count > 0, "Should have at least one dead wrapper");

  // 重新分发操作 op 到栈上的张量
  op.callBoxed(stack);
}

// TensorWrapper 后端的回退机制：解封装并继续执行。

TORCH_LIBRARY_IMPL(_, FuncTorchGradWrapper, m) {
  // 注册回退函数，使用给定的函数指针<&dead_tensor_wrapper_fallback>
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dead_tensor_wrapper_fallback>());
}

} // namespace at::functorch
```