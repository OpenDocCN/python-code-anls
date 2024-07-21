# `.\pytorch\aten\src\ATen\FunctionalTensorWrapper.h`

```py
#pragma once
// 预处理指令：#pragma once 确保头文件只被编译一次，避免重复包含

#include <ATen/ArrayRef.h>
#include <ATen/FunctionalStorageImpl.h>
#include <ATen/core/IListRef.h>
#include <ATen/core/List.h>
#include <ATen/core/boxing/BoxedKernel.h>
#include <ATen/core/boxing/impl/boxing.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <c10/core/DispatchKey.h>

namespace at {

// 命名空间 at，定义了一系列与张量操作相关的功能

// Note [Functionalization Pass In Core]
// The Functionalization pass is used to remove aliasing from a pytorch program.
// Functionalization pass 用于从 PyTorch 程序中移除别名。

// This is useful for backends that don't support aliasing, like XLA and Vulkan.
// 适用于不支持别名的后端，如 XLA 和 Vulkan。

// It's also necessary in order to remove mutation from a program, which is
// needed in Functorch.
// 同时也需要移除程序中的变异，这对于 Functorch 是必需的。

// Consider this program:
// a = torch.ones(...)
// b = a.view(...)
// b.add_(1)
// 在这个程序中，b 由于使用了 view()，意味着和 a 是别名。程序结束时，a 和 b 都变成了全为 2 的张量。

// However, backends that don't support aliasing aren't able to correctly implement the view() operator.
// 然而，不支持别名的后端无法正确实现 view() 运算符。

// Instead, they can opt into the Functionalization pass, which will
// sit between the user and the backend, and provide the necessary aliasing logic.
// 可以选择使用 Functionalization pass，它会位于用户和后端之间，并提供必要的别名逻辑。

// The functionalization pass will turn the above program into a slightly
// different program that has the same semantics, transparently to the user,
// functionalization pass 会将上述程序转换成一个略有不同但语义相同的程序，对用户透明，

// that backends like XLA/Vulkan are able to implement
// 后端如 XLA/Vulkan 可以实现

// a = torch.ones(...)
// b = a.view_copy(...)  # view() 替换为 view_copy()
// Backends like XLA/Vulkan can implement this!
// 后端如 XLA/Vulkan 可以实现这种方式！

// b.add_(1)
// a.add_(1)
// Our functionalization pass machinery knows that a and b are aliased - it applies b's mutation to a too.
// 我们的 functionalization pass 机制知道 a 和 b 是别名 - 它将 b 的变异应用于 a。

// So, how does the functionalization pass keep track of which tensors are aliased?
// functionalization pass 如何跟踪哪些张量是别名的？

// The pass works by wrapping EVERY tensor in the program inside of a FunctionalTensorWrapper,
// which knows about its alias'd tensors.
// functionalization pass 通过将程序中的每个张量包装在 FunctionalTensorWrapper 中工作，
// 这个包装器知道它的别名张量。

// See Note [Functionalization: Alias Removal] for details on the aliasing machinery.
// 详见 Note [Functionalization: Alias Removal]，了解有关别名机制的详细信息。

// See Note [Functionalization: Mutation Removal] for details on mutation removal.
// 详见 Note [Functionalization: Mutation Removal]，了解有关变异移除的详细信息。

struct TORCH_API FunctionalTensorWrapper : public c10::TensorImpl {
  // FunctionalTensorWrapper 结构体，继承自 c10::TensorImpl

  explicit FunctionalTensorWrapper(const Tensor& value);
  // 显式构造函数：从张量 value 创建 FunctionalTensorWrapper 对象

  // Additional constructor to create a FunctionalTensorWrapper directly from an
  // underlying tensor that was created from a view.
  // 附加构造函数：直接从由 view 创建的底层张量创建 FunctionalTensorWrapper 对象。

  // For example, the code b = a.view1() will generate a constructor call to
  // FunctionalTensorWrapper(b, a, view1_meta)
  // 例如，代码 b = a.view1() 会生成对 FunctionalTensorWrapper(b, a, view1_meta) 的构造函数调用。

  explicit FunctionalTensorWrapper(
      const Tensor& view_value,
      const FunctionalTensorWrapper* base,
      const functionalization::ViewMeta& meta);

  // Get the underlying, actual tensor, that doesn't know anything about functionalization.
  // 获取底层的实际张量，该张量对 functionalization 一无所知。

  const Tensor& value() const {
    return value_;
  };

  // The concept of "level" is only ever important to functorch; it's exposed
  // here as more of a hook for functorch to use.
  // “level” 概念仅对 functorch 重要；它在这里更像是 functorch 使用的钩子。

  int64_t level() const {
    return level_;
  };

  void set_level(int64_t level) {
    level_ = level;
  }

  bool has_metadata_mutation() const {
    // 返回是否具有元数据变异标志
    return has_metadata_mutation_;
    };
    
    // 标记变异操作
    void mark_mutation() {
      functional_storage_impl()->mark_mutation();
    }
    
    // 标记被自动微分隐藏的变异操作，
    // 例如为了将张量传递给 Triton 内核
    void mark_mutation_hidden_from_autograd() {
      functional_storage_impl()->mark_mutation_hidden_from_autograd();
    }
    
    // 在无梯度或推理模式期间标记变异操作
    void mark_mutation_during_no_grad_or_inference_mode() {
      functional_storage_impl()->mark_mutation_during_no_grad_or_inference_mode();
    }
    
    // 检查张量上的所有变异操作是否都被自动微分隐藏
    bool are_all_mutations_hidden_from_autograd() const {
      return functional_storage_impl()->are_all_mutations_hidden_from_autograd();
    }
    
    // 检查所有变异操作是否都发生在无梯度或推理模式下
    // （此处也需要完全忽略被自动微分隐藏的变异操作）
    bool are_all_mutations_under_no_grad_or_inference_mode() const {
      return functional_storage_impl()
          ->are_all_mutations_under_no_grad_or_inference_mode();
    }
    
    // 可能标记符号化操作，根据给定的视图元数据
    void maybe_mark_symbolic(const functionalization::ViewMeta& meta) {
      is_symbolic_ = is_symbolic_ | meta.has_symbolic_inputs;
    }
    
    // 返回是否具有符号化标志
    bool is_symbolic() const {
  // 返回是否当前 FunctionalTensorWrapper 曾经被 set_() 调用过
  bool was_storage_changed() {
    // 直接返回 was_storage_changed_ 的值
    return was_storage_changed_;
  }

  // 获取存储的大小
  c10::SymInt get_storage_size(bool before) {
  // 调用 functional_storage_impl() 方法获取 FunctionalTensor 的实现对象，
  // 然后调用 get_storage_size(before) 方法获取存储大小并返回
  return functional_storage_impl()->get_storage_size(before);
}

// 返回 FunctionalTensor 是否经历了 untyped_storage().resize_() 调用
bool was_inductor_storage_resized() {
  // 调用 functional_storage_impl() 方法获取 FunctionalTensor 的实现对象，
  // 然后调用 was_inductor_storage_resized() 方法检查是否调用了 resize_()
  return functional_storage_impl()->was_inductor_storage_resized();
}

// 功能化传递可以用于移除突变操作。
// 它通过用对应的非原地操作替换任何突变操作来实现这一点，然后调用 replace_()。
// 例如：
//
// a.add_(1)
//
// 将变为：
//
// tmp = a.add(1)
// a.replace_(tmp)
//
// replace_() 将 value_ 中的包装张量与 tmp 交换。
void replace_(const Tensor& other, bool from_lazy_regenerate = false);

bool is_multi_output_view() {
};

// Utility functions for the functionalization pass.

// 声明 functionalization 命名空间内部的实现细节

namespace functionalization {
namespace impl {

// 获取 Tensor 对象的 FunctionalTensorWrapper 指针，假设此操作是安全的
// 如果无法转换为 FunctionalTensorWrapper，会触发断言错误
TORCH_API inline FunctionalTensorWrapper* unsafeGetFunctionalWrapper(
    const Tensor& tensor) {
  auto functional_impl =
      static_cast<FunctionalTensorWrapper*>(tensor.unsafeGetTensorImpl());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_impl != nullptr);
  return functional_impl;
}

// 检查给定的 Tensor 是否是 functional tensor
TORCH_API bool isFunctionalTensor(const at::Tensor& tensor);

// 检查给定的 std::optional<Tensor> 是否是 functional tensor
TORCH_API bool isFunctionalTensor(const std::optional<Tensor>& t);

// 检查给定的 c10::List<std::optional<Tensor>> 是否包含 functional tensor
TORCH_API bool isFunctionalTensor(
    const c10::List<std::optional<Tensor>>& t_list);

// 检查给定的 ITensorListRef 是否包含 functional tensor
TORCH_API bool isFunctionalTensor(ITensorListRef list);

// 将普通 Tensor 转换为 functional tensor
TORCH_API Tensor to_functional_tensor(const Tensor& tensor);

// 将 std::optional<Tensor> 转换为 functional tensor
TORCH_API std::optional<Tensor> to_functional_tensor(
    const std::optional<Tensor>& tensor);

// 将 c10::List<std::optional<Tensor>> 转换为 functional tensor 的列表
TORCH_API c10::List<std::optional<Tensor>> to_functional_tensor(
    const c10::List<std::optional<Tensor>>& t_list);

// 将 ITensorListRef 转换为包含 functional tensor 的 std::vector<Tensor>
TORCH_API std::vector<Tensor> to_functional_tensor(ITensorListRef t_list);

// 冻结 functional tensor，阻止其在计算图中的梯度传播
TORCH_API void freeze_functional_tensor(const Tensor& tensor);

// 从 functional tensor 中获取普通 Tensor
// 如果 assert_functional 为 true，会断言输入确实是 functional tensor
TORCH_API Tensor
from_functional_tensor(const Tensor& tensor, bool assert_functional = true);

// 从 std::optional<Tensor> 中获取普通 Tensor
// 如果 assert_functional 为 true，会断言输入确实是 functional tensor
TORCH_API std::optional<Tensor> from_functional_tensor(
    const std::optional<Tensor>& t,
    bool assert_functional = true);

// 从 c10::List<std::optional<Tensor>> 中获取普通 Tensor 的列表
TORCH_API c10::List<std::optional<Tensor>> from_functional_tensor(
    const c10::List<std::optional<Tensor>>& t_list);

// 从 ITensorListRef 中获取包含普通 Tensor 的 std::vector<Tensor>
TORCH_API std::vector<Tensor> from_functional_tensor(ITensorListRef t_list);

// 同步操作，确保 functional tensor 与其原始数据的同步
TORCH_API void sync(const at::Tensor& t);

// 同步操作，确保 std::optional<Tensor> 与其原始数据的同步
TORCH_API void sync(const std::optional<Tensor>& t);

// 同步操作，确保 c10::List<std::optional<Tensor>> 内所有元素与其原始数据的同步
TORCH_API void sync(const c10::List<std::optional<Tensor>>& t_list);

// 同步操作，确保 ITensorListRef 内所有元素与其原始数据的同步
TORCH_API void sync(ITensorListRef t_list);

// 替换 functional tensor 的数据为其他 Tensor 的数据
TORCH_API void replace_(const Tensor& functional_tensor, const Tensor& other);

// 替换 functional tensor 列表的数据为其他 Tensor 列表的数据
TORCH_API void replace_(
    const ITensorListRef functional_tensor,
    ITensorListRef other);

// 提交对 functional tensor 的更新
TORCH_API void commit_update(const Tensor& functional_tensor);

// 提交对 functional tensor 列表的更新
TORCH_API void commit_update(ITensorListRef functional_tensor);

// 不安全地重置 functional tensor 的存储，慎用
TORCH_API void unsafe_reset_storage(const Tensor& functional_tensor);

// 标记 functional tensor 在自动求导中的变异操作为隐藏状态
TORCH_API void mark_mutation_hidden_from_autograd(
    const Tensor& functional_tensor);

// 检查 functional tensor 是否所有变异操作都对自动求导隐藏
TORCH_API bool are_all_mutations_hidden_from_autograd(
    const Tensor& functional_tensor);

// 检查 functional tensor 是否所有变异操作都在无梯度或推断模式下
TORCH_API bool are_all_mutations_under_no_grad_or_inference_mode(
    const Tensor& functional_tensor);

// XLA 特定逻辑，对于普通 functionalization 流程无效
// 用于传播 XLA 数据
TORCH_API void propagate_xla_data(
    const Tensor& functional_tensor,
    const Tensor& other);

// XLA 特定逻辑，对于普通 functionalization 流程无效
// 用于传播 XLA 数据
TORCH_API void propagate_xla_data(
    const ITensorListRef functional_tensor,
    ITensorListRef other);

// 使用视图元数据创建带有视图信息的 functional tensor
TORCH_API Tensor create_functional_tensor_with_view_meta(
    const Tensor& view_to_wrap,
    const Tensor& base,
    functionalization::ViewMeta meta,
    int64_t out_idx = 0);

// 使用视图元数据创建带有视图信息的 functional tensor 列表
TORCH_API std::vector<Tensor> create_functional_tensor_with_view_meta(
    ITensorListRef view_to_wrap,
    const Tensor& base,
    // 声明一个名为 functionalization 的命名空间中的 ViewMeta 类型的常量引用参数 meta
    const functionalization::ViewMeta& meta;
// 声明一个函数 mutate_view_meta，接受一个 Tensor 和 functionalization::ViewMeta 对象作为参数
void mutate_view_meta(
    const Tensor& self,
    const functionalization::ViewMeta& meta);

// 声明一个函数 set_sizes_strides_offset，接受两个 Tensor 参数，用于设置输出 Tensor 的大小、步幅和偏移量
void set_sizes_strides_offset(const Tensor& out, const Tensor& meta_out);

// 声明一个函数 set_sizes_strides_offset，接受两个 std::vector<Tensor> 参数，用于设置多个输出 Tensor 的大小、步幅和偏移量
void set_sizes_strides_offset(
    const std::vector<Tensor>& outs,
    const std::vector<Tensor>& meta_outs);

//  ~~~~~ TLS used in functionalization ~~~~~

// 获取功能化重应用视图的 TLS 状态
TORCH_API bool getFunctionalizationReapplyViewsTLS();
// 设置功能化重应用视图的 TLS 状态
TORCH_API void setFunctionalizationReapplyViewsTLS(bool reapply_views);

// FunctionalizationReapplyViewsGuard 类，用于管理功能化重应用视图的 TLS 状态
class TORCH_API FunctionalizationReapplyViewsGuard {
 public:
  // 构造函数，保存当前的 TLS 状态并设置新的状态
  FunctionalizationReapplyViewsGuard(bool reapply_views)
      : prev_(getFunctionalizationReapplyViewsTLS()) {
    setFunctionalizationReapplyViewsTLS(reapply_views);
  }

  // 析构函数，恢复之前保存的 TLS 状态
  ~FunctionalizationReapplyViewsGuard() {
    setFunctionalizationReapplyViewsTLS(prev_);
  }

  // 禁用拷贝构造函数和赋值运算符重载
  FunctionalizationReapplyViewsGuard(
      const FunctionalizationReapplyViewsGuard&) = delete;
  FunctionalizationReapplyViewsGuard operator=(
      const FunctionalizationReapplyViewsGuard&) = delete;
  // 禁用移动构造函数和移动赋值运算符重载
  FunctionalizationReapplyViewsGuard(FunctionalizationReapplyViewsGuard&&) =
      delete;
  FunctionalizationReapplyViewsGuard operator=(
      FunctionalizationReapplyViewsGuard&&) = delete;

 private:
  bool prev_;  // 保存前一个 TLS 状态的成员变量
};

} // namespace impl

// 辅助函数，调用一个不在原地执行的组合 aten 内核，可能在内部使用变异/视图，并对其进行功能化处理
TORCH_API void functionalize_op_helper(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack);

// 模板结构 _functionalize_aten_op，用于功能化处理 aten 操作
template <class Op, bool symint, class ReturnType, class... ParameterTypes>
struct _functionalize_aten_op final {};

// 特化模板 _functionalize_aten_op，定义了调用 aten 操作的静态成员函数 call
template <class Op, bool symint, class ReturnType, class... ParameterTypes>
struct _functionalize_aten_op<Op, symint, ReturnType(ParameterTypes...)> final {
  static ReturnType call(
      typename c10::maybe_keep_symint<symint, ParameterTypes>::type... args) {
    using FuncType = ReturnType(
        typename c10::maybe_keep_symint<symint, ParameterTypes>::type...);
    auto op = c10::Dispatcher::singleton()
                  .findSchemaOrThrow(
                      (const char*)Op::name, (const char*)Op::overload_name)
                  .typed<FuncType>();

    // 调用 boxed kernel wrapper 的 call 函数，将操作包装为 BoxedKernel，并传递参数
    return c10::impl::BoxedKernelWrapper<FuncType>::call(
        c10::BoxedKernel::makeFromFunction<functionalize_op_helper>(),
        op,
        // BoxedKernelWrapper 知道忽略此 keyset 参数，因为 functionalize_op_helper 不接受 DispatchKeySet
        c10::DispatchKeySet(),
        args...);
  }
};

// 使用 Op::schema 定义功能化 aten 操作的类型别名
template <class Op>
using functionalize_aten_op =
    _functionalize_aten_op<Op, false, typename Op::schema>;

// 使用 Op::schema 定义符号整数化 aten 操作的类型别名
template <class Op>
using functionalize_aten_op_symint =
    _functionalize_aten_op<Op, true, typename Op::schema>;

} // namespace functionalization
} // namespace at
```