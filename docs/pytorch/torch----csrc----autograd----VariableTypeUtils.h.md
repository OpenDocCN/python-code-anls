# `.\pytorch\torch\csrc\autograd\VariableTypeUtils.h`

```
#pragma once

#include <c10/util/irange.h>  // 包含 c10 库中的整数范围工具

#include <ATen/core/boxing/KernelFunction.h>  // 包含 ATen 核心模块的 KernelFunction 定义
#include <ATen/core/dispatch/Dispatcher.h>  // 包含 ATen 核心模块的调度器定义

#include <torch/csrc/autograd/edge.h>  // 包含 Torch 自动求导模块的边界定义
#include <torch/csrc/autograd/function.h>  // 包含 Torch 自动求导模块的函数定义
#include <torch/csrc/autograd/functions/basic_ops.h>  // 包含 Torch 自动求导模块的基本操作函数定义
#include <torch/csrc/autograd/functions/tensor.h>  // 包含 Torch 自动求导模块的张量函数定义
#include <torch/csrc/autograd/grad_mode.h>  // 包含 Torch 自动求导模块的梯度模式定义
#include <torch/csrc/autograd/saved_variable.h>  // 包含 Torch 自动求导模块的保存变量定义
#include <torch/csrc/autograd/variable.h>  // 包含 Torch 自动求导模块的变量定义

#include <torch/csrc/autograd/functions/utils.h>  // 包含 Torch 自动求导模块的工具函数定义
#include <torch/csrc/autograd/jit_decomp_interface.h>  // 包含 Torch 自动求导模块的 JIT 分解接口定义
#include <torch/csrc/utils/variadic.h>  // 包含 Torch 工具模块的可变参数定义

#include <cstddef>  // 包含标准库的 cstddef 头文件，定义了一些与 C++ 标准库中 <stddef.h> 里定义的类型、宏等相似的东西
#include <functional>  // 包含标准库的 functional 头文件，提供了函数对象的类模板
#include <memory>  // 包含标准库的 memory 头文件，提供了一些与动态内存分配、智能指针等有关的工具
#include <utility>  // 包含标准库的 utility 头文件，提供了一些与 STL 模板库有关的通用工具
#include <vector>  // 包含标准库的 vector 头文件，提供了可变大小数组的容器类模板

#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

namespace torch {
namespace autograd {

enum class can_mutate_inplace_result {
  success,  // 操作成功
  non_default_backward_view,  // 非默认反向视图
  view_of_leaf,  // 叶子变量的视图
  is_leaf,  // 叶子变量
};

// The requires_grad argument is used to know if the inplace operation needs
// gradient to be setup for it.
// In particular, we can have tensor.requires_grad() != requires_grad when
// writing a Tensor that requires gradients inplace into a Tensor that does not
// require gradients: a = torch.rand(2) b = torch.rand(2, requires_grad=True)
// a.copy_(b)
inline can_mutate_inplace_result can_mutate_inplace(
    const at::Tensor& tensor,
    bool requires_grad) {
  if (!requires_grad || !GradMode::is_enabled()) {
    return can_mutate_inplace_result::success;  // 如果不需要梯度或者梯度模式未启用，直接返回成功
  }
  auto diff_view_meta = impl::get_view_autograd_meta(tensor);  // 获取张量的视图自动求导元数据
  if (diff_view_meta && diff_view_meta->has_bw_view()) {
    if (diff_view_meta->get_creation_meta() != CreationMeta::DEFAULT) {
      return can_mutate_inplace_result::non_default_backward_view;  // 如果存在非默认的反向视图，返回相应结果
    }
    if (tensor.requires_grad() && tensor._base().is_leaf()) {
      return can_mutate_inplace_result::view_of_leaf;  // 如果是叶子变量的视图，返回相应结果
    }
  }
  if (tensor.requires_grad() && tensor.is_leaf()) {
    return can_mutate_inplace_result::is_leaf;  // 如果是叶子变量，返回相应结果
  }
  return can_mutate_inplace_result::success;  // 其他情况返回成功
}

// 检查是否可以就地修改张量
inline void check_inplace(const at::Tensor& tensor, bool requires_grad) {
  switch (can_mutate_inplace(tensor, requires_grad)) {
    case can_mutate_inplace_result::success:
      return;  // 如果可以成功就地修改，则直接返回
    case can_mutate_inplace_result::non_default_backward_view: {
      return handle_view_on_rebase(impl::get_view_autograd_meta(tensor));  // 处理非默认反向视图的情况
    }
    case can_mutate_inplace_result::view_of_leaf:
      TORCH_CHECK(
          false,
          "a view of a leaf Variable that requires grad is being used in an in-place operation.");  // 检查叶子变量视图在就地操作中的使用
      break;

    case can_mutate_inplace_result::is_leaf:
      TORCH_CHECK(
          false,
          "a leaf Variable that requires grad is being used in an in-place operation.");  // 检查叶子变量在就地操作中的使用
      break;
  }
  TORCH_INTERNAL_ASSERT(false);  // 如果到达此处，表明出现了不应该发生的情况，内部断言失败
}

// 检查是否可以就地修改张量列表
inline void check_inplace(at::ITensorListRef tensors, bool requires_grad) {
  for (const auto& tensor : tensors) {
    check_inplace(tensor, requires_grad);  // 遍历每个张量，检查是否可以就地修改
  }
}
}  // namespace autograd
}  // namespace torch
// 抛出错误，指示具有 out=... 参数的函数不支持自动微分，
// 但其中一个参数需要梯度。
inline void throw_error_out_requires_grad(const char* name) {
  AT_ERROR(
      name,
      "(): functions with out=... arguments don't support automatic differentiation, "
      "but one of the arguments requires grad.");
}

// 如果张量需要梯度，则抛出错误，指示不支持具有复杂数据类型的输出的自动微分。
inline void throw_error_for_complex_autograd(
    const at::Tensor& tensor,
    const char* name) {
  if (tensor.requires_grad()) {
    TORCH_CHECK(
        !tensor.is_complex(),
        name,
        " does not support automatic differentiation for outputs with complex dtype.");
  }
}

// 抛出错误，如果基础张量与给定张量相同，提示不再允许视图操作返回与输入基础张量相同的张量。
inline void throw_error_if_base_and_tensor_are_same(
    const at::Tensor& base,
    const at::Tensor& tensor) {
  TORCH_CHECK(
      base.unsafeGetTensorImpl() != tensor.unsafeGetTensorImpl(),
      "View operation returned a tensor that is the same as the input base tensor.  This "
      "is no longer allowed; you must explicitly create a new tensor (e.g., using .detach()). "
      "As a user, you could have made a mistake implementing __torch_dispatch__ or a Python "
      "operator decomposition or meta registration; if that's not the case, please "
      "report a bug to PyTorch or the backend you are using.");
}

// 对于张量列表，调用上面的函数检查每个张量是否支持自动微分。
inline void throw_error_for_complex_autograd(
    at::ITensorListRef tensorlist,
    const char* name) {
  for (const auto& tensor : tensorlist) {
    throw_error_for_complex_autograd(tensor, name);
  }
}

// 重设变量的历史记录，用于梯度传播。
inline void rebase_history(const Variable& var, std::shared_ptr<Node> grad_fn) {
  if (grad_fn && var.defined()) {
    grad_fn->add_input_metadata(var);
    impl::rebase_history(var, {std::move(grad_fn), 0});
  }
}

// 对变量列表中的每个变量重设历史记录，用于梯度传播。
inline void rebase_history(
    const std::vector<Variable>& vars,
    const std::shared_ptr<Node>& grad_fn) {
  if (grad_fn) {
    for (auto& var : vars) {
      if (var.defined()) {
        auto output_nr = grad_fn->add_input_metadata(var);
        impl::rebase_history(var, {grad_fn, output_nr});
      } else {
        grad_fn->add_input_metadata(Node::undefined_input());
      }
    }
  }
}

// 增加张量的版本号。
inline void increment_version(const at::Tensor& t) {
  impl::bump_version(t);
}

// 结构体，用于扁平化张量参数列表。
struct Flatten : IterArgs<Flatten> {
  Flatten(variable_list& out) : out(out) {}
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  variable_list& out;
  void operator()(const at::Tensor& x) {
    out.emplace_back(x);
  }
  void operator()(const std::optional<at::Tensor>& x) {
    if (x.has_value())
      out.emplace_back(x.value());
  }
  void operator()(at::ArrayRef<at::Tensor> xs) {
    out.insert(out.end(), xs.begin(), xs.end());
  }
};

// 模板函数，用于扁平化任意数量的张量参数。
template <typename... Args>
inline variable_list flatten_tensor_args(Args&&... args) {
  variable_list out;
  out.reserve(count_tensors(std::forward<Args>(args)...));
  Flatten(out).apply(std::forward<Args>(args)...);
  return out; // RVO
}

// 根据指定的基础张量和张量创建一个视图张量，并指定其是否支持反向和前向的微分。
// 详细信息请参阅注释 [ Autograd View Variables ]。
inline at::Tensor as_view(
    const at::Tensor& base,
    const at::Tensor& tensor,
    bool is_bw_differentiable,
    bool is_fw_differentiable,
    std::unique_ptr<ViewFunc> view_func = nullptr,
    // 定义一个可管理的ViewFunc指针，默认为空指针
    std::function<at::Tensor(const at::Tensor&)> rev_view_func = nullptr,
    // 定义一个函数对象，接受一个Tensor参数并返回Tensor，初始化为空
    CreationMeta creation_meta = CreationMeta::DEFAULT,
    // 创建元信息，默认为DEFAULT
    bool allow_tensor_metadata_change = true) {
    // 是否允许张量元数据更改，默认为true

  // Note [View of inference tensor]
  // 推理张量的视图
  // 对于推理张量，此代码只能在InferenceMode之外执行，
  // 因为ADInplaceOrView位于default_included_set中。
  // 如果Inplace和View是分开的调度键，我们可以将Inplace放入default_included_set中，
  // 这样即使在InferenceMode之外，视图操作也不必通过as_view。
  if (base.is_inference())
    return tensor;

  auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(base);
  // 获取不同视图的自动求导元信息

  // To speed up the most common case, we specially handle when both the forward
  // and backward view infos are the same, and so a single shared ViewInfo can
  // be used for both of them.
  // 为了加速最常见的情况，我们特别处理前向和后向视图信息相同的情况，
  // 因此可以为它们两者使用单个共享的ViewInfo。
  if ((!diff_view_meta || diff_view_meta->shared_view_info()) &&
      is_bw_differentiable && is_fw_differentiable) {
    // 如果没有不同视图的元信息或者视图信息是共享的，
    // 并且可以在反向和前向都可微的情况下
    throw_error_if_base_and_tensor_are_same(base, tensor);
    // 如果基础和张量相同，抛出错误

    if (diff_view_meta) {
      creation_meta = propagate_creation_meta(
          diff_view_meta->get_creation_meta(), creation_meta);
      // 传播创建元信息
      return make_variable_differentiable_view(
          tensor,
          diff_view_meta->get_backward_view().chain(
              base, tensor, std::move(view_func), std::move(rev_view_func)),
          c10::nullopt,
          /*shared_view_info*/ true,
          creation_meta,
          allow_tensor_metadata_change);
    } else {
      return make_variable_differentiable_view(
          tensor,
          ViewInfo(base, std::move(view_func), std::move(rev_view_func)),
          c10::nullopt,
          /*shared_view_info*/ true,
          creation_meta,
          allow_tensor_metadata_change);
    }
  }

  // If they cannot be shared, create the required view infos
  // 如果不能共享，创建必需的视图信息
  std::optional<ViewInfo> new_bw_info;
  std::optional<ViewInfo> new_fw_info;

  if (is_bw_differentiable) {
    auto bw_view_func = view_func ? view_func->clone_and_set() : nullptr;
    // 如果可反向微分，复制并设置视图函数

    if (diff_view_meta && diff_view_meta->has_bw_view()) {
      const auto& base_bw_info = diff_view_meta->get_backward_view();
      new_bw_info = base_bw_info.chain(
          base, tensor, std::move(bw_view_func), rev_view_func);
      // 如果有反向视图，创建反向视图链
    } else {
      new_bw_info = ViewInfo(base, std::move(bw_view_func), rev_view_func);
      // 否则创建新的反向视图信息
    }
  } else {
    TORCH_CHECK(
        creation_meta == CreationMeta::DEFAULT,
        "Non-backward differentiable views must have creation_meta=CreationMeta::DEFAULT");
    // 如果不可反向微分，则检查创建元信息是否为DEFAULT
  }

  if (is_fw_differentiable) {
    // Check if base is a forward differentiable view
    // 检查基础是否是前向可微分视图
    if (diff_view_meta && diff_view_meta->has_fw_view()) {
      const auto& base_fw_info = diff_view_meta->get_forward_view();
      new_fw_info = base_fw_info.chain(
          base, tensor, std::move(view_func), std::move(rev_view_func));
      // 如果有前向视图，创建前向视图链
  } else {
      // 如果不是视图，则创建一个新的视图信息对象
      new_fw_info =
          ViewInfo(base, std::move(view_func), std::move(rev_view_func));
  }
}

// 检查是否前向或后向可微分，处理创建元数据传播和检查基础张量是否与张量相同
if (is_fw_differentiable || is_bw_differentiable) {
    // 如果差分视图元数据存在并且有后向视图，则传播创建元数据
    if (diff_view_meta && diff_view_meta->has_bw_view()) {
        creation_meta = propagate_creation_meta(
            diff_view_meta->get_creation_meta(), creation_meta);
    }
    // 抛出错误，如果基础张量与张量相同
    throw_error_if_base_and_tensor_are_same(base, tensor);
    // 返回一个可微分视图变量
    return make_variable_differentiable_view(
        tensor,
        std::move(new_bw_info),
        std::move(new_fw_info),
        /*shared_view_info*/ false,
        creation_meta,
        allow_tensor_metadata_change);
} else {
    // 返回一个不可微分视图变量
    return make_variable_non_differentiable_view(
        base, tensor, allow_tensor_metadata_change);
}
// 检查张量是否不需要梯度，并根据情况抛出错误信息
inline void check_no_requires_grad(
    const at::Tensor& tensor,    // 输入张量
    const char* name,            // 张量名称
    const char* fn_name = "",    // 函数名称，默认为空字符串
    bool check_grad_mode = true) {  // 是否检查梯度模式，默认为true
  TORCH_CHECK(
      !(tensor.defined() && tensor.requires_grad()) ||   // 检查张量是否定义且需要梯度
          !(check_grad_mode && GradMode::is_enabled()),  // 检查梯度模式是否开启
      "The function '",                                 // 错误信息的开始
      fn_name, "', is not differentiable with respect to argument '",  // 函数名称和错误信息继续
      name, "'. This input cannot have requires_grad True.");  // 张量名称和错误信息结尾
}

// 检查可选张量是否不需要梯度，并根据情况调用上述函数检查
inline void check_no_requires_grad(
    const std::optional<at::Tensor>& tensor,  // 可选的输入张量
    const char* name,                        // 张量名称
    const char* fn_name = "") {              // 函数名称，默认为空字符串
  if (tensor.has_value()) {                 // 检查可选张量是否有值
    check_no_requires_grad(*tensor, name, fn_name);  // 调用检查函数检查张量是否不需要梯度
  }
}

// 检查张量列表是否不需要梯度，并根据情况调用上述函数检查
inline void check_no_requires_grad(
    at::ITensorListRef tensors,    // 张量列表的引用
    const char* name,              // 张量名称
    const char* fn_name = "") {    // 函数名称，默认为空字符串
  // 梯度模式检查开销较大，所以仅在GradMode未启用时执行检查
  if (!GradMode::is_enabled()) {
    return;   // 如果梯度模式未开启，直接返回
  }
  for (auto& tensor : tensors) {   // 遍历张量列表
    check_no_requires_grad(tensor, name, fn_name, /*check_grad_mode*/ false);  // 调用检查函数，并禁止梯度模式检查
  }
}

// 检查可选张量列表是否不需要梯度，并根据情况调用上述函数检查
inline void check_no_requires_grad(
    const c10::List<std::optional<at::Tensor>>& tensors,  // 可选张量列表
    const char* name,                  // 张量名称
    const char* fn_name = "") {        // 函数名称，默认为空字符串
  // 梯度模式检查开销较大，所以仅在GradMode未启用时执行检查
  if (!GradMode::is_enabled()) {
    return;   // 如果梯度模式未开启，直接返回
  }
  for (std::optional<at::Tensor> tensor : tensors) {   // 遍历可选张量列表
    if (tensor.has_value()) {   // 如果可选张量有值
      check_no_requires_grad(*tensor, name, fn_name, /*check_grad_mode*/ false);  // 调用检查函数，并禁止梯度模式检查
    }
  }
}

// 假定保存的张量列表不是原位输出的函数
inline std::vector<SavedVariable> make_saved_variable_list(
    at::ITensorListRef tensors,   // 张量列表的引用
    const bool is_output = false) {  // 是否是输出张量，默认为false
  return fmap(tensors, [&is_output](const at::Tensor& tensor) -> SavedVariable {
    return SavedVariable{tensor, is_output /* is output */};  // 返回保存的变量对象
  });
}

// 假定保存的张量列表不是原位输出的函数
inline std::vector<SavedVariable> make_saved_variable_list(
    const c10::List<std::optional<at::Tensor>>& tensors,  // 可选张量列表
    const bool is_output = false) {    // 是否是输出张量，默认为false
  return fmap(
      tensors,
      [&is_output](const std::optional<at::Tensor>& tensor) -> SavedVariable {
        if (tensor.has_value()) {   // 如果可选张量有值
          return SavedVariable{*tensor, is_output /* is output */};  // 返回保存的变量对象
        } else {
          return SavedVariable{at::Tensor(), is_output /* is output */};  // 返回空的保存变量对象
        }
      });
}

// 将张量列表转换为其大小的二维向量
inline std::vector<std::vector<int64_t>> to_args_sizes(
    at::ITensorListRef tensors) {   // 张量列表的引用
  std::vector<std::vector<int64_t>> args_sizes(tensors.size());  // 初始化大小与张量列表相同的二维向量
  size_t i = 0;
  for (const auto& t : tensors) {   // 遍历张量列表
    args_sizes[i++] = t.sizes().vec();  // 将每个张量的大小转换为向量并存储在二维向量中
  }
  return args_sizes;   // 返回包含所有张量大小的二维向量
}

// 将张量列表的符号大小转换为其大小的二维向量
inline std::vector<std::vector<c10::SymInt>> to_args_sizes_symint(
    at::ITensorListRef tensors) {   // 张量列表的引用
  std::vector<std::vector<c10::SymInt>> args_sizes(tensors.size());  // 初始化大小与张量列表相同的二维向量
  size_t i = 0;
  for (const auto& t : tensors) {   // 遍历张量列表
    args_sizes[i++] = t.sym_sizes().vec();  // 将每个张量的符号大小转换为向量并存储在二维向量中
  }
  return args_sizes;   // 返回包含所有张量符号大小的二维向量
}

// 将张量列表转换为其标量类型的向量
inline std::vector<c10::ScalarType> to_args_scalartypes(
    at::ITensorListRef tensors) {   // 张量列表的引用
    # 根据传入的张量列表创建一个标量类型的向量
    std::vector<c10::ScalarType> args_scalartypes(tensors.size());
    # 初始化一个计数器
    size_t i = 0;
    # 遍历传入的张量列表
    for (const auto& t : tensors) {
        # 将当前张量的标量类型存入args_scalartypes向量中对应位置
        args_scalartypes[i++] = t.scalar_type();
    }
    # 返回存有所有张量标量类型的向量
    return args_scalartypes;
}

namespace impl {

namespace {

// 如果 run_jit_decomposition 不是成员函数，我们可以将其作为模板参数传递给 c10::Boxedkernel::makeFromFunction。
// 然而，成员函数无法通过这种方式传递 - 因此我们通过这个函数对象包装它，以便可以传递给 c10::BoxedKernel::makeFromFunctor。
class WrapperFunctor final : public c10::OperatorKernel {
 public:
  // 构造函数，接受 JitDecompInterface 指针作为参数
  WrapperFunctor(JitDecompInterface* impl) : impl_(impl){};

  // 函数调用运算符重载，用于执行运算符的 JIT 分解
  void operator()(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet ks,
      torch::jit::Stack* stack) {
    impl_->run_jit_decomposition(op, stack);
  }
  // 指向 JitDecompInterface 的指针成员
  JitDecompInterface* impl_;
};

} // namespace

// 使用参数运行 JIT 分解，为了 JVP (Jacobian Vector Product)
template <class Return, class... Args>
Return run_jit_decomposition_with_args_for_jvp(
    c10::string_view name,
    const c10::OperatorHandle& opHandle,
    c10::DispatchKeySet dispatchKeySet,
    Args&&... args) {
  // 查找 JitDecompInterface 的实现
  JitDecompInterface* impl = getJitDecompImpl();

  // 检查是否实现了操作符的 JIT 分解，否则抛出错误
  TORCH_CHECK_NOT_IMPLEMENTED(
      impl && impl->has_jit_decomposition(opHandle.schema()),
      "Trying to use forward AD with ",
      name,
      " that does not support it because it has not been implemented yet.\nPlease file an issue "
      "to PyTorch at https://github.com/pytorch/pytorch/issues/new?template=feature-request.yml "
      "so that we can prioritize its implementation or submit a PR adding the implementation to "
      "derivatives.yaml");

  // 创建包装了 WrapperFunctor 的 BoxedKernel，并使用 KernelFunction 调用它
  return c10::KernelFunction::makeFromBoxedKernel(
             c10::BoxedKernel::makeFromFunctor(
                 std::make_unique<WrapperFunctor>(impl)))
      .call<Return, Args...>(
          opHandle, dispatchKeySet, std::forward<Args>(args)...);
}

} // namespace impl

} // namespace autograd
} // namespace torch
```