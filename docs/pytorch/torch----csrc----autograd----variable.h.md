# `.\pytorch\torch\csrc\autograd\variable.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/utils/python_stub.h>
// 包含 Python 互操作的工具函数头文件

#include <torch/csrc/Export.h>
// 导出宏定义头文件

#include <torch/csrc/autograd/cpp_hook.h>
// C++ 钩子函数支持的头文件

#include <torch/csrc/autograd/edge.h>
// 自动求导图中边的定义头文件

#include <torch/csrc/autograd/forward_grad.h>
// 前向梯度的头文件

#include <torch/csrc/autograd/function_hook.h>
// 函数钩子的头文件

#include <ATen/NamedTensorUtils.h>
// 命名张量工具函数的头文件

#include <ATen/core/Tensor.h>
// 张量核心定义头文件

#include <ATen/core/VariableHooksInterface.h>
// 变量钩子接口定义头文件

#include <c10/util/Exception.h>
// C10 库中异常处理的头文件

#include <cstdint>
// 标准整数类型头文件

#include <memory>
// 内存管理头文件

#include <mutex>
// 互斥量头文件，用于多线程同步

#include <string>
// 标准字符串头文件

#include <utility>
// 实用程序头文件

#include <vector>
// 标准向量头文件

namespace torch::autograd {

/// `Variable` 与 `Tensor` 完全相同（即 `using Variable = at::Tensor`）。
/// 这意味着你可以对 `Variable` 执行所有常规的数学和其他操作，就像对 `Tensor` 一样。
///
/// 我们保留 `Variable` 类的唯一原因是与外部用户的遗留 C++ 前端代码向后兼容。
/// 我们打算在不久的将来消除 `Variable` 类。
using Variable = at::Tensor;

} // namespace torch::autograd

// 以下是所有内部 API，不应显示在 libtorch 文档中。
// 因此，我们用 `#ifndef DOXYGEN_SHOULD_SKIP_THIS` ... `#endif` 包装以下代码

#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace torch::autograd {

/// 检查此类型是否被自动求导引擎支持。
/// 如果更改此处内容，请更新 torch/autograd/__init__.py 文件顶部的文档和
/// test/test_autograd.py 中的 "test_set_requires_grad_only_for_continuous_types"。
static inline bool isDifferentiableType(at::ScalarType t) {
  return isFloatingType(t) || isComplexType(t);
}

struct Node;

///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///                                Variable
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// `Variable` 在我们的自动求导机制中增强了 `Tensor` 的交互能力。
/// 概念上，`Variable` 在自动求导图中的 `Node` 之间通过 `Edge` 传播。
/// `Variable` 可以是叶子节点，如神经网络中的权重，也可以是内部变量，当它是变量之间操作的结果时。
/// 每个 `Variable` 还存储另一个称为 `grad` 的 `Variable`（梯度）。如果变量是叶子节点，则其梯度将累积到此变量中。
///
/// 每个 Tensor 都是一个 Variable，但有时我们口头上将不需要梯度的 Variable 称为 Tensor
/// （因为不适用于 Variable 的任何自动求导机制）。历史上，Variable 和 Tensor 是不同的概念，
/// 但现在它们完全相同（即 `using Variable = at::Tensor`）。
///
///                              Gradient Edges
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// 此外，`Variable` 还有 `gradient_edge` 的概念，这是
/// edge in the autograd graph that connects the variable to a particular input
/// of the gradient function that will be invoked with the variable during the
/// backward pass. More precisely, this gradient function can be one of two
/// things:
/// 1. A `grad_fn`, if the variable is in the interior of the graph. This is the
///    gradient of the function that produced the variable.
/// 2. A `grad_accumulator`, if the variable is a leaf, which accumulates a
///    scalar gradient value into its `grad` variable.
///
///                               Versioning
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Another major feature of `Variable`s are *versions*. Versions are
/// incremented when an in-place mutation of a variable occurs. Versions are
/// useful when constructing `SavedVariable`s, which take a snapshot of a
/// `Variable` at a certain version. You can retrieve a `Variable`'s version
/// through its `current_version()` method.
///
///                                 Views
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// It is possible for a  `Variable` to be a *view* of another `Variable`, in
/// which case it tracks that `Variable`'s data and autograd history. Beyond
/// construction, the interface of a view is identical to that of a regular
/// `Variable`. You can determine whether `Variable` is in fact a view by
/// probing its `is_view()` method. Note that the *view* semantics are only
/// meaningful for `Variable` relations that are relevant to autograd.
/// See NOTE [ Autograd View Variables ] for more details.
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct AutogradMeta;
struct DifferentiableViewMeta;

// Private-ish functions for manipulating variables; we don't want to put them
// on Tensor proper
namespace impl {

// WARNING: This may return a nullptr.  If you require AutogradMeta to return
// a materialized structure, use materialize_autograd_meta instead.
TORCH_API AutogradMeta* get_autograd_meta(const at::TensorBase&);

// WARNING: This will return a nullptr if the Tensor is not a view.
TORCH_API DifferentiableViewMeta* get_view_autograd_meta(const at::TensorBase&);

// Returns the current autograd meta, materializing it if it was previously
// none.  This counts as a *mutating* operation, so do not call it on
// "read-only" operators; in particular, this is NOT thread safe
TORCH_API AutogradMeta* materialize_autograd_meta(const at::TensorBase&);

/// Set the gradient accumulator of the `Variable`. This is only applicable to
/// leaf variables. Interior variables should call `set_gradient_edge()`.
TORCH_API void set_grad_accumulator(
    const Variable&,
    std::weak_ptr<Node> grad_accumulator);

/// Attempts to get a pointer to the gradient accumulator of the `Variable`,
/// if it still exists. If the gradient accumulator function has been
/// destroyed, returns a `nullptr`.
/// 尝试获取给定变量的梯度累加器，如果存在则返回，否则动态创建并返回。
TORCH_API std::shared_ptr<Node> try_get_grad_accumulator(const Variable&);

/// 获取变量的梯度累加器。如果变量是内部变量，则返回梯度函数；否则返回梯度累加器。
/// 如果变量是内部变量，返回的`Edge`对象会在其`input_nr`字段中存储连接到该变量的节点的输入索引。
/// 对于叶节点，`input_nr`始终为零。注意`set_gradient_edge`和`gradient_edge`不是对称的。
/// 必须使用`set_gradient_edge`设置`grad_fn`，使用`set_grad_accumulator`设置累加器。
TORCH_API std::shared_ptr<Node> grad_accumulator(const Variable&);

/// 返回该变量的“规范”梯度边缘，即如果是内部变量则返回梯度函数，否则返回梯度累加器。
/// 如果变量是内部变量，返回的`Edge`将存储与该变量连接的节点的输入索引。
/// 对于叶节点，`input_nr`始终为零。注意`set_gradient_edge`和`gradient_edge`不是对称的。
TORCH_API Edge gradient_edge(const Variable&);

/// 设置变量的梯度边缘，即`grad_fn`和`input_nr`。
/// 注意：这始终会设置`grad_fn`，即使这是一个叶节点变量，而不会设置`grad_accumulator`。
/// 对于后者，请使用`set_grad_accumulator`。这允许在内部变量的延迟构造过程中设置`Variable`。
TORCH_API void set_gradient_edge(const Variable&, Edge edge);

// Autograd Graph Interaction
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// 更新现有变量的`grad_fn`。在原地修改之后调用。
///
/// 对于视图变量：
/// 在原地修改之后调用。修改基础变量的`grad_fn`。
TORCH_API void rebase_history(const Variable&, Edge gradient_edge);

/// 获取当前变量的原始梯度函数指针，无论其当前是什么。
TORCH_API Node* grad_fn_unsafe(const Variable&);

/// 增加此变量的版本计数。
TORCH_API void bump_version(const Variable&);

/// 设置此变量的版本计数器。
TORCH_API void set_version_counter(
    const Variable&,
    const c10::VariableVersion& version_counter);

/// 检索此变量的版本计数器。
TORCH_API const c10::VariableVersion& version_counter(const Variable&);

/// 设置变量的名称。
TORCH_API void set_name(const Variable&, const std::string& name);

/// 为给定的`TensorBase`添加钩子。
TORCH_API void add_hook(
    const at::TensorBase&,
    std::unique_ptr<FunctionPreHook> hook);

/// 返回给定变量的所有钩子。
TORCH_API std::vector<std::unique_ptr<FunctionPreHook>>& hooks(const Variable&);

/// 清除给定`TensorBase`的所有钩子。
TORCH_API void clear_hooks(const at::TensorBase&);

/// 设置给定变量的后累积梯度钩子。
TORCH_API void set_post_acc_grad_hooks(
    const at::TensorBase&,
    std::unique_ptr<PostAccumulateGradHook> dict);

/// 返回给定变量的后累积梯度钩子。
TORCH_API std::unique_ptr<PostAccumulateGradHook>& post_acc_grad_hooks(
    const Variable&);

/// 创建C++钩子。
TORCH_API void create_cpp_hook(
    const at::TensorBase&,
    bool is_retains_grad_hooks = false);
} // namespace impl
# 每个 `Variable` 对象都有一个唯一的 `AutogradMeta` 结构体，用于存储跟踪该变量的自动求导历史所必需的元数据字段。
# 作为优化，一个 Variable 可能会存储一个空指针（nullptr），而不是一个默认构造的 AutogradMeta。
// 定义 AutogradMeta 结构体，继承自 AutogradMetaInterface 接口
struct TORCH_API AutogradMeta : public c10::AutogradMetaInterface {
  // 存储变量的名称
  std::string name_;

  // 梯度变量
  Variable grad_;

  // 梯度函数节点的共享指针
  std::shared_ptr<Node> grad_fn_;

  // 梯度累加器的弱引用
  std::weak_ptr<Node> grad_accumulator_;

  // 用于存储与 AutogradMeta 对象（及其对应的 Tensor）相关的所有前向自动微分梯度
  // 此字段延迟填充，与 AutogradMeta 和 ForwardGrad 之间有语义上的一对一对应关系
  mutable std::shared_ptr<ForwardGrad> fw_grad_;

  // hooks_ 字段被 Python 和 C++ 逻辑共同使用
  // 对于两种情况，都有一个数据结构（cpp_hooks_list_ 或 dict）作为规范副本
  // 然后，对于两种情况，我们总是在 hooks_ 注册一个单一钩子，该钩子包装列表/字典中的所有钩子
  // 如果 Tensor 上存在 grad_fn，则还会额外向 grad_fn 注册一个单一钩子
  std::vector<std::unique_ptr<FunctionPreHook>> hooks_;
  std::shared_ptr<hooks_list> cpp_hooks_list_;

  // post_acc_grad_hooks_ 字段仅存储 Python 钩子（PyFunctionTensorPostAccGradHooks）
  // 这些钩子在将 .grad 字段累积后调用，比 hooks_ 字段要简单得多
  std::unique_ptr<PostAccumulateGradHook> post_acc_grad_hooks_ = nullptr;

  // 仅对叶子变量有意义（否则必须为 false）
  bool requires_grad_{false};

  // 仅对非叶子变量有意义（否则必须为 false）
  bool retains_grad_{false};

  bool is_view_{false};

  // 此变量的“输出编号”；例如，如果此变量是函数的第二个输出，则 output_nr_ == 1
  // 我们使用这个字段来确保在将此变量传递给另一个函数时正确设置反向跟踪
  uint32_t output_nr_;

  // 互斥锁，确保并发读取操作修改内部状态仍然是线程安全的
  // 被 grad_fn()、grad_accumulator()、fw_grad() 和 set_fw_grad() 使用
  // mutable 是因为我们需要能够在此类的 const 版本中获取它
  mutable std::mutex mutex_;

  /// 设置 `Variable` 的 `requires_grad` 属性
  /// 对于希望累积梯度的叶子变量，此属性应为 true；对于所有其他变量，应为 false
  void set_requires_grad(bool requires_grad, at::TensorImpl* self_impl) final {
    TORCH_CHECK(
        !requires_grad ||
            isDifferentiableType(at::typeMetaToScalarType(self_impl->dtype())),
        "Only Tensors of floating point and complex dtype can require gradients");
    // 检查是否可以要求梯度，要求梯度的情况下确保张量的数据类型支持微分
    requires_grad_ = requires_grad;
  }

  bool requires_grad() const override {
    return requires_grad_ || grad_fn_;
  }
  // 返回当前是否需要计算梯度，如果 requires_grad_ 为 true 或者存在 grad_fn_ 则返回 true

  /// Accesses the gradient `Variable` of this `Variable`.
  Variable& mutable_grad() override {
    return grad_;
  }
  // 返回可变的梯度变量 grad_

  const Variable& grad() const override {
    return grad_;
  }
  // 返回当前的梯度变量 grad_

  const Variable& fw_grad(uint64_t level, const at::TensorBase& self)
      const override;
  // 声明前向梯度函数 fw_grad，但未在当前代码段实现

  void set_fw_grad(
      const at::TensorBase& new_grad,
      const at::TensorBase& self,
      uint64_t level,
      bool is_inplace_op) override;
  // 设置前向梯度的函数，但未在当前代码段实现

  AutogradMeta(
      at::TensorImpl* self_impl = nullptr,
      bool requires_grad = false,
      Edge gradient_edge = Edge())
      : grad_fn_(std::move(gradient_edge.function)),
        output_nr_(gradient_edge.input_nr) {
    // 构造函数，初始化 grad_fn_ 和 output_nr_
    // 如果 requires_grad 为 true，则调用 set_requires_grad 来设置是否需要梯度
    if (requires_grad) {
      TORCH_INTERNAL_ASSERT(self_impl);
      set_requires_grad(requires_grad, self_impl);
    }
    // 检查错误条件，确保 grad_fn_ 与 requires_grad_ 不同时为 true
    TORCH_CHECK(
        !grad_fn_ || !requires_grad_,
        "requires_grad should be false if grad_fn is set");
  }

  ~AutogradMeta() override {
    // 析构函数，如果存在前向梯度 fw_grad_，则清空它
    // 参见文档中的注释 [ Using ForwardGrad ]
    if (fw_grad_) {
      fw_grad_->clear();
    }
  }
/// Base class for view functions, providing reapplication of a view on a new
/// base. Each view op should get a codegenerated subclass of this class
/// containing any state needed to reconstruct the view. The class also provides
/// convenience accessors for saved SymInts / tensor state. This is useful for
/// e.g. fake-ification, where we want to use symbolic values or fake tensors
/// instead.
struct TORCH_API ViewFunc {
  virtual ~ViewFunc() {}
  
  /// Returns any SymInts in the saved state.
  virtual std::vector<c10::SymInt> get_symints() const {
    return {};
  }
  
  /// Returns the number of SymInts in the saved state.
  virtual size_t num_symints() const {
    return 0;
  }
  
  /// Returns any tensors in the saved state.
  virtual std::vector<at::Tensor> get_tensors() const {
    return {};
  }
  
  /// Returns the number of tensors in the saved state.
  virtual size_t num_tensors() const {
    return 0;
  }
  
  /// Reapplies the view on the given base using the saved state.
  virtual at::Tensor operator()(const at::Tensor&) const = 0;
  
  /// Returns a clone of this ViewFunc, optionally with the specified saved
  /// state.
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = c10::nullopt,
      std::optional<std::vector<at::Tensor>> = c10::nullopt) const = 0;

 protected:
  /// Sets the values of any SymInts in the saved state. The input vector size
  /// must match the number of SymInts in the saved state (i.e. the size of the
  /// list returned by get_symints()).
  virtual void set_symints(std::vector<c10::SymInt>) {}
  
  /// Sets the values of any Tensors in the saved state. The input vector size
  /// must match the number of Tensors in the saved state (i.e. the size of the
  /// list returned by get_tensors()).
  virtual void set_tensors(std::vector<at::Tensor>) {}
};

/// ViewFunc that represents a chain of two ViewFuncs.
struct ChainedViewFunc : public ViewFunc {
  ChainedViewFunc(
      std::unique_ptr<ViewFunc> first,
      std::unique_ptr<ViewFunc> second)
      : first(std::move(first)), second(std::move(second)) {}

  virtual ~ChainedViewFunc() override{};
  
  /// Returns combined list of SymInts from both chained ViewFuncs.
  virtual std::vector<c10::SymInt> get_symints() const override;
  
  /// Returns total number of SymInts from both chained ViewFuncs.
  virtual size_t num_symints() const override {
    return first->num_symints() + second->num_symints();
  }
  
  /// Returns combined list of tensors from both chained ViewFuncs.
  virtual std::vector<at::Tensor> get_tensors() const override;
  
  /// Returns total number of tensors from both chained ViewFuncs.
  virtual size_t num_tensors() const override {
    return first->num_tensors() + second->num_tensors();
  }
  
  /// Applies both chained ViewFuncs sequentially on the input tensor.
  virtual at::Tensor operator()(const at::Tensor&) const override;
  
  /// Returns a clone of this ChainedViewFunc, optionally with the specified saved
  /// state for each ViewFunc in the chain.
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = c10::nullopt,
      std::optional<std::vector<at::Tensor>> = c10::nullopt) const override;

 private:
  std::unique_ptr<ViewFunc> first;   ///< First ViewFunc in the chain.
  std::unique_ptr<ViewFunc> second;  ///< Second ViewFunc in the chain.
};
// 定义一个继承自 ViewFunc 的 ErroringViewFunc 结构体，用于处理视图操作中的错误情况
struct ErroringViewFunc : public ViewFunc {
  // 构造函数，接受一个错误消息作为参数，并存储在成员变量 error_msg 中
  ErroringViewFunc(const std::string& error_msg) : error_msg(error_msg) {}
  // 虚析构函数，用于释放资源
  virtual ~ErroringViewFunc() override{};
  // 重载操作符 ()，当调用时，使用 TORCH_CHECK 断言来抛出错误，信息为 error_msg
  virtual at::Tensor operator()(const at::Tensor&) const override {
    TORCH_CHECK(false, error_msg);
  }
  // 克隆函数，创建并返回当前对象的唯一指针副本
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = c10::nullopt,
      std::optional<std::vector<at::Tensor>> = c10::nullopt) const override {
    return std::make_unique<ErroringViewFunc>(error_msg);
  }

 private:
  std::string error_msg; // 存储错误消息的成员变量
};

// 定义一个 TORCH_API 的结构体 ViewInfo，用于存储关于视图操作的信息
struct TORCH_API ViewInfo {
  /// The base `Variable`
  /// 如果此 ViewInfo 表示前向（反向）AD 梯度，则此张量不能是前向（反向）视图。
  Variable base_; // 存储基本 Variable 的成员变量

  /// By default we use as_strided to recover views which is more efficient.
  /// view_fn is only saved when as_strided is not supported.
  /// 如果 as_strided 支持，则默认使用其恢复视图，这更有效率。
  /// 仅当 as_strided 不支持时，才保存 view_fn。
  std::unique_ptr<ViewFunc> view_fn_; // 存储视图函数的唯一指针

  /// Analogue of view_fn but in reverse: given a view -> produce the base by
  /// applying the inverse view.
  /// rev_view_fn 是 view_fn 的反向操作：给定一个视图 -> 通过应用逆视图来生成基本 Variable。
  std::function<Variable(const Variable&)> rev_view_fn_; // 存储反向视图函数的函数对象

  /// Accessors for the view function
  /// 获取视图函数的访问器
  bool has_view_fn() const {
    // 假设 view_fn_ 和 rev_view_fn_ 要么都存在，要么都不存在
    return view_fn_ != nullptr;
  }

  const ViewFunc& view_fn() const {
    // 断言：仅当存在视图函数时才能访问视图函数
    TORCH_CHECK(
        has_view_fn(), "Can only access the view function if it exists.");
    return *view_fn_;
  }

  std::function<Variable(const Variable&)> rev_view_fn() const {
    // 断言：仅当存在反向视图函数时才能访问反向视图函数
    TORCH_CHECK(
        has_view_fn(),
        "Can only access the reverse view function if it exists.");
    return rev_view_fn_;
  }

  /// The chain function can be used to build a new ViewInfo for a
  /// differentiable view function. It will return a new view info that
  /// accurately represents how "tensor" is a view of this instance's "base_".
  /// The "base" and "tensor" are respectively the input and output of the
  /// differentiable view function that happened. They are required to properly
  /// set the optional view_fn_ when it is not provided. The "view_func", if
  /// provided, should be a function that allows to re-do the view between
  /// "base" and "tensor".
  /// chain 函数可用于为可微视图函数构建一个新的 ViewInfo。
  /// 它将返回一个新的视图信息，准确表示 "tensor" 如何是此实例的 "base_" 的视图。
  /// "base" 和 "tensor" 分别是发生的可微视图函数的输入和输出。
  /// 当未提供 view_fn_ 时，需要它们来正确设置可选的 view_fn_。
  /// 如果提供了 "view_func"，则应为允许在 "base" 和 "tensor" 之间重新进行视图的函数。
  ViewInfo chain(
      const Variable& base,
      const Variable& tensor,
      std::unique_ptr<ViewFunc> view_func = nullptr,
      std::function<Variable(const Variable&)> rev_view_func = nullptr) const;

  // 构造函数，初始化 ViewInfo 对象
  ViewInfo(
      Variable base,
      std::unique_ptr<ViewFunc> view_fn,
      std::function<Variable(const Variable&)> rev_view_fn)
      : base_(std::move(base)),
        view_fn_(std::move(view_fn)),
        rev_view_fn_(std::move(rev_view_fn)) {
    // 断言：确保 base_ 已定义
    TORCH_CHECK(base_.defined(), "base is undefined");
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                     DifferentiableViewMeta
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// NOTE [ Autograd View Variables ]
///
/// Many operations return Variable that shares storage with an input Variable.
/// The returned Variable is called a **view** Variable on the input **base**
/// Variable.
///
/// In PyTorch, we have two types of views: differentiable views, and
/// non-differentiable views. In either type, to support proper version
/// checking, the base and view Variables must always share the same
/// version_counter.
///
///
/// Differentiable Views
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// This class allows to track both forward and backward AD differentiable
/// views. These views can have different base as non-differentiable view for
/// forward and backward mode AD are not the same.
///
/// Most function are either both forward and backward differentiable views (for
/// example: view, select, narrow, transpose, etc) or both not forward and not
/// backward differentiable views (for example: indices, values, eq, lt, etc).
/// But there are also functions that are forward but not backward
/// differentiable views (only detach for now) or functions that are backward
/// but not forward differentiable view (only make_dual and unpack dual for
/// now).
///
/// A concrete example of two views with different bases is as follow:
///
///     # Have:
///     #   dual is a dual Tensor that is neither a forward or backward view
///     detached_dual = dual.detach()
///     view = detached_dual.view_as(dual)
///     # The forward base of view is dual
///     # The backward base of view is detached_dual
///
/// - Backward Mode View
/// Differentiable views are the view variables where you want gradients to flow
/// back to the base variables. Out-of-place operations on views are quite
/// straightforward, but in-place ones are very tricky. Even if the base
/// variable may not require grad when we create the view, we still need to
/// track the view relation because future in-place ops may require back-proping
/// through it. For example, we need to support
///
///   (1) in-place operation on view, e.g.,
///
///     # Have:
///     #   base.requires_grad = False
///     #   var.requires_grad = True
///     base[1] = var  # i.e., base[1].copy_(var)
///     torch.autograd.grad(base.sum(), var)  <- should return an all ones
///     tensor
///
///   (2) in-place operation on base after view is created, e.g.,
///
///     # Have:
///     #   base.requires_grad = False
///     #   var.requires_grad = True
///     view = base[1]
///     base.copy_(var)
///     torch.autograd.grad(view.sum(), var)  <- should return a tensor with
///                                              var[1] filled with all ones and
///                                              zeros everywhere else
///
/// - Forward Mode View
/// Forward differentiable views follow the same semantic as backward ones but
/// are used in forward mode AD.
# show up differently as they are computed along with the forward evaluation.
# The hard examples above are thus very similar
#
#   (1) in-place operation on view, e.g.,
#
#     # Have:
#     #   base is a regular Tensor
#     #   var is a dual Tensor whose tangent is all ones
#     base[1] = var  # i.e., base[1].copy_(var)
#     # Now, base is a dual Tensor
#     _, fw_grad = fwAD.unpack_dual(base)  # fw_grad should be a tensor with
#                                           # fw_grad[1] filled with all ones
#                                           # and zeros everywhere else
#
#   (2) in-place operation on base after view is created, e.g.,
#
#     # Have:
#     #   base is a regular Tensor
#     #   var is a dual Tensor whose tangent is all ones
#     view = base[1]
#     base.copy_(var)
#     _, fw_grad = fwAD.unpack_dual(view)  # fw_grad should be an all ones
#                                           # tensor
#
# See Note [Forward Grad View/inplace] for more details on how we handle these
# hard cases.
#
#
# DifferentiableViewMeta is created to support gradient tracking of
# such **in-place** operations. In particular,
#   + if an in-place op is done on base, the grad_fn field of the view may
#     become stale. So accesses should always go through grad_fn(), which
#     reconstructs an updated grad_fn if the version_counter has incremented.
#     All other fields are always valid.
#   + if an in-place op is done on view, in rebase_history() of view, which is
#     called after every in-place op in VariableType.cpp, the grad_fn of base
#     is updated.
#   + if a single autograd Node returns multiple differentiable views, if any
#     output is modified by an inplace operation, the autograd engine will
#     make an equivalent graph (corresponding to the view operations) without
#     using equivalent graph, where each output is treated as if it were
#     produced by a distinct view operation. This discards the original (e.g.,
#     user provided) grad_fn. If the provided grad_fn does more than the
#     backward of the view, then the DifferentiableViewMeta must be created
#     with creation_meta= CreationMeta::MULTI_OUTPUT_NODE to prevent the
#     engine from ignoring the provided grad_fn.
#
# Interaction with GradMode:
# The particular case that we consider here is:
#
#     # Have:
#     #   base.requires_grad = True or False
#     with torch.no_grad():
#         view = base[1]
#     base.requires_grad_()
#     view.copy_(var)
#     torch.autograd.grad(base.sum(), var)  # what should it return?
#
# Given that this particular code example is ambiguous and can easily be
# replace by either moving both inside the no_grad block or both outside, we
# explicitly forbid it. For now, it is deprecated by a warning. This is
# achieved by setting creation_meta=CreationMeta::NO_GRAD_MODE for all
# differentiable views created in no_grad mode.
/// See Note [View + Inplace update for base tensor]
/// and Note [View + Inplace update for view tensor] for the details how
/// autograd handles inplace update with view ops.
///
/// Non-Differentiable Views
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// In certain cases, although function outputs share storage with inputs, they
/// will **never** require gradient history tracking. Instead of registering the
/// view relation via DifferentiableViewMeta in autograd, the views will be
/// using usual AutogradMeta and just share the version counters with the base
/// Variables.
/// Such views include:
///   1. Views created from .detach()
///   2. Views that are non-differentiable by its nature.
///      E.g., `sparse_tensor.indices()` is a integral view on a (possibly)
///      floating point tensor.
///      See top of `derivatives.yaml` on how to specify that outputs of a
///      function are non-differentiable.
/// These are called non-differentiable views as the gradients do not flow
/// through the view relation.
///
/// Relevant logic for both differentiable and non-differentiable views is
/// implemented in make_variable_(non_)differentiable_view below, and
/// wrap_output of gen_variable_type.py.
/// 根据创建元信息进行传播，若新视图的创建元信息为 DEFAULT，则使用前一个视图的创建元信息；
/// 否则，如果前一个视图的创建元信息为 INFERENCE_MODE，则保持不变，否则使用新视图的创建元信息。
inline CreationMeta propagate_creation_meta(
    CreationMeta prev_view_creation_meta,
    CreationMeta new_view_creation_meta) {
  return (new_view_creation_meta == CreationMeta::DEFAULT)
      ? prev_view_creation_meta
      : (prev_view_creation_meta == CreationMeta::INFERENCE_MODE
             ? prev_view_creation_meta
             : new_view_creation_meta);
}

/// 统一处理重新基准时的错误检查
/// indirect=true 表示调用者未直接进行 inplace 操作，而是 inplace 操作发生在其他地方。
TORCH_API void handle_view_on_rebase(
    DifferentiableViewMeta* diff_view_meta,
    bool indirect = false);

struct TORCH_API DifferentiableViewMeta : public AutogradMeta {
 private:
  /// 视图信息
  std::optional<ViewInfo> backward_info_;
  std::optional<ViewInfo> forward_info_;

  // 优化以减少创建的 ViewInfo 数量。
  // 在 backward_info_ == forward_info_ 的常见情况下，只填充 backward_info_
  // （用作前向和后向视图信息），并设置 shared_view_info_ = true。不变条件：
  //   - 如果 shared_view_info_ 为 false，则 backward_info_ 和 forward_info_ 没有特殊约束。
  //   - 如果 shared_view_info_ 为 true，则必须满足：
  //      - backward_info_.has_value() == true
  //      - forward_info_.has_value() == false
  bool shared_view_info_;

  /// 用于确保对此后向视图的任何操作都有效的额外信息。

  /// grad_fn 创建时版本计数器的值。如果 attr_version_ != version_counter.current_version()，
  /// 则 grad_fn 字段已过期。
  uint32_t attr_version_;
  CreationMeta creation_meta_;

 public:
  /// requires_grad 是反向自动微分字段，因此我们仅对反向可微视图使用视图特定逻辑
  bool requires_grad() const override {
    return requires_grad_ || grad_fn_ ||
        (has_bw_view() && get_backward_view().base_.requires_grad());
  }

  bool shared_view_info() const {
    return shared_view_info_;
  }

  bool has_bw_view() const {
    return backward_info_.has_value();
  }

  const ViewInfo& get_backward_view() const {
    TORCH_CHECK(
        has_bw_view(), "backward view info can only exist for backward views.");
    return backward_info_.value();
  }

  uint32_t get_attr_version() const {
    TORCH_CHECK(
        has_bw_view(), "attr_version can only exist for backward views.");
    return attr_version_;
  }

  void set_attr_version(uint32_t new_attr_version) {
    TORCH_CHECK(
        has_bw_view(), "attr_version can only exist for backward views.");
    attr_version_ = new_attr_version;

# 设置对象的属性版本号为新的属性版本号

  CreationMeta get_creation_meta() const {
    TORCH_CHECK(
        has_bw_view(), "creation_meta can only exist for backward views.");
    return creation_meta_;
  }

# 获取对象的创建元数据
# 仅当对象有后向视图时才存在创建元数据

  void set_creation_meta(CreationMeta new_creation_meta) {
    TORCH_CHECK(
        has_bw_view(), "creation_meta can only exist for backward views.");
    creation_meta_ = new_creation_meta;
  }

# 设置对象的创建元数据为新的创建元数据
# 仅当对象有后向视图时才能设置创建元数据

  bool has_fw_view() const {
    return shared_view_info_ || forward_info_.has_value();
  }

# 检查对象是否有前向视图

  const ViewInfo& get_forward_view() const {
    TORCH_CHECK(
        has_fw_view(), "forward view info can only exist for forward views.");
    TORCH_CHECK(
        !shared_view_info_ || has_bw_view(),
        "forward view info can only exist for forward views.");
    return shared_view_info_ ? backward_info_.value() : forward_info_.value();
  }

# 获取对象的前向视图信息
# 仅当对象有前向视图时才存在前向视图信息
# 如果前向视图信息是共享的，则必须同时存在后向视图

  DifferentiableViewMeta(
      at::TensorImpl* self_impl,
      std::optional<ViewInfo> backward_info,
      std::optional<ViewInfo> forward_info,
      bool shared_view_info,
      CreationMeta creation_meta = CreationMeta::DEFAULT);

# DifferentiableViewMeta 类的构造函数声明，接受多个参数进行初始化
# 包括对象自身的实现指针，可选的后向视图信息、前向视图信息、共享视图信息以及创建元数据
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                        Variable Implementation
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Factory Functions
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a `Variable` that is a *view* of another (*base*) variable.
/// The `gradient_edge` is an optional (gradient_function, input_number) pair.
/// `is_differentiable` is a bool that specifies whether this view is
/// differentiable, i.e., whether the relation should be tracked by autograd.
/// See NOTE [ Autograd View Variables ] for details.

/// NOTE: `allow_tensor_metadata_change` is set to true by default, because
/// there are a lot of call sites to these factory functions that need to change
/// the variable's size or storage afterwards, and they don't expect the
/// original tensor (where the variable is created from) to be updated. Setting
/// `allow_tensor_metadata_change_` to false by default would unnecessarily
/// prevent those changes from happening and is undesirable.

// See NOTE [ Autograd View Variables ] for details.
// Differentiable view. Track history with DifferentiableViewMeta.
inline Variable make_variable_differentiable_view(
    const at::Tensor& data,
    std::optional<ViewInfo> backward_info,
    std::optional<ViewInfo> forward_info,
    bool shared_view_info,
    CreationMeta creation_meta,
    bool allow_tensor_metadata_change = true) {
  // Check if the input data tensor is defined
  if (data.defined()) {
    // Ensure that the input data tensor doesn't already have autograd metadata
    TORCH_CHECK(
        data.getIntrusivePtr()->autograd_meta() == nullptr,
        "Attempted to make a tensor into a differentiable view, but the "
        "tensor already had autograd metadata associated with it.  If you are "
        "using a __torch_dispatch__ mode, the most common cause for this "
        "problem is that you used torch.overrides.enable_reentrant_dispatch() "
        "improperly; tensors created within the extent of reentrant dispatch "
        "MUST NOT be directly returned from __torch_dispatch__; instead, they "
        "must be wrapped into fresh tensors that serve as the output.  If you "
        "are not using wrappers, you probably don't need reentrant dispatch.  "
        "If this doesn't seem applicable, please file a bug to PyTorch.");
    
    // Retrieve the underlying implementation of the data tensor
    at::TensorImpl* data_impl = data.unsafeGetTensorImpl();
    // Set whether changes to tensor metadata (size or storage) are allowed
    data_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
    // Attach autograd metadata to the tensor, marking it as a differentiable view
    data_impl->set_autograd_meta(std::make_unique<DifferentiableViewMeta>(
        data_impl,
        std::move(backward_info),
        std::move(forward_info),
        shared_view_info,
        creation_meta));
    // Return the input data tensor, now marked as a differentiable view
    return data;
  }
  // Return an empty Variable if the input data tensor is not defined
  return Variable();
}

// See NOTE [ Autograd View Variables ] for details.
// Non-differentiable view. Just share version counter.
inline Variable make_variable_non_differentiable_view(
    const Variable& base,
    const at::Tensor& data,
  bool allow_tensor_metadata_change = true) {
  // 检查传入的数据是否已定义
  if (data.defined()) {
    // 当前所有不可微视图操作（detach/_indices/_values）共享与其基础张量相同的 TensorImpl。
    // 因此，这里需要分配一个新的 TensorImpl。
    auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
        /*version_counter=*/impl::version_counter(base),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    // 清除自动求导元数据，将其设置为nullptr
    data_impl_copy->set_autograd_meta(nullptr);
    // 返回一个新的 Variable 对象，使用复制的数据实现
    return Variable(data_impl_copy);
  }
  // 如果数据未定义，返回一个空的 Variable 对象
  return Variable();
}
/// 结束上一个函数定义，声明下一个函数，该函数创建一个 `Variable` 对象，
/// 从给定的 `Tensor` 复制其底层 `TensorImpl`。`requires_grad` 应该只对叶节点设置，
/// 并确定 `Variable` 是否会累积梯度。注意：`data` 不能已经是一个 `Variable`，
/// 它的动态类型必须是 `Tensor`。
inline Variable make_variable(
    at::Tensor data,
    bool requires_grad = false,
    bool allow_tensor_metadata_change = true) {
  // 检查输入的 Tensor 是否已定义
  if (data.defined()) {
    // 如果 Tensor 的引用计数为 1，并且具有唯一版本
    if (data.getIntrusivePtr().use_count() == 1 &&
        data.getIntrusivePtr()->unique_version()) {
      // 释放 Tensor 的 IntrusivePtr 并设置元数据选项
      auto data_impl = data.unsafeReleaseIntrusivePtr();
      data_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      // 如果需要梯度，则设置 AutogradMeta
      if (requires_grad) {
        data_impl->set_autograd_meta(
            std::make_unique<AutogradMeta>(data_impl.get(), requires_grad));
      } else {
        data_impl->set_autograd_meta(nullptr);
      }
      // 返回一个包含移动的 TensorImpl 的 Variable 对象
      return Variable(std::move(data_impl));
    } else {
      // 创建 TensorImpl 的浅拷贝并分离
      auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
          /*version_counter=*/0,
          /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
      // 如果需要梯度，则设置 AutogradMeta
      if (requires_grad) {
        data_impl_copy->set_autograd_meta(std::make_unique<AutogradMeta>(
            data_impl_copy.get(), requires_grad));
      } else {
        data_impl_copy->set_autograd_meta(nullptr);
      }
      // 返回一个包含拷贝的 TensorImpl 的 Variable 对象
      return Variable(data_impl_copy);
    }
  }
  // 如果输入的 Tensor 未定义，则返回一个空的 Variable 对象
  return Variable();
}

/// 从给定的 `Tensor` 创建一个 `Variable` 对象，复制其底层 `TensorImpl`。
/// `gradient_edge` 应该是一个 (function, input_nr) 对，指定在 autograd 图中的函数，
/// 以及该变量连接到该函数的特定输入。
inline Variable make_variable(
    const at::Tensor& data,
    Edge gradient_edge,
    bool allow_tensor_metadata_change = true) {
  // 检查输入的 Tensor 是否已定义
  if (data.defined()) {
    // 创建 TensorImpl 的浅拷贝并分离
    auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
        /*version_counter=*/0,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    // 设置 AutogradMeta，包括梯度边缘信息
    data_impl_copy->set_autograd_meta(std::make_unique<AutogradMeta>(
        data_impl_copy.get(), false, std::move(gradient_edge)));
    // 返回一个包含拷贝的 TensorImpl 的 Variable 对象
    return Variable(data_impl_copy);
  }
  // 如果输入的 Tensor 未定义，则返回一个空的 Variable 对象
  return Variable();
}
struct VariableHooks final : at::impl::VariableHooksInterface {
  // 重写接口函数，返回 TensorBase 的 tensor_data
  at::TensorBase tensor_data(const at::TensorBase&) const override;
  // 重写接口函数，返回 TensorBase 的 variable_data
  at::TensorBase variable_data(const at::TensorBase&) const override;
  // 重写接口函数，返回 TensorBase 的 grad_fn
  const std::shared_ptr<torch::autograd::Node>& grad_fn(
      const at::TensorBase&) const override;
  // 重写接口函数，注册 hook 到 TensorBase
  unsigned _register_hook(
      const at::TensorBase&,
      std::function<at::TensorBase(const at::TensorBase&)> hook) const override;
  // 重写接口函数，移除 TensorBase 的 hook
  void remove_hook(const at::TensorBase&, unsigned pos) const override;
  // 重写接口函数，判断 TensorBase 是否是 view
  bool is_view(const at::TensorBase&) const override;
  // 重写接口函数，返回 TensorBase 的 base
  const at::TensorBase& base(const at::TensorBase&) const override;
  // 重写接口函数，返回 TensorBase 的 name
  const std::string& name(const at::TensorBase&) const override;
  // 重写接口函数，判断 TensorBase 是否是叶子节点
  bool is_leaf(const at::TensorBase&) const override;
  // 重写接口函数，返回 TensorBase 的 output_nr
  int64_t output_nr(const at::TensorBase&) const override;
  // 重写接口函数，设置 TensorBase 的数据
  void set_data(const at::TensorBase& self, const at::TensorBase& new_data)
      const override;
  // 重写接口函数，返回 TensorBase 的数据
  at::TensorBase data(const at::TensorBase& self) const override;
  // 重写接口函数，返回 TensorBase 的版本号
  int64_t _version(const at::TensorBase& self) const override;
  // 重写接口函数，保留 TensorBase 的梯度
  void retain_grad(const at::TensorBase& self) const override;
  // 重写接口函数，判断 TensorBase 是否保留梯度
  bool retains_grad(const at::TensorBase& self) const override;
  // 重写接口函数，执行反向传播
  void _backward(
      const at::Tensor& self,
      at::TensorList inputs,
      const std::optional<at::Tensor>& gradient,
      std::optional<bool> keep_graph,
      bool create_graph) const override;
  // 重写接口函数，设置 TensorBase 是否需要梯度
  void requires_grad_(const at::TensorBase& self, bool _requires_grad)
      const override;
  // 重写接口函数，基础自动求导未实现的回调函数
  void basic_autograd_not_implemented_fallback(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet dispatch_keys,
      torch::jit::Stack* stack) const override;
};

namespace utils {

// 检查两个 Variable 对象是否具有相同的元信息
TORCH_API bool has_same_meta(const Variable& base, const Variable& other);

} // namespace utils
```