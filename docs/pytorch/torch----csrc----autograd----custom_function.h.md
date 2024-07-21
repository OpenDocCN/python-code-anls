# `.\pytorch\torch\csrc\autograd\custom_function.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/core/ivalue.h>
// 包含 ATen 库的 IValue 类头文件

#include <c10/core/SymInt.h>
// 包含 c10 库的 SymInt 类头文件

#include <c10/util/flat_hash_map.h>
// 包含 c10 库的 flat_hash_map 类头文件

#include <c10/util/irange.h>
// 包含 c10 库的 irange 类头文件

#include <torch/csrc/autograd/function.h>
// 包含 torch 库的 function 头文件

#include <torch/csrc/autograd/variable.h>
// 包含 torch 库的 variable 头文件

#include <torch/csrc/autograd/variable_info.h>
// 包含 torch 库的 variable_info 头文件

#include <torch/csrc/dynamo/compiled_autograd.h>
// 包含 torch 库的 compiled_autograd 头文件

#include <vector>
// 包含 vector 头文件

namespace torch::autograd {
// 定义命名空间 torch::autograd

using optional_variable_list = std::vector<std::optional<Variable>>;
// 使用 std::vector 存储 std::optional<Variable> 的类型别名 optional_variable_list

using _jvp_fn_t = std::function<variable_list(variable_list, variable_list)>;
// 使用 std::function 定义接受两个 variable_list 参数并返回 variable_list 的函数类型别名 _jvp_fn_t

using _view_as_self_fn_t = std::function<at::Tensor(at::Tensor)>;
// 使用 std::function 定义接受一个 at::Tensor 参数并返回 at::Tensor 的函数类型别名 _view_as_self_fn_t

TORCH_API std::vector<std::optional<Variable>> _wrap_outputs(
    const variable_list& input_vars,
    const std::unordered_set<at::TensorImpl*>& non_differentiable,
    const std::unordered_set<at::TensorImpl*>& dirty_inputs,
    const at::ArrayRef<std::optional<Variable>> raw_outputs,
    const std::shared_ptr<Node>& cdata,
    const _jvp_fn_t& jvp_user_function,
    const std::unordered_set<at::TensorImpl*>& to_save_if_setup_context,
    const _view_as_self_fn_t& view_as_self_fn);
// 声明 _wrap_outputs 函数，接受多个参数，返回 std::vector<std::optional<Variable>> 类型，声明在 TORCH_API 中

TORCH_API void check_variable_result(
    const at::TensorBase& original,
    const at::TensorBase& result,
    const std::string& hook_name);
// 声明 check_variable_result 函数，接受三个参数，返回 void 类型，声明在 TORCH_API 中

// Get the return type of the forward function of the custom Function class X
template <typename X, typename... Args>
using forward_t = decltype(X::forward(nullptr, std::declval<Args>()...));
// 定义模板 forward_t，获取自定义 Function 类 X 的 forward 函数返回类型

/// To use custom autograd operations, implement a Function subclass with
/// static forward and backward functions:
///
/// `forward` can take as many arguments as you want and should return either a
/// variable list or a Variable. Use of any direct Variable arguments will be
/// registered in the graph but no vectors/sets or any other data structures
/// will be traversed. You can use std::optional<Tensor> as one of the arguments
/// and it will be registered as a variable in the graph if the argument has a
/// value. It should take a pointer to `torch::autograd::AutogradContext` as the
/// first argument. Variables can be saved in the `ctx` using
/// `ctx->save_for_backward`
/// (see `torch::autograd::AutogradContext::save_for_backward`) and other data
/// can be saved in the `ctx->saved_data` map
/// (see `torch::autograd::AutogradContext::saved_data`)
/// in the form of `<std::string, at::IValue>` pairs.
///
/// `backward` should take a pointer to `torch::autograd::AutogradContext`
/// and a variable list containing as many Variables as there were outputs from
/// `forward` as arguments. It should return as many Variables as there were
/// inputs with each of them containing the gradient w.r.t. its corresponding
/// input. Variables saved in `forward` can be accessed with
/// `ctx->get_saved_variables` (see
/// `torch::autograd::AutogradContext::get_saved_variables`) and other saved
/// data can be accessed from `ctx->saved_data`.
/// To enable compiled autograd support (torch.compile for backward) for your
// 多行注释，描述如何使用自定义 autograd 操作，并实现 Function 子类的静态 forward 和 backward 函数
/// custom autograd operation, you can set MyFunction::is_traceable
/// (see Function::istraceable notes below).
///
/// For example:
/// ```
/// class MyFunction : public Function<MyFunction> {
///   public:
///   static constexpr bool is_traceable = true;
///
///   static variable_list forward(AutogradContext *ctx, int n, Variable var) {
///      // Save data for backward in context
///      ctx->saved_data["n"] = n;
///      var.mul_(2);
///      // Mark var as modified by inplace operation
///      ctx->mark_dirty({var});
///      return {var};
///   }
///
///   static variable_list backward(AutogradContext *ctx, variable_list
///   grad_output) {
///      // Use data saved in forward
///      auto n = ctx->saved_data["n"].toInt();
///      return {grad_output[0]*n};
///   }
/// };
/// ```
///
/// To use `MyFunction`:
/// ```
/// Variable x;
/// auto y = MyFunction::apply(6, x);
/// // Example backward call
/// y[0].sum().backward();
/// ```
template <class T>
struct TORCH_API Function {
  // We need to use a different template parameter than T here because T will
  // inherit from Function, and when Function<T> is instantiated, T::forward
  // is not declared yet.
  // The enable_if check is to ensure that the user doesn't explicitly provide
  // the parameter X.
  template <typename X = T, typename... Args>
  static auto apply(Args&&... args)
      -> std::enable_if_t<std::is_same_v<X, T>, forward_t<X, Args...>>;

  // This flag is for an experimental feature: compiled autograd. Not all
  // built-in APIs are supported at the moment e.g. mark_dirty and
  // mark_non_differentiable. Before setting this flag to enable tracing for
  // your custom function <T>, you need to ensure that the backward function is
  // traceable i.e. any variables accessed in the backward other than the input
  // arguments must be handled in a similar manner to built-ins in
  // CppNode::compiled_args and CppNode::apply_with_saved.
  static constexpr bool is_traceable = false;
};

/// Context to save information during `forward` that can be accessed in
/// `backward` in custom autograd operations (see `torch::autograd::Function`
/// for details).
struct TORCH_API AutogradContext {
  // 默认构造函数，构造一个 AutogradContext 对象
  AutogradContext() = default;
  // 删除复制构造函数，禁止通过复制构造函数创建 AutogradContext 对象
  AutogradContext(const AutogradContext& other) = delete;
  // 删除赋值运算符重载，禁止通过赋值运算符重载赋值 AutogradContext 对象
  AutogradContext& operator=(const AutogradContext& other) = delete;

  /// Can be used to save non-variable data for `backward`.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  // 使用 ska::flat_hash_map 存储字符串到 at::IValue 的映射，用于保存非变量数据供反向传播使用
  ska::flat_hash_map<std::string, at::IValue> saved_data;

  /// Saves the list of variables for a future call to `backward`. This
  /// should be called at most once from inside of `forward`.
  // 保存变量列表，以备将来调用 `backward` 使用。应该在 `forward` 函数内最多调用一次。
  void save_for_backward(variable_list to_save);
  /// Marks variables in the list as modified in an in-place operation. This
  /// should be called at most once from inside of `forward` and all arguments
  /// should be inputs.
  // 标记列表中的变量在原地操作中被修改。应该在 `forward` 函数内最多调用一次，并且所有参数应为输入。
  void mark_dirty(const variable_list& inputs);
  /// Marks outputs in the list as not requiring gradients. This should be
  /// called at most once from inside of `forward` and all arguments should be
  /// outputs.
  // 标记列表中的输出不需要梯度。应该在 `forward` 函数内最多调用一次，并且所有参数应为输出。
  void mark_non_differentiable(const variable_list& outputs);
  // Sets whether undefined output grad tensors should be expanded to tensors
  // full of zeros before calling backward function. Default value is true.
  // 设置是否在调用反向传播函数之前，将未定义的输出梯度张量扩展为全零张量。默认值为 true。
  void set_materialize_grads(bool value);

  /// Get the list of variables that were saved in `forward` using
  /// `save_for_backward()`. Before returning them to the user, a check is made
  /// to ensure that they were not modified by any in-place operations.
  // 获取通过 `save_for_backward()` 在 `forward` 函数中保存的变量列表。在返回给用户之前，会检查这些变量是否被任何原地操作修改。
  variable_list get_saved_variables() const;
  // 获取并返回被标记为脏的张量的无序集合。
  const std::unordered_set<at::TensorImpl*>& get_and_bump_dirty() const;
  // 获取并返回被标记为不可区分的张量的无序集合。
  const std::unordered_set<at::TensorImpl*>& get_non_differentiable() const;

  /// Expose the Node's `task_should_compute_output` method to the cpp
  /// custom autograd Function as `needs_input_grad`.
  // 将 Node 的 `task_should_compute_output` 方法公开给 C++ 自定义自动求导函数，作为 `needs_input_grad`。
  bool needs_input_grad(size_t output_edge_index) const;
  bool needs_input_grad(std::initializer_list<IndexRange> idxs) const;

 private:
  // 存储被标记为不可区分的张量的无序集合
  std::unordered_set<at::TensorImpl*> non_differentiable_;
  // 存储被标记为脏的输入张量的无序集合
  std::unordered_set<at::TensorImpl*> dirty_inputs_;
  // 存储已保存变量的 vector
  std::vector<torch::autograd::SavedVariable> saved_variables_;
  // 待保存的变量列表
  variable_list to_save_;
  // 控制是否在调用反向传播函数之前，将未定义的输出梯度张量扩展为全零张量。默认为 true。
  bool materialize_grads_{true};

  // 拥有该 AutogradContext 的自动求导图中的 CppNode。我们使用 weak_ptr 避免引用循环。
  // 因为 grad_fn_ 拥有这个 AutogradContext，所以在我们使用它时它始终是存活的。
  std::weak_ptr<Node> grad_fn_;
  // 标记是否已释放缓冲区
  bool has_freed_buffers_{false};

  // 保存变量的私有方法
  void save_variables();

  template <class T>
  friend struct CppNode;
};

// CppNode<T> is the Node in the autograd graph that represents the user defined
// backward function for Function<T>. Calls to CppNode::apply are forward to
// T::backward().
// CppNode<T> 是自动求导图中表示用户定义的 Function<T> 的反向函数的节点。对 CppNode::apply 的调用会转发到 T::backward()。
template <class T>
`
struct CppNode : public Node {
  // 重写 apply 函数，返回一个变量列表
  variable_list apply(variable_list&& inputs) override;
  AutogradContext ctx_; // 自动求导上下文对象
  std::vector<bool> is_variable_input_; // 标记输入是否为变量
  std::vector<VariableInfo> input_info_; // 输入信息
  std::vector<VariableInfo> output_info_; // 输出信息

  // 重写释放变量函数
  void release_variables() override;

  // 设置上下文的梯度函数
  void set_ctx_grad_fn(const std::shared_ptr<Node>& node);
  // 保存变量到上下文
  void save_variables_to_ctx();

  // 重写编译参数函数
  void compiled_args(CompiledNodeArgs& args) override {
    if (!T::is_traceable) { // 如果节点不可追踪，抛出异常
      throw std::runtime_error(
          std::string(
              "compiled_args not implemented for non-traceable node: ") +
          name());
    }

    // 收集类型信息
    args.collect(static_cast<uint64_t>(typeid(T).hash_code()));
    args.collect(std::string(typeid(T).name()));

    // 收集上下文数据
    args.collect(ctx_.saved_data);
    TORCH_INTERNAL_ASSERT(ctx_.non_differentiable_.empty()); // 确保上下文中的不可微分变量为空
    TORCH_INTERNAL_ASSERT(ctx_.dirty_inputs_.empty()); // 确保上下文中的脏输入为空
    args.collect(ctx_.saved_variables_);
    TORCH_INTERNAL_ASSERT(ctx_.to_save_.empty()); // 确保待保存变量为空
    args.collect(ctx_.materialize_grads_);
    args.collect(ctx_.has_freed_buffers_);
    args.collect(is_variable_input_);
    args.collect(input_info_);
    args.collect(output_info_);
  }

  // 重写带保存变量的 apply 函数
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override {
    saved.before(ctx_.saved_data); // 保存数据前
    TORCH_INTERNAL_ASSERT(ctx_.non_differentiable_.empty()); // 确保不可微分变量为空
    TORCH_INTERNAL_ASSERT(ctx_.dirty_inputs_.empty()); // 确保脏输入为空
    saved.before(ctx_.saved_variables_); // 保存变量前
    TORCH_INTERNAL_ASSERT(ctx_.to_save_.empty()); // 确保待保存变量为空
    saved.before(ctx_.materialize_grads_);
    saved.before(ctx_.has_freed_buffers_);
    saved.before(input_info_);
    saved.before(output_info_);
    auto results = apply(variable_list(inputs)); // 调用 apply 函数
    saved.after(ctx_.saved_data); // 保存数据后
    TORCH_INTERNAL_ASSERT(ctx_.non_differentiable_.empty()); // 确保不可微分变量为空
    TORCH_INTERNAL_ASSERT(ctx_.dirty_inputs_.empty()); // 确保脏输入为空
    saved.after(ctx_.saved_variables_); // 保存变量后
    TORCH_INTERNAL_ASSERT(ctx_.to_save_.empty()); // 确保待保存变量为空
    saved.after(ctx_.materialize_grads_);
    saved.after(ctx_.has_freed_buffers_);
    saved.after(input_info_);
    saved.after(output_info_);
    return results; // 返回结果
  }
};

struct ExtractVariables : IterArgs<ExtractVariables> {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  std::vector<bool>& is_var_; // 引用变量标记向量
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  variable_list& list_; // 引用变量列表
  ExtractVariables(std::vector<bool>& is_var, variable_list& list)
      : is_var_(is_var), list_(list) {} // 构造函数初始化引用
  void operator()(const std::optional<at::Tensor>& x) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (x.has_value() && x.value().defined()) { // 如果 Tensor 存在且定义
      is_var_.push_back(true); // 标记为变量
      list_.emplace_back(x.value()); // 添加到变量列表
    } else {
      is_var_.push_back(false); // 标记为非变量
    }
  }
  void operator()(const at::Tensor& x) {
    is_var_.push_back(true); // 标记为变量
    list_.emplace_back(x);

将变量 x 添加到 list_ 中。


  }

结束当前函数。


  void operator()(const at::TensorList& list) {

定义一个函数调用运算符重载，接受 at::TensorList 类型的参数 list。


    for (const at::Tensor& x : list) {

对参数 list 中的每个元素 x 进行迭代。


      is_var_.push_back(true);

将 true 添加到 is_var_ 向量中，表示当前处理的元素是一个变量。


      list_.emplace_back(x);

将变量 x 添加到 list_ 向量中。


    }
  }

结束当前函数调用运算符重载。


  template <typename T>
  void operator()(const T& x) {

定义一个模板函数调用运算符重载，接受任意类型 T 的参数 x。


    is_var_.push_back(false);

将 false 添加到 is_var_ 向量中，表示当前处理的元素不是一个变量。
  const auto& functorch_tls = at::functorch::functorchTLSAccessor();
  // 获取functorch的TLS访问器

  if (functorch_tls) {
    // 如果TLS访问器存在
    // 在这里处理functorch对函数的支持，Python中处理。
    // 这里我们处理的是（C++）函数，不支持。
    // 而不是悄悄地不正确，让我们引发一个错误。
    functorch_tls->checkSupportsCppAutogradFunction();
  }

  // 创建一个新的CppNode<T>的智能指针，并指定了一个删除器deleteNode
  std::shared_ptr<CppNode<T>> node(new CppNode<T>(), deleteNode);

  // 定义变量列表input_vars
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  variable_list input_vars;

  // 获取输入参数的数量
  const size_t num_inputs = sizeof...(Args);
  // 预留input_vars的空间
  input_vars.reserve(num_inputs);
  // 预留node->is_variable_input_的空间
  node->is_variable_input_.reserve(num_inputs);

  // 调用extract_vars函数，将is_var和list与args进行解包并应用
  // TODO 在此处添加跟踪
  extract_vars(node->is_variable_input_, input_vars, args...);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool is_executable =
      GradMode::is_enabled() && any_variable_requires_grad(input_vars);
  // 如果梯度模式启用并且任何变量需要梯度，则is_executable为true
  auto next_edges =
      (is_executable ? collect_next_edges(input_vars) : edge_list());
  // 如果可执行，则收集下一个边；否则返回空的边列表
  node->set_ctx_grad_fn(node);
  // 设置node的上下文梯度函数为node自身
  node->set_next_edges(std::move(next_edges));
  // 设置node的下一个边为移动构造的next_edges
  node->clear_input_metadata();
  // 清除node的输入元数据

  // 预留node->input_info_的空间
  node->input_info_.reserve(input_vars.size());
  // 将input_vars中的每个变量加入node->input_info_
  for (auto& var : input_vars) {
    node->input_info_.emplace_back(var);
  }

  // 定义forward_return_t类型为forward_t<X, Args...>的返回类型
  using forward_return_t = forward_t<X, Args...>;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  forward_return_t outputs;

  // 在AutoGradMode为false的情况下执行以下代码块
  {
    AutoGradMode grad_mode(false);
    // 在此处执行函数的代码
  // 调用模板函数 T::forward，传递 node->ctx_ 和参数 args，并获取输出结果
  outputs = T::forward(&node->ctx_, std::forward<Args>(args)...);
}

_jvp_fn_t jvp_fn = [](const variable_list& inputs,
                      const variable_list& gI) -> variable_list {
  // 抛出错误，指示不支持自定义 Function 的 C++ API 中的 jvp 操作
  TORCH_CHECK(
      false,
      "jvp is not implemented for the c++ API of custom Function yet.",
      "Please open a feature request on GitHub if you need this.");
};

auto view_as_self_fn = [](const at::Tensor& x) -> at::Tensor {
  // 返回输入张量 x 的视图
  return x.view_as(x);
};

auto wrapped_outputs = _wrap_outputs(
    input_vars,
    node->ctx_.get_non_differentiable(),
    node->ctx_.get_and_bump_dirty(),
    to_optional(outputs),
    is_executable ? node : nullptr,
    jvp_fn,
    {},
    view_as_self_fn);

node->output_info_.reserve(wrapped_outputs.size());
for (auto& output : wrapped_outputs) {
  if (is_executable && output.has_value()) {
    // 如果 is_executable 为真且输出有值，则将值添加到 node->output_info_
    node->output_info_.emplace_back(output.value());
  } else if (is_executable) {
    // 如果 is_executable 为真但输出为空，则添加一个空对象到 node->output_info_
    node->output_info_.emplace_back();
  }
}

if (is_executable) {
  // 如果 is_executable 为真，则将变量保存到 node->ctx_ 中
  node->save_variables_to_ctx();
}

// wrapped_outputs 是一个 variable_list，将其转换为正确的返回类型
// 只有 Variable 和 variable_list 被接受作为返回类型
return to_output_type<forward_return_t>(wrapped_outputs);
}

// 这里的逻辑与 PyNode::apply 相同，因此更改应在两个地方同时进行
// template<class T> 开始了一个模板函数，泛型类型为 T
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved) 禁止Linter对右值引用参数未移动的警告

variable_list CppNode<T>::apply(variable_list&& inputs) {
  // OptionalDeviceGuard _device_guard 用于管理设备的可选保护
  at::OptionalDeviceGuard _device_guard;

  auto num_inputs = inputs.size();  // 获取输入变量列表的大小
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables) 禁止Linter对变量初始化的建议
  variable_list backward_inputs;  // 创建一个变量列表 backward_inputs
  backward_inputs.reserve(num_inputs);  // 预留足够的空间以容纳 num_inputs 个元素
  for (const auto i : c10::irange(num_inputs)) {  // 遍历输入变量列表的索引
    if (inputs[i].defined() || !ctx_.materialize_grads_) {
      backward_inputs.emplace_back(std::move(inputs[i]));  // 如果输入变量已定义或者不需要生成梯度，则移动到 backward_inputs
    } else {
      backward_inputs.emplace_back(output_info_[i].zeros(_device_guard));  // 否则，在输出信息中生成零值
    }
  }

  // 获取锁以保护自定义 C++ Autograd 节点的线程安全性
  // 对于自定义 Autograd 节点，这是必需的，因为我们不知道用户定义的节点在反向传播期间是否会写入共享数据
  // 参见注释 [Thread Safety on Autograd Node]
  std::lock_guard<std::mutex> lock(mutex_);

  auto outputs = T::backward(&ctx_, backward_inputs);  // 调用 T 类的静态方法 backward，传入上下文和反向输入

  const auto num_forward_inputs =
      static_cast<int64_t>(is_variable_input_.size());  // 获取变量输入的数量
  auto num_outputs = static_cast<int64_t>(outputs.size());  // 获取输出的数量
  // 如果输出数量超过了变量输入的数量，则检查是否全部为未定义状态
  if (num_outputs > num_forward_inputs) {
    bool all_undef = true;
    for (const auto i : c10::irange(num_forward_inputs, num_outputs)) {
      all_undef &= (!outputs[i].defined());  // 检查每个输出是否未定义
    }
    // 如果全部为未定义，则将结果向量截断为变量输入的数量
    if (all_undef) {
      outputs.resize(num_forward_inputs);
      num_outputs = num_forward_inputs;
    }
  }

  // 如果输出数量与变量输入数量不一致，则抛出异常
  if (num_outputs != num_forward_inputs) {
    std::string msg("function ");
    msg += name() + " returned an incorrect number of gradients (expected ";
    msg += std::to_string(num_forward_inputs) + ", got ";
    msg += std::to_string(num_outputs) + ")";
    throw std::runtime_error(msg);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables) 禁止Linter对变量初始化的建议
  variable_list results;  // 创建结果变量列表
  results.reserve(num_outputs);  // 预留足够的空间以容纳 num_outputs 个元素
  for (const auto i : c10::irange(num_outputs)) {  // 遍历输出的索引
    if (!is_variable_input_[i]) {  // 如果不是变量输入
      if (outputs[i].defined()) {  // 如果输出被定义了
        std::string msg("function ");
        msg += name() +
            " returned a gradient different that is defined at position ";
        msg += std::to_string(i + 1) +
            ", std the corresponding forward input was not a Variable";
        throw std::runtime_error(msg);  // 抛出异常，表明梯度与期望不一致
      }
      continue;  // 否则继续下一次循环
    }
    results.emplace_back(outputs[i]);  // 将输出添加到结果列表中
  }
  return results;  // 返回结果列表
}

template <class T>
void CppNode<T>::release_variables() {
  // 获取锁以确保线程安全性，参见注释 [Thread Safety on Autograd Node]
  std::lock_guard<std::mutex> lock(mutex_);
  ctx_.saved_variables_.clear();  // 清除保存的变量
  ctx_.has_freed_buffers_ = true;  // 设置已释放缓冲区的标志
}

template <class T>
void CppNode<T>::save_variables_to_ctx() {
  ctx_.save_variables();  // 将变量保存到上下文中
}

template <class T>
// 设置当前节点的梯度函数，将给定节点设置为上下文的梯度函数
void CppNode<T>::set_ctx_grad_fn(const std::shared_ptr<Node>& node) {
    // 将给定节点作为当前节点的梯度函数存储在上下文中
    ctx_.grad_fn_ = node;
}

// 结束 torch::autograd 命名空间
} // namespace torch::autograd
```