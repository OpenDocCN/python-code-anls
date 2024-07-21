# `.\pytorch\torch\csrc\autograd\functions\tensor.cpp`

```py
// 包含 Torch 的自动求导相关头文件
#include <torch/csrc/autograd/functions/tensor.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/graph_task.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/dynamo/compiled_autograd.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

// Torch 的命名空间
namespace torch {
namespace autograd {

// CopyBackwards 类的 apply 方法实现
auto CopyBackwards::apply(variable_list&& grads) -> variable_list {
  // 检查输入变量的有效性和数量
  check_input_variables("CopyBackwards", grads, 1, -1, true);
  // 借用 grads[0] 的梯度张量
  auto grad = c10::MaybeOwned<at::Tensor>::borrowed(grads[0]);
  // 创建包含两个变量的 grad_inputs 列表
  variable_list grad_inputs(2);
  if (grad->defined()) {
    if (task_should_compute_output(0)) {
      // 如果需要计算输出0，则初始化一个与 grad 形状相同的零张量
      grad_inputs[0] = at::zeros_like(*grad, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (task_should_compute_output(1)) {
      // 如果需要计算输出1
      // 处理实数到复数的拷贝操作，避免产生警告
      const auto src_type = src_options.dtype().toScalarType();
      if (!c10::isComplexType(src_type) && grad->is_complex()) {
        grad = c10::MaybeOwned<at::Tensor>::owned(at::real(grads[0]));
      }
      // 设置目标设备并将 grad 转换为指定的 src_options
      at::DeviceGuard device_guard(src_options.device());
      grad_inputs[1] = grad->to(src_options);
    }
  }
  // 返回 grad_inputs 列表
  return grad_inputs;
}

// CopyBackwards 类的 compiled_args 方法实现
void CopyBackwards::compiled_args(CompiledNodeArgs& args) {
  // 收集 src_options 到 args
  args.collect(src_options);
}

// CopyBackwards 类的 apply_with_saved 方法实现
variable_list CopyBackwards::apply_with_saved(
    const variable_list& inputs,
    SwapSavedVariables& saved) {
  // 在应用之前保存 src_options
  saved.before(src_options);
  // 调用 apply 方法处理输入变量并返回结果
  auto result = apply(variable_list(inputs));
  // 在应用之后恢复 src_options
  saved.after(src_options);
  // 返回结果
  return result;
}

// CopySlices 类的构造函数实现
CopySlices::CopySlices(
    const Variable& base_var,
    at::TensorGeometry view_,
    std::unique_ptr<ViewFunc> view_fn_,
    std::shared_ptr<Node> fn_)
    : Node(),
      base(base_var),
      view(std::move(view_)),
      view_fn(std::move(view_fn_)),
      fn(std::move(fn_)) {
  // 将 fn 的 next_edges 赋值给当前对象的 next_edges_，除了 index 0
  // index 0 对应 base_var，而不是 view_
  add_input_metadata(base_var);
  const auto num_outputs = fn->num_outputs();
  next_edges_.reserve(num_outputs);
  add_next_edge(impl::gradient_edge(base_var));
  for (const auto i : c10::irange(1, num_outputs)) {
    add_next_edge(fn->next_edge(i));
  }
}

// CopySlices 类的 apply_impl 方法模板实现
template <typename T>
inline variable_list CopySlices::apply_impl(
    variable_list&& inputs,
    const T& call_fn) {
  // 检查输入变量的有效性和数量
  check_input_variables("CopySlices", inputs, 1, -1, true);
  // 获取梯度张量的引用
  auto& grad = inputs[0];
  if (!grad.defined()) {
    // 如果梯度未定义，则返回空的变量列表
    return variable_list(num_outputs());
  }

  // 获取 fn 的互斥锁，保证线程安全性
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (!fn) {
    // 如果 fn 为空，则抛出运行时错误
    throw std::runtime_error(ERR_BACKWARD_TWICE);
  }

  // 创建一个新的与 base 的符号尺寸和步长相匹配的空张量
  auto result =
      grad.new_empty_strided_symint(base.sym_sizes(), base.sym_strides());
  // 将 grad 拷贝到 result 中
  result.copy_(grad);

  at::Tensor grad_slice;
  if (view_fn) {
    // 如果存在 view_fn，则执行相应的操作
    // （未完整）
  // 调用视图函数（view_fn），并将结果存储到 grad_slice 中
  grad_slice = (*view_fn)(result);
} else {
  // 计算视图相对于基础视图的偏移量
  auto offset = view.sym_storage_offset() - base.sym_storage_offset();
  // 使用视图的大小、步长和偏移量创建梯度切片（grad_slice）
  grad_slice =
      result.as_strided_symint(view.sym_sizes(), view.sym_strides(), offset);
}

// 查看当前图任务的执行信息
const auto exec_info = get_current_graph_task_exec_info();
if (exec_info && !exec_info->empty()) {
  // 获取函数 fn 和当前对象（this）的第一个输入的梯度边缘
  const auto& fn_edge = fn->next_edge(0);
  const auto& this_edge = this->next_edge(0);
  // 断言函数 fn_edge 和当前对象的梯度边缘是否有效
  TORCH_INTERNAL_ASSERT(fn_edge.is_valid() == this_edge.is_valid());
  if (fn_edge.is_valid()) {
    // 获取函数 fn 的下一个节点，并在当前图任务的执行信息中查找该节点
    const auto fn_next_node = fn_edge.function.get();
    auto it = exec_info->find(fn_next_node);
    if (it == exec_info->end()) {
      // 如果节点不在执行信息中，则根据需要计算输出，将节点添加到当前图任务的执行信息中
      if (task_should_compute_output(0)) {
        add_node_to_current_graph_task_exec_info(fn_next_node);
      }
    } else {
      // 断言节点是否应执行与计算输出相对应的操作
      TORCH_INTERNAL_ASSERT(
          it->second.should_execute() == task_should_compute_output(0));
    }
  }
}

// 断言图是否在修改后仍保持只读状态
TORCH_INTERNAL_ASSERT(num_outputs() == fn->num_outputs());
for (const auto i : c10::irange(1, this->num_outputs())) {
  // 断言对于每个输出，函数 fn 和当前对象的下一个边缘是否相同
  TORCH_INTERNAL_ASSERT(
      fn->next_edge(i).function.get() == this->next_edge(i).function.get());
}

// TODO: 我们克隆 grad_slice 是因为我们会在下面修改它，而 "fn" 可能会保存它以供 res 的反向传播使用。
// 如果禁用双向反向传播，可能可以避免克隆（clone()）。
auto res = call_fn({grad_slice.clone(at::MemoryFormat::Contiguous)});

// 创建一个变量列表 grad_inputs，大小为输出数量
variable_list grad_inputs(num_outputs());
for (const auto i : c10::irange(res.size())) {
    // 如果需要计算第 i 个任务的输出
    if (task_should_compute_output(i)) {
      // 如果输出未定义，将其视为零张量处理
      // 这种情况可能发生在用户定义自定义函数时
      if (!res[i].defined()) {
        continue; // 继续下一个迭代
      }
      // 如果 i 等于 0
      if (i == 0) {
        grad_slice.copy_(res[i]);
        // 使用 std::move 将 result 移动到 grad_inputs[i] 中
        grad_inputs[i] = std::move(result); // NOLINT(bugprone-use-after-move)
      } else {
        // 使用 std::move 将 res[i] 移动到 grad_inputs[i] 中
        grad_inputs[i] = std::move(res[i]);
      }
    }
  }

  // 返回梯度输入的数组
  return grad_inputs;
}

// 释放变量函数的实现
void CopySlices::release_variables() {
  // 使用互斥锁保护，确保线程安全
  std::lock_guard<std::mutex> lock(mutex_);
  // 将 fn 指针置为 nullptr，释放变量
  fn = nullptr;
}

// 编译后参数的收集函数
void CopySlices::compiled_args(CompiledNodeArgs& args) {
  // 检查是否禁止使用 view_fn，因为编译自动求导不支持视图函数
  TORCH_CHECK(!view_fn, "view_fn not supported by compiled autograd")
  // 内部断言，确保 fn 指针不为空
  TORCH_INTERNAL_ASSERT((bool)fn);
  // 收集 base、view 和 fn 的参数
  args.collect(base);
  args.collect(view);
  args.collect(fn);
  // 调用 fn 的 compiled_args 函数，收集其参数
  fn->compiled_args(args);
}

// 应用保存的变量进行操作的函数
variable_list CopySlices::apply_with_saved(
    const variable_list& grads,
    SwapSavedVariables& saved) {
  // 在操作之前保存 base 和 view 的变量状态
  saved.before(base);
  saved.before(view);
  int call_count = 0;
  // 执行应用实现，并传入 lambda 函数，该函数递归调用 fn 的 apply_with_saved 方法
  variable_list result = apply_impl(
      variable_list(grads),
      [this, &saved, &call_count](const variable_list& inputs2) {
        call_count++;
        return fn->apply_with_saved(inputs2, saved);
      });
  // 内部断言，确保调用次数为 1
  TORCH_INTERNAL_ASSERT(call_count == 1);
  // 恢复 base 和 view 的变量状态
  saved.after(base);
  saved.after(view);
  // 返回操作结果
  return result;
}

// 应用函数的实现，接受移动语义的输入参数
auto CopySlices::apply(variable_list&& inputs1) -> variable_list {
  // 调用 apply_impl 函数，传入 lambda 表达式，执行 fn 的 apply 方法
  return apply_impl(std::move(inputs1), [this](variable_list&& inputs2) {
    return (*fn)(std::move(inputs2));
  });
}

// 命名空间 autograd 结束
} // namespace autograd
// 命名空间 torch 结束
} // namespace torch
```