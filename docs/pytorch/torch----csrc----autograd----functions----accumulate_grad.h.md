# `.\pytorch\torch\csrc\autograd\functions\accumulate_grad.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/CachedTensorUtils.h>
// 包含 CachedTensorUtils 头文件，提供了关于缓存张量工具的功能

#include <ATen/LegacyBatchedTensorImpl.h>
// 包含 LegacyBatchedTensorImpl 头文件，支持遗留的批处理张量实现

#include <ATen/TensorOperators.h>
// 包含 TensorOperators 头文件，提供了张量操作的函数

#include <torch/csrc/Export.h>
// 包含 Export 头文件，用于导出符号

#include <torch/csrc/autograd/function.h>
// 包含 function 头文件，实现了自动求导的函数

#include <torch/csrc/autograd/utils/grad_layout_contract.h>
// 包含 grad_layout_contract 头文件，提供了自动求导过程中的梯度布局协议工具函数

#include <torch/csrc/autograd/variable.h>
// 包含 variable 头文件，定义了自动求导中的变量类

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 如果未定义 AT_PER_OPERATOR_HEADERS，包含 Functions 头文件，提供张量操作的函数集合
#else
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS，包含 _sparse_coo_tensor_unsafe 头文件，提供不安全的稀疏 COO 张量操作
#endif

#include <mutex>
// 包含互斥量头文件，用于多线程同步

namespace torch {
namespace autograd {

#define CHECK_RESULT(RESULT, VAR)                                          \
  if (!(RESULT.is_sparse() || VAR.is_sparse() || RESULT.is_sparse_csr() || \
        VAR.is_sparse_csr())) {                                            \
    // 检查梯度和参数是否遵循梯度布局协议，如果不是则发出警告
    if (!utils::obeys_layout_contract(RESULT, VAR)) {                      \
      TORCH_WARN_ONCE(                                                     \
          "grad and param do not obey the gradient layout contract. "      \
          "This is not an error, but may impair performance.\n"            \
          "grad.sizes() = ",                                               \
          RESULT.sizes(),                                                  \
          ", strides() = ",                                                \
          RESULT.strides(),                                                \
          "\n",                                                            \
          "param.sizes() = ",                                              \
          VAR.sizes(),                                                     \
          ", strides() = ",                                                \
          VAR.strides());                                                  \
    }                                                                      \
  }

struct TORCH_API AccumulateGrad : public Node {
  explicit AccumulateGrad(Variable variable_);
  // AccumulateGrad 结构体，用于累积梯度的节点，继承自 Node 类，以变量 variable_ 初始化

  variable_list apply(variable_list&& grads) override;
  // 应用梯度列表到节点，重写父类的 apply 方法

  std::vector<std::unique_ptr<FunctionPreHook>>& tensor_pre_hooks() noexcept
      override {
    // 获取张量的预钩子列表，重写父类的方法
    // 注意：由于 AccumulateGrad 节点只是从张量的弱引用，可以在张量仍然存活时销毁，因此必须在此处延迟读取张量钩子。
    return impl::hooks(variable);
  }

  std::unique_ptr<PostAccumulateGradHook>& tensor_post_acc_grad_hooks() noexcept
      override {
    // 获取张量的后累积梯度钩子，重写父类的方法
    // 注意：由于 AccumulateGrad 节点只是从张量的弱引用，可以在张量仍然存活时销毁，因此必须在此处延迟读取张量钩子。
    // 如果变量的梯度未定义
    if (!variable_grad.defined()) {
      // 如果不在梯度模式下，并且新梯度不是稀疏的，并且不是稀疏的 CSR 格式，并且不是稀疏的 CSR 格式的变量，
      // 并且新梯度的使用计数小于或等于预期引用数，并且（新梯度是 MKLDNN 张量或者新梯度遵循布局约定与变量相同）
      // 在这些条件下，可以在不进行深度复制的情况下窃取新梯度。
      update_grad(new_grad.detach());
    } else if (
        // 如果不在梯度模式下，并且新梯度是稀疏的，并且新梯度的索引和值都是连续的，并且索引和值的使用计数都应该小于或等于1，
        // 因为稀疏张量应该是唯一持有这些引用的张量。
        !GradMode::is_enabled() && new_grad.is_sparse() &&
        new_grad._indices().is_contiguous() &&
        new_grad._values().is_contiguous() &&
        new_grad._indices().use_count() <= 1 &&
        new_grad._values().use_count() <= 1 &&
        new_grad.use_count() <= num_expected_refs) {
      // 不能分离稀疏张量（因为分离后不允许元数据更改），因此为梯度创建一个新的稀疏张量，这是一个浅拷贝。
      // 我们需要一个浅拷贝，以确保修改原始梯度张量不会修改我们累积的梯度。
      // 我们只有在索引和值本身是连续的情况下才跳过克隆，出于向后兼容性的原因。
      // 如果不进行这种优化，我们将克隆整个稀疏张量，包括克隆索引和值。
      // 有关详细信息，请参阅 https://github.com/pytorch/pytorch/issues/34375。

      // 当前没有预期此条件为真的情况
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          !at::caching::is_cached_tensor(new_grad._indices()) &&
          !at::caching::is_cached_tensor(new_grad._values()) &&
          !at::caching::is_cached_tensor(new_grad));

      // 更新梯度为不安全的 COO 格式稀疏张量
      update_grad(at::_sparse_coo_tensor_unsafe(
          new_grad._indices(),
          new_grad._values(),
          new_grad.sizes(),
          new_grad.options()));
    } else {
      // 否则，根据情况更新梯度
      if (new_grad.is_sparse() || new_grad.is_sparse_csr() ||
          new_grad.is_nested()) {
        // 如果是稀疏张量、稀疏 CSR 格式或者是嵌套的张量，进行克隆更新
        update_grad(new_grad.clone());
      } else {
        if (new_grad.is_mkldnn()) {
          // 如果是 MKLDNN 张量，进行克隆更新
          update_grad(new_grad.clone());
        } else {
          // 否则，根据 "Gradient Layout Contract" 深度复制新梯度
          update_grad(utils::clone_obey_contract(new_grad, variable));
        }
      }
    }
    } else {
      // 定义一个 Tensor 类型的变量 result
      at::Tensor result;
      // 如果 variable_grad 是稀疏张量并且 new_grad 不是稀疏张量
      if (variable_grad.is_sparse() && !new_grad.is_sparse()) {
        // 在 CPU 后端中，稀疏张量加密集会抛出错误，因此这里优先选择密集加稀疏的方式
        result = new_grad + variable_grad;
      } else {
        // 假设 operator+ 的结果通常匹配第一个参数的步长，同时希望 variable_grad 最初遵守布局约定
        result = variable_grad + new_grad;
      }
      // 检查 result 是否符合预期，如果不符合则会抛出异常
      CHECK_RESULT(result, variable);
      // 调用 update_grad 函数，传递 result 的移动语义
      update_grad(std::move(result));
      // ^ 我们可以通过以下方式更积极地强制执行约定：
      // if (obeys_layout_contract(new_grad, variable)) {
      //   update_grad(new_grad + variable_grad);
      // } else {
      //   update_grad(variable_grad + new_grad);
      // }
      // 以上代码会在其中一个张量已经具有正确步长的情况下，确保保存的梯度也具有正确步长。
      // 也可以通过以下方式确保约定：
      // auto result = variable_grad + new_grad (或者反过来)，检查 result 的布局，并在 update_grad 之前复制到符合约定的克隆体中。这种复制将需要额外的内存传递。
      // 由于 GradMode 在此分支中已启用，并且 add_out 操作不可微，因此无法创建具有正确布局的空结果，然后通过单个内核进行 add_out 操作。也许这种方法不值得麻烦。
    }
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  Variable variable;
};

#undef CHECK_RESULT

} // namespace autograd
} // namespace torch
```