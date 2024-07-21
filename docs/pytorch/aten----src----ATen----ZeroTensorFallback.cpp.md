# `.\pytorch\aten\src\ATen\ZeroTensorFallback.cpp`

```
// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含调度器的头文件
#include <ATen/core/dispatch/Dispatcher.h>
// 包含操作注册的头文件
#include <ATen/core/op_registration/op_registration.h>
// 包含一元操作的头文件
#include <ATen/native/UnaryOps.h>
// 包含 ATen 库的原生函数头文件
#include <ATen/NativeFunctions.h>
// 包含 C10 实用工具中的范围工具头文件
#include <c10/util/irange.h>
// 包含 Torch 库的头文件
#include <torch/library.h>
// 包含数学位下降列表的头文件
#include <ATen/native/MathBitFallThroughLists.h>

// ATen 命名空间开始
namespace at {

  // TODO: 添加一条注释来解释设计决策
  // ZeroTensors 被设计为不可变的。因此，当对 ZeroTensors 执行原地操作时，我们会报错。
  static void zeroTensorFallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
    // 获取操作的参数列表
    const auto& arguments = op.schema().arguments();
    // 获取参数的数量
    const auto num_arguments = arguments.size();
    // 栈的起始位置
    const auto stack_start = stack->size() - num_arguments;

    // 可选的写入标志
    std::optional<bool> is_write;
    // 遍历参数
    for (const auto i : c10::irange(num_arguments)) {
      // 获取参数的别名信息
      const auto& alias_info = arguments[i].alias_info();
      // 如果别名信息不为空
      if (alias_info != nullptr) {
        // 如果写入标志已经被设置
        if (is_write.has_value()) {
          // 检查是否写入与当前参数的别名信息一致
          TORCH_CHECK(*is_write == alias_info->isWrite(),
            "Unsupported operator for ", "ZeroTensorFallback: ", op.schema().name(),
            "ZeroTensor fallback doesn't work for operators with a mix "
            "mutable and non-mutable inputs that alias with outputs, "
            "this must be implemented manually.  "
            "If you got this error on a core op, please report a bug to PyTorch.");
        } else {
          // 设置写入标志
          is_write = alias_info->isWrite();
        }
      }
    }

    // 如果写入标志已设置且不为写入操作
    if (is_write.has_value() && !*is_write) {
      // 假设视图操作会正确处理 ZeroTensor 位，通过传播调度键集中的调度键
      // 这并不总是正确的，因此您应该测试这些情况。
      op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::ZeroTensor), stack);
      return;
    }
    // 对于参数列表中的每个索引 i，循环处理
    for (const auto i : c10::irange(num_arguments)) {
      // 获取当前参数在堆栈中的引用
      auto& ivalue = (*stack)[stack_start + i];
      // 如果当前值不是 Tensor 或 Tensor 列表类型，则跳过处理
      if (!(ivalue.isTensor() || ivalue.isTensorList())) {
        continue;
      }
      // 获取当前参数的详细信息
      const auto& argument = arguments[i];
      bool mut_arg = false;

      // 如果参数有别名信息
      if (argument.alias_info()) {
        // 在上面的 is_write 循环中已经测试过了
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(argument.alias_info()->isWrite());
        mut_arg = true;
      }

      // 如果当前值是 Tensor 类型
      if (ivalue.isTensor()) {
        // 将当前值移动并转换为 Tensor
        auto tensor = std::move(ivalue).toTensor();
        // 如果是零 Tensor
        if (tensor._is_zerotensor()) {
          // 检查是否为可变参数，零 Tensor 是不可变的
          TORCH_CHECK(!mut_arg, "ZeroTensors are immutable. Please use the materialized zero tensor ",
                      "obtained using .clone() if you want a mutable tensor.");
          // 创建一个与 tensor 类型和设备相同的全零 Tensor，并扩展到相同的大小
          tensor = at::zeros({}, tensor.options()).expand(tensor.sizes());
        }
        // 将处理后的 Tensor 放回到堆栈中
        (*stack)[stack_start + i] = std::move(tensor);
      } else if (ivalue.isTensorList()) {  // 如果当前值是 Tensor 列表类型
        // 将当前值移动并转换为 Tensor 列表
        auto tensors = std::move(ivalue).toTensorList();
        // 遍历 Tensor 列表中的每个 Tensor
        for(const auto j : c10::irange(tensors.size())) {
          const Tensor& tensor = tensors[j];
          // 如果是零 Tensor
          if (tensor._is_zerotensor()) {
            // 检查是否为可变参数，零 Tensor 是不可变的
            TORCH_CHECK(!mut_arg, "ZeroTensors are immutable. Please use the materialized zero tensor ",
                        "obtained using .clone() if you want a mutable tensor.");
            // 创建一个与 tensor 类型和设备相同的全零 Tensor，并扩展到相同的大小
            tensors[j] = at::zeros({}, tensor.options()).expand(tensor.sizes());
          }
        }
        // 将处理后的 Tensor 列表放回到堆栈中
        (*stack)[stack_start + i] = std::move(tensors);
      }
    }

    // 使用 dispatch_keys 中的信息重新调度操作
    op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::ZeroTensor), stack);
  }


  // 实现一个 Torch 库函数的回退，使用指定的 C++ 函数
  TORCH_LIBRARY_IMPL(_, ZeroTensor, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&zeroTensorFallback>());
  }

  // 实现一个 Torch 库函数的具体操作，用于 aten 命名空间的 ZeroTensor
  TORCH_LIBRARY_IMPL(aten, ZeroTensor, m) {
    // 实现 "zeros_like" 函数的回退
    m.impl("zeros_like", torch::CppFunction::makeFallthrough());
    // 实现 "mul.Scalar" 函数的回退
    m.impl("mul.Scalar", torch::CppFunction::makeFallthrough());
    // 实现 "add.Scalar" 函数的回退
    m.impl("add.Scalar", torch::CppFunction::makeFallthrough());
    // 实现 "copy_" 函数的回退
    m.impl("copy_", torch::CppFunction::makeFallthrough());
    // 实现 "clone" 函数的回退
    m.impl("clone", torch::CppFunction::makeFallthrough());
    // 实现 "dot" 函数的回退
    m.impl("dot", torch::CppFunction::makeFallthrough());
    // 实现 "vdot" 函数的回退
    m.impl("vdot", torch::CppFunction::makeFallthrough());
    // 下列函数在 native_functions.yaml 中有特定的注册，不使用回退功能
    // m.impl("mul.Tensor", torch::CppFunction::makeFallthrough());
    // m.impl("add.Tensor", torch::CppFunction::makeFallthrough());
    // m.impl("linalg_cross", torch::CppFunction::makeFallthrough());

    // 注册视图函数
    TORCH_VIEW_FNS(m)
    // 注册 Tensor 工具函数和构造函数
    TENSOR_UTILITIES_AND_CONSTRUCTORS(m)
  }
} // namespace at
```