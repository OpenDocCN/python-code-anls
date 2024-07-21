# `.\pytorch\aten\src\ATen\native\CPUFallback.cpp`

```py
// 定义预处理器指令，用于仅包含方法操作符的Torch模式
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含CPU回退相关的头文件
#include <ATen/native/CPUFallback.h>

// 包含IValue相关的核心头文件
#include <ATen/core/ivalue.h>
// 包含堆栈操作相关的核心头文件
#include <ATen/core/stack.h>
// 包含分发器相关的核心头文件
#include <ATen/core/dispatch/Dispatcher.h>

// 包含字符串流的标准头文件
#include <sstream>
// 包含向量的标准头文件
#include <vector>

// 条件编译指令，若未定义每个操作符头文件，则包含通用操作函数头文件，否则包含特定操作头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/_to_cpu.h>
#endif

// ATen命名空间下的native命名空间
namespace at::native {

// 将张量列表转换为CPU张量的便捷助手函数
static std::vector<at::Tensor> to_cpu(const at::TensorList& tensors) {
    // 不能简单地对整个张量列表调用at::to_cpu()，因为未定义的张量会导致错误。先分离未定义的张量。
    std::vector<at::Tensor> cpu_tensors(tensors.size());
    std::vector<at::Tensor> valid_tensors;
    std::vector<bool> to_translate(tensors.size());
    for (const auto i : c10::irange(tensors.size())) {
        const at::Tensor& tensor = tensors[i];
        // 明确处理这里的未定义张量，而不是让`at::_to_cpu`处理它。
        // 否则，我们需要让所有后端都有自己的`_to_cpu`实现来正确处理未定义的张量。
        if (tensor.defined()) {
            to_translate[i] = true;
            valid_tensors.push_back(tensor);
        } else {
            cpu_tensors[i] = tensor;
        }
    }
    // 将有效的张量列表移动到CPU上
    auto cpu_valid_tensors = at::_to_cpu(valid_tensors);
    for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
        if (to_translate[i]) {
            cpu_tensors[i] = std::move(cpu_valid_tensors[defined_pos++]);
        }
    }
    // 返回CPU张量列表
    return cpu_tensors;
}

// 计算目标设备，返回输出张量的设备
static std::optional<c10::Device> compute_target_device(std::vector<at::Tensor>& t_args, std::vector<c10::List<at::Tensor>> tlist_args) {
    // 决定将输出张量移动到的设备
    // 当前约定是使用第一个张量参数选择设备
    if (!t_args.empty()) {
        return t_args[0].device();
    } else {
        // 遍历所有的（可能有多个）张量列表参数
        // 例如，如果第一个为空但第二个不为空。
        for (auto& tens_list : tlist_args) {
            for (const auto i : c10::irange(tens_list.size())) {
                return tens_list.get(i).device();
            }
        }
    }
    // 返回空设备选项
    return c10::nullopt;
}

// 验证张量列表是否至少有一个已定义的张量
static bool validate_tensor_list(const c10::List<at::Tensor>& tensorlist) {
    bool flag = false;

    // 遍历张量列表，检查是否有已定义的张量
    for (const auto& i : c10::irange(tensorlist.size())) {
        if (tensorlist[i].defined())
            flag = true;
    }

    // 返回标志指示是否有已定义的张量
    return flag;
}
// 处理 CPU 回退的函数，将非 CPU 张量输入转换为 CPU 张量，并将它们放置在正确的索引位置上。
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool error_on_views) {
  // 获取操作符的参数模式
  auto& schema_args = op.schema().arguments();
  // 参数的数量
  const auto num_arguments = schema_args.size();
  // 获取最后 num_arguments 个参数
  auto arguments = torch::jit::last(stack, num_arguments);
  // 参数起始位置
  const auto arguments_begin = stack->size() - num_arguments;

  // 用于存储张量类型的参数
  std::vector<at::Tensor> tensor_args;
  // 用于存储张量类型参数的索引
  std::vector<int> tensor_args_indices;

  // 用于存储张量列表类型的参数
  std::vector<c10::List<at::Tensor>> tensorlist_args;
  // 用于存储张量列表类型参数的索引
  std::vector<int> tensorlist_args_indices;

  // 目标设备，初始值为空
  std::optional<c10::Device> tgt_device = c10::nullopt;
  // 保存转换后的 CPU 张量，用于 TensorList
  std::vector<c10::IValue> tensorlist_cpu_args;

  // Step 1: Convert all non-CPU tensor inputs into CPU tensors
  // and put them on the stack at the correct indices.
  // 步骤1：将所有非 CPU 张量输入转换为 CPU 张量，并将它们放置在正确的索引位置上。
  for (const auto idx : c10::irange(arguments.size())) {
    const auto& ivalue = arguments[idx];
    // 如果是张量类型
    if (ivalue.isTensor()) {
      tensor_args.push_back(ivalue.toTensor());
      tensor_args_indices.push_back(idx);
    }
    // 如果是张量列表类型
    else if (ivalue.isTensorList()) {
      // 注意：为了方便起见，我们逐个将每个张量列表参数复制到 CPU，但 XLA 可以通过一次性在 CPU 上材料化所有张量和张量列表参数来受益。
      tensorlist_args.push_back(ivalue.toTensorList());
      tensorlist_args_indices.push_back(idx);
      // 将每个张量列表参数转换为 CPU，并存储在 tensorlist_cpu_args 中
      auto cpu_ivalue = c10::IValue(c10::List<at::Tensor>(to_cpu(ivalue.toTensorList().vec())));
      tensorlist_cpu_args.push_back(cpu_ivalue);
      // 在栈上的正确索引位置设置 CPU 张量
      (*stack)[arguments_begin + idx] = std::move(cpu_ivalue);
    }
    // 如果是可选张量列表类型
    else if (ivalue.isOptionalTensorList()) {
      auto opt_tensors = ivalue.toOptionalTensorList().vec();
      std::vector<at::Tensor> need_convert_tensors;
      std::vector<int> need_convert_tensors_index;
      // 遍历处理每个张量
      for (auto i : c10::irange(opt_tensors.size())) {
        if (!opt_tensors[i].has_value() || !opt_tensors[i]->defined()) continue;
        need_convert_tensors.push_back(opt_tensors[i].value());
        need_convert_tensors_index.push_back(i);
      }
      // 转换为 CPU 张量
      auto cpu_tensors = to_cpu(need_convert_tensors);
      // 将转换后的 CPU 张量放回原始位置
      for (const auto i : c10::irange(need_convert_tensors_index.size())) {
        auto idx = need_convert_tensors_index[i];
        opt_tensors[idx] = cpu_tensors[i];
      }
      // 更新栈上的参数为转换后的张量列表
      (*stack)[arguments_begin + idx] = c10::IValue(opt_tensors);
    }
    // 如果是设备类型
    else if (ivalue.isDevice()) {
      tgt_device = ivalue.toDevice();
      // 将栈上的参数设为 CPU 设备
      (*stack)[arguments_begin + idx] = c10::IValue(c10::Device(kCPU));
    }
  }
  // XLA 要求将所有张量参数收集起来，并一起转换为 CPU。
  auto cpu_tensors = to_cpu(tensor_args);

  // 将转换后的 CPU 张量放回原始位置
  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto idx = tensor_args_indices[i];
    // 继续处理 CPU 张量的后续步骤
    // (未提供完整的代码，根据上下文推断)
  // 将 CPU tensors 转换为 IValue 并存入堆栈中的指定位置
  (*stack)[arguments_begin + idx] = c10::IValue(cpu_tensors[i]);

  // 步骤 2: 调用操作符的 CPU 实现
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);

  // 步骤 3: 需要特别处理可变别名：
  // 如果任何输入张量是可变别名，则需要将更新后的数据直接复制回原始输入的 CPU 张量。
  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto tensor_idx = tensor_args_indices[i];
    const AliasInfo* alias_info = schema_args[tensor_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      at::_copy_from_and_resize(cpu_tensors[i], tensor_args[i]);
    }
  }

  // 还需要显式地重新应用对张量列表输入的输入突变
  for (const auto i : c10::irange(tensorlist_args_indices.size())) {
    auto tensorlist_idx = tensorlist_args_indices[i];
    const AliasInfo* alias_info = schema_args[tensorlist_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      const auto& cpu_tensors = tensorlist_cpu_args[i].toTensorList().vec();
      for (const auto idx : c10::irange(tensorlist_args[i].size())) {
        at::_copy_from_and_resize(cpu_tensors[idx], tensorlist_args[i][idx]);
      }
    }
  }

  // 步骤 4: 将任何 CPU 输出张量转换回原始输入设备。
  // 对于可变别名的输出，我们还需要特别注意将原始输入张量移回堆栈中，代替我们创建的临时 CPU 输出张量。
  //
  // 注解 [CPU Fallback Does Not Handle View Operators]
  // 还需要注意，我们无法正确处理不可变别名。
  // 为什么？
  // 具有不可变别名的张量输出对应于视图操作符。
  // 例如，native_functions.yaml 中的 `view_as` 模式：
  // `view_as(Tensor(a) self, Tensor other) -> Tensor(a)`
  // 我们无法正确处理这些操作，因为视图操作应返回一个与原始张量共享相同存储的新张量。
  // 但我们创建的新张量无法共享相同的存储，因为它位于 CPU 上，而原始张量位于不同的设备上。
  // 因此，如果尝试在视图操作符上调用 CPU 回退，我们会发出警告（这是为了保持 XLA 对视图操作符的 BC 兼容性）。
  const auto& schema_returns = op.schema().returns();
  const auto& num_returns = schema_returns.size();
  auto returns = torch::jit::last(stack, num_returns);
  const auto returns_begin = stack->size() - num_returns;

  // 如果目标设备未指定，则计算目标设备
  if (tgt_device == c10::nullopt) {
    tgt_device = compute_target_device(tensor_args, tensorlist_args);
  }

  // 遍历返回值，并处理别名信息
  for (const auto idx : c10::irange(returns.size())) {
    const AliasInfo* alias_info = schema_returns[idx].alias_info();
    // 检查 alias_info 不为空且是可写的情况
    if (alias_info != nullptr && alias_info->isWrite()) {
      // 情况（1）：可变别名情况。
      // 将输入的 ivalue 直接移到堆栈中，替换现有的 CPU 输出张量。
      bool found_alias = false;
      // 如果返回值是张量并且已定义
      if (returns[idx].isTensor() && returns[idx].toTensor().defined()) {
        // 如果需要提高性能，可以在函数模式上存储一些额外的元数据以避免此处的循环。
        // 遍历张量参数索引的范围
        for (const auto i : c10::irange(tensor_args_indices.size())) {
          auto input_tensor_idx = tensor_args_indices[i];
          const auto& input_tensor = cpu_tensors[i];
          // 获取输入张量的别名信息
          const AliasInfo* input_alias_info =
              schema_args[input_tensor_idx].alias_info();
          // 在上面已检查；添加断言以防止由于更改上面的 if 测试而破坏以下条件。
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alias_info != nullptr);
          // 如果输入张量已定义且与当前输出别名相同，或者其别名信息相同
          if (input_tensor.defined() &&
              (alias_info == input_alias_info ||
               (input_alias_info != nullptr &&
                *alias_info == *input_alias_info))) {
            // 找到了与当前输出别名相对应的原始输入张量。将其包装为 IValue 并直接放在堆栈上。
            (*stack)[returns_begin + idx] = c10::IValue(tensor_args[i]);
            found_alias = true;
            break;
          }
        }
      } else if (
          // 如果返回值是张量列表，并且验证张量列表有效
          returns[idx].isTensorList() &&
          validate_tensor_list(returns[idx].toTensorList())) {
        // 遍历张量列表参数索引的范围
        for (const auto i : c10::irange(tensorlist_args_indices.size())) {
          auto input_tensor_idx = tensorlist_args_indices[i];
          // 获取输入张量列表的别名信息
          const AliasInfo* input_alias_info =
              schema_args[input_tensor_idx].alias_info();
          // 在上面已检查；添加断言以防止由于更改上面的 if 测试而破坏以下条件。
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alias_info != nullptr);
          // 如果验证张量列表有效，并且当前输出别名与输入别名相同，或者其别名信息相同
          if (validate_tensor_list(tensorlist_args[i]) &&
              (alias_info == input_alias_info ||
               (input_alias_info != nullptr &&
                *alias_info == *input_alias_info))) {
            // 找到了与当前输出别名相对应的原始输入张量。将其包装为 IValue 并直接放在堆栈上。
            (*stack)[returns_begin + idx] = c10::IValue(tensorlist_args[i]);
            found_alias = true;
            break;
          }
        }
      }
      // 检查是否找到了别名，否则抛出错误
      TORCH_CHECK(
          found_alias,
          "The operator ",
          op.schema().operator_name(),
          " appears to have invalid alias information. ",
          "Found a return tensor argument with a mismatched mutable alias: ",
          schema_returns[idx]);
    } else {
      // 如果不是 Case (1) 的情况，即非直接返回视图或拷贝操作时执行以下代码块

      if (alias_info != nullptr && !alias_info->isWrite()) {
        // 如果存在不可写的别名信息，即不可变别名（视图）情况
        // 在这里发出警告，因为我们正在进行拷贝操作而非创建视图。
        // 如果需要此操作符，后端应提供相应的内核。
        // 参见注释 [CPU Fallback Does Not Handle View Operators]
        std::stringstream dev_str;
        if (tgt_device) {
          dev_str << *tgt_device;
        } else {
          dev_str << "<none>";
        }
        
        if (error_on_views) {
          // 如果视图错误标志为真，抛出错误
          TORCH_CHECK(
              false,
              "The operator ",
              op.schema().operator_name(),
              " appears to be a view operator, ",
              "but it has no implementation for the backend \"",
              dev_str.str(),
              "\". View operators don't support ",
              "since the tensor's storage cannot be shared across devices.");
        } else {
          // 否则，发出警告
          TORCH_WARN(
              false,
              "The operator ",
              op.schema().operator_name(),
              " appears to be a view operator, ",
              "but it has no implementation for the backend \"",
              dev_str.str(),
              "\". View operators don't support falling back to run on the CPU, ",
              "since the tensor's storage cannot be shared across devices.");
        }
      }

      // Case (2): copy case.
      // 情况 (2): 拷贝操作
      // 将 CPU 输出张量复制到原始设备

      // 技术上可能不存在目标设备，例如使用空列表调用 torch.cat()
      // 在这种情况下，我们不应该有任何需要在设备间传递的张量。
      if (tgt_device) {
        if (returns[idx].isTensor() && returns[idx].toTensor().defined()) {
          // 如果返回值是张量且已定义
          (*stack)[returns_begin + idx] =
              c10::IValue(returns[idx].toTensor().to(*tgt_device));
        } else if (
            returns[idx].isTensorList() &&
            validate_tensor_list(returns[idx].toTensorList())) {
          // 如果返回值是张量列表且通过验证
          const auto& cpu_tensors = returns[idx].toTensorList().vec();
          std::vector<at::Tensor> tensors;
          tensors.reserve(cpu_tensors.size());

          // 将所有 CPU 张量转换到目标设备
          for (const auto& tensor : cpu_tensors) {
            tensors.push_back(tensor.to(*tgt_device));
          }
          (*stack)[returns_begin + idx] =
              c10::IValue(c10::List<at::Tensor>(tensors));
        }
      }
    }
  }
}

// 这段代码是在C++中结束一个命名空间`at::native`。
// 命名空间用于将全局作用域划分为更小的区域，以避免名称冲突并组织代码。
// 在此处，结束命名空间`at::native`的定义。
```