# `.\pytorch\aten\src\ATen\native\MathBitsFallback.h`

```py
// 包含 ATen 核心库中的头文件
#include <ATen/core/Tensor.h>
// 包含 ATen 核心调度器的头文件
#include <ATen/core/dispatch/Dispatcher.h>
// 包含 ATen 操作注册的头文件
#include <ATen/core/op_registration/op_registration.h>
// 包含 ATen 本地库的一元操作的头文件
#include <ATen/native/UnaryOps.h>
// 包含 ATen 本地库的调整大小操作的头文件
#include <ATen/native/Resize.h>
// 包含 c10 实用工具的范围库的头文件
#include <c10/util/irange.h>
// 包含 Torch 库的头文件
#include <torch/library.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含 ATen 函数库的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 否则，包含 ATen 克隆操作的头文件和一些实用工具的头文件
#else
#include <ATen/ops/clone.h>
#include <utility>
#endif

// 定义 at::native 命名空间
namespace at::native {

// 这是一个用于处理数学操作回退的结构体
// 该回退仅应用于自反操作，并且具有对应的张量位（使用 DispatchKey 内部实现）
struct MathOpFallback {
  // 构造函数，初始化操作的 DispatchKey 和操作名称
  MathOpFallback(DispatchKey key_, string op_name_) : key(key_), op_name(std::move(op_name_)) {}
  
  // 纯虚函数，用于检查张量是否设置了特定的数学位
  virtual bool is_bit_set(const Tensor&) = 0;
  
  // 实现回退的具体功能，根据操作句柄、DispatchKey 集合和 Torch 堆栈
  void fallback_impl(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
    /*
      处理的情况：
        1. 非就地操作。简单地将所有输入材料化然后调用。
        2. 就地操作。将 x.add_(2) 转换为 x.conj_().add_(2).conj_()。其他输入材料化如 (1)。
        3. out= 操作。将 add(x, 2, out=y) 转换为 y.copy_(add(x, 2))。其他输入材料化如 (1)。
        
        能够分辨我们是否从参数中读取以及是否向参数写入是很重要的。
        保守的方法是假定我们总是从参数中读取，但在 out= 操作中可以跳过从未使用的输入的共轭化。
        在当前模式下，我们无法轻易分辨操作是就地操作还是 out= 操作。
        
        注意：
        1. 不允许包含数学位设置为 true 的可变张量列表。
        2. 无条件克隆数学位设置为 true 的可变张量，以确保在可变张量与非可变参数共享内存的情况下正确行为。
           如果我们要就地解析可变输入的数学位，则在以下情况下，共享部分或全部内存的非可变输入将读入错误的值：
             1. 非可变输入其数学位设置为 false。
             2. 在克隆非可变输入（其数学位设置为 true 并与一个或多个可变参数共享内存）之前，可变输入的数学位已解析。
           最终，栈中的可变参数的最终值被复制到原始输入的可变张量输入中。
    */
    // 获取操作的参数列表
    const auto& arguments = op.schema().arguments();
    // 获取参数的数量
    const auto num_arguments = arguments.size();
    // 计算堆栈中参数起始位置
    const auto stack_start = stack->size() - num_arguments;

    // 初始化是否写操作的可选值
    std::optional<bool> is_write;
    // 遍历参数列表
    for (const auto i : c10::irange(num_arguments)) {
      // 获取当前参数的别名信息
      const AliasInfo* alias_info = arguments[i].alias_info();
      // 如果别名信息不为 nullptr
      if (alias_info != nullptr) {
        // 如果 is_write 已经有值
        if (is_write.has_value()) {
          // 检查当前参数的写操作状态与之前参数的写操作状态是否一致
          TORCH_CHECK(*is_write == alias_info->isWrite(),
            "Unsupported operator for ", op_name, " fallback: ", op.schema().name(),
            op_name, " fallback doesn't work for operators with a mix "
            "mutable and non-mutable inputs that alias with outputs, "
            "this must be implemented manually.  "
            "If you got this error on a core op, please report a bug to PyTorch.");
        } else {
          // 设置当前参数的写操作状态
          is_write = alias_info->isWrite();
        }
      }
    }

    // 如果 is_write 有值且为 false
    if (is_write.has_value() && !*is_write) {
      // 假设视图操作可以通过传播 dispatch key 在 key_set 中正确处理数学位
      // 这并不总是正确的，因此应该测试这些情况。
      // 重新调度操作，并返回
      op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, key), stack);
      return;
    }

    // 可变输入和它们的克隆，存储为向量对
    std::vector<std::pair<Tensor, Tensor>> mutable_inputs_with_their_clones;
    // 遍历操作数的范围，处理每个操作数
    for (const auto i : c10::irange(num_arguments)) {
      // 获取当前操作数的引用
      auto& ivalue = (*stack)[stack_start + i];
      // 如果操作数不是张量或张量列表，则跳过处理
      if (!(ivalue.isTensor() || ivalue.isTensorList())) {
        continue;
      }
      // 获取当前操作数对应的参数信息
      const auto& argument = arguments[i];
      // 标记是否为可变参数
      bool mut_arg = false;
      // 如果参数有别名信息
      if (argument.alias_info()) {
        // 断言参数别名信息表明其是写操作
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(argument.alias_info()->isWrite());
        mut_arg = true;
      }
      // 如果操作数是张量
      if (ivalue.isTensor()) {
        // 如果张量的位被设置为假，则跳过处理
        if (!is_bit_set(ivalue.toTensor())) {
          continue;
        }
        // 移动当前操作数的张量值，并复制生成解析后的张量
        auto tensor = std::move(ivalue).toTensor();
        auto resolved_tensor = at::clone(tensor);
        // 如果是可变参数，确保只有一个可变参数与其克隆值的映射
        if (mut_arg) {
          TORCH_CHECK(mutable_inputs_with_their_clones.empty(), op_name, " fallback does not support operators with more than one mutable tensors with ",
            op_name, "bit set to true.");
          mutable_inputs_with_their_clones.emplace_back(std::move(tensor), resolved_tensor);
        }
        // 将解析后的张量值设置回操作数栈
        (*stack)[stack_start + i] = std::move(resolved_tensor);
      } else if (ivalue.isTensorList()) {
        // 如果操作数是张量列表
        auto tensors = std::move(ivalue).toTensorList();
        // 遍历张量列表中的每一个张量
        for(const auto j : c10::irange(tensors.size())) {
          const auto& tensor = tensors[j];
          // 如果张量的位被设置为假，则跳过处理
          if (!is_bit_set(tensor)) {
            continue;
          }
          // 如果是可变参数，则不支持当前情况，抛出错误
          TORCH_CHECK(!mut_arg, " fallback doesn't currently support mutable TensorLists with ",
              op_name, " inputs. Please materialize all the ", op_name, " input tensor(s) in the mutable TensorList inputs before calling ",
              op.schema().name());
          // 复制并替换当前张量的克隆值
          tensors[j] = at::clone(tensor);
        }
        // 将更新后的张量列表设置回操作数栈
        (*stack)[stack_start + i] = std::move(tensors);
      }
    }

    // 重新分派包装后的操作
    op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, key), stack);

    // 断言确保只有一个或没有可变参数与其克隆值的映射
    TORCH_INTERNAL_ASSERT(mutable_inputs_with_their_clones.size() <= 1);

    // 遍历所有可变参数及其克隆值
    for (std::pair<Tensor, Tensor> mut_tensors: mutable_inputs_with_their_clones) {
      // 获取可变参数及其克隆值的引用
      auto& mutable_input =  mut_tensors.first;
      auto& cloned_mutable_input =  mut_tensors.second;
      // 获取当前操作数栈顶的值作为返回的输出张量
      auto& ivalue = (*stack)[stack_start];
      auto returned_output = std::move(ivalue).toTensor();

      // 断言以确保操作数栈顶的张量与其克隆值存在别名关系
      TORCH_INTERNAL_ASSERT(cloned_mutable_input.is_same(returned_output));

      // 调整输出张量的大小以匹配可变参数的尺寸
      at::native::resize_output(mutable_input, returned_output.sizes());

      // 将返回的输出张量复制到可变参数中
      mutable_input.copy_(returned_output);
      // 将更新后的可变参数设置回操作数栈顶
      (*stack)[stack_start] = std::move(mutable_input);
    }
  }

  // 虚析构函数，默认实现
  virtual ~MathOpFallback() = default;

  // 分派键和操作名称
  DispatchKey key;
  string op_name;
};

// 结束 at::native 命名空间定义
} // namespace at::native
```