# `.\pytorch\torch\csrc\lazy\ts_backend\ts_eager_fallback.cpp`

```
// 引入 Torch 和 ATen 库的相关头文件

#include <torch/csrc/lazy/ts_backend/ts_eager_fallback.h>

#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Functions.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/native/CPUFallback.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/library.h>

// 引入标准库头文件
#include <sstream>
#include <unordered_map>

// Torch 命名空间
namespace torch {
// Torch Lazy 模块命名空间
namespace lazy {
namespace {

// 将一组 Tensor 转换为 eager 模式的函数
std::vector<at::Tensor> _to_eager(
    at::TensorList tensors,           // 输入的 Tensor 列表
    c10::DeviceType device_type) {    // 设备类型参数
  switch (device_type) {
    case at::kCPU: {                 // 如果设备类型是 CPU
      return at::_to_cpu(tensors);   // 调用 ATen 的 _to_cpu 函数将 Tensor 转换为 CPU 模式
    }
    default: {                       // 其他设备类型
      std::vector<at::Tensor> eager_tensors;  // 创建一个存放 eager Tensor 的向量
      for (const auto& t : tensors) {
        c10::TensorOptions options = t.options().device(device_type);  // 获取 Tensor 的选项并设置设备类型
        at::Tensor eager_tensor = t.to(
            options,                // 设置的选项
            /*non_blocking*/ false, // 是否非阻塞
            /*copy*/ false);        // 是否拷贝数据
        eager_tensors.push_back(eager_tensor);  // 将转换后的 Tensor 放入结果向量
      }
      return eager_tensors;          // 返回所有转换后的 eager Tensor
    }
  }
}

// 将一组 Tensor 转换为 eager 模式的便捷函数
std::vector<at::Tensor> to_eager(
    const at::TensorList& tensors,   // 输入的 Tensor 列表
    c10::DeviceType device_type) {   // 设备类型参数
  // 首先分离未定义的 Tensor，防止 _to_eager 处理时出错
  std::vector<at::Tensor> eager_tensors(tensors.size());  // 创建一个与输入大小相同的 eager Tensor 向量
  std::vector<at::Tensor> valid_tensors;  // 存放有效 Tensor 的向量
  std::vector<bool> to_translate(tensors.size());  // 标记需要转换的 Tensor
  for (size_t i = 0; i < tensors.size(); ++i) {
    const at::Tensor& tensor = tensors[i];
    if (tensor.defined()) {         // 如果 Tensor 是有效定义的
      to_translate[i] = true;       // 标记需要转换
      valid_tensors.push_back(tensor);  // 放入有效 Tensor 向量
    } else {                        // 如果 Tensor 未定义
      eager_tensors[i] = tensor;    // 直接放入结果向量
    }
  }
  auto eager_valid_tensors = _to_eager(valid_tensors, device_type);  // 将有效 Tensor 转换为 eager 模式
  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (to_translate[i]) {          // 如果需要转换当前 Tensor
      eager_tensors[i] = std::move(eager_valid_tensors[defined_pos++]);  // 将转换后的 Tensor 放入结果向量
    }
  }
  return eager_tensors;             // 返回所有转换后的 eager Tensor
}

// 将一组可能为空的 Tensor 转换为 eager 模式的函数
std::vector<std::optional<at::Tensor>> to_eager(
    const std::vector<std::optional<at::Tensor>>& tensors,  // 输入的可能为空的 Tensor 列表
    c10::DeviceType device_type) {                          // 设备类型参数
  std::vector<std::optional<at::Tensor>> eager_tensors(tensors.size());  // 创建一个与输入大小相同的 optional eager Tensor 向量
  std::vector<at::Tensor> valid_tensors;  // 存放有效 Tensor 的向量
  std::vector<bool> to_translate(tensors.size());  // 标记需要转换的 Tensor
  for (size_t i = 0; i < tensors.size(); ++i) {
    const std::optional<at::Tensor>& tensor = tensors[i];
    if (tensor.has_value()) {       // 如果 Tensor 有值
      to_translate[i] = true;       // 标记需要转换
      valid_tensors.push_back(*tensor);  // 放入有效 Tensor 向量
    } else {                        // 如果 Tensor 为空
      eager_tensors[i] = tensor;    // 直接放入结果向量
    }
  }
  auto eager_valid_tensors = _to_eager(valid_tensors, device_type);  // 将有效 Tensor 转换为 eager 模式
  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (to_translate[i]) {          // 如果需要转换当前 Tensor
      eager_tensors[i] = std::move(eager_valid_tensors[defined_pos++]);  // 将转换后的 Tensor 放入结果向量
    }
  }
  return eager_tensors;             // 返回所有转换后的 eager Tensor
}

} // namespace
} // namespace lazy
} // namespace torch
  // 检查张量是否有值且已定义，如果是，则标记为需要转换，并添加到有效张量列表中
  if (tensor.has_value() && tensor->defined()) {
    to_translate[i] = true;  // 标记索引 i 需要进行转换
    valid_tensors.push_back(*tensor);  // 将有效的张量添加到有效张量列表中
  } else {
    eager_tensors[i] = tensor;  // 否则将张量存储在急切张量数组中
  }
}
// 将有效张量转换为急切模式，并指定设备类型
auto eager_valid_tensors = _to_eager(valid_tensors, device_type);
// 根据标记进行张量的转换
for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
  if (to_translate[i]) {
    eager_tensors[i] = std::move(eager_valid_tensors[defined_pos++]);  // 移动已转换的急切张量到相应位置
  }
}
// 返回包含急切张量的数组
return eager_tensors;
}

// 根据设备类型返回对应的调度键
c10::DispatchKey dispatch_key(c10::DeviceType device_type) {
  switch (device_type) {
    case at::kCPU: {
      return c10::DispatchKey::CPU;  // 返回 CPU 调度键
    }
    case at::kCUDA: {
      return c10::DispatchKey::CUDA;  // 返回 CUDA 调度键
    }
    default: {
      AT_ERROR("Unsupported device type: ", device_type);  // 抛出错误，不支持的设备类型
    }
  }
}

// 计算目标设备，决定将输出张量移动到哪个设备
std::optional<c10::Device> compute_target_device(
    std::vector<at::Tensor>& t_args,
    std::vector<c10::List<at::Tensor>> tlist_args,
    std::vector<c10::List<std::optional<at::Tensor>>> opt_tlist_args) {
  if (!t_args.empty()) {
    return t_args[0].device();  // 如果有张量参数，返回第一个张量的设备
  } else {
    // 遍历 TensorList 参数，找到第一个非空张量并返回其设备
    for (auto& tens_list : tlist_args) {
      for (const auto i : c10::irange(tens_list.size())) {
        return tens_list.get(i).device();
      }
    }
    // 遍历包含可选张量的 TensorList 参数，找到第一个有值的张量并返回其设备
    for (auto& tens_list : opt_tlist_args) {
      for (const auto i : c10::irange(tens_list.size())) {
        if (tens_list.get(i).has_value()) {
          return tens_list.get(i)->device();
        }
      }
    }
  }
  return c10::nullopt;  // 如果没有有效的设备，返回空的 std::optional
}

} // namespace

// 静态全局变量，用于存储惰性回退计数器的映射
static std::unordered_map<std::string, ::torch::lazy::Counter*> _eager_fallback_counters;

// 检查是否强制使用惰性回退的宏
bool force_eager_fallback(c10::Symbol op) {
  auto force_str = getLTCForceFallback();
  if (!force_str.empty()) {
    static auto force_sym = c10::Symbol::fromQualString(std::string(force_str));
    if (op == force_sym) {
      return true;  // 如果操作符与强制符号匹配，返回 true
    }
  }
  if (op == at::aten::nonzero) {
    // 当符号形状模式未启用时，非零形状函数返回不正确的结果
    return !symbolicShapeEnabled();  // 如果符号形状未启用，返回 true
  }

  return false;  // 默认返回 false
}

// 处理惰性回退的函数
void ltc_eager_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  // TODO(whc) this FN_TRACK thing hasn't been used so far in LTC iirc but could
  // land/re-enable it LTC_FN_TRACK(3);;
  const auto name = c10::toString(op.operator_name());

  // 手动应用 TORCH_LAZY_COUNTER 宏
  // 需要显式地保持计数器映射，因为这个函数被多个操作符使用
  // 宏在调用时在代码位置创建一个具有固定名称的静态计数器对象
  if (_eager_fallback_counters.find(name) == _eager_fallback_counters.end()) {
    _eager_fallback_counters[name] = new ::torch::lazy::Counter(name);
  }
  _eager_fallback_counters[name]->AddValue(1);  // 计数器加一

  auto& args = op.schema().arguments();
  auto arguments = torch::jit::last(stack, args.size());

  // 记录每个张量参数
  for (const auto& ivalue : arguments) {
    if (ivalue.isTensor()) {
      VLOG(3) << ivalue.toTensor().toString();  // 输出张量的字符串表示
    }
  }



// 结束了一个函数或代码块的声明和定义

  // Call the actual boxed CPU fallback.
  // 调用实际的 CPU 回退处理函数。
  ts_eager_fallback(
      op, stack, torch::lazy::getBackend()->EagerFallbackDeviceType());


这些注释解释了代码块的结束以及一行调用函数的作用，帮助读者理解它们的功能和意图。
}

// 注册 TorchScript 的延迟回退函数
void register_ts_ltc_eager_fallback() {
  // 创建静态局部变量 m，使用 TORCH_LIBRARY_IMPL 宏注册到 Lazy 库中
  static auto m = MAKE_TORCH_LIBRARY_IMPL(_, Lazy);

  // 大多数后端使用 TORCH_LIBRARY_* 宏在静态库初始化时注册它们的分发函数，
  // 但是延迟 TorchScript 后端不这样做，因为它是在主 torch 库中构建的，但并非总是使用。
  // 特别地，如果另一个外部后端想要向相同的键（Lazy）注册自己，TorchScript 后端就不应该初始化。
  // 注册 torch::CppFunction 到 m.fallback，使用 ltc_eager_fallback 函数的包装
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&ltc_eager_fallback>());
}

// TorchScript 的急切回退函数
void ts_eager_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    c10::DeviceType device_type) {
  // 获取操作符的参数 schema
  auto& schema_args = op.schema().arguments();
  const auto num_arguments = schema_args.size();
  // 从栈中获取最后 num_arguments 个参数
  auto arguments = torch::jit::last(stack, num_arguments);
  const auto arguments_begin = stack->size() - num_arguments;

  // 用于存储各种类型的参数
  std::vector<at::Tensor> tensor_args;
  std::vector<int> tensor_args_indices;

  std::vector<c10::List<at::Tensor>> tensorlist_args;
  std::vector<c10::List<std::optional<at::Tensor>>> opt_tensorlist_args;

  // 步骤 1: 将所有非急切张量输入转换为急切张量，并将它们放置在正确的索引位置上栈中
  for (size_t idx = 0; idx < arguments.size(); ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      // 如果是张量，则将其添加到 tensor_args 并记录索引
      tensor_args.push_back(ivalue.toTensor());
      tensor_args_indices.push_back(idx);
    } else if (ivalue.isTensorList()) {
      // 如果是张量列表，则将其转换为急切张量并更新栈中对应位置的值，并将原始的 tensorlist_args 添加到容器中
      auto eager_ivalue = c10::IValue(c10::List<at::Tensor>(
          to_eager(ivalue.toTensorVector(), device_type)));
      (*stack)[arguments_begin + idx] = std::move(eager_ivalue);
      tensorlist_args.push_back(ivalue.toTensorList());
    } else if (ivalue.isOptionalTensorList()) {
      // 如果是可选的张量列表，则将其转换为急切张量并更新栈中对应位置的值，并将原始的 opt_tensorlist_args 添加到容器中
      auto eager_ivalue = c10::IValue(c10::List<std::optional<at::Tensor>>(
          to_eager(ivalue.toOptionalTensorVector(), device_type)));
      (*stack)[arguments_begin + idx] = std::move(eager_ivalue);
      opt_tensorlist_args.push_back(ivalue.toOptionalTensorList());
    }
  }

  // XLA 需要将所有张量参数收集起来并一起转换为 CPU 张量
  auto eager_tensors = to_eager(tensor_args, device_type);

  // 将急切张量放回栈中原来的索引位置
  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto idx = tensor_args_indices[i];
    (*stack)[arguments_begin + idx] = std::move(eager_tensors[i]);
  }
}
  (*stack)[arguments_begin + idx] = c10::IValue(eager_tensors[i]);
  // 将 eager_tensors[i] 转换为 c10::IValue，并存储在 stack 中的正确位置

  // Step 2: Call the underlying eager implementation of the operator
  // 调用操作符的基础 eager 实现
  op.redispatchBoxed(c10::DispatchKeySet(dispatch_key(device_type)), stack);

  // Step 3: We need to take special care to handle mutable aliases properly:
  // 如果任何输入张量是可变别名，则需要正确处理：
  // 如果有可变别名，需要将更新后的 eager tensors 直接复制回原始输入张量。
  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto tensor_idx = tensor_args_indices[i];
    const auto alias_info = schema_args[tensor_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      at::_copy_from_and_resize(eager_tensors[i], tensor_args[i]);
      // 调用 _copy_from_and_resize 函数，将 eager_tensors[i] 复制并调整大小到 tensor_args[i]
    }
  }

  // Step 4: Convert any eager output tensors back to the original input device.
  // For mutable alias'd outputs, we also need to take special care
  // to move the ORIGINAL input tensor back onto the stack, in place of
  // the temporary eager output tensor that we created.
  //
  // Note [Eager Fallback Does Not Handle View Operators]
  // Also note that we are incapable of handling immutable alises properly.
  // Why?
  // Schemas with an immutable alias'd tensor outputs correspond to view
  // operators. For example, the `view_as` schema from native_functions.yaml:
  // `view_as(Tensor(a) self, Tensor other) -> Tensor(a)`
  // We can't handle these ops properly, because view ops are supposed to return
  // a NEW tensor that shares the SAME storage as the original tensor.
  // However, the new tensor that we created cannot share the same storage,
  // since it lives on the eager CPU / CUDA device and the original tensor lives
  // on a different device. Because of that, we warn if someone attempts to call
  // the eager fallback on a view operator (this is to maintain BC for view ops
  // for XLA that fall back to CPU).
  // 将任何 eager 输出张量转换回原始输入设备。
  // 对于可变别名的输出，还需要特别注意将原始输入张量移到栈上，
  // 替换我们创建的临时 eager 输出张量。
  const auto& schema_returns = op.schema().returns();
  const auto& num_returns = schema_returns.size();
  auto returns = torch::jit::last(stack, num_returns);
  const auto returns_begin = stack->size() - num_returns;

  for (const auto idx : c10::irange(returns.size())) {
    // 循环处理返回的每个元素
  }
} // 结束 torch 命名空间

} // 结束 lazy 命名空间
} // 结束 torch 命名空间
```