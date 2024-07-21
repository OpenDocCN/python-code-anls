# `.\pytorch\torch\csrc\jit\mobile\compatibility\runtime_compatibility.cpp`

```
/*
 * 包含必要的头文件
 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/type_factory.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/compatibility/runtime_compatibility.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/custom_class.h>
#include <unordered_map>

/*
 * 定义命名空间 c10 和 torch::jit
 */
namespace c10 {
    TypePtr parseType(const std::string& pythonStr);
} // namespace c10

namespace torch {
namespace jit {

/*
 * 返回当前支持的最大字节码版本
 */
uint64_t _get_runtime_bytecode_version() {
  return caffe2::serialize::kMaxSupportedBytecodeVersion;
}

/*
 * 返回运行时支持的字节码版本范围
 */
std::pair<uint64_t, uint64_t> _get_runtime_bytecode_min_max_versions() {
  return std::pair<uint64_t, uint64_t>(
      caffe2::serialize::kMinSupportedBytecodeVersion,
      caffe2::serialize::kMaxSupportedBytecodeVersion);
}

/*
 * 返回运行时支持的运算符文件格式版本范围
 */
std::pair<uint64_t, uint64_t> _get_runtime_operators_min_max_versions() {
  return std::pair<uint64_t, uint64_t>(
      caffe2::serialize::kMinSupportedFileFormatVersion,
      caffe2::serialize::kMaxSupportedFileFormatVersion);
}

/*
 * 返回所有注册的 PyTorch 运算符及其版本信息
 */
std::unordered_map<std::string, OperatorInfo> _get_runtime_ops_and_info() {
  std::unordered_map<std::string, OperatorInfo> result;

  // 获取所有非调度运算符
  auto nonDispatcherOperators = torch::jit::getAllOperators();
  for (const auto& full_op : nonDispatcherOperators) {
    auto op = full_op->schema();
    int num_schema_args = op.arguments().size();
    auto op_name = op.name();
    if (!op.overload_name().empty()) {
      op_name += ("." + op.overload_name());
    }
    result.emplace(op_name, OperatorInfo{num_schema_args});
  }

  // 获取所有调度运算符
  auto dispatcherOperators = c10::Dispatcher::singleton().getAllOpNames();
  for (auto& op : dispatcherOperators) {
    // 获取运算符模式
    const auto op_handle = c10::Dispatcher::singleton().findOp(op);
    std::optional<int> num_schema_args;
    if (op_handle->hasSchema()) {
      num_schema_args = op_handle->schema().arguments().size();
    }
    auto op_name = op.name;
    if (!op.overload_name.empty()) {
      op_name += ("." + op.overload_name);
    }
    result.emplace(op_name, OperatorInfo{num_schema_args});
  }

  return result;
}

/*
 * 返回运行时兼容性信息
 */
RuntimeCompatibilityInfo RuntimeCompatibilityInfo::get() {
  return RuntimeCompatibilityInfo{
      _get_runtime_bytecode_min_max_versions(),
      _get_runtime_ops_and_info(),
      _get_mobile_supported_types(),
      _get_runtime_operators_min_max_versions()};
}

/*
 * 返回移动设备支持的类型集合
 */
std::unordered_set<std::string> _get_mobile_supported_types() {
  std::unordered_set<std::string> supported_types;
  for (const auto& it : c10::DynamicTypeFactory::basePythonTypes()) {
    supported_types.insert(it.first);
  }
  supported_types.insert(
      at::TypeParser::getNonSimpleType().begin(),
      at::TypeParser::getNonSimpleType().end());
  supported_types.insert(
      at::TypeParser::getCustomType().begin(),
      at::TypeParser::getCustomType().end());

  return supported_types;
}

} // namespace jit
} // namespace torch
# 定义一个函数 `_get_loaded_custom_classes()`，返回一个无序集合，集合元素类型为字符串
TORCH_API std::unordered_set<std::string> _get_loaded_custom_classes() {
    # 调用 Torch 库函数 `torch::getAllCustomClassesNames()`，返回所有已加载的自定义类名集合
    return torch::getAllCustomClassesNames();
}

# 命名空间闭合：jit
} // namespace jit

# 命名空间闭合：torch
} // namespace torch
```