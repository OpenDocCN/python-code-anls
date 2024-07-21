# `.\pytorch\torch\csrc\jit\mobile\compatibility\model_compatibility.h`

```py
#pragma once

#include <c10/macros/Export.h>
#include <torch/csrc/jit/mobile/compatibility/runtime_compatibility.h>

#include <istream>
#include <memory>
#include <unordered_map>

namespace caffe2 {
namespace serialize {
class PyTorchStreamReader;
class ReadAdapterInterface;
} // namespace serialize
} // namespace caffe2

namespace torch {
namespace jit {

// The family of methods below to get bytecode version from a model
// Throws if not passed in a well formed model
TORCH_API uint64_t _get_model_bytecode_version(std::istream& in);

// Overload to get bytecode version from a model file by filename
TORCH_API uint64_t _get_model_bytecode_version(const std::string& filename);

// Overload to get bytecode version from a model using a ReadAdapterInterface
TORCH_API uint64_t _get_model_bytecode_version(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

// Overload to get bytecode version from bytecode IValues
uint64_t _get_model_bytecode_version(
    const std::vector<c10::IValue>& bytecode_ivalues);

// The family of methods below to get the operator version from a model
// Throws if not passed in a well formed model
TORCH_API uint64_t _get_model_operator_version(std::istream& in);

// Overload to get operator version from a model file by filename
TORCH_API uint64_t _get_model_operator_version(const std::string& filename);

// Overload to get operator version from a model using a ReadAdapterInterface
TORCH_API uint64_t _get_model_operator_version(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

// Utility Functions

// Retrieve bytecode IValues from a PyTorchStreamReader
std::vector<c10::IValue> get_bytecode_ivalues(
    caffe2::serialize::PyTorchStreamReader& reader);

// Read an archive from a stream reader by name
c10::IValue readArchive(
    const std::string& archive_name,
    caffe2::serialize::PyTorchStreamReader& stream_reader);

// Check if the provided ReadAdapterInterface represents a ZIP file
bool check_zip_file(
    const std::shared_ptr<caffe2::serialize::ReadAdapterInterface>& rai);

// The family of methods below to get the root ops and information from a model
TORCH_API std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::istream& in);

// Overload to get root ops and info from a model file by filename
TORCH_API std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    const std::string& filename);

// Overload to get root ops and info from a model using a ReadAdapterInterface
TORCH_API std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

// The family of methods below to get contained types from a model
// Throws if not passed in a well formed model
TORCH_API std::unordered_set<std::string> _get_mobile_model_contained_types(
    std::istream& in);

// Overload to get contained types from a model file by filename
TORCH_API std::unordered_set<std::string> _get_mobile_model_contained_types(
    const std::string& filename);

// Overload to get contained types from a model using a ReadAdapterInterface
TORCH_API std::unordered_set<std::string> _get_mobile_model_contained_types(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

// Overload to get contained types from bytecode IValues
std::unordered_set<std::string> _get_mobile_model_contained_types(
    const std::vector<c10::IValue>& bytecode_ivalues);

// The family of methods below return the compatibility information of a model
// 定义一个结构体 ModelCompatibilityInfo，用于存储模型兼容性信息
struct ModelCompatibilityInfo {
  uint64_t bytecode_version; // 存储字节码版本号的无符号64位整数
  std::unordered_map<std::string, OperatorInfo> operator_info; // 存储操作符信息的无序映射，映射关系为字符串到 OperatorInfo 对象
  std::unordered_set<std::string> type_table; // 存储类型表的无序集合，集合中存储字符串表示的类型名称
  uint64_t operator_version; // 存储操作符版本号的无符号64位整数

  // 工厂方法声明，用于创建 ModelCompatibilityInfo 对象
  static TORCH_API ModelCompatibilityInfo get(std::istream& in); // 从输入流中获取信息
  static TORCH_API ModelCompatibilityInfo get(const std::string& filename); // 从文件名获取信息
  static TORCH_API ModelCompatibilityInfo
  get(std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai); // 从读取适配器接口获取信息
};

// 枚举类型 ModelCompatibilityStatus，表示模型兼容性状态
enum ModelCompatibilityStatus {
  OK = 1,    // 模型与运行时兼容
  ERROR = 2, // 模型与运行时不兼容
};

// 结构体 ModelCompatCheckResult，用于存储模型兼容性检查的结果
struct ModelCompatCheckResult {
  ModelCompatibilityStatus status; // 存储兼容性检查的状态
  std::vector<std::string> errors; // 存储兼容性检查中的错误信息列表
};

// is_compatible 函数声明，用于检查运行时和模型是否兼容，并返回检查结果
TORCH_API ModelCompatCheckResult is_compatible(
    RuntimeCompatibilityInfo runtime_info, // 运行时信息对象
    ModelCompatibilityInfo model_info); // 模型信息对象

} // namespace jit
} // namespace torch
```