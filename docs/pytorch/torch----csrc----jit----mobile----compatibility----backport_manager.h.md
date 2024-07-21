# `.\pytorch\torch\csrc\jit\mobile\compatibility\backport_manager.h`

```py
#pragma once

#include <functional>  // 包含函数对象的头文件
#include <memory>      // 包含智能指针的头文件
#include <unordered_map>  // 包含无序映射的头文件

namespace c10 {
struct IValue;  // 声明 IValue 结构体
}

namespace caffe2 {
namespace serialize {
class IStreamAdapter;         // 声明 IStreamAdapter 类
class ReadAdapterInterface;   // 声明 ReadAdapterInterface 类
class PyTorchStreamWriter;    // 声明 PyTorchStreamWriter 类
class PyTorchStreamReader;    // 声明 PyTorchStreamReader 类
} // namespace serialize
} // namespace caffe2

namespace torch {
namespace jit {

/*
BackportManager manages a list of backport from n to n-1 function, and provides
function to check if a specific function exists.
BackportManager 管理从版本 n 到版本 n-1 的回溯函数列表，并提供检查特定函数是否存在的功能。
*/
class BackportManager final {
 public:
  // Check if a bytecode backport function exists for a given from_version.
  // 检查是否存在给定 from_version 的字节码回溯函数。
  bool hasBytecodeBackportFunction(const int64_t from_version) const;

  // Get a reference to the map of bytecode backport functions.
  // 返回字节码回溯函数的映射表的引用。
  std::unordered_map<
      int64_t,
      std::function<std::stringstream(std::stringstream&)>>&
  bytecodeBackportFunctions() const;

  // Backport function that performs conversion from one version to another.
  // 进行从一个版本到另一个版本的转换的回溯函数。
  bool backport(
      std::istream& oss,
      caffe2::serialize::PyTorchStreamWriter& final_writer,
      int64_t from_version,
      int64_t to_version) const;

  // Default constructor for BackportManager.
  // BackportManager 的默认构造函数。
  BackportManager();

  // Disable copy constructor and assignment operator.
  // 禁用复制构造函数和赋值运算符。
  BackportManager(BackportManager const&) = delete;
  BackportManager& operator=(BackportManager const&) = delete;

 private:
  // Register a bytecode backport function for a specific from_version.
  // 为特定的 from_version 注册字节码回溯函数。
  void registerBytecodeBackportFunction(
      const int64_t from_version,
      const std::function<std::stringstream(std::stringstream&)>&
          backport_function);

  // Registry of backport functions.
  // 回溯函数的注册表。
};

} // namespace jit
} // namespace torch
```