# `.\pytorch\torch\csrc\jit\mobile\debug_info.h`

```py
/*
 * MobileDebugTable:
 * Deserializes debug_pkl and callstack_map records from PT model's zip archive
 * and stores them in a map of debug handles to DebugInfoPair. Debug handles are
 * unique per model and runtime, be in lite interpreter or delegate, an
 * exception of BackendRuntimeException should raised using debug handles.
 * getSourceDebugString method is responsible for translating debug
 * handles to correspond debug information.
 * This debug informatin includes stack trace of model level source code and
 * module hierarchy where the exception occurred.
 */

#pragma once // 确保头文件只被编译一次

#include <c10/util/flat_hash_map.h> // 包含使用的第三方库的头文件
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/ir/scope.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>

namespace torch {
namespace jit {

/*
 * MobileDebugTable 类
 * 从 PT 模型的 zip 存档中反序列化 debug_pkl 和 callstack_map 记录，
 * 并将它们存储在一个从 debug handle 到 DebugInfoPair 的映射中。
 * Debug handles 在每个模型和运行时中是唯一的，无论是在轻量解释器还是委托中，
 * 如果出现 BackendRuntimeException，则应使用 debug handle 抛出异常。
 * getSourceDebugString 方法负责将 debug handle 转换为相应的调试信息。
 * 此调试信息包括模型级源代码的堆栈跟踪和发生异常的模块层次结构。
 */
class MobileDebugTable {
 public:
  MobileDebugTable() = default; // 默认构造函数
  
  // 构造函数，从 PyTorch 的 StreamReader 和 CompilationUnit 创建 MobileDebugTable
  MobileDebugTable(
      std::unique_ptr<caffe2::serialize::PyTorchStreamReader>& reader,
      const std::shared_ptr<CompilationUnit>& cu);

  // 从迭代器范围构造 MobileDebugTable
  template <typename It>
  MobileDebugTable(It begin, It end) : callstack_ptr_map_(begin, end) {}

  // 获取给定 debug handle 对应的源调试字符串
  std::string getSourceDebugString(
      const int64_t debug_handle,
      const std::string& top_module_type_name = "ModuleTypeUnknown") const;
  
  // 获取给定一组 debug handles 对应的源调试字符串
  std::string getSourceDebugString(
      const std::vector<int64_t>& debug_handles,
      const std::string& top_module_type_name = "ModuleTypeUnknown") const;
  
  // 获取给定 debug handle 对应的模块层次结构信息
  std::string getModuleHierarchyInfo(
      const int64_t debug_handle,
      const std::string& top_module_type_name = "ModuleTypeUnknown") const;
  
  // 获取给定一组 debug handles 对应的模块层次结构信息
  std::string getModuleHierarchyInfo(
      const std::vector<int64_t>& debug_handles,
      const std::string& top_module_type_name = "ModuleTypeUnknown") const;

  // 获取调用堆栈指针映射的常量引用
  const ska::flat_hash_map<int64_t, DebugInfoTuple>& getCallStackPtrMap()
      const {
    return callstack_ptr_map_;
  }

 private:
  // 根据给定的 debug handles 和顶级模块类型名获取源调试模块层次信息
  std::pair<std::string, std::string> getSourceDebugModuleHierarchyInfo(
      const std::vector<int64_t>& debug_handles,
      const std::string& top_module_type_name = "ModuleTypeUnknown") const;

  // 调用堆栈指针映射，从 debug handle 到 DebugInfoTuple
  ska::flat_hash_map<int64_t, DebugInfoTuple> callstack_ptr_map_;
};

} // namespace jit
} // namespace torch
```