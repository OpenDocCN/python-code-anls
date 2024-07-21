# `.\pytorch\torch\csrc\jit\serialization\callstack_debug_info_serialization.h`

```py
#pragma once

#include <c10/core/Allocator.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/scope.h>

#include <ATen/core/ivalue.h>

#include <vector>

#include <c10/util/flat_hash_map.h>

namespace c10 {
struct IValue;
}

namespace torch::jit {

// 声明 Pickler 类，用于序列化调用堆栈调试信息
class Pickler;

// InlinedCallStackSerializer 类，用于序列化内联调用堆栈
class InlinedCallStackSerializer {
 public:
  // 序列化内联调用堆栈
  // SerializedInlinedCallStack =
  // [module_info, source range tag, SerializedInlinedCallStack]
  // module_info = [ClassType.qualifiedName, instance_name]
  // source_range_tag = 唯一的源代码范围标识符
  c10::IValue serialize(
      const InlinedCallStackPtr& cs_ptr,
      const SourceRangeTagMap& source_range_tags);

 private:
  // 序列化模块实例信息
  // module_info = [ClassType.qualifiedName, instance_name]
  c10::IValue serialize_module_instance_info(
      const std::optional<ModuleInstanceInfo>& m);

  // 缓存序列化后的内联调用堆栈指针，因为多个 InlinedCallStackPtr 可能引用同一个指针
  ska::flat_hash_map<InlinedCallStackPtr, c10::IValue> serialized_inlined_callstack_;

  // 缓存序列化后的模块实例信息
  // 可能有多个节点属于同一个父模块、祖父模块等
  ska::flat_hash_map<std::string, c10::IValue> serialized_module_instance_info_;
};

// CallStackDebugInfoPickler 类，用于序列化调用堆栈的调试信息
class TORCH_API CallStackDebugInfoPickler {
 public:
  CallStackDebugInfoPickler() = default;

  // 使用 Pickler 对象序列化调用堆栈调试信息
  std::vector<char> pickle(
      const std::unordered_map<int64_t, DebugInfoTuple>& callstack_ptrs,
      const SourceRangeTagMap& source_range_tags);

 private:
  InlinedCallStackSerializer css_;
};

// InlinedCallStackDeserializer 类，用于反序列化内联调用堆栈
class InlinedCallStackDeserializer {
 public:
  // 反序列化 IValue 为 InlinedCallStackPtr
  InlinedCallStackPtr deserialize(
      const c10::IValue& iv,
      const ska::flat_hash_map<int64_t, SourceRange>& source_range_map,
      const std::shared_ptr<CompilationUnit>& cu);

 private:
  // 反序列化模块实例信息
  std::optional<ModuleInstanceInfo> deserialize_module_instance_info(
      const c10::IValue& iv,
      const std::shared_ptr<CompilationUnit>& cu);

  // 缓存反序列化后的内联调用堆栈指针
  ska::flat_hash_map<c10::intrusive_ptr<c10::ivalue::Tuple>, InlinedCallStackPtr>
      cached_inlined_callstacks_;

  // 缓存反序列化后的模块实例信息
  ska::flat_hash_map<c10::intrusive_ptr<c10::ivalue::Tuple>, ModuleInstanceInfo>
      cached_module_instance_info_;
};

// CallStackDebugInfoUnpickler 类，用于反序列化调用堆栈调试信息
class TORCH_API CallStackDebugInfoUnpickler {
 public:
  // 解析二进制数据以获取调用堆栈调试信息
  ska::flat_hash_map<int64_t, DebugInfoTuple> unpickle(
      at::DataPtr&& data,
      size_t size,
      const ska::flat_hash_map<int64_t, SourceRange>& source_range_map,
      const std::shared_ptr<CompilationUnit>& cu);

 private:
  InlinedCallStackDeserializer csds_;
};

} // namespace torch::jit
```