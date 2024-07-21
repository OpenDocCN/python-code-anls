# `.\pytorch\torch\csrc\jit\mobile\debug_info.cpp`

```py
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/mobile/debug_info.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/serialization/callstack_debug_info_serialization.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include <c10/util/string_view.h>

namespace torch {
namespace jit {

namespace {

// 返回未找到调试信息的消息，包括未找到的调试句柄字符串
C10_ALWAYS_INLINE std::string debugHandlesNotFoundMessage(
    const std::string& debug_handles_string) {
  return "Debug info for handle(s): " + debug_handles_string +
      ", was not found.";
}

// 获取包含模块层次结构的堆栈跟踪信息及其模块信息
std::pair<std::vector<StackEntry>, std::string> getStackTraceWithModuleHierarchy(
    const DebugInfoTuple& source_callstack,
    const std::string& caller_name) {
  std::vector<StackEntry> entries;  // 堆栈条目列表

  const SourceRange& range =
      std::get<kDebugInfoTupleSourceRangeIndex>(source_callstack);  // 获取源代码范围
  InlinedCallStackPtr callstack_ptr =
      std::get<kDebugInfoTupleInlinedCSIndex>(source_callstack);  // 获取内联调用堆栈指针
  std::string prev_function_name = caller_name;  // 调用者名称
  std::string module_info;  // 模块信息字符串
  if (!callstack_ptr) {
    // 如果没有调用堆栈指针，则为顶层节点
    entries.emplace_back(StackEntry{prev_function_name, range});
    return {std::move(entries), std::move(module_info)};
  } else {
    while (callstack_ptr) {
      const auto& opt_module_instance_info = callstack_ptr->module_instance();
      if (opt_module_instance_info.has_value()) {
        const auto& module_instance_info = opt_module_instance_info.value();
        // 在某些情况下（例如，在降级后端），我们使用类型名称增强实例名称，而不是丢失类型名称。
        // 在这些情况下，实例名称包括实例名称和类型名称。
        if (module_instance_info.class_type()) {
          module_info.append(".").append(
              utils::get_module_info(module_instance_info));  // 获取模块信息
        } else {
          module_info.append(".").append(module_instance_info.instance_name());  // 获取实例名称
        }
      } else {
        module_info.append(".UNKNOWN_INSTANCE(UNKNOWN_TYPE)");  // 未知实例和类型
      }
      // 将源码范围信息添加到堆栈中
      entries.emplace_back(
          StackEntry{prev_function_name, callstack_ptr->source_range()});
      prev_function_name = callstack_ptr->function_name();  // 更新前一个函数名称
      // 在这里添加函数名称
      // 将其重命名为prev_function_name，因为对于StackEntry，函数名称将在下一次迭代中添加。
      // 这是format_stack_trace期望函数名称的格式。
      module_info.append("::").append(prev_function_name);  // 添加函数名称到模块信息

      if (callstack_ptr->callee()) {
        callstack_ptr = callstack_ptr->callee().value();  // 获取调用的指针
      } else {
        callstack_ptr = c10::intrusive_ptr<InlinedCallStack>();  // 设置为空的内联调用堆栈指针
      }
    }
    entries.emplace_back(StackEntry{prev_function_name, range});  // 最后一个堆栈条目
    return {std::move(entries), std::move(module_info)};
  }
}
// Construct stacktrace with module hierarchy based on source callstacks,
// root scope string, and top module type name.
std::pair<std::string, std::string> getStackTraceWithModuleHierarchy(
    const std::vector<DebugInfoTuple>& source_callstacks,
    const std::string& root_scope_string,
    const std::string& top_module_type_name) {
  
  // Initialize vector to store stack entries
  std::vector<StackEntry> stack_entries;

  // Initialize module_info with root scope and top module type
  std::string module_info =
      root_scope_string + "(" + top_module_type_name + ")";

  // Initialize caller function name as "<unknown>"
  std::string caller_fn_name = "<unknown>";
  
  // Append caller function name to module_info
  module_info.append("::").append(caller_fn_name);
  
  // Iterate through source_callstacks
  for (const auto& debug_info : source_callstacks) {
    // Recursively get stack trace with module hierarchy for debug_info
    auto debug_info_pair =
        getStackTraceWithModuleHierarchy(debug_info, caller_fn_name);
    
    // Extract entries from debug_info_pair and move them to stack_entries
    auto entries = std::move(debug_info_pair.first);
    stack_entries.insert(stack_entries.end(), entries.begin(), entries.end());
    
    // Append debug_info_pair's module_info to module_info
    module_info.append(debug_info_pair.second);
  }
  
  // Only the last entry in source_callstacks has a node name of interest
  auto last_entry = source_callstacks.back();
  const std::string& node_name =
      std::get<kDebugInfoTupleNodeNameIndex>(last_entry);
  
  // Append node_name to module_info
  module_info.append(".").append(node_name);
  
  // Prepare stack trace string
  std::ostringstream ss;
  ss << "Module hierarchy:" << module_info << "\n";
  
  // Format stack trace using stack_entries
  format_stack_trace(ss, stack_entries);
  
  // Return formatted stack trace and module_info
  return {ss.str(), std::move(module_info)};
}

// End of namespace declaration
} // namespace

// Constructor for MobileDebugTable, initializes source_range_map
// and processes all record names from the reader.
MobileDebugTable::MobileDebugTable(
    std::unique_ptr<caffe2::serialize::PyTorchStreamReader>& reader,
    const std::shared_ptr<CompilationUnit>& cu) {
  
  // Initialize source_range_map for mapping int64_t to SourceRange
  ska::flat_hash_map<int64_t, SourceRange> source_range_map;
  
  // Get all record names from the reader
  const std::vector<std::string>& record_names = reader->getAllRecords();
  
  // Suffix for debug records
  const c10::string_view suffix(".debug_pkl");
  
  // Iterate through each record_name in record_names
  for (const auto& record_name : record_names) {
    // 检查记录名是否以指定后缀结尾
    if (c10::string_view(record_name).ends_with(suffix)) {
      // 获取指定记录名的调试数据和其大小
      auto [debug_data, debug_size] = reader->getRecord(record_name);
      // 使用jit::unpickle解析调试数据，得到ivalueTuple
      auto ivalueTuple = jit::unpickle(
          reinterpret_cast<const char*>(debug_data.get()),
          debug_size,
          nullptr,
          {},
          c10::parseType);
      // 获取ivalueTuple中元组的元素
      const auto& ivalues = ivalueTuple.toTuple()->elements();
      // 定义IValue类型的lines和SourceRangeDeserializer类型的deserializer
      IValue lines;
      std::unique_ptr<SourceRangeDeserializer> deserializer;
      // 根据ivalues的大小和内容选择不同的处理分支
      if (ivalues.size() == 3 && ivalues[0].isString() &&
          kFormatWithStringTable == ivalues[0].toStringRef()) {
        // 使用新格式，创建SourceRangeDeserializer对象和lines对象
        deserializer = std::make_unique<SourceRangeDeserializer>(ivalues[1]);
        lines = ivalues[2];
      } else {
        // 使用旧格式，创建默认的SourceRangeDeserializer对象和lines对象
        deserializer = std::make_unique<SourceRangeDeserializer>();
        lines = ivalueTuple;
      }

      // 遍历lines中的每个元素
      for (auto& val : lines.toTuple()->elements()) {
        // 将元素转换为元组
        auto tup_elems = std::move(*std::move(val).toTuple()).elements();
        // 检查元组大小是否为3，解码包含以下内容的元组：byte_offset、debug_handle（源范围标签）、源范围
        if (tup_elems.size() == 3) {
          // 获取debug_handle并反序列化源范围
          int64_t debug_handle = tup_elems[kSourceRangeTagIndex].toInt();
          auto source_range =
              deserializer->deserialize(tup_elems[kSourceRangeIndex]);
          // 将debug_handle和源范围映射存入source_range_map中
          source_range_map.emplace(debug_handle, std::move(source_range));
        }
      }
    }
  }
  // 设置调用栈调试文件名
  const std::string callstack_debug_file("callstack_debug_map.pkl");
  // 检查读取器是否包含调用栈调试记录
  if (reader->hasRecord("callstack_debug_map.pkl")) {
    // 获取调用栈调试数据和其大小
    auto [callstack_data, callstack_data_size] =
        reader->getRecord(callstack_debug_file);
    // 创建CallStackDebugInfoUnpickler对象
    CallStackDebugInfoUnpickler unpickler;
    // 使用unpickler解析调用栈调试数据，得到调用栈指针映射表callstack_ptr_map_
    callstack_ptr_map_ = unpickler.unpickle(
        std::move(callstack_data), callstack_data_size, source_range_map, cu);
  }
} // 结束命名空间 torch 和 jit

std::string MobileDebugTable::getModuleHierarchyInfo(
    const int64_t debug_handle,
    const std::string& top_module_type_name) const {
  // 在 callstack_ptr_map_ 中查找 debug_handle 对应的迭代器
  const auto it = callstack_ptr_map_.find(debug_handle);
  // 如果未找到，则返回相应的调试句柄未找到消息
  if (it == callstack_ptr_map_.end()) {
    return debugHandlesNotFoundMessage(std::to_string(debug_handle));
  }
  // 调用 getStackTraceWithModuleHierarchy 函数获取模块层次结构的堆栈跟踪信息，并返回其第二个元素
  return (getStackTraceWithModuleHierarchy(
              {it->second}, "top", top_module_type_name))
      .second;
}

std::string MobileDebugTable::getModuleHierarchyInfo(
    const std::vector<int64_t>& debug_handles,
    const std::string& top_module_type_name) const {
  // 调用 getSourceDebugModuleHierarchyInfo 函数获取调试模块层次结构信息的第二个元素
  return getSourceDebugModuleHierarchyInfo(debug_handles, top_module_type_name)
      .second;
}

std::string MobileDebugTable::getSourceDebugString(
    const int64_t debug_handle,
    const std::string& top_module_type_name) const {
  // 在 callstack_ptr_map_ 中查找 debug_handle 对应的迭代器
  const auto it = callstack_ptr_map_.find(debug_handle);
  // 如果未找到，则返回相应的调试句柄未找到消息
  if (it == callstack_ptr_map_.end()) {
    return debugHandlesNotFoundMessage(std::to_string(debug_handle));
  }
  // 调用 getStackTraceWithModuleHierarchy 函数获取模块层次结构的堆栈跟踪信息，并返回其第一个元素
  return (getStackTraceWithModuleHierarchy(
              {it->second}, "top", top_module_type_name))
      .first;
}

std::string MobileDebugTable::getSourceDebugString(
    const std::vector<int64_t>& debug_handles,
    const std::string& top_module_type_name) const {
  // 调用 getSourceDebugModuleHierarchyInfo 函数获取调试模块层次结构信息的第一个元素
  return getSourceDebugModuleHierarchyInfo(debug_handles, top_module_type_name)
      .first;
}

std::pair<std::string, std::string> MobileDebugTable::
    getSourceDebugModuleHierarchyInfo(
        const std::vector<int64_t>& debug_handles,
        const std::string& top_module_type_name) const {
  // 存储调试信息的向量
  std::vector<DebugInfoTuple> debug_infos;
  // 标志，指示是否有未找到的调试句柄
  bool debug_handle_not_found{false};
  // 反向迭代处理调试句柄
  for (auto it = debug_handles.rbegin(); it != debug_handles.rend(); ++it) {
    auto debug_handle = *it;
    // 在 callstack_ptr_map_ 中查找 debug_handle 对应的迭代器
    const auto cs_it = callstack_ptr_map_.find(debug_handle);
    // 如果未找到，则标记为未找到，并跳出循环
    if (cs_it == callstack_ptr_map_.end()) {
      debug_handle_not_found = true;
      break;
    }
    // 将调试信息添加到 debug_infos 中
    debug_infos.emplace_back(cs_it->second);
  }
  // 如果有未找到的调试句柄，则返回相应的未找到消息
  if (debug_handle_not_found) {
    std::string debug_handles_string = "debug_handles:{";
    for (const auto debug_handle : debug_handles) {
      debug_handles_string += std::to_string(debug_handle);
    }
    debug_handles_string += "}";
    debug_handles_string = debugHandlesNotFoundMessage(debug_handles_string);
    return {debug_handles_string, debug_handles_string};
  }
  // 调用 getStackTraceWithModuleHierarchy 函数获取模块层次结构的堆栈跟踪信息，并返回其结果
  return (getStackTraceWithModuleHierarchy(
      debug_infos, "top", top_module_type_name));
}
```