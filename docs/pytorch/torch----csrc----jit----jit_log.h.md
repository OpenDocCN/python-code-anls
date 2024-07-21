# `.\pytorch\torch\csrc\jit\jit_log.h`

```
#pragma once
#include <torch/csrc/Export.h>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>

// `TorchScript` offers a simple logging facility that can enabled by setting an
// environment variable `PYTORCH_JIT_LOG_LEVEL`.

// Logging is enabled on a per file basis. To enable logging in
// `dead_code_elimination.cpp`, `PYTORCH_JIT_LOG_LEVEL` should be
// set to `dead_code_elimination.cpp` or, simply, to `dead_code_elimination`
// (i.e. `PYTORCH_JIT_LOG_LEVEL=dead_code_elimination`).

// Multiple files can be logged by separating each file name with a colon `:` as
// in the following example,
// `PYTORCH_JIT_LOG_LEVEL=dead_code_elimination:guard_elimination`

// There are 3 logging levels available for your use ordered by the detail level
// from lowest to highest.

// * `GRAPH_DUMP` should be used for printing entire graphs after optimization
// passes
// * `GRAPH_UPDATE` should be used for reporting graph transformations (i.e.
// node deletion, constant folding, etc)
// * `GRAPH_DEBUG` should be used for providing information useful for debugging
//   the internals of a particular optimization pass or analysis

// The default logging level is `GRAPH_DUMP` meaning that only `GRAPH_DUMP`
// statements will be enabled when one specifies a file(s) in
// `PYTORCH_JIT_LOG_LEVEL`.

// `GRAPH_UPDATE` can be enabled by prefixing a file name with an `>` as in
// `>alias_analysis`.
// `GRAPH_DEBUG` can be enabled by prefixing a file name with an `>>` as in
// `>>alias_analysis`.
// `>>>` is also valid and **currently** is equivalent to `GRAPH_DEBUG` as there
// is no logging level that is higher than `GRAPH_DEBUG`.

// Namespace declaration for TorchScript
namespace torch {
namespace jit {

// Declaration of a structure for representing a Node
struct Node;
// Declaration of a structure for representing a Graph
struct Graph;

// Enumeration defining different levels of JIT logging
enum class JitLoggingLevels {
  GRAPH_DUMP = 0,
  GRAPH_UPDATE,
  GRAPH_DEBUG,
};

// Function declarations for JIT logging

// Get the current JIT logging levels as a string
TORCH_API std::string get_jit_logging_levels();

// Set the JIT logging levels using a string
TORCH_API void set_jit_logging_levels(std::string level);

// Set the output stream for JIT logging
TORCH_API void set_jit_logging_output_stream(std::ostream& out_stream);

// Get the current output stream for JIT logging
TORCH_API std::ostream& get_jit_logging_output_stream();

// Get a header string for a Node in a Graph
TORCH_API std::string getHeader(const Node* node);

// Log a function represented by a shared pointer to a Graph
TORCH_API std::string log_function(const std::shared_ptr<Graph>& graph);

// Get a string with each line in the multiline string IN_STR prefixed with PREFIX
TORCH_API std::string jit_log_prefix(
    const std::string& prefix,
    const std::string& in_str);

// Get a string with each line in the multiline string IN_STR prefixed with a logging level and other information
TORCH_API std::string jit_log_prefix(
    ::torch::jit::JitLoggingLevels level,
    const char* fn,
    int l,
    const std::string& in_str);

// Check if logging is enabled for a specific file at a given logging level
TORCH_API bool is_enabled(
    const char* cfname,
    ::torch::jit::JitLoggingLevels level);

// Overloaded operator for streaming a JitLoggingLevels enum to an output stream
TORCH_API std::ostream& operator<<(
    std::ostream& out,
    ::torch::jit::JitLoggingLevels level);

// Macro for conditional logging based on the enabled logging level for the current file
#define JIT_LOG(level, ...)                                         \
  if (is_enabled(__FILE__, level)) {                                 \
    ::torch::jit::get_jit_logging_output_stream()                   \
        << ::torch::jit::jit_log_prefix(                            \
               level, __FILE__, __LINE__, ::c10::str(__VA_ARGS__)); \
  }



    // 调用 Torch JIT 模块中的日志输出流函数，将日志级别、文件名、行号及可变参数信息作为前缀输出到流中
    ::torch::jit::get_jit_logging_output_stream()                   \
        // 调用 Torch JIT 模块中的日志前缀函数，传入日志级别、当前文件名、行号和可变参数作为参数
        << ::torch::jit::jit_log_prefix(                            \
               level, __FILE__, __LINE__, ::c10::str(__VA_ARGS__)); \
  }


这段代码是在使用 Torch 的 JIT 模块进行日志输出。
// 定义宏 SOURCE_DUMP，用于记录源码还原的日志消息和图形对象 G
#define SOURCE_DUMP(MSG, G)                       \
  JIT_LOG(                                        \
      ::torch::jit::JitLoggingLevels::GRAPH_DUMP, \
      MSG,                                        \
      "\n",                                       \
      ::torch::jit::log_function(G));

// 定义宏 GRAPH_DUMP，用于在优化过程后记录图形对象 G 的日志消息
#define GRAPH_DUMP(MSG, G) \
  JIT_LOG(                 \
      ::torch::jit::JitLoggingLevels::GRAPH_DUMP, MSG, "\n", (G)->toString());

// 定义宏 GRAPH_UPDATE，用于报告图形变换（如节点删除、常量折叠、公共子表达式消除）的日志消息
#define GRAPH_UPDATE(...) \
  JIT_LOG(::torch::jit::JitLoggingLevels::GRAPH_UPDATE, __VA_ARGS__);

// 定义宏 GRAPH_DEBUG，用于调试特定优化过程时提供有用信息的日志消息
#define GRAPH_DEBUG(...) \
  JIT_LOG(::torch::jit::JitLoggingLevels::GRAPH_DEBUG, __VA_ARGS__);

// 定义宏 GRAPH_EXPORT，用于导出图形对象 G 的 IR 数据，以便可以通过脚本加载
#define GRAPH_EXPORT(MSG, G)                       \
  JIT_LOG(                                         \
      ::torch::jit::JitLoggingLevels::GRAPH_DEBUG, \
      MSG,                                         \
      "\n<GRAPH_EXPORT>\n",                        \
      (G)->toString(),                             \
      "</GRAPH_EXPORT>");

// 定义宏 GRAPH_DUMP_ENABLED，用于检查 GRAPH_DUMP 日志级别是否启用
#define GRAPH_DUMP_ENABLED \
  (is_enabled(__FILE__, ::torch::jit::JitLoggingLevels::GRAPH_DUMP))

// 定义宏 GRAPH_UPDATE_ENABLED，用于检查 GRAPH_UPDATE 日志级别是否启用
#define GRAPH_UPDATE_ENABLED \
  (is_enabled(__FILE__, ::torch::jit::JitLoggingLevels::GRAPH_UPDATE))

// 定义宏 GRAPH_DEBUG_ENABLED，用于检查 GRAPH_DEBUG 日志级别是否启用
#define GRAPH_DEBUG_ENABLED \
  (is_enabled(__FILE__, ::torch::jit::JitLoggingLevels::GRAPH_DEBUG))

// 命名空间 jit 结束
} // namespace jit

// 命名空间 torch 结束
} // namespace torch
```