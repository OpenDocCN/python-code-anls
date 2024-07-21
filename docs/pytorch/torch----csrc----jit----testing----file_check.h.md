# `.\pytorch\torch\csrc\jit\testing\file_check.h`

```py
#pragma once


// 预处理指令，确保此头文件只被编译一次


#include <torch/csrc/Export.h>
#include <memory>
#include <string>


// 引入其他必要的头文件，包括导出宏、内存管理、字符串处理等


namespace torch {
namespace jit {


// 进入 torch::jit 命名空间


struct Graph;


// 声明 Graph 结构体，假设在此文件中 Graph 结构体已经定义


namespace testing {


// 进入 testing 命名空间


struct FileCheckImpl;


// 声明 FileCheckImpl 结构体，实现可能在其他地方定义


struct FileCheck {
 public:


// FileCheck 结构体定义开始，公有部分


  TORCH_API explicit FileCheck();


// 显式构造函数声明，使用 TORCH_API 宏导出


  TORCH_API ~FileCheck();


// 析构函数声明，使用 TORCH_API 宏导出


  // Run FileCheck against test string
  TORCH_API void run(const std::string& test_string);


// 运行 FileCheck 对测试字符串执行检查，使用 TORCH_API 宏导出


  // Run FileCheck against dump of graph IR
  TORCH_API void run(const Graph& graph);


// 运行 FileCheck 对图形 IR 的转储执行检查，使用 TORCH_API 宏导出


  // Parsing input checks string and run against test string / dump of graph IR
  TORCH_API void run(
      const std::string& input_checks_string,
      const std::string& test_string);


// 解析输入的检查字符串，并对测试字符串或图形 IR 的转储执行检查，使用 TORCH_API 宏导出


  TORCH_API void run(
      const std::string& input_checks_string,
      const Graph& graph);


// 解析输入的检查字符串，并对图形 IR 的转储执行检查，使用 TORCH_API 宏导出


  // Checks that the string occurs, starting at the end of the most recent match
  TORCH_API FileCheck* check(const std::string& str);


// 检查字符串是否出现在最近匹配的末尾，使用 TORCH_API 宏导出


  // Checks that the string does not occur between the previous match and next
  // match. Consecutive check_nots test against the same previous match and next
  // match
  TORCH_API FileCheck* check_not(const std::string& str);


// 检查字符串是否不出现在上一个匹配和下一个匹配之间，连续的 check_not 对同一前一个匹配和后一个匹配进行测试，使用 TORCH_API 宏导出


  // Checks that the string occurs on the same line as the previous match
  TORCH_API FileCheck* check_same(const std::string& str);


// 检查字符串是否出现在与前一个匹配相同的行上，使用 TORCH_API 宏导出


  // Checks that the string occurs on the line immediately following the
  // previous match
  TORCH_API FileCheck* check_next(const std::string& str);


// 检查字符串是否出现在紧接着前一个匹配的下一行上，使用 TORCH_API 宏导出


  // Checks that the string occurs count number of times, starting at the end
  // of the previous match. If exactly is true, checks that there are exactly
  // count many matches
  TORCH_API FileCheck* check_count(
      const std::string& str,
      size_t count,
      bool exactly = false);


// 检查字符串是否出现 count 次，从前一个匹配的末尾开始。如果 exactly 为 true，则检查是否恰好有 count 个匹配，使用 TORCH_API 宏导出


  // A series of consecutive check_dags get turned into a group of checks
  // which can appear in any order relative to each other. The checks begin
  // at the end of the previous match, and the match for the check_dag group
  // is the minimum match of all individual checks to the maximum match of all
  // individual checks.
  TORCH_API FileCheck* check_dag(const std::string& str);


// 一系列连续的 check_dag 被转换为一个可以以任意顺序出现的检查组。这些检查从前一个匹配的末尾开始，check_dag 组的匹配是所有单个检查的最小匹配到最大匹配之间的匹配，使用 TORCH_API 宏导出


  // Checks that source token is highlighted in str (usually an error message).
  TORCH_API FileCheck* check_source_highlighted(const std::string& str);


// 检查源标记是否在 str 中高亮显示（通常是错误消息），使用 TORCH_API 宏导出


  // Checks that the regex matched string occurs, starting at the end of the
  // most recent match
  TORCH_API FileCheck* check_regex(const std::string& str);


// 检查正则表达式匹配的字符串是否出现在最近匹配的末尾，使用 TORCH_API 宏导出


  // reset checks
  TORCH_API void reset();


// 重置检查状态，使用 TORCH_API 宏导出


 private:
  bool has_run = false;
  std::unique_ptr<FileCheckImpl> fcImpl;
};


// FileCheck 结构体定义结束，包括私有成员变量 has_run 和 fcImpl


} // namespace testing
} // namespace jit
} // namespace torch


// 结束 testing、jit 和 torch 命名空间
```