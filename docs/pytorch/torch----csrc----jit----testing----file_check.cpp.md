# `.\pytorch\torch\csrc\jit\testing\file_check.cpp`

```py
//==-- llvm/Support/FileCheck.h ---------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// API modified from llvm::FileCheck

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/StringUtil.h>
#include <c10/util/irange.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

// 命名空间 torch::jit::testing 下定义了用于测试的文件检查功能

namespace torch {
namespace jit {
namespace testing {

// 枚举不同的检查类型，用于区分不同的检查方式
enum CheckType {
  CHECK,                     // 普通检查
  CHECK_NEXT,                // 检查下一个
  CHECK_SAME,                // 检查相同
  CHECK_NOT,                 // 检查不存在
  CHECK_COUNT,               // 检查计数
  CHECK_DAG,                 // 检查有向无环图
  CHECK_SOURCE_HIGHLIGHTED,   // 检查高亮源码
  CHECK_REGEX                // 正则表达式检查
};

// 结构体 Check 表示一个检查的实例
struct Check {
  Check(
      CheckType type,
      std::string str,
      std::optional<size_t> count = c10::nullopt)
      : type_(type), count_(count), search_str_(std::move(str)) {}

  // 另一种构造函数，接受 c10::string_view 作为参数，委托给上面的构造函数
  Check(
      CheckType type,
      c10::string_view str,
      std::optional<size_t> count = c10::nullopt)
      : Check(type, std::string(str.begin(), str.end()), count) {}

  // 成员变量定义
  CheckType type_;
  std::optional<size_t> count_;
  const std::string search_str_;

  // 重载 << 运算符，输出检查的类型和搜索字符串
  friend std::ostream& operator<<(std::ostream& out, const Check& c);
};

// 实现 << 运算符的函数，根据检查的类型输出相应的标识符和搜索字符串
std::ostream& operator<<(std::ostream& out, const Check& c) {
  switch (c.type_) {
    case CHECK:
      out << "CHECK";
      break;
    case CHECK_NEXT:
      out << "CHECK-NEXT";
      break;
    case CHECK_SAME:
      out << "CHECK-SAME";
      break;
    case CHECK_NOT:
      out << "CHECK-NOT";
      break;
    case CHECK_DAG:
      out << "CHECK-DAG";
      break;
    case CHECK_COUNT:
      out << "CHECK-COUNT-" << *c.count_;
      break;
    case CHECK_SOURCE_HIGHLIGHTED:
      out << "CHECK-SOURCE-HIGHLIGHTED";
      break;
    case CHECK_REGEX:
      out << "CHECK-REGEX";
      break;
  }
  out << ": " << c.search_str_;  // 输出检查类型和搜索字符串
  return out;
};

// 匿名命名空间，定义了辅助函数 assertFind，用于在指定范围内查找子字符串
namespace {

// assertFind 函数，用于在指定的源码范围内查找子字符串，若找不到则抛出异常
size_t assertFind(
    const SourceRange& search_range,               // 搜索范围
    const std::string& sub,                       // 要搜索的子字符串
    const std::function<void(std::ostream& out)>& extra_msg = nullptr) {   // 可选的额外信息生成函数
  auto pos = search_range.source()->text_str().find(sub, search_range.start());  // 在源码字符串中查找子字符串的位置
  if (pos == std::string::npos || (pos + sub.size()) > search_range.end()) {    // 如果未找到或找到的位置超出了搜索范围的末尾
    auto found_range =
        SourceRange(search_range.source(), search_range.start(), sub.size());   // 获取找到的范围
    std::stringstream ss;
    ss << "Expected to find ";
    c10::printQuotedString(ss, sub);                // 输出期望找到的字符串
    ss << " but did not find it" << std::endl;      // 输出未找到的提示信息
    ss << "Searched string:" << std::endl;
    found_range.highlight(ss);                      // 高亮显示找到的范围
    if (extra_msg) {
      extra_msg(ss);                               // 若有额外信息生成函数，则调用生成额外信息
    }
    throw std::runtime_error(ss.str());             // 抛出运行时异常，包含错误信息
  }
  return pos;                                       // 返回找到的位置
}

// 另一个 assertFind 函数的重载，用于在指定的源码范围内查找子字符串
size_t assertFind(
    const SourceRange& search_range,               // 搜索范围
    const std::string& sub,
    // 调用 assertFind 函数，传入 search_range、sub 以及一个 lambda 函数作为回调
    // lambda 函数接受一个 std::ostream 对象 out，并向其写入一条日志信息
    // 日志信息格式为 "From " 后跟 check 对象的字符串表示，并换行
  return assertFind(search_range, sub, [&](std::ostream& out) {
    out << "From " << check << "\n";
  });
} // namespace

// 结构体定义：FileCheckImpl
struct FileCheckImpl {
  // TORCH_API 指定的默认构造函数
  TORCH_API explicit FileCheckImpl() = default;

  // 运行测试文件检查
  TORCH_API void run(const std::string& test_file) {
    // 标记已经运行过
    has_run = true;

    // 如果检查组为空或第一个检查组为空，则抛出异常
    if (groups.empty() || groups[0].empty()) {
      throw std::runtime_error(
          "No checks have been added to this instance of"
          " Filecheck! Check for bad input.");
    }

    // 对测试文件执行检查
    doChecks(std::make_shared<Source>(test_file));
  }

  // 使用检查文件和测试文件运行检查
  TORCH_API void run(
      const std::string& checks_file,
      const std::string& test_file) {
    auto source = std::make_shared<Source>(checks_file);
    parseStrings(source);
    run(test_file);
  }

  // 添加检查对象到检查组中
  TORCH_API void addCheck(const Check& check) {
    // 连续的 CHECK_DAGs 和 CHECK_NOTs 需要作为一个组进行评估
    if (groups.empty() ||
        (check.type_ != CHECK_NOT && check.type_ != CHECK_DAG)) {
      groups.push_back({check});
    } else {
      auto& last_group = groups.back();
      if (last_group.at(0).type_ == check.type_) {
        last_group.push_back(check);
      } else {
        groups.push_back({check});
      }
    }
  // 初始设定为 false，表示程序尚未运行
  has_run = false;
}

// 将指定类型、字符串及可选计数添加至检查列表
TORCH_API void addCheck(
    CheckType type,
    const std::string& s,
    std::optional<size_t> count = c10::nullopt) {
  addCheck(Check(type, s, count));
}

// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
// 声明并初始化私有变量 has_run，用于追踪程序是否已运行
bool has_run = false;

// 声明友元函数，用于输出 FileCheckImpl 类的对象信息至流
friend std::ostream& operator<<(std::ostream& out, const FileCheckImpl& fc);

private:
// 解析单个检查内容
bool parseSingleCheck(const std::shared_ptr<Source>& source, size_t* start) {
  // 定义检查类型及其后缀字符串的静态对应关系
  const static std::vector<std::pair<CheckType, std::string>> check_pairs = {
      {CHECK, ": "},
      {CHECK_NEXT, "-NEXT: "},
      {CHECK_SAME, "-SAME: "},
      {CHECK_NOT, "-NOT: "},
      {CHECK_DAG, "-DAG: "},
      {CHECK_COUNT, "-COUNT-"}, // 需要特殊解析
      {CHECK_SOURCE_HIGHLIGHTED, "-SOURCE-HIGHLIGHTED: "},
      {CHECK_REGEX, "-REGEX: "},
  };

  // 遍历各种检查类型及其后缀
  for (const auto& check_pair : check_pairs) {
    const std::string& check_suffix = check_pair.second;
    // 查找在文本中指定位置开始的后缀字符串
    auto suffix_pos = source->text_str().find(check_suffix, *start);
    // 如果未找到匹配的后缀，则继续下一个检查类型
    if (suffix_pos != *start) {
      continue;
    }
    // 计算检查字符串的结束位置
    size_t end_check_string = suffix_pos + check_suffix.size();
    // 获取当前检查类型
    CheckType type = check_pair.first;
    std::optional<size_t> count = c10::nullopt;
    // 查找检查字符串结束后的换行符位置
    auto end_line = source->text_str().find("\n", end_check_string);
    bool exactly = false;
    // 如果当前检查类型为 CHECK_COUNT
    if (type == CHECK_COUNT) {
      const std::string exact = "EXACTLY-";
      // 如果检查字符串包含 "EXACTLY-" 字符串
      if (source->text_str().find(exact, end_check_string) ==
          end_check_string) {
        exactly = true;
        end_check_string += exact.size();
      }
      // 查找到 ":" 符号的位置，确定计数值的结束位置
      size_t end =
          assertFind(SourceRange(source, end_check_string, end_line), ":");
      // 提取并转换计数值
      auto count_view = source->text_str()
                            .substr(end_check_string, end - end_check_string)
                            .str();
      count = std::stoll(std::string(count_view.begin(), count_view.end()));
      end_check_string = end + 2; // 添加 ':' 及空格的偏移量
    }
    // 构造 Check 对象
    auto check = Check(
        type,
        source->text_str()
            .substr(end_check_string, end_line - end_check_string)
            .str(),
        count);
    // 将构造的检查对象添加至检查列表
    addCheck(check);
    // 如果是 EXACTLY 类型的检查，还需添加 CHECK_NOT 类型的检查
    if (exactly) {
      addCheck(CHECK_NOT, check.search_str_);
    }
    *start = end_line;
    return true;
  }
  return false;
}

// 在给定源文本及前一检查结束位置后，查找下一个检查开始位置
size_t findNextStart(const std::shared_ptr<Source>& source, size_t prev_end) {
  // 查找下一个 '#' 符号的位置
  size_t start = source->text_str().find("#", prev_end);
  // 如果未找到 '#' 符号，则返回未找到的标志
  if (start == std::string::npos) {
    return start;
  }
  start += 1;
  // 最大允许的空白字符数
  static constexpr size_t max_whitespace = 6;
  size_t i = 0;
  // 在指定范围内查找最大允许空白字符数
  while (start + i < source->size() && i < max_whitespace) {
    auto c = source->char_at(start + i);
    // 如果字符不是空格或制表符，则结束循环
    if (c != ' ' && c != '\t') {
      break;
    }
    i++;
  }
  // 需要查找的关键字字符串 "CHECK"
  static const std::string check = "CHECK";
    if (source->text_str().substr(start + i, check.size()) == check) {
      // 检查从起始位置开始的子字符串是否与指定的检查字符串匹配
      return start + i + check.size();
    } else {
      // 如果不匹配，则递归查找下一个起始位置
      return findNextStart(source, start + i + 1);
    }
  }

  void parseStrings(const std::shared_ptr<Source>& source) {
    size_t start = 0;
    // 找到下一个起始位置
    start = findNextStart(source, 0);
    while (start != std::string::npos) {
      // 尝试解析单个检查
      bool found_match = parseSingleCheck(source, &start);
      if (!found_match) {
        // 如果未能解析检查项，则抛出运行时异常
        std::ostringstream ss;
        ss << "Could not parse check at:\n";
        SourceRange(source, start, start + 1).highlight(ss);
        ss << "Check for bad input.";
        has_run = true;
        throw std::runtime_error(ss.str());
      }
      // 继续查找下一个起始位置
      start = findNextStart(source, start);
    }
  }

  void doCheckNot(
      const std::vector<Check>& nots,
      const std::shared_ptr<Source>& source,
      const SourceRange& prev,
      const SourceRange& next) {
    auto start = prev.end(); // inclusive
    auto end = next.start(); // exclusive
    if (end < start) {
      return;
    }
    // 对于每一个“not”检查，确保在指定范围内未找到特定的搜索字符串
    for (const auto& check : nots) {
      AT_ASSERT(check.type_ == CHECK_NOT);
      assertNotFind(SourceRange(source, start, end), check.search_str_, check);
    }
  }

  // 检查源代码标记是否被高亮显示，但不会推进搜索范围。
  void doCheckSourceHighlighted(
      const Check& check,
      const std::shared_ptr<Source>& source,
      size_t start_offset) {
    // 匿名函数，用于构造错误信息并抛出异常
    auto construct_error_and_throw = [&](size_t error_start_pos) {
      SourceRange error_range(
          source, error_start_pos, check.search_str_.size());
      std::stringstream ss;
      ss << "Expected to find ";
      c10::printQuotedString(ss, check.search_str_);
      ss << "highlighted but it is not." << std::endl;
      error_range.highlight(ss);
      throw std::runtime_error(ss.str());
    };

    size_t search_start_offset = start_offset;
    bool found_token_at_least_once = false;
    size_t pos = search_start_offset;
    // 当前位置在源文本大小之内时执行循环
    while (pos < source->size()) {
      // 在源文本中查找目标字符串的位置，从指定的偏移开始搜索
      pos = source->text_str().find(check.search_str_, search_start_offset);
      // 如果未找到目标字符串，跳出循环
      if (pos == std::string::npos) {
        break;
      }

      // 标记至少找到一个目标字符串
      found_token_at_least_once = true;

      // 获取目标字符串的行号和列号
      auto lineno = source->lineno_for_offset(pos);
      auto col = pos - source->offset_for_line(lineno);
      auto highlight_lineno = lineno + 1;

      // 如果高亮行号超过源文本的总行数，抛出构造的错误
      if (highlight_lineno >= source->num_lines()) {
        construct_error_and_throw(pos);
      }

      // 计算高亮部分的起始和结束偏移量
      auto highlight_start_offset =
          source->offset_for_line(highlight_lineno) + col;
      auto highlight_end_offset = std::min(
          highlight_start_offset + check.search_str_.size(), source->size());

      // 如果高亮部分的结束偏移量超过了源文本的大小，抛出构造的错误
      if (highlight_end_offset >= source->size()) {
        construct_error_and_throw(pos);
      }

      // 检查高亮部分是否完全由波浪符('~')组成
      bool found_highlight = true;
      for (const auto posi :
           c10::irange(highlight_start_offset, highlight_end_offset)) {
        if (source->char_at(posi) != '~') {
          found_highlight = false;
        }
      }

      // 如果高亮部分完全由波浪符('~')组成，执行断言检查
      if (found_highlight) {
        assertNotFind(
            SourceRange(
                source, highlight_start_offset - 1, highlight_start_offset),
            "~",
            check);
        assertNotFind(
            SourceRange(source, highlight_end_offset, highlight_end_offset + 1),
            "~",
            check);
        // 直接返回，不再继续处理
        return;
      }

      // 更新搜索起始偏移量，准备下一次循环
      search_start_offset = pos + 1;
    }

    // 如果未至少找到一个目标字符串，执行断言检查
    if (!found_token_at_least_once) {
      // 确保生成错误消息失败
      assertFind(source, check.search_str_, start_offset, check);
    }

    // 抛出构造的错误，指定起始偏移量
    construct_error_and_throw(start_offset);
  }

  // 匹配DAG（有向无环图）组，返回匹配的源范围
  SourceRange matchDagGroup(
      const std::vector<Check>& group,
      const std::shared_ptr<Source>& source,
      const SourceRange& prev) {
    // 初始化组的起始和结束位置
    size_t group_beg = std::string::npos;
    size_t group_end = 0;

    // 断言组不能为空
    AT_ASSERT(!groups.empty());

    // 遍历每个检查项
    for (const auto& check : group) {
      // 断言检查类型与组中第一个检查项类型相同
      AT_ASSERT(check.type_ == group[0].type_);
      // 在源文本中查找并断言找到目标字符串的位置，从给定的偏移开始搜索
      auto pos = assertFind(source, check.search_str_, prev.end(), check);
      // 更新组的起始和结束位置
      group_beg = std::min(pos, group_beg);
      group_end = std::max(pos + check.search_str_.size(), group_end);
    }

    // 返回匹配的源范围
    return SourceRange(source, group_beg, group_end);
  }

  // 匹配组，返回匹配的源范围
  SourceRange matchGroup(
      const std::vector<Check>& group,
      const std::shared_ptr<Source>& source,
      const SourceRange& prev) {
    // 断言组不能为空
    AT_ASSERT(!group.empty());
    // 获取第一个检查项的类型
    CheckType type = group[0].type_;

    // 如果类型为CHECK_DAG，则调用matchDagGroup函数进行匹配
    if (type == CHECK_DAG) {
      return matchDagGroup(group, source, prev);
    }

    // 断言类型不为CHECK_NOT，并且组的大小为1
    AT_ASSERT(type != CHECK_NOT);
    AT_ASSERT(group.size() == 1);

    // 获取组中的唯一检查项
    const auto& check = group[0];
    // 设置起始和结束的范围
    size_t start_range = prev.end();
    size_t end_range = start_range;
    switch (check.type_) {
      case CHECK: {
        // 调用 assertFind 函数查找指定字符串，并更新起始和结束范围
        start_range = assertFind(source, check.search_str_, start_range, check);
        end_range = start_range + check.search_str_.size();
      } break;
      case CHECK_SAME: {
        // 在指定范围内查找字符串，并确保未找到换行符
        auto pos = assertFind(source, check.search_str_, start_range, check);
        assertNotFind(SourceRange(source, prev.end(), pos), "\n", check);
        start_range = pos;
        end_range = pos + check.search_str_.size();
      } break;
      case CHECK_NEXT: {
        // 在下一行查找指定字符串，并确保未找到换行符
        auto line_end = assertFind(source, "\n", start_range, check);
        auto pos = assertFind(source, check.search_str_, line_end + 1, check);
        assertNotFind(SourceRange(source, line_end + 1, pos), "\n", check);
        start_range = pos;
        end_range = pos + check.search_str_.size();
      } break;
      case CHECK_COUNT: {
        // 查找指定数量的字符串出现次数，并更新起始和结束范围
        auto group_start_range = std::string::npos;
        AT_ASSERT(check.count_ && *check.count_ != 0);
        for (size_t i = 0; i < *check.count_; ++i) {
          start_range =
              assertFind(source, check.search_str_, start_range, check);
          group_start_range = std::min(start_range, group_start_range);
          end_range = start_range + check.search_str_.size();
          start_range = end_range;
        }
        start_range = group_start_range;
      } break;
      case CHECK_SOURCE_HIGHLIGHTED: {
        // 对源代码进行高亮检查
        doCheckSourceHighlighted(check, source, start_range);
        break;
      }
      case CHECK_REGEX: {
        // 使用正则表达式查找指定字符串，并更新起始和结束范围
        start_range =
            assertFindRegex(source, check.search_str_, start_range, check);
        end_range = start_range + check.search_str_.size();
        break;
      }
      case CHECK_DAG: {
        // 如果类型为 CHECK_DAG，则抛出错误
        AT_ERROR();
      } break;
      case CHECK_NOT: {
        // 如果类型为 CHECK_NOT，则抛出错误
        AT_ERROR();
      } break;
    }
    // 返回更新后的源范围对象
    return SourceRange(source, start_range, end_range);
  }

  void doChecks(const std::shared_ptr<Source>& source) {
    // 初始化前一个匹配的范围
    SourceRange prev(source, 0, 0);
    // 遍历每个检查组
    for (size_t i = 0; i < groups.size(); i++) {
      const auto& curr_group = groups[i];
      // 获取当前检查组的类型
      CheckType type = curr_group.at(0).type_;
      if (type != CHECK_NOT) {
        // 如果当前类型不是 CHECK_NOT，则执行匹配并更新 prev
        prev = matchGroup(curr_group, source, prev);
      } else {
        // 如果当前类型是 CHECK_NOT
        if (i + 1 < groups.size()) {
          const auto& next_group = groups[i + 1];
          // 确保下一个组的类型不是 CHECK_NOT
          AT_ASSERT(next_group.at(0).type_ != CHECK_NOT);
          // 执行 CHECK_NOT 检查
          SourceRange after_not = matchGroup(next_group, source, prev);
          doCheckNot(curr_group, source, prev, after_not);
          prev = after_not;
          ++i; // 已经检查了下一个组
        } else {
          // 如果当前是最后一个组，则在文件末尾执行 CHECK_NOT 检查
          SourceRange end_of_file(
              source, source->size() + 1, source->size() + 1);
          doCheckNot(curr_group, source, prev, end_of_file);
        }
      }
    }
  }

  std::vector<Check> checks;
  std::vector<std::vector<Check>> groups;
};

FileCheck::FileCheck() : fcImpl(new FileCheckImpl()){};

// 输出 FileCheckImpl 对象的检查内容到流中
std::ostream& operator<<(std::ostream& out, const FileCheckImpl& fc) {
  out << "FileCheck checks:\n";
  // 遍历 FileCheckImpl 对象中的每个检查项目，并输出到流中
  for (const Check& c : fc.checks) {
    out << "\t" << c << "\n";
  }
  return out;
};

// FileCheck 类的析构函数，如果实例尚未运行，则输出相应的提示信息和检查内容
FileCheck::~FileCheck() {
  if (!fcImpl->has_run) {
    std::cout << "You have not run this instance of FileCheck!\n";
    std::cout << *fcImpl;
  }
  fcImpl.reset(); // 重置 fcImpl 智能指针，释放资源
};

// 运行文件检查，将测试文件名作为参数传递给内部实现
void FileCheck::run(const std::string& test_file) {
  fcImpl->run(test_file);
};

// 运行图形检查，将图形对象转换为字符串并作为参数传递给内部实现
void FileCheck::run(const Graph& graph) {
  std::stringstream graph_str;
  graph_str << graph;
  fcImpl->run(graph_str.str());
};

// 运行检查，将输入检查字符串和测试字符串作为参数传递给内部实现
void FileCheck::run(
    const std::string& input_checks_string,
    const std::string& test_string) {
  fcImpl->run(input_checks_string, test_string);
}

// 运行检查，将输入检查字符串和图形对象转换后的字符串作为参数传递给内部实现
void FileCheck::run(
    const std::string& input_checks_string,
    const Graph& graph) {
  std::stringstream graph_str;
  graph_str << graph;
  fcImpl->run(input_checks_string, graph_str.str());
}

// 添加一般性检查项目，并返回 FileCheck 对象的指针
FileCheck* FileCheck::check(const std::string& str) {
  fcImpl->addCheck(CHECK, str);
  return this;
}

// 添加非包含检查项目，并返回 FileCheck 对象的指针
FileCheck* FileCheck::check_not(const std::string& str) {
  fcImpl->addCheck(CHECK_NOT, str);
  return this;
}

// 添加相同检查项目，并返回 FileCheck 对象的指针
FileCheck* FileCheck::check_same(const std::string& str) {
  fcImpl->addCheck(CHECK_SAME, str);
  return this;
}

// 添加下一个检查项目，并返回 FileCheck 对象的指针
FileCheck* FileCheck::check_next(const std::string& str) {
  fcImpl->addCheck(CHECK_NEXT, str);
  return this;
}

// 添加计数检查项目，并返回 FileCheck 对象的指针
FileCheck* FileCheck::check_count(
    const std::string& str,
    size_t count,
    bool exactly) {
  TORCH_INTERNAL_ASSERT(
      count != 0 || exactly, "Count == 0 && !exactly doesn't do anything");
  if (count) {
    fcImpl->addCheck(CHECK_COUNT, str, count);
  }
  if (exactly) {
    fcImpl->addCheck(CHECK_NOT, str);
  }
  return this;
}

// 添加 DAG 检查项目，并返回 FileCheck 对象的指针
FileCheck* FileCheck::check_dag(const std::string& str) {
  fcImpl->addCheck(CHECK_DAG, str);
  return this;
}

// 添加源代码高亮检查项目，并返回 FileCheck 对象的指针
FileCheck* FileCheck::check_source_highlighted(const std::string& str) {
  fcImpl->addCheck(CHECK_SOURCE_HIGHLIGHTED, str);
  return this;
}

// 添加正则表达式检查项目，并返回 FileCheck 对象的指针
FileCheck* FileCheck::check_regex(const std::string& str) {
  fcImpl->addCheck(CHECK_REGEX, str);
  return this;
}

// 结束命名空间声明
} // namespace testing
} // namespace jit
} // namespace torch
```