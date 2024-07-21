# `.\pytorch\torch\csrc\jit\frontend\source_range.cpp`

```py
// 包含头文件 <c10/util/irange.h>
// 用于范围迭代的实用工具函数
#include <c10/util/irange.h>

// 包含头文件 <torch/csrc/jit/frontend/source_range.h>
// 提供了源代码范围的定义和操作
#include <torch/csrc/jit/frontend/source_range.h>

// 包含头文件 <torch/csrc/jit/serialization/source_range_serialization.h>
// 提供了源代码范围序列化相关的功能
#include <torch/csrc/jit/serialization/source_range_serialization.h>

// 包含头文件 <iostream>
// 标准输入输出流库
#include <iostream>

// 命名空间 torch::jit
namespace torch::jit {

// StringCordView 类的默认构造函数
StringCordView::StringCordView() {
  accumulated_sizes_.push_back(0);
}

// StringCordView 类的构造函数，接受多个 string_view 和 shared_ptr 的输入
StringCordView::StringCordView(
    std::vector<c10::string_view> inputs,
    std::vector<std::shared_ptr<std::string>> ownerships)
    : pieces_(std::move(inputs)), owned_strings_(std::move(ownerships)) {
  accumulated_sizes_.push_back(0);  // 初始化累积大小向量
  size_t running_sum = 0;
  for (auto& s : pieces_) {
    if (!s.empty()) {
      running_sum += s.size();  // 计算累积大小
      accumulated_sizes_.push_back(running_sum);  // 记录每个片段的累积大小
    }
  }
}

// 在 StringCordView 中查找指定的字符串 tok，从位置 start 开始
size_t StringCordView::find(const std::string& tok, size_t start) const {
  if (tok.empty()) {
    return 0;  // 如果目标字符串为空，直接返回 0
  }

  if ((size() - start) < tok.size()) {
    return std::string::npos;  // 如果剩余长度不足以匹配 tok，返回 npos
  }

  Iterator begin = iter_for_pos(start);  // 获取从 start 位置开始的迭代器
  Iterator end_iter = end();  // 获取结束位置的迭代器
  size_t offset = start;
  for (; begin != end_iter; ++begin, ++offset) {
    if (*begin == tok[0]) {  // 如果找到第一个字符匹配
      auto mis = std::mismatch(begin, end_iter, tok.begin(), tok.end());
      if (mis.second == tok.end()) {
        // 没有不匹配，并且目标字符串 tok 已经匹配完全
        return offset;  // 返回找到的位置
      }
      if (mis.first == end_iter) {
        // 当前字符串片段已经匹配完，但是 tok 还没有匹配完全
        return std::string::npos;  // 返回 npos
      }
    }
  }
  return std::string::npos;  // 没有找到匹配的字符串，返回 npos
}

// 在 StringCordView 中使用正则表达式 tok 查找，从位置 start 开始
size_t StringCordView::find_regex(const std::string& tok, size_t start) const {
  if (tok.empty()) {
    return 0;  // 如果正则表达式为空，返回 0
  }

  const std::string& target = this->substr(start, this->size()).str();  // 获取子串并转换为字符串
  std::smatch sm;
  const std::regex re(tok);  // 编译正则表达式

  auto regex_found = std::regex_search(target, sm, re);  // 在 target 中查找正则表达式

  return regex_found ? sm.position(0) : std::string::npos;  // 返回匹配位置或者 npos
}

// 返回 StringCordView 的子串，从 start 开始，长度为 size
StringCordView StringCordView::substr(size_t start, size_t size) const {
  std::vector<c10::string_view> pieces;
  std::vector<std::shared_ptr<std::string>> ownerships;
  if (start >= this->size()) {
    // 如果 start 超出了字符串长度，返回空的 StringCordView
    return StringCordView();
  }
  if (start + size >= this->size()) {
    size = this->size() - start;  // 如果长度超出范围，截断为剩余长度
  }
  Iterator begin = iter_for_pos(start);  // 获取开始位置的迭代器
  Iterator end = iter_for_pos(start + size);  // 获取结束位置的迭代器

  if (begin.line_ == end.line_) {
    // 如果在同一行
    pieces.push_back(pieces_[begin.line_].substr(begin.pos_, size));  // 添加子串片段
  } else {
    pieces.push_back(pieces_[begin.line_].substr(begin.pos_));  // 添加第一行的子串

    size_t last_line = pieces_.size();
    if (end != this->end() && end.line_ < last_line) {
      last_line = end.line_;  // 结束位置在有效范围内
    }
    for (size_t i = begin.line_ + 1; i < last_line; i++) {
      pieces.push_back(pieces_[i]);  // 添加中间行的完整片段
    }
    if (end != this->end()) {
      pieces.push_back(pieces_[end.line_].substr(0, end.pos_));  // 添加最后一行的子串
    }
  }
  // 返回新的 StringCordView 对象，包含所需的子串
  return StringCordView(std::move(pieces), std::move(ownerships));
}
    }
  }


// 结束循环和函数的定义

// share ownership
std::copy(
    owned_strings_.begin(),
    owned_strings_.end(),
    std::back_inserter(ownerships));


// 将 owned_strings_ 中的所有元素复制到 ownerships 容器的末尾

return StringCordView(std::move(pieces), std::move(ownerships));


// 返回一个 StringCordView 对象，使用 std::move 将 pieces 和 ownerships 的所有权移动到新对象中
}

// 重载操作符==，用于比较StringCordView对象与std::string对象是否相等
bool StringCordView::operator==(const std::string& rhs) const {
    // 如果大小不相等，则返回false
    if (size() != rhs.size()) {
        return false;
    }
    // 使用std::mismatch比较StringCordView和std::string的内容，直到两者都耗尽
    auto res = std::mismatch(begin(), end(), rhs.begin(), rhs.end());
    // 返回比较结果：两者都耗尽则相等
    return res.first == end() && res.second == rhs.end();
}

// 重载操作符==，用于比较两个StringCordView对象是否相等
bool StringCordView::operator==(const StringCordView& rhs) const {
    // 如果大小不相等，则返回false
    if (size() != rhs.size()) {
        return false;
    }
    // 使用std::mismatch比较两个StringCordView对象的内容，直到两者都耗尽
    auto res = std::mismatch(begin(), end(), rhs.begin(), rhs.end());
    // 返回比较结果：两者都耗尽则相等
    return res.first == end() && res.second == rhs.end();
}

// 返回指定位置的迭代器
StringCordView::Iterator StringCordView::iter_for_pos(size_t pos) const {
    // 如果位置为0，返回begin()迭代器
    if (pos == 0) {
        return begin();
    }
    // 如果位置超过大小，返回end()迭代器
    if (pos >= size()) {
        return end();
    }
    // 使用std::upper_bound找到累积大小中第一个大于pos的位置
    auto upper = std::upper_bound(
        accumulated_sizes_.begin(), accumulated_sizes_.end(), pos);
    // 如果upper等于累积大小的末尾，返回end()迭代器
    if (upper == accumulated_sizes_.end()) {
        return end();
    }
    // 计算行数，并进行断言检查
    size_t line = upper - accumulated_sizes_.begin() - 1;
    assert(accumulated_sizes_[line] <= pos);
    assert(accumulated_sizes_[line + 1] > pos);
    // 返回新的迭代器
    return Iterator(this, line, pos - accumulated_sizes_[line], size() - pos);
}

// 自定义哈希函数，用于torch::jit::SourceRange对象的哈希计算
size_t SourceRangeHasher::operator()(const torch::jit::SourceRange& key) const {
    return (
        std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(key.source().get())) ^
        std::hash<size_t>()(key.start()) ^ std::hash<size_t>()(key.end()));
}

// 查找生成给定SourceRange的原始SourceRange
std::optional<SourceRange> Source::findSourceRangeThatGenerated(
    const SourceRange& range) {
    // 如果gen_ranges_为空，返回空的optional对象
    if (!gen_ranges_) {
        return c10::nullopt;
    }
    // 调用gen_ranges_的findSourceRangeThatGenerated方法
    return gen_ranges_->findSourceRangeThatGenerated(range);
}

// 在输出流中高亮显示SourceRange
void SourceRange::highlight(std::ostream& out) const {
    // 获取原始SourceRange，如果存在的话
    if (auto orig_source_range = findSourceRangeThatGenerated()) {
        // 调用原始SourceRange的highlight方法
        orig_source_range->highlight(out);
        out << "Serialized ";
    }
    // 调用print_with_context方法输出当前SourceRange的上下文
    print_with_context(out, CONTEXT, true, "");
}

// 格式化堆栈跟踪信息
void format_stack_trace(
    std::ostream& out,
    const std::vector<StackEntry>& entries) {
    // 是否存在原始范围的标志和原始范围的容器
    bool has_orig_ranges = false;
    std::vector<SourceRange> orig_ranges;
    // 收集原始范围，如果某些帧没有原始范围，则用当前范围替代
    for (const StackEntry& entry : entries) {
        // 查找生成给定range的原始SourceRange
        if (auto orig_source_range = entry.range.findSourceRangeThatGenerated()) {
            orig_ranges.emplace_back(std::move(orig_source_range.value()));
            has_orig_ranges = true;
        } else {
            orig_ranges.emplace_back(entry.range);
        }
    }
    // 输出TorchScript的堆栈跟踪信息
    out << "Traceback of TorchScript";
    // 如果存在原始范围，追加"，serialized code"信息
    if (has_orig_ranges) {
        out << ", serialized code";
    }
    // 输出最近调用的提示信息
    out << " (most recent call last):\n";
    // 逐个打印堆栈条目的上下文
    for (const StackEntry& entry : entries) {
        entry.range.print_with_context(
            out, SourceRange::CONTEXT, true, entry.filename);
    }
    // 如果存在原始范围，输出原始代码的堆栈跟踪信息
    if (has_orig_ranges) {
        out << "\nTraceback of TorchScript, original code (most recent call last):\n";
        // 从entries的起始位置开始打印
        auto it = entries.begin();
    // 对 orig_ranges 中的每个 SourceRange 对象进行遍历
    for (const SourceRange& range : orig_ranges) {
      // 调用 SourceRange 对象的 print_with_context 方法，
      // 将范围信息打印到输出流 out 中，打印时包含上下文 CONTEXT，打印完整行并且输出文件名
      range.print_with_context(
          out, SourceRange::CONTEXT, true, (*it++).filename);
    }
  }
}

// 打印带有上下文的源代码范围
void SourceRange::print_with_context(
    std::ostream& out,                     // 输出流对象
    size_t context,                        // 上下文行数
    bool highlight,                        // 是否高亮显示
    const std::string& funcname) const {   // 函数名称参数

  // 如果源视图为空，则表示空的SourceRange，作为一个哨兵值
  if (!source_view_) {
    return;
  }

  auto str = source_view_->text_str().str(); // 获取源代码文本字符串

  // 如果范围大小与整个文件大小相同，则打印整个文件内容
  if (size() == str.size()) {
    out << str; // 输出整个文件内容
    return;
  }

  size_t range_end =
      (str.size() < end()
           ? str.size()
           : end()); // 计算实际范围结束位置，避免超出字符串长度

  // 确定要高亮显示的上下文行范围
  size_t begin_line = start(); // 高亮显示的起始行
  size_t end_line = range_end; // 高亮显示的结束行
  if (begin_line > str.size()) {
    return;
  }
  while (begin_line > 0 && str[begin_line - 1] != '\n')
    --begin_line;
  while (end_line < str.size() && str[end_line] != '\n')
    ++end_line;
  AT_ASSERT(begin_line == 0 || str[begin_line - 1] == '\n');
  AT_ASSERT(end_line == str.size() || str[end_line] == '\n');

  size_t begin_context = begin_line; // 上下文开始位置，高亮行之前的CONTEXT行数
  for (size_t i = 0; begin_context > 0; --begin_context) {
    if (str[begin_context - 1] == '\n') {
      ++i;
    }
    if (i >= context) {
      break;
    }
  }
  AT_ASSERT(begin_context == 0 || str[begin_context - 1] == '\n');

  size_t end_context =
      end_line; // 上下文结束位置，高亮行之后的CONTEXT行数
  for (size_t i = 0; end_context < str.size(); ++end_context) {
    if (str[end_context] == '\n') {
      ++i;
    }
    if (i >= context) {
      break;
    }
  }
  AT_ASSERT(end_context == str.size() || str[end_context] == '\n');

  // 打印位置信息
  if (auto flc = file_line_col()) {
    auto [filename, line, col] = *flc;
    out << "  File \"" << filename << "\", line " << line;
    if (!funcname.empty()) {
      out << ", in " << funcname;
    }
    out << "\n";
  }

  // 打印初始上下文内容
  out << str.substr(begin_context, start() - begin_context);
  size_t line_start = start();
  size_t line_end = range_end;
  if (highlight) {
    line_end = start(); // 如果需要高亮显示，则修改结束行位置
    while (line_start < range_end) {
      // 当前行的起始位置小于结束位置时，继续处理
      // 将 line_end 移动到行末尾
      while (line_end < str.size() && str[line_end] != '\n') {
        ++line_end;
      }
      // 截取实际行的内容
      auto actual_line = str.substr(line_start, (line_end - line_start) + 1);
      out << actual_line;
      // 如果实际行最后一个字符不是换行符，添加换行符
      if (actual_line.back() != '\n') {
        out << "\n";
      }

      // 初始化空白和高亮空间计数器
      size_t empty_space = 0;
      size_t highlight_space = 0;
      size_t hightlight_begin = line_start;
      size_t highlight_end = line_start;
      // 确定当前高亮的行的起始位置
      while (hightlight_begin > 0 && str[hightlight_begin - 1] != '\n') {
        --hightlight_begin;
      }
      // 确定当前高亮的行的结束位置
      while (highlight_end < range_end && str[highlight_end] != '\n') {
        ++highlight_end;
      }
      // 断言高亮行的起始和结束位置的正确性
      AT_ASSERT(hightlight_begin == 0 || str[hightlight_begin - 1] == '\n');
      AT_ASSERT(highlight_end == range_end || str[highlight_end] == '\n');
      
      // 计算当前行中空白和高亮空间的数量
      for (const auto i : c10::irange(hightlight_begin, highlight_end)) {
        if (str[i] == ' ' || i < start()) {
          empty_space++;
        } else {
          break;
        }
      }
      highlight_space = highlight_end - hightlight_begin - empty_space;
      // 如果存在高亮空间，则输出波浪线或指示位置
      if (highlight_space > 0) {
        // 检查是否有更多行需要打印
        bool more_lines = false;
        for (size_t i = line_end; i <= range_end; i++) {
          if (str[i] != '\n' && str[i] != ' ') {
            more_lines = true;
          }
        }
        // 输出空白和高亮波浪线
        out << std::string(empty_space, ' ');
        out << std::string(highlight_space, '~');
        out << (more_lines && line_end != range_end ? "\n" : " <--- HERE\n");
      }
      // 更新 line_end 和 line_start
      ++line_end;
      line_start = line_end;
    }
  } else {
    // 若不需要高亮，则直接打印代码
    out << str.substr(start(), range_end - start());
  }
  // 打印末尾的上下文
  if (line_end <= str.size()) {
    auto line_substr = str.substr(line_end, end_context - line_end);
    out << line_substr;
    // 如果末尾上下文不为空且最后一个字符不是换行符，则添加换行符
    if (!line_substr.empty() && line_substr.back() != '\n') {
      out << "\n";
    }
  }
}

} // namespace torch::jit
```