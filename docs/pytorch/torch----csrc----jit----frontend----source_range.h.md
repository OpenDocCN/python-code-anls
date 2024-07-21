# `.\pytorch\torch\csrc\jit\frontend\source_range.h`

```
#pragma once
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace torch::jit {

// 前向声明，以便在类声明前使用
class SourceRangeUnpickler;
struct SourceRange;

// 一个由 vector<string_view> 支持的类，表示逻辑上连接的字符串
// 这种方式的优势在于不需要连续的内存
struct TORCH_API StringCordView {
  // 默认构造函数
  StringCordView();
  // 拷贝构造函数
  StringCordView(const StringCordView&) = default;
  // 移动构造函数
  StringCordView(StringCordView&&) noexcept = default;
  // 构造函数，接受输入的 string_view 向量和共享指针的向量
  StringCordView(
      std::vector<c10::string_view> inputs,
      std::vector<std::shared_ptr<std::string>> ownerships);

  // 拷贝赋值运算符重载
  StringCordView& operator=(const StringCordView&) = default;
  // 移动赋值运算符重载
  StringCordView& operator=(StringCordView&&) noexcept = default;

  // 返回字符串的总长度
  size_t size() const {
    return accumulated_sizes_.back();
  }

  // 在字符串中从指定位置开始查找给定的字符串 tok，返回找到的位置
  size_t find(const std::string& tok, size_t start) const;
  // 在字符串中从指定位置开始使用正则表达式查找给定的字符串 tok，返回找到的位置
  size_t find_regex(const std::string& tok, size_t start) const;
  // 返回从指定位置开始指定长度的子字符串
  StringCordView substr(size_t start, size_t size) const;

  // 返回指定索引处的字符
  char at(size_t index) const {
    return *iter_for_pos(index);
  }
  // 重载下标操作符，返回指定索引处的字符
  char operator[](size_t index) const {
    return at(index);
  }

  // 将所有的片段拼接成一个完整的字符串并返回
  std::string str() const {
    std::stringstream ss;
    for (auto s : pieces_) {
      ss << std::string(s);
    }
    return ss.str();
  }

  // 重载相等比较操作符，比较该对象与给定字符串是否相等
  bool operator==(const std::string& rhs) const;

  // 重载相等比较操作符，比较该对象与另一个 StringCordView 对象是否相等
  bool operator==(const StringCordView& rhs) const;

  // 返回指定索引处的片段（string_view）
  c10::string_view piece(size_t index) const {
    return pieces_[index];
  }

  // 内部类：迭代器，用于遍历 StringCordView 对象的字符
  struct Iterator {
    // 构造函数，初始化迭代器的位置和状态
    Iterator(
        const StringCordView* str,
        size_t start_line,
        size_t start_pos,
        size_t size)
        : line_(start_line), pos_(start_pos), str_(str), size_(size) {}
    // 显式构造函数，初始化迭代器在整个 StringCordView 中的位置
    explicit Iterator(const StringCordView* str)
        : Iterator(str, 0, 0, str->size()) {}

    // 默认构造函数，初始化迭代器为无效状态
    Iterator() : Iterator(nullptr, 0, 0, 0) {}

    // 拷贝构造函数
    Iterator(const Iterator&) = default;
    // 移动构造函数
    Iterator(Iterator&&) = default;
    // 拷贝赋值运算符重载
    Iterator& operator=(const Iterator&) = default;
    // 移动赋值运算符重载
    Iterator& operator=(Iterator&&) = default;

    // 前置递增运算符重载，使迭代器指向下一个字符位置
    Iterator operator++() {
      if (size_ == 0) {
        return *this;
      }
      if ((pos_ + 1) < str_->pieces_[line_].size()) {
        pos_++;
      } else {
        line_++;
        pos_ = 0;
      }
      return *this;
    }

    // 后置递增运算符重载，返回递增前的迭代器，并使迭代器指向下一个字符位置
    Iterator operator++(int) {
      Iterator prev(*this);
      ++(*this);
      return prev;
    }

    // 返回指向下一个字符的迭代器
    Iterator next_iter() const {
      Iterator next(*this);
      ++next;
      return next;
    }
    // 重载运算符+=，用于迭代器向前移动指定数量的位置
    Iterator& operator+=(size_t num) {
      // 如果已经没有下一个位置可用，则直接返回当前迭代器
      if (!has_next()) {
        return *this;
      }
      // 计算目标位置
      size_t target_pos = pos_ + num;
      // 如果目标位置在当前行的范围内，则更新位置并返回当前迭代器
      if (target_pos >= str_->accumulated_sizes_[line_] &&
          (line_ + 1) < str_->accumulated_sizes_.size() &&
          target_pos < str_->accumulated_sizes_[line_ + 1]) {
        pos_ = target_pos;
        return *this;
      }

      // 计算目标绝对位置并更新当前迭代器为该位置对应的迭代器
      size_t target_abs_pos = pos() + num;
      *this = str_->iter_for_pos(target_abs_pos);
      return *this;
    }

    // 重载运算符==，用于比较两个迭代器是否相等
    bool operator==(const Iterator& rhs) const {
      // 如果两个迭代器都没有下一个位置，则认为它们相等
      if (!has_next() && !rhs.has_next()) {
        return true;
      }
      // 比较两个迭代器的所属字符串、行号和位置是否相同
      return (str_ == rhs.str_) && (line_ == rhs.line_) && (pos_ == rhs.pos_);
    }

    // 重载运算符!=，用于比较两个迭代器是否不相等
    bool operator!=(const Iterator& rhs) {
      return !((*this) == rhs);
    }

    // 返回迭代器是否还有下一个位置
    bool has_next() const {
      return size_ > 0 && (line_ < str_->pieces_.size());
    }

    // 解引用操作符*，返回当前迭代器位置的字符
    char operator*() const {
      // 断言当前行号小于字符串片段数，并且当前位置小于当前行的长度
      TORCH_INTERNAL_ASSERT(line_ < str_->pieces_.size());
      TORCH_INTERNAL_ASSERT(pos_ < str_->pieces_[line_].size());
      return str_->pieces_[line_].at(pos_);
    }

    // 返回当前迭代器所在位置到行末尾的子串
    // 如果当前行号超过了字符串片段数，则返回空串
    c10::string_view rest_line() const {
      if (line_ >= str_->pieces_.size()) {
        return "";
      }

      // 获取当前行的字符串视图，并返回从当前位置到行末尾的子串
      c10::string_view cur_line = str_->pieces_[line_];
      return cur_line.substr(pos_, std::string::npos);
    }

    // 返回当前迭代器的绝对位置
    size_t pos() const {
      // 如果迭代器的大小为0，则返回0
      if (size_ == 0) {
        return 0;
      }
      // 返回当前行的累积大小加上当前位置
      return str_->accumulated_sizes_[line_] + pos_;
    }
// Source 表示一个代码段。它跟踪以下内容：
// - text_view：代码段的文本视图
// - filename（可选）：如果存在，则表示代码段所属文件的名称
// - starting_line_no：表示代码段在原始文件中的起始行号
struct TORCH_API Source {
  // 是否应该在构造函数中复制传入的字符串。
  enum CopiesString { COPIES_STRING, DONT_COPY };

  // 构造函数，接受一个字符串视图、可选的文件名、起始行号、生成范围反序列化器和字符串复制方式。
  explicit Source(
      c10::string_view text_view,
      std::optional<std::string> filename = c10::nullopt,
      size_t starting_line_no = 0,
      std::shared_ptr<SourceRangeUnpickler> gen_ranges = nullptr,
      CopiesString copies_str = COPIES_STRING)
      : filename_(std::move(filename)),
        starting_line_no_(starting_line_no),
        gen_ranges_(std::move(gen_ranges)) {
    // 如果 copies_str 为 COPIES_STRING，则复制 text_view 中的内容到新分配的字符串中。
    if (copies_str == COPIES_STRING) {
      std::shared_ptr<std::string> allocated_str =
          std::make_shared<std::string>(text_view.data(), text_view.size());
      // 使用 StringCordView 包装新分配的字符串，创建 text_view_。
      text_view_ = StringCordView({*allocated_str}, {allocated_str});
    } else {
      // 直接使用 text_view 创建 StringCordView。
      text_view_ = StringCordView({text_view}, {});
    }

    // 计算行的起始偏移量。
    calc_line_start_offsets();
  }

  // 构造函数，接受一个 StringCordView、可选的文件名、起始行号和生成范围反序列化器。
  explicit Source(
      StringCordView str,
      std::optional<std::string> filename = c10::nullopt,
      size_t starting_line_no = 0,
      std::shared_ptr<SourceRangeUnpickler> gen_ranges = nullptr)
      : text_view_(std::move(str)),
        filename_(std::move(filename)),
        starting_line_no_(starting_line_no),
        gen_ranges_(std::move(gen_ranges)) {
    // 计算行的起始偏移量。
    calc_line_start_offsets();
  }

  // 给定行号（在源码中），返回该行的起始字节偏移量。
  size_t offset_for_line(size_t line) const {
    return line_starting_offsets_.at(line);
  }

  // 返回源码中的总行数。
  size_t num_lines() const {
    return line_starting_offsets_.size();
  }

  // 计算给定偏移量所在的行号（在代码段内部）。
  size_t lineno_for_offset(size_t offset) const {
    auto iter = std::upper_bound(
        line_starting_offsets_.begin(), line_starting_offsets_.end(), offset);
    return iter - line_starting_offsets_.begin() - 1;
  }

  // 计算给定行号在原始源文件中的行号（如果有的话）。
  size_t lineno_to_source_lineno(size_t lineno) const {
    if (filename_) {
      return lineno + starting_line_no_;
    } else {
      return lineno;
    }
  }

  // 获取指定行号的文本视图。
  StringCordView get_line(size_t lineno) const {
    auto start = offset_for_line(lineno);
    auto size = (lineno + 1) < num_lines() ? offset_for_line(lineno + 1) - start
                                           : text_view_.size() - start;
    return text_view_.substr(start, size);
  }

  // 返回代码段的文本视图。
  const StringCordView& text_str() const {
    return text_view_;
  }

  // 返回指定索引处的字符。
  char char_at(size_t index) const {
  // 返回 text_view_ 中指定索引位置的字符
  return text_view_.at(index);
}

size_t size() const {
  // 返回 text_view_ 的大小
  return text_view_.size();
}

std::optional<std::string>& filename() {
  // 返回 filename_ 的可选引用
  return filename_;
}

size_t starting_line_no() const {
  // 返回 starting_line_no_ 的值，表示起始行号
  return starting_line_no_;
}

// 查找生成给定源代码范围的源代码范围
std::optional<SourceRange> findSourceRangeThatGenerated(
    const SourceRange& range);

// 默认析构函数
~Source() = default;

private:
void calc_line_start_offsets() {
  // 清空并重新计算行起始偏移量
  line_starting_offsets_.clear();
  line_starting_offsets_.push_back(0);
  size_t pos = 0;
  // 查找每行的起始偏移量，并存储到 line_starting_offsets_
  while ((pos = text_view_.find("\n", pos)) != std::string::npos) {
    line_starting_offsets_.push_back(++pos);
  }
}

StringCordView text_view_;  // 源代码的字符串视图

std::optional<std::string> filename_;  // 源文件名（可选）

size_t starting_line_no_;  // 起始行号（如果 filename_ 不存在，则此值不重要）

std::vector<size_t> line_starting_offsets_;  // 每行起始偏移量的集合

std::shared_ptr<SourceRangeUnpickler> gen_ranges_;  // 生成的源代码范围解析器
// `SourceRange` 结构体表示源码的子集，由源码文本中的 `start` 和 `end` 字节偏移量指定。
struct TORCH_API SourceRange {

  // 构造函数，初始化 `SourceRange` 对象。
  // 参数：
  //   - source_view: 源视图的智能指针，用于访问源码
  //   - start_: 起始偏移量
  //   - end_: 结束偏移量
  SourceRange(std::shared_ptr<Source> source_view, size_t start_, size_t end_)
      : source_view_(std::move(source_view)), start_(start_), end_(end_) {
    // 如果存在源视图，根据起始偏移量获取对应的迭代器
    if (source_view_) {
      start_iter_ = source_view_->text_str().iter_for_pos(start_);
    }
  }

  // 默认构造函数，创建一个空的 `SourceRange` 对象
  SourceRange() : source_view_(nullptr), start_(0), end_(0) {}

  // 构造函数，使用迭代器初始化 `SourceRange` 对象。
  // 参数：
  //   - source_view_: 源视图的智能指针，用于访问源码
  //   - start_iter: 文本迭代器，指定起始位置
  //   - end_: 结束偏移量
  SourceRange(
      std::shared_ptr<Source> source_view_,
      StringCordView::Iterator start_iter,
      size_t end_)
      : source_view_(std::move(source_view_)),
        start_(start_iter.pos()),
        end_(end_),
        start_iter_(start_iter) {}

  // 返回从 `start` 到 `end` 的文本片段
  const c10::string_view token_text() const {
    size_t size = end() - start();
    return start_iter_.rest_line().substr(0, size);
  }

  // 返回从 `start` 到 `end` 的完整文本
  const StringCordView text() const {
    return source_view_->text_str().substr(start(), end() - start());
  }

  // 返回 `SourceRange` 的长度
  size_t size() const {
    return end() - start();
  }

  // `highlight` 方法的常量版本，用于在输出流中显示源码高亮信息
  static const size_t CONTEXT = 3;
  void highlight(std::ostream& out) const;

  // 自定义版本的 `highlight` 方法，用于在输出流中以上下文方式打印源码信息
  void print_with_context(
      std::ostream& out,
      size_t context,
      bool highlight,
      const std::string& funcname) const;

  // 返回源视图的智能指针
  const std::shared_ptr<Source>& source() const {
    return source_view_;
  }

  // 返回 `SourceRange` 的起始偏移量
  size_t start() const {
    return start_;
  }

  // 返回 `SourceRange` 的结束偏移量
  size_t end() const {
    return end_;
  }

  // 返回 `SourceRange` 的高亮字符串表示
  std::string str() const {
    std::stringstream ss;
    highlight(ss);
    return ss.str();
  }

  // 返回源码文件、行号和列号的可选元组
  std::optional<std::tuple<std::string, size_t, size_t>> file_line_col() const {
    if (!source_view_ || !source()->filename()) {
      return c10::nullopt;
    }

    auto lineno = source_view_->lineno_for_offset(start_);
    auto col_offset = (int)start_ - (int)source_view_->offset_for_line(lineno);
    return std::make_tuple<std::string, size_t, size_t>(
        source_view_->filename().value_or(""),
        source_view_->lineno_to_source_lineno(lineno),
        (size_t)col_offset);
  }

  // 比较运算符重载，用于比较两个 `SourceRange` 对象是否相等
  bool operator==(const SourceRange& rhs) const {
    return start() == rhs.start() && end() == rhs.end() &&
        source() == rhs.source();
  }

  // 比较运算符重载，用于比较两个 `SourceRange` 对象是否不相等
  bool operator!=(const SourceRange& rhs) const {
    return !(*this == rhs);
  }

  // 查找生成当前 `SourceRange` 的源码范围，返回一个可选的 `SourceRange` 对象
  std::optional<SourceRange> findSourceRangeThatGenerated() const {
    if (!source_view_) {
      return c10::nullopt;
    }
    return source_view_->findSourceRangeThatGenerated(*this);
  }

 protected:
  std::shared_ptr<Source> source_view_; // 源视图的智能指针，用于访问源码

 private:
  size_t start_; // `SourceRange` 的起始偏移量
  size_t end_;   // `SourceRange` 的结束偏移量
  StringCordView::Iterator start_iter_; // 迭代器，指向 `SourceRange` 的起始位置
};
// 定义一个继承自 SourceRange 的 OwnedSourceRange 结构体
struct OwnedSourceRange : public SourceRange {
  // 显式构造函数，接受一个 SourceRange 对象作为参数，并调用基类的构造函数进行初始化
  explicit OwnedSourceRange(const SourceRange& source_range)
      : SourceRange(source_range) {
    // 获取源码范围的引用
    const auto& source = source_range.source();
    // 如果源码范围有效
    if (source) {
      // 使用源码的文本内容、文件名和起始行号创建一个共享的 Source 对象
      source_view_ = std::make_shared<Source>(
          source->text_str().str(),
          source->filename(),
          source->starting_line_no());
    }
  }
};

// 定义一个用于哈希 SourceRange 的结构体 SourceRangeHasher
struct TORCH_API SourceRangeHasher {
 public:
  // 重载括号操作符，接受一个 torch::jit::SourceRange 对象并返回其哈希值
  size_t operator()(const torch::jit::SourceRange& key) const;
};

// 表示堆栈条目的结构体 StackEntry
struct StackEntry {
  // 文件名
  std::string filename;
  // 源码范围
  SourceRange range;
};

// 格式化堆栈跟踪输出到给定的输出流的函数声明
TORCH_API void format_stack_trace(
    std::ostream& out,
    const std::vector<StackEntry>& entries);

// 重载输出流操作符，以便于输出 SourceRange 对象的高亮信息
inline std::ostream& operator<<(std::ostream& out, const SourceRange& range) {
  range.highlight(out);  // 调用 SourceRange 的 highlight 方法进行高亮输出
  return out;
}

// 描述输出流中特定段落的 (字节偏移量, SourceRange) 对
struct TaggedRange {
  // 构造函数，接受字节偏移量和 SourceRange 对象作为参数，并进行初始化
  TaggedRange(size_t bytes, SourceRange range)
      : bytes(bytes), range(std::move(range)) {}
  size_t bytes;  // 字节偏移量
  SourceRange range;  // 源码范围
};
// 使用 TaggedRange 对象组成的向量来表示源码范围记录
using SourceRangeRecords = std::vector<TaggedRange>;
// 使用哈希表来映射 SourceRange 到整数的结构，以用于标记
using SourceRangeTagMap =
    std::unordered_map<SourceRange, int64_t, SourceRangeHasher>;

// 定义在 torch::jit 命名空间中的内容结束
} // namespace torch::jit

// 为了实现迭代器的特性，特化了 StringCordView::Iterator 的 iterator_traits
namespace std {
template <>
struct iterator_traits<torch::jit::StringCordView::Iterator> {
  using value_type = char;  // 值类型为 char
  using difference_type = ptrdiff_t;  // 差值类型为 ptrdiff_t
  using pointer = char*;  // 指针类型为 char*
  using reference = char&;  // 引用类型为 char&
  using iterator_category = std::forward_iterator_tag;  // 迭代器类型为前向迭代器
};
} // namespace std
```