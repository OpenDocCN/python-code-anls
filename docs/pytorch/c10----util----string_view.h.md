# `.\pytorch\c10\util\string_view.h`

```py
#pragma once

#include <algorithm>  // 包含标准库算法头文件
#include <cstddef>  // 包含标准库大小类型头文件
#include <cstring>  // 包含 C 字符串操作头文件
#include <functional>  // 包含函数对象头文件
#include <iterator>  // 包含迭代器头文件
#include <limits>  // 包含数值限制头文件
#include <ostream>  // 包含输出流头文件
#include <stdexcept>  // 包含标准异常头文件
#include <string>  // 包含字符串头文件
#include <string_view>  // 包含字符串视图头文件

#include <c10/macros/Macros.h>  // 包含 c10 宏定义头文件

namespace c10 {

/**
 * Port of std::string_view with methods from C++20.
 * Implemented following the interface definition in
 * https://en.cppreference.com/w/cpp/string/basic_string_view
 * See there for the API documentation.
 *
 * Difference: We don't have a Traits template parameter because
 * std::char_traits isn't constexpr and we'd have to reimplement
 * std::char_traits if we wanted to use it with our constexpr basic_string_view.
 */
template <class CharT>
class basic_string_view final {
 public:
  using value_type = CharT;  // 定义字符类型
  using pointer = CharT*;  // 定义指针类型
  using const_pointer = const CharT*;  // 定义常量指针类型
  using reference = CharT&;  // 定义引用类型
  using const_reference = const CharT&;  // 定义常量引用类型
  using const_iterator = const CharT*;  // 定义常量迭代器类型
  using iterator = const_iterator;  // 使用常量迭代器类型作为迭代器类型
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;  // 定义常量反向迭代器类型
  using reverse_iterator = const_reverse_iterator;  // 使用常量反向迭代器类型作为反向迭代器类型
  using size_type = std::size_t;  // 定义大小类型为无符号大小类型
  using difference_type = std::ptrdiff_t;  // 定义差别类型为指针间距类型

  static constexpr size_type npos = size_type(-1);  // 定义静态常量 npos，表示无效位置

  constexpr basic_string_view() noexcept : begin_(nullptr) {}  // 默认构造函数，初始化 begin_ 为 nullptr

  explicit constexpr basic_string_view(const_pointer str, size_type count)
      : begin_(str), size_(count) {}  // 显式构造函数，使用给定指针和长度初始化

  /* implicit */ constexpr basic_string_view(const_pointer str)
      : basic_string_view(str, strlen_(str)) {}  // 隐式构造函数，使用给定指针和长度初始化，长度由 strlen_ 函数确定

  /* implicit */ basic_string_view(const ::std::basic_string<CharT>& str)
      : basic_string_view(str.data(), str.size()) {}  // 隐式构造函数，使用 std::basic_string 对象初始化

  constexpr basic_string_view(const basic_string_view&) noexcept = default;  // 拷贝构造函数，默认 noexcept

  constexpr basic_string_view& operator=(
      const basic_string_view& rhs) noexcept {  // 赋值运算符重载，将 rhs 的值赋给当前对象
    begin_ = rhs.begin_;
    size_ = rhs.size_;
    return *this;
  }

  explicit operator ::std::basic_string<CharT>() const {  // 显式类型转换运算符，将当前对象转换为 std::basic_string 对象
    return ::std::basic_string<CharT>(data(), size());
  }

  constexpr const_iterator begin() const noexcept {  // 返回指向首字符的常量迭代器
    return cbegin();
  }

  constexpr const_iterator cbegin() const noexcept {  // 返回指向首字符的常量迭代器
    return begin_;
  }

  constexpr const_iterator end() const noexcept {  // 返回指向尾后位置的常量迭代器
    return cend();
  }

  constexpr const_iterator cend() const noexcept {  // 返回指向尾后位置的常量迭代器
    return begin_ + size_;
  }

  constexpr const_reverse_iterator rbegin() const noexcept {  // 返回指向反向首字符的常量迭代器
    return crbegin();
  }

  constexpr const_reverse_iterator crbegin() const noexcept {  // 返回指向反向首字符的常量迭代器
    return const_reverse_iterator(this->end());
  }

  constexpr const_reverse_iterator rend() const noexcept {  // 返回指向反向尾后位置的常量迭代器
    return crend();
  }

  constexpr const_reverse_iterator crend() const noexcept {  // 返回指向反向尾后位置的常量迭代器
    return const_reverse_iterator(this->begin());
  }

  friend constexpr const_iterator begin(basic_string_view sv) noexcept {  // 友元函数，返回指向首字符的常量迭代器
    return sv.begin();
  }

  friend constexpr const_iterator end(basic_string_view sv) noexcept {  // 友元函数，返回指向尾后位置的常量迭代器
    return sv.end();
  }

  constexpr const_reference operator[](size_type pos) const {  // 下标运算符重载，返回指定位置的常量引用
    // 返回在指定位置 `pos` 的元素的引用
    return at_(pos);
    }
    
    // constexpr 修饰的常量成员函数，用于返回在指定位置 `pos` 的元素的常量引用
    constexpr const_reference at(size_type pos) const {
#if !defined( \
    __CUDA_ARCH__) // CUDA不支持在设备代码中使用std::out_of_range
    // 如果位置超出字符串长度，则抛出out_of_range异常，否则返回指定位置的字符引用
    return C10_UNLIKELY(pos >= size_)
        ? (throw std::out_of_range(
               "string_view::operator[] or string_view::at() out of range. Index: " +
               std::to_string(pos) + ", size: " + std::to_string(size())),
           at_(0))
        : at_(pos);
#else
    // 在CUDA设备代码中直接返回指定位置的字符引用
    return at_(pos);
#endif
  }

  // 返回字符串视图的第一个字符的引用
  constexpr const_reference front() const {
    return *begin_;
  }

  // 返回字符串视图的最后一个字符的引用
  constexpr const_reference back() const {
    return *(begin_ + size_ - 1);
  }

  // 返回指向字符串视图数据的指针
  constexpr const_pointer data() const noexcept {
    return begin_;
  }

  // 返回字符串视图的大小
  constexpr size_type size() const noexcept {
    return size_;
  }

  // 返回字符串视图的长度，与size()函数相同
  constexpr size_type length() const noexcept {
    return size();
  }

  // 返回字符串视图可容纳的最大元素数量
  constexpr size_type max_size() const noexcept {
    return std::numeric_limits<difference_type>::max();
  }

  // 检查字符串视图是否为空
  C10_NODISCARD constexpr bool empty() const noexcept {
    return size() == 0;
  }

  // 移除字符串视图的前缀，抛出out_of_range异常如果n大于视图大小
  constexpr void remove_prefix(size_type n) {
    if (n > size()) {
      throw std::out_of_range(
          "basic_string_view::remove_prefix: out of range. PrefixLength: " +
          std::to_string(n) + ", size: " + std::to_string(size()));
    }
    begin_ += n;
    size_ -= n;
  }

  // 移除字符串视图的后缀，抛出out_of_range异常如果n大于视图大小
  constexpr void remove_suffix(size_type n) {
    if (n > size()) {
      throw std::out_of_range(
          "basic_string_view::remove_suffix: out of range. SuffixLength: " +
          std::to_string(n) + ", size: " + std::to_string(size()));
    }
    size_ -= n;
  }

  // 交换两个字符串视图的内容
  constexpr void swap(basic_string_view& sv) noexcept {
    auto tmp = *this;
    *this = sv;
    sv = tmp;
  }

  // 将字符串视图的一部分复制到目标指针dest中，返回实际复制的元素数量
  size_type copy(pointer dest, size_type count, size_type pos = 0) const {
    if (pos > size_) {
      throw std::out_of_range(
          "basic_string_view::copy: out of range. Index: " +
          std::to_string(pos) + ", size: " + std::to_string(size()));
    }
    size_type copy_length = std::min(count, size_ - pos);
    for (auto iter = begin() + pos, end = iter + copy_length; iter != end;) {
      *(dest++) = *(iter++);
    }
    return copy_length;
  }

  // 返回从指定位置pos开始的子字符串视图，如果pos超出视图大小则抛出out_of_range异常
  constexpr basic_string_view substr(size_type pos = 0, size_type count = npos)
      const {
#if !defined( \
    __CUDA_ARCH__) // CUDA不支持在设备代码中使用std::out_of_range
    // 如果pos超出视图大小，则抛出out_of_range异常，否则返回指定位置和长度的子视图
    return (pos > size_)
        ? (throw std::out_of_range(
               "basic_string_view::substr parameter out of bounds. Index: " +
               std::to_string(pos) + ", size: " + std::to_string(size())),
           substr_())
        : substr_(pos, count);
#else
    // 在CUDA设备代码中直接返回指定位置和长度的子视图
    return substr_(pos, count);
#endif
  }

  // 比较两个字符串视图，返回-1、0或1，表示当前视图小于、等于或大于rhs视图
  constexpr int compare(basic_string_view rhs) const noexcept {
    // 逐字符比较，返回第一个不相等字符的比较结果，或视图大小的比较结果
    for (size_t i = 0, end = std::min(size(), rhs.size()); i < end; ++i) {
      if (at_(i) < rhs.at_(i)) {
        return -1;
      } else if (at_(i) > rhs.at_(i)) {
        return 1;
      }
    }
    // 如果所有字符相等，则比较视图的大小
    if (size() < rhs.size()) {
      return -1;
  } else if (size() > rhs.size()) {
    // 如果当前字符串长度大于 rhs 的长度，则当前字符串大于 rhs，返回 1
    return 1;
  }
  // 否则返回 0，表示两个字符串相等
  return 0;
}

// 比较从指定位置开始的子串与给定的 basic_string_view 对象 v
constexpr int compare(size_type pos1, size_type count1, basic_string_view v) const {
  // 返回从 pos1 位置开始长度为 count1 的子串与 v 的比较结果
  return substr(pos1, count1).compare(v);
}

// 比较两个子串的方法重载，其中一个子串从 pos1 开始，长度为 count1，另一个子串从 v 的 pos2 开始，长度为 count2
constexpr int compare(
    size_type pos1,
    size_type count1,
    basic_string_view v,
    size_type pos2,
    size_type count2) const {
  // 返回从 pos1 开始长度为 count1 的子串与从 v 的 pos2 开始长度为 count2 的子串的比较结果
  return substr(pos1, count1).compare(v.substr(pos2, count2));
}

// 比较当前字符串与以 null 结尾的字符串 s 的方法重载
constexpr int compare(const_pointer s) const {
  // 调用基本的 compare 方法，将 s 转换为 basic_string_view 对象进行比较
  return compare(basic_string_view(s));
}

// 比较从 pos1 开始长度为 count1 的子串与以 null 结尾的字符串 s 的方法重载
constexpr int compare(size_type pos1, size_type count1, const_pointer s) const {
  // 调用基本的 compare 方法，将 s 转换为 basic_string_view 对象进行比较
  return substr(pos1, count1).compare(basic_string_view(s));
}

// 比较从 pos1 开始长度为 count1 的子串与以 null 结尾的字符串 s 的前 count2 个字符的方法重载
constexpr int compare(
    size_type pos1,
    size_type count1,
    const_pointer s,
    size_type count2) const {
  // 调用基本的 compare 方法，将 s 转换为 basic_string_view 对象进行比较
  return substr(pos1, count1).compare(basic_string_view(s, count2));
}

// 比较运算符重载，判断两个 basic_string_view 对象是否相等
friend constexpr bool operator==(basic_string_view lhs, basic_string_view rhs) noexcept {
  // 调用 equals_ 方法比较两个对象
  return lhs.equals_(rhs);
}

// 比较运算符重载，判断两个 basic_string_view 对象是否不相等
friend constexpr bool operator!=(basic_string_view lhs, basic_string_view rhs) noexcept {
  // 直接通过 == 运算符得到结果取反
  return !(lhs == rhs);
}

// 比较运算符重载，判断左侧的 basic_string_view 对象是否小于右侧的对象
friend constexpr bool operator<(basic_string_view lhs, basic_string_view rhs) noexcept {
  // 调用 compare 方法比较，判断 lhs 是否小于 rhs
  return lhs.compare(rhs) < 0;
}

// 比较运算符重载，判断左侧的 basic_string_view 对象是否大于等于右侧的对象
friend constexpr bool operator>=(basic_string_view lhs, basic_string_view rhs) noexcept {
  // 判断 lhs 是否不小于 rhs
  return !(lhs < rhs);
}

// 比较运算符重载，判断左侧的 basic_string_view 对象是否大于右侧的对象
friend constexpr bool operator>(basic_string_view lhs, basic_string_view rhs) noexcept {
  // 调用 < 运算符的重载，判断 rhs 是否小于 lhs
  return rhs < lhs;
}

// 比较运算符重载，判断左侧的 basic_string_view 对象是否小于等于右侧的对象
friend constexpr bool operator<=(basic_string_view lhs, basic_string_view rhs) noexcept {
  // 判断 lhs 是否不大于 rhs
  return !(lhs > rhs);
}

// 检查当前字符串是否以给定的前缀开头
constexpr bool starts_with(basic_string_view prefix) const noexcept {
  // 若前缀长度大于当前字符串长度，则不可能是前缀，返回 false；否则比较前缀与从头开始相同长度的子串
  return (prefix.size() > size()) ? false
                                  : prefix.equals_(substr_(0, prefix.size()));
}

// 检查当前字符串是否以单个字符 prefix 开头
constexpr bool starts_with(CharT prefix) const noexcept {
  // 非空字符串且第一个字符等于 prefix 则返回 true
  return !empty() && prefix == front();
}

// 检查当前字符串是否以以 null 结尾的字符串 prefix 开头
constexpr bool starts_with(const_pointer prefix) const {
  // 调用 starts_with 方法，将 prefix 转换为 basic_string_view 对象进行检查
  return starts_with(basic_string_view(prefix));
}

// 检查当前字符串是否以给定的后缀结尾
constexpr bool ends_with(basic_string_view suffix) const noexcept {
  // 若后缀长度大于当前字符串长度，则不可能是后缀，返回 false；否则比较后缀与从末尾开始相同长度的子串
  return (suffix.size() > size())
      ? false
      : suffix.equals_(substr_(size() - suffix.size(), suffix.size()));
}

// 检查当前字符串是否以单个字符 suffix 结尾
constexpr bool ends_with(CharT suffix) const noexcept {
  // 非空字符串且最后一个字符等于 suffix 则返回 true
  return !empty() && suffix == back();
}

// 检查当前字符串是否以以 null 结尾的字符串 suffix 结尾
constexpr bool ends_with(const_pointer suffix) const {
  // 调用 ends_with 方法，将 suffix 转换为 basic_string_view 对象进行检查
  return ends_with(basic_string_view(suffix));
}

// 在当前字符串中查找子串 v，从指定位置 pos 开始查找，默认从头开始
constexpr size_type find(basic_string_view v, size_type pos = 0) const noexcept {
  // 如果子串长度为 0，直接返回 pos，如果 pos 超过当前字符串长度，返回 npos
  if (v.size() == 0) {
    return pos <= size() ? pos : npos;
  }

  // 从 pos 开始遍历当前字符串，直到 size() - v.size()，与子串 v 比较
  for (size_type cur = pos, end = size() - v.size(); cur <= end; ++cur) {
    // 如果第一个字符匹配，并且剩余部分与子串相同，则返回当前位置 cur
    if (v.at_(0) == at_(cur) &&
        v.substr_(1).equals_(substr_(cur + 1, v.size() - 1))) {
      return cur;
    }
  }
    // 返回 npos，表示未找到指定元素或字符串的位置
    return npos;
  }

  // 在字符串中查找字符 ch 的位置，从指定位置 pos 开始搜索
  constexpr size_type find(CharT ch, size_type pos = 0) const noexcept {
    // 调用 find_first_if_ 函数，查找第一个满足条件 charIsEqual_{ch} 的位置
    return find_first_if_(pos, charIsEqual_{ch});
  }

  // 在字符串中查找指定长度为 count 的字符串 s 的位置，从 pos 开始搜索
  constexpr size_type find(const_pointer s, size_type pos, size_type count)
      const {
    // 调用 find 函数，使用 basic_string_view(s, count) 构造视图并进行查找
    return find(basic_string_view(s, count), pos);
  }

  // 在字符串中查找以 null 结尾的字符串 s 的位置，从 pos 开始搜索
  constexpr size_type find(const_pointer s, size_type pos = 0) const {
    // 调用 find 函数，使用 basic_string_view(s) 构造视图并进行查找
    return find(basic_string_view(s), pos);
  }

  // 在字符串中反向查找 basic_string_view v 的位置，从 pos 开始向前搜索
  constexpr size_type rfind(basic_string_view v, size_type pos = npos)
      const noexcept {
    // 使用迭代方式实现，性能更高
    if (v.size() == 0) {
      // 如果 v 为空，则返回 pos，或者字符串长度 size()，取较小值
      return pos <= size() ? pos : size();
    }

    if (v.size() <= size()) {
      // 将 pos 限制在字符串长度内
      pos = std::min(size() - v.size(), pos);
      do {
        // 检查当前位置 pos 的字符与 v 的第一个字符是否相等，
        // 以及从 pos+1 开始的子串是否与 v 的剩余部分相等
        if (v.at_(0) == at_(pos) &&
            v.substr_(1).equals_(substr_(pos + 1, v.size() - 1))) {
          return pos;
        }
      } while (pos-- > 0);
    }
    // 未找到匹配，则返回 npos
    return npos;
  }

  // 在字符串中反向查找字符 ch 的位置，从 pos 开始向前搜索
  constexpr size_type rfind(CharT ch, size_type pos = npos) const noexcept {
    // 调用 find_last_if_ 函数，查找最后一个满足条件 charIsEqual_{ch} 的位置
    return find_last_if_(pos, charIsEqual_{ch});
  }

  // 在字符串中反向查找指定长度为 count 的字符串 s 的位置，从 pos 开始向前搜索
  constexpr size_type rfind(const_pointer s, size_type pos, size_type count)
      const {
    // 调用 rfind 函数，使用 basic_string_view(s, count) 构造视图并进行反向查找
    return rfind(basic_string_view(s, count), pos);
  }

  // 在字符串中反向查找以 null 结尾的字符串 s 的位置，从 pos 开始向前搜索
  constexpr size_type rfind(const_pointer s, size_type pos = npos) const {
    // 调用 rfind 函数，使用 basic_string_view(s) 构造视图并进行反向查找
    return rfind(basic_string_view(s), pos);
  }

  // 在字符串中查找第一个与 basic_string_view v 中任何字符匹配的位置，从 pos 开始搜索
  constexpr size_type find_first_of(basic_string_view v, size_type pos = 0)
      const noexcept {
    // 调用 find_first_if_ 函数，查找第一个满足条件 stringViewContainsChar_{v} 的位置
    return find_first_if_(pos, stringViewContainsChar_{v});
  }

  // 在字符串中查找第一个与字符 ch 匹配的位置，从 pos 开始搜索
  constexpr size_type find_first_of(CharT ch, size_type pos = 0)
      const noexcept {
    // 调用 find_first_if_ 函数，查找第一个满足条件 charIsEqual_{ch} 的位置
    return find_first_if_(pos, charIsEqual_{ch});
  }

  // 在字符串中查找第一个与以 null 结尾的字符串 s 中任何字符匹配的位置，从 pos 开始搜索
  constexpr size_type find_first_of(
      const_pointer s,
      size_type pos,
      size_type count) const {
    // 调用 find_first_of 函数，使用 basic_string_view(s, count) 构造视图并进行查找
    return find_first_of(basic_string_view(s, count), pos);
  }

  // 在字符串中查找第一个与以 null 结尾的字符串 s 中任何字符匹配的位置，从 pos 开始搜索
  constexpr size_type find_first_of(const_pointer s, size_type pos = 0) const {
    // 调用 find_first_of 函数，使用 basic_string_view(s) 构造视图并进行查找
    return find_first_of(basic_string_view(s), pos);
  }

  // 在字符串中反向查找第一个与 basic_string_view v 中任何字符匹配的位置，从 pos 开始向前搜索
  constexpr size_type find_last_of(basic_string_view v, size_type pos = npos)
      const noexcept {
    // 调用 find_last_if_ 函数，查找最后一个满足条件 stringViewContainsChar_{v} 的位置
    return find_last_if_(pos, stringViewContainsChar_{v});
  }

  // 在字符串中反向查找第一个与字符 ch 匹配的位置，从 pos 开始向前搜索
  constexpr size_type find_last_of(CharT ch, size_type pos = npos)
      const noexcept {
    // 调用 find_last_if_ 函数，查找最后一个满足条件 charIsEqual_{ch} 的位置
    return find_last_if_(pos, charIsEqual_{ch});
  }

  // 在字符串中反向查找第一个与以 null 结尾的字符串 s 中任何字符匹配的位置，从 pos 开始向前搜索
  constexpr size_type find_last_of(
      const_pointer s,
      size_type pos,
      size_type count) const {
    // 调用 find_last_of 函数，使用 basic_string_view(s, count) 构造视图并进行反向查找
    return find_last_of(basic_string_view(s, count), pos);
  }

  // 在字符串中反向查找第一个与以 null 结尾的字符串 s 中任何字符匹配的位置，从 pos 开始向前搜索
  constexpr size_type find_last_of(const_pointer s, size_type pos = npos)
      const {
    // 调用 find_last_of 函数，使用 basic_string_view(s) 构造视图并进行反向查找
    return find_last_of(basic_string_view(s), pos);
  }

  // 在字符串中查找第一个不与 basic_string_view v 中任何字符匹配的位置，从 pos 开始搜索
  constexpr size_type find_first_not_of(basic_string_view v, size_type pos = 0)
      const noexcept {
    // 调用 find_first_if_ 函数，查找第一个满足条件 stringViewDoesNotContainChar_{v} 的位置
    return find_first_if_(pos, stringViewDoesNotContainChar_{v});
  }

  // 在字符串中查找第一个不与字符 ch 匹配的位置，从 pos 开始搜索
  constexpr size_type find_first_not_of(CharT ch, size_type pos = 0)
      const noexcept {
    // 调用 find_first_if_ 函数，查找第一个满足条件 !charIsEqual_{ch} 的位置
    return find_first_if_(pos, charIsNotEqual_{ch});
  }
  // 返回第一个不满足条件的字符的位置
  return find_first_if_(pos, charIsNotEqual_{ch});
}

// 在字符串视图中查找第一个不在指定位置范围内的字符
constexpr size_type find_first_not_of(
    const_pointer s,
    size_type pos,
    size_type count) const {
  return find_first_not_of(basic_string_view(s, count), pos);
}

// 在以空字符结尾的字符数组中查找第一个不在指定位置范围内的字符
constexpr size_type find_first_not_of(const_pointer s, size_type pos = 0)
    const {
  return find_first_not_of(basic_string_view(s), pos);
}

// 在字符串视图中查找最后一个不满足条件的字符
constexpr size_type find_last_not_of(
    basic_string_view v,
    size_type pos = npos) const noexcept {
  return find_last_if_(pos, stringViewDoesNotContainChar_{v});
}

// 在字符串中查找最后一个不等于指定字符的位置
constexpr size_type find_last_not_of(CharT ch, size_type pos = npos)
    const noexcept {
  return find_last_if_(pos, charIsNotEqual_{ch});
}

// 在以空字符结尾的字符数组中查找最后一个不满足条件的字符
constexpr size_type find_last_not_of(
    const_pointer s,
    size_type pos,
    size_type count) const {
  return find_last_not_of(basic_string_view(s, count), pos);
}

// 在以空字符结尾的字符数组中查找最后一个不在指定位置范围内的字符
constexpr size_type find_last_not_of(const_pointer s, size_type pos = npos)
    const {
  return find_last_not_of(basic_string_view(s), pos);
}

private:
// 计算以空字符结尾的字符数组的长度
static constexpr size_type strlen_(const_pointer str) noexcept {
  const_pointer current = str;
  while (*current != '\0') {
    ++current;
  }
  return current - str;
}

// 获取指定位置的字符的引用
constexpr const_reference at_(size_type pos) const noexcept {
  return *(begin_ + pos);
}

// 返回从指定位置开始的子字符串视图
constexpr basic_string_view substr_(size_type pos = 0, size_type count = npos)
    const {
  return basic_string_view{begin_ + pos, std::min(count, size() - pos)};
}

template <class Condition>
// 在字符串中查找第一个满足条件的字符的位置
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr size_type find_first_if_(size_type pos, Condition&& condition)
    const noexcept {
  if (pos + 1 <= size()) {
    for (size_type cur = pos; cur < size(); ++cur) {
      if (condition(at_(cur))) {
        return cur;
      }
    }
  }
  return npos;
}

template <class Condition>
// 在字符串中逆向查找第一个满足条件的字符的位置
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr size_type find_last_if_(size_type pos, Condition&& condition)
    const noexcept {
  // 使用迭代方式实现，速度更快
  if (size() > 0) {
    pos = std::min(size() - 1, pos);
    do {
      if (condition(at_(pos))) {
        return pos;
      }
    } while (pos-- > 0);
  }
  return npos;
}

// 判断当前字符串视图是否与另一个视图相等
constexpr bool equals_(basic_string_view rhs) const {
  // 这里手动实现比较函数，只比较相等性以便进行更优化的代码生成
#if defined(__GNUC__) && !defined(__CUDACC__)
// 如果编译器是 GCC，并且不是 CUDA 编译环境
    return size() == rhs.size() &&
        0 == __builtin_memcmp(data(), rhs.data(), size());
// 比较两个字符串视图的大小和内容是否完全相同，使用内置的内存比较函数
#else
// 否则
    if (size() != rhs.size()) {
      return false;
    }
    // 是的，使用 memcmp 比这个循环要快，但是 memcmp 不是 constexpr 的
    // 而且我不想实现一个 constexpr 的 memcmp 变种。
    // TODO：在某个时候，这个功能可能应该实现，包括一些技巧，比如每次比较一个机器字而不是一个字节。
    for (typename basic_string_view<CharT>::size_type pos = 0; pos < size();
         ++pos) {
      if (at_(pos) != rhs.at_(pos)) {
        return false;
      }
    }
    return true;
// 通过循环逐字符比较两个字符串视图的内容，如果任何字符不同则返回 false，否则返回 true
#endif
  }

  struct charIsEqual_ final {
    CharT expected;
    constexpr bool operator()(CharT actual) const noexcept {
      return expected == actual;
    }
  };

  struct charIsNotEqual_ final {
    CharT expected;
    constexpr bool operator()(CharT actual) const noexcept {
      return expected != actual;
    }
  };

  struct stringViewContainsChar_ final {
    basic_string_view expected;
    constexpr bool operator()(CharT ch) const noexcept {
      return npos != expected.find(ch);
    }
  };

  struct stringViewDoesNotContainChar_ final {
    basic_string_view expected;
    constexpr bool operator()(CharT ch) const noexcept {
      return npos == expected.find(ch);
    }
  };

  const_pointer begin_;
  size_type size_{};
};

template <class CharT>
inline std::basic_ostream<CharT>& operator<<(
    std::basic_ostream<CharT>& stream,
    basic_string_view<CharT> sv) {
  // operator<< 的规则相当复杂，所以我们推迟到 STL 的实现。
  using std_string_type = ::std::basic_string_view<CharT>;
  return stream << std_string_type(sv.data(), sv.size());
  // 使用流操作符输出字符串视图到流，调用 STL 的基础类型以确保正确性。
}

template <class CharT>
constexpr inline void swap(
    basic_string_view<CharT>& lhs,
    basic_string_view<CharT>& rhs) noexcept {
  lhs.swap(rhs);
  // 交换两个字符串视图的内容。
}

using string_view = basic_string_view<char>;

} // namespace c10

namespace std {
template <class CharT>
struct hash<::c10::basic_string_view<CharT>> {
  size_t operator()(::c10::basic_string_view<CharT> x) const {
    // 标准规定 std::string_view 的哈希函数必须与 std::string 的哈希函数相同，但是具体的 std::string 的哈希函数细节由实现者决定。
    // 因此，为了符合标准，我们需要重新使用现有 STL 类型的哈希函数。使用 std::string 的回退方式可能会慢，但这是符合标准的唯一方法。

    using std_string_type = ::std::basic_string_view<CharT>;
    return ::std::hash<std_string_type>{}(std_string_type(x.data(), x.size()));
    // 使用 std::string_view 的数据和大小调用 STL 的哈希函数，确保一致性。
  }
};
} // namespace std
```