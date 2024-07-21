# `.\pytorch\torch\csrc\profiler\containers.h`

```py
// 防止头文件被多次包含
#pragma once

// 包含标准库和第三方库的头文件
#include <algorithm>  // 包含算法库，用于各种算法操作
#include <array>      // 包含数组容器的头文件
#include <cstddef>    // 包含标准库定义的常量
#include <cstdint>    // 包含固定大小整数类型的头文件
#include <forward_list>  // 包含单向链表容器的头文件
#include <new>        // 包含内存分配相关的头文件
#include <utility>    // 包含实用工具的头文件，如 std::forward

// 包含 PyTorch 相关的头文件
#include <c10/macros/Macros.h>   // 包含 PyTorch 的宏定义
#include <c10/util/ArrayRef.h>   // 包含 PyTorch 的数组引用类
#include <c10/util/Exception.h>  // 包含 PyTorch 的异常处理类

namespace torch::profiler::impl {

// ============================================================================
// == AppendOnlyList ==========================================================
// ============================================================================

// 在分析性能时，我们有一个非常可预测的访问模式：只向容器的末尾添加元素。
// 我们可以特化并优化 std::vector（必须重新分配内存）和 std::deque（执行双重间接访问），
// 这种操作类似于性能热路径中的专用操作：
//   https://godbolt.org/z/rTjozf1c4
//   https://quick-bench.com/q/mmfuu71ogwaiULDCJyHdKnHZms4    (Prototype #1,
//   int) https://quick-bench.com/q/5vWDW6jjdXVdoffev2zst8D09no    (Prototype
//   #1, int pair) https://quick-bench.com/q/IfEkfAQMeJSNBA52xtMP6Agcl-Q
//   (Prototype #2, int pair)
//   https://quick-bench.com/q/wJV2lKmuXL4XyGJzcI5hs4gEHFg    (Prototype #3, int
//   pair) https://quick-bench.com/q/xiO8ZaBEkYRYUA9dFrMuPLlW9fo    (Full impl,
//   int pair)
// AppendOnlyList 的 emplace 操作的开销比更通用的 STL 容器低 2 倍。

// 最优的 ChunkSize 值会因用例而异，但测试显示，1024 的值很好地摊销了增长的 malloc 成本。
// 对于更大的值，性能会下降，因此如果性能绝对关键，建议进行逐个案例的测试。

template <
    typename T,
    size_t ChunkSize,
    template <typename U, size_t N> class block_t = std::array>
class AppendOnlyList {
 public:
  using array_t = block_t<T, ChunkSize>;

  // 静态断言，检查 array_t 是 std::array<T, ChunkSize> 的子类
  static_assert(
      std::is_base_of_v<std::array<T, ChunkSize>, array_t>,
      "AppendOnlyList expects raw low level pointer storage.");

  // 静态断言，检查 ChunkSize 大于 0
  static_assert(ChunkSize > 0, "Block cannot be empty.");

  // 构造函数，初始化 buffer_last_
  AppendOnlyList() : buffer_last_{buffer_.before_begin()} {}

  // 禁用拷贝构造函数和赋值运算符
  AppendOnlyList(const AppendOnlyList&) = delete;
  AppendOnlyList& operator=(const AppendOnlyList&) = delete;

  // 返回 AppendOnlyList 的大小
  size_t size() const {
    return n_blocks_ * ChunkSize - (size_t)(end_ - next_);
  }

  // emplace_back 函数，向 AppendOnlyList 的末尾插入元素
  template <class... Args>
  T* emplace_back(Args&&... args) {
    maybe_grow();  // 确保有足够的空间

    // 如果 T 和 array_t 都是平凡析构的，则使用 placement new 构造对象
    if constexpr (
        std::is_trivially_destructible_v<T> &&
        std::is_trivially_destructible_v<array_t>) {
      ::new ((void*)next_) T{std::forward<Args>(args)...};
    } else {
      *next_ = T{std::forward<Args>(args)...};  // 否则直接赋值构造
    }

    return next_++;  // 返回插入的元素的指针并移动指针到下一个位置
  }

  // copy 函数，复制给定的数组到 AppendOnlyList
  template <typename T0>
  std::enable_if_t<std::is_same_v<T0, T> && std::is_trivially_copyable_v<T>>
  copy(c10::ArrayRef<T0> src) {
    size_t n = src.size();
    if (C10_UNLIKELY(n == 0)) {  // 如果源数组为空，直接返回
      return;
    }
    maybe_grow();  // 确保有足够的空间
    if (C10_LIKELY(next_ && (next_ + n <= end_))) {
      // 如果条件成立，说明还有足够的空间可以直接使用 memcpy 进行内存拷贝
      std::memcpy((void*)next_, (void*)src.begin(), n * sizeof(T0));
      next_ += n;
    } else {
      // 如果条件不成立，说明空间不足，需要逐个元素插入到列表中
      // 虽然可以分块使用多个 `memcpy`，但由于我们预期这种情况很少发生（n << ChunkSize），性能影响可以忽略不计
      for (auto i : src) {
        emplace_back(i);
      }
    }
  }

  void clear() {
    // 清空 buffer_ 中的所有数据
    buffer_.clear();
    // 重置 buffer_last_ 为 buffer_ 的起始位置之前的位置
    buffer_last_ = buffer_.before_begin();
    // 重置块计数器为 0
    n_blocks_ = 0;
    // 重置 next_ 和 end_ 指针为空指针
    next_ = nullptr;
    end_ = nullptr;
  }

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;

    Iterator(std::forward_list<array_t>& buffer, const size_t size)
        : block_{buffer.begin()}, size_{size} {}

    // End iterator.
    Iterator() = default;

    bool exhausted() const {
      // 检查是否已经迭代完所有元素
      return current_ >= size_;
    }

    reference operator*() const {
      // 返回当前迭代器指向的元素的引用
      return *current_ptr(/*checked=*/true);
    }
    pointer operator->() {
      // 返回当前迭代器指向的元素的指针
      return current_ptr(/*checked=*/true);
    }

    // 前缀递增运算符
    Iterator& operator++() {
      // 如果当前位置为块的末尾，移动到下一个块的开头
      if (!(++current_ % ChunkSize)) {
        block_++;
      }
      return *this;
    }

    // 后缀递增运算符
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const Iterator& a, const Iterator& b) {
      // 比较两个迭代器是否指向相同的位置
      return a.current_ptr() == b.current_ptr();
    }
    friend bool operator!=(const Iterator& a, const Iterator& b) {
      // 比较两个迭代器是否指向不同的位置
      return a.current_ptr() != b.current_ptr();
    }

    std::pair<array_t*, size_t> address() const {
      // 返回当前迭代器指向的元素在数组中的地址
      if (current_ >= size_) {
        return {nullptr, 0};
      }
      return {&(*block_), current_ % ChunkSize};
    }

   private:
    T* current_ptr(bool checked = false) const {
      // 获取当前迭代器指向的元素的指针
      auto a = address();
      if (a.first == nullptr) {
        // 如果地址无效，则根据 checked 参数决定是否断言报错
        TORCH_INTERNAL_ASSERT(!checked, "Invalid access on AppendOnlyList.");
        return nullptr;
      }
      return a.first->data() + a.second;
    }

    typename std::forward_list<array_t>::iterator block_;
    size_t current_{0};
    size_t size_{0};
  };

  Iterator begin() {
    // 返回指向 buffer_ 开始的迭代器
    return Iterator(buffer_, size());
  }
  Iterator end() {
    // 返回表示结束的迭代器
    return Iterator();
  }
  // TODO: cbegin and cend()

 private:
  void maybe_grow() {
    // 如果 next_ 指针已经指向 end_ 指针所在的位置，表示需要增加 buffer_ 的大小
    if (C10_UNLIKELY(next_ == end_)) {
      // 在 buffer_ 的末尾插入一个新块
      buffer_last_ = buffer_.emplace_after(buffer_last_);
      // 块计数器增加
      n_blocks_++;
      // 更新 next_ 指针为新块的起始位置
      next_ = buffer_last_->data();
      // 更新 end_ 指针为新块的末尾位置
      end_ = next_ + ChunkSize;
    }
  }

  std::forward_list<array_t> buffer_;

  // 维护对 buffer_ 的最后一个元素的指针，以便在 O(1) 时间内在末尾插入新元素
  size_t n_blocks_{0};
  T* next_{nullptr};
  T* end_{nullptr};

 protected:
  typename std::forward_list<array_t>::iterator buffer_last_;
};

// 结束 torch::profiler::impl 命名空间的定义
} // namespace torch::profiler::impl
```