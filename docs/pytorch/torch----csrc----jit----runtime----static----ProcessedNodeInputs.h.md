# `.\pytorch\torch\csrc\jit\runtime\static\ProcessedNodeInputs.h`

```
/**
 * 指令告诉编译器只包含此头文件一次，即使在多个文件中引用。
 */
#pragma once

#include <cstddef>  // 包含用于大小的标准库头文件
#include <cstdint>  // 包含用于固定大小整数类型的标准库头文件

#include <memory>   // 包含智能指针等内存管理工具的标准库头文件

#include <c10/macros/Macros.h>   // 包含C10宏定义的头文件
#include <c10/util/Logging.h>    // 包含C10日志记录工具的头文件

/**
 * ProcessedNodeInputs类，用于存储处理节点的压缩表示的输入索引。
 */
class ProcessedNodeInputs {
 private:
  // 将输入和输出的大小保持在16字节以内；
  // 我们使用12字节，然后使用两个2字节整数存储输出。
  static constexpr size_t kMaxInlineInputs = 5;  // 最大内联输入数为5个

 public:
  /**
   * 默认构造函数，创建一个空的ProcessedNodeInputs对象。
   */
  ProcessedNodeInputs() : ProcessedNodeInputs(0) {}

  /**
   * 带参数的构造函数，根据输入大小创建ProcessedNodeInputs对象。
   */
  explicit ProcessedNodeInputs(size_t size) {
    TORCH_DCHECK_LT(size, (1 << 16));  // 确保输入大小小于65536
    if (size <= kMaxInlineInputs) {
      repr_.inline_repr_.size = size;  // 如果大小小于等于最大内联输入数，使用内联表示
    } else {
      new (&repr_.outline_repr_) HeapArrayPtr(size);  // 否则使用堆数组表示
    }
  }

  /**
   * 重载[]操作符，用于访问输入索引处的值。
   */
  uint16_t operator[](uint16_t idx) const {
    return (*const_cast<ProcessedNodeInputs*>(this))[idx];
  }

  /**
   * 重载[]操作符的非const版本，用于修改输入索引处的值。
   */
  uint16_t& operator[](uint16_t idx) {
    if (C10_LIKELY(repr_.is_inline())) {  // 如果使用内联表示
      TORCH_DCHECK_LT(idx, repr_.inline_repr_.size);  // 确保索引小于内联数组大小
      return repr_.inline_repr_.inputs[idx];  // 返回内联数组中索引处的值的引用
    } else {
      return repr_.outline_repr_[idx];  // 返回堆数组中索引处的值的引用
    }
  }

  /**
   * 返回输入的数量。
   */
  C10_NODISCARD uint16_t size() const {
    if (C10_LIKELY(repr_.is_inline())) {
      return repr_.inline_repr_.size;  // 如果使用内联表示，返回内联数组的大小
    } else {
      return repr_.outline_repr_.size();  // 否则返回堆数组的大小
    }
  }

  /**
   * 判断ProcessedNodeInputs对象是否为空。
   */
  C10_NODISCARD bool empty() const {
    return size() == 0;  // 判断输入的数量是否为0，即对象是否为空
  }

 private:
  /**
   * 内部类HeapArrayPtr，用于处理堆数组的封装。
   */
  class HeapArrayPtr {
   public:
    /**
     * 默认构造函数，创建一个空的HeapArrayPtr对象。
     */
    HeapArrayPtr() = default;

    /**
     * 带参数的构造函数，根据大小创建HeapArrayPtr对象。
     */
    explicit HeapArrayPtr(uint16_t size) : array_(alloc(size)) {}

    /**
     * 复制构造函数，根据另一个HeapArrayPtr对象创建新对象。
     */
    HeapArrayPtr(const HeapArrayPtr& rhs) : array_(alloc(rhs.size())) {
      if (rhs.array_) {
        std::memcpy(
            array_.get(),
            rhs.array_.get(),
            (rhs.size() + 1) * sizeof(uint16_t));  // 复制数组内容
      }
    }

    /**
     * 移动构造函数，根据另一个HeapArrayPtr对象创建新对象（移动语义）。
     */
    HeapArrayPtr(HeapArrayPtr&&) noexcept = default;

    /**
     * 赋值运算符重载，将另一个HeapArrayPtr对象的内容赋值给当前对象。
     */
    HeapArrayPtr& operator=(const HeapArrayPtr& rhs) {
      if (&rhs == this) {
        return *this;  // 如果是同一个对象，直接返回
      }

      if (size() != rhs.size()) {
        array_ = alloc(rhs.size());  // 如果大小不同，重新分配内存
      }

      if (rhs.array_) {
        std::memcpy(
            array_.get(),
            rhs.array_.get(),
            (rhs.size() + 1) * sizeof(uint16_t));  // 复制数组内容
      }
      return *this;
    }

    /**
     * 移动赋值运算符重载，将另一个HeapArrayPtr对象的内容移动给当前对象（移动语义）。
     */
    HeapArrayPtr& operator=(HeapArrayPtr&&) noexcept = default;

    /**
     * 返回HeapArrayPtr对象是否为空。
     */
    C10_NODISCARD bool empty() const {
      return size() != 0;  // 判断堆数组是否为空
    }

    /**
     * 返回HeapArrayPtr对象的大小。
     */
    C10_NODISCARD uint16_t size() const {
      return array_ ? array_[0] : 0;  // 返回堆数组的大小
    }

    /**
     * 重载[]操作符，用于访问数组索引处的值。
     */
    uint16_t operator[](uint16_t idx) const {
      TORCH_DCHECK_LT(idx, size());  // 确保索引小于数组大小
      return array_[idx + 1];  // 返回索引处的值
    }

    /**
     * 重载[]操作符的非const版本，用于修改数组索引处的值。
     */
    uint16_t& operator[](uint16_t idx) {
      TORCH_DCHECK_LT(idx, size());  // 确保索引小于数组大小
      return array_[idx + 1];  // 返回索引处的值的引用
    }

   private:
    std::unique_ptr<uint16_t[]> array_;  // 使用智能指针管理数组内存

    // alloc函数，用于分配指定大小的内存并返回指针
    std::unique_ptr<uint16_t[]> alloc(uint16_t size) {
      return std::make_unique<uint16_t[]>(size + 1);  // 分配内存并返回智能指针
    }
  };

  // union repr_，用于存储内联表示或堆数组表示的选择
  union {
    struct {
      uint16_t size;             // 内联表示的大小
      uint16_t inputs[kMaxInlineInputs];  // 内联表示的输入数组
    } inline_repr_;

    HeapArrayPtr outline_repr_;   // 堆数组表示的封装
  } repr_;  // 表示ProcessedNodeInputs对象的联合数据结构
};
    // 定义静态方法 alloc，返回一个 std::unique_ptr<uint16_t[]>，用于分配一定数量的 uint16_t 类型的数组内存
    static std::unique_ptr<uint16_t[]> alloc(uint16_t num_elts) {
      // 检查 num_elts 是否非零
      if (num_elts) {
        // 使用 std::make_unique 分配 num_elts + 1 个 uint16_t 的内存，并初始化第一个元素为 num_elts
        auto result = std::make_unique<uint16_t[]>(num_elts + 1);
        result[0] = num_elts;
        // 返回分配的内存
        return result;
      } else {
        // 如果 num_elts 为零，则返回空指针
        return nullptr;
      }
    }
  };

  // 我们希望 ProcessedNode 能够在其 ProcessedNodeInputs 后再打包两个 uint16_t 字段，
  // 并且最终会对齐到 8 字节边界。我们可以通过移动 ProcessedNode::outputs_offset_ 和
  // ProcessedNode::num_outputs_ 到这个类来避免使用这个 pragma，但这样做可能会显得笨拙。
#pragma pack(push, 2)
union Repr {
  // 检查当前对象是否使用内联表示
  C10_NODISCARD bool is_inline() const {
    uint8_t tag;
    // 使用 reinterpret_cast 将当前对象的首字节解释为 uint8_t，这是合法的行为；
    // 参考：https://en.cppreference.com/w/cpp/language/reinterpret_cast
    std::memcpy(&tag, reinterpret_cast<const uint8_t*>(this), 1);
    // 当 inline_repr_ 活跃时，HeapArrayPtr 将以普通指针的形式表示，
    // 其对齐至少为2字节边界（因为是 uint16_t*），更可能是8字节或16字节边界，
    // 因为 malloc 倾向于将所有内容对齐到其中之一。因此，我们设置 tag 为1，
    // 以便能够区分这两种情况。
    return (tag & 1) != 0;
  }

  // NOLINTNEXTLINE(modernize-use-equals-default)
  Repr() {}

  // 析构函数，根据当前对象的表示形式销毁其内部资源
  ~Repr() {
    destroyIfOutline();
  }

  // 复制构造函数，根据 rhs 的表示形式创建新对象
  Repr(const Repr& rhs) {
    if (rhs.is_inline()) {
      std::memcpy(&inline_repr_, &rhs.inline_repr_, sizeof(inline_repr_));
    } else {
      new (&outline_repr_) OutlineRepr(rhs.outline_repr_);
    }
  }

  // 复制赋值运算符，根据 rhs 的表示形式复制对象内容
  Repr& operator=(const Repr& rhs) {
    if (&rhs == this) {
      return *this;
    }
    if (rhs.is_inline()) {
      destroyIfOutline();
      new (&inline_repr_) InlineRepr();
      std::memcpy(&inline_repr_, &rhs.inline_repr_, sizeof(inline_repr_));
    } else {
      if (is_inline()) {
        new (&outline_repr_) OutlineRepr(rhs.outline_repr_);
      } else {
        outline_repr_ = rhs.outline_repr_;
      }
    }
    return *this;
  }

  // 移动构造函数，根据 rhs 的表示形式创建新对象（移动语义）
  Repr(Repr&& rhs) noexcept {
    if (rhs.is_inline()) {
      std::memcpy(&inline_repr_, &rhs.inline_repr_, sizeof(inline_repr_));
    } else {
      new (&outline_repr_) OutlineRepr(std::move(rhs.outline_repr_));
    }
  }

  // 移动赋值运算符，根据 rhs 的表示形式复制对象内容（移动语义）
  Repr& operator=(Repr&& rhs) noexcept {
    if (&rhs == this) {
      return *this;
    }

    if (rhs.is_inline()) {
      destroyIfOutline();
      new (&inline_repr_) InlineRepr();
      std::memcpy(&inline_repr_, &rhs.inline_repr_, sizeof(inline_repr_));
    } else {
      if (is_inline()) {
        new (&outline_repr_) OutlineRepr(std::move(rhs.outline_repr_));
      } else {
        outline_repr_ = std::move(rhs.outline_repr_);
      }
    }

    return *this;
  }

  // 内联表示结构体
  struct InlineRepr {
    uint8_t tag = 0x1;
    uint8_t size;
    uint16_t inputs[kMaxInlineInputs];
  };

  // 外部表示类型定义
  using OutlineRepr = HeapArrayPtr;

  // 内联表示对象
  InlineRepr inline_repr_{};
  // 外部表示对象
  OutlineRepr outline_repr_;

 private:
  // 如果当前对象使用外部表示，则销毁其 outline_repr_
  void destroyIfOutline() {
    if (!is_inline()) {
      outline_repr_.~OutlineRepr();
    }
  }
} repr_;
#pragma pack(pop)
};

// 确保 ProcessedNodeInputs 类型的大小为12字节，否则抛出错误信息
static_assert(
    sizeof(ProcessedNodeInputs) == 12,
    "ProcessedNodeInputs has the wrong size!");
```