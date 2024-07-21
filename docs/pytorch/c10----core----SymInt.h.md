# `.\pytorch\c10\core\SymInt.h`

```py
#pragma once

#include <c10/core/SymBool.h>               // 包含 SymBool 相关头文件
#include <c10/core/SymNodeImpl.h>           // 包含 SymNodeImpl 相关头文件
#include <c10/macros/Export.h>              // 包含导出宏定义头文件
#include <c10/macros/Macros.h>              // 包含宏定义头文件
#include <c10/util/Exception.h>             // 包含异常处理头文件
#include <c10/util/Optional.h>              // 包含可选项头文件

#include <cstdint>                          // 包含标准整数类型头文件
#include <iterator>                         // 包含迭代器头文件
#include <numeric>                          // 包含数值算法头文件
#include <ostream>                          // 包含输出流头文件
#include <type_traits>                      // 包含类型特性头文件

namespace c10 {

class SymFloat;

// SymInt 表示一个 int64_t 类型的常规整数，或者一个符号整数（以类型抹除的方式表示为 SymNode）。
// SymInt 的意图是在运算符内核中进行形状计算时表示符号大小。这允许在程序中跟踪，而不是在内核调用中固定具体大小。
//
// SymInt 具有等效于 int64_t 的 API。特别地，它是一个值类型。
// 在内部，SymInt 以一种巧妙的紧凑方式表示，因此它只占用一个字的空间；但从道德上讲，它是 int64_t 和 SymNodeImpl 的指针之间的联合。
//
// 不变性：引用的 SymNodeImpl 保证是 SymNode，其中 is_int() 返回 true。

class C10_API SymInt {
 public:
  enum Unchecked {
    UNCHECKED,
  };

  /*implicit*/ SymInt(int64_t d) : data_(d) {
    if (is_heap_allocated()) {
      // 如果数值过大为负数，则进行堆分配
      promote_to_negative();
    }
  };
  SymInt() : data_(0) {}                       // 默认构造函数，初始化为0
  SymInt(SymNode n);                          // SymNode 构造函数声明

  // unchecked c-tor accepting raw `data_`
  // 用于在知道非负的情况下构造 symint 的一种适当用法
  SymInt(Unchecked, int64_t d) : data_(d) {}

  // TODO: these implementations are not optimal because they allocate a
  // temporary and then use the move constructor/assignment

  // 拷贝构造函数，如果 s 是堆分配的，则使用 SymNode 构造
  SymInt(const SymInt& s) : data_(0) {
    if (s.is_heap_allocated()) {
      *this = SymInt(s.toSymNode());
    } else {
      data_ = s.data_;
    }
  }

  // 移动构造函数，使用 noexcept 保证不抛出异常
  SymInt(SymInt&& s) noexcept : data_(s.data_) {
    s.data_ = 0;
  }

  // 拷贝赋值运算符重载
  SymInt& operator=(const SymInt& s) {
    if (this != &s) {
      if (s.is_heap_allocated()) {
        *this = SymInt(s.toSymNode());
      } else {
        data_ = s.data_;
      }
    }
    return *this;
  }

  // 移动赋值运算符重载
  SymInt& operator=(SymInt&& s) noexcept {
    if (this != &s) {
      release_();                     // 释放当前的 SymNode（如果有的话）
      data_ = s.data_;
      if (s.is_heap_allocated())
        s.data_ = 0;
    };
    return *this;
  }

  // 返回非拥有的 SymNodeImpl 指针
  SymNodeImpl* toSymNodeImplUnowned() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(is_heap_allocated());
    uint64_t unextended_bits = static_cast<uint64_t>(data_) & ~MASK;
    uint64_t sign_bit_mask = 1ULL << (62 - 1);
    // https://stackoverflow.com/questions/42534749/signed-extension-from-24-bit-to-32-bit-in-c
    uint64_t extended_bits = (unextended_bits ^ sign_bit_mask) - sign_bit_mask;
    // 返回不拥有的 SymNodeImpl 指针

    return reinterpret_cast<SymNodeImpl*>(extended_bits);
  }

 private:
  int64_t data_;               // 数据成员，存储 int64_t 类型数据

  // 内部方法，判断是否堆分配
  bool is_heap_allocated() const {
    return (data_ & HEAP_ALLOCATED_BIT) != 0;
  }

  // 内部方法，释放当前的 SymNode
  void release_() {
    if (is_heap_allocated()) {
      delete reinterpret_cast<SymNodeImpl*>(data_ & ~HEAP_ALLOCATED_BIT);
    }
  }

  // 内部方法，提升为负数
  void promote_to_negative() {
    // 逻辑：将堆分配位设置为1
    data_ |= HEAP_ALLOCATED_BIT;
  }

  static constexpr uint64_t HEAP_ALLOCATED_BIT = 1ULL << 63;   // 堆分配位标志

  static constexpr uint64_t MASK = HEAP_ALLOCATED_BIT - 1;     // 掩码
};
    return static_cast<SymNodeImpl*>(
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<void*>(static_cast<uintptr_t>(extended_bits)));
  }



  // 将 extended_bits 强制类型转换为 SymNodeImpl* 类型并返回
  return static_cast<SymNodeImpl*>(
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<void*>(static_cast<uintptr_t>(extended_bits)));
  }



  void release_() {
    // 如果对象是堆分配的，则释放它
    if (is_heap_allocated()) {
      SymNode::reclaim(toSymNodeImplUnowned()); // steal
    }
  }



  // 摄取（Steal）操作：如果对象是堆分配的，则释放它
  void release_() {
    if (is_heap_allocated()) {
      SymNode::reclaim(toSymNodeImplUnowned()); // steal
    }
  }



  SymNodeImpl* release() && {



  // 移动语义版本的 release 函数，返回 SymNodeImpl* 类型的指针
  SymNodeImpl* release() && {
#ifndef C10_MOBILE
    // 断言是否是堆分配的对象
    TORCH_INTERNAL_ASSERT(is_heap_allocated());
    // 获取未拥有的 SymNodeImpl 对象指针
    auto* r = toSymNodeImplUnowned();
    // 数据归零，转移所有权
    data_ = 0; // transfer ownership
    // 返回未拥有的 SymNodeImpl 对象指针
    return r;
#else
    // 移动端条件不满足时断言失败
    TORCH_INTERNAL_ASSERT(false);
#endif
  }

  // 只有当对象是堆分配的时候才有效
  SymNode toSymNode() const;

  // 确保返回一个 SymNode，必要时使用基类进行封装
  SymNode wrap_node(const SymNode& base) const;

  ~SymInt() {
    release_();
  }

  // 要求该整数非符号化，如果是符号化的则抛出错误
  // 对于那些暂时无法处理符号形状的 C++ 代码是安全的使用方式，
  // 如果在 C++ 中尝试触发路径，会适当地抛出错误
  int64_t expect_int() const {
    if (auto r = maybe_as_int()) {
      return *r;
    }
    // 总是显示 C++ 堆栈跟踪
    TORCH_CHECK_ALWAYS_SHOW_CPP_STACKTRACE(
        false, "when unpacking SymInt, expected int but got ", *this);
  }

  // 检查是否有该整数的提示信息（例如，guard_int 可以使用）
  // 大多数情况下返回 true；仅当未支持的 SymInt 时返回 false
  bool has_hint() const;

  // 插入一个保护条件，使整数成为其具体值，并返回该值
  // 即使整数是符号化的，只要我们知道其基础值是多少，这个操作总是有效的
  // 不要随意使用这个方法；这可能导致 PyTorch 程序的过度特化
  // 应该使用 guard_int(__FILE__, __LINE__) 进行调用，用于诊断过度特化的文件和行号
  int64_t guard_int(const char* file, int64_t line) const;

  // 插入一个保护条件，确保该 SymInt 必须是大小类似的，返回 true 如果整数确实 >= 0
  // 与手动执行 >= 0 测试不同，如果 SymInt 是未支持的 SymInt（或者未来可能包含未支持的 SymInt），我们也将对未支持的 SymInt 进行静态测试 >= 2
  bool expect_size(const char* file, int64_t line) const;

  // 区分实际符号值与存储在堆上的常量
  bool is_symbolic() const {
    return is_heap_allocated() &&
        !toSymNodeImplUnowned()->constant_int().has_value();
  }

  // 注意：在头文件中保持这个定义非常重要
  // 因为我们期望如果检查在移动端编译时会折叠掉无用的代码路径
  C10_ALWAYS_INLINE bool is_heap_allocated() const {
#ifdef C10_MOBILE
    return false;
#else
    return !check_range(data_);
#endif
  }
  #endif
  }

  // 重载运算符 +，返回两个 SymInt 相加的结果
  SymInt operator+(const SymInt& sci) const;
  // 重载运算符 -，返回两个 SymInt 相减的结果
  SymInt operator-(const SymInt& sci) const;
  // 重载运算符 *，返回两个 SymInt 相乘的结果
  SymInt operator*(const SymInt& sci) const;
  // 重载运算符 /，返回两个 SymInt 相除的结果
  SymInt operator/(const SymInt& sci) const;
  // 重载运算符 %，返回两个 SymInt 取模的结果
  SymInt operator%(const SymInt& sci) const;
  // 重载运算符 *=，将当前 SymInt 乘以另一个 SymInt 的结果赋值给当前对象
  void operator*=(const SymInt& sci);
  // 重载运算符 +=，将当前 SymInt 加上另一个 SymInt 的结果赋值给当前对象
  void operator+=(const SymInt& sci);
  // 重载运算符 /=，将当前 SymInt 除以另一个 SymInt 的结果赋值给当前对象
  void operator/=(const SymInt& sci);

  // 返回当前 SymInt 的副本
  SymInt clone() const;

  // 比较当前 SymInt 和另一个 SymInt 是否符号相等，返回 SymBool 对象
  SymBool sym_eq(const SymInt&) const;
  // 比较当前 SymInt 和另一个 SymInt 是否符号不相等，返回 SymBool 对象
  SymBool sym_ne(const SymInt&) const;
  // 比较当前 SymInt 是否小于另一个 SymInt，返回 SymBool 对象
  SymBool sym_lt(const SymInt&) const;
  // 比较当前 SymInt 是否小于等于另一个 SymInt，返回 SymBool 对象
  SymBool sym_le(const SymInt&) const;
  // 比较当前 SymInt 是否大于另一个 SymInt，返回 SymBool 对象
  SymBool sym_gt(const SymInt&) const;
  // 比较当前 SymInt 是否大于等于另一个 SymInt，返回 SymBool 对象
  SymBool sym_ge(const SymInt&) const;

  // 重载运算符 ==，比较当前 SymInt 和另一个 SymInt 是否相等，返回布尔值
  bool operator==(const SymInt& o) const {
    return sym_eq(o).guard_bool(__FILE__, __LINE__);
  }
  // 重载运算符 !=，比较当前 SymInt 和另一个 SymInt 是否不相等，返回布尔值
  bool operator!=(const SymInt& o) const {
    return sym_ne(o).guard_bool(__FILE__, __LINE__);
  }
  // 重载运算符 <，比较当前 SymInt 是否小于另一个 SymInt，返回布尔值
  bool operator<(const SymInt& o) const {
    return sym_lt(o).guard_bool(__FILE__, __LINE__);
  }
  // 重载运算符 <=，比较当前 SymInt 是否小于等于另一个 SymInt，返回布尔值
  bool operator<=(const SymInt& o) const {
    return sym_le(o).guard_bool(__FILE__, __LINE__);
  }
  // 重载运算符 >，比较当前 SymInt 是否大于另一个 SymInt，返回布尔值
  bool operator>(const SymInt& o) const {
    return sym_gt(o).guard_bool(__FILE__, __LINE__);
  }
  // 重载运算符 >=，比较当前 SymInt 是否大于等于另一个 SymInt，返回布尔值
  bool operator>=(const SymInt& o) const {
    return sym_ge(o).guard_bool(__FILE__, __LINE__);
  }

  // 返回当前 SymInt 和另一个 SymInt 中的最小值
  SymInt min(const SymInt& sci) const;
  // 返回当前 SymInt 和另一个 SymInt 中的最大值
  SymInt max(const SymInt& sci) const;

  // 如果两个 SymInt 共享相同节点（用于符号计算），返回 true；如果普通比较相等，返回 false
  bool is_same(const SymInt& other) const;

  // 将 SymInt 转换为 SymFloat
  operator SymFloat() const;

  // 不要使用此函数，推荐使用 maybe_as_int 替代
  // 直接返回当前 SymInt 对象的数据，未检查是否在堆上分配
  int64_t as_int_unchecked() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!is_heap_allocated());
    return data_;
  }

  // 返回当前 SymInt 可能的整数值，使用 std::optional 包装
  std::optional<int64_t> maybe_as_int() const {
    // 如果不在堆上分配，直接返回当前数据
    if (!is_heap_allocated()) {
      return c10::make_optional(data_);
    }
    // 否则，获取节点的常量整数值，或者尝试获取可能的整数值
    auto* node = toSymNodeImplUnowned();
    if (auto c = node->constant_int()) {
      return c;
    }
    return node->maybe_as_int();
  }

  // 检查给定整数是否在不需要堆分配的范围内
  static bool check_range(int64_t i) {
    return i > MAX_UNREPRESENTABLE_INT;
  }

  // 返回作为 SymInt 的最小可表示整数值，无需堆分配
  static constexpr int64_t min_representable_int() {
    // 返回最大不可表示整数 + 1
    return MAX_UNREPRESENTABLE_INT + 1;
    }
    
    private:
    // 提升为负数的函数声明
    void promote_to_negative();
    
    // 内部表示的约束条件：
    //
    // - 应表示正数和小的负数整数
    // - 操作整数时无需转换
    // - 必须表示有效的64位指针
    // - is symbolic test 必须快速执行（两条算术指令太多）。
    //   此代码作为热路径基于 is_heap_allocated() 的 Strobelight 分析。
    //   仅适用于 FB：https://fburl.com/strobelight/5l50ncxd
    //   （您需要更改时间窗口）。
    //
    // 因此，方案是保留大的负数（假设为二进制补码）：
    //
    // - 0b0.... 表示正整数
    // - 0b11... 表示小的负整数
    // - 0b10... 表示指针。这意味着[-2^63, -2^62-1]不能表示为整数。
    //   实际上我们不需要所有这些空间，因为在x86_64上，顶部的16位未使用。
    static constexpr uint64_t MASK = 1ULL << 63 | 1ULL << 62 | 1ULL << 61;
    static constexpr uint64_t IS_SYM = 1ULL << 63 | 1ULL << 61;
    // 我们必须手动将位模式测试转换为大于测试，因为编译器无法自动解析：
    // https://godbolt.org/z/356aferaW
    static constexpr int64_t MAX_UNREPRESENTABLE_INT =
        -1LL & static_cast<int64_t>(~(1ULL << 62));
    int64_t data_;
};

/// Sum of a list of SymInt; accumulates into the c10::SymInt expression
template <
    typename C,
    typename std::enable_if_t<
        std::is_same_v<typename C::value_type, c10::SymInt>,
        int> = 0>
inline c10::SymInt multiply_integers(const C& container) {
  // 使用 std::accumulate 算法对容器中的 SymInt 元素进行累乘，初始值为 c10::SymInt(1)
  return std::accumulate(
      container.begin(),
      container.end(),
      c10::SymInt(1),
      [](const c10::SymInt& a, const c10::SymInt& b) { return a * b; });
}

template <
    typename Iter,
    typename = std::enable_if_t<std::is_same_v<
        typename std::iterator_traits<Iter>::value_type,
        c10::SymInt>>>
inline c10::SymInt multiply_integers(Iter begin, Iter end) {
  // 使用 std::accumulate 算法对迭代器范围内的 SymInt 元素进行累乘，初始值为 c10::SymInt(1)
  return std::accumulate(
      begin,
      end,
      c10::SymInt(1),
      [](const c10::SymInt& a, const c10::SymInt& b) { return a * b; });
}

#define DECLARE_SYMINT_OP_INTONLY(scalar_t, RetTy)      \
  C10_API RetTy operator%(const SymInt& a, scalar_t b); \
  C10_API RetTy operator%(scalar_t a, const SymInt& b);

#define DECLARE_SYMINT_OP(scalar_t, RetTy)              \
  C10_API RetTy operator+(const SymInt& a, scalar_t b); \
  C10_API RetTy operator-(const SymInt& a, scalar_t b); \
  C10_API RetTy operator*(const SymInt& a, scalar_t b); \
  C10_API RetTy operator/(const SymInt& a, scalar_t b); \
  C10_API RetTy operator+(scalar_t a, const SymInt& b); \
  C10_API RetTy operator-(scalar_t a, const SymInt& b); \
  C10_API RetTy operator*(scalar_t a, const SymInt& b); \
  C10_API RetTy operator/(scalar_t a, const SymInt& b); \
  C10_API bool operator==(const SymInt& a, scalar_t b); \
  C10_API bool operator!=(const SymInt& a, scalar_t b); \
  C10_API bool operator<(const SymInt& a, scalar_t b);  \
  C10_API bool operator<=(const SymInt& a, scalar_t b); \
  C10_API bool operator>(const SymInt& a, scalar_t b);  \
  C10_API bool operator>=(const SymInt& a, scalar_t b); \
  C10_API bool operator==(scalar_t a, const SymInt& b); \
  C10_API bool operator!=(scalar_t a, const SymInt& b); \
  C10_API bool operator<(scalar_t a, const SymInt& b);  \
  C10_API bool operator<=(scalar_t a, const SymInt& b); \
  C10_API bool operator>(scalar_t a, const SymInt& b);  \
  C10_API bool operator>=(scalar_t a, const SymInt& b);

DECLARE_SYMINT_OP_INTONLY(int64_t, SymInt)
DECLARE_SYMINT_OP_INTONLY(int32_t, SymInt)
DECLARE_SYMINT_OP_INTONLY(uint64_t, SymInt)
DECLARE_SYMINT_OP_INTONLY(uint32_t, SymInt)
DECLARE_SYMINT_OP(int64_t, SymInt)
DECLARE_SYMINT_OP(int32_t, SymInt) // make sure constants work
DECLARE_SYMINT_OP(uint64_t, SymInt)
DECLARE_SYMINT_OP(uint32_t, SymInt)
DECLARE_SYMINT_OP(double, SymFloat)
DECLARE_SYMINT_OP(float, SymFloat) // just for completeness

// On OSX size_t is different than uint64_t so we have to
// define it separately
#if defined(__APPLE__)
DECLARE_SYMINT_OP_INTONLY(size_t, SymInt)
DECLARE_SYMINT_OP(size_t, SymInt)
#endif

#undef DECLARE_SYMINT_OP

C10_API std::ostream& operator<<(std::ostream& os, const SymInt& s);
C10_API SymInt operator-(const SymInt& s);
// 比较两个 int64_t 类型的数值是否相等，返回布尔值结果
inline bool sym_eq(int64_t a, int64_t b) {
    return a == b;
}

// 比较两个 SymInt 对象是否相等，调用 SymInt 类的 sym_eq 方法进行比较，返回 SymBool 结果
inline SymBool sym_eq(const SymInt& a, const SymInt& b) {
    return a.sym_eq(b);
}

// 比较两个 int64_t 类型的数值是否不相等，返回布尔值结果
inline bool sym_ne(int64_t a, int64_t b) {
    return a != b;
}

// 比较两个 SymInt 对象是否不相等，调用 SymInt 类的 sym_ne 方法进行比较，返回 SymBool 结果
inline SymBool sym_ne(const SymInt& a, const SymInt& b) {
    return a.sym_ne(b);
}

// 比较两个 int64_t 类型的数值是否小于，返回布尔值结果
inline bool sym_lt(int64_t a, int64_t b) {
    return a < b;
}

// 比较两个 SymInt 对象是否小于，调用 SymInt 类的 sym_lt 方法进行比较，返回 SymBool 结果
inline SymBool sym_lt(const SymInt& a, const SymInt& b) {
    return a.sym_lt(b);
}

// 比较两个 int64_t 类型的数值是否小于等于，返回布尔值结果
inline bool sym_le(int64_t a, int64_t b) {
    return a <= b;
}

// 比较两个 SymInt 对象是否小于等于，调用 SymInt 类的 sym_le 方法进行比较，返回 SymBool 结果
inline SymBool sym_le(const SymInt& a, const SymInt& b) {
    return a.sym_le(b);
}

// 比较两个 int64_t 类型的数值是否大于，返回布尔值结果
inline bool sym_gt(int64_t a, int64_t b) {
    return a > b;
}

// 比较两个 SymInt 对象是否大于，调用 SymInt 类的 sym_gt 方法进行比较，返回 SymBool 结果
inline SymBool sym_gt(const SymInt& a, const SymInt& b) {
    return a.sym_gt(b);
}

// 比较两个 int64_t 类型的数值是否大于等于，返回布尔值结果
inline bool sym_ge(int64_t a, int64_t b) {
    return a >= b;
}

// 比较两个 SymInt 对象是否大于等于，调用 SymInt 类的 sym_ge 方法进行比较，返回 SymBool 结果
inline SymBool sym_ge(const SymInt& a, const SymInt& b) {
    return a.sym_ge(b);
}

// 判断给定的 SymBool 对象是否具有提示信息，并调用 guard_bool 方法进行布尔守卫，返回布尔值结果
inline bool definitely_true(
    const c10::SymBool& b,
    const char* file,
    int64_t line) {
    return b.has_hint() && b.guard_bool(file, line);
}

} // namespace c10
```