# `.\pytorch\c10\core\impl\LocalDispatchKeySet.h`

```
#pragma once
// 指示编译器仅包含此头文件一次

#include <c10/core/DispatchKeySet.h>
#include <c10/macros/Export.h>
// 包含所需的其他头文件：DispatchKeySet.h 和 Export.h

// TLS management for DispatchKeySet (the "local" DispatchKeySet(s))
// TLS 管理用于 DispatchKeySet 的线程局部存储（"本地" DispatchKeySet）

// This manages two thread-local DispatchKeySets:
// 这管理两个线程局部的 DispatchKeySet：

//  - The included type set, which adds a tensor type for consideration
//    in dispatch.  (For example, you might add Profiling to
//    the included type set to turn on profiling on all tensor operations.)
//    (用于包含的类型集合，将张量类型添加到调度中，例如可以添加 Profiling 以在所有张量操作上开启性能分析)

//  - The excluded type set, which disqualifies a tensor type from dispatch.
//    (Exclusion wins over inclusion.)
//    (用于排除的类型集合，排除指定的张量类型不参与调度，排除优先于包含)

// NB: Originally, I implemented the excluded type set as storing the inverted
// set, but TLS is defined to be zero-initialized, so this doesn't actually work
// (if it's inverted, you want the set to be -1 initialized).
// 注意：最初我将排除类型集合实现为存储反转的集合，但 TLS 被定义为零初始化，因此这实际上不起作用
// （如果是反转的话，你希望集合是 -1 初始化）

namespace c10::impl {

// POD version of LocalDispatchKeySet.  Declared here just so that
// we can put it in the guards.
// This struct encapsulates special handling for TLS initialization
// in set_included()/included() API so that they reflect the truth.
// If you want to create PODLocalDispatchKeySet with non-zero state,
// use set_included() instead of default constructor.
// PODLocalDispatchKeySet 的 POD 版本。这里声明它只是为了能够将其放在 guard 中。
// 此结构体封装了 TLS 初始化中 set_included()/included() API 的特殊处理，以反映它们的真实状态。
// 如果想要创建非零状态的 PODLocalDispatchKeySet，请使用 set_included() 而不是默认构造函数。

struct C10_API PODLocalDispatchKeySet {
  uint64_t included_;
  uint64_t excluded_;

  // See Note [TLS Initialization]
  // 见注释 [TLS 初始化]

  DispatchKeySet included() const {
    return DispatchKeySet(DispatchKeySet::RAW, included_) ^
        c10::default_included_set;
  }
  // 返回包含的 DispatchKeySet，按位异或默认的包含集合

  DispatchKeySet excluded() const {
    return DispatchKeySet(DispatchKeySet::RAW, excluded_) ^
        c10::default_excluded_set;
  }
  // 返回排除的 DispatchKeySet，按位异或默认的排除集合

  void set_included(DispatchKeySet x) {
    included_ = (x ^ c10::default_included_set).raw_repr();
  }
  // 设置包含的 DispatchKeySet，按位异或默认的包含集合，存储原始表示

  void set_excluded(DispatchKeySet x) {
    excluded_ = (x ^ c10::default_excluded_set).raw_repr();
  }
  // 设置排除的 DispatchKeySet，按位异或默认的排除集合，存储原始表示
};

static_assert(
    std::is_trivial_v<PODLocalDispatchKeySet>,
    "PODLocalDispatchKeySet must be a POD type.");
// 静态断言：确保 PODLocalDispatchKeySet 是一个 POD 类型

struct C10_API LocalDispatchKeySet {
  /* implicit */ LocalDispatchKeySet(PODLocalDispatchKeySet x)
      : included_(x.included()), excluded_(x.excluded()) {}
  // 隐式构造函数：从 PODLocalDispatchKeySet 构造 LocalDispatchKeySet

  DispatchKeySet included_;
  DispatchKeySet excluded_;
  // 包含的 DispatchKeySet 和排除的 DispatchKeySet
};

// thread_local variables cannot be C10_API on Windows.
// Inlining this seems to break AutoDispatchBelowAutograd on Android.
// 在 Windows 上，thread_local 变量不能是 C10_API。
// 在 Android 上，内联这个函数似乎会破坏 AutoDispatchBelowAutograd。

#if defined(_MSC_VER) || defined(C10_ANDROID) || defined(C10_IPHONE)
C10_API LocalDispatchKeySet tls_local_dispatch_key_set();
#else // defined(_MSC_VER) || defined(C10_ANDROID) || defined(C10_IPHONE)
extern C10_API thread_local PODLocalDispatchKeySet raw_local_dispatch_key_set;

inline C10_API LocalDispatchKeySet tls_local_dispatch_key_set() {
  // Don't let people fiddle with the thread_local directly just
  // because they include this header.
  // 不要让人们直接操作 thread_local 变量，只是因为他们包含了这个头文件。
  return raw_local_dispatch_key_set;
}
#endif // defined(_MSC_VER) || defined(C10_ANDROID) || defined(C10_IPHONE)

// Internal, use ThreadLocalStateGuard
// 内部使用，使用 ThreadLocalStateGuard
// 设置一个 TLS API，用于强制设置本地调度键集
C10_API void _force_tls_local_dispatch_key_set(LocalDispatchKeySet key_set);

// RAII API，用于操作线程本地调度状态的管理类。

class C10_API IncludeDispatchKeyGuard {
 public:
  // 构造函数，设置包含的调度键集
  IncludeDispatchKeyGuard(DispatchKeySet);
  // 构造函数，设置包含单个调度键
  IncludeDispatchKeyGuard(DispatchKey k)
      : IncludeDispatchKeyGuard(DispatchKeySet(k)) {}
  IncludeDispatchKeyGuard(const IncludeDispatchKeyGuard&) = delete;
  IncludeDispatchKeyGuard operator=(const IncludeDispatchKeyGuard&) = delete;
  IncludeDispatchKeyGuard(IncludeDispatchKeyGuard&&) = delete;
  IncludeDispatchKeyGuard operator=(IncludeDispatchKeyGuard&&) = delete;
  // 析构函数，负责清理工作
  ~IncludeDispatchKeyGuard();

 private:
  // 微小的性能优化，避免在析构时调用 tls_get_addr 函数
  PODLocalDispatchKeySet* tls_;
  DispatchKeySet include_;
};

class C10_API ExcludeDispatchKeyGuard {
 public:
  // 构造函数，设置排除的调度键集
  ExcludeDispatchKeyGuard(DispatchKeySet);
  // 构造函数，设置排除单个调度键
  ExcludeDispatchKeyGuard(DispatchKey k)
      : ExcludeDispatchKeyGuard(DispatchKeySet(k)) {}
  ExcludeDispatchKeyGuard(const ExcludeDispatchKeyGuard&) = delete;
  ExcludeDispatchKeyGuard operator=(const ExcludeDispatchKeyGuard&) = delete;
  ExcludeDispatchKeyGuard(ExcludeDispatchKeyGuard&&) = delete;
  ExcludeDispatchKeyGuard operator=(ExcludeDispatchKeyGuard&&) = delete;
  // 析构函数，负责清理工作
  ~ExcludeDispatchKeyGuard();

 private:
  // 微小的性能优化，避免在析构时调用 tls_get_addr 函数
  PODLocalDispatchKeySet* tls_;
  DispatchKeySet exclude_;
};

// 结构体，用于强制调度键的守卫类
struct C10_API ForceDispatchKeyGuard {
 public:
  // 默认构造函数，保存当前线程的本地调度键集
  ForceDispatchKeyGuard()
      : saved_keyset_(c10::impl::tls_local_dispatch_key_set()) {}
  // 构造函数，强制设置新的本地调度键集
  ForceDispatchKeyGuard(c10::impl::LocalDispatchKeySet key_set)
      : ForceDispatchKeyGuard() {
    c10::impl::_force_tls_local_dispatch_key_set(key_set);
  }
  // 构造函数，设置包含和排除的调度键集合
  ForceDispatchKeyGuard(
      c10::DispatchKeySet include,
      c10::DispatchKeySet exclude)
      : ForceDispatchKeyGuard() {
    auto updated_set = saved_keyset_;
    updated_set.included_ = include;
    updated_set.excluded_ = exclude;
    c10::impl::_force_tls_local_dispatch_key_set(updated_set);
  }
  // 析构函数，恢复之前保存的本地调度键集
  ~ForceDispatchKeyGuard() {
    c10::impl::_force_tls_local_dispatch_key_set(saved_keyset_);
  }

 private:
  c10::impl::LocalDispatchKeySet saved_keyset_;
};

// 非 RAII API，用于操作线程本地调度状态。
// 在可使用 RAII 守卫的情况下，请优先使用 RAII API。
// 非 RAII API 可能在需要在 Python 到 C++ 的多次调用中跨多个调用时
// 需要包含/排除某个 DispatchKey 状态时有用。
//
// 示例用法：一个 Python 上下文管理器包含某个调度键，以确保在上下文管理器下运行的操作
// 通过该调度键的注册覆盖进行调度。
//
// 非 RAII API 不如 RAII 守卫效率高，因为获取器和设置器都会执行 tls_getaddr 查询（RAII 结构体只需要一次！）
C10_API bool tls_is_dispatch_key_excluded(DispatchKey x);
# 设置一个特定的调度键排除在外的函数声明
C10_API void tls_set_dispatch_key_excluded(DispatchKey x, bool desired_state);

# 检查一个特定的调度键是否包含在内的函数声明
C10_API bool tls_is_dispatch_key_included(DispatchKey x);

# 设置一个特定的调度键包含在内的函数声明
C10_API void tls_set_dispatch_key_included(DispatchKey x, bool desired_state);

# 检查一组调度键集合是否被排除在外的函数声明
C10_API bool tls_is_dispatch_keyset_excluded(DispatchKeySet ks);

# 检查一组调度键集合是否被包含在内的函数声明
C10_API bool tls_is_dispatch_keyset_included(DispatchKeySet ks);

} // namespace c10::impl
```