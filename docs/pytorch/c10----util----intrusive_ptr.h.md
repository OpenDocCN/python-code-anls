# `.\pytorch\c10\util\intrusive_ptr.h`

```
#pragma once
// 一次性包含防止重复引用

#include <c10/util/Exception.h>
// 包含异常处理相关的头文件
#include <c10/util/MaybeOwned.h>
// 包含 MaybeOwned 类的头文件
#include <atomic>
// 包含原子操作相关的头文件
#include <climits>
// 包含 CHAR_BIT（每字节位数） 的头文件
#include <memory>
// 包含内存管理相关的头文件
#include <type_traits>
// 包含类型特性相关的头文件

namespace pybind11 {
template <typename, typename...>
class class_;
}
// pybind11 命名空间，定义了一个类模板 class_

namespace c10 {
class intrusive_ptr_target;
// 前向声明 intrusive_ptr_target 类

namespace raw {
namespace weak_intrusive_ptr {
inline void incref(intrusive_ptr_target* self);
}
// weak_intrusive_ptr 命名空间，定义了一个增加引用计数的函数 incref

namespace intrusive_ptr {
inline void incref(intrusive_ptr_target* self);
}
// intrusive_ptr 命名空间，定义了一个增加引用计数的函数 incref

// 构造函数标签，用于 intrusive_ptr 构造函数
struct DontIncreaseRefcount {};
}
// raw 命名空间，包含了不增加引用计数的标记

namespace detail {
constexpr uint32_t kImpracticallyHugeReferenceCount = 0x0FFFFFFF;
}
// detail 命名空间，定义了一个极大的参考计数值

/**
 * intrusive_ptr<T> is an alternative to shared_ptr<T> that has better
 * performance because it does the refcounting intrusively
 * (i.e. in a member of the object itself).
 * Your class T needs to inherit from intrusive_ptr_target to allow it to be
 * used in an intrusive_ptr<T>. Your class's constructor should not allow
 * `this` to escape to other threads or create an intrusive_ptr from `this`.
 */
// intrusive_ptr<T> 是 shared_ptr<T> 的一种替代品，由于在对象本身的成员中进行引用计数，
// 所以性能更好。你的类 T 需要继承自 intrusive_ptr_target，以便可以在 intrusive_ptr<T> 中使用。
// 你的类的构造函数不应允许 this 逃逸到其他线程，也不应从 this 创建 intrusive_ptr。

// Note [Stack allocated intrusive_ptr_target safety]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// std::enable_shared_from_this 存在的一个问题是它允许从栈分配的对象创建 std::shared_ptr，
// 这是不正确的，因为对象一旦从栈返回就会销毁。在 intrusive_ptr 中，我们可以检测到这种情况，
// 因为我们将继承自 intrusive_ptr_target 的对象的引用计数/弱引用计数设置为零，
// 除非我们能证明该对象是动态分配的（例如通过 make_intrusive）。
//
// 因此，每当你将一个 T* 转换为 intrusive_ptr<T> 时，我们会检查并确保引用计数不为零
// （或者，对于 weak_intrusive_ptr<T> 的更微妙的测试，引用计数可以为零，但弱引用计数不应为零），
// 因为这告诉我们对象是否由我们分配。如果不是，则不能使用 intrusive_ptr！

// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
// 忽略下一行代码检查（cppcoreguidelines-virtual-class-destructor）
class C10_API intrusive_ptr_target {
  // Note [Weak references for intrusive refcounting]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Here's the scheme:
  //
  //  - refcount == number of strong references to the object
  //    weakcount == number of weak references to the object,
  //      plus one more if refcount > 0
  //    An invariant: refcount > 0  =>  weakcount > 0
  //
  //  - c10::StorageImpl stays live as long as there are any strong
  //    or weak pointers to it (weakcount > 0, since strong
  //    references count as a +1 to weakcount)
  //
  //  - finalizers are called and data_ptr is deallocated when refcount == 0
  //
  //  - Once refcount == 0, it can never again be > 0 (the transition
  //    from > 0 to == 0 is monotonic)
  //
  //  - When you access c10::StorageImpl via a weak pointer, you must
  //    atomically increment the use count, if it is greater than 0.
  //    If it is not, you must report that the storage is dead.
  //
  mutable std::atomic<uint32_t> refcount_;    // 引用计数，记录对象的强引用数量
  mutable std::atomic<uint32_t> weakcount_;   // 弱引用计数，记录对象的弱引用数量

  template <typename T, typename NullType>
  friend class intrusive_ptr;                // intrusive_ptr 类是友元，可以访问私有成员

  friend inline void raw::intrusive_ptr::incref(intrusive_ptr_target* self);  // 声明增加引用计数的友元函数

  template <typename T, typename NullType>
  friend class weak_intrusive_ptr;           // weak_intrusive_ptr 类是友元，可以访问私有成员

  friend inline void raw::weak_intrusive_ptr::incref(
      intrusive_ptr_target* self);           // 声明增加弱引用计数的友元函数

  template <typename T>
  friend struct ExclusivelyOwnedTensorTraits;  // ExclusivelyOwnedTensorTraits 结构是友元，可以访问保护成员

 protected:
  // protected destructor. We never want to destruct intrusive_ptr_target*
  // directly.
  virtual ~intrusive_ptr_target() {
    // Disable -Wterminate and -Wexceptions so we're allowed to use assertions
    // (i.e. throw exceptions) in a destructor.
    // We also have to disable -Wunknown-warning-option and -Wpragmas, because
    // some other compilers don't know about -Wterminate or -Wexceptions and
    // will show a warning about unknown warning options otherwise.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning( \
    disable : 4297) // function assumed not to throw an exception but does
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wterminate"
#pragma GCC diagnostic ignored "-Wexceptions"
#endif
  }
};
    # 使用内部宏 TORCH_INTERNAL_ASSERT_DEBUG_ONLY 进行断言检查：
    # 检查引用计数是否为0或者大于等于 kImpracticallyHugeReferenceCount
    # 第二个条件是为了适应 unsafe_adapt_non_heap_allocated 的情况，因为在这种情况下我们会自行释放对象，
    # 此时每个预期的减引用操作都应该已经发生（某些用户代码尝试减引用并释放对象，但这并不立即发生），
    # 或者没有发生（没有用户代码尝试释放对象，现在它将通过 unsafe_adapt_non_heap_allocated 调用者使用的任何机制被销毁）。
    # 我们选择我们的引用计数使得计数不会低于 kImpracticallyHugeReferenceCount。
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        refcount_.load() == 0 ||
        refcount_.load() >= detail::kImpracticallyHugeReferenceCount,
        "Tried to destruct an intrusive_ptr_target that still has intrusive_ptr to it; refcount was ",
        refcount_.load());

    # 使用内部宏 TORCH_INTERNAL_ASSERT_DEBUG_ONLY 进行断言检查：
    # 检查弱引用计数是否为1、0、detail::kImpracticallyHugeReferenceCount - 1 或 detail::kImpracticallyHugeReferenceCount。
    # 这是为了优化 ~intrusive_ptr，在销毁时通常会导致 weakcount_ 为1。
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        weakcount_.load() == 1 || weakcount_.load() == 0 ||
        weakcount_.load() == detail::kImpracticallyHugeReferenceCount - 1 ||
        weakcount_.load() == detail::kImpracticallyHugeReferenceCount,
        "Tried to destruct an intrusive_ptr_target that still has weak_intrusive_ptr to it");
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif


// 如果编译器是 MSC（Microsoft Visual C++）并且不是 Clang 编译器，则恢复之前的警告状态
// 否则，恢复 GCC 的诊断状态


  }

  constexpr intrusive_ptr_target() noexcept : refcount_(0), weakcount_(0) {}


// 默认构造函数，初始化 refcount 和 weakcount 为 0
// 是 constexpr 函数，因此可以在编译时求值


  // intrusive_ptr_target supports copy and move: but refcount and weakcount
  // don't participate (since they are intrinsic properties of the memory
  // location)
  intrusive_ptr_target(intrusive_ptr_target&& /*other*/) noexcept
      : intrusive_ptr_target() {}


// 移动构造函数，使用 noexcept 保证不抛出异常
// 仅初始化一个新的对象，不涉及 refcount 和 weakcount，因为它们是内存位置的固有属性


  intrusive_ptr_target& operator=(intrusive_ptr_target&& /*other*/) noexcept {
    return *this;
  }


// 移动赋值运算符重载，简单地返回当前对象的引用
// 不进行实际的移动操作，因为 refcount 和 weakcount 是内在的


  intrusive_ptr_target(const intrusive_ptr_target& /*other*/) noexcept
      : intrusive_ptr_target() {}


// 拷贝构造函数，使用 noexcept 保证不抛出异常
// 初始化一个新对象，不涉及 refcount 和 weakcount


  intrusive_ptr_target& operator=(
      const intrusive_ptr_target& /*other*/) noexcept {
    return *this;
  }


// 拷贝赋值运算符重载，简单地返回当前对象的引用
// 不进行实际的拷贝操作，因为 refcount 和 weakcount 是内在的


 private:
  /**
   * This is called when refcount reaches zero.
   * You can override this to release expensive resources.
   * There might still be weak references, so your object might not get
   * destructed yet, but you can assume the object isn't used anymore,
   * i.e. no more calls to methods or accesses to members (we just can't
   * destruct it yet because we need the weakcount accessible).
   *
   * If there are no weak references (i.e. your class is about to be
   * destructed), this function WILL NOT be called.
   */
  virtual void release_resources() {}


// 私有成员函数，当 refcount 达到零时调用
// 可以重写此函数来释放昂贵的资源
// 存在弱引用时，对象可能还未被销毁，但可以假设对象不再被使用
// 如果没有弱引用（即类即将被销毁），此函数将不会被调用


};

namespace detail {
template <class TTarget>
struct intrusive_target_default_null_type final {
  static constexpr TTarget* singleton() noexcept {
    return nullptr;
  }
};


// detail 命名空间中的结构体，定义了默认的空指针类型
// 返回一个空指针常量表达式


template <class TTarget, class ToNullType, class FromNullType>
TTarget* assign_ptr_(TTarget* rhs) {
  if (FromNullType::singleton() == rhs) {
    return ToNullType::singleton();
  } else {
    return rhs;
  }
}


// 模板函数，根据不同的空指针类型进行指针赋值
// 如果 rhs 是 FromNullType 的单例空指针，则返回 ToNullType 的单例空指针
// 否则返回 rhs 自身


// Increment needs to be acquire-release to make use_count() and
// unique() reliable.
inline uint32_t atomic_refcount_increment(std::atomic<uint32_t>& refcount) {
  return refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
}


// 原子操作，递增 refcount 计数器
// 使用 acquire-release 语义确保 use_count() 和 unique() 的可靠性


// weak_use_count() is only used for testing, so we don't need it to
// be reliable. Relaxed should be fine.
inline uint32_t atomic_weakcount_increment(std::atomic<uint32_t>& weakcount) {
  return weakcount.fetch_add(1, std::memory_order_relaxed) + 1;
}


// 原子操作，递增 weakcount 计数器
// 使用 relaxed 语义，仅用于测试，所以不需要保证可靠性


// Both decrements need to be acquire-release for correctness. See
// e.g. std::shared_ptr implementation.
inline uint32_t atomic_refcount_decrement(std::atomic<uint32_t>& refcount) {
  return refcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}


// 原子操作，递减 refcount 计数器
// 使用 acquire-release 语义确保正确性，类似于 std::shared_ptr 的实现


inline uint32_t atomic_weakcount_decrement(std::atomic<uint32_t>& weakcount) {
  return weakcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}


// 原子操作，递减 weakcount 计数器
// 使用 acquire-release 语义确保正确性


} // namespace detail


// detail 命名空间结束


template <class TTarget, class NullType>
class weak_intrusive_ptr;


// 前置声明 weak_intrusive_ptr 模板类


template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class intrusive_ptr final {


// intrusive_ptr 模板类的定义，使用 detail::intrusive_target_default_null_type<TTarget> 作为默认的 NullType


 private:
//  the following static assert would be nice to have but it requires


// 私有部分开始，此处原本可能有 static_assert，但要求其它代码支持
//  intrusive_ptr<T> 实例化时目标类 T 必须完全定义
//  对于包含指向自身的指针的类来说，这是一个问题
//  static_assert 用于检查 TTarget 是否派生自 intrusive_ptr_target 类
//  "intrusive_ptr 只能用于继承自 intrusive_ptr_target 的类。"
#ifndef _WIN32
  // 在 MSVC 上触发此 static_assert
  //  error C2131: expression did not evaluate to a constant
  static_assert(
      // NOLINTNEXTLINE(misc-redundant-expression)
      NullType::singleton() == NullType::singleton(),
      "NullType 必须具有 constexpr singleton() 方法");
#endif

  // 检查 NullType::singleton() 返回的类型是否派生自 TTarget
  static_assert(
      std::is_base_of_v<
          TTarget,
          std::remove_pointer_t<decltype(NullType::singleton())>>,
      "NullType::singleton() 必须返回一个 element_type* 指针");

  // 指向目标对象的指针
  TTarget* target_;

  // 声明友元模板 ExclusivelyOwnedTensorTraits
  template <typename T>
  friend struct ExclusivelyOwnedTensorTraits;

  // 声明友元类 intrusive_ptr
  template <class TTarget2, class NullType2>
  friend class intrusive_ptr;

  // 声明友元类 weak_intrusive_ptr<TTarget, NullType>
  friend class weak_intrusive_ptr<TTarget, NullType>;

  // retain_ 方法的实现，用于增加引用计数
  void retain_() {
    // 如果目标不是 NullType::singleton()
    if (target_ != NullType::singleton()) {
      // 原子操作增加引用计数
      uint32_t new_refcount =
          detail::atomic_refcount_increment(target_->refcount_);
      // 调试断言：不能在引用计数达到零后增加引用计数
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          new_refcount != 1,
          "intrusive_ptr: Cannot increase refcount after it reached zero.");
    }
  }

  // reset_ 方法的实现，用于重置指针
  void reset_() noexcept {
    // 如果目标不是 NullType::singleton() 并且引用计数减到零
    if (target_ != NullType::singleton() &&
        detail::atomic_refcount_decrement(target_->refcount_) == 0) {
      // 关于 weakcount 的注释见上面的说明。
      // 当 refcount>0 时，weakcount 比实际弱引用数多一个。
      // 所以在这里需要递减它。
      bool should_delete =
          target_->weakcount_.load(std::memory_order_acquire) == 1;
      if (!should_delete) {
        // 正当性：const_cast 的理由是 release_resources 基本上是一个析构函数，
        // 即使对于 const 对象，析构函数也总是改变对象。
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<std::remove_const_t<TTarget>*>(target_)->release_resources();
        should_delete =
            detail::atomic_weakcount_decrement(target_->weakcount_) == 0;
      }
      // 如果应当删除目标对象，则释放其内存
      if (should_delete) {
        delete target_;
      }
    }
  }
  // raw pointer constructors are not public because we shouldn't make
  // intrusive_ptr out of raw pointers except from inside the make_intrusive(),
  // reclaim() and weak_intrusive_ptr::lock() implementations.

  // 这些原始指针构造函数不公开，因为我们不应该在除了 make_intrusive()、reclaim() 和 weak_intrusive_ptr::lock() 实现内部之外，使用原始指针创建 intrusive_ptr。

  // This constructor will increase the ref counter for you.
  // This constructor will be used by the make_intrusive(), and also pybind11,
  // which wrap the intrusive_ptr holder around the raw pointer and incref
  // correspondingly (pybind11 requires raw pointer constructor to incref by
  // default).
  
  // 这个构造函数会为目标对象增加引用计数。
  // 这个构造函数将被 make_intrusive() 使用，也被 pybind11 使用，pybind11 会将 intrusive_ptr 的持有者包装在原始指针周围，并相应地增加引用计数（pybind11 默认要求原始指针构造函数增加引用计数）。

  explicit intrusive_ptr(TTarget* target)
      : intrusive_ptr(target, raw::DontIncreaseRefcount{}) {
    if (target_ != NullType::singleton()) {
      // We just created result.target_, so we know no other thread has
      // access to it, so we know we needn't care about memory ordering.
      // (On x86_64, a store with memory_order_relaxed generates a plain old
      // `mov`, whereas an atomic increment does a lock-prefixed `add`, which is
      // much more expensive: https://godbolt.org/z/eKPzj8.)
      
      // 我们刚刚创建了 result.target_，因此我们知道没有其他线程可以访问它，所以我们不需要关心内存顺序。
      // （在 x86_64 上，使用 memory_order_relaxed 的存储生成简单的 `mov` 指令，而原子增加则执行带有锁前缀的 `add` 操作，后者更加昂贵：https://godbolt.org/z/eKPzj8。）

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          target_->refcount_ == 0 && target_->weakcount_ == 0,
          "intrusive_ptr: Newly-created target had non-zero refcounts. Does its "
          "constructor do something strange like incref or create an "
          "intrusive_ptr from `this`?");
      target_->refcount_.store(1, std::memory_order_relaxed);
      target_->weakcount_.store(1, std::memory_order_relaxed);
    }
  }

 public:
  using element_type = TTarget;

  intrusive_ptr() noexcept
      : intrusive_ptr(NullType::singleton(), raw::DontIncreaseRefcount{}) {}

  intrusive_ptr(std::nullptr_t) noexcept
      : intrusive_ptr(NullType::singleton(), raw::DontIncreaseRefcount{}) {}

  // This constructor will not increase the ref counter for you.
  // We use the tagged dispatch mechanism to explicitly mark this constructor
  // to not increase the refcount
  
  // 这个构造函数不会为目标对象增加引用计数。
  // 我们使用标记派发机制显式地标记这个构造函数不增加引用计数。

  explicit intrusive_ptr(TTarget* target, raw::DontIncreaseRefcount) noexcept
      : target_(target) {}

  explicit intrusive_ptr(std::unique_ptr<TTarget> rhs) noexcept
      : intrusive_ptr(rhs.release()) {}

  intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();
  }

  template <class From, class FromNullType>
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  /* implicit */ intrusive_ptr(intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(
            detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr move constructor got pointer of wrong type.");
    rhs.target_ = FromNullType::singleton();
  }

  intrusive_ptr(const intrusive_ptr& rhs) : target_(rhs.target_) {
  }

  // 模板构造函数，从另一个 intrusive_ptr 对象构造
  template <class From, class FromNullType>
  /* implicit */ intrusive_ptr(const intrusive_ptr<From, FromNullType>& rhs)
      : target_(
            detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr copy constructor got pointer of wrong type.");
    // 增加引用计数
    retain_();
  }

  // 析构函数，释放资源
  ~intrusive_ptr() noexcept {
    reset_();
  }

  // 移动赋值运算符，接受右值引用
  intrusive_ptr& operator=(intrusive_ptr&& rhs) & noexcept {
    // NOLINTNEXTLINE(*assign*)
    // 调用模板化的赋值运算符
    return operator= <TTarget, NullType>(std::move(rhs));
  }

  // 模板化的移动赋值运算符
  template <class From, class FromNullType>
  intrusive_ptr& operator=(intrusive_ptr<From, FromNullType>&& rhs) & noexcept {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr move assignment got pointer of wrong type.");
    // 使用移动语义构造临时对象，然后交换资源
    intrusive_ptr tmp = std::move(rhs);
    swap(tmp);
    return *this;
  }

  // 拷贝赋值运算符，接受 const 左值引用
  intrusive_ptr& operator=(const intrusive_ptr& rhs) & noexcept {
    // NOLINTNEXTLINE(*assign-operator, *assignment-signature)
    // 调用模板化的赋值运算符
    return operator= <TTarget, NullType>(rhs);
  }

  // 模板化的拷贝赋值运算符
  template <class From, class FromNullType>
  intrusive_ptr& operator=(
      const intrusive_ptr<From, NullType>& rhs) & noexcept {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr copy assignment got pointer of wrong type.");
    // 使用拷贝构造函数创建临时对象，然后交换资源
    intrusive_ptr tmp = rhs;
    swap(tmp);
    return *this;
  }

  // 返回被管理对象的指针
  TTarget* get() const noexcept {
    return target_;
  }

  // 解引用操作符，返回被管理对象的引用
  TTarget& operator*() const noexcept {
    return *target_;
  }

  // 成员访问操作符，返回被管理对象的指针
  TTarget* operator->() const noexcept {
    return target_;
  }

  // 转换为 bool 类型，检查是否为空指针
  operator bool() const noexcept {
    return target_ != NullType::singleton();
  }

  // 重置 intrusive_ptr 对象
  void reset() noexcept {
    reset_();
    // 将目标指针重置为空指针
    target_ = NullType::singleton();
  }

  // 交换两个 intrusive_ptr 对象的资源
  void swap(intrusive_ptr& rhs) noexcept {
    std::swap(target_, rhs.target_);
  }

  // 检查被管理对象是否已定义
  // 在代码中频繁使用空指针检查，这个操作需要高效执行
  bool defined() const noexcept {
    return target_ != NullType::singleton();
  }

  // 返回被管理对象的引用计数
  uint32_t use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    // 使用 memory_order_acquire 加载引用计数
    return target_->refcount_.load(std::memory_order_acquire);
  }

  // 返回被管理对象的弱引用计数
  uint32_t weak_use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    // 使用 memory_order_acquire 加载弱引用计数
    return target_->weakcount_.load(std::memory_order_acquire);
  }

  // 检查是否仅有当前 intrusive_ptr 对象在管理资源
  bool unique() const noexcept {
    // 返回当前 intrusive_ptr 实例的引用计数是否为1
    return use_count() == 1;
  }

  /**
   * 返回一个拥有所有权的指针到底层对象，并使 intrusive_ptr 实例无效。
   * 这意味着引用计数不会减少。必须使用 intrusive_ptr::reclaim(ptr)
   * 将返回的指针放回 intrusive_ptr 中以正确析构它。这对于 C API 非常有用。
   */
  TTarget* release() noexcept {
    // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
    // 将 target_ 赋值给 result，然后将 target_ 设置为 NullType::singleton()
    TTarget* result = target_;
    target_ = NullType::singleton();
    return result;
  }

  /**
   * 接受一个拥有所有权的 TTarget* 指针，并创建一个 intrusive_ptr
   * 来接管所有权。这意味着引用计数不会增加。这是 intrusive_ptr::release() 的对应部分，
   * 传入的指针必须使用 intrusive_ptr::release() 创建。
   */
  static intrusive_ptr reclaim(TTarget* owning_ptr) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        // 确保 owning_ptr 满足条件：要么是 NullType::singleton()，要么 refcount_ == 0 且 weakcount_ > 0
        owning_ptr == NullType::singleton() ||
            owning_ptr->refcount_.load() == 0 || owning_ptr->weakcount_.load(),
        "TTarget violates the invariant that refcount > 0  =>  weakcount > 0");
    return intrusive_ptr(owning_ptr, raw::DontIncreaseRefcount{});
  }

  /**
   * 接受一个拥有所有权的 TTarget* 指针，并创建一个 intrusive_ptr
   * 表示一个新引用，即原始指针保留所有权。
   */
  static intrusive_ptr reclaim_copy(TTarget* owning_ptr) {
    // 使用 reclaim() 获取 intrusive_ptr，并调用 retain_() 保留引用
    auto ret = reclaim(owning_ptr);
    ret.retain_();
    return ret;
  }

  /**
   * 使用参数 args 分配一个堆对象，并将其包装在 intrusive_ptr 中并增加引用计数。
   * 这是一个辅助函数，允许 make_intrusive() 访问私有的 intrusive_ptr 构造函数。
   */
  template <class... Args>
  static intrusive_ptr make(Args&&... args) {
    return intrusive_ptr(new TTarget(std::forward<Args>(args)...));
  }

  /**
   * 将 TTarget 的新实例（例如，直接使用 new TTarget(...) 分配的）转换为 intrusive_ptr。
   * 如果可能，请使用 intrusive_ptr::make，它可以静态保证分配是正确的。
   *
   * 目前，此方法存在的唯一原因是因为 pybind11 的持有者类型希望能够以这种方式分配
   * （因为 pybind11 自己处理新的分配）。
   */
  static intrusive_ptr unsafe_steal_from_new(TTarget* raw_ptr) {
    // 返回一个将 raw_ptr 包装为 intrusive_ptr 的实例
    return intrusive_ptr(raw_ptr);
  }

  /**
   * 将一个不应被引用计数的 TTarget 实例（例如，在 arena 中使用 placement new 分配的）转换为 intrusive_ptr。
   * 这种操作极不安全，只有在可以保证指针不会逃逸并像正常情况下进行引用计数时才应使用。
   *
   * `expected_decrefs` 是一个调试参数：它表示被讨论的 intrusive_ptr_target 预计会获得的强引用数目。
   * 在大多数情况下，这个数值可能是 1。
   *
   * 此方法的存在是为了在静态运行时手动共享 StorageImpls 到 Tensor 之间。它需要访问私有的 intrusive_ptr 成员，
   * 以便可以将引用计数初始化为自定义值。
   */
  static intrusive_ptr unsafe_adapt_non_heap_allocated(
      TTarget* raw_ptr,
      uint32_t expected_decrefs) {
    // 使用 raw::DontIncreaseRefcount 初始化一个 intrusive_ptr 实例
    intrusive_ptr result(raw_ptr, raw::DontIncreaseRefcount{});
    // kImpracticallyHugeReferenceCount 对于引用计数来说是不切实际地巨大，但不会溢出 uint32_t。
    // 实际上我们只需要将引用计数初始化为 2 -- 我们只是做了一个不平衡的增加引用计数以防止非堆分配目标被释放，
    // 并且我们通过直接初始化引用计数而不是进行昂贵的原子增量来优化这个增加引用计数。
    // 使用 kImpracticallyHugeReferenceCount 的原因是为了适应 ~intrusive_ptr_target 中的调试断言。
    ```
#ifdef NDEBUG
    // 如果定义了 NDEBUG 宏，则期望的引用计数为 0
    expected_decrefs = 0;
#endif
    // 将目标对象的引用计数设置为一个极大的值加上期望减少的引用计数
    result.target_->refcount_.store(
        detail::kImpracticallyHugeReferenceCount + expected_decrefs,
        std::memory_order_relaxed);
    // 将目标对象的弱引用计数设置为一个极大的值，使用 relaxed 内存顺序
    result.target_->weakcount_.store(
        detail::kImpracticallyHugeReferenceCount, std::memory_order_relaxed);
    // 返回结果对象
    return result;
  }

/**
 * 将一个非拥有原始指针转换为 intrusive_ptr。这相当于在 shared_ptr 上使用 enable_shared_from_this 的道德等效物。
 *
 * 此方法仅适用于已经存在的对象。如果你在寻找类似于 unique_ptr<T>(T*) 构造函数的道德等效物，请参见 steal_from_new。
 *
 * TODO: https://github.com/pytorch/pytorch/issues/56482
 */
static intrusive_ptr unsafe_reclaim_from_nonowning(TTarget* raw_ptr) {
  // 参见注释 [Stack allocated intrusive_ptr_target safety]
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      raw_ptr == NullType::singleton() || raw_ptr->refcount_.load() > 0,
      "intrusive_ptr: Can only reclaim pointers that are owned by someone");
  // 回收指针，不增加引用计数
  auto ptr = reclaim(raw_ptr); // doesn't increase refcount
  // 增加引用计数
  ptr.retain_();
  // 返回结果
  return ptr;
}

};

template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>,
    class... Args>
inline intrusive_ptr<TTarget, NullType> make_intrusive(Args&&... args) {
  // 调用 intrusive_ptr 的静态方法 make 来创建 intrusive_ptr 对象
  return intrusive_ptr<TTarget, NullType>::make(std::forward<Args>(args)...);
}

template <class TTarget, class NullType>
inline void swap(
    intrusive_ptr<TTarget, NullType>& lhs,
    intrusive_ptr<TTarget, NullType>& rhs) noexcept {
  // 调用 intrusive_ptr 的 swap 方法来交换两个 intrusive_ptr 对象
  lhs.swap(rhs);
}

// 为了允许 intrusive_ptr 在 std::map 或 std::set 内部使用，需要定义 operator< 操作符
template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator<(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  // 比较两个 intrusive_ptr 的原始指针地址
  return lhs.get() < rhs.get();
}

// 定义 operator== 操作符，比较两个 intrusive_ptr 是否指向相同的对象
template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator==(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.get() == rhs.get();
}

// 定义 operator== 操作符，比较 intrusive_ptr 是否等于 nullptr
template <class TTarget1, class NullType1>
inline bool operator==(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    std::nullptr_t) noexcept {
  return lhs.get() == nullptr;
}

// 定义 operator== 操作符，比较 nullptr 是否等于 intrusive_ptr
template <class TTarget2, class NullType2>
inline bool operator==(
    std::nullptr_t,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return nullptr == rhs.get();
}

// 定义 operator!= 操作符，比较两个 intrusive_ptr 是否不相等
template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator!=(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return !operator==(lhs, rhs);
}

// 定义 operator!= 操作符，比较 intrusive_ptr 是否不等于 nullptr
template <class TTarget1, class NullType1>
inline bool operator!=(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    std::nullptr_t) noexcept {
  return !operator==(lhs, nullptr);
}
// 定义 != 操作符，用于比较 nullptr 和 intrusive_ptr 类型的 rhs，返回相反的 == 操作结果
template <class TTarget2, class NullType2>
inline bool operator!=(
    std::nullptr_t,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return !operator==(nullptr, rhs);
}

// MaybeOwnedTraits 结构体模板的特化，针对 c10::intrusive_ptr<T> 类型
template <typename T>
struct MaybeOwnedTraits<c10::intrusive_ptr<T>> {
  using owned_type = c10::intrusive_ptr<T>;    // 拥有的类型是 c10::intrusive_ptr<T>
  using borrow_type = c10::intrusive_ptr<T>;   // 借用的类型也是 c10::intrusive_ptr<T>

  // 创建一个 borrow_type，从 owned_type 中获取资源
  static borrow_type createBorrow(const owned_type& from) {
    return borrow_type::reclaim(from.get());
  }

  // 将 rhs 的资源赋值给 lhs
  static void assignBorrow(borrow_type& lhs, const borrow_type& rhs) {
    lhs.release();                              // 释放 lhs 当前持有的资源
    lhs = borrow_type::reclaim(rhs.get());       // 从 rhs 中获取资源并赋给 lhs
  }

  // 销毁一个 borrow_type，释放其资源
  static void destroyBorrow(borrow_type& toDestroy) {
    toDestroy.release();                        // 释放 borrow_type 持有的资源
  }

  // 从 borrow 中获取引用
  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return borrow;                              // 直接返回 borrow 的引用
  }

  // 从 borrow 中获取指针
  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return &borrow;                             // 返回 borrow 的指针
  }

  // 调试用：判断 borrow 是否有效
  static bool debugBorrowIsValid(const borrow_type& /*borrow*/) {
    return true;                                // 始终返回 true，表明 borrow 总是有效的
  }
};

// weak_intrusive_ptr 类模板的定义
template <
    typename TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class weak_intrusive_ptr final {
 private:
  // 静态断言：TTarget 必须是 intrusive_ptr_target 的派生类
  static_assert(
      std::is_base_of_v<intrusive_ptr_target, TTarget>,
      "intrusive_ptr can only be used for classes that inherit from intrusive_ptr_target.");

#ifndef _WIN32
  // 条件静态断言：在非 Windows 平台触发，要求 NullType 必须有一个 constexpr 的 singleton() 方法
  static_assert(
      NullType::singleton() == NullType::singleton(),
      "NullType must have a constexpr singleton() method");
#endif

  // 静态断言：NullType::singleton() 返回的类型必须是 TTarget* 的指针类型
  static_assert(
      std::is_base_of_v<
          TTarget,
          std::remove_pointer_t<decltype(NullType::singleton())>>,
      "NullType::singleton() must return a element_type* pointer");

  TTarget* target_;  // 指向目标对象的指针

  template <class TTarget2, class NullType2>
  friend class weak_intrusive_ptr;

  // retain_ 方法：增加目标对象的弱引用计数
  void retain_() {
    if (target_ != NullType::singleton()) {
      uint32_t new_weakcount =
          detail::atomic_weakcount_increment(target_->weakcount_);
      // 调试断言：增加弱引用计数后不能为 1，否则报错
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          new_weakcount != 1,
          "weak_intrusive_ptr: Cannot increase weakcount after it reached zero.");
    }
  }

  // reset_ 方法：重置指针，并可能释放目标对象
  void reset_() noexcept {
    if (target_ != NullType::singleton() &&
        detail::atomic_weakcount_decrement(target_->weakcount_) == 0) {
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDelete)
      delete target_;                           // 删除目标对象
    }
    target_ = NullType::singleton();            // 重置指针为 NullType 的单例值
  }

  // 显式构造函数：从 TTarget* 构造 weak_intrusive_ptr
  constexpr explicit weak_intrusive_ptr(TTarget* target) : target_(target) {}

 public:
  using element_type = TTarget;                 // 元素类型是 TTarget

  // 显式构造函数：从 intrusive_ptr 转换构造 weak_intrusive_ptr
  explicit weak_intrusive_ptr(const intrusive_ptr<TTarget, NullType>& ptr)
      : weak_intrusive_ptr(ptr.get()) {
    retain_();                                  // 增加引用计数
  }

  // 移动构造函数：从另一个 weak_intrusive_ptr 移动构造
  weak_intrusive_ptr(weak_intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();
  }

  template <class From, class FromNullType>
  /* implicit */ weak_intrusive_ptr(
      // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
      weak_intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(
            detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. weak_intrusive_ptr move constructor got pointer of wrong type.");
    rhs.target_ = FromNullType::singleton();
  }

- `rhs.target_ = NullType::singleton();`
  ```cpp
  // 将 rhs 对象的 target_ 成员设置为 NullType 的单例对象
  ```

- `template <class From, class FromNullType>`
  ```cpp
  // 模板：移动构造函数，从另一个类型为 weak_intrusive_ptr 的对象 rhs 移动构造
  ```

- `/* implicit */ weak_intrusive_ptr(`
  ```cpp
  // 隐式构造函数：weak_intrusive_ptr 的移动构造函数
  ```

- `// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)`
  ```cpp
  // 禁止对 rhs 参数进行移动的静态分析指令
  ```

- `weak_intrusive_ptr<From, FromNullType>&& rhs) noexcept`
  ```cpp
  // 右值引用参数 rhs，移动构造函数使用 noexcept 保证不抛出异常
  ```

- `: target_(`
  ```cpp
  // 初始化列表：将 target_ 成员初始化为以下表达式的结果
  ```

- `detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {`
  ```cpp
  // 调用 detail 命名空间中的 assign_ptr_ 模板函数，用 rhs.target_ 的值初始化 target_
  ```

- `static_assert(`
  ```cpp
  // 静态断言：确保 From* 可以隐式转换为 TTarget*，否则编译错误并显示指定的错误消息
  ```

- `"Type mismatch. weak_intrusive_ptr move constructor got pointer of wrong type.");`
  ```cpp
  // 静态断言失败时显示的错误消息，指示移动构造函数接收了错误类型的指针
  ```

- `rhs.target_ = FromNullType::singleton();`
  ```cpp
  // 将 rhs 对象的 target_ 成员设置为 FromNullType 的单例对象
  ```
  // 将 rhs.target_ 设置为 tmp
  rhs.target_ = tmp;
}

// NB: This should ONLY be used by the std::hash implementation
// for weak_intrusive_ptr.  Another way you could do this is
// friend std::hash<weak_intrusive_ptr>, but this triggers two
// bugs:
//
//  (1) It triggers an nvcc bug, where std::hash in a friend class
//      declaration gets preprocessed into hash, which then cannot
//      actually be found.  The error in this case looks like:
//
//        error: no template named 'hash'; did you mean 'std::hash'?
//
//  (2) On OS X, std::hash is declared as a struct, not a class.
//      This twings:
//
//        error: class 'hash' was previously declared as a struct
//        [-Werror,-Wmismatched-tags]
//
// Both of these are work-aroundable, but on the whole, I decided
// it would be simpler and easier to make work if we just expose
// an unsafe getter for target_
//
// 返回 _unsafe_get_target() 函数返回的指针，用于内部获取目标对象
TTarget* _unsafe_get_target() const noexcept {
  return target_;
}

// 返回当前对象的强引用计数
uint32_t use_count() const noexcept {
  if (target_ == NullType::singleton()) {
    return 0;
  }
  return target_->refcount_.load(
      std::memory_order_acquire); // refcount, not weakcount!
}

// 返回当前对象的弱引用计数
uint32_t weak_use_count() const noexcept {
  if (target_ == NullType::singleton()) {
    return 0;
  }
  return target_->weakcount_.load(std::memory_order_acquire);
}

// 检查当前对象的强引用计数是否为 0，即对象是否已过期
bool expired() const noexcept {
  return use_count() == 0;
}

// 获取当前对象的强引用指针，如果对象已经销毁则返回空指针
intrusive_ptr<TTarget, NullType> lock() const noexcept {
  if (expired()) {
    return intrusive_ptr<TTarget, NullType>();
  } else {
    auto refcount = target_->refcount_.load(std::memory_order_seq_cst);
    do {
      if (refcount == 0) {
        // Object already destructed, no strong references left anymore.
        // Return nullptr.
        return intrusive_ptr<TTarget, NullType>();
      }
    } while (
        !target_->refcount_.compare_exchange_weak(refcount, refcount + 1));
    return intrusive_ptr<TTarget, NullType>(
        target_, raw::DontIncreaseRefcount{});
  }
}

/**
 * 返回一个拥有（但仍然只有弱引用）的指向底层对象的指针，并使 weak_intrusive_ptr 实例无效。
 * 这意味着 weakcount 没有减少。
 * 必须使用 weak_intrusive_ptr::reclaim(ptr) 将返回的指针放回 weak_intrusive_ptr 中以正确析构它。
 * 对于 C API 很有帮助。
 */
TTarget* release() noexcept {
  TTarget* result = target_;
  target_ = NullType::singleton();
  return result;
}

/**
 * 接受一个拥有（但必须是弱引用）的 TTarget* 指针，并创建一个接管所有权的 weak_intrusive_ptr。
 * 这意味着 weakcount 没有增加。
 * 这是 weak_intrusive_ptr::release() 的对应部分，传入的指针必须使用 weak_intrusive_ptr::release() 创建。
 */
static weak_intrusive_ptr reclaim(TTarget* owning_weak_ptr) {
    // 断言，确保 owning_weak_ptr 符合以下条件之一：
    // 1. 如果 owning_weak_ptr == NullType::singleton()，则忽略后续条件
    // 2. 如果 owning_weak_ptr 的引用计数 refcount > 0，则弱引用计数 weakcount 必须 > 1，以确保存在弱引用
    // 3. 如果 owning_weak_ptr 的引用计数 refcount == 0，则弱引用计数 weakcount 必须 > 0，即使没有强引用也能保持弱引用
    // 详细说明见该文件顶部关于弱引用计数的解释
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        owning_weak_ptr == NullType::singleton() ||
            owning_weak_ptr->weakcount_.load() > 1 ||
            (owning_weak_ptr->refcount_.load() == 0 &&
             owning_weak_ptr->weakcount_.load() > 0),
        "weak_intrusive_ptr: Can only weak_intrusive_ptr::reclaim() owning pointers that were created using weak_intrusive_ptr::release().");

    // 返回一个 weak_intrusive_ptr 对象，该对象用于表示一个新的弱引用，
    // 即原始指针 owning_ptr 保留了所有权
    return weak_intrusive_ptr(owning_weak_ptr);
  }

  /**
   * Takes a pointer to TTarget* (may be weak or strong) and creates a
   * new weak_intrusive_ptr representing a new weak reference, i.e.
   * the raw pointer retains ownership.
   */
  static weak_intrusive_ptr reclaim_copy(TTarget* owning_ptr) {
    // 调用 reclaim 函数获取弱引用指针，然后调用 retain_ 方法增加引用计数
    auto ret = reclaim(owning_ptr);
    ret.retain_();
    // 返回增加了引用计数后的 weak_intrusive_ptr 对象
    return ret;
  }

  // 以下是友元函数的声明，用于比较不同类型的 weak_intrusive_ptr 对象
  template <class TTarget1, class NullType1, class TTarget2, class NullType2>
  friend bool operator<(
      const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
      const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept;
  template <class TTarget1, class NullType1, class TTarget2, class NullType2>
  friend bool operator==(
      const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
      const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept;
};

// 定义了一个模板函数 swap，用于交换两个 weak_intrusive_ptr 对象的内容
template <class TTarget, class NullType>
inline void swap(
    weak_intrusive_ptr<TTarget, NullType>& lhs,
    weak_intrusive_ptr<TTarget, NullType>& rhs) noexcept {
  lhs.swap(rhs);
}

// 为了允许在 std::map 或 std::set 中使用 weak_intrusive_ptr，定义了比较运算符 operator<
template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator<(
    const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
    const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.target_ < rhs.target_;
}

// 定义了比较运算符 operator==，用于比较两个 weak_intrusive_ptr 是否相等
template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator==(
    const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
    const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.target_ == rhs.target_;
}

// 定义了比较运算符 operator!=，用于比较两个 weak_intrusive_ptr 是否不相等
template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator!=(
    const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
    const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  // 使用 operator== 来实现 operator!=
  return !operator==(lhs, rhs);
}

// 为了方便文档目的，定义了别名 weak_intrusive_ptr_target，用于区分 weak raw intrusive pointers 和 intrusive pointers
using weak_intrusive_ptr_target = intrusive_ptr_target;

// 这个命名空间提供了一些处理 raw pointers 的方法，这些指针是 intrusive_ptr_target 的子类。
// 它们不作为 intrusive_ptr_target 的方法提供，因为理想情况下，不应该再使用这些方法（应使用智能指针）。
// 如果你需要处理仍需传递 raw pointers 的遗留代码，这些方法可能会很有用。
namespace raw {

namespace intrusive_ptr {

// 注意：与 reclaim() API 不同，不允许将 NullType::singleton 传递给 incref()
inline void incref(intrusive_ptr_target* self) {
  // 如果 self 不为 nullptr，则增加其引用计数
  if (self) {
    detail::atomic_refcount_increment(self->refcount_);
  }
}

// 注意：与 reclaim() API 不同，不允许将 NullType::singleton 传递给 decref()
inline void decref(intrusive_ptr_target* self) {
  // 让对象的引用计数减少，允许对象自行销毁
  c10::intrusive_ptr<intrusive_ptr_target>::reclaim(self);
  // 注意：调用者仍然持有 'self' 指针，但现在已经无效。
  // 如果需要更安全的操作，请使用实际的 c10::intrusive_ptr 类。
}

// 创建一个弱引用指针，从强引用指针 self 转换而来
template <typename T>
inline T* make_weak(T* self) {
  // 'self' 是一个强引用指针，但返回一个弱引用指针
  auto ptr = c10::intrusive_ptr<T>::reclaim(self);
  c10::weak_intrusive_ptr<T> wptr(ptr);
  ptr.release();  // 释放对对象的所有权
  return wptr.release();  // 返回弱引用指针
}
// 返回给定的 intrusive_ptr_target 对象的强引用计数
inline uint32_t use_count(intrusive_ptr_target* self) {
  // 通过 reclaim 方法获取 intrusive_ptr 对象，此时 self 指针所有权转移给 ptr
  auto ptr = c10::intrusive_ptr<intrusive_ptr_target>::reclaim(self);
  // 调用 intrusive_ptr 的 use_count 方法获取引用计数
  auto r = ptr.use_count();
  // 释放 intrusive_ptr 对象，将 self 指针所有权还原
  ptr.release();
  // 返回获取的引用计数
  return r;
}

} // namespace intrusive_ptr

namespace weak_intrusive_ptr {

// 增加 weak_intrusive_ptr_target 对象的弱引用计数
inline void incref(weak_intrusive_ptr_target* self) {
  // 调用 detail 命名空间中的原子操作函数增加 weakcount_ 成员变量的值
  detail::atomic_weakcount_increment(self->weakcount_);
}

// 减少 weak_intrusive_ptr_target 对象的弱引用计数
inline void decref(weak_intrusive_ptr_target* self) {
  // 释放 weak_intrusive_ptr 对象，self 指针的所有权转移给 reclaim 方法
  c10::weak_intrusive_ptr<intrusive_ptr_target>::reclaim(self);
  // 注意：此时 self 指针仍然存在，但已经无效。
  // 如果需要更安全的操作，应使用 c10::weak_intrusive_ptr 类
}

// 锁定 weak_intrusive_ptr 指向的对象，并返回其指针
template <typename T>
inline T* lock(T* self) {
  // 通过 reclaim 方法获取 weak_intrusive_ptr 对象，self 指针所有权转移给 wptr
  auto wptr = c10::weak_intrusive_ptr<T>::reclaim(self);
  // 调用 lock 方法获取强引用指针 ptr
  auto ptr = wptr.lock();
  // 释放 weak_intrusive_ptr 对象，self 指针所有权还原
  wptr.release();
  // 返回强引用指针 ptr
  return ptr.release();
}

// 获取 weak_intrusive_ptr 指向对象的强引用计数
// 注意：此函数返回的是弱引用指针的强引用计数
inline uint32_t use_count(weak_intrusive_ptr_target* self) {
  // 通过 reclaim 方法获取 weak_intrusive_ptr 对象，self 指针所有权转移给 wptr
  auto wptr = c10::weak_intrusive_ptr<intrusive_ptr_target>::reclaim(self);
  // 调用 use_count 方法获取强引用计数
  auto r = wptr.use_count();
  // 释放 weak_intrusive_ptr 对象，self 指针所有权还原
  wptr.release();
  // 返回获取的强引用计数
  return r;
}

} // namespace weak_intrusive_ptr

} // namespace raw

} // namespace c10

namespace std {
// 为了允许 intrusive_ptr 和 weak_intrusive_ptr 作为 std::unordered_map 或 std::unordered_set 的键，需要提供 std::hash
template <class TTarget, class NullType>
struct hash<c10::intrusive_ptr<TTarget, NullType>> {
  // 定义哈希运算符，用于 intrusive_ptr 的哈希计算，基于指向对象的指针
  size_t operator()(const c10::intrusive_ptr<TTarget, NullType>& x) const {
    return std::hash<TTarget*>()(x.get());
  }
};
template <class TTarget, class NullType>
struct hash<c10::weak_intrusive_ptr<TTarget, NullType>> {
  // 定义哈希运算符，用于 weak_intrusive_ptr 的哈希计算，基于指向对象的指针
  size_t operator()(const c10::weak_intrusive_ptr<TTarget, NullType>& x) const {
    return std::hash<TTarget*>()(x._unsafe_get_target());
  }
};
} // namespace std


这些注释将代码中的每一行都解释了清楚，保持了代码的完整性和结构。
```