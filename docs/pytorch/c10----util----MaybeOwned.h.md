# `.\pytorch\c10\util\MaybeOwned.h`

```py
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <c10/macros/Macros.h>
// 包含 c10 库的宏定义

#include <c10/util/Exception.h>
// 包含 c10 库的异常处理工具

#include <memory>
// 包含标准库中与智能指针相关的头文件

#include <type_traits>
// 包含标准库中与类型特性相关的头文件

#include <utility>
// 包含标准库中与实用工具相关的头文件

namespace c10 {

/// MaybeOwnedTraits<T> describes how to borrow from T.  Here is how we
/// can implement borrowing from an arbitrary type T using a raw
/// pointer to const:
// 定义了 MaybeOwnedTraits<T> 结构体模板，描述了如何从类型 T 中进行借用
// 提供了使用常量指针实现对任意类型 T 借用的方法

template <typename T>
struct MaybeOwnedTraitsGenericImpl {
  using owned_type = T;
  // 使用 T 类型作为 owned_type

  using borrow_type = const T*;
  // 使用常量指针 const T* 作为 borrow_type

  static borrow_type createBorrow(const owned_type& from) {
    return &from;
    // 返回指向 from 的常量指针，实现从 owned_type 到 borrow_type 的借用
  }

  static void assignBorrow(borrow_type& lhs, borrow_type rhs) {
    lhs = rhs;
    // 将 rhs 赋值给 lhs，实现 borrow_type 的赋值操作
  }

  static void destroyBorrow(borrow_type& /*toDestroy*/) {}
  // 销毁 borrow_type 对象，但本例中没有实际操作

  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return *borrow;
    // 返回 borrow 指向的对象的引用，实现从 borrow_type 到 owned_type 的引用
  }

  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return borrow;
    // 返回 borrow 指向的对象的指针，实现从 borrow_type 到 owned_type 的指针
  }

  static bool debugBorrowIsValid(const borrow_type& borrow) {
    return borrow != nullptr;
    // 判断 borrow 是否有效，实现 borrow_type 的有效性检查
  }
};

/// It is possible to eliminate the extra layer of indirection for
/// borrows for some types that we control. For examples, see
/// intrusive_ptr.h and TensorBody.h.
// 可以消除某些我们控制的类型的借用中的额外间接层次。例如，请参阅 intrusive_ptr.h 和 TensorBody.h。

template <typename T>
struct MaybeOwnedTraits;
// MaybeOwnedTraits<T> 结构体模板的前向声明

// Explicitly enable MaybeOwned<shared_ptr<T>>, rather than allowing
// MaybeOwned to be used for any type right away.
// 明确地启用 MaybeOwned<shared_ptr<T>>，而不是立即允许 MaybeOwned 用于任何类型。

template <typename T>
struct MaybeOwnedTraits<std::shared_ptr<T>>
    : public MaybeOwnedTraitsGenericImpl<std::shared_ptr<T>> {};
// 特化 MaybeOwnedTraits 结构体模板，针对 std::shared_ptr<T> 类型，继承自 MaybeOwnedTraitsGenericImpl<std::shared_ptr<T>>

/// A smart pointer around either a borrowed or owned T. When
/// constructed with borrowed(), the caller MUST ensure that the
/// borrowed-from argument outlives this MaybeOwned<T>. Compare to
/// Rust's std::borrow::Cow
/// (https://doc.rust-lang.org/std/borrow/enum.Cow.html), but note
/// that it is probably not suitable for general use because C++ has
/// no borrow checking. Included here to support
/// Tensor::expect_contiguous.
// 包裹一个借用或拥有的 T 类型的智能指针。当用 borrowed() 构造时，调用者必须确保借用的原始对象在 MaybeOwned<T> 对象之前不会被销毁。与 Rust 的 std::borrow::Cow 相比较，但请注意，这可能不适合一般用途，因为 C++ 没有借用检查。这里包含以支持 Tensor::expect_contiguous。

template <typename T>
class MaybeOwned final {
  using borrow_type = typename MaybeOwnedTraits<T>::borrow_type;
  // 使用 MaybeOwnedTraits<T> 得到 borrow_type 类型

  using owned_type = typename MaybeOwnedTraits<T>::owned_type;
  // 使用 MaybeOwnedTraits<T> 得到 owned_type 类型

  bool isBorrowed_;
  // 表示对象是否为借用状态的布尔值

  union {
    borrow_type borrow_;
    // 联合体，用于存储 borrow_type 类型的数据
  /// 这个类用于封装一个可能是拥有（owned）或者借用（borrowed）状态的对象
  struct MaybeOwned {
    // 标识当前对象是否为借用状态
    bool isBorrowed_;
    // 如果是借用状态，存储借用对象的指针
    borrow_type borrow_;
    // 如果是拥有状态，存储拥有对象的实例
    owned_type own_;

    /// 不要使用这个构造函数；使用 borrowed() 替代。
    explicit MaybeOwned(const owned_type& t)
        : isBorrowed_(true), borrow_(MaybeOwnedTraits<T>::createBorrow(t)) {}

    /// 不要使用这个构造函数；使用 owned() 替代。
    explicit MaybeOwned(T&& t) noexcept(std::is_nothrow_move_constructible_v<T>)
        : isBorrowed_(false), own_(std::move(t)) {}

    /// 不要使用这个构造函数；使用 owned() 替代。
    template <class... Args>
    explicit MaybeOwned(std::in_place_t, Args&&... args)
        : isBorrowed_(false), own_(std::forward<Args>(args)...) {}

  public:
    /// 默认构造函数，创建一个空的 MaybeOwned 对象，初始为借用状态
    explicit MaybeOwned() : isBorrowed_(true), borrow_() {}

    // 复制一个借用对象会得到原始对象的另一个借用对象，就像 T* 一样。
    // 复制一个拥有对象会得到一个新的拥有对象，保证安全：默认情况下不会形成借用链！
    // （注意，如果需要这种行为，可以使用 MaybeOwned<T>::borrowed(*rhs) 实现。）
    MaybeOwned(const MaybeOwned& rhs) : isBorrowed_(rhs.isBorrowed_) {
      if (C10_LIKELY(rhs.isBorrowed_)) {
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
      } else {
        new (&own_) T(rhs.own_);
      }
    }

    /// 赋值运算符的实现
    MaybeOwned& operator=(const MaybeOwned& rhs) {
      if (this == &rhs) {
        return *this;
      }
      if (C10_UNLIKELY(!isBorrowed_)) {
        if (rhs.isBorrowed_) {
          // 如果当前对象是拥有状态，先析构当前拥有对象
          own_.~T();
          // 使用借用对象赋值，并标记为借用状态
          MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
          isBorrowed_ = true;
        } else {
          // 如果当前对象是拥有状态，直接赋值拥有对象
          own_ = rhs.own_;
        }
      } else {
        if (C10_LIKELY(rhs.isBorrowed_)) {
          // 如果当前对象是借用状态，使用借用对象赋值
          MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
        } else {
          // 如果当前对象是借用状态，析构当前借用对象，然后移动构造拥有对象
          MaybeOwnedTraits<T>::destroyBorrow(borrow_);
          new (&own_) T(rhs.own_);
          isBorrowed_ = false;
        }
      }
      // 断言当前对象和 rhs 对象的借用状态应当一致
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isBorrowed_ == rhs.isBorrowed_);
      return *this;
    }

    /// 移动构造函数的实现
    MaybeOwned(MaybeOwned&& rhs) noexcept(
        std::is_nothrow_move_constructible_v<T> &&
        std::is_nothrow_move_assignable_v<borrow_type>)
        : isBorrowed_(rhs.isBorrowed_) {
      if (C10_LIKELY(rhs.isBorrowed_)) {
        // 如果 rhs 是借用状态，移动构造借用对象
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
      } else {
        // 如果 rhs 是拥有状态，移动构造拥有对象
        new (&own_) T(std::move(rhs.own_));
      }
    }

    /// 移动赋值运算符的实现
    MaybeOwned& operator=(MaybeOwned&& rhs) noexcept(
        std::is_nothrow_move_assignable_v<T> &&
        std::is_nothrow_move_assignable_v<borrow_type> &&
        std::is_nothrow_move_constructible_v<T> &&
        std::is_nothrow_destructible_v<T> &&
        std::is_nothrow_destructible_v<borrow_type>) {
      if (this == &rhs) {
        return *this;
      }
      if (C10_UNLIKELY(!isBorrowed_)) {
        if (rhs.isBorrowed_) {
          // 如果当前对象是拥有状态，析构当前拥有对象，然后移动构造借用对象
          own_.~T();
          MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
          isBorrowed_ = true;
        } else {
          // 如果当前对象是拥有状态，移动赋值拥有对象
          own_ = std::move(rhs.own_);
        }
      } else {
        if (C10_LIKELY(rhs.isBorrowed_)) {
          // 如果当前对象是借用状态，移动赋值借用对象
          MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
        } else {
          // 如果当前对象是借用状态，析构当前借用对象，然后移动构造拥有对象
          MaybeOwnedTraits<T>::destroyBorrow(borrow_);
          new (&own_) T(std::move(rhs.own_));
          isBorrowed_ = false;
        }
      }
      return *this;
    }
  // 如果当前对象已经持有了某个对象（owned），则执行析构该对象，然后以移动语义构造新对象rhs.own_
  // 如果rhs是借用状态（borrowed），则执行借用赋值操作
  // 否则，销毁当前持有的对象，以移动语义构造新对象rhs.own_，并标记为非借用状态
  } else {
    if (C10_LIKELY(rhs.isBorrowed_)) {
      MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
    } else {
      MaybeOwnedTraits<T>::destroyBorrow(borrow_);
      new (&own_) T(std::move(rhs.own_));
      isBorrowed_ = false;
    }
  }
  // 返回当前对象的引用
  return *this;
}

// 通过给定的常量引用t创建一个MaybeOwned对象，并返回该对象
static MaybeOwned borrowed(const T& t) {
  return MaybeOwned(t);
}

// 通过给定的右值引用t创建一个MaybeOwned对象（带有移动语义），并返回该对象
static MaybeOwned owned(T&& t) noexcept(
    std::is_nothrow_move_constructible_v<T>) {
  return MaybeOwned(std::move(t));
}

// 使用in_place_t标志和参数Args创建一个MaybeOwned对象，并返回该对象
template <class... Args>
static MaybeOwned owned(std::in_place_t, Args&&... args) {
  return MaybeOwned(std::in_place, std::forward<Args>(args)...);
}

// 析构函数，析构MaybeOwned对象
// noexcept指定该函数不会抛出异常
// 如果当前对象不是借用状态（owned），则调用析构函数销毁own_中的对象
// 否则，调用destroyBorrow销毁borrow_中的对象
~MaybeOwned() noexcept(
    // NOLINTNEXTLINE(*-noexcept-destructor)
    std::is_nothrow_destructible_v<T> &&
    std::is_nothrow_destructible_v<borrow_type>) {
  if (C10_UNLIKELY(!isBorrowed_)) {
    own_.~T();
  } else {
    MaybeOwnedTraits<T>::destroyBorrow(borrow_);
  }
}

// 返回当前对象是否是借用状态的标志
bool unsafeIsBorrowed() const {
  return isBorrowed_;
}

// 解引用操作符，返回对象的常量引用
const T& operator*() const& {
  // 如果当前对象是借用状态，则进行断言验证borrow的有效性
  if (isBorrowed_) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        MaybeOwnedTraits<T>::debugBorrowIsValid(borrow_));
  }
  // 返回借用状态下borrow的引用，否则返回own_的引用
  return C10_LIKELY(isBorrowed_)
      ? MaybeOwnedTraits<T>::referenceFromBorrow(borrow_)
      : own_;
}

// 指针访问操作符，返回对象的常量指针
const T* operator->() const {
  // 如果当前对象是借用状态，则进行断言验证borrow的有效性
  if (isBorrowed_) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        MaybeOwnedTraits<T>::debugBorrowIsValid(borrow_));
  }
  // 返回借用状态下borrow的指针，否则返回own_的地址
  return C10_LIKELY(isBorrowed_)
      ? MaybeOwnedTraits<T>::pointerFromBorrow(borrow_)
      : &own_;
}

// 解引用操作符（移动语义版本），返回对象的值
T operator*() && {
  // 如果当前对象是借用状态，则进行断言验证borrow的有效性，并返回borrow的引用
  // 否则，以移动语义返回own_的值
  if (isBorrowed_) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        MaybeOwnedTraits<T>::debugBorrowIsValid(borrow_));
    return MaybeOwnedTraits<T>::referenceFromBorrow(borrow_);
  } else {
    return std::move(own_);
  }
}
};

} // namespace c10
```