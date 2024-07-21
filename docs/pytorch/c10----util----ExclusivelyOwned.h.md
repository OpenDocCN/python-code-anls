# `.\pytorch\c10\util\ExclusivelyOwned.h`

```
// 防止头文件被多次包含
#pragma once

// 包含 <utility> 头文件，用于提供一些泛型编程组件，如 std::forward
#include <utility>

// 声明 c10 命名空间
namespace c10 {

// ExclusivelyOwnedTraits 结构模板的前向声明，用于定义 ExclusivelyOwned 的行为特征
template <typename T>
struct ExclusivelyOwnedTraits;

// ExclusivelyOwned 是对某种类型 T 的独占所有智能指针样式的包装器
template <typename T>
class ExclusivelyOwned {
  // 使用 ExclusivelyOwnedTraits<T> 中定义的 repr_type 类型来存储 T 的实例
  typename ExclusivelyOwnedTraits<T>::repr_type repr_;

 public:
  // 默认构造函数，初始化 repr_ 为 nullRepr()
  ExclusivelyOwned() : repr_(ExclusivelyOwnedTraits<T>::nullRepr()) {}

  // 移动构造函数，使用 moveToRepr 将传入的 T&& 移动到 repr_ 中
  explicit ExclusivelyOwned(T&& t) : repr_(ExclusivelyOwnedTraits<T>::moveToRepr(std::move(t))) {}

  // in_place 构造函数模板，使用 createInPlace 在原地创建 T 实例并移动到 repr_
  template <class... Args>
  explicit ExclusivelyOwned(std::in_place_t, Args&&... args)
      : repr_(ExclusivelyOwnedTraits<T>::createInPlace(std::forward<Args>(args)...)) {}

  // 复制构造函数被删除，禁止使用复制构造 ExclusivelyOwned 对象
  ExclusivelyOwned(const ExclusivelyOwned&) = delete;

  // 移动构造函数，移动另一个 ExclusivelyOwned 对象的 repr_ 到当前对象中
  ExclusivelyOwned(ExclusivelyOwned&& rhs) noexcept
      : repr_(std::move(rhs.repr_)) {
    rhs.repr_ = ExclusivelyOwnedTraits<T>::nullRepr();
  }

  // 复制赋值操作符被删除，禁止使用复制赋值操作符
  ExclusivelyOwned& operator=(const ExclusivelyOwned&) = delete;

  // 移动赋值操作符，销毁当前 repr_，然后将 rhs 的 repr_ 移动到当前对象，并将 rhs 的 repr_ 置为 nullRepr()
  ExclusivelyOwned& operator=(ExclusivelyOwned&& rhs) noexcept {
    ExclusivelyOwnedTraits<T>::destroyOwned(repr_);
    repr_ = std::move(rhs.repr_);
    rhs.repr_ = ExclusivelyOwnedTraits<T>::nullRepr();
    return *this;
  }
  return *this;
}

ExclusivelyOwned& operator=(T&& rhs) noexcept {
  // 销毁当前对象的资源
  EOT::destroyOwned(repr_);
  // 将右值 rhs 移动到当前对象的 repr_ 中
  repr_ = EOT::moveToRepr(std::move(rhs));
  // 返回当前对象的引用
  return *this;
}

~ExclusivelyOwned() {
  // 销毁当前对象的资源
  EOT::destroyOwned(repr_);
  // 不需要调用 repr_ 的析构函数，因为在 destroyOwned 中已经专门处理了
  // 独占所有权情况的资源释放！
}

// 我们不提供此操作符，因为这将要求我们能够区分拥有但空的 T 和缺少 T 的情况。
// 对于 Tensor 来说尤其棘手，它希望使用未定义的 Tensor 作为其空状态。
explicit operator bool() const noexcept = delete;

operator T() && {
  // 调用 take() 方法获取当前对象的资源，并返回
  return take();
}

// 注意：在 MaybeOwned 上，对应的操作是移动操作符 *()。
// 对于 ExclusivelyOwned，take() 和 operator*() 可能有不同的返回类型，因此它们是不同的函数。
T take() && {
  // 调用 take() 方法获取当前对象的资源，并返回
  return EOT::take(repr_);
}

typename EOT::const_pointer_type operator->() const {
  // 调用 get() 方法获取当前对象的资源指针，并返回常量指针类型
  return get();
}

typename EOT::const_pointer_type get() const {
  // 调用 getImpl() 方法获取当前对象的资源指针，并返回常量指针类型
  return EOT::getImpl(repr_);
}

typename EOT::pointer_type operator->() {
  // 调用 get() 方法获取当前对象的资源指针，并返回非常量指针类型
  return get();
}

typename EOT::pointer_type get() {
  // 调用 getImpl() 方法获取当前对象的资源指针，并返回非常量指针类型
  return EOT::getImpl(repr_);
}

std::remove_pointer_t<typename EOT::const_pointer_type>& operator*() const {
  // 解引用 get() 方法获取的常量指针类型资源，并返回引用
  return *get();
}

std::remove_pointer_t<typename EOT::pointer_type>& operator*() {
  // 解引用 get() 方法获取的非常量指针类型资源，并返回引用
  return *get();
}
};

// 结束命名空间 c10
} // namespace c10
```