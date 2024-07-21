# `.\pytorch\c10\util\ExclusivelyOwnedTensorTraits.h`

```
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <c10/core/TensorImpl.h>
// 引入 c10 库中的 TensorImpl 头文件
#include <c10/core/UndefinedTensorImpl.h>
// 引入 c10 库中的 UndefinedTensorImpl 头文件

#include <utility>
// 引入标准库 utility，用于提供通用的编程组件

namespace c10 {
// 进入 c10 命名空间

// 共享的 ExclusivelyOwnedTraits 实现，用于 caffe2::Tensor 和 at::TensorBase 之间
template <typename TensorType>
struct ExclusivelyOwnedTensorTraits {
  // 定义 ExclusivelyOwnedTensorTraits 模板结构体，适用于 TensorType 类型

  using repr_type = TensorType;
  // 定义 repr_type 为 TensorType 类型
  using pointer_type = TensorType*;
  // 定义 pointer_type 为 TensorType 指针类型
  using const_pointer_type = const TensorType*;
  // 定义 const_pointer_type 为常量 TensorType 指针类型

  static repr_type nullRepr() {
    // 静态方法，返回默认构造的 TensorType 对象
    return TensorType();
  }

  template <class... Args>
  static repr_type createInPlace(Args&&... args) {
    // 静态方法模板，以传递的参数 Args&&... args 构造并返回 TensorType 对象
    return TensorType(std::forward<Args>(args)...);
  }

  static repr_type moveToRepr(TensorType&& x) {
    // 静态方法，移动语义地返回传入的 TensorType 对象 x
    return std::move(x);
  }

  static void destroyOwned(TensorType& x) {
    // 静态方法，销毁拥有的 TensorType 对象 x

    TensorImpl* const toDestroy = x.unsafeReleaseTensorImpl();
    // 调用 x 的 unsafeReleaseTensorImpl 方法，获取需要销毁的 TensorImpl 对象指针

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        toDestroy != nullptr, "Tensor somehow got null TensorImpl?");
    // 断言：确保 toDestroy 不为空指针，否则输出错误信息

    const bool isUndefined = toDestroy == UndefinedTensorImpl::singleton();
    // 检查 toDestroy 是否是 UndefinedTensorImpl 的单例对象

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        toDestroy->refcount_ == 1 || (toDestroy->refcount_ == 0 && isUndefined),
        "ExclusivelyOwned<Tensor> destroyed with isUndefined ",
        isUndefined,
        " and refcount ",
        toDestroy->refcount_,
        ", expected 1 or, if isUndefined, 0!");
    // 断言：确保 refcount 符合预期值，若 isUndefined 为真，则 refcount 应为 0

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        toDestroy->weakcount_ == 1 ||
            (toDestroy->weakcount_ == 0 &&
             toDestroy == UndefinedTensorImpl::singleton()),
        "ExclusivelyOwned<Tensor> destroyed with isUndefined ",
        isUndefined,
        " and weakcount ",
        toDestroy->weakcount_,
        ", expected 1 or, if isUndefined, 0!");
    // 断言：确保 weakcount 符合预期值，若 isUndefined 为真，则 weakcount 应为 0

    if (!isUndefined) {
      // 如果不是 UndefinedTensorImpl 的单例对象

#ifndef NDEBUG
      // 如果处于调试模式

      // 必须设置为 0 才能通过 ~intrusive_ptr_target 中的调试断言
      toDestroy->refcount_ = 0;
      toDestroy->weakcount_ = 0;
#endif

      // 删除 toDestroy 指向的 TensorImpl 对象
      delete toDestroy;
    }
  }

  static TensorType take(TensorType& x) {
    // 静态方法，移动语义地返回传入的 TensorType 对象 x
    return std::move(x);
  }

  static pointer_type getImpl(repr_type& x) {
    // 静态方法，返回指向传入的 TensorType 对象 x 的指针
    return &x;
  }

  static const_pointer_type getImpl(const repr_type& x) {
    // 静态方法，返回指向常量传入的 TensorType 对象 x 的指针
    return &x;
  }
};
} // namespace c10
// 退出 c10 命名空间
```