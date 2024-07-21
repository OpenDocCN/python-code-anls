# `.\pytorch\aten\src\ATen\core\Tensor.h`

```
#pragma`
#pragma once
// 防止头文件被多次包含

#include <ATen/core/TensorBody.h>
// 引入 ATen 库中的 TensorBody 头文件

#include <c10/util/Exception.h>
// 引入 c10 库中的 Exception 头文件

namespace at {
// 进入 at 命名空间

class TORCH_API OptionalTensorRef {
// 定义 OptionalTensorRef 类，用于包装可选的 Tensor 引用

 public:
  OptionalTensorRef() = default;
  // 默认构造函数

  ~OptionalTensorRef() {
    ref_.unsafeReleaseTensorImpl();
    // 析构函数，释放 TensorImpl，不安全释放
  }

  OptionalTensorRef(const TensorBase& src)
      : ref_(Tensor::unsafe_borrow_t{}, src) {
    // 从 TensorBase 构造 OptionalTensorRef，使用不安全借用构造 Tensor
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src.defined());
    // 调试模式下，断言 src 是已定义的
  }

  OptionalTensorRef(const OptionalTensorRef& rhs)
      : ref_(Tensor::unsafe_borrow_t{}, rhs.ref_) {}
      // OptionalTensorRef 的复制构造函数，使用不安全借用构造 Tensor

  OptionalTensorRef& operator=(OptionalTensorRef rhs) {
    std::swap(ref_, rhs.ref_);
    return *this;
    // OptionalTensorRef 的赋值运算符重载，交换操作数的 Tensor 引用
  }

  bool has_value() const {
    return ref_.defined();
    // 检查是否存在值，即 Tensor 是否已定义
  }

  const Tensor& getTensorRef() const & {
    return ref_;
    // 获取 Tensor 的常量引用
  }

  const Tensor& operator*() const & {
    return ref_;
    // 解引用操作符，返回 Tensor 的常量引用
  }

  const Tensor* operator->() const & {
    return &ref_;
    // 指针访问操作符，返回 Tensor 的常量指针
  }

  operator bool() const {
    return ref_.defined();
    // 转换为 bool 类型，检查 Tensor 是否已定义
  }

 private:
  Tensor ref_;
  // 私有成员变量，存储 Tensor 引用
};

// 用于将可能未定义的 TensorBase 转换为 at::Tensor，而无需增加引用计数
class TORCH_API TensorRef {
 public:
  ~TensorRef() {
    ref_.unsafeReleaseTensorImpl();
    // 析构函数，释放 TensorImpl，不安全释放
  }

  TensorRef(const TensorBase& src)
      : ref_(Tensor::unsafe_borrow_t{}, src) {}
      // TensorRef 构造函数，使用不安全借用构造 Tensor

  const Tensor& operator*() const & {
    return ref_;
    // 解引用操作符，返回 Tensor 的常量引用
  }

 private:
  Tensor ref_;
  // 私有成员变量，存储 Tensor 引用
};

template <typename T>
// 模板函数，用于注册 hook 到 Tensor

auto Tensor::register_hook(T&& hook) const -> Tensor::hook_return_void_t<T> {
  // 返回 void 类型 hook 的 grad 参数，以便具有 Tensor 返回类型的 std::function
  static_assert(std::is_same<decltype(hook(Tensor())), void>::value,
                "Expected hook to return void");
  // 静态断言，确保 hook 返回 void 类型
  return _register_hook([fn=std::forward<T>(hook)](const TensorBase& grad_base) {
    TensorRef grad(grad_base);
    fn(*grad);
    return Tensor();
    // 注册 hook，调用传入的 hook 函数，返回一个空的 Tensor
  });
}

template <typename T>
// 模板函数，用于注册 hook 到 Tensor

auto Tensor::register_hook(T&& hook) const -> Tensor::hook_return_var_t<T> {
  return _register_hook([fn=std::forward<T>(hook)](const TensorBase& grad_base) {
    TensorRef grad(grad_base);
    Tensor ret = fn(*grad);
    return TensorBase(std::move(ret));
    // 注册 hook，调用传入的 hook 函数，返回包装的 TensorBase
  });
}

} // namespace at
// 结束 at 命名空间
```