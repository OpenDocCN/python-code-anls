# `.\pytorch\aten\src\ATen\core\UnsafeFromTH.h`

```
#pragma once
// 包含 ATen 库中的 Tensor 头文件
#include <ATen/core/Tensor.h>

// 定义命名空间 at
namespace at {

// 定义一个内联函数，从 TH 指针创建不安全的 Tensor
inline Tensor unsafeTensorFromTH(void * th_pointer, bool retain) {
    // 通过 th_pointer 恢复 TensorImpl 指针，并封装为引用计数指针 tensor_impl
    auto tensor_impl = c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(static_cast<TensorImpl*>(th_pointer));
    // 如果需要保留并且 tensor_impl 不是 UndefinedTensorImpl 的单例，则增加其引用计数
    if (retain && tensor_impl.get() != UndefinedTensorImpl::singleton()) {
        c10::raw::intrusive_ptr::incref(tensor_impl.get());
    }
    // 返回一个 Tensor 对象，使用移动语义将 tensor_impl 转移
    return Tensor(std::move(tensor_impl));
}

// 定义一个内联函数，从 TH 指针创建不安全的 Storage
inline Storage unsafeStorageFromTH(void * th_pointer, bool retain) {
    // 如果需要保留并且 th_pointer 非空，则增加其引用计数
    if (retain && th_pointer) {
        c10::raw::intrusive_ptr::incref(static_cast<StorageImpl*>(th_pointer));
    }
    // 返回一个 Storage 对象，使用移动语义将从 th_pointer 恢复的 StorageImpl 转移
    return Storage(c10::intrusive_ptr<StorageImpl>::reclaim(static_cast<StorageImpl*>(th_pointer)));
}

// 命名空间 at 结束
}
```