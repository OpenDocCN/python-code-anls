# `.\pytorch\c10\core\impl\COW.h`

```py
#pragma once
// 使用 pragma once 确保头文件只被编译一次

#include <c10/macros/Macros.h>
// 包含 c10 库中的宏定义头文件

#include <c10/util/intrusive_ptr.h>
// 包含 c10 库中的 intrusive_ptr 实现头文件

namespace c10 {
struct StorageImpl;
class DataPtr;
}; // namespace c10
// 声明 c10 命名空间，包含 StorageImpl 结构体和 DataPtr 类

namespace c10::impl::cow {

// 创建给定 storage 的 Copy-on-write（COW）克隆。如果 storage 不是 COW，
// 则会将其转换为 COW 存储。
//
// 如果 storage 的 DataPtr 具有非等于数据指针的上下文（`DataPtr::get_context()`），
// 则无法成功将其转换为 COW 存储。在这种情况下，返回 nullptr。
C10_API c10::intrusive_ptr<StorageImpl> lazy_clone_storage(
    StorageImpl& storage);
// 声明 lazy_clone_storage 函数，返回一个 intrusive_ptr 智能指针，指向 StorageImpl 结构体

// 检查 storage 是否具有简单的 DataPtr，没有异常上下文
C10_API bool has_simple_data_ptr(const c10::StorageImpl& storage);
// 声明 has_simple_data_ptr 函数，检查给定 StorageImpl 是否有简单的 DataPtr

// 检查 DataPtr 是否为 COW
C10_API bool is_cow_data_ptr(const c10::DataPtr& data_ptr);
// 声明 is_cow_data_ptr 函数，检查给定 DataPtr 是否为 COW

// 立即复制 COW 存储的数据，将其转换为非 COW 存储
C10_API void materialize_cow_storage(StorageImpl& storage);
// 声明 materialize_cow_storage 函数，用于将 COW 存储的数据立即复制，转换为非 COW 存储

} // namespace c10::impl::cow
// c10::impl::cow 命名空间结束
```