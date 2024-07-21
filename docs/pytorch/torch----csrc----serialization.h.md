# `.\pytorch\torch\csrc\serialization.h`

```py
#ifndef THP_SERIALIZATION_INC
#define THP_SERIALIZATION_INC

#include <c10/core/StorageImpl.h>
#include <c10/util/intrusive_ptr.h>

// 定义了两个函数模板，用于从文件描述符读取数据和向文件描述符写入数据
template <class io>
void doRead(io fildes, void* buf, size_t nbytes);

template <class io>
void doWrite(io fildes, void* buf, size_t nbytes);

// 注意：此函数接受可变的 StorageImpl 对象，因为它可能传递到 at::from_blob 函数。
// 从给定的文件描述符 fd 中写入 StorageImpl 对象 self 的原始数据。
// 如果 save_size 为 true，则保存存储大小信息；element_size 表示每个元素的大小。
template <class io>
void THPStorage_writeFileRaw(
    c10::StorageImpl* self,
    io fd,
    bool save_size,
    uint64_t element_size);

// 从给定的文件描述符 fd 中读取原始数据，并将其填充到 storage 指向的 StorageImpl 对象中。
// element_size 表示每个元素的大小。
template <class io>
c10::intrusive_ptr<c10::StorageImpl> THPStorage_readFileRaw(
    io fd,
    c10::intrusive_ptr<c10::StorageImpl> storage,
    uint64_t element_size);

#endif
```