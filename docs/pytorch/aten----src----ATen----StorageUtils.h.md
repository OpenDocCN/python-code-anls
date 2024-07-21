# `.\pytorch\aten\src\ATen\StorageUtils.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <c10/core/Storage.h>
// 包含 ATen 的存储相关头文件
#include <c10/core/StorageImpl.h>
// 包含 ATen 的存储实现相关头文件
#include <c10/util/intrusive_ptr.h>
// 包含 ATen 的指针工具头文件

namespace at {

class TensorBase;
// 声明 TensorBase 类，用于后续使用

// 这里定义了一系列工具函数，用于创建/操作基于 ATen 的 c10 存储实现。

/**
 * Create a new shared memory storage impl managed by file descriptor
 *
 * @param size  size in bytes
 */
C10_EXPORT c10::intrusive_ptr<c10::StorageImpl> new_shm_fd_storage(size_t size);
// 创建一个由文件描述符管理的新的共享内存存储实现的函数声明，返回存储实现的指针

/**
 * Copy src to dst
 * Caller must guarantee the validness of the storage objects
 * during the entire copy process, esp. when it's async.
 *
 * This can probably live in c10 namespace later if needed,
 * but for now keep it in at to keep implementation simple.
 *
 * @param dst  dst tensor
 * @param src  src tensor
 * @param non_blocking  (default false) whether this operation blocks caller
 */
C10_EXPORT void storage_copy(
    c10::Storage& dst,
    const c10::Storage& src,
    bool non_blocking = false);
// 将源存储对象 src 复制到目标存储对象 dst 的函数声明，非阻塞模式由 non_blocking 参数控制

/**
 * In place change the storage to shm based.
 *
 * This is only applicable to CPU tensors not already shared.
 * Otherwise, it's a no op to mirror the THP tensor behavior:
 * https://pytorch.org/docs/stable/generated/torch.Tensor.share_memory_.html
 *
 * @param t  a tensor
 */
C10_EXPORT void share_memory_(TensorBase& t);
// 将张量 t 的存储改变为基于共享内存的函数声明，仅适用于尚未共享的 CPU 张量

} // namespace at
// 结束命名空间 at
```