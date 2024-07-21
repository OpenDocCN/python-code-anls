# `.\pytorch\aten\src\ATen\StorageUtils.cpp`

```
// 包含 ATen 库的头文件，提供各种张量和存储操作的函数和类
#include <ATen/Functions.h>
#include <ATen/MapAllocator.h>
#include <ATen/StorageUtils.h>
#include <c10/core/TensorOptions.h>

// ATen 命名空间
namespace at {

// 创建一个新的共享内存文件描述符存储，返回 StorageImpl 智能指针
C10_EXPORT c10::intrusive_ptr<c10::StorageImpl> new_shm_fd_storage(
    size_t size) {
  // 定义共享内存分配器的标志
  int flags = ALLOCATOR_MAPPED_SHAREDMEM | ALLOCATOR_MAPPED_EXCLUSIVE |
      ALLOCATOR_MAPPED_KEEPFD | ALLOCATOR_MAPPED_UNLINK;
  // 获取一个新的进程范围的共享内存句柄
  std::string handle = NewProcessWideShmHandle();
  // 使用共享内存句柄创建 MapAllocator 数据指针
  auto sptr = MapAllocator::makeDataPtr(
      handle.c_str(), flags, size * sizeof(uint8_t), nullptr);
  // 创建并返回一个不可调整大小的 StorageImpl 对象
  return c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size,
      std::move(sptr),
      /*allocator=*/nullptr,
      /*resizable=*/false);
}

// 在不同存储之间复制数据，支持非阻塞操作
C10_EXPORT void storage_copy(
    c10::Storage& dst,
    const c10::Storage& src,
    bool non_blocking) {
  // 构造目标存储的 TensorOptions
  auto dst_options = c10::TensorOptions().device(dst.device()).dtype(at::kByte);
  // 创建一个空张量 dst_t，设置其存储为 dst
  auto dst_t = at::empty({0}, dst_options).set_(dst);

  // 构造源存储的 TensorOptions
  auto src_options = c10::TensorOptions().device(src.device()).dtype(at::kByte);
  // 创建一个空张量 src_t，设置其存储为 src
  auto src_t = at::empty({0}, src_options).set_(src);
  // 将 src_t 数据复制到 dst_t 中，支持非阻塞操作
  dst_t.copy_(src_t, non_blocking);
}

// 将张量的存储转换为共享内存存储（如果尚未共享）
C10_EXPORT void share_memory_(TensorBase& t) {
  // 如果张量不在 CPU 上，则返回
  if (t.device() != at::kCPU) {
    return;
  }

  // 获取原始存储
  const at::Storage& origStorage = t.storage();

  // 如果原始存储已经是共享的，则直接返回
  if (MapAllocator::fromDataPtr(origStorage.data_ptr()) != nullptr) {
    // already shared
    return;
  }

  // 创建新的共享内存存储
  at::Storage newStorage(new_shm_fd_storage(origStorage.nbytes()));
  // 将原始存储的数据复制到新的共享内存存储中
  storage_copy(newStorage, origStorage);

  // 替换原始存储的数据指针和分配器为新创建的共享内存存储的内容
  c10::StorageImpl* origStorageImpl = origStorage.unsafeGetStorageImpl();
  c10::StorageImpl* newStorageImpl = newStorage.unsafeGetStorageImpl();
  origStorageImpl->set_data_ptr(std::move(newStorageImpl->mutable_data_ptr()));
  origStorageImpl->set_allocator(newStorageImpl->allocator());
}

} // namespace at
```