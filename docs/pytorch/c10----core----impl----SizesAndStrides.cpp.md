# `.\pytorch\c10\core\impl\SizesAndStrides.cpp`

```
#include <c10/core/impl/SizesAndStrides.h>  // 引入 SizesAndStrides 类的头文件

namespace c10::impl {

void SizesAndStrides::resizeSlowPath(
    const size_t newSize,
    const size_t oldSize) {  // SizesAndStrides 类的 resizeSlowPath 方法定义

  if (newSize <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE) {  // 如果 newSize 小于等于内联存储的最大尺寸

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        !isInline(),
        "resizeSlowPath called when fast path should have been hit!");  // 断言检查，确保不在内联状态下调用 resizeSlowPath

    int64_t* tempStorage = outOfLineStorage_;  // 临时存储指针指向非内联存储
    memcpy(
        &inlineStorage_[0],  // 将内联存储的内容复制到临时存储
        &tempStorage[0],
        C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE * sizeof(inlineStorage_[0]));
    memcpy(
        &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE],  // 将旧尺寸之后的内容复制到内联存储的后半部分
        &tempStorage[oldSize],
        C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE * sizeof(inlineStorage_[0]));
    // 不能在这里使用 freeOutOfLineStorage()！outOfLineStorage_ 已经被覆盖!
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    free(tempStorage);  // 释放临时存储的内存空间

  } else {  // 如果 newSize 大于内联存储的最大尺寸

    if (isInline()) {  // 如果当前处于内联状态

      // 不能在这里使用 allocateOutOfLineStorage(newSize)！会覆盖 inlineStorage_!
      int64_t* tempStorage =
          // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
          static_cast<int64_t*>(malloc(storageBytes(newSize)));  // 分配足够大小的内存空间
      TORCH_CHECK(
          tempStorage,
          "Could not allocate memory to change Tensor SizesAndStrides!");  // 检查内存分配是否成功

      const auto bytesToCopy = oldSize * sizeof(inlineStorage_[0]);  // 需要复制的字节数量
      const auto bytesToZero = (newSize > oldSize)
          ? (newSize - oldSize) * sizeof(tempStorage[0])
          : 0;  // 需要清零的字节数量

      memcpy(&tempStorage[0], &inlineStorage_[0], bytesToCopy);  // 复制旧尺寸部分的数据到新分配的内存中
      if (bytesToZero) {
        memset(&tempStorage[oldSize], 0, bytesToZero);  // 清零多出的部分
      }

      memcpy(
          &tempStorage[newSize],  // 复制旧尺寸之后的内容到新分配的内存中
          &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE],
          bytesToCopy);
      if (bytesToZero) {
        memset(&tempStorage[newSize + oldSize], 0, bytesToZero);  // 清零多出的部分
      }

      outOfLineStorage_ = tempStorage;  // 更新非内联存储指针为新分配的内存地址

    } else {  // 如果当前不处于内联状态

      const bool isGrowing = oldSize < newSize;  // 判断是否是扩展操作

      if (isGrowing) {
        // 在移动之前调整大小，以确保有足够的空间。
        resizeOutOfLineStorage(newSize);  // 调整非内联存储的大小
      }

      // 移动旧的步幅到它们的新起始点。注意，上述的内联路径中不会发生这种情况，因为步幅的起始点不会移动。
      memmove(
          outOfLineStorage_ + newSize,
          outOfLineStorage_ + oldSize,
          std::min(oldSize, newSize) * sizeof(outOfLineStorage_[0]));

      if (!isGrowing) {
        // 在移动之后调整大小，以防止数据丢失。
        resizeOutOfLineStorage(newSize);  // 调整非内联存储的大小
      } else {
        // 将大小部分的末尾清零。
        const auto bytesToZero =
            (newSize - oldSize) * sizeof(outOfLineStorage_[0]);  // 需要清零的字节数量
        memset(&outOfLineStorage_[oldSize], 0, bytesToZero);  // 清零多出的部分
        memset(&outOfLineStorage_[newSize + oldSize], 0, bytesToZero);  // 清零多出的部分
      }
    }
  }
  size_ = newSize;  // 更新对象的大小属性
}

} // namespace c10::impl  // c10::impl 命名空间结束
```