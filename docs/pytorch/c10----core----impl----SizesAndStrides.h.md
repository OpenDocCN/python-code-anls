# `.\pytorch\c10\core\impl\SizesAndStrides.h`

```py
#pragma once

#include <algorithm>  // 包含标准库算法头文件
#include <cstdint>    // 包含标准整数类型头文件

#include <c10/macros/Macros.h>  // 包含 C10 宏定义头文件
#include <c10/util/ArrayRef.h>  // 包含 C10 数组引用工具头文件
#include <c10/util/SmallVector.h>  // 包含 C10 小向量头文件

#define C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE 5  // 定义最大内联大小为 5

namespace c10::impl {

// TensorImpl 尺寸和步幅的紧凑容器。
// 这种设计改进了先前使用 c10::SmallVector<int64_t, 5> 的方法，
// 通过专门为我们实际使用的操作进行优化，并强制尺寸和步幅数量相同。
// 内存布局如下：
//
// 1 个 size_t 用于大小
// 5 个八字节的内联尺寸和 5 个八字节的内联步幅，或者指向外部数组的指针
class C10_API SizesAndStrides {
 public:
  // TODO: different iterator types for sizes & strides to prevent
  // mixing the two accidentally.
  using sizes_iterator = int64_t*;                 // 尺寸的迭代器类型
  using sizes_const_iterator = const int64_t*;      // 尺寸的常量迭代器类型
  using strides_iterator = int64_t*;               // 步幅的迭代器类型
  using strides_const_iterator = const int64_t*;    // 步幅的常量迭代器类型

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  SizesAndStrides() {
    size_at_unchecked(0) = 0;      // 尺寸未检查时置零
    stride_at_unchecked(0) = 1;    // 步幅未检查时置一
  }

  ~SizesAndStrides() {
    if (C10_UNLIKELY(!isInline())) {
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      free(outOfLineStorage_);    // 如果不是内联，则释放内存
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  SizesAndStrides(const SizesAndStrides& rhs) : size_(rhs.size_) {
    if (C10_LIKELY(rhs.isInline())) {
      copyDataInline(rhs);    // 如果 rhs 是内联的，则复制数据到内联存储
    } else {
      allocateOutOfLineStorage(size_);   // 否则分配外部存储
      copyDataOutline(rhs);    // 复制数据到外部存储
    }
  }

  SizesAndStrides& operator=(const SizesAndStrides& rhs) {
    if (this == &rhs) {
      return *this;   // 自我赋值检查
    }
    if (C10_LIKELY(rhs.isInline())) {
      if (C10_UNLIKELY(!isInline())) {
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(outOfLineStorage_);    // 如果不是内联，释放外部存储
      }
      copyDataInline(rhs);    // 复制数据到内联存储
    } else {
      if (isInline()) {
        allocateOutOfLineStorage(rhs.size_);   // 如果是内联，分配外部存储
      } else {
        resizeOutOfLineStorage(rhs.size_);     // 否则调整外部存储大小
      }
      copyDataOutline(rhs);    // 复制数据到外部存储
    }
    size_ = rhs.size_;    // 更新大小
    return *this;
  }

  // Move from rhs. rhs.size() == 0 afterwards.
  SizesAndStrides(SizesAndStrides&& rhs) noexcept : size_(rhs.size_) {
    if (C10_LIKELY(isInline())) {
      memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
    } else {
      outOfLineStorage_ = rhs.outOfLineStorage_;
      rhs.outOfLineStorage_ = nullptr;
    }

    rhs.size_ = 0;    // 移动后 rhs 的大小置零
  }

  // Move from rhs. rhs.size() == 0 afterwards.
  SizesAndStrides& operator=(SizesAndStrides&& rhs) noexcept {
    if (this == &rhs) {
      return *this;   // 自我移动检查
    }
    if (C10_LIKELY(rhs.isInline())) {
      if (C10_UNLIKELY(!isInline())) {
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(outOfLineStorage_);    // 如果不是内联，释放外部存储
      }
      copyDataInline(rhs);    // 复制数据到内联存储
    } else {
      if (isInline()) {
        allocateOutOfLineStorage(rhs.size_);   // 如果是内联，分配外部存储
      } else {
        resizeOutOfLineStorage(rhs.size_);     // 否则调整外部存储大小
      }
      copyDataOutline(rhs);    // 复制数据到外部存储
    }
    size_ = rhs.size_;    // 更新大小
    return *this;
  }

 private:
  size_t size_;                           // 尺寸大小
  union {
    int64_t inlineStorage_[10];           // 内联存储，包括尺寸和步幅各 5 个八字节
    int64_t* outOfLineStorage_;           // 外部存储指针
  };

  bool isInline() const {
    return size_ <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE;   // 判断是否内联存储
  }

  int64_t& size_at_unchecked(size_t i) {
    return inlineStorage_[i];   // 返回指定位置的尺寸引用
  }

  int64_t& stride_at_unchecked(size_t i) {
    return inlineStorage_[i + C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];   // 返回指定位置的步幅引用
  }

  void allocateOutOfLineStorage(size_t size) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    outOfLineStorage_ = static_cast<int64_t*>(malloc(2 * size * sizeof(int64_t)));   // 分配外部存储空间
  }

  void resizeOutOfLineStorage(size_t size) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    outOfLineStorage_ = static_cast<int64_t*>(realloc(outOfLineStorage_, 2 * size * sizeof(int64_t)));   // 调整外部存储空间大小
  }

  void copyDataInline(const SizesAndStrides& rhs) {
    memcpy(inlineStorage_, rhs.inlineStorage_, 10 * sizeof(int64_t));   // 复制数据到内联存储
  }

  void copyDataOutline(const SizesAndStrides& rhs) {
    memcpy(outOfLineStorage_, rhs.outOfLineStorage_, 2 * rhs.size_ * sizeof(int64_t));   // 复制数据到外部存储
  }
};

}  // namespace c10::impl
    } else {
      // 如果是非内联存储，则需要偷取其向量数据。
      if (!isInline()) {
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        // 如果不是内联存储，释放当前的非内联存储空间
        free(outOfLineStorage_);
      }
      // 将 rhs 对象的非内联存储指针赋值给当前对象的非内联存储指针
      outOfLineStorage_ = rhs.outOfLineStorage_;
      // 将 rhs 对象的非内联存储指针置空
      rhs.outOfLineStorage_ = nullptr;
    }
    // 将 rhs 对象的 size_ 成员赋值给当前对象的 size_ 成员
    size_ = rhs.size_;
    // 将 rhs 对象的 size_ 成员置零
    rhs.size_ = 0;

    // 返回当前对象的引用
    return *this;
  }

  // 返回当前对象的 size_ 成员
  size_t size() const noexcept {
    return size_;
  }

  // 返回当前对象的 sizes 数据指针，用于只读访问
  const int64_t* sizes_data() const noexcept {
    // 如果是内联存储，则返回内联存储的地址
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      // 否则返回非内联存储的地址
      return &outOfLineStorage_[0];
    }
  }

  // 返回当前对象的 sizes 数据指针，用于可修改访问
  int64_t* sizes_data() noexcept {
    // 如果是内联存储，则返回内联存储的地址
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      // 否则返回非内联存储的地址
      return &outOfLineStorage_[0];
    }
  }

  // 返回 sizes 数据的常量迭代器的起始位置
  sizes_const_iterator sizes_begin() const noexcept {
    return sizes_data();
  }

  // 返回 sizes 数据的迭代器的起始位置
  sizes_iterator sizes_begin() noexcept {
    return sizes_data();
  }

  // 返回 sizes 数据的常量迭代器的结束位置
  sizes_const_iterator sizes_end() const noexcept {
    // 返回 sizes 数据的结束位置
    return sizes_begin() + size();
  }

  // 返回 sizes 数据的迭代器的结束位置
  sizes_iterator sizes_end() noexcept {
    // 返回 sizes 数据的结束位置
    return sizes_begin() + size();
  }

  // 返回 sizes 数据的 IntArrayRef 对象
  IntArrayRef sizes_arrayref() const noexcept {
    // 返回包含 sizes 数据的 IntArrayRef 对象
    return IntArrayRef{sizes_data(), size()};
  }

  // 设置当前对象的 sizes 数据
  void set_sizes(IntArrayRef newSizes) {
    // 调整当前对象的大小以适应新的 sizes 数据
    resize(newSizes.size());
    // 将新的 sizes 数据复制到当前对象的 sizes 数据中
    std::copy(newSizes.begin(), newSizes.end(), sizes_begin());
  }

  // 设置当前对象的 strides 数据
  void set_strides(IntArrayRef strides) {
    // 断言 strides 的大小与当前对象的 size_ 成员相等
    TORCH_INTERNAL_ASSERT(strides.size() == size());
    // 将 strides 数据复制到当前对象的 strides 数据中
    std::copy(strides.begin(), strides.end(), strides_begin());
  }

  // 返回当前对象的 strides 数据指针，用于只读访问
  const int64_t* strides_data() const noexcept {
    // 如果是内联存储，则返回内联存储后 C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE 处的地址
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      // 否则返回非内联存储后 size_ 处的地址
      return &outOfLineStorage_[size()];
    }
  }

  // 返回当前对象的 strides 数据指针，用于可修改访问
  int64_t* strides_data() noexcept {
    // 如果是内联存储，则返回内联存储后 C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE 处的地址
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      // 否则返回非内联存储后 size_ 处的地址
      return &outOfLineStorage_[size()];
    }
  }

  // 返回 strides 数据的常量迭代器的起始位置
  strides_const_iterator strides_begin() const noexcept {
    // 如果是内联存储，则返回内联存储后 C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE 处的地址
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      // 否则返回非内联存储后 size_ 处的地址
      return &outOfLineStorage_[size()];
    }
  }

  // 返回 strides 数据的迭代器的起始位置
  strides_iterator strides_begin() noexcept {
    // 如果是内联存储，则返回内联存储后 C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE 处的地址
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      // 否则返回非内联存储后 size_ 处的地址
      return &outOfLineStorage_[size()];
    }
  }

  // 返回 strides 数据的常量迭代器的结束位置
  strides_const_iterator strides_end() const noexcept {
    // 返回 strides 数据的结束位置
    return strides_begin() + size();
  }

  // 返回 strides 数据的迭代器的结束位置
  strides_iterator strides_end() noexcept {
    // 返回 strides 数据的结束位置
    return strides_begin() + size();
  }

  // 返回 strides 数据的 IntArrayRef 对象
  IntArrayRef strides_arrayref() const noexcept {
    // 返回包含 strides 数据的 IntArrayRef 对象
    return IntArrayRef{strides_data(), size()};
  }

  // 访问指定索引的 size 数据，带断言检查索引合法性
  int64_t size_at(size_t idx) const noexcept {
    // 断言索引 idx 小于当前对象的 size_
    assert(idx < size());
    // 返回指定索引处的 size 数据
    return sizes_data()[idx];
  }

  // 访问指定索引的 size 数据的引用，带断言检查索引合法性
  int64_t& size_at(size_t idx) noexcept {
    // 断言索引 idx 小于当前对象的 size_
    assert(idx < size());
    // 返回指定索引处 size 数据的引用
    return sizes_data()[idx];
  }

  // 不带断言检查地访问指定索引的 size 数据
  int64_t size_at_unchecked(size_t idx) const noexcept {
  // 返回给定索引处的大小数据
  return sizes_data()[idx];
}

  // 返回给定索引处的大小数据的可变引用
int64_t& size_at_unchecked(size_t idx) noexcept {
  return sizes_data()[idx];
}

  // 返回给定索引处的步幅数据
int64_t stride_at(size_t idx) const noexcept {
  assert(idx < size());
  return strides_data()[idx];
}

  // 返回给定索引处的步幅数据的可变引用
int64_t& stride_at(size_t idx) noexcept {
  assert(idx < size());
  return strides_data()[idx];
}

  // 返回给定索引处的步幅数据（无检查）
int64_t stride_at_unchecked(size_t idx) const noexcept {
  return strides_data()[idx];
}

  // 返回给定索引处的步幅数据的可变引用（无检查）
int64_t& stride_at_unchecked(size_t idx) noexcept {
  return strides_data()[idx];
}

  // 调整大小为给定的 newSize
void resize(size_t newSize) {
  const auto oldSize = size();
  // 如果新大小等于旧大小，则直接返回
  if (newSize == oldSize) {
    return;
  }
  // 如果新大小小于等于最大内联大小，并且当前对象是内联的
  if (C10_LIKELY(
          newSize <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE && isInline())) {
    // 如果扩展大小大于旧大小，则对内联存储进行扩展并清零新增部分
    if (oldSize < newSize) {
      const auto bytesToZero =
          (newSize - oldSize) * sizeof(inlineStorage_[0]);
      memset(&inlineStorage_[oldSize], 0, bytesToZero);
      memset(
          &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE + oldSize],
          0,
          bytesToZero);
    }
    // 更新对象的大小
    size_ = newSize;
  } else {
    // 否则，调用慢路径调整大小
    resizeSlowPath(newSize, oldSize);
  }
}

  // 慢路径调整大小的具体实现
void resizeSlowPath(size_t newSize, size_t oldSize);

private:
  // 判断对象是否是内联存储
  bool isInline() const noexcept {
    return size_ <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE;
  }

  // 从另一个 SizesAndStrides 对象复制内联数据
  void copyDataInline(const SizesAndStrides& rhs) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.isInline());
    memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
  }

  // 计算给定大小所需的存储字节数
  static size_t storageBytes(size_t size) noexcept {
    return size * 2 * sizeof(int64_t);
  }

  // 分配足够大的非内联存储
  void allocateOutOfLineStorage(size_t size) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    outOfLineStorage_ = static_cast<int64_t*>(malloc(storageBytes(size)));
    TORCH_CHECK(
        outOfLineStorage_,
        "Could not allocate memory for Tensor SizesAndStrides!");
  }

  // 调整非内联存储的大小
  void resizeOutOfLineStorage(size_t newSize) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!isInline());
    outOfLineStorage_ = static_cast<int64_t*>(
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        realloc(outOfLineStorage_, storageBytes(newSize)));
    TORCH_CHECK(
        outOfLineStorage_,
        "Could not allocate memory for Tensor SizesAndStrides!");
  }

  // 从另一个 SizesAndStrides 对象复制非内联数据
  void copyDataOutline(const SizesAndStrides& rhs) noexcept {
    memcpy(outOfLineStorage_, rhs.outOfLineStorage_, storageBytes(rhs.size_));
  }

  size_t size_{1};
  union {
    int64_t* outOfLineStorage_;
    // NOLINTNEXTLINE(*c-array*)
    int64_t inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE * 2]{};
  };
};

} // namespace c10::impl
```