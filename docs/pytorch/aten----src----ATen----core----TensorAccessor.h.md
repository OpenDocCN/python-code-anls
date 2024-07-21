# `.\pytorch\aten\src\ATen\core\TensorAccessor.h`

```
#pragma once

#include <c10/macros/Macros.h>  // 引入 C10 库的宏定义
#include <c10/util/ArrayRef.h>  // 引入 C10 库的 ArrayRef 类
#include <c10/util/Deprecated.h>  // 引入 C10 库的 Deprecated 功能
#include <c10/util/Exception.h>  // 引入 C10 库的异常处理功能
#include <c10/util/irange.h>  // 引入 C10 库的 irange 功能
#include <cstddef>  // 标准库：定义了 size_t 类型
#include <cstdint>  // 标准库：定义了 int64_t 类型
#include <type_traits>  // 标准库：定义了类型特性

namespace at {

// TensorAccessorBase 和 TensorAccessor 用于 CPU 和 CUDA 张量。
// 对于 CUDA 张量，它仅在设备代码中使用。这意味着我们限制自己只使用那里可用的函数和类型（例如，没有 IntArrayRef）。

// PtrTraits 参数仅在 CUDA 中才相关，用于支持 `__restrict__` 指针。
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class TensorAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;  // 定义指针类型 PtrType

  C10_HOST_DEVICE TensorAccessorBase(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : data_(data_), sizes_(sizes_), strides_(strides_) {}  // 构造函数，初始化 data_, sizes_, strides_

  C10_HOST IntArrayRef sizes() const {
    return IntArrayRef(sizes_,N);  // 返回 sizes_ 的 IntArrayRef
  }

  C10_HOST IntArrayRef strides() const {
    return IntArrayRef(strides_,N);  // 返回 strides_ 的 IntArrayRef
  }

  C10_HOST_DEVICE index_t stride(index_t i) const {
    return strides_[i];  // 返回第 i 维的步长
  }

  C10_HOST_DEVICE index_t size(index_t i) const {
    return sizes_[i];  // 返回第 i 维的大小
  }

  C10_HOST_DEVICE PtrType data() {
    return data_;  // 返回数据指针
  }

  C10_HOST_DEVICE const PtrType data() const {
    return data_;  // 返回数据指针（常量版本）
  }

protected:
  PtrType data_;  // 数据指针
  const index_t* sizes_;  // 大小数组指针
  const index_t* strides_;  // 步长数组指针
};

// TensorAccessor 通常用于 CPU 张量，使用 `Tensor.accessor<T, N>()`。
// 对于 CUDA 张量，在主机上使用 `GenericPackedTensorAccessor`，仅在设备上使用 `TensorAccessor` 进行索引。
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class TensorAccessor : public TensorAccessorBase<T,N,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;  // 定义指针类型 PtrType

  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : TensorAccessorBase<T, N, PtrTraits, index_t>(data_,sizes_,strides_) {}  // 构造函数，调用基类构造函数初始化

  C10_HOST_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
    // 返回一个新的 TensorAccessor，表示对第一维进行索引操作
  }

  C10_HOST_DEVICE const TensorAccessor<T, N-1, PtrTraits, index_t> operator[](index_t i) const {
    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
    // 返回一个新的 TensorAccessor（常量版本），表示对第一维进行索引操作
  }
};

}  // namespace at
// 定义一个模板类 TensorAccessor，用于访问一维张量的数据，继承自 TensorAccessorBase
template<typename T, template <typename U> class PtrTraits, typename index_t>
class TensorAccessor<T,1,PtrTraits,index_t> : public TensorAccessorBase<T,1,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  // 构造函数，接受数据指针 data_、大小数组 sizes_ 和步长数组 strides_
  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : TensorAccessorBase<T, 1, PtrTraits, index_t>(data_,sizes_,strides_) {}

  // 重载运算符[]，返回索引 i 处的数据引用，通过步长 strides_[0] 计算偏移量
  C10_HOST_DEVICE T & operator[](index_t i) {
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    return this->data_[this->strides_[0]*i];
  }

  // 常量版本的重载运算符[]，返回索引 i 处的数据引用，不可修改
  C10_HOST_DEVICE const T & operator[](index_t i) const {
    return this->data_[this->strides_[0]*i];
  }
};


// GenericPackedTensorAccessorBase 和 GenericPackedTensorAccessor 用于处理 CUDA 上的张量
// 与 TensorAccessor 不同，它们在实例化时复制步长和大小（在主机上）
// 这样在调用内核时可以在设备上传递它们
// 在设备上，多维张量的索引给予 TensorAccessor
// 如果要将张量数据指针标记为 __restrict__，请使用 RestrictPtrTraits 作为 PtrTraits
// 只在主机上需要从数据、大小和步长实例化，因为设备上不可用 std::copy
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class GenericPackedTensorAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  // 构造函数，接受数据指针 data_、大小数组 sizes_ 和步长数组 strides_
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  C10_HOST GenericPackedTensorAccessorBase(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : data_(data_) {
    // 复制 sizes_ 和 strides_ 到对象的大小和步长数组中
    std::copy(sizes_, sizes_ + N, std::begin(this->sizes_));
    std::copy(strides_, strides_ + N, std::begin(this->strides_));
  }

  // 如果 index_t 不是 int64_t，我们希望有一个 int64_t 的构造函数
  template <typename source_index_t, class = std::enable_if_t<std::is_same_v<source_index_t, int64_t>>>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  C10_HOST GenericPackedTensorAccessorBase(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : data_(data_) {
    // 通过循环将 sizes_ 和 strides_ 复制到对象的大小和步长数组中
    for (const auto i : c10::irange(N)) {
      this->sizes_[i] = sizes_[i];
      this->strides_[i] = strides_[i];
    }
  }

  // 返回索引 i 处的步长
  C10_HOST_DEVICE index_t stride(index_t i) const {
    return strides_[i];
  }

  // 返回索引 i 处的大小
  C10_HOST_DEVICE index_t size(index_t i) const {
    return sizes_[i];
  }

  // 返回数据指针
  C10_HOST_DEVICE PtrType data() {
    return data_;
  }

  // 返回常量数据指针
  C10_HOST_DEVICE const PtrType data() const {
    return data_;
  }

protected:
  PtrType data_;         // 数据指针
  // NOLINTNEXTLINE(*c-arrays*)
  index_t sizes_[N];     // 大小数组
  // NOLINTNEXTLINE(*c-arrays*)
  index_t strides_[N];   // 步长数组

  // 保护方法，用于检查索引是否超出边界
  C10_HOST void bounds_check_(index_t i) const {

  }
};
    # 使用 TORCH_CHECK_INDEX 宏检查索引 i 是否在有效范围内，确保其值大于等于 0 并且小于维度 N 的值
    TORCH_CHECK_INDEX(
        0 <= i && i < index_t{N},
        "Index ",
        i,
        " is not within bounds of a tensor of dimension ",
        N);
};

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class GenericPackedTensorAccessor : public GenericPackedTensorAccessorBase<T,N,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  // 构造函数，初始化泛型压缩张量访问器，使用给定的数据指针、尺寸数组和步幅数组
  C10_HOST GenericPackedTensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

  // 如果 index_t 不是 int64_t 类型，则定义一个接受 int64_t 类型的构造函数
  template <typename source_index_t, class = std::enable_if_t<std::is_same_v<source_index_t, int64_t>>>
  C10_HOST GenericPackedTensorAccessor(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

  // 重载运算符，返回一个降维后的张量访问器对象，索引为 i
  C10_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
    index_t* new_sizes = this->sizes_ + 1;
    index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
  }

  // 重载运算符，返回一个常量版本的降维后的张量访问器对象，索引为 i
  C10_DEVICE const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) const {
    const index_t* new_sizes = this->sizes_ + 1;
    const index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
  }

  /// 返回一个经过转置后的相同维度的压缩张量访问器
  /// 通过交换尺寸/步幅数组中的元素实现转置，不实际移动元素
  /// 如果维度无效，则会断言错误
  C10_HOST GenericPackedTensorAccessor<T, N, PtrTraits, index_t> transpose(
      index_t dim1,
      index_t dim2) const {
    // 检查维度是否在有效范围内
    this->bounds_check_(dim1);
    this->bounds_check_(dim2);
    // 创建结果对象，与当前对象共享数据、尺寸和步幅数组
    GenericPackedTensorAccessor<T, N, PtrTraits, index_t> result(
        this->data_, this->sizes_, this->strides_);
    // 交换指定维度的步幅和尺寸
    std::swap(result.strides_[dim1], result.strides_[dim2]);
    std::swap(result.sizes_[dim1], result.sizes_[dim2]);
    return result;
  }
};

template<typename T, template <typename U> class PtrTraits, typename index_t>
class GenericPackedTensorAccessor<T,1,PtrTraits,index_t> : public GenericPackedTensorAccessorBase<T,1,PtrTraits,index_t> {
public:
  // 定义 PtrType 别名，用于指向模板类型 T 的指针
  typedef typename PtrTraits<T>::PtrType PtrType;

  // 构造函数，用于创建 GenericPackedTensorAccessor 对象
  C10_HOST GenericPackedTensorAccessor(
      PtrType data_,                      // 数据指针
      const index_t* sizes_,              // 尺寸数组指针
      const index_t* strides_)            // 步长数组指针
      : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}

  // 当 index_t 不是 int64_t 时，提供一个 int64_t 类型的构造函数
  template <typename source_index_t, class = std::enable_if_t<std::is_same_v<source_index_t, int64_t>>>
  C10_HOST GenericPackedTensorAccessor(
      PtrType data_,                      // 数据指针
      const source_index_t* sizes_,       // 尺寸数组指针
      const source_index_t* strides_)     // 步长数组指针
      : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}

  // 设备函数，用于返回索引 i 处的数据引用
  C10_DEVICE T & operator[](index_t i) {
    return this->data_[this->strides_[0] * i];
  }

  // 设备函数，用于返回索引 i 处的常量数据引用
  C10_DEVICE const T& operator[](index_t i) const {
    return this->data_[this->strides_[0]*i];
  }

  // 在一维情况下，返回一个与原始对象相同的 PackedTensorAccessor 的副本
  // 在通用 N 维情况下也适用，但需要注意一维情况下返回的对象总是与原对象相同
  C10_HOST GenericPackedTensorAccessor<T, 1, PtrTraits, index_t> transpose(
      index_t dim1,                       // 维度1
      index_t dim2) const {               // 维度2
    this->bounds_check_(dim1);           // 检查维度1的有效性
    this->bounds_check_(dim2);           // 检查维度2的有效性
    return GenericPackedTensorAccessor<T, 1, PtrTraits, index_t>(
        this->data_, this->sizes_, this->strides_);  // 返回新的 PackedTensorAccessor 对象
  }
};

// 由于宏函数参数中存在逗号，不能直接放入宏函数参数中，因此定义了 AT_X 宏
#define AT_X GenericPackedTensorAccessor<T, N, PtrTraits, index_t>

// 旧名为 `GenericPackedTensorAccessor` 的模板别名
template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
C10_DEFINE_DEPRECATED_USING(PackedTensorAccessor, AT_X)

// 取消宏定义 AT_X
#undef AT_X

// 使用 int32_t 作为 index_t 类型的 PackedTensorAccessor 模板别名
template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor32 = GenericPackedTensorAccessor<T, N, PtrTraits, int32_t>;

// 使用 int64_t 作为 index_t 类型的 PackedTensorAccessor 模板别名
template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor64 = GenericPackedTensorAccessor<T, N, PtrTraits, int64_t>;

} // namespace at
```