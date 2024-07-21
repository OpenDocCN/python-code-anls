# `.\pytorch\aten\src\ATen\native\StridedRandomAccessor.h`

```
// 使用 #pragma once 指令确保头文件只被编译一次
#pragma once

// 命名空间 at::native 包含了本地的实现
namespace at::native {

// ConstStridedRandomAccessor 是一个常量的随机访问迭代器，
// 用于访问一个步进数组。

// 下面的 traits 用于在不同平台上引入 __restrict__ 修饰符。

template <typename T>
struct DefaultPtrTraits {
  using PtrType = T*;  // 默认的指针类型为 T*
};

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

template <typename T>
struct RestrictPtrTraits {
  using PtrType = T* RESTRICT;  // 使用平台相关的 __restrict__ 修饰符定义指针类型
};

template <
  typename T,
  typename index_t = int64_t,
  template <typename U> class PtrTraits = DefaultPtrTraits
>
class ConstStridedRandomAccessor {
public:
  using difference_type = index_t;  // 定义差值类型为 index_t
  using value_type = const T;  // 定义值类型为常量 T
  using pointer = const typename PtrTraits<T>::PtrType;  // 定义指针类型为常量 T 的指针类型
  using reference = const value_type&;  // 定义引用类型为常量值类型的引用
  using iterator_category = std::random_access_iterator_tag;  // 定义迭代器类别为随机访问迭代器

  using PtrType = typename PtrTraits<T>::PtrType;  // 使用 PtrTraits 定义的指针类型
  using index_type = index_t;  // 使用 index_t 定义索引类型

  // 构造函数 {
  C10_HOST_DEVICE
  ConstStridedRandomAccessor(PtrType ptr, index_t stride)
    : ptr{ptr}, stride{stride}
  {}

  C10_HOST_DEVICE
  explicit ConstStridedRandomAccessor(PtrType ptr)
    : ptr{ptr}, stride{static_cast<index_t>(1)}
  {}

  C10_HOST_DEVICE
  ConstStridedRandomAccessor()
    : ptr{nullptr}, stride{static_cast<index_t>(1)}
  {}
  // }

  // 指针类操作 {
  C10_HOST_DEVICE
  reference operator*() const {
    return *ptr;  // 解引用操作符，返回指针所指向的值的引用
  }

  C10_HOST_DEVICE
  const value_type* operator->() const {
    return reinterpret_cast<const value_type*>(ptr);  // 指针成员访问操作符，返回指针转换后的常量值类型指针
  }

  C10_HOST_DEVICE
  reference operator[](index_t idx) const {
    return ptr[idx * stride];  // 下标操作符，返回指定偏移量的值的引用
  }
  // }

  // 前缀/后缀自增/自减 {
  C10_HOST_DEVICE
  ConstStridedRandomAccessor& operator++() {
    ptr += stride;  // 前缀自增操作，使指针向前移动一个步长
    return *this;
  }

  C10_HOST_DEVICE
  ConstStridedRandomAccessor operator++(int) {
    ConstStridedRandomAccessor copy(*this);  // 后缀自增操作，返回先前的副本并移动指针
    ++*this;
    return copy;
  }

  C10_HOST_DEVICE
  ConstStridedRandomAccessor& operator--() {
    ptr -= stride;  // 前缀自减操作，使指针向后移动一个步长
    return *this;
  }

  C10_HOST_DEVICE
  ConstStridedRandomAccessor operator--(int) {
    ConstStridedRandomAccessor copy(*this);  // 后缀自减操作，返回先前的副本并移动指针
    --*this;
    return copy;
  }
  // }

  // 算术操作 {
  C10_HOST_DEVICE
  ConstStridedRandomAccessor& operator+=(index_t offset) {
    ptr += offset * stride;  // 复合赋值加法操作，移动指针以指定的偏移量乘以步长
    return *this;
  }

  C10_HOST_DEVICE
  ConstStridedRandomAccessor operator+(index_t offset) const {
    return ConstStridedRandomAccessor(ptr + offset * stride, stride);  // 加法操作，返回新的迭代器以给定偏移量乘以步长移动的位置
  }

  C10_HOST_DEVICE
  friend ConstStridedRandomAccessor operator+(
    index_t offset,
    const ConstStridedRandomAccessor& accessor
  ) {
    return accessor + offset;  // 友元加法操作，实现 offset + accessor
  }

  C10_HOST_DEVICE
  ConstStridedRandomAccessor& operator-=(index_t offset) {
    ptr -= offset * stride;  // 复合赋值减法操作，向后移动指针以指定的偏移量乘以步长
    return *this;
  }

  C10_HOST_DEVICE
  ConstStridedRandomAccessor operator-(index_t offset) const {
    // 减法操作，返回新的迭代器以给定偏移量乘以步长向后移动的位置

    return ConstStridedRandomAccessor(ptr - offset * stride, stride);
  }
  // }
};
    // 返回一个 ConstStridedRandomAccessor 对象，其指针为 ptr - offset * stride，步长为 stride
    return ConstStridedRandomAccessor(ptr - offset * stride, stride);
    }
    
    // 注意：当 `this` 和 `other` 表示相同的序列时，该运算符是有定义的，即当：
    // 1. this.stride == other.stride，
    // 2. |other - this| / this.stride 是一个整数。
    C10_HOST_DEVICE
    difference_type operator-(const ConstStridedRandomAccessor& other) const {
        // 返回两个对象指针之间的差除以步长的结果
        return (ptr - other.ptr) / stride;
    }
    
    // Comparison operators {
    C10_HOST_DEVICE
    bool operator==(const ConstStridedRandomAccessor& other) const {
        // 比较两个对象的指针和步长是否相等
        return (ptr == other.ptr) && (stride == other.stride);
    }
    
    C10_HOST_DEVICE
    bool operator!=(const ConstStridedRandomAccessor& other) const {
        // 使用相等运算符确定两个对象是否不相等
        return !(*this == other);
    }
    
    C10_HOST_DEVICE
    bool operator<(const ConstStridedRandomAccessor& other) const {
        // 使用指针比较确定当前对象是否小于另一个对象
        return ptr < other.ptr;
    }
    
    C10_HOST_DEVICE
    bool operator<=(const ConstStridedRandomAccessor& other) const {
        // 使用小于运算符和相等运算符确定当前对象是否小于或等于另一个对象
        return (*this < other) || (*this == other);
    }
    
    C10_HOST_DEVICE
    bool operator>(const ConstStridedRandomAccessor& other) const {
        // 使用小于等于运算符确定当前对象是否大于另一个对象
        return !(*this <= other);
    }
    
    C10_HOST_DEVICE
    bool operator>=(const ConstStridedRandomAccessor& other) const {
        // 使用小于运算符确定当前对象是否大于或等于另一个对象
        return !(*this < other);
    }
    // }
protected:
  PtrType ptr;  // 指向数据的指针，类型为PtrType
  index_t stride;  // 步长，用于在数据中跳跃访问元素的间隔
};

template <
  typename T,
  typename index_t = int64_t,
  template <typename U> class PtrTraits = DefaultPtrTraits
>
class StridedRandomAccessor
  : public ConstStridedRandomAccessor<T, index_t, PtrTraits> {
public:
  using difference_type = index_t;  // 差异类型，用于表示两个迭代器之间的距离
  using value_type = T;  // 值类型，表示迭代器指向的元素类型
  using pointer = typename PtrTraits<T>::PtrType;  // 指针类型，指向T类型数据的指针
  using reference = value_type&;  // 引用类型，表示迭代器指向的元素的引用类型

  using BaseType = ConstStridedRandomAccessor<T, index_t, PtrTraits>;
  using PtrType = typename PtrTraits<T>::PtrType;  // 指向T类型数据的指针类型

  // Constructors {
  C10_HOST_DEVICE
  StridedRandomAccessor(PtrType ptr, index_t stride)
    : BaseType(ptr, stride)
  {}  // 使用指针和步长构造StridedRandomAccessor对象

  C10_HOST_DEVICE
  explicit StridedRandomAccessor(PtrType ptr)
    : BaseType(ptr)
  {}  // 显式指定指针构造StridedRandomAccessor对象

  C10_HOST_DEVICE
  StridedRandomAccessor()
    : BaseType()
  {}  // 默认构造函数，使用基类默认构造函数构造对象
  // }

  // Pointer-like operations {
  C10_HOST_DEVICE
  reference operator*() const {
    return *this->ptr;  // 返回指针指向的元素的引用
  }

  C10_HOST_DEVICE
  value_type* operator->() const {
    return reinterpret_cast<value_type*>(this->ptr);  // 返回指向ptr指向数据类型的指针
  }

  C10_HOST_DEVICE
  reference operator[](index_t idx) const {
    return this->ptr[idx * this->stride];  // 返回跳跃访问后的元素的引用
  }
  // }

  // Prefix/postfix increment/decrement {
  C10_HOST_DEVICE
  StridedRandomAccessor& operator++() {
    this->ptr += this->stride;  // 前缀递增操作符，指针前进stride步长
    return *this;
  }

  C10_HOST_DEVICE
  StridedRandomAccessor operator++(int) {
    StridedRandomAccessor copy(*this);  // 后缀递增操作符，创建副本对象
    ++*this;
    return copy;
  }

  C10_HOST_DEVICE
  StridedRandomAccessor& operator--() {
    this->ptr -= this->stride;  // 前缀递减操作符，指针后退stride步长
    return *this;
  }

  C10_HOST_DEVICE
  StridedRandomAccessor operator--(int) {
    StridedRandomAccessor copy(*this);  // 后缀递减操作符，创建副本对象
    --*this;
    return copy;
  }
  // }

  // Arithmetic operations {
  C10_HOST_DEVICE
  StridedRandomAccessor& operator+=(index_t offset) {
    this->ptr += offset * this->stride;  // 加法赋值操作符，移动指针offset * stride的距离
    return *this;
  }

  C10_HOST_DEVICE
  StridedRandomAccessor operator+(index_t offset) const {
    return StridedRandomAccessor(this->ptr + offset * this->stride, this->stride);  // 加法操作符，返回新的对象
  }

  C10_HOST_DEVICE
  friend StridedRandomAccessor operator+(
    index_t offset,
    const StridedRandomAccessor& accessor
  ) {
    return accessor + offset;  // 友元加法操作符，将偏移量加到迭代器上
  }

  C10_HOST_DEVICE
  StridedRandomAccessor& operator-=(index_t offset) {
    this->ptr -= offset * this->stride;  // 减法赋值操作符，移动指针-offset * stride的距离
    return *this;
  }

  C10_HOST_DEVICE
  StridedRandomAccessor operator-(index_t offset) const {
    return StridedRandomAccessor(this->ptr - offset * this->stride, this->stride);  // 减法操作符，返回新的对象
  }

  // Note that here we call BaseType::operator- version
  C10_HOST_DEVICE
  difference_type operator-(const BaseType& other) const {
    return (static_cast<const BaseType&>(*this) - other);  // 计算迭代器之间的距离
  }
  // }
};

} // namespace at::native
```