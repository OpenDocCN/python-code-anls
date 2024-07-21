# `.\pytorch\aten\src\ATen\cpu\vec\vec_mask.h`

```py
/**
 * This pragma ensures that this header file is included only once during compilation.
 */
#pragma once

/**
 * Include necessary headers for vectorized operations.
 */
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_n.h>

/**
 * Define the namespace `at::vec` and an inline namespace `CPU_CAPABILITY` within it.
 */
namespace at::vec {
inline namespace CPU_CAPABILITY {

/**
 * The `VecMask` class provides an interface for handling vectorized masks in SIMD operations.
 * It encapsulates a `Vectorized<T, N>` mask for use in masked vectorized operations.
 * It supports various operations on the mask:
 * 1. `from` and `to`: Conversion between vector of boolean values and vectorized mask.
 * 2. `cast`: Casts the mask to a different base type.
 * 3. `all_zero`: Checks if all mask elements are zero.
 * 4. `is_masked`: Checks if a specific element is masked.
 * 5. `loadu`: Loads data from memory using the mask.
 * 6. `all_masked`: Checks if all mask elements are masked.
 *
 * Helper template classes are provided for specialized operations on `VecMask`:
 * 1. `VecMaskLoad`: Loads data from memory using the mask.
 * 2. `VecMaskTo`: Converts the mask to boolean.
 * 3. `VecMaskCast`: Casts the mask to a different base type.
 *
 */
template <typename T, int N>
class VecMask;

/**
 * Template specialization for loading data from memory using the mask.
 * Applies when enabled.
 */
template <
    typename data_t,
    int data_n,
    typename mask_t,
    int mask_n,
    typename Enabled = void>
struct VecMaskLoad {
  /**
   * Applies the mask to load data from memory pointed to by `ptr`.
   * Constructs data and mask vectors, applies the mask, and loads data.
   * Returns a `VectorizedN<data_t, data_n>` object containing loaded data.
   */
  static inline VectorizedN<data_t, data_n> apply(
      const data_t* ptr,
      const VecMask<mask_t, mask_n>& vec_mask) {
    constexpr typename VecMask<mask_t, mask_n>::size_type size =
        VecMask<mask_t, mask_n>::size();
    static_assert(VectorizedN<data_t, data_n>::size() >= size);
    __at_align__ data_t data[size];
    __at_align__ mask_t mask[size];
    auto mask_ = VectorizedN<mask_t, mask_n>(vec_mask);
    mask_.store(mask);
    for (int i = 0; i < size; i++) {
      data[i] = mask[i] ? ptr[i] : static_cast<data_t>(0);
    }
    return VectorizedN<data_t, data_n>::loadu(data, size);
  }
};

/**
 * Template specialization for converting the mask to a boolean.
 * Applies when enabled.
 */
template <
    typename dst_t,
    int dst_n,
    typename src_t,
    int src_n,
    typename Enabled = void>
struct VecMaskTo {
  /**
   * Converts the mask `vec_mask` to a boolean vector.
   * Returns a `VecMask<dst_t, dst_n>` object.
   */
  static inline VecMask<dst_t, dst_n> apply(
      const VecMask<src_t, src_n>& vec_mask) {
    auto zeros = VectorizedN<dst_t, dst_n>(static_cast<dst_t>(0));
    auto ones = VectorizedN<dst_t, dst_n>(static_cast<dst_t>(1));
    return VectorizedN<dst_t, dst_n>::blendv(
        zeros, ones, vec_mask.template cast<dst_t, dst_n>());
  }
};

/**
 * Template specialization for casting the mask to a different base type.
 * Applies when enabled.
 */
template <typename dst_t, int dst_n, typename src_t, int src_n>
struct VecMaskCast {
  /**
   * Casts the mask `vec_mask` to a `VecMask<dst_t, dst_n>`.
   * Returns the casted `VecMask`.
   */
  static inline VecMask<dst_t, dst_n> apply(
      const VecMask<src_t, src_n>& vec_mask) {
    return VecMask<dst_t, dst_n>::from(VectorizedN<src_t, src_n>(vec_mask));
  }
};

/**
 * Template specialization for identity casting of the mask to the same type.
 * Applies when `dst_t` and `src_t` are the same.
 */
template <typename T, int N>
struct VecMaskCast<T, N, T, N> {
  /**
   * Returns the `vec_mask` unchanged when `dst_t` equals `src_t`.
   */
  static inline VecMask<T, N> apply(const VecMask<T, N>& vec_mask) {
    return vec_mask;
  }
};

/**
 * Definition of the `VecMask` class.
 * Template class for handling vectorized masks.
 */
template <typename T, int N>
class VecMask {
 public:
  using size_type = int;
  /**
   * Returns the size of the vectorized mask.
   * The size is determined by `VecMask<T, N>`.
   */
  static constexpr size_type size() {

    /**
     * Returns the size of the vectorized mask.
     * The size is determined by `VecMask<T, N>`.
     */
    return N;
  }
    return VectorizedN<T, N>::size();
  }

 private:
  VectorizedN<T, N> mask_;

 public:
  // 默认构造函数，初始化 mask_ 为 0
  VecMask() : mask_(static_cast<T>(0)) {}

  // 使用 VectorizedN<T, N> 类型的对象初始化 mask_
  VecMask(const VectorizedN<T, N>& mask) : mask_(mask) {}

  // 当 N == 1 时，使用 Vectorized<T> 对象初始化 mask_
  template <int L = N, typename std::enable_if_t<L == 1, int> = 0>
  VecMask(const Vectorized<T>& mask) : mask_(mask) {}

  // 根据 VectorizedN<U, L> 对象创建 VecMask<T, N>
  template <typename U, int L>
  static VecMask<T, N> from(const VectorizedN<U, L>& b_vec) {
    __at_align__ U b_buf[size()];  // 声明一个数组 b_buf，类型为 U，长度为 size()

    // 如果当前 VecMask 的 size() 大于等于 b_vec 的 size()
    if constexpr (size() >= VectorizedN<U, L>::size()) {
      // 将 b_vec 的数据存储到 b_buf 中
      b_vec.store(b_buf);
      // 对于 b_buf 中未被填充的部分，置为 0
      for (int i = VectorizedN<U, L>::size(); i < size(); i++) {
        b_buf[i] = static_cast<U>(0);
      }
    } else {
      // 将 b_vec 的前 size() 个数据存储到 b_buf 中
      b_vec.store(b_buf, size());
    }
    // 从 b_buf 中创建并返回 VecMask<T, N> 对象
    return from(b_buf);
  }

  // 根据单个值 b 创建 VecMask<T, N> 对象
  template <typename U>
  static VecMask<T, N> from(U b) {
    using int_t = int_same_size_t<T>;
    // 根据 b 的值设置 mask，如果 b 为真，则为全 1，否则为全 0
    T mask = b ? c10::bit_cast<T>((int_t)(~(int_t)0)) : (T)0;
    return VectorizedN<T, N>(mask);
  }

  // 根据指针 b 创建 VecMask<T, N> 对象
  template <typename U>
  static VecMask<T, N> from(U* b) {
    using int_t = int_same_size_t<T>;
    __at_align__ T mask[size()];  // 声明一个数组 mask，类型为 T，长度为 size()
#ifndef __msvc_cl__
#pragma unroll
#endif
// 如果不是使用 MSVC 编译器，对循环进行展开优化
for (int i = 0; i < size(); i++) {
  // 将布尔向量 b 转换为整数类型 int_t，并根据其值设置掩码 mask 中对应位置的位
  *(int_t*)(mask + i) = b[i] ? ~(int_t)0 : (int_t)0;
}
// 将设置好的掩码 mask 转换为 VectorizedN<T, N> 类型的对象并返回
return VectorizedN<T, N>(VectorizedN<T, N>::loadu(mask));
}

// 静态函数，使用 c、b、a 中的掩码进行混合操作，并返回混合结果
static VecMask<T, N> blendv(
  const VecMask<T, N>& c,
  const VecMask<T, N>& b,
  const VecMask<T, N>& a) {
  VectorizedN<T, N> result = VectorizedN<T, N>::blendv(
    VectorizedN<T, N>(c),
    VectorizedN<T, N>(b),
    VectorizedN<T, N>(a));
  return result;
}

// 将掩码对象存储到布尔数组 b 中，可选参数 count 指定存储的元素个数，默认为 size()
void store(bool* b, int count = size()) {
  // 计算所需的布尔类型数组 res 的长度 L
  constexpr int L = (VectorizedN<T, N>::size() + Vectorized<bool>::size() - 1)/ Vectorized<bool>::size();
  // 将当前对象转换为布尔类型数组 res，并将其存储到数组 b 中
  auto res = this->to<bool, L>();
  res.store(b, count);
  return;
}

// 模板函数，将当前对象转换为另一种类型的掩码对象，L >= 2 时实现
template <typename U, int L, std::enable_if_t<L >= 2, int> = 0>
inline VectorizedN<U, L> to() const {
  return VecMaskTo<U, L, T, N>::apply(*this);
}

// 模板函数，将当前对象转换为另一种类型的掩码对象，L == 1 时实现
template <typename U, int L, std::enable_if_t<L == 1, int> = 0>
inline Vectorized<U> to() const {
  return VecMaskTo<U, L, T, N>::apply(*this);
}

// 将当前对象转换为另一种类型的掩码对象，使用 VecMaskCast 实现
template <typename U, int L>
inline VecMask<U, L> cast() const {
  return VecMaskCast<U, L, T, N>::apply(*this);
}

// 检查当前掩码对象是否全为零
inline bool all_zero() const {
  __at_align__ T mask[size()];
  mask_.store(mask);
  // 使用 std::all_of 检查 mask 数组中的所有元素是否都等于零
  return std::all_of(
      mask, mask + size(), [](T m) { return m == static_cast<T>(0); });
}

// 检查当前掩码对象是否全为非零
inline bool all_masked() const {
  __at_align__ T mask[size()];
  mask_.store(mask);
  // 使用 std::all_of 检查 mask 数组中的所有元素是否都不等于零
  return std::all_of(
      mask, mask + size(), [](T m) { return m != static_cast<T>(0); });
}

// 检查给定索引 i 处的掩码值是否非零
inline bool is_masked(int i) const {
  __at_align__ T mask[size()];
  mask_.store(mask);
  // 检查 mask 数组中索引为 i 的元素是否不等于零
  return mask[i] != static_cast<T>(0);
}

// 将当前掩码对象转换为 VectorizedN<T, N> 类型的对象并返回
inline operator VectorizedN<T, N>() const {
  return mask_;
}

// 当 N == 1 时，将当前掩码对象转换为 Vectorized<T> 类型的对象并返回
template <int L = N, typename std::enable_if_t<L == 1, int> = 0>
inline operator Vectorized<T>() const {
  return mask_[0];
}

// 返回当前掩码对象中索引为 i 的 Vectorized<T> 类型的掩码值
inline Vectorized<T> operator[](int i) const {
  return mask_[i];
}

// 从指针 ptr 处加载数据到 VectorizedN<U, L> 类型的对象，当 L >= 2 时实现
template <
    typename U,
    int L,
    std::enable_if_t<L >= 2 && VectorizedN<U, L>::size() >= size(), int> = 0>
VectorizedN<U, L> loadu(const U* ptr) const {
  return VecMaskLoad<U, L, T, N>::apply(ptr, *this);
}

// 从指针 ptr 处加载数据到 Vectorized<U> 类型的对象，当 L == 1 时实现
template <
    typename U,
    int L,
    std::enable_if_t<L == 1 && Vectorized<U>::size() >= size(), int> = 0>
Vectorized<U> loadu(const U* ptr) const {
  return VecMaskLoad<U, L, T, N>::apply(ptr, *this);
}
};

// 定义全局的一元操作符宏，对 VecMask<T, N> 类型的对象执行操作
#define VEC_MASK_DEFINE_UNARY_OP_GLOBAL(op)         \
template <typename T, int N>                        \
inline VecMask<T, N> op(const VecMask<T, N>& a) {   \
  return op(VectorizedN<T, N>(a));                  \
}
#define VEC_MASK_DEFINE_BINARY_OP_GLOBAL(op)                                  \
  // 定义一个模板函数，实现全局的二元运算符重载，操作符为 op
  template <                                                                  \
      typename T,                                                             \
      int N,                                                                  \
      typename V,                                                             \
      int M,                                                                  \
      std::enable_if_t<VecMask<T, N>::size() == VecMask<V, M>::size(), int> = \
          0>                                                                  \
  inline VecMask<T, N> op(const VecMask<T, N>& a, const VecMask<V, M>& b) {   \
    // 调用操作符 op，并将参数转换为相应的 VectorizedN 类型
    return op(                                                                \
        VectorizedN<T, N>(a), VectorizedN<T, N>(b.template cast<T, N>()));    \
  }

#define VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(op, EXPR)                  \
  // 定义一个模板函数，实现全局的二元运算符重载，操作符为 op，使用给定的表达式 EXPR
  template <                                                                  \
      typename T,                                                             \
      int N,                                                                  \
      typename V,                                                             \
      int M,                                                                  \
      std::enable_if_t<VecMask<T, N>::size() == VecMask<V, M>::size(), int> = \
          0>                                                                  \
  inline VecMask<T, N> op(const VecMask<T, N>& a, const VecMask<V, M>& b) {   \
    // 返回给定的表达式 EXPR 的结果
    return EXPR;                                                              \
  }

// 定义全局的按位非运算符重载
VEC_MASK_DEFINE_UNARY_OP_GLOBAL(operator~)
// 定义全局的按位与运算符重载
VEC_MASK_DEFINE_BINARY_OP_GLOBAL(operator&)
// 定义全局的按位或运算符重载
VEC_MASK_DEFINE_BINARY_OP_GLOBAL(operator|)
// 定义全局的按位异或运算符重载
VEC_MASK_DEFINE_BINARY_OP_GLOBAL(operator^)
// 定义全局的带有给定表达式的大于运算符重载
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator>, a & ~b)
// 定义全局的带有给定表达式的小于运算符重载
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator<, ~a & b)
// 定义全局的带有给定表达式的等于运算符重载
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator==, ~(a ^ b))
// 定义全局的带有给定表达式的大于等于运算符重载
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator>=, (a == b) | (a > b))
// 定义全局的带有给定表达式的小于等于运算符重载
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator<=, (a == b) | (a < b))
// 定义全局的带有给定表达式的不等于运算符重载
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator!=, (a ^ b))

#undef VEC_MASK_DEFINE_UNARY_OP_GLOBAL
#undef VEC_MASK_DEFINE_BINARY_OP_GLOBAL
#undef VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL

} // namespace CPU_CAPABILITY
} // namespace at::vec
```