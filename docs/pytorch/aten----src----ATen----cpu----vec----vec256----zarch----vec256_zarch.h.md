# `.\pytorch\aten\src\ATen\cpu\vec\vec256\zarch\vec256_zarch.h`

```py
// 引入数学函数、字符串处理、数值极限、类型特性以及实用工具
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

// 根据编译器类型选择 Sleef 库和向量指令库的包含文件
#if defined(__clang__)
#include <sleef.h>
#elif defined(__GNUC__) || defined(__GNUG__)
#include <sleef.h>
#include <vecintrin.h>
#endif

// 引入 ATen 库中的向量化指令和基础向量类
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>

// 引入 C10 库中的复数处理工具
#include <c10/util/complex.h>

namespace at {
namespace vec {

// CPU_CAPABILITY 命名空间的内联命名空间，见注释 [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

// 判断给定类型 T 是否支持 Z 系列指令集的实现
template <typename T>
constexpr bool is_zarch_implemented() {
  return (
      std::is_same<T, float>::value || std::is_same<T, double>::value ||
      std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
      std::is_same<T, uint16_t>::value || std::is_same<T, int16_t>::value ||
      std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value);
}

// 判断给定类型 T 是否支持 Z 系列指令集的量化实现
template <typename T>
constexpr bool is_zarch_implemented_quant() {
  return (
      std::is_same<T, c10::qint32>::value ||
      std::is_same<T, c10::qint8>::value ||
      std::is_same<T, c10::quint8>::value);
}

// 判断给定类型 T 是否支持 Z 系列指令集的复数实现
template <typename T>
constexpr bool is_zarch_implemented_complex() {
  return std::is_same<T, c10::complex<float>>::value ||
      std::is_same<T, c10::complex<double>>::value;
}

// 定义常量 offset0 和 offset16
constexpr int offset0 = 0;
constexpr int offset16 = 16;

// 根据模板参数 N 定义 VecBinaryType 结构体，指定每个向量元素的类型为 uintmax_t
template <int N>
struct VecBinaryType {
  using type __attribute__((vector_size(16))) = uintmax_t;
};

// 对 N = 8 的特化，使用 unsigned long long 作为向量元素类型
template <>
struct VecBinaryType<8> {
  using type = __attribute__((vector_size(16))) unsigned long long;
};

// 对 N = 4 的特化，使用 unsigned int 作为向量元素类型
template <>
struct VecBinaryType<4> {
  using type = __attribute__((vector_size(16))) unsigned int;
};

// 对 N = 2 的特化，使用 unsigned short 作为向量元素类型
template <>
struct VecBinaryType<2> {
  using type = __attribute__((vector_size(16))) unsigned short;
};

// 对 N = 1 的特化，使用 unsigned char 作为向量元素类型
template <>
struct VecBinaryType<1> {
  using type = __attribute__((vector_size(16))) unsigned char;
};

// 根据模板参数 T 定义 VecInnerType 结构体
template <typename T>
struct VecInnerType {
  using Type __attribute__((vector_size(16))) = T;
  using BinaryType = typename VecBinaryType<sizeof(T)>::type;
  using ElementType = T;
  static constexpr int size = 16 / sizeof(T);
};

// 对 int64_t 类型的特化，使用 signed long long 作为向量内部类型
template <>
struct VecInnerType<int64_t> {
  using Type = __attribute__((vector_size(16))) signed long long;
  using ElementType = signed long long;
  using BinaryType = typename VecBinaryType<sizeof(signed long long)>::type;
  static constexpr int size = 16 / sizeof(signed long long);
};

// 定义 ZSimdVect<T> 类型别名，表示类型 T 对应的 Z 系列 SIMD 向量类型
template <typename T>
using ZSimdVect = typename VecInnerType<T>::Type;

// 定义 ZSimdVectBinary<T> 类型别名，表示类型 T 对应的 Z 系列 SIMD 二进制向量类型
template <typename T>
using ZSimdVectBinary = typename VecInnerType<T>::BinaryType;

// 定义 ZSimdVectElement<T> 类型别名，表示类型 T 对应的 Z 系列 SIMD 向量元素类型
template <typename T>
using ZSimdVectElement = typename VecInnerType<T>::ElementType;

// 定义 blendChoiceInner 函数，根据给定的 mask 值进行内部向量的混合选择
constexpr int blendChoiceInner(
    const uint64_t mask,
    const uint64_t half1 = 0xF,
    const uint64_t half2 = 0xF0) {
  uint64_t none = 0;
  uint64_t both = half1 | half2;
  // 将 mask 值限制在 0 和 both 之间
  auto res_mask = mask & both;
  // 根据 res_mask 的值返回不同的选择结果
  if (res_mask == none)
    return 0;
  else if (res_mask == both)
    return 1;
}
    return 1;
  // 如果 res_mask 等于 0，返回结果 1
  else if (res_mask == half1)
    return 2;
  // 如果 res_mask 等于 half1，返回结果 2
  else if (res_mask == half2)
    return 3;
  // 如果 res_mask 等于 half2，返回结果 3
  else if (res_mask > 0 && res_mask < half1)
    return 4;
  // 如果 res_mask 大于 0 并且小于 half1，返回结果 4
  else if ((res_mask & half2) == half2)
    return 5;
  // 如果 res_mask 按位与 half2 等于 half2，返回结果 5
  else if ((res_mask & half1) == 0 && res_mask > half1)
    return 6;
  // 如果 res_mask 按位与 half1 等于 0 且大于 half1，返回结果 6
  else if ((res_mask & half1) == half1 && res_mask > half1)
    return 7;
  // 如果 res_mask 按位与 half1 等于 half1 且大于 half1，返回结果 7
  // 默认情况，返回结果 8
  return 8;
template <>
// 当模板参数 Z 为 8 时，根据给定的 mask 值执行 blendChoiceInner 函数，使用掩码 0x3 和 0xC 进行位混合
constexpr int blendChoice<8>(const uint64_t mask) {
  // clamp it 0 and 0xF
  return blendChoiceInner(mask, 0x3, 0xC);
}

template <int N>
// 返回一个类型为 VecBinaryType<N>::type 的值，其实际内容为空的结构体或类
constexpr auto GetMask1(const uint64_t mask) {
  return typename VecBinaryType<N>::type{};
}

template <int N>
// 返回一个类型为 VecBinaryType<N>::type 的值，其实际内容为空的结构体或类
constexpr auto GetMask2(const uint64_t mask) {
  return typename VecBinaryType<N>::type{};
}

template <>
// 当模板参数 N 为 1 时，基于给定的 mask 值生成一个长度为 16 的 VecBinaryType<1>::type 结构，每个元素为 mask 中对应位的 t 值
constexpr auto GetMask1<1>(const uint64_t mask) {
  constexpr uint8_t t = (int)0xFF;
  uint8_t g0 = (mask & 1) * t;
  uint8_t g1 = ((mask & 2) >> 1) * t;
  uint8_t g2 = ((mask & 4) >> 2) * t;
  uint8_t g3 = ((mask & 8) >> 3) * t;
  uint8_t g4 = ((mask & 16) >> 4) * t;
  uint8_t g5 = ((mask & 32) >> 5) * t;
  uint8_t g6 = ((mask & 64) >> 6) * t;
  uint8_t g7 = ((mask & 128) >> 7) * t;
  uint8_t g8 = ((mask & 256) >> 8) * t;
  uint8_t g9 = ((mask & 512) >> 9) * t;
  uint8_t g10 = ((mask & 1024) >> 10) * t;
  uint8_t g11 = ((mask & 2048) >> 11) * t;
  uint8_t g12 = ((mask & 4096) >> 12) * t;
  uint8_t g13 = ((mask & 8192) >> 13) * t;
  uint8_t g14 = ((mask & 16384) >> 14) * t;
  uint8_t g15 = ((mask & 32768) >> 15) * t;
  // 返回一个结构体，包含由 mask 的每个位计算得到的值
  return (typename VecBinaryType<1>::type){
      g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15};
}

template <>
// 当模板参数 N 为 1 时，基于给定的 mask 值生成一个长度为 16 的 VecBinaryType<1>::type 结构，每个元素为 mask 中对应位的 t 值
constexpr auto GetMask2<1>(const uint64_t mask) {
  // 将 mask 的低 32 位抽取出来，再调用 GetMask1<1> 生成 VecBinaryType<1>::type 结构
  uint64_t mask2 = (mask & 0xFFFFFFFF) >> 16;
  return GetMask1<1>(mask2);
}

template <>
// 当模板参数 N 为 2 时，基于给定的 mask 值生成一个长度为 8 的 VecBinaryType<2>::type 结构，每个元素为 mask 中对应位的 t 值
constexpr auto GetMask1<2>(const uint64_t mask) {
  constexpr uint16_t t = (int)0xFFFF;
  uint16_t g0 = (mask & 1) * t;
  uint16_t g1 = ((mask & 2) >> 1) * t;
  uint16_t g2 = ((mask & 4) >> 2) * t;
  uint16_t g3 = ((mask & 8) >> 3) * t;
  uint16_t g4 = ((mask & 16) >> 4) * t;
  uint16_t g5 = ((mask & 32) >> 5) * t;
  uint16_t g6 = ((mask & 64) >> 6) * t;
  uint16_t g7 = ((mask & 128) >> 7) * t;
  // 返回一个结构体，包含由 mask 的每个位计算得到的值
  return (typename VecBinaryType<2>::type){g0, g1, g2, g3, g4, g5, g6, g7};
}

template <>
// 当模板参数 N 为 2 时，基于给定的 mask 值生成一个长度为 8 的 VecBinaryType<2>::type 结构，每个元素为 mask 中对应位的 t 值
constexpr auto GetMask2<2>(const uint64_t mask) {
  // 将 mask 的低 16 位抽取出来，再调用 GetMask1<2> 生成 VecBinaryType<2>::type 结构
  uint64_t mask2 = (mask & 0xFFFF) >> 8;
  return GetMask1<2>(mask2);
}

template <>
// 当模板参数 N 为 4 时，基于给定的 mask 值生成一个长度为 4 的 VecBinaryType<4>::type 结构，每个元素为 mask 中对应位的 0xffffffff 或 0 值
constexpr auto GetMask1<4>(const uint64_t mask) {
  uint32_t g0 = (mask & 1) * 0xffffffff;
  uint32_t g1 = ((mask & 2) >> 1) * 0xffffffff;
  uint32_t g2 = ((mask & 4) >> 2) * 0xffffffff;
  uint32_t g3 = ((mask & 8) >> 3) * 0xffffffff;
  // 返回一个结构体，包含由 mask 的每个位计算得到的值
  return (typename VecBinaryType<4>::type){g0, g1, g2, g3};
}

template <>
// 当模板参数 N 为 4 时，基于给定的 mask 值生成一个长度为 4 的 VecBinaryType<4>::type 结构，每个元素为 mask 中对应位的 0xffffffff 或 0 值
constexpr auto GetMask2<4>(const uint64_t mask) {
  // 将 mask 的低 8 位抽取出来，再调用 GetMask1<4> 生成 VecBinaryType<4>::type 结构
  uint64_t mask2 = (mask & 0xFF) >> 4;
  return GetMask1<4>(mask2);
}
constexpr auto GetMask1<8>(const uint64_t mask) {
  // 计算第一个 64 位掩码位的处理结果
  uint64_t g0 = (mask & 1) * 0xffffffffffffffff;
  // 计算第二个 64 位掩码位的处理结果
  uint64_t g1 = ((mask & 2) >> 1) * 0xffffffffffffffff;
  // 返回结果作为 VecBinaryType<8>::type 的对象
  return (typename VecBinaryType<8>::type){g0, g1};
}

template <>
constexpr auto GetMask2<8>(const uint64_t mask) {
  // 提取 mask 中的低 4 位并右移 2 位
  uint64_t mask2 = (mask & 0xF) >> 2;
  // 调用 GetMask1<8>() 处理 mask2，返回结果
  return GetMask1<8>(mask2);
}

template <int Z>
constexpr int maskForComplex(uint32_t mask) {
  // 默认情况下返回 0
  return 0;
}

template <>
constexpr int maskForComplex<8>(uint32_t mask) {
  // 仅保留 mask 的低 4 位
  mask = mask & 0xF;
  // 初始化复杂掩码为 0
  int complex_mask = 0;
  // 根据 mask 的每个位设置相应的复杂掩码位
  if (mask & 1)
    complex_mask |= 3;
  if (mask & 2)
    complex_mask |= (3 << 2);
  if (mask & 4)
    complex_mask |= (3 << 4);
  if (mask & 8)
    complex_mask |= (3 << 6);
  // 返回构建好的复杂掩码
  return complex_mask;
}

template <>
constexpr int maskForComplex<16>(uint32_t mask) {
  // 仅保留 mask 的低 2 位
  mask = mask & 0x3;
  // 初始化复杂掩码为 0
  int complex_mask = 0;
  // 根据 mask 的每个位设置相应的复杂掩码位
  if (mask & 1)
    complex_mask |= 3;
  if (mask & 2)
    complex_mask |= (3 << 2);
  // 返回构建好的复杂掩码
  return complex_mask;
}

template <typename T = c10::complex<float>>
constexpr int blend_choice() {
  // 默认情况下返回 0xAA
  return 0xAA;
}

template <>
constexpr int blend_choice<c10::complex<double>>() {
  // 当 T 为 c10::complex<double> 时返回 0x0A
  return 0x0A;
}

constexpr int64_t allbitset(int16_t x) {
  // 构造一个所有位都被设置为 1 的掩码并返回
  int64_t onex = 1;
  return (onex << x) - onex;
}

namespace { /* unnamed namespace */

ZSimdVect<float> vec_mergee(ZSimdVect<float> x, ZSimdVect<float> y) {
  // 静态定义合并掩码 mergee_mask，对 x 和 y 进行合并操作并返回结果
  constexpr ZSimdVectBinary<uint8_t> mergee_mask{
      0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27};
  return vec_perm(x, y, mergee_mask);
}

ZSimdVect<double> vec_mergee(ZSimdVect<double> x, ZSimdVect<double> y) {
  // 对双精度向量 x 和 y 执行水平合并操作并返回结果
  return vec_mergeh(x, y);
}

ZSimdVect<float> vec_mergeo(ZSimdVect<float> x, ZSimdVect<float> y) {
  // 静态定义合并掩码 mergeo_mask，对 x 和 y 进行合并操作并返回结果
  constexpr ZSimdVectBinary<uint8_t> mergeo_mask{
      4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31};
  return vec_perm(x, y, mergeo_mask);
}

ZSimdVect<double> vec_mergeo(ZSimdVect<double> x, ZSimdVect<double> y) {
  // 对双精度向量 x 和 y 执行垂直合并操作并返回结果
  return vec_mergel(x, y);
}

} /* unnamed namespace */

//
template <typename T>
constexpr auto GetBpermZeroMask() {
  // 返回静态定义的 T 类型的零掩码
  return ZSimdVectBinary<uint8_t>{
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      96,
      64,
      32,
      0};
}

template <>
constexpr auto GetBpermZeroMask<double>() {
  // 返回双精度类型的零掩码
  return ZSimdVectBinary<uint8_t>{
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      64,
      0};
}

constexpr auto GetSwapMaskFloat() {
  // 返回静态定义的浮点数交换掩码
  return ZSimdVectBinary<uint8_t>{
      4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11};
}

template <typename T>
// 定义模板类 Vectorized，用于支持 Z 架构已实现的类型 T
template <typename T, std::enable_if_t<is_zarch_implemented<T>()>> {
public:
    // 定义类型别名
    using value_type = T;
    using vtype = ZSimdVect<T>;
    using vmaskType = ZSimdVectBinary<T>;
    using size_type = int;
    // 由于 gcc 对于 int64_t 的不一致处理，这里使用 ElementType 而不是 value_type
    using ElementType = ZSimdVectElement<T>;
    using vinner_data = std::pair<vtype, vtype>;

private:
    // 私有成员变量，存储两个 ZSimdVect<T> 类型的对象
    vtype _vec0;
    vtype _vec1;

public:
    // 静态成员函数，返回 Vectorized 对象的大小
    static constexpr size_type size() {
        return VECTOR_WIDTH / sizeof(ElementType);
    }
    // 默认构造函数
    Vectorized() {}

    // 构造函数，使用单个 vtype 类型参数初始化 _vec0 和 _vec1
    C10_ALWAYS_INLINE Vectorized(vtype v) : _vec0{v}, _vec1{v} {}
    // 构造函数，使用 vinner_data 对象初始化 _vec0 和 _vec1
    C10_ALWAYS_INLINE Vectorized(const vinner_data &v) : _vec0{v.first}, _vec1{v.second} {}
    // 构造函数，使用两个 vtype 类型参数分别初始化 _vec0 和 _vec1
    C10_ALWAYS_INLINE Vectorized(vtype v1, vtype v2) : _vec0{v1}, _vec1{v2} {}
    // 构造函数，使用标量 s 初始化 _vec0 和 _vec1
    C10_ALWAYS_INLINE Vectorized(T s)
        : _vec0{vec_splats((ElementType)s)}, _vec1{vec_splats((ElementType)s)} {}

    // 模板结构体 LoaduHelper，定义 loadu 方法用于从指针加载数据到 Vectorized<T> 对象
    template <typename U, typename DUMMY = void>
    struct LoaduHelper {
        static Vectorized<T> C10_ALWAYS_INLINE
        loadu(const U* ptr, int count = size()) {
            // 创建临时数组 tmp_values，从 ptr 复制数据到 tmp_values
            __at_align__ ElementType tmp_values[size()] = {};
            std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(ElementType));

            // 返回使用 tmp_values 初始化的 Vectorized<T> 对象
            return {
                vec_xl(offset0, &(tmp_values[0])),
                vec_xl(offset16, &(tmp_values[0]))};
        }
    };

    // 特化的 LoaduHelper 结构体，用于 ElementType 类型的指针加载数据
    template <typename DUMMY>
    struct LoaduHelper<ElementType, DUMMY> {
        static Vectorized<T> C10_ALWAYS_INLINE
        loadu(const ElementType* ptr, int count = size()) {
            if (count == size()) {
                // 如果 count 等于 size()，直接从 ptr 加载数据到 Vectorized<T> 对象
                return {
                    vec_xl(offset0, ptr),
                    vec_xl(offset16, ptr)};
            }

            // 创建临时数组 tmp_values，从 ptr 复制数据到 tmp_values
            __at_align__ ElementType tmp_values[size()] = {};
            std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(ElementType));

            // 返回使用 tmp_values 初始化的 Vectorized<T> 对象
            return {
                vec_xl(offset0, &(tmp_values[0])),
                vec_xl(offset16, &(tmp_values[0]))};
        }
    };

    // 静态成员函数 loadu，用于从任意类型 U 的指针加载数据到 Vectorized<T> 对象
    template <typename U>
    static Vectorized<T> C10_ALWAYS_INLINE
    loadu(const U* ptr, int count = size()) {
        return LoaduHelper<U>::loadu(ptr, count);
    }

    // 静态成员函数 loadu_one_fourth，用于从 U 类型的指针加载前 8 字节数据到 Vectorized<T> 对象
    template <typename U>
    static Vectorized<T> C10_ALWAYS_INLINE
    loadu_one_fourth(const U* ptr) {
        // 加载指定长度的数据，这里用于 uint8_t 类型的数据加载
        return loadu(ptr, 8 / sizeof(ElementType));
    }

    // 模板结构体 StoreHelper，定义 store 方法用于将 Vectorized<T> 对象存储到 U 类型的指针
    template <typename U, typename DUMMY = void>
    struct StoreHelper {
        static void C10_ALWAYS_INLINE store(const Vectorized<T> &vec, U* ptr, int count = size()) {
            if (count > 0) {
                // 创建临时数组 tmp_values，将 vec._vec0 和 vec._vec1 的数据存储到 tmp_values
                __at_align__ ElementType tmp_values[size()];
                vec_xst(vec._vec0, offset0, &(tmp_values[0]));
                vec_xst(vec._vec1, offset16, &(tmp_values[0]));
                // 将 tmp_values 的数据复制到 ptr
                std::memcpy(
                    ptr, tmp_values, std::min(count, size()) * sizeof(ElementType));
            }
        }
    };

    // 特化的 StoreHelper 结构体，用于 ElementType 类型的指针存储数据
    template <typename DUMMY>
    struct StoreHelper<ElementType, DUMMY> {
        // store 方法的特化实现
        static void C10_ALWAYS_INLINE store(const Vectorized<T> &vec, ElementType* ptr, int count = size()) {
            // 如果 count 大于 0，将 vec._vec0 和 vec._vec1 的数据直接存储到 ptr
            if (count > 0) {
                vec_xst(vec._vec0, offset0, ptr);
                vec_xst(vec._vec1, offset16, ptr);
            }
        }
    };
};
    static void C10_ALWAYS_INLINE store(const Vectorized<T> &vec, ElementType* ptr, int count = size()) {
      // 如果传入的 count 等于 size()，表示要存储全部元素
      if (count == size()) {
        // 将 vec._vec0 中的数据存储到 ptr 中的偏移量为 offset0 的位置
        vec_xst(vec._vec0, offset0, ptr);
        // 将 vec._vec1 中的数据存储到 ptr 中的偏移量为 offset16 的位置
        vec_xst(vec._vec1, offset16, ptr);
      } else if (count > 0) {
        // 否则，如果 count 大于 0，则部分存储
        __at_align__ ElementType tmp_values[size()]; // 创建临时数组 tmp_values，大小为 size()
        // 将 vec._vec0 中的数据存储到 tmp_values 数组的起始位置
        vec_xst(vec._vec0, offset0, &(tmp_values[0]));
        // 将 vec._vec1 中的数据存储到 tmp_values 数组的偏移量为 offset16 的位置
        vec_xst(vec._vec1, offset16, &(tmp_values[0]));
        // 将 tmp_values 数组的部分内容拷贝到 ptr 中，拷贝长度为 std::min(count, size()) * sizeof(ElementType)
        std::memcpy(
            ptr, tmp_values, std::min(count, size()) * sizeof(ElementType));
      }
    }
  };

  template <typename U>
  void C10_ALWAYS_INLINE store(U* ptr, int count = size()) const {
    // 调用 StoreHelper 模板类的 store 方法，将当前对象 (*this) 存储到 ptr 中，count 为存储元素个数
    return StoreHelper<U>::store(*this, ptr, count);
  }

  C10_ALWAYS_INLINE const vtype& vec0() const {
    // 返回当前对象的 _vec0 成员
    return _vec0;
  }

  C10_ALWAYS_INLINE const vtype& vec1() const {
    // 返回当前对象的 _vec1 成员
    return _vec1;
  }

  C10_ALWAYS_INLINE vinner_data data() const {
    // 返回由当前对象的 _vec0 和 _vec1 构成的 vinner_data 对象
    return std::make_pair<>(_vec0, _vec1);
  }

  C10_ALWAYS_INLINE operator vinner_data() const {
    // 将当前对象转换为 vinner_data 类型，即调用 data() 方法
    return data();
  }

  C10_ALWAYS_INLINE const vmaskType vecb0() const {
    // 将当前对象的 _vec0 成员强制转换为 vmaskType 类型后返回
    return (vmaskType)_vec0;
  }
  C10_ALWAYS_INLINE const vmaskType vecb1() const {
    // 将当前对象的 _vec1 成员强制转换为 vmaskType 类型后返回
    return (vmaskType)_vec1;
  }

  static Vectorized<T> C10_ALWAYS_INLINE blendv(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      const Vectorized<T>& mask) {
    template <typename U = T, std::enable_if_t<(sizeof(U) == 8), int> = 0>
    C10_ALWAYS_INLINE Vectorized(T s1, T s2, T s3, T s4)
        : _vec0{s1, s2}, _vec1{s3, s4} {}
    
    
    
    // 当模板参数 U 是 8 字节大小的类型时，构造函数初始化 Vectorized 对象
    template <typename U = T, std::enable_if_t<(sizeof(U) == 8), int> = 0>
    C10_ALWAYS_INLINE Vectorized(T s1, T s2, T s3, T s4)
        : _vec0{s1, s2}, _vec1{s3, s4} {}
    
    
    
    template <typename U = T, std::enable_if_t<(sizeof(U) == 4), int> = 0>
    C10_ALWAYS_INLINE Vectorized(T s1, T s2, T s3, T s4, T s5, T s6, T s7, T s8)
        : _vec0{s1, s2, s3, s4}, _vec1{s5, s6, s7, s8} {}
    
    
    
    // 当模板参数 U 是 4 字节大小的类型时，构造函数初始化 Vectorized 对象
    template <typename U = T, std::enable_if_t<(sizeof(U) == 4), int> = 0>
    C10_ALWAYS_INLINE Vectorized(T s1, T s2, T s3, T s4, T s5, T s6, T s7, T s8)
        : _vec0{s1, s2, s3, s4}, _vec1{s5, s6, s7, s8} {}
    
    
    
    template <typename U = T, std::enable_if_t<(sizeof(U) == 2), int> = 0>
    C10_ALWAYS_INLINE Vectorized(
        T s1,
        T s2,
        T s3,
        T s4,
        T s5,
        T s6,
        T s7,
        T s8,
        T s9,
        T s10,
        T s11,
        T s12,
        T s13,
        T s14,
        T s15,
        T s16)
        : _vec0{s1, s2, s3, s4, s5, s6, s7, s8},
          _vec1{s9, s10, s11, s12, s13, s14, s15, s16} {}
    
    
    
    // 当模板参数 U 是 2 字节大小的类型时，构造函数初始化 Vectorized 对象
    template <typename U = T, std::enable_if_t<(sizeof(U) == 2), int> = 0>
    C10_ALWAYS_INLINE Vectorized(
        T s1,
        T s2,
        T s3,
        T s4,
        T s5,
        T s6,
        T s7,
        T s8,
        T s9,
        T s10,
        T s11,
        T s12,
        T s13,
        T s14,
        T s15,
        T s16)
        : _vec0{s1, s2, s3, s4, s5, s6, s7, s8},
          _vec1{s9, s10, s11, s12, s13, s14, s15, s16} {}
    
    
    
    template <typename U = T, std::enable_if_t<(sizeof(U) == 1), int> = 0>
    C10_ALWAYS_INLINE Vectorized(
        T s1,
        T s2,
        T s3,
        T s4,
        T s5,
        T s6,
        T s7,
        T s8,
        T s9,
        T s10,
        T s11,
        T s12,
        T s13,
        T s14,
        T s15,
        T s16,
        T s17,
        T s18,
        T s19,
        T s20,
        T s21,
        T s22,
        T s23,
        T s24,
        T s25,
        T s26,
        T s27,
        T s28,
        T s29,
        T s30,
        T s31,
        T s32)
        : _vec0{s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16},
          _vec1{
              s17,
              s18,
              s19,
              s20,
              s21,
              s22,
              s23,
              s24,
              s25,
              s26,
              s27,
              s28,
              s29,
              s30,
              s31,
              s32} {}
    
    
    
    // 当模板参数 U 是 1 字节大小的类型时，构造函数初始化 Vectorized 对象
    template <typename U = T, std::enable_if_t<(sizeof(U) == 1), int> = 0>
    C10_ALWAYS_INLINE Vectorized(
        T s1,
        T s2,
        T s3,
        T s4,
        T s5,
        T s6,
        T s7,
        T s8,
        T s9,
        T s10,
        T s11,
        T s12,
        T s13,
        T s14,
        T s15,
        T s16,
        T s17,
        T s18,
        T s19,
        T s20,
        T s21,
        T s22,
        T s23,
        T s24,
        T s25,
        T s26,
        T s27,
        T s28,
        T s29,
        T s30,
        T s31,
        T s32)
        : _vec0{s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16},
          _vec1{
              s17,
              s18,
              s19,
              s20,
              s21,
              s22,
              s23,
              s24,
              s25,
              s26,
              s27,
              s28,
              s29,
              s30,
              s31,
              s32} {}
    
    
    
    template <typename step_t, typename U = T>
    static std::enable_if_t<sizeof(U) == 8, Vectorized<T>> arange(
        T base = 0,
        step_t step = static_cast<step_t>(1)) {
      return Vectorized<T>(base, base + step, base + 2 * step, base + 3 * step);
    }
    
    
    
    // 当模板参数 U 是 8 字节大小的类型时，返回一个 Vectorized 对象，使用给定的步长和基础值进行初始化
    template <typename step_t, typename U = T>
    static std::enable_if_t<sizeof(U) == 8, Vectorized<T>> arange(
        T base = 0,
        step_t step = static_cast<step_t>(1)) {
      return Vectorized<T>(base, base + step, base + 2 * step, base + 3 * step);
    }
    
    
    
    template <typename step_t, typename U = T>
    static std::enable_if_t<sizeof(U) == 4, Vectorized<T>> arange(
        T base = 0,
        step_t step = static_cast<step_t>(1)) {
      return Vectorized<T>(
          base,
          base + step,
          base + 2 * step,
          base + 3 * step,
          base + 4 * step,
          base + 5 * step,
          base + 6 * step,
          base + 7 * step);
    }
    
    
    
    // 当模板参数 U 是 4 字节大小的类型时，返回一个 Vectorized 对象，使用给定的步长和基础值进行初始化
    template <typename step_t, typename U = T>
    static std::enable_if_t<sizeof(U) == 4, Vectorized<T>> arange(
        T base = 0,
        step_t step = static_cast<step_t>(1)) {
      return Vectorized<T>(
          base,
          base + step,
          base + 2 * step,
          base + 3 * step,
          base + 4 * step,
          base + 5 * step,
          base + 6 * step,
          base + 7 * step);
    }
    
    
    
    template <typename step_t, typename U = T>
    static std::enable_if_t<sizeof(U) == 2, Vectorized<T>> arange(
        T base = 0,
        step_t step = static_cast<step_t>(1)) {
    
    
    
    // 当模板参数 U 是 2 字节大小的类型时，返回一个 Vectorized 对象，使用给定的步长和基础值进行初始化
    template <typename step_t, typename U = T>
    static std::enable_if_t<
    return Vectorized<T>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step);
  }


  template <typename step_t, typename U = T>
  // 如果 T 是 1 字节大小的类型，则启用该函数用于生成 Vectorized<T> 对象
  static std::enable_if_t<sizeof(U) == 1, Vectorized<T>> arange(
      T base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step,
        base + 16 * step,
        base + 17 * step,
        base + 18 * step,
        base + 19 * step,
        base + 20 * step,
        base + 21 * step,
        base + 22 * step,
        base + 23 * step,
        base + 24 * step,
        base + 25 * step,
        base + 26 * step,
        base + 27 * step,
        base + 28 * step,
        base + 29 * step,
        base + 30 * step,
        base + 31 * step);
  }


  // blend section
  template <int64_t mask>
  // 如果 blendChoice<sizeof(T)>(mask) 的结果为 0，则选择 a 的向量
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 0, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    return a;
  }

  template <int64_t mask>
  // 如果 blendChoice<sizeof(T)>(mask) 的结果为 1，则选择 b 的向量
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 1, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    return b;
  }

  template <int64_t mask>
  // 如果 blendChoice<sizeof(T)>(mask) 的结果为 2，则混合 a 和 b 的向量
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 2, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    return {b._vec0, a._vec1};
  }

  template <int64_t mask>
  // 如果 blendChoice<sizeof(T)>(mask) 的结果为 3，则混合 a 和 b 的向量
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 3, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    return {a._vec0, b._vec1};
  }

  template <int64_t mask>
  // 如果 blendChoice<sizeof(T)>(mask) 的结果为 4，则按位选择 a 和 b 的向量
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 4, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    const vmaskType mask_1st = GetMask1<sizeof(T)>(mask);
    return {(vtype)vec_sel(a._vec0, b._vec0, mask_1st), a._vec1};
  }

  template <int64_t mask>
  // 如果 blendChoice<sizeof(T)>(mask) 的结果为 5，则按位选择 a 和 b 的向量
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 5, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    const vmaskType mask_1st = GetMask1<sizeof(T)>(mask);
  return {(vtype)vec_sel(a._vec0, b._vec0, mask_1st), b._vec1};


// 如果 blendChoice 返回的掩码是 6，使用给定的掩码来混合两个 Vectorized 对象
template <int64_t mask>
static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 6, Vectorized<T>>
    C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 获取第二个掩码
  const vmaskType mask_2nd = GetMask2<sizeof(T)>(mask);
  // 返回混合后的结果，第一个向量保持不变，第二个向量按照第二个掩码进行混合
  return {a._vec0, (vtype)vec_sel(a._vec1, b._vec1, mask_2nd)};
}



  return {a._vec0, (vtype)vec_sel(a._vec1, b._vec1, mask_2nd)};


// 如果 blendChoice 返回的掩码是 7，使用给定的掩码来混合两个 Vectorized 对象
template <int64_t mask>
static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 7, Vectorized<T>>
    C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 获取第二个掩码
  const vmaskType mask_2nd = GetMask2<sizeof(T)>(mask);
  // 返回混合后的结果，第一个向量替换为第二个向量，第二个向量按照第二个掩码进行混合
  return {b._vec0, (vtype)vec_sel(a._vec1, b._vec1, mask_2nd)};
}



  return {
      (vtype)vec_sel(a._vec0, b._vec0, mask_1st),
      (vtype)vec_sel(a._vec1, b._vec1, mask_2nd)};


// 如果 blendChoice 返回的掩码是 8，使用给定的掩码来混合两个 Vectorized 对象
template <int64_t mask>
static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 8, Vectorized<T>>
    C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 获取第一个和第二个掩码
  const vmaskType mask_1st = GetMask1<sizeof(T)>(mask);
  const vmaskType mask_2nd = GetMask2<sizeof(T)>(mask);
  // 返回混合后的结果，两个向量分别按照各自的掩码进行混合
  return {
      (vtype)vec_sel(a._vec0, b._vec0, mask_1st),
      (vtype)vec_sel(a._vec1, b._vec1, mask_2nd)};
}



  return b;


// 当 Z 大于等于 C 时，返回向量 b
template <int16_t Z, int16_t C>
static inline std::enable_if_t<(Z >= C), Vectorized<T>> set_inner(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    size_t count) {
  return b;
}



  if (count == Z)
    return blend<allbitset(Z)>(a, b);
  else
    return set_inner<Z + 1, C>(a, b, count);


// 当 Z 小于 C 时，根据 count 的值来选择调用 blend 或递归调用 set_inner
template <int16_t Z, int16_t C>
static inline std::enable_if_t<(Z < C), Vectorized<T>> set_inner(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    size_t count) {
  if (count == Z)
    return blend<allbitset(Z)>(a, b);
  else
    return set_inner<Z + 1, C>(a, b, count);
}



  if (count == 0)
    return a;
  return set_inner<1, size()>(a, b, count);


// 设置向量的操作，根据 count 的值决定是返回向量 a 还是调用 set_inner 来处理
static Vectorized<T> set(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    size_t count = size()) {
  if (count == 0)
    return a;
  return set_inner<1, size()>(a, b, count);
}



  return {(vtype)vec_nor(vecb0(), vecb0()), (vtype)vec_nor(vecb1(), vecb1())};


// 对向量执行逻辑非操作
Vectorized<T> _not() const {
  // 对两个内部向量执行逻辑 NOR 操作，得到结果向量
  return {(vtype)vec_nor(vecb0(), vecb0()), (vtype)vec_nor(vecb1(), vecb1())};
}



  return (*this == other) & Vectorized<T>((T)1.0);


// 向量执行相等比较操作
Vectorized<T> C10_ALWAYS_INLINE eq(const Vectorized<T>& other) const {
  // 返回比较结果的逻辑与操作，结果向量中的元素为 1.0
  return (*this == other) & Vectorized<T>((T)1.0);
}



  return (*this != other) & Vectorized<T>((T)1.0);


// 向量执行不等比较操作
Vectorized<T> C10_ALWAYS_INLINE ne(const Vectorized<T>& other) const {
  // 返回比较结果的逻辑与操作，结果向量中的元素为 1.0
  return (*this != other) & Vectorized<T>((T)1.0);
}



  return (*this > other) & Vectorized<T>((T)1.0);


// 向量执行大于比较操作
Vectorized<T> C10_ALWAYS_INLINE gt(const Vectorized<T>& other) const {
  // 返回比较结果的逻辑与操作，结果向量中的元素为 1.0
  return (*this > other) & Vectorized<T>((T)1.0);
}



  return (*this >= other) & Vectorized<T>((T)1.0);


// 向量执行大于等于比较操作
Vectorized<T> C10_ALWAYS_INLINE ge(const Vectorized<T>& other) const {
  // 返回比较结果的逻辑与操作，结果向量中的元素为 1.0
  return (*this >= other) & Vectorized<T>((T)1.0);
}



  return (*this < other) & Vectorized<T>((T)1.0);


// 向量执行小于比较操作
Vectorized<T> C10_ALWAYS_INLINE lt(const Vectorized<T>& other) const {
  // 返回比较结果的逻辑与操作，结果向量中的元素为 1.0
  return (*this < other) & Vectorized<T>((T)1.0);
}



  return (*this <= other) & Vectorized<T>((T)1.0);


// 向量执行小于等于比较操作
Vectorized<T> C10_ALWAYS_INLINE le(const Vectorized<T>& other) const {
  // 返回比较结果的逻辑与操作，结果向量中的元素为 1.0
  return (*this <= other) & Vectorized<T>((T)1.0);
}
  template <
      typename U = T,
      std::enable_if_t<!std::is_unsigned<U>::value, int> = 0>
  // 如果类型 U 不是无符号整数，则返回向量的绝对值
  Vectorized<U> C10_ALWAYS_INLINE abs() const {
    return {vec_abs(_vec0), vec_abs(_vec1)};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_unsigned<U>::value, int> = 0>
  // 如果类型 U 是无符号整数，则返回向量本身（因为无符号整数没有负值）
  Vectorized<U> C10_ALWAYS_INLINE abs() const {
    return {_vec0, _vec1};
  }

  // 返回向量的负值
  Vectorized<T> C10_ALWAYS_INLINE neg() const {
    return {-_vec0, -_vec1};
  }

  // 检查向量中每个元素是否是 NaN（非数值）
  Vectorized<T> isnan() const {
    auto x = *this;
    auto ret = (x == x);  // 检查每个元素是否等于自身，NaN 与任何数比较都是 false
    return ret._not();    // 返回取反后的结果，即将非 NaN 的元素置为 true
  }

  // 检查向量中是否包含无穷大或 NaN
  bool has_inf_nan() const {
    for (const auto i : c10::irange(size()/2)) {
      if(_isnan(_vec0[i]) || _isinf(_vec0[i])) {  // 检查_vec0数组中的元素是否是 NaN 或 无穷大
        return true;
      }
    }
    for (const auto i : c10::irange(size()/2)) {
      if(_isnan(_vec1[i]) || _isinf(_vec1[i])) {  // 检查_vec1数组中的元素是否是 NaN 或 无穷大
        return true;
      }
    }
    return false;  // 向量中没有包含 NaN 或 无穷大的元素
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  // 计算向量中每个元素的角度，返回角度值
  Vectorized<U> angle() const {
    auto tmp = blendv(
        Vectorized<U>(0), Vectorized<U>(c10::pi<U>), *this < Vectorized<U>(0));
    return blendv(tmp, *this, isnan());
  }

  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point<U>::value, int> = 0>
  // 计算向量中每个元素的角度，返回角度值（如果不是浮点数类型，忽略isnan）
  Vectorized<U> angle() const {
    return blendv(
        Vectorized<U>(0), Vectorized<U>(c10::pi<U>), *this < Vectorized<U>(0));
  }

  // 返回向量的实部，即返回自身
  Vectorized<T> real() const {
    return *this;
  }

  // 返回向量的虚部，即返回全零向量
  Vectorized<T> imag() const {
    return Vectorized<T>{0};
  }

  // 返回向量的共轭，即返回自身
  Vectorized<T> conj() const {
    return *this;
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  // 返回向量中每个元素是否为零的掩码（mask）
  int zero_mask() const {
    auto cmp = (*this == Vectorized<U>(0));
    constexpr auto mask_zero_bits = GetBpermZeroMask<U>();
    ZSimdVectBinary<uint64_t> result0 =
        vec_bperm_u128((ZSimdVectBinary<uint8_t>)cmp.vecb0(), mask_zero_bits);
    ZSimdVectBinary<uint64_t> result1 =
        vec_bperm_u128((ZSimdVectBinary<uint8_t>)cmp.vecb1(), mask_zero_bits);
    return (result0[0] | (result1[0] << (size() / 2)));  // 将结果合并成一个整数作为返回值
  }

  // 返回向量中每个元素的下取整结果
  Vectorized<T> C10_ALWAYS_INLINE floor() const {
    return {vec_floor(_vec0), vec_floor(_vec1)};
  }

  // 返回向量中每个元素的上取整结果
  Vectorized<T> C10_ALWAYS_INLINE ceil() const {
    return {vec_ceil(_vec0), vec_ceil(_vec1)};
  }

  // 返回向量中每个元素的四舍五入结果
  Vectorized<T> C10_ALWAYS_INLINE round() const {
    return {vec_round(_vec0), vec_round(_vec1)};
  }

  // 返回向量中每个元素的最接近整数的浮点数值
  Vectorized<T> C10_ALWAYS_INLINE rint() const {
    return {vec_rint(_vec0), vec_rint(_vec1)};
  }

  // 返回向量中每个元素的截断整数部分
  Vectorized<T> C10_ALWAYS_INLINE trunc() const {
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }

  // 返回向量中每个元素的小数部分
  Vectorized<T> C10_ALWAYS_INLINE frac() const {
    return *this - trunc();
  }

  // 返回向量中每个元素的平方根
  Vectorized<T> C10_ALWAYS_INLINE sqrt() const {
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }
  // 返回向量中每个元素的倒数
  Vectorized<T> C10_ALWAYS_INLINE reciprocal() const {
  return Vectorized<T>((T)1) / (*this);


  # 返回一个向量，其每个元素都是常数 1 的倒数除以当前向量的对应元素
  return Vectorized<T>((T)1) / (*this);



  Vectorized<T> C10_ALWAYS_INLINE rsqrt() const {
    return sqrt().reciprocal();
  }


  # 计算当前向量元素的平方根的倒数向量
  Vectorized<T> C10_ALWAYS_INLINE rsqrt() const {
    return sqrt().reciprocal();
  }



  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, float>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(float (*const f)(float)) const {
    float a00 = f(_vec0[0]);
    float a01 = f(_vec0[1]);
    float a02 = f(_vec0[2]);
    float a03 = f(_vec0[3]);
    float a10 = f(_vec1[0]);
    float a11 = f(_vec1[1]);
    float a12 = f(_vec1[2]);
    float a13 = f(_vec1[3]);
    return Vectorized<T>{a00, a01, a02, a03, a10, a11, a12, a13};
  }


  # 将给定的浮点数函数应用于当前向量中每个元素，返回结果构成的新向量
  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, float>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(float (*const f)(float)) const {
    float a00 = f(_vec0[0]);
    float a01 = f(_vec0[1]);
    float a02 = f(_vec0[2]);
    float a03 = f(_vec0[3]);
    float a10 = f(_vec1[0]);
    float a11 = f(_vec1[1]);
    float a12 = f(_vec1[2]);
    float a13 = f(_vec1[3]);
    return Vectorized<T>{a00, a01, a02, a03, a10, a11, a12, a13};
  }



  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, double>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(double (*const f)(double)) const {
    return Vectorized<T>(f(_vec0[0]), f(_vec0[1]), f(_vec1[0]), f(_vec1[1]));
  }


  # 将给定的双精度浮点数函数应用于当前向量中每个元素，返回结果构成的新向量
  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, double>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(double (*const f)(double)) const {
    return Vectorized<T>(f(_vec0[0]), f(_vec0[1]), f(_vec1[0]), f(_vec1[1]));
  }



  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, float>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(
      float (*const f)(float, float),
      const Vectorized<T>& b) const {
    float a00 = f(_vec0[0], b._vec0[0]);
    float a01 = f(_vec0[1], b._vec0[1]);
    float a02 = f(_vec0[2], b._vec0[2]);
    float a03 = f(_vec0[3], b._vec0[3]);
    float a10 = f(_vec1[0], b._vec1[0]);
    float a11 = f(_vec1[1], b._vec1[1]);
    float a12 = f(_vec1[2], b._vec1[2]);
    float a13 = f(_vec1[3], b._vec1[3]);
    return Vectorized<T>{a00, a01, a02, a03, a10, a11, a12, a13};
  }


  # 将给定的双参数浮点数函数应用于当前向量和另一个向量中的对应元素，返回结果构成的新向量
  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, float>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(
      float (*const f)(float, float),
      const Vectorized<T>& b) const {
    float a00 = f(_vec0[0], b._vec0[0]);
    float a01 = f(_vec0[1], b._vec0[1]);
    float a02 = f(_vec0[2], b._vec0[2]);
    float a03 = f(_vec0[3], b._vec0[3]);
    float a10 = f(_vec1[0], b._vec1[0]);
    float a11 = f(_vec1[1], b._vec1[1]);
    float a12 = f(_vec1[2], b._vec1[2]);
    float a13 = f(_vec1[3], b._vec1[3]);
    return Vectorized<T>{a00, a01, a02, a03, a10, a11, a12, a13};
  }



  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, double>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(
      double (*const f)(double, double),
      const Vectorized<T>& b) const {
    return Vectorized<T>(
        f(_vec0[0], b._vec0[0]),
        f(_vec0[1], b._vec0[1]),
        f(_vec1[0], b._vec1[0]),
        f(_vec1[1], b._vec1[1]));
  }


  # 将给定的双参数双精度浮点数函数应用于当前向量和另一个向量中的对应元素，返回结果构成的新向量
  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, double>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(
      double (*const f)(double, double),
      const Vectorized<T>& b) const {
    return Vectorized<T>(
        f(_vec0[0], b._vec0[0]),
        f(_vec0[1], b._vec0[1]),
        f(_vec1[0], b._vec1[0]),
        f(_vec1[1], b._vec1[1]));
  }



  template <
      typename FloatOp,
      typename DoubleOp,
      typename U = T,
      std::enable_if_t<std::is_same<U, float>::value, int> = 0>
  inline Vectorized<T> mapSleef(FloatOp f, DoubleOp d) const {
    vtype a0 = f(_vec0);
    vtype a1 = f(_vec1);
    return Vectorized<T>{a0, a1};
  }


  # 将给定的 Sleef 浮点数操作函数分别应用于当前向量的每个分量，返回结果构成的新向量
  template <
      typename FloatOp,
      typename DoubleOp,
      typename U = T,
      std::enable_if_t<std::is_same<U, float>::value, int> = 0>
  inline Vectorized<T> mapSleef(FloatOp f, DoubleOp d) const {
    vtype a0 = f(_vec0);
    vtype a1 = f(_vec1);
    return Vectorized<T>{a0, a1};
  }



  template <
      typename FloatOp,
      typename DoubleOp,
      typename U = T,
      std::enable_if_t<std::is_same<U, double>::value, int> = 0>
  inline Vectorized<T> mapSleef(FloatOp f, DoubleOp d) const {
    return Vectorized<T>(d(_vec0), d(_vec1));
  }


  # 将给定的 Sleef 双精度浮点数操作函数分别应用于当前向量的每个分量，返回结果构成的新向量
  template <
      typename FloatOp,
      typename DoubleOp,
      typename U = T,
      std::enable_if_t<std::is_same<U, double>::value, int> = 0>
  inline Vectorized<T> mapSleef(FloatOp f, DoubleOp d
  Vectorized<T> acos() const {
    // 使用 Sleef 库中的 acos 函数对向量进行逐元素计算
    return mapSleef(Sleef_acosf4_u10, Sleef_acosd2_u10);
  }
  Vectorized<T> asin() const {
    // 使用 Sleef 库中的 asin 函数对向量进行逐元素计算
    return mapSleef(Sleef_asinf4_u10, Sleef_asind2_u10);
  }
  Vectorized<T> atan() const {
    // 使用 Sleef 库中的 atan 函数对向量进行逐元素计算
    return mapSleef(Sleef_atanf4_u10, Sleef_atand2_u10);
  }
  Vectorized<T> atanh() const {
    // 使用 Sleef 库中的 atanh 函数对向量进行逐元素计算
    return mapSleef(Sleef_atanhf4_u10, Sleef_atanhd2_u10);
  }

  Vectorized<T> erf() const {
    // 使用 Sleef 库中的 erf 函数对向量进行逐元素计算
    return mapSleef(Sleef_erff4_u10, Sleef_erfd2_u10);
  }
  Vectorized<T> erfc() const {
    // 使用 Sleef 库中的 erfc 函数对向量进行逐元素计算
    return mapSleef(Sleef_erfcf4_u15, Sleef_erfcd2_u15);
  }

  Vectorized<T> exp() const {
    // 使用 Sleef 库中的 exp 函数对向量进行逐元素计算
    return mapSleef(Sleef_expf4_u10, Sleef_expd2_u10);
  }
  Vectorized<T> exp2() const {
    // 使用 Sleef 库中的 exp2 函数对向量进行逐元素计算
    return mapSleef(Sleef_exp2f4_u10, Sleef_exp2d2_u10);
  }
  Vectorized<T> expm1() const {
    // 使用 Sleef 库中的 expm1 函数对向量进行逐元素计算
    return mapSleef(Sleef_expm1f4_u10, Sleef_expm1d2_u10);
  }
  Vectorized<T> exp_u20() const {
    // 调用 exp 函数，返回结果
    return exp();
  }

  Vectorized<T> log() const {
    // 使用 Sleef 库中的 log 函数对向量进行逐元素计算
    return mapSleef(Sleef_logf4_u10, Sleef_logd2_u10);
  }
  Vectorized<T> log2() const {
    // 使用 Sleef 库中的 log2 函数对向量进行逐元素计算
    return mapSleef(Sleef_log2f4_u10, Sleef_log2d2_u10);
  }
  Vectorized<T> log10() const {
    // 使用 Sleef 库中的 log10 函数对向量进行逐元素计算
    return mapSleef(Sleef_log10f4_u10, Sleef_log10d2_u10);
  }
  Vectorized<T> log1p() const {
    // 使用 Sleef 库中的 log1p 函数对向量进行逐元素计算
    return mapSleef(Sleef_log1pf4_u10, Sleef_log1pd2_u10);
  }

  Vectorized<T> sin() const {
    // 使用 Sleef 库中的 sin 函数对向量进行逐元素计算
    return mapSleef(Sleef_sinf4_u10, Sleef_sind2_u10);
  }
  Vectorized<T> sinh() const {
    // 使用 Sleef 库中的 sinh 函数对向量进行逐元素计算
    return mapSleef(Sleef_sinhf4_u10, Sleef_sinhd2_u10);
  }
  Vectorized<T> cos() const {
    // 使用 Sleef 库中的 cos 函数对向量进行逐元素计算
    return mapSleef(Sleef_cosf4_u10, Sleef_cosd2_u10);
  }
  Vectorized<T> cosh() const {
    // 使用 Sleef 库中的 cosh 函数对向量进行逐元素计算
    return mapSleef(Sleef_coshf4_u10, Sleef_coshd2_u10);
  }

  Vectorized<T> tan() const {
    // 使用 Sleef 库中的 tan 函数对向量进行逐元素计算
    return mapSleef(Sleef_tanf4_u10, Sleef_tand2_u10);
  }
  Vectorized<T> tanh() const {
    // 使用 Sleef 库中的 tanh 函数对向量进行逐元素计算
    return mapSleef(Sleef_tanhf4_u10, Sleef_tanhd2_u10);
  }

  Vectorized<T> lgamma() const {
    // 使用 Sleef 库中的 lgamma 函数对向量进行逐元素计算
    return mapSleef(Sleef_lgammaf4_u10, Sleef_lgammad2_u10);
  }

  Vectorized<T> atan2(const Vectorized<T>& b) const {
    // 使用 Sleef 库中的 atan2 函数对向量进行逐元素计算，传入第二个向量 b
    return mapSleef(Sleef_atan2f4_u10, Sleef_atan2d2_u10, b);
  }
  Vectorized<T> copysign(const Vectorized<T>& sign) const {
    // 使用 Sleef 库中的 copysign 函数对向量进行逐元素计算，传入第二个向量 sign
    return mapSleef(Sleef_copysignf4, Sleef_copysignd2, sign);
  }
  Vectorized<T> fmod(const Vectorized<T>& q) const {
    // 使用 Sleef 库中的 fmod 函数对向量进行逐元素计算，传入第二个向量 q
    return mapSleef(Sleef_fmodf4, Sleef_fmodd2, q);
  }

  Vectorized<T> hypot(const Vectorized<T>& b) const {
    // 使用 Sleef 库中的 hypot 函数对向量进行逐元素计算，传入第二个向量 b
    return mapSleef(Sleef_hypotf4_u05, Sleef_hypotd2_u05, b);
  }

  Vectorized<T> pow(const Vectorized<T>& b) const {
    // 使用 Sleef 库中的 pow 函数对向量进行逐元素计算，传入第二个向量 b
    return mapSleef(Sleef_powf4_u10, Sleef_powd2_u10, b);
  }

  Vectorized<T> nextafter(const Vectorized<T>& b) const {
    // 使用 Sleef 库中的 nextafter 函数对向量进行逐元素计算，传入第二个向量 b
    return mapSleef(Sleef_nextafterf4, Sleef_nextafterd2, b);
  }

  Vectorized<T> erfinv() const {
    // 调用普通的函数 calc_erfinv 进行逐元素计算
    return mapOrdinary(calc_erfinv);
  }

  Vectorized<T> digamma() const {
    // 调用普通的函数 calc_digamma 进行逐元素计算
    return mapOrdinary(calc_digamma);
  }

  Vectorized<T> igamma(const Vectorized<T>& x) const {
    // 使用普通的函数 calc_igamma 对向量进行逐元素计算，传入第二个向量 x
    return mapOrdinary(calc_igamma, x);
  }

  Vectorized<T> igammac(const Vectorized<T>& x) const {
    // 使用普通的函数 calc_igammac 对向量进行逐元素计算，传入第二个向量 x
    return mapOrdinary(calc_igammac, x);
  }
    // 返回通过 mapOrdinary 应用 calc_igammac 函数后的结果
    return mapOrdinary(calc_igammac, x);
  }

  // 返回通过 mapOrdinary 应用 calc_i0 函数后的结果
  Vectorized<T> i0() const {
    return mapOrdinary(calc_i0);
  }

  // 返回通过 mapOrdinary 应用 calc_i0e 函数后的结果
  Vectorized<T> i0e() const {
    return mapOrdinary(calc_i0e);
  }

  // 返回两个向量中每个元素的最小值
  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> minimum(const Vectorized<T>& other) const {
    return {vec_min(_vec0, other._vec0), vec_min(_vec1, other._vec1)};
  }

  /* Propagates NaN if either input is a NaN. */
  // 返回两个向量中每个元素的最小值，如果有 NaN，则传播 NaN
  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> minimum(const Vectorized<T>& other) const {
    Vectorized<T> tmp = {vec_min(_vec0, other._vec0), vec_min(_vec1, other._vec1)};
    tmp = blendv(tmp, *this, isnan());
    return blendv(tmp, other, other.isnan());
  }

  // 返回两个向量中每个元素的最大值
  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> maximum(const Vectorized<T>& other) const {
    return {vec_max(_vec0, other._vec0), vec_max(_vec1, other._vec1)};
  }

  /* Propagates NaN if either input is a NaN. */
  // 返回两个向量中每个元素的最大值，如果有 NaN，则传播 NaN
  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> maximum(const Vectorized<T>& other) const {
    Vectorized<T> tmp = {vec_max(_vec0, other._vec0), vec_max(_vec1, other._vec1)};
    tmp = blendv(tmp, *this, isnan());
    return blendv(tmp, other, other.isnan());
  }

  // 返回两个向量中每个元素与给定最小向量中每个元素的最大值
  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> clamp_min(const Vectorized<T>& min) const {
    return {vec_max(_vec0, min._vec0), vec_max(_vec1, min._vec1)};
  }

  /* Keeps NaN if actual value is NaN */
  // 返回两个向量中每个元素与给定最小向量中每个元素的最大值，如果有 NaN，则保留 NaN
  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> clamp_min(const Vectorized<T>& min) const {
    Vectorized<T> tmp = {vec_max(_vec0, min._vec0), vec_max(_vec1, min._vec1)};
    return blendv(tmp, *this, isnan());
  }

  // 返回两个向量中每个元素与给定最大向量中每个元素的最小值
  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> clamp_max(const Vectorized<T>& max) const {
    return {vec_min(_vec0, max._vec0), vec_min(_vec1, max._vec1)};
  }

  /* Keeps NaN if actual value is NaN */
  // 返回两个向量中每个元素与给定最大向量中每个元素的最小值，如果有 NaN，则保留 NaN
  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> clamp_max(const Vectorized<T>& max) const {
    Vectorized<T> tmp = {vec_min(_vec0, max._vec0), vec_min(_vec1, max._vec1)};
    return blendv(tmp, *this, isnan());
  }

  // 返回浮点类型向量的交换后的结果
  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, float>::value, int> = 0>
  Vectorized<T> swapped() const {
    auto swap_mask = GetSwapMaskFloat();
    vtype v0 = vec_perm(_vec0, _vec0, swap_mask);
    vtype v1 = vec_perm(_vec1, _vec1, swap_mask);
    return {v0, v1};
  }

  // 返回双精度类型向量的交换后的结果
  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, double>::value, int> = 0>
  Vectorized<T> swapped() const {
    vtype v0 = vec_permi(_vec0, _vec0, 2);
    // 使用 vec_permi 函数对 _vec0 进行排列操作，返回结果存储在 v0 中
    vtype v1 = vec_permi(_vec1, _vec1, 2);
    // 使用 vec_permi 函数对 _vec1 进行排列操作，返回结果存储在 v1 中
    return {v0, v1};
    // 返回包含 v0 和 v1 的向量，作为函数的结果
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  static Vectorized<T> mergee(Vectorized<T>& first, Vectorized<T>& second) {
    // 合并两个浮点数向量，通过 vec_mergee 函数实现
    return {
        vec_mergee(first._vec0, second._vec0),
        vec_mergee(first._vec1, second._vec1)};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  static Vectorized<T> mergeo(Vectorized<T>& first, Vectorized<T>& second) {
    // 合并两个浮点数向量，通过 vec_mergeo 函数实现
    return {
        vec_mergeo(first._vec0, second._vec0),
        vec_mergeo(first._vec1, second._vec1)};
  }

  static Vectorized<T> horizontal_add_perm(
      Vectorized<T>& first,
      Vectorized<T>& second) {
    // 使用 6 条指令模拟水平加法操作
    // 对第二个向量进行排列，以便进行加法操作以获取水平和
    auto first_perm = first.swapped(); // 2perm
    auto second_perm = second.swapped(); // 2perm
    // 计算
    auto first_ret = first + first_perm; // 2add
    auto second_ret = second + second_perm; // 2 add
    // 选择偶数位置的元素
    return mergee(first_ret, second_ret); // 2 mergee's
  }

  static Vectorized<T> horizontal_sub_perm(
      Vectorized<T>& first,
      Vectorized<T>& second) {
    // 使用 6 条指令模拟水平减法操作
    // 对第二个向量进行排列，以便进行减法操作以获取水平差
    auto first_perm = first.swapped(); // 2perm
    auto second_perm = second.swapped(); // 2perm
    // 计算
    auto first_ret = first - first_perm; // 2sub
    auto second_ret = second - second_perm; // 2 sub
    // 选择偶数位置的元素
    return mergee(first_ret, second_ret); // 2 mergee's
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> mergee() const {
    // 合并当前浮点数向量的两个部分，通过 vec_mergee 函数实现
    return {vec_mergee(_vec0, _vec0), vec_mergee(_vec1, _vec1)};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> mergeo() const {
    // 合并当前浮点数向量的两个部分，通过 vec_mergeo 函数实现
    return {vec_mergeo(_vec0, _vec0), vec_mergeo(_vec1, _vec1)};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, uint8_t>::value, int> = 0>
  Vectorized<int32_t> to_vec_float_helper() const {
    // 辅助函数，将当前向量的浮点数部分转换为 int32_t 类型的向量
    int32_t values[8] = {
      _vec0[0],
      _vec0[1],
      _vec0[2],
      _vec0[3],
      _vec0[4],
      _vec0[5],
      _vec0[6],
      _vec0[7],
    };

    return Vectorized<int32_t>{
      values[0], values[1], values[2], values[3],
      values[4], values[5], values[6], values[7]
    };
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, int32_t>::value, int> = 0>
  Vectorized<uint8_t> to_vec_uint8_helper() const {
    // 辅助函数，用于将浮点数向量转换为 uint8_t 类型的向量
    // 这里可能还有待补充的代码
    # 创建一个包含特定向量元素的 uint8_t 数组，从两个不同的向量 _vec0 和 _vec1 中提取元素，并转换为 uint8_t 类型
    uint8_t values[8] = {
      static_cast<uint8_t>(_vec0[0]),   # 从 _vec0 中提取第一个元素，并转换为 uint8_t 类型
      static_cast<uint8_t>(_vec0[1]),   # 从 _vec0 中提取第二个元素，并转换为 uint8_t 类型
      static_cast<uint8_t>(_vec0[2]),   # 从 _vec0 中提取第三个元素，并转换为 uint8_t 类型
      static_cast<uint8_t>(_vec0[3]),   # 从 _vec0 中提取第四个元素，并转换为 uint8_t 类型
      static_cast<uint8_t>(_vec1[0]),   # 从 _vec1 中提取第一个元素，并转换为 uint8_t 类型
      static_cast<uint8_t>(_vec1[1]),   # 从 _vec1 中提取第二个元素，并转换为 uint8_t 类型
      static_cast<uint8_t>(_vec1[2]),   # 从 _vec1 中提取第三个元素，并转换为 uint8_t 类型
      static_cast<uint8_t>(_vec1[3]),   # 从 _vec1 中提取第四个元素，并转换为 uint8_t 类型
    };
    
    # 创建一个 Vectorized<uint8_t> 类型的对象，其中包含从 values 数组中提取的前八个元素，后续元素为零填充
    return Vectorized<uint8_t>{
      values[0], values[1], values[2], values[3],   # 使用 values 数组的前四个元素初始化 Vectorized 对象
      values[4], values[5], values[6], values[7],   # 使用 values 数组的后四个元素初始化 Vectorized 对象
      0, 0, 0, 0,                                   # 对象的后续元素进行零填充
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    };
# 定义宏，为类型 `typex` 生成向量化操作符重载
#define ZVECTOR_OPERATORS(typex)                                                                           \
  # 定义加法操作符重载：两个向量化对象相加
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator+(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    return Vectorized<typex>{a.vec0() + b.vec0(), a.vec1() + b.vec1()};                                    \
  }                                                                                                        \
                                                                                                           \
  # 定义减法操作符重载：两个向量化对象相减
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator-(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    return Vectorized<typex>{a.vec0() - b.vec0(), a.vec1() - b.vec1()};                                    \
  }                                                                                                        \
                                                                                                           \
  # 定义乘法操作符重载：两个向量化对象相乘
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator*(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    return Vectorized<typex>{a.vec0() * b.vec0(), a.vec1() * b.vec1()};                                    \
  }                                                                                                        \
                                                                                                           \
  # 定义除法操作符重载：两个向量化对象相除
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator/(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    return Vectorized<typex>{a.vec0() / b.vec0(), a.vec1() / b.vec1()};                                    \
  }                                                                                                        \
                                                                                                           \
  # 定义按位与操作符重载：两个向量化对象按位与
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator&(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    // 返回按位与的向量结果
    return Vectorized<typex>{
        // 将第一个向量的第一个部分与第二个向量的第一个部分按位与，转换为相应类型
        (Vectorized<typex>::vtype)(a.vecb0() & b.vecb0()),
        // 将第一个向量的第二个部分与第二个向量的第二个部分按位与，转换为相应类型
        (Vectorized<typex>::vtype)(a.vecb1() & b.vecb1())};
    }
    
    // 返回按位或的向量结果
    template <>
    Vectorized<typex> C10_ALWAYS_INLINE operator|(const Vectorized<typex>& a, const Vectorized<typex>& b) {
      return Vectorized<typex>{
          // 将第一个向量的第一个部分与第二个向量的第一个部分按位或，转换为相应类型
          (Vectorized<typex>::vtype)(a.vecb0() | b.vecb0()),
          // 将第一个向量的第二个部分与第二个向量的第二个部分按位或，转换为相应类型
          (Vectorized<typex>::vtype)(a.vecb1() | b.vecb1())};
    }
    
    // 返回按位异或的向量结果
    template <>
    Vectorized<typex> C10_ALWAYS_INLINE operator^(const Vectorized<typex>& a, const Vectorized<typex>& b) {
      return Vectorized<typex>{
          // 将第一个向量的第一个部分与第二个向量的第一个部分按位异或，转换为相应类型
          (Vectorized<typex>::vtype)(a.vecb0() ^ b.vecb0()),
          // 将第一个向量的第二个部分与第二个向量的第二个部分按位异或，转换为相应类型
          (Vectorized<typex>::vtype)(a.vecb1() ^ b.vecb1())};
    }
    
    // 返回向量是否相等的比较结果向量
    Vectorized<typex> C10_ALWAYS_INLINE operator==(const Vectorized<typex>& a, const Vectorized<typex>& b) {
      return Vectorized<typex>{
          // 比较第一个向量的第一个部分与第二个向量的第一个部分是否相等
          vec_cmpeq(a.vec0(), b.vec0()),
          // 比较第一个向量的第二个部分与第二个向量的第二个部分是否相等
          vec_cmpeq(a.vec1(), b.vec1())};
    }
    
    // 返回向量是否不相等的比较结果向量
    Vectorized<typex> C10_ALWAYS_INLINE operator!=(const Vectorized<typex>& a, const Vectorized<typex>& b) {
    // 返回两个 Vectorized<typex> 对象的逐元素大于操作的结果
    Vectorized<typex> C10_ALWAYS_INLINE operator>(const Vectorized<typex>& a, const Vectorized<typex>& b) {
        // 创建一个新的 Vectorized<typex> 对象，其中包含逐元素的大于比较结果
        return Vectorized<typex>{
            vec_cmpgt(a.vec0(), b.vec0()),  // 对第一个向量的大于比较操作
            vec_cmpgt(a.vec1(), b.vec1())   // 对第二个向量的大于比较操作
        };
    }
    
    // 返回两个 Vectorized<typex> 对象的逐元素大于等于操作的结果
    Vectorized<typex> C10_ALWAYS_INLINE operator>=(const Vectorized<typex>& a, const Vectorized<typex>& b) {
        // 创建一个新的 Vectorized<typex> 对象，其中包含逐元素的大于等于比较结果
        return Vectorized<typex>{
            vec_cmpge(a.vec0(), b.vec0()),  // 对第一个向量的大于等于比较操作
            vec_cmpge(a.vec1(), b.vec1())   // 对第二个向量的大于等于比较操作
        };
    }
    
    // 返回两个 Vectorized<typex> 对象的逐元素小于操作的结果
    Vectorized<typex> C10_ALWAYS_INLINE operator<(const Vectorized<typex>& a, const Vectorized<typex>& b) {
        // 创建一个新的 Vectorized<typex> 对象，其中包含逐元素的小于比较结果
        return Vectorized<typex>{
            vec_cmplt(a.vec0(), b.vec0()),  // 对第一个向量的小于比较操作
            vec_cmplt(a.vec1(), b.vec1())   // 对第二个向量的小于比较操作
        };
    }
    
    // 返回两个 Vectorized<typex> 对象的逐元素小于等于操作的结果
    Vectorized<typex> C10_ALWAYS_INLINE operator<=(const Vectorized<typex>& a, const Vectorized<typex>& b) {
        // 创建一个新的 Vectorized<typex> 对象，其中包含逐元素的小于等于比较结果
        return Vectorized<typex>{
            vec_cmple(a.vec0(), b.vec0()),  // 对第一个向量的小于等于比较操作
            vec_cmple(a.vec1(), b.vec1())   // 对第二个向量的小于等于比较操作
        };
    }
// 定义宏 ZVECTOR_OPERATORS，用于生成特定类型的位移操作符重载函数模板
ZVECTOR_OPERATORS(float)
ZVECTOR_OPERATORS(double)
ZVECTOR_OPERATORS(int8_t)
ZVECTOR_OPERATORS(uint8_t)
ZVECTOR_OPERATORS(uint16_t)
ZVECTOR_OPERATORS(int16_t)
ZVECTOR_OPERATORS(int32_t)
ZVECTOR_OPERATORS(int64_t)

// 取消宏 ZVECTOR_OPERATORS 的定义
#undef ZVECTOR_OPERATORS

// 重新定义宏 ZVECTOR_OPERATORS
#define ZVECTOR_OPERATORS(typex)                                                                           \
  // 定义模板特化，为特定类型 typex 的 Vectorized 向量执行左移操作符重载函数
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator<<(const Vectorized<typex>& a, const Vectorized<typex>& b) { \
    // 计算 typex 类型的最大位移值
    constexpr Vectorized<typex>::ElementType max_shift                                                     \
      = sizeof(Vectorized<typex>::ElementType) * CHAR_BIT;                                                 \
                                                                                                           \
    // 声明 a、b、c 数组，用于存储向量化操作的数据
    Vectorized<typex>::ElementType a_array[Vectorized<typex>::size()];                                     \
    Vectorized<typex>::ElementType b_array[Vectorized<typex>::size()];                                     \
    Vectorized<typex>::ElementType c_array[Vectorized<typex>::size()];                                     \
                                                                                                           \
    // 将向量 a 存储到 a_array 数组中
    a.store(a_array);                                                                                      \
    // 将向量 b 存储到 b_array 数组中
    b.store(b_array);                                                                                      \
                                                                                                           \
    // 循环处理向量中的每个元素
    for (int i = 0; i != Vectorized<typex>::size(); i++) {                                                 \
      // 获取位移值 shift
      typex shift = b_array[i];                                                                            \
      // 如果位移值为负数或超出最大位移值，则将结果 c_array[i] 设置为 0
      if ((static_cast<std::make_signed_t<typex>>(shift) < 0) || (shift >= max_shift)) {                   \
        c_array[i] = 0;                                                                                    \
      } else {                                                                                             \
        // 否则，执行无符号左移操作并存储结果到 c_array[i]
        c_array[i] = static_cast<std::make_unsigned_t<typex>>(a_array[i]) << shift;                        \
      }                                                                                                    \
    }                                                                                                      \
                                                                                                           \
    /* 返回一个Vectorized<typex>对象，加载未对齐的C数组 */
    return Vectorized<typex>::loadu(c_array);                                                              \
  }                                                                                                        \
                                                                                                           \
  template <>                                                                                              \
  /* 重载右移操作符>>，用于Vectorized<typex>类型 */                                                         \
  Vectorized<typex> C10_ALWAYS_INLINE operator>>(const Vectorized<typex>& a, const Vectorized<typex>& b) { \
    /* 右移操作时保留符号位（对于有符号类型）或不保留位（对于无符号类型）的值 */                          \
    constexpr Vectorized<typex>::ElementType max_shift                                                     \
      = sizeof(typex) * CHAR_BIT - std::is_signed_v<typex>;                                                \
                                                                                                           \
    Vectorized<typex>::ElementType a_array[Vectorized<typex>::size()];                                     \
    Vectorized<typex>::ElementType b_array[Vectorized<typex>::size()];                                     \
    Vectorized<typex>::ElementType c_array[Vectorized<typex>::size()];                                     \
                                                                                                           \
    /* 将Vectorized<typex>对象a的值存储到数组a_array中 */                                                  \
    a.store(a_array);                                                                                      \
    /* 将Vectorized<typex>对象b的值存储到数组b_array中 */                                                  \
    b.store(b_array);                                                                                      \
                                                                                                           \
    /* 遍历Vectorized<typex>对象的元素 */                                                                   \
    for (int i = 0; i != Vectorized<typex>::size(); i++) {                                                 \
      typex shift = b_array[i];                                                                            \
      /* 如果位移量为负数或超过最大允许的位移值，则右移a_array[i]值对应的最大位数 */                      \
      if ((static_cast<std::make_signed_t<typex>>(shift) < 0) || (shift >= max_shift)) {                   \
        c_array[i] = a_array[i] >> max_shift;                                                              \
      } else {                                                                                             \
        /* 否则，按位移量shift对a_array[i]进行右移操作 */                                                \
        c_array[i] = a_array[i] >> shift;                                                                  \
      }                                                                                                    \
    }                                                                                                      \
                                                                                                           \
    // 返回对Vectorized<typex>类型的对象应用loadu函数后的结果，该函数加载从c_array指针开始的未对齐数据
    return Vectorized<typex>::loadu(c_array);                                                              \
  }                                                                                                        \
                                                                                                           \
  // 特化模板：按位取反运算符重载函数
  template <>                                                                                              \
  // 定义按位取反运算符~的重载，对Vectorized<typex>类型的对象a执行按位取反操作
  inline Vectorized<typex> operator~(const Vectorized<typex>& a) {                                         \
    // 返回a对象的_not()成员函数的结果，该函数执行按位取反操作
    return a._not();                                                                                       \
  }
// 宏 ZVECTOR_OPERATORS(int8_t)：展开一组操作符重载的模板定义，针对 int8_t 类型
ZVECTOR_OPERATORS(int8_t)
// 宏 ZVECTOR_OPERATORS(uint8_t)：展开一组操作符重载的模板定义，针对 uint8_t 类型
ZVECTOR_OPERATORS(uint8_t)
// 宏 ZVECTOR_OPERATORS(uint16_t)：展开一组操作符重载的模板定义，针对 uint16_t 类型
ZVECTOR_OPERATORS(uint16_t)
// 宏 ZVECTOR_OPERATORS(int16_t)：展开一组操作符重载的模板定义，针对 int16_t 类型
ZVECTOR_OPERATORS(int16_t)
// 宏 ZVECTOR_OPERATORS(int32_t)：展开一组操作符重载的模板定义，针对 int32_t 类型
ZVECTOR_OPERATORS(int32_t)
// 宏 ZVECTOR_OPERATORS(int64_t)：展开一组操作符重载的模板定义，针对 int64_t 类型
ZVECTOR_OPERATORS(int64_t)

#undef ZVECTOR_OPERATORS

// 宏 DEFINE_MAXMIN_FUNCS(operand_type)：定义模板函数 maximum 和 minimum，针对特定类型 operand_type
#define DEFINE_MAXMIN_FUNCS(operand_type)                                     \
  template <>                                                                 \
  Vectorized<operand_type> inline maximum(                                    \
      const Vectorized<operand_type>& a, const Vectorized<operand_type>& b) { \
    return a.maximum(b);                                                      \
  }                                                                           \
  template <>                                                                 \
  Vectorized<operand_type> inline minimum(                                    \
      const Vectorized<operand_type>& a, const Vectorized<operand_type>& b) { \
    return a.minimum(b);                                                      \
  }

// 宏 DEFINE_CLAMP_MAXMIN_FUNCS(typex)：定义模板函数 clamp_min、clamp_max 和 clamp，针对特定类型 typex
#define DEFINE_CLAMP_MAXMIN_FUNCS(typex)                          \
  DEFINE_MAXMIN_FUNCS(typex)                                      \
  template <>                                                     \
  Vectorized<typex> C10_ALWAYS_INLINE clamp_min(                  \
      const Vectorized<typex>& a, const Vectorized<typex>& min) { \
    return a.clamp_min(min);                                      \
  }                                                               \
  template <>                                                     \
  Vectorized<typex> C10_ALWAYS_INLINE clamp_max(                  \
      const Vectorized<typex>& a, const Vectorized<typex>& max) { \
    return a.clamp_max(max);                                      \
  }                                                               \
  template <>                                                     \
  Vectorized<typex> C10_ALWAYS_INLINE clamp(                      \
      const Vectorized<typex>& a,                                 \
      const Vectorized<typex>& min,                               \
      const Vectorized<typex>& max) {                             \
    return clamp_max(clamp_min(a, min), max);                     \
  }

// 宏定义模板函数集，针对不同的数据类型进行函数定义
DEFINE_CLAMP_MAXMIN_FUNCS(int8_t)
DEFINE_CLAMP_MAXMIN_FUNCS(uint8_t)
DEFINE_CLAMP_MAXMIN_FUNCS(int16_t)
DEFINE_CLAMP_MAXMIN_FUNCS(int32_t)
DEFINE_CLAMP_MAXMIN_FUNCS(int64_t)
DEFINE_CLAMP_MAXMIN_FUNCS(float)
DEFINE_CLAMP_MAXMIN_FUNCS(double)

// 匿名命名空间，包含以下条件编译的定义
namespace { /* unnamed namespace */

// 如果条件未满足，则发出警告提示信息，建议在 z15 平台上编译以获得更好的性能表现
#if !defined(vec_float) || __ARCH__ < 13
#warning \
    "float->int and int->float conversion is simulated. compile for z15 for improved performance"
// 定义函数 vec_int_flt，实现从整数向浮点数的模拟转换
inline ZSimdVect<float> vec_int_flt(const ZSimdVect<int> x) {
  return ZSimdVect<float>{float(x[0]), float(x[1]), float(x[2]), float(x[3])};
}
// 定义函数 vec_flt_int，实现从浮点数向整数的模拟转换
inline ZSimdVect<int> vec_flt_int(const ZSimdVect<float> x) {
  return ZSimdVect<int>{int(x[0]), int(x[1]), int(x[2]), int(x[3])};
}
#else
// 如果条件满足，则定义 vec_int_flt 为 vec_float，vec_flt_int 为 vec_signed
#define vec_int_flt vec_float
#define vec_flt_int vec_signed
#endif

Vectorized<float> zvec_convert_to_float(const Vectorized<int32_t>& x) {
  // 将Vectorized<int32_t>类型向量转换为Vectorized<float>类型向量
  return {vec_int_flt(x.vec0()), vec_int_flt(x.vec1())};
}

Vectorized<int32_t> zvec_convert_to_int(const Vectorized<float>& x) {
  // 将Vectorized<float>类型向量转换为Vectorized<int32_t>类型向量
  return {vec_flt_int(x.vec0()), vec_flt_int(x.vec1())};
}

Vectorized<double> zvec_convert_to_float(const Vectorized<int64_t>& x) {
  // 将Vectorized<int64_t>类型向量转换为Vectorized<double>类型向量
  return {vec_double(x.vec0()), vec_double(x.vec1())};
}

Vectorized<int64_t> zvec_convert_to_int(const Vectorized<double>& x) {
  // 将Vectorized<double>类型向量转换为Vectorized<int64_t>类型向量
  return {vec_signed(x.vec0()), vec_signed(x.vec1())};
}

} /* unnamed namespace */

template <typename T, typename V>
Vectorized<V> cast_zvector(const Vectorized<T>& x) {
  using cast_type = typename Vectorized<V>::vtype;
  // 将Vectorized<T>类型向量转换为Vectorized<V>类型向量
  return Vectorized<V>{(cast_type)x.vec0(), (cast_type)x.vec1()};
}

template <>
Vectorized<float> C10_ALWAYS_INLINE fmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  // 特化的浮点数向量乘加运算
  return Vectorized<float>{
      __builtin_s390_vfmasb(a.vec0(), b.vec0(), c.vec0()),
      __builtin_s390_vfmasb(a.vec1(), b.vec1(), c.vec1())};
}

template <>
Vectorized<double> C10_ALWAYS_INLINE fmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  // 特化的双精度浮点数向量乘加运算
  return Vectorized<double>{
      __builtin_s390_vfmadb(a.vec0(), b.vec0(), c.vec0()),
      __builtin_s390_vfmadb(a.vec1(), b.vec1(), c.vec1())};
}

template <>
Vectorized<int16_t> C10_ALWAYS_INLINE fmadd(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b,
    const Vectorized<int16_t>& c) {
  // 特化的int16_t类型向量乘加运算
  return Vectorized<int16_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}

template <>
Vectorized<int32_t> C10_ALWAYS_INLINE fmadd(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b,
    const Vectorized<int32_t>& c) {
  // 特化的int32_t类型向量乘加运算
  return Vectorized<int32_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE fmadd(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b,
    const Vectorized<int64_t>& c) {
  // 特化的int64_t类型向量乘加运算
  return Vectorized<int64_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE
convert_to_int_of_same_size<double>(const Vectorized<double>& src) {
  // 将双精度浮点数向量转换为相同大小的整型向量
  return zvec_convert_to_int(src);
}

template <>
Vectorized<int32_t> C10_ALWAYS_INLINE
convert_to_int_of_same_size<float>(const Vectorized<float>& src) {
  // 将单精度浮点数向量转换为相同大小的整型向量
  return zvec_convert_to_int(src);
}

template <>
inline void convert(const int32_t* src, float* dst, int64_t n) {
  // int32_t和float类型的数据大小相同的转换操作
  int64_t i;
  for (i = 0; i <= (n - Vectorized<float>::size());
       i += Vectorized<float>::size()) {
    const int32_t* src_a = src + i;
    float* dst_a = dst + i;
    auto input_vec = Vectorized<int32_t>::loadu(src_a);
    auto output_vec = zvec_convert_to_float(input_vec);
    output_vec.store(dst_a);
  }

  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

template <>
// 将长度为n的int64_t数组src转换为长度为n的double数组dst
inline void convert(const int64_t* src, double* dst, int64_t n) {
  int64_t i;
  // 使用向量化操作处理数组，每次处理Vectorized<double>::size()个元素
  for (i = 0; i <= (n - Vectorized<double>::size());
       i += Vectorized<double>::size()) {
    const int64_t* src_a = src + i;   // 指向src数组的当前片段起始位置
    double* dst_a = dst + i;          // 指向dst数组的当前片段起始位置
    auto input_vec = Vectorized<int64_t>::loadu(src_a);  // 加载int64_t向量
    auto output_vec = zvec_convert_to_float(input_vec);  // 转换为double向量
    output_vec.store(dst_a);          // 存储转换后的double向量到dst_a
  }
  // 处理剩余不足一个向量长度的元素
  for (; i < n; i++) {
    dst[i] = static_cast<double>(src[i]);  // 执行单个元素的转换
  }
}

// 定义reinterpret_cast的模板函数，将类型Fst转换为类型Cst的向量化操作
#define DEFINE_REINTERPRET_CAST_FUNCS(Fst, Cst)     \
  template <>                                       \
  C10_ALWAYS_INLINE Vectorized<Cst> cast<Cst, Fst>( \
      const Vectorized<Fst>& src) {                 \
    return cast_zvector<Fst, Cst>(src);             \
  }

// 定义reinterpret_cast的模板函数，将类型Fst转换为double、float、int64_t、int32_t、int16_t的向量化操作
#define DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(Fst) \
  DEFINE_REINTERPRET_CAST_FUNCS(Fst, double)      \
  DEFINE_REINTERPRET_CAST_FUNCS(Fst, float)       \
  DEFINE_REINTERPRET_CAST_FUNCS(Fst, int64_t)     \
  DEFINE_REINTERPRET_CAST_FUNCS(Fst, int32_t)     \
  DEFINE_REINTERPRET_CAST_FUNCS(Fst, int16_t)

// 分别对float、double、int64_t、int32_t、int16_t调用定义的reinterpret_cast函数
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(float)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(double)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int64_t)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int32_t)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int16_t)

// 解除之前定义的reinterpret_cast函数宏定义
#undef DEFINE_REINTERPRET_CAST_FUNCS

// unpack_type模板，将T解包成其对应的类型
template <typename T>
struct unpack_type {
  using type = T;
};
template <>
struct unpack_type<int8_t> {
  using type = int16_t;
};
template <>
struct unpack_type<uint8_t> {
  using type = int16_t;
};
template <>
struct unpack_type<int16_t> {
  using type = int32_t;
};

// pack_type模板，将T打包成其对应的类型
template <typename T>
struct pack_type {
  using type = T;
};
template <>
struct pack_type<int16_t> {
  using type = int8_t;
};
template <>
struct pack_type<int32_t> {
  using type = int16_t;
};

// 匿名命名空间，定义模板函数unpack，用于将T类型向量x解包成V类型的向量对
namespace { /* unnamed namespace */

template <typename T, typename V = typename unpack_type<T>::type>
std::pair<Vectorized<V>, Vectorized<V>> unpack(const Vectorized<T>& x) {
  auto vec0 = vec_unpackh(x.vec0());  // 解包x的高位向量
  auto vec1 = vec_unpackl(x.vec0());  // 解包x的低位向量
  auto vec2 = vec_unpackh(x.vec1());  // 解包x的第二个高位向量
  auto vec3 = vec_unpackl(x.vec1());  // 解包x的第二个低位向量
  return {Vectorized<V>{vec0, vec1}, Vectorized<V>{vec2, vec3}};  // 返回解包后的向量对
}

// 特化模板函数unpack，用于将uint8_t类型向量x解包成int16_t类型的向量对
template <>
std::pair<Vectorized<int16_t>, Vectorized<int16_t>> unpack<uint8_t, int16_t>(
    const Vectorized<uint8_t>& x) {
  using typeX = typename Vectorized<uint16_t>::vtype;
  typeX vec0 = vec_unpackh(x.vec0());  // 解包x的高位向量
  typeX vec1 = vec_unpackl(x.vec0());  // 解包x的低位向量
  typeX vec2 = vec_unpackh(x.vec1());  // 解包x的第二个高位向量
  typeX vec3 = vec_unpackl(x.vec1());  // 解包x的第二个低位向量
  // 进行uint16_t到int16_t的强制类型转换，并返回转换后的向量对
  return {
      cast_zvector<uint16_t, int16_t>(Vectorized<uint16_t>{vec0, vec1}),
      cast_zvector<uint16_t, int16_t>(Vectorized<uint16_t>{vec2, vec3})};
}

// pack_type模板，将T打包成其对应的类型V
template <typename T, typename V = typename pack_type<T>::type>
// 向量打包函数，将两个给定类型为 Vectorized<T> 的对象合并成一个 Vectorized<V> 对象
Vectorized<V> pack(const Vectorized<T>& first, const Vectorized<T>& second) {
  // 调用 vec_packs 函数将 first 的两个子向量打包到 vec0
  auto vec0 = vec_packs(first.vec0(), first.vec1());
  // 调用 vec_packs 函数将 second 的两个子向量打包到 vec1
  auto vec1 = vec_packs(second.vec0(), second.vec1());
  // 返回一个新的 Vectorized<V> 对象，包含两个打包后的向量
  return Vectorized<V>{vec0, vec1};
}

// 特化模板，用于将两个 Vectorized<int16_t> 对象打包成 Vectorized<uint8_t> 对象
template <>
Vectorized<uint8_t> pack(
    const Vectorized<int16_t>& first,
    const Vectorized<int16_t>& second) {
  // 调用 vec_packsu 函数将 first 的两个子向量打包到 vec0
  auto vec0 = vec_packsu(first.vec0(), first.vec1());
  // 调用 vec_packsu 函数将 second 的两个子向量打包到 vec1
  auto vec1 = vec_packsu(second.vec0(), second.vec1());
  // 返回一个新的 Vectorized<uint8_t> 对象，包含两个打包后的向量
  return Vectorized<uint8_t>{vec0, vec1};
}

} /* unnamed namespace */

//////////////////////////////////QUANT///////////////////////////////////////////

// 对于支持量化操作的特化模板 Vectorized<T, std::enable_if_t<is_zarch_implemented_quant<T>()>> 的定义
template <typename T>
struct Vectorized<T, std::enable_if_t<is_zarch_implemented_quant<T>()>> {
 public:
  using value_type = typename T::underlying;  // 定义 value_type 作为 T::underlying 的别名
  using vtype = ZSimdVect<value_type>;        // 定义 vtype 作为 ZSimdVect<value_type> 的别名
  using vmaskType = ZSimdVectBinary<value_type>;  // 定义 vmaskType 作为 ZSimdVectBinary<value_type> 的别名
  using vinner_type = Vectorized<value_type>; // 定义 vinner_type 作为 Vectorized<value_type> 的别名
  using size_type = int;                      // 定义 size_type 为 int 类型

  // 返回该向量类型的宽度
  static constexpr size_type size() {
    return VECTOR_WIDTH / sizeof(value_type);
  }

  // 返回浮点数向量的数量
  static constexpr size_t float_num_vecs() {
    return size() / Vectorized<float>::size();
  }

  // 返回整数向量的数量，与浮点数向量数量相同
  static constexpr int int_num_vecs() {
    return float_num_vecs();
  }

  // 定义 float_vec_return_type 为包含多个 Vectorized<float> 对象的数组类型
  using float_vec_return_type = std::array<Vectorized<float>, float_num_vecs()>;

  // 定义 int_vec_return_type 为包含多个 Vectorized<c10::qint32> 对象的数组类型
  using int_vec_return_type =
      std::array<Vectorized<c10::qint32>, int_num_vecs()>;

 private:
  vinner_type _vec;  // 私有成员变量 _vec，类型为 vinner_type

 public:
  // 默认构造函数
  Vectorized() {}

  // 显式构造函数，初始化 _vec 成员变量
  explicit C10_ALWAYS_INLINE Vectorized(vinner_type v) : _vec{v} {}

  // 构造函数，将 T 类型对象 val 初始化为 _vec 成员变量
  Vectorized(const T& val) : _vec(val.val_) {}

  // 返回 _vec 成员变量的引用
  C10_ALWAYS_INLINE const vinner_type& vec() const {
    return _vec;
  }

  // 加载未对齐的数据到 Vectorized<T> 对象，返回 Vectorized<T> 对象
  template <typename U>
  static Vectorized<T> C10_ALWAYS_INLINE
  loadu(const U* ptr, int count = size()) {
    return Vectorized<T>{vinner_type::loadu(ptr, count)};
  }

  // 将 Vectorized<T> 对象的数据存储到指定的内存位置 ptr，count 指定存储的元素数量
  template <typename U>
  void C10_ALWAYS_INLINE store(U* ptr, int count = size()) const {
    _vec.store(ptr, count);
  }

  // 对 Vectorized<T> 对象执行 ReLU 操作，返回 Vectorized<T> 对象
  Vectorized<T> relu(Vectorized<T> zero_point) const {
    return Vectorized<T>{_vec.maximum(zero_point._vec)};
  }

  // 对 Vectorized<T> 对象执行 ReLU6 操作，返回 Vectorized<T> 对象
  Vectorized<T> relu6(Vectorized<T> zero_point, Vectorized<T> q_six) const {
    auto ret_max = _vec.maximum(zero_point._vec);
    auto ret_min = ret_max.minimum(q_six._vec);
    return Vectorized<T>{ret_min};
  }

  // 宽化减法操作，对 Vectorized<T> 对象执行宽化减法，返回 int_vec_return_type 对象数组
  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 1, int> = 0>
  int_vec_return_type widening_subtract(Vectorized<T> b) const {
    return {*this - b};
  }

  // 解量化操作，将 Vectorized<T> 对象解量化为浮点数向量，返回 float_vec_return_type 对象数组
  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 1, int> = 0>
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
    auto float_val = zvec_convert_to_float(_vec);
  // 对量化结果进行反量化操作，返回浮点数向量类型
  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 1, int> = 0>
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
    // 将向量转换为浮点数向量
    auto float_val = zvec_convert_to_float(_vec);
    // 应用量化逆操作：先减去零点，再乘以缩放因子
    return {(float_val - zero_point) * scale};
  }

  // 静态方法：将浮点数向量量化为整数向量
  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 1, int> = 0>
  static Vectorized<T> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    // 将浮点数向量转换为 float_vec_return_type，并乘以逆缩放因子
    Vectorized<float> vecf = rhs[0];
    vecf = vecf * Vectorized<float>(inverse_scale);
    // 四舍五入到最接近的整数，并加上零点
    vecf = vecf.rint() + Vectorized<float>((float)(zero_point));
    // 将浮点数向量转换为整数向量
    auto veci = zvec_convert_to_int(vecf);

    return Vectorized<T>{veci};
  }

  // 静态方法：从整数向量重新量化为 T 类型向量
  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::int_num_vecs() == 1, int> = 0>
  static Vectorized<T> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    // 获取输入整数向量的第一个元素
    Vectorized<T> vi = inp[0];
    // 将整数向量转换为浮点数向量，并乘以倍乘因子
    auto vecf = zvec_convert_to_float(vi.vec());
    vecf = vecf * Vectorized<float>(multiplier);
    // 四舍五入到最接近的整数
    vecf = vecf.rint();
    // 将浮点数向量转换为整数向量，并加上零点
    auto veci = zvec_convert_to_int(vecf) + Vectorized<int>(zero_point);

    return Vectorized<T>{veci};
  }

  // 静态方法：宽化减法操作，将两个向量中的每个分量相减
  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::int_num_vecs() == 4, int> = 0>
  int_vec_return_type widening_subtract(Vectorized<U> b) const {
    // 将当前向量和参数向量按16位和32位解包
    auto ret16 = unpack(_vec);
    auto ret16B = unpack(b.vec());
    auto ret32_0 = unpack(ret16.first);
    auto ret32_1 = unpack(ret16.second);
    auto ret32B_0 = unpack(ret16B.first);
    auto ret32B_1 = unpack(ret16B.second);

    // 返回四个宽化减法结果的整数向量
    return {
        Vectorized<c10::qint32>(ret32_0.first - ret32B_0.first),
        Vectorized<c10::qint32>(ret32_0.second - ret32B_0.second),
        Vectorized<c10::qint32>(ret32_1.first - ret32B_1.first),
        Vectorized<c10::qint32>(ret32_1.second - ret32B_1.second)};
  }

  // 对四个浮点数向量进行反量化操作
  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 4, int> = 0>
  float_vec_return_type C10_ALWAYS_INLINE dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
    // 将当前向量按16位解包
    auto ret16 = unpack(_vec);
    auto ret32_0 = unpack(ret16.first);
    auto ret32_1 = unpack(ret16.second);

    // 将每个32位整数向量转换为浮点数向量
    auto vecf_0 = zvec_convert_to_float(ret32_0.first);
    auto vecf_1 = zvec_convert_to_float(ret32_0.second);
    auto vecf_2 = zvec_convert_to_float(ret32_1.first);
    auto vecf_3 = zvec_convert_to_float(ret32_1.second);
    return {
        // 计算向量运算：scale * vecf_0 + scale_zp_premul
        fmadd(scale, vecf_0, scale_zp_premul),
        // 计算向量运算：scale * vecf_1 + scale_zp_premul
        fmadd(scale, vecf_1, scale_zp_premul),
        // 计算向量运算：scale * vecf_2 + scale_zp_premul
        fmadd(scale, vecf_2, scale_zp_premul),
        // 计算向量运算：scale * vecf_3 + scale_zp_premul
        fmadd(scale, vecf_3, scale_zp_premul)};
  }

  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 4, int> = 0>
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
    // 将无符号整数解包为有符号整数
    auto ret16 = unpack(_vec);
    // 进一步解包为两个 32 位整数向量
    auto ret32_0 = unpack(ret16.first);
    auto ret32_1 = unpack(ret16.second);

    // 将解包后的整数向量转换为浮点数向量
    auto vecf_0 = zvec_convert_to_float(ret32_0.first);
    auto vecf_1 = zvec_convert_to_float(ret32_0.second);
    auto vecf_2 = zvec_convert_to_float(ret32_1.first);
    auto vecf_3 = zvec_convert_to_float(ret32_1.second);

    // 返回四个浮点数向量进行零点和缩放的解量化结果
    return {
        (vecf_0 - zero_point) * scale,
        (vecf_1 - zero_point) * scale,
        (vecf_2 - zero_point) * scale,
        (vecf_3 - zero_point) * scale };
  }

  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 4, int> = 0>
  static Vectorized<T> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    // 创建一个包含逆缩放因子的浮点数向量
    auto vec_inverse = Vectorized<float>(inverse_scale);
    // 创建一个包含零点的浮点数向量
    auto vec_zero_point = Vectorized<float>((float)zero_point);

    // 分别获取输入向量中的四个浮点数元素
    auto vecf0 = rhs[0];
    auto vecf2 = rhs[1];
    auto vecf4 = rhs[2];
    auto vecf6 = rhs[3];

    // 对每个浮点数元素乘以逆缩放因子
    vecf0 = vecf0 * vec_inverse;
    vecf2 = vecf2 * vec_inverse;
    vecf4 = vecf4 * vec_inverse;
    vecf6 = vecf6 * vec_inverse;

    // 四舍五入到最接近的整数，并加上零点
    vecf0 = vecf0.rint() + vec_zero_point;
    vecf2 = vecf2.rint() + vec_zero_point;
    vecf4 = vecf4.rint() + vec_zero_point;
    vecf6 = vecf6.rint() + vec_zero_point;

    // 将浮点数向量转换为整数向量
    auto veci0 = zvec_convert_to_int(vecf0);
    auto veci2 = zvec_convert_to_int(vecf2);
    auto veci4 = zvec_convert_to_int(vecf4);
    auto veci6 = zvec_convert_to_int(vecf6);

    // 将两个整数向量打包成一个带有给定模板参数的向量
    auto vecshi0 = pack(veci0, veci2);
    auto vecshi2 = pack(veci4, veci6);
    auto ret = pack<int16_t, typename U::underlying>(vecshi0, vecshi2);

    // 返回带有模板参数 T 的向量化对象
    return Vectorized<T>{ret};
  }

  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::int_num_vecs() == 4, int> = 0>
  static Vectorized<U> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    // 创建包含倍增因子的浮点数向量
    Vectorized<float> vec_multiplier = Vectorized<float>(multiplier);
    // 创建包含零点的整数向量
    Vectorized<int32_t> vec_zero_point = Vectorized<int32_t>(zero_point);

    // 解包输入向量中的四个整数向量
    Vectorized<c10::qint32> vi0 = inp[0];
    Vectorized<c10::qint32> vi1 = inp[1];
    Vectorized<c10::qint32> vi2 = inp[2];
    Vectorized<c10::qint32> vi3 = inp[3];

    // 将整数向量转换为浮点数向量
    auto vecf0 = zvec_convert_to_float(vi0.vec());
    auto vecf2 = zvec_convert_to_float(vi1.vec());
    auto vecf4 = zvec_convert_to_float(vi2.vec());
    auto vecf6 = zvec_convert_to_float(vi3.vec());

    // 每个浮点数向量乘以倍增因子
    vecf0 = vecf0 * vec_multiplier;
    vecf2 = vecf2 * vec_multiplier;
    vecf4 = vecf4 * vec_multiplier;
    // 将 vecf6 向量乘以 vec_multiplier 中的每个元素
    vecf6 = vecf6 * vec_multiplier;

    // 将 vecf0、vecf2、vecf4、vecf6 中的每个元素四舍五入到最接近的整数
    vecf0 = vecf0.rint();
    vecf2 = vecf2.rint();
    vecf4 = vecf4.rint();
    vecf6 = vecf6.rint();

    // 将 vecf0、vecf2、vecf4、vecf6 向量中的浮点数元素转换为整数元素
    auto veci0 = zvec_convert_to_int(vecf0);
    auto veci2 = zvec_convert_to_int(vecf2);
    auto veci4 = zvec_convert_to_int(vecf4);
    auto veci6 = zvec_convert_to_int(vecf6);

    // 将 veci0、veci2、veci4、veci6 向量中的每个元素加上 vec_zero_point 向量中对应位置的元素
    veci0 = veci0 + vec_zero_point;
    veci2 = veci2 + vec_zero_point;
    veci4 = veci4 + vec_zero_point;
    veci6 = veci6 + vec_zero_point;

    // 将 veci0 和 veci2 向量打包为一个包含 int32_t 和 int16_t 的结构体
    auto vecshi0 = pack<int32_t, int16_t>(veci0, veci2);
    // 将 veci4 和 veci6 向量打包为一个包含 int32_t 和 int16_t 的结构体
    auto vecshi2 = pack<int32_t, int16_t>(veci4, veci6);

    // 将 vecshi0 和 vecshi2 向量打包为一个包含 int16_t 和 U 类型的结构体，返回结果
    auto ret = pack<int16_t, typename U::underlying>(vecshi0, vecshi2);

    // 返回一个新的 Vectorized<U> 对象，其内部向量为 ret
    return Vectorized<U>{ret};
  }

  // 比较当前 Vectorized 对象与另一个 Vectorized 对象 other 是否相等，返回结果
  Vectorized<T> C10_ALWAYS_INLINE eq(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.eq(other._vec)};
  }

  // 比较当前 Vectorized 对象与另一个 Vectorized 对象 other 是否不相等，返回结果
  Vectorized<T> C10_ALWAYS_INLINE ne(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.ne(other._vec)};
  }

  // 比较当前 Vectorized 对象是否大于另一个 Vectorized 对象 other，返回结果
  Vectorized<T> C10_ALWAYS_INLINE gt(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.gt(other._vec)};
  }

  // 比较当前 Vectorized 对象是否大于等于另一个 Vectorized 对象 other，返回结果
  Vectorized<T> C10_ALWAYS_INLINE ge(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.ge(other._vec)};
  }

  // 比较当前 Vectorized 对象是否小于另一个 Vectorized 对象 other，返回结果
  Vectorized<T> C10_ALWAYS_INLINE lt(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.lt(other._vec)};
  }

  // 比较当前 Vectorized 对象是否小于等于另一个 Vectorized 对象 other，返回结果
  Vectorized<T> C10_ALWAYS_INLINE le(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.le(other._vec)};
  }

  // 将当前 Vectorized 对象中的元素与给定的最小值向量 min 中的对应元素逐个比较，返回比较结果
  Vectorized<T> clamp_min(const Vectorized<T>& min) const {
    return Vectorized<T>{_vec.clamp_min(min._vec)};
  }

  // 将当前 Vectorized 对象中的元素与给定的最大值向量 max 中的对应元素逐个比较，返回比较结果
  Vectorized<T> clamp_max(const Vectorized<T>& max) const {
    return Vectorized<T>{_vec.clamp_max(max._vec)};
  }

  // 返回当前 Vectorized 对象中的元素与另一个 Vectorized 对象 other 中对应元素的最小值
  Vectorized<T> minimum(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.minimum(other._vec)};
  }

  // 返回当前 Vectorized 对象中的元素与另一个 Vectorized 对象 other 中对应元素的最大值
  Vectorized<T> maximum(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.maximum(other._vec)};
  }
};

#define ZVECTOR_OPERATORS(typex)                                                                           \
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator+(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    return Vectorized<typex>{a.vec() + b.vec()};                                                           \
  }                                                                                                        \
                                                                                                           \
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator-(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    return Vectorized<typex>{a.vec() - b.vec()};                                                           \
  }                                                                                                        \
                                                                                                           \
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator*(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    return Vectorized<typex>{a.vec() * b.vec()};                                                           \
  }                                                                                                        \
                                                                                                           \
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator/(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    return Vectorized<typex>{a.vec() / b.vec()};                                                           \
  }                                                                                                        \
                                                                                                           \
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator&(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    // 定义按位与运算符重载，对应类型为 Vectorized<typex>，操作数为 a 和 b，返回值为对应元素的按位与结果的 Vectorized<typex> 对象
    return Vectorized<typex>{a.vec() & b.vec()};                                                           \
  }
    # 返回按位与的向量化操作结果
    return Vectorized<typex>{a.vec() & b.vec()};
    
    
    
    # 返回按位或的向量化操作结果
    return Vectorized<typex>{a.vec() | b.vec()};
    
    
    
    # 返回按位异或的向量化操作结果
    return Vectorized<typex>{a.vec() ^ b.vec()};
    
    
    
    # 返回按位等于的向量化操作结果
    return Vectorized<typex>{a.vec() == b.vec()};
    
    
    
    # 返回按位不等于的向量化操作结果
    return Vectorized<typex>{a.vec() != b.vec()};
    
    
    
    # 返回向量中每个元素是否大于的向量化操作结果
    return Vectorized<typex>{a.vec() > b.vec()};
    
    
    
    # 返回向量中每个元素是否大于或等于的向量化操作结果
    return Vectorized<typex>{a.vec() >= b.vec()};
    // 定义重载运算符>=，比较两个向量化类型的对象的每个元素，返回每个元素对应比较的结果向量
    return Vectorized<typex>{a.vec() >= b.vec()};                                                          \
    // 定义重载运算符<，比较两个向量化类型的对象的每个元素，返回每个元素对应比较的结果向量
    }                                                                                                        \
                                                                                                               \
    // 定义重载运算符<=，比较两个向量化类型的对象的每个元素，返回每个元素对应比较的结果向量
    Vectorized<typex> C10_ALWAYS_INLINE operator<(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    // 返回一个新的向量化类型对象，包含每个元素对应的小于比较结果向量
    return Vectorized<typex>{a.vec() < b.vec()};                                                           \
    }                                                                                                        \
                                                                                                               \
    // 定义重载运算符<=，比较两个向量化类型的对象的每个元素，返回每个元素对应比较的结果向量
    Vectorized<typex> C10_ALWAYS_INLINE operator<=(const Vectorized<typex>& a, const Vectorized<typex>& b) { \
    // 返回一个新的向量化类型对象，包含每个元素对应的小于等于比较结果向量
    return Vectorized<typex>{a.vec() <= b.vec()};                                                          \
    }
template <typename U = float>
constexpr auto real_mask() {
  return (ZSimdVect<U>)ZSimdVectBinary<float>{0xFFFFFFFF, 0, 0xFFFFFFFF, 0};
}



// 返回一个适用于模板类型 U 的实部掩码，以 ZSimdVect 封装 SIMD 向量类型
template <>
constexpr auto real_mask<double>() {
  return (ZSimdVect<double>)ZSimdVectBinary<double>{0xFFFFFFFFFFFFFFFF, 0};
}



template <typename U = float>
constexpr auto image_mask() {
  return (ZSimdVect<U>)ZSimdVectBinary<U>{0, 0xFFFFFFFF, 0, 0xFFFFFFFF};
}



// 返回一个适用于模板类型 U 的虚部掩码，以 ZSimdVect 封装 SIMD 向量类型
template <>
constexpr auto image_mask<double>() {
  return (ZSimdVect<double>)ZSimdVectBinary<double>{0, 0xFFFFFFFFFFFFFFFF};
}



template <typename U = float>
constexpr auto rsign_mask() {
  return ZSimdVect<U>{-0.f, 0.f, -0.f, 0.f};
}



// 返回一个适用于模板类型 U 的实部符号掩码，以 ZSimdVect 封装 SIMD 向量类型
template <>
constexpr auto rsign_mask<double>() {
  return ZSimdVect<double>{-0.0, 0.f};
}



template <typename U = float>
constexpr auto isign_mask() {
  return ZSimdVect<U>{0.0, -0.f, 0.0, -0.f};
}



// 返回一个适用于模板类型 U 的虚部符号掩码，以 ZSimdVect 封装 SIMD 向量类型
template <>
constexpr auto isign_mask<double>() {
  return ZSimdVect<double>{0.0, -0.0};
}



template <typename U = float>
constexpr auto image_one() {
  return ZSimdVect<U>{0, 1.f, 0, 1.f};
}



// 返回一个适用于模板类型 U 的虚部为 1 的掩码，以 ZSimdVect 封装 SIMD 向量类型
template <>
constexpr auto image_one<double>() {
  return ZSimdVect<double>{0.0, 1.0};
}



template <typename U = float>
constexpr auto pi_half() {
  return ZSimdVect<U>{(float)(M_PI / 2.0), 0.f, (float)(M_PI / 2.0), 0.f};
}



// 返回一个适用于模板类型 U 的π/2 的掩码，以 ZSimdVect 封装 SIMD 向量类型
template <>
constexpr auto pi_half<double>() {
  return ZSimdVect<double>{M_PI / 2.0, 0.0};
}



template <typename U = float>
constexpr auto image_half() {
  return ZSimdVect<U>{0, 0.5f, 0, 0.5f};
}



// 返回一个适用于模板类型 U 的虚部为 0.5 的掩码，以 ZSimdVect 封装 SIMD 向量类型
template <>
constexpr auto image_half<double>() {
  return ZSimdVect<double>{0.0, 0.5};
}



template <typename U>
constexpr U log2e_inv() {
  return static_cast<U>(1.4426950408889634);
}



// 返回对数 2 的倒数的适用于模板类型 U 的值
template <typename U>
constexpr U log10e_inv() {
  return static_cast<U>(0.43429448190325176);
}



template <typename T>
struct Vectorized<T, std::enable_if_t<is_zarch_implemented_complex<T>()>> {
 public:
  using underline_type = decltype(std::declval<T>().imag());
  using value_type = T;
  using vtype = ZSimdVect<underline_type>;
  using vmaskType = ZSimdVectBinary<underline_type>;
  using vinner_type = Vectorized<underline_type>;
  using size_type = int;
  using vinner_data = typename Vectorized<underline_type>::vinner_data;

  static constexpr size_type size() {



// 用于模板类型 T，当 is_zarch_implemented_complex<T>() 为 true 时，定义 Vectorized 结构体
template <typename T>
struct Vectorized<T, std::enable_if_t<is_zarch_implemented_complex<T>()>> {
 public:
  // T 类型的虚部类型
  using underline_type = decltype(std::declval<T>().imag());
  using value_type = T;
  using vtype = ZSimdVect<underline_type>;
  using vmaskType = ZSimdVectBinary<underline_type>;
  using vinner_type = Vectorized<underline_type>;
  using size_type = int;
  // Vectorized<underline_type> 的内部数据类型
  using vinner_data = typename Vectorized<underline_type>::vinner_data;

  // 返回 size_type 类型的常量，代表向量的大小
  static constexpr size_type size() {
  return VECTOR_WIDTH / sizeof(value_type);
}

private:
vinner_type _vec;

public:
Vectorized() {}

// 构造函数，使用给定的 vinner_data 初始化 _vec
C10_ALWAYS_INLINE Vectorized(const vinner_data &v) : _vec{v.first, v.second} {}

// 如果 T 的大小为 16 字节，则使用 s1 和 s2 初始化 _vec
template <typename U = T, std::enable_if_t<(sizeof(U) == 16), int> = 0>
C10_ALWAYS_INLINE Vectorized(T s1, T s2)
    : _vec{s1.real(), s1.imag(), s2.real(), s2.imag()} {}

// 如果 T 的大小为 8 字节，则使用 s1, s2, s3, s4 初始化 _vec
template <typename U = T, std::enable_if_t<(sizeof(U) == 8), int> = 0>
C10_ALWAYS_INLINE Vectorized(T s1, T s2, T s3, T s4)
    : _vec{
          s1.real(),
          s1.imag(),
          s2.real(),
          s2.imag(),
          s3.real(),
          s3.imag(),
          s4.real(),
          s4.imag()} {}

// 如果 T 的大小为 16 字节，则使用 s 初始化 _vec
template <typename U = T, std::enable_if_t<(sizeof(U) == 16), int> = 0>
C10_ALWAYS_INLINE Vectorized(T s) : Vectorized<T>(s, s) {}

// 如果 T 的大小为 8 字节，则使用 s 初始化 _vec
template <typename U = T, std::enable_if_t<(sizeof(U) == 8), int> = 0>
C10_ALWAYS_INLINE Vectorized(T s) : Vectorized<T>(s, s, s, s) {}

// 将 _vec 转换为 vinner_type 类型并返回
C10_ALWAYS_INLINE operator vinner_type() const {
  return _vec;
}

// 返回 _vec 的常量引用
C10_ALWAYS_INLINE const vinner_type& vec() const {
  return _vec;
}

// 将 _vec 转换为 vinner_data 类型并返回
C10_ALWAYS_INLINE operator vinner_data() const {
  return _vec.data();
}

// 返回 _vec 的数据成员 vinner_data
C10_ALWAYS_INLINE vinner_data data() const {
  return _vec.data();
}

// 从指针 ptr 加载数据并返回 Vectorized<T> 对象
template <typename U>
static Vectorized<T> C10_ALWAYS_INLINE
loadu(const U* ptr, int count = size()) {
  return Vectorized<T>{vinner_type::loadu(ptr, 2 * count)};
}

// 将 _vec 的数据存储到 ptr 指向的内存中
template <typename U>
void C10_ALWAYS_INLINE store(U* ptr, int count = size()) const {
  return _vec.store(ptr, 2 * count);
}

// 使用 blendv 操作将 a 和 b 按照 mask 进行混合
static Vectorized<T> blendv(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& mask) {
  // 将复杂类型 std::complex<V> 的掩码转换为 V 类型的掩码：xy -> xxyy
  vinner_type vmask = mask.vec();
  auto mask_complex = vinner_type(
      vec_mergeh(vmask.vec0(), vmask.vec0()),
      vec_mergeh(vmask.vec1(), vmask.vec1()));
  return Vectorized<T>{vinner_type::blendv(a.vec(), b.vec(), mask_complex)};
}

// 使用 blend 操作将 a 和 b 按照 mask 进行混合
template <int64_t mask>
static auto C10_ALWAYS_INLINE
blend(const Vectorized<T>& a, const Vectorized<T>& b) {
  constexpr int mask_complex = maskForComplex<sizeof(T)>(mask);
  return Vectorized<T>{
      vinner_type::template blend<mask_complex>(a.vec(), b.vec())};
}

// 生成一个范围为 base 到 base+step 的序列，步长为 step_t 类型
template <typename step_t, typename U = T>
static std::enable_if_t<sizeof(U) == 16, Vectorized<T>> arange(
    T base = 0,
    step_t step = static_cast<step_t>(1)) {
  return Vectorized<T>(base, base + step);
}

// 生成一个范围为 base 到 base+step 的序列，步长为 step_t 类型
template <typename step_t, typename U = T>
static std::enable_if_t<sizeof(U) == 8, Vectorized<T>> arange(
    T base = 0,
    step_t step = static_cast<step_t>(1)) {
  template <int16_t Z, int16_t C>
  static inline std::enable_if_t<(Z >= C), Vectorized<T>> set_inner(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      size_t count) {
    // 如果 Z 大于等于 C，返回 b
    return b;
  }

  template <int16_t Z, int16_t C>
  static inline std::enable_if_t<(Z < C), Vectorized<T>> set_inner(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      size_t count) {
    // 如果 Z 小于 C，递归调用 set_inner 增加 Z 直到 Z >= C，然后根据 count 返回 blend 或者 b
    if (count == Z)
      return blend<allbitset(Z)>(a, b);  // 使用 blend 函数根据 allbitset(Z) 来混合 a 和 b
    else
      return set_inner<Z + 1, C>(a, b, count);  // 递归调用 set_inner，增加 Z 直到 Z >= C
  }

  static Vectorized<T> set(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      size_t count = size()) {
    // 如果 count 等于 0，返回 a
    if (count == 0)
      return a;
    // 调用 set_inner 开始递归设置向量中的元素
    return set_inner<1, size()>(a, b, count);
  }

  const T& operator[](int idx) const = delete;  // 删除 const 版本的数组下标操作符

  T& operator[](int idx) = delete;  // 删除非 const 版本的数组下标操作符

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, c10::complex<float>>::value, int> = 0>
  Vectorized<T> mapOrdinary(T (*const f)(const T&)) const {
    // 如果 T 是 c10::complex<float>，对向量的每个元素应用函数 f，并返回结果向量
    auto v0 = _vec.vec0();
    auto v1 = _vec.vec1();
    return Vectorized<T>{
        f(T(v0[0], v0[1])),
        f(T(v0[2], v0[3])),
        f(T(v1[0], v1[1])),
        f(T(v1[2], v1[3]))};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, c10::complex<double>>::value, int> = 0>
  Vectorized<U> mapOrdinary(T (*const f)(const T&)) const {
    // 如果 T 是 c10::complex<double>，对向量的每个元素应用函数 f，并返回结果向量
    auto v0 = _vec.vec0();
    auto v1 = _vec.vec1();
    return Vectorized<T>{f(T(v0[0], v0[1])), f(T(v1[0], v1[1]))};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, c10::complex<float>>::value, int> = 0>
  Vectorized<T> mapOrdinary(T (*const f)(T)) const {
    // 如果 T 是 c10::complex<float>，对向量的每个元素应用函数 f，并返回结果向量
    auto v0 = _vec.vec0();
    auto v1 = _vec.vec1();
    return Vectorized<T>{
        f(T(v0[0], v0[1])),
        f(T(v0[2], v0[3])),
        f(T(v1[0], v1[1])),
        f(T(v1[2], v1[3]))};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, c10::complex<double>>::value, int> = 0>
  Vectorized<T> mapOrdinary(T (*const f)(T)) const {
    // 如果 T 是 c10::complex<double>，对向量的每个元素应用函数 f，并返回结果向量
    auto v0 = _vec.vec0();
    auto v1 = _vec.vec1();
    return Vectorized<T>{f(T(v0[0], v0[1])), f(T(v1[0], v1[1]))};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, c10::complex<float>>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(
      T (*const f)(const T&, const T&),
      const Vectorized<T>& b) const {
    // 如果 T 是 c10::complex<float>，对向量的每个元素与向量 b 的对应元素应用函数 f，并返回结果向量
    auto v0 = _vec.vec0();
    auto v1 = _vec.vec1();
    auto bvec = b.vec();
    auto b0 = bvec.vec0();
    auto b1 = bvec.vec1();
    T a00 = f(T(v0[0], v0[1]), T(b0[0], b0[1]));
    T a01 = f(T(v0[2], v0[3]), T(b0[2], b0[3]));
    T a02 = f(T(v1[0], v1[1]), T(b1[0], b1[1]));
    T a03 = f(T(v1[2], v1[3]), T(b1[2], b1[3]));
    // 返回应用函数 f 后的结果向量
    return Vectorized<T>{a00, a01, a02, a03};
  }
  Vectorized<T> angle2_() const {
    // 获取向量 _vec 的交换部分，并计算其反正切角度
    auto b_a = _vec.swapped(); // b        a
    // 返回结果，其中角度被转换为实部
    return Vectorized<T>{_vec.atan2(b_a).swapped()};
  }

  Vectorized<T> angle() const {
    // 调用 angle2_() 方法获取角度向量，并返回其实部
    return angle2_().real();
  }

  Vectorized<T> atan() const {
    // 计算 arctangent 函数值，使用公式 atan(x) = i/2 * ln((i + z)/(i - z))
    auto ione = Vectorized<T>{vinner_type(image_one<underline_type>())}; // 定义复数单位向量 i
    auto sum = ione + *this; // 计算 i + z
    auto sub = ione - *this; // 计算 i - z
    auto ln = (sum / sub).log(); // 计算 ln((i + z)/(i - z))
    // 返回 arctangent 的结果，乘以 i/2
    return ln *
        Vectorized<T>{vinner_type(image_half<underline_type>())}; // i/2*ln()
  }

  Vectorized<T> atanh() const {
    // 调用 mapOrdinary 函数，传递 std::atanh 作为参数，计算双参数函数
    return mapOrdinary(std::atanh);
  }

  Vectorized<T> asin() const {
    // 计算 arcsine 函数值，使用公式 asin(x) = -i * ln(iz + sqrt(1 - z^2))
    auto ione = Vectorized<T>{vinner_type(image_one<underline_type>())}; // 定义复数单位向量 i
    auto z_squared = *this * *this; // 计算 z 的平方
    auto sqrt_term = (ione - z_squared).sqrt(); // 计算 sqrt(1 - z^2)
    auto ln_arg = (ione * *this - sqrt_term).log(); // 计算 ln(iz + sqrt(1 - z^2))
    // 返回 arcsine 的结果，乘以 -i
    return -ione * ln_arg;
  }
#if 1
    // 计算共轭向量 cnj
    vinner_type cnj = conj().vec();
    // 计算交换后的向量 b_a
    vinner_type b_a = cnj.swapped();
    // 计算 ab = cnj * b_a
    vinner_type ab = cnj * b_a;
    // 计算 im = ab + ab
    vinner_type im = ab + ab;
    // 计算 val_2 = _vec * _vec
    vinner_type val_2 = _vec * _vec;
    // 计算交换后的 val_2_swapped
    vinner_type val_2_swapped = val_2.swapped();
    // 计算 re = horizontal_sub_perm(val_2, val_2_swapped)
    vinner_type re = vinner_type::horizontal_sub_perm(val_2, val_2_swapped);
    // re = 1 - re
    re = vinner_type(static_cast<underline_type>(1)) - re;
    // 根据 blend_choice<T>() 生成 blend_mask
    constexpr int blend_mask = blend_choice<T>(); // 0x0A for complex<double> , 0xAA for complex<float>
    // 使用 blend_mask 合并 re 和 im
    vinner_type blendx = vinner_type::template blend<blend_mask>(re, im);
    // 对 blendx 执行平方根操作
    auto root = Vectorized<T>(blendx).sqrt();
    // 计算 ln = log(b_a + root)
    auto ln = Vectorized<T>(Vectorized<T>(b_a) + root).log();
    // 返回 ln 向量的共轭
    return Vectorized<T>(ln.vec().swapped()).conj();
#else
    // 如果条件不满足，返回 mapOrdinary(std::asin)
    return mapOrdinary(std::asin);
#endif
  }
  // 返回一个新的向量，其中每个元素是当前向量与给定向量的按位异或结果
  return Vectorized<T>(_vec ^ vinner_type(isign_mask<underline_type>()));

  // 计算当前向量每个元素的平方，然后将每个元素与其对应位置的元素相加，最后返回结果的数据部分
  vinner_data abs_2_() const {
    auto a = _vec * _vec;
    a = a + a.swapped();
    return a.mergee().data();
  }

  // 静态辅助函数，返回给定值的绝对值
  static T abs_helper(const T &value)
  {
    return T(std::abs(value));
  }

  // 返回一个新的向量，其中每个元素是当前向量中每个元素的绝对值
  Vectorized<T> abs() const {
    return mapOrdinary(abs_helper);
  }

  // 返回一个新的向量，其中每个元素是当前向量中每个元素的指数值
  Vectorized<T> exp() const {
    return mapOrdinary(std::exp);
  }

  // 返回一个新的向量，其中每个元素是当前向量中每个元素的2的指数值
  Vectorized<T> exp2() const {
    return mapOrdinary(exp2_impl);
  }

  // 返回一个新的向量，其中每个元素是当前向量中每个元素的指数减1的值
  Vectorized<T> expm1() const {
    return mapOrdinary(std::expm1);
  }

  // 返回一个新的向量，其中每个元素是当前向量中每个元素的自然对数值
  Vectorized<T> log() const {
    return mapOrdinary(std::log);
  }

  // 返回一个新的向量，其中每个元素是当前向量中每个元素的以2为底的对数值
  Vectorized<T> log2() const {
    // log2eB_inv
    auto ret = log();
    return Vectorized<T>{ret._vec * vinner_type(log2e_inv<underline_type>())};
  }

  // 返回一个新的向量，其中每个元素是当前向量中每个元素的以10为底的对数值
  Vectorized<T> log10() const {
    auto ret = log();
    return Vectorized<T>{ret._vec * vinner_type(log10e_inv<underline_type>())};
  }

  // 返回一个新的向量，其中每个元素是当前向量中每个元素的自然对数减1的值
  Vectorized<T> log1p() const {
    return mapOrdinary(std::log1p);
  }

  // 返回一个新的向量，其中每个元素是当前向量中每个元素的符号值
  Vectorized<T> sgn() const {
    return mapOrdinary(at::native::sgn_impl);
  }

  // 返回一个新的向量，其中每个元素是当前向量中每个元素与给定向量按位取幂后的结果
  Vectorized<T> pow(const Vectorized<T>& exp) const {
    return mapOrdinary(std::pow, exp);
  }

  // 返回一个新的向量，其中每个元素是当前向量中每个元素的平方根值
  Vectorized<T> sqrt() const {
    return mapOrdinary(std::sqrt);
  }

  // 返回一个新的向量，其中每个元素是当前向量中每个元素的倒数
  Vectorized<T> reciprocal() const {
    // re + im*i = (a + bi)  / (c + di)
    // re = (ac + bd)/abs_2() = c/abs_2()
    // im = (bc - ad)/abs_2() = d/abs_2()
    vinner_type c_d = _vec ^ vinner_type(isign_mask<underline_type>());
    vinner_type abs = abs_2_();
    return Vectorized<T>{c_d / abs};
  }

  // 返回一个新的向量，其中每个元素是当前向量中每个元素的平方根的倒数
  Vectorized<T> rsqrt() const {
    return sqrt().reciprocal();
  }

  // 抛出异常，表示不支持复数类型的比较操作
  Vectorized<T> lt(const Vectorized<T>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  // 抛出异常，表示不支持复数类型的比较操作
  Vectorized<T> le(const Vectorized<T>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  // 抛出异常，表示不支持复数类型的比较操作
  Vectorized<T> gt(const Vectorized<T>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  // 抛出异常，表示不支持复数类型的比较操作
  Vectorized<T> ge(const Vectorized<T>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
};

#define ZVECTOR_OPERATORS(typex)                                                                           \
  // 定义模板函数，实现向量化类型的加法操作
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator+(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    return Vectorized<typex>{a.vec() + b.vec()};                                                           \
  }                                                                                                        \
                                                                                                           \
  // 定义模板函数，实现向量化类型的减法操作
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator-(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    return Vectorized<typex>{a.vec() - b.vec()};                                                           \
  }                                                                                                        \
                                                                                                           \
  // 定义模板函数，实现向量化类型的乘法操作
  template <>                                                                                              \
  Vectorized<typex> inline operator*(const Vectorized<typex>& a, const Vectorized<typex>& b) {             \
    // 进行复数乘法运算，计算公式为 (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    Vectorized<typex>::vinner_type bv = b.vec();                                                           \
                                                                                                           \
    // 根据 Z 架构的友好性，使用 mergeo 和 mergee 方法而非在 x86 上模拟 horizontal
    Vectorized<typex>::vinner_type vi = bv.mergeo();                                                       \
    Vectorized<typex>::vinner_type vr = bv.mergee();                                                       \
    vi = vi ^ Vectorized<typex>::vinner_type(rsign_mask<Vectorized<typex>::underline_type>());             \
    Vectorized<typex>::vinner_type ret = a.vec() * vr;                                                     \
    Vectorized<typex>::vinner_type vx_swapped = a.vec().swapped();                                         \
    ret = fmadd(vx_swapped, vi, ret);                                                                      \
                                                                                                           \
    return Vectorized<typex>{ret};                                                                         \
  }                                                                                                        \
                                                                                                           \
  template <>                                                                                              \
  Vectorized<typex> inline operator/(const Vectorized<typex>& a, const Vectorized<typex>& b) {             \
    /* Unfortunately, this breaks some tests */                                                            \
    /* Implement it like it's done for avx2 */                                                             \
    // 计算 b 的绝对值
    auto fabs_cd = b.vec().abs();                               /* |c|    |d| */                           \
    // 交换 b 的两个元素
    auto fabs_dc = fabs_cd.swapped();                           /* |d|    |c| */                           \
    // 计算缩放因子，确保 c 和 d 的最大值非零
    auto scale = Vectorized<typex>::vinner_type {1.0} / maximum(fabs_cd, fabs_dc); /* 1/sc     1/sc */     \
    // 对 a 和 b 应用缩放因子
    auto a2 = a.vec() * scale;                                  /* a/sc     b/sc */                        \
    auto b2 = b.vec() * scale;                                  /* c/sc     d/sc */                        \
    // 计算 ac 和 bd 的乘积
    auto acbd2 = a2 * b2;                                       /* ac/sc^2  bd/sc^2 */                     \
                                                                                                           \
    // 交换 b2 的两个元素
    auto dc2 = b2.swapped();                                    /* d/sc         c/sc */                    \
    // 对 dc2 中的元素取负值
    dc2 = Vectorized<typex>::real_neg(dc2);                     /* -d/|c,d|        c/sc */                 \
    // 计算 -ad 和 bc 的乘积
    auto adbc2 = a2 * dc2;                                      /* -ad/sc^2      bc/sc^2 */                \
    // 计算 ac+bd 和 bc-ad 的和
    auto sum1 = acbd2 + acbd2.swapped();                        /* (ac+bd)/sc^2  (ac+bd)/sc^2 */           \
    auto sum2 = adbc2 + adbc2.swapped();                        /* (bc-ad)/sc^2  (bc-ad)/sc^2 */           \
    // 合并结果 sum1 和 sum2
    auto res2 = Vectorized<typex>::vinner_type::mergee(sum1, sum2);  /* (ac+bd)/sc^2  (bc-ad)/sc^2 */      \
                                                                                                           \
    // 计算分母
    Vectorized<typex>::vinner_type denom2 = Vectorized<typex>{b2}.abs_2_(); /* (c^2+d^2)/sc^2   (c^2+d^2)/sc^2 */ \
    // 对 res2 应用分母
    res2 = res2 / denom2;                                                                                  \
  return Vectorized<typex>{ res2 };                                                                      \
```  
返回一个 `Vectorized<typex>` 对象，该对象使用变量 `res2` 初始化。


  }                                                                                                        \
```py  
结束函数定义。


                                                                                                           \
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator&(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
```  
定义模板特化的按位与（&）操作符重载函数，其返回一个 `Vectorized<typex>` 对象。


    return Vectorized<typex>{a.vec() & b.vec()};                                                           \
```py  
返回一个 `Vectorized<typex>` 对象，它包含 `a` 和 `b` 向量执行按位与操作的结果。


  }                                                                                                        \
```  
结束函数定义。


                                                                                                           \
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator|(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
```py  
定义模板特化的按位或（|）操作符重载函数，其返回一个 `Vectorized<typex>` 对象。


    return Vectorized<typex>{a.vec() | b.vec()};                                                           \
```  
返回一个 `Vectorized<typex>` 对象，它包含 `a` 和 `b` 向量执行按位或操作的结果。


  }                                                                                                        \
```py  
结束函数定义。


                                                                                                           \
  template <>                                                                                              \
  Vectorized<typex> C10_ALWAYS_INLINE operator^(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
```  
定义模板特化的按位异或（^）操作符重载函数，其返回一个 `Vectorized<typex>` 对象。


    return Vectorized<typex>{a.vec() ^ b.vec()};                                                           \
```py  
返回一个 `Vectorized<typex>` 对象，它包含 `a` 和 `b` 向量执行按位异或操作的结果。


  }                                                                                                        \
```  
结束函数定义。


                                                                                                           \
  Vectorized<typex> C10_ALWAYS_INLINE operator==(const Vectorized<typex>& a, const Vectorized<typex>& b) { \
```py  
定义重载的相等（==）操作符函数，其返回一个 `Vectorized<typex>` 对象。


    return Vectorized<typex>{a.vec() == b.vec()};                                                          \
```  
返回一个 `Vectorized<typex>` 对象，它包含 `a` 和 `b` 向量执行相等比较操作的结果。


  }                                                                                                        \
```py  
结束函数定义。


                                                                                                           \
  Vectorized<typex> C10_ALWAYS_INLINE operator!=(const Vectorized<typex>& a, const Vectorized<typex>& b) { \
```  
定义重载的不等（!=）操作符函数，其返回一个 `Vectorized<typex>` 对象。


    return Vectorized<typex>{a.vec() != b.vec()};                                                          \
```py  
返回一个 `Vectorized<typex>` 对象，它包含 `a` 和 `b` 向量执行不等比较操作的结果。


  }                                                                                                        \
```  
结束函数定义。


                                                                                                           \
  Vectorized<typex> C10_ALWAYS_INLINE operator<(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
```py  
定义重载的小于（<）操作符函数，其返回一个 `Vectorized<typex>` 对象。
    // 对于复数不支持此操作，触发运行时检查，并抛出错误信息
    TORCH_CHECK(false, "not supported for complex numbers");                                               \
  }                                                                                                        \
                                                                                                           \
  // 重载运算符<=，对于复数不支持此操作，触发运行时检查，并抛出错误信息
  Vectorized<typex> C10_ALWAYS_INLINE operator<=(const Vectorized<typex>& a, const Vectorized<typex>& b) { \
    TORCH_CHECK(false, "not supported for complex numbers");                                               \
  }                                                                                                        \
                                                                                                           \
  // 重载运算符>，对于复数不支持此操作，触发运行时检查，并抛出错误信息
  Vectorized<typex> C10_ALWAYS_INLINE operator>(const Vectorized<typex>& a, const Vectorized<typex>& b) {  \
    TORCH_CHECK(false, "not supported for complex numbers");                                               \
  }                                                                                                        \
                                                                                                           \
  // 重载运算符>=，对于复数不支持此操作，触发运行时检查，并抛出错误信息
  Vectorized<typex> C10_ALWAYS_INLINE operator>=(const Vectorized<typex>& a, const Vectorized<typex>& b) { \
    TORCH_CHECK(false, "not supported for complex numbers");                                               \
  }
// 定义宏 ZVECTOR_OPERATORS，用于生成复数向量操作的代码，针对单精度复数类型
ZVECTOR_OPERATORS(c10::complex<float>)

// 定义宏 ZVECTOR_OPERATORS，用于生成复数向量操作的代码，针对双精度复数类型
ZVECTOR_OPERATORS(c10::complex<double>)

// 取消宏 ZVECTOR_OPERATORS 的定义，确保后续代码不再使用该宏
#undef ZVECTOR_OPERATORS

// 如果模板参数 T 的大小为 8 字节，则定义函数 inner_interleave2
template <typename T, std::enable_if_t<(sizeof(T) == 8), int> = 0>
std::pair<Vectorized<T>, Vectorized<T>> inline inner_interleave2(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  // inputs:
  //   a      = {a0, a1, a2, a3}
  //   b      = {b0, b1, b2, b3}
  // 使用 Vectorized<T> 类型的 vtype 作为向量类型
  using vtype = typename Vectorized<T>::vtype;
  // 对 a.vec0() 和 b.vec0() 进行混合排列，选取索引为 0 和 3 的元素
  vtype ab00 = vec_permi(a.vec0(), b.vec0(), 0);
  vtype ab11 = vec_permi(a.vec0(), b.vec0(), 3);
  // 对 a.vec1() 和 b.vec1() 进行混合排列，选取索引为 0 和 3 的元素
  vtype ab2_00 = vec_permi(a.vec1(), b.vec1(), 0);
  vtype ab2_11 = vec_permi(a.vec1(), b.vec1(), 3);
  //   return {a0, b0, a1, b1}
  //          {a2, b2, a3, b3}
  // 返回混合排列后的结果作为一对 Vectorized<T> 对象
  return std::make_pair(
      Vectorized<T>{ab00, ab11}, Vectorized<T>{ab2_00, ab2_11});
}

// 如果模板参数 T 的大小为 8 字节，则定义函数 inner_deinterleave2
template <typename T, std::enable_if_t<(sizeof(T) == 8), int> = 0>
std::pair<Vectorized<T>, Vectorized<T>> inline inner_deinterleave2(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1}
  //   b = {a2, b2, a3, b3}
  // 使用 Vectorized<T> 类型的 vtype 作为向量类型
  using vtype = typename Vectorized<T>::vtype;
  // 对 a.vec0() 和 a.vec1() 进行混合排列，选取索引为 0 的元素
  vtype aa01 = vec_permi(a.vec0(), a.vec1(), 0);
  // 对 b.vec0() 和 b.vec1() 进行混合排列，选取索引为 0 的元素
  vtype aa23 = vec_permi(b.vec0(), b.vec1(), 0);

  // 对 a.vec0() 和 a.vec1() 进行混合排列，选取索引为 3 的元素
  vtype bb_01 = vec_permi(a.vec0(), a.vec1(), 3);
  // 对 b.vec0() 和 b.vec1() 进行混合排列，选取索引为 3 的元素
  vtype bb_23 = vec_permi(b.vec0(), b.vec1(), 3);

  // swap lanes:
  //   return {a0, a1, a2, a3}
  //          {b0, b1, b2, b3}
  // 返回交错排列后的结果作为一对 Vectorized<T> 对象
  return std::make_pair(Vectorized<T>{aa01, aa23}, Vectorized<T>{bb_01, bb_23});
}

// 如果模板参数 T 的大小为 4 字节，则定义函数 inner_interleave2
template <typename T, std::enable_if_t<(sizeof(T) == 4), int> = 0>
std::pair<Vectorized<T>, Vectorized<T>> inline inner_interleave2(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  // inputs:
  //   a = {a0, a1, a2, a3,, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3,, b4, b5, b6, b7}
  // 使用 Vectorized<T> 类型的 vtype 作为向量类型
  using vtype = typename Vectorized<T>::vtype;
  // 对 a.vec0() 和 b.vec0() 进行水平合并
  vtype ab0011 = vec_mergeh(a.vec0(), b.vec0());
  // 对 a.vec0() 和 b.vec0() 进行垂直合并
  vtype ab2233 = vec_mergel(a.vec0(), b.vec0());

  // 对 a.vec1() 和 b.vec1() 进行水平合并
  vtype ab2_0011 = vec_mergeh(a.vec1(), b.vec1());
  // 对 a.vec1() 和 b.vec1() 进行垂直合并
  vtype ab2_2233 = vec_mergel(a.vec1(), b.vec1());
  // group cols crossing lanes:
  //   return {a0, b0, a1, b1,, a2, b2, a3, b3}
  //          {a4, b4, a5, b5,, a6, b6, a7, b7}
  // 返回交叉排列后的结果作为一对 Vectorized<T> 对象
  return std::make_pair(
      Vectorized<T>{ab0011, ab2233}, Vectorized<T>{ab2_0011, ab2_2233});
}

// 如果模板参数 T 的大小为 4 字节，则定义函数 inner_deinterleave2
template <typename T, std::enable_if_t<(sizeof(T) == 4), int> = 0>
std::pair<Vectorized<T>, Vectorized<T>> inline inner_deinterleave2(
    const Vectorized<T>& a,
    // 定义一个模板函数，接受两个 Vectorized<T> 类型的引用参数 a 和 b
    const Vectorized<T>& b) {
      // inputs:
      //   a = {a0, b0, a1, b1,, a2, b2, a3, b3}
      //   b = {a4, b4, a5, b5,, a6, b6, a7, b7}
      
      // 使用 Vectorized<T> 类型中的 vtype 作为局部类型别名
      using vtype = typename Vectorized<T>::vtype;
      
      // {a0,a2,b0,b2} {a1,a3,b1,b3}，将输入向量 a 的元素重新组合
      vtype a0a2b0b2 = vec_mergeh(a.vec0(), a.vec1());
      vtype a1a3b1b3 = vec_mergel(a.vec0(), a.vec1());
    
      // 将重新组合的向量再次合并为两个矢量
      vtype aa0123 = vec_mergeh(a0a2b0b2, a1a3b1b3);
      vtype bb0123 = vec_mergel(a0a2b0b2, a1a3b1b3);
    
      // 将输入向量 b 的元素重新组合
      vtype a0a2b0b2_2 = vec_mergeh(b.vec0(), b.vec1());
      vtype a1a3b1b3_2 = vec_mergel(b.vec0(), b.vec1());
    
      // 将重新组合的向量再次合并为两个矢量
      vtype aa0123_2 = vec_mergeh(a0a2b0b2_2, a1a3b1b3_2);
      vtype bb0123_2 = vec_mergel(a0a2b0b2_2, a1a3b1b3_2);
    
      // it could be done with vec_perm ,too
      // 使用 vec_perm 函数也可以实现相同的功能
      // swap lanes:
      //   return {a0, a1, a2, a3,, a4, a5, a6, a7}
      //          {b0, b1, b2, b3,, b4, b5, b6, b7}
    
      // 返回一个 std::pair 对象，包含两个 Vectorized<T> 类型的对象
      return std::make_pair(
          Vectorized<T>{aa0123, aa0123_2}, Vectorized<T>{bb0123, bb0123_2});
    }
} // 关闭命名空间 at
} // 关闭命名空间 vec
} // 关闭命名空间 at
```