# `.\pytorch\aten\src\ATen\native\cuda\jit_utils.cpp`

```
#define TORCH_ASSERT_NO_OPERATORS
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>
#include <c10/util/hash.h>
#include <c10/util/Optional.h>
#include <ATen/jit_macros.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/code_template.h>
#include <ATen/OpMathType.h>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/cuda/llvm_jit_strings.h>
#include <ATen/native/cuda/reduction_template.cuh>

#include <sstream>
#include <fstream>
#include <cstdio>
#include <iterator> // istreambuf_iterator
#include <cstdlib>
#include <string>

// TODO: C++17 has the filesystem header, which may replace these
#ifdef _WIN32
  // On Windows, the POSIX implementations are considered deprecated. We simply map to the newer variant.
  #include <process.h>
  #include <direct.h>
  #include <io.h>
  #define access _access
  #define getpid _getpid
  #define R_OK    4
  #define W_OK    2
  #define F_OK    0
#else
  #include <sys/types.h>
  #include <sys/stat.h> // mkdir
  #include <unistd.h>
#endif

namespace at::cuda::jit {

// hiprtc already includes some traits, so this removes duplicate definitions of
// integral_constant, is_same, is_integral, enable_if, is_floating_point, is_arithmetic.
// Copied from aten/src/ATen/cuda/llvm_basic.cpp, then modified as above.
// If not compiling for ROCm, return the original get_traits_string().
// 获取一组类型特征的字符串表示，但在使用 hiprtc 时要避免重复定义
std::string get_traits_string_but_hiprtc_safe() {
#ifdef USE_ROCM
    return R"ESCAPE(
namespace std {

template <class _Tp>
_Tp&& __declval(int);
template <class _Tp>
_Tp __declval(long);
template <class _Tp>
decltype(__declval<_Tp>(0)) declval() noexcept;

template <class _Tp> struct remove_const            {typedef _Tp type;};
template <class _Tp> struct remove_const<const _Tp> {typedef _Tp type;};
template <class _Tp> using remove_const_t = typename remove_const<_Tp>::type;

template <class _Tp> struct remove_volatile               {typedef _Tp type;};
template <class _Tp> struct remove_volatile<volatile _Tp> {typedef _Tp type;};
template <class _Tp> using remove_volatile_t = typename remove_volatile<_Tp>::type;

template <class _Tp> struct remove_cv
{typedef typename remove_volatile<typename remove_const<_Tp>::type>::type type;};
template <class _Tp> using remove_cv_t = typename remove_cv<_Tp>::type;

template <class _Tp> struct __libcpp_is_floating_point              : public false_type {};
template <>          struct __libcpp_is_floating_point<float>       : public true_type {};
template <>          struct __libcpp_is_floating_point<double>      : public true_type {};
template <>          struct __libcpp_is_floating_point<long double> : public true_type {};

template <class _Tp>
inline constexpr bool is_arithmetic_v = is_arithmetic<_Tp>::value;

template <class _Tp>
struct __numeric_type {
    using type = typename conditional<
        is_arithmetic_v<remove_cv_t<_Tp>>,
        remove_cv_t<_Tp>,
        void
    >::type;
};

} // namespace std
)ESCAPE";
#else
    // 若不是为 ROCm 编译，则返回原始的类型特征字符串
    return get_traits_string();
#endif
}

} // namespace at::cuda::jit
{
   static void __test(...);
   // 声明一个接受任意参数的静态函数 __test

   static float __test(float);
   // 声明一个接受 float 类型参数并返回 float 类型结果的静态函数 __test

   static double __test(char);
   // 声明一个接受 char 类型参数并返回 double 类型结果的静态函数 __test

   static double __test(int);
   // 声明一个接受 int 类型参数并返回 double 类型结果的静态函数 __test

   static double __test(unsigned);
   // 声明一个接受 unsigned 类型参数并返回 double 类型结果的静态函数 __test

   static double __test(long);
   // 声明一个接受 long 类型参数并返回 double 类型结果的静态函数 __test

   static double __test(unsigned long);
   // 声明一个接受 unsigned long 类型参数并返回 double 类型结果的静态函数 __test

   static double __test(long long);
   // 声明一个接受 long long 类型参数并返回 double 类型结果的静态函数 __test

   static double __test(unsigned long long);
   // 声明一个接受 unsigned long long 类型参数并返回 double 类型结果的静态函数 __test

   static double __test(double);
   // 声明一个接受 double 类型参数并返回 double 类型结果的静态函数 __test

   static long double __test(long double);
   // 声明一个接受 long double 类型参数并返回 long double 类型结果的静态函数 __test

   typedef decltype(__test(declval<_Tp>())) type;
   // 定义一个类型 type，表示调用 __test 函数并传入 _Tp 类型参数的返回类型

   static const bool value = !is_same<type, void>::value;
   // 如果 type 不是 void 类型，则 value 为 true；否则为 false
};

template <>
struct __numeric_type<void>
{
   static const bool value = true;
   // 对于 __numeric_type<void> 特化，value 值为 true
};

// __promote

template <class _A1, class _A2 = void, class _A3 = void,
          bool = __numeric_type<_A1>::value &&
                 __numeric_type<_A2>::value &&
                 __numeric_type<_A3>::value>
class __promote_imp
{
public:
    static const bool value = false;
    // 如果 _A1、_A2、_A3 任意一个不是数值类型，则 value 为 false
};

template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true>
{
private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
    typedef typename __promote_imp<_A3>::type __type3;
public:
    typedef decltype(__type1() + __type2() + __type3()) type;
    // 定义一个类型 type，表示 __type1、__type2 和 __type3 的加法结果的类型
    static const bool value = true;
    // 如果 _A1、_A2、_A3 都是数值类型，则 value 为 true
};

template <class _A1, class _A2>
class __promote_imp<_A1, _A2, void, true>
{
private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
public:
    typedef decltype(__type1() + __type2()) type;
    // 定义一个类型 type，表示 __type1 和 __type2 的加法结果的类型
    static const bool value = true;
    // 如果 _A1、_A2 都是数值类型，则 value 为 true
};

template <class _A1>
class __promote_imp<_A1, void, void, true>
{
public:
    typedef typename __numeric_type<_A1>::type type;
    // 定义一个类型 type，表示 __numeric_type<_A1>::type 的类型
    static const bool value = true;
    // 如果 _A1 是数值类型，则 value 为 true
};

template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3> {};

} // namespace std
)ESCAPE";
#else
    return get_traits_string();
#endif
}

#ifdef USE_ROCM
const std::string jit_preamble = R"ESCAPE(
#pragma clang force_cuda_host_device begin
)ESCAPE";
// 如果使用 ROCm，定义 jit_preamble 字符串为包含指定预处理指令的字符串

const std::string jit_epilogue = R"ESCAPE(
#pragma clang force_cuda_host_device end
)ESCAPE";
// 如果使用 ROCm，定义 jit_epilogue 字符串为包含指定预处理指令的字符串
#else
const std::string jit_preamble;
// 如果不使用 ROCm，定义空字符串 jit_preamble

const std::string jit_epilogue;
// 如果不使用 ROCm，定义空字符串 jit_epilogue
#endif

    for (int i = 0; i < size; i++) {
      data[i] = x;
    }
    // 将 x 赋值给数组 data 中的每个元素，循环执行 size 次
  }
  };

  ${half_string}
  ${bfloat16_string}
  ${complex_body_string}
  ${complex_half_body_string}
  ${complex_math_string}


)ESCAPE";

// 将 half_string、bfloat16_string、complex_body_string、complex_half_body_string、complex_math_string 插入代码

//we need to include half, bfloat16 and complex strings to all kernels with half arguments and to all kernels with type casting
//regardless of whether they have half arguments (because fetch_and_cast and cast_and_store loop over all types)
const std::string jiterator_half_support_literal = R"ESCAPE(
namespace at {
struct alignas(2) Half {
  unsigned short x;
  // 定义一个结构体 Half，包含一个无符号短整型成员 x

  Half() = default;
  // 默认构造函数

  inline __host__ __device__ Half(float value){
#ifdef __HIPCC__
    x = __half_as_short(__float2half(value));
    // 如果使用 HIPCC 编译器，将 float 类型 value 转换为半精度浮点数，并存储在 x 中
#else
    asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(x) : "f"(value));
    // 使用内联汇编将 float 类型 value 转换为半精度浮点数，并存储在 x 中
#endif
  }

  inline __host__ __device__ operator float() const{
    // 类型转换运算符，将 Half 结构体转换为 float 类型
#ifdef __HIPCC__
      // 如果编译环境为 HIPCC，则将输入的 __half 类型转换为 float 类型并返回
      return __half2float(*reinterpret_cast<const __half*>(&x));
#else
      // 否则，使用内联汇编将输入的 __half 类型转换为 float 类型并返回
      float val;
      asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(x)); // 这里是否需要进行 const 强制转换？
      return val;
#endif
  }
};
}
)ESCAPE";

const std::string jiterator_bfloat16_support_literal = R"ESCAPE(
namespace at {
struct alignas(2) BFloat16 {
  unsigned short x;

  __device__ unsigned short __internal_float2bfloat16(
      const float f,
      unsigned int& sign,
      unsigned int& remainder) {
    unsigned int x;

    x = __float_as_uint(f);

    // 如果输入的浮点数超出了 BFloat16 能表示的范围，则返回最大值 0x7fff
    if ((x & 0x7fffffffU) > 0x7f800000U) {
      sign = 0U;
      remainder = 0U;
      return static_cast<unsigned short>(0x7fffU);
    }
    // 计算符号位和余数部分，并返回 BFloat16 类型的值
    sign = x >> 31;
    remainder = x << 16;
    return static_cast<unsigned short>(x >> 16);
  }


  BFloat16() = default;
  // 根据输入的浮点数值进行初始化 BFloat16 类型变量
  inline __host__ __device__ BFloat16(float value){
  #if __CUDA_ARCH__ >= 800
  // 在支持的 CUDA 架构版本中，使用内联汇编将浮点数转换为 BFloat16 类型
  asm("{  cvt.rn.bf16.f32 %0, %1;}\n" : "=h"(x) : "f"(value));
  )ESCAPE"
  R"ESCAPE(
  #else
  // 对于不支持的 CUDA 架构版本，手动实现浮点数到 BFloat16 的转换
  unsigned int sign;
  unsigned int remainder;
  x = __internal_float2bfloat16(value, sign, remainder);
  // 调整 BFloat16 的值，确保舍入正确性
  if ((remainder > 0x80000000U) ||
      ((remainder == 0x80000000U) && ((x & 0x1U) != 0U))) {
    x++;
  }
  #endif
  }

  // 将 BFloat16 类型转换为 float 类型
  inline __host__ __device__ operator float() const{
#ifdef __HIPCC__
    // 如果编译环境为 HIPCC，则使用联合体将 BFloat16 转换为 float 类型并返回
    union
    {
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(x) << 16};
    return u.fp32;
#else
    // 否则，使用内联汇编将 BFloat16 转换为 float 类型并返回
    float val;
    asm("{ mov.b32 %0, {0,%1};}\n" : "=f"(val) : "h"(x)); // 这里是否需要进行 const 强制转换？
    return val;
#endif
  }

};
}
)ESCAPE";

// From c10/util/Load.h
const std::string load_support_literal = R"ESCAPE(

  namespace c10 {
    // LoadImpl 结构模板，用于加载特定类型的数据
    template <typename T>
    struct LoadImpl {
      // 从给定地址加载类型 T 的数据
      __device__ static T apply(const void *src) {
        return *reinterpret_cast<const T*>(src);
      }
    };

    // 特化模板，用于加载 bool 类型数据
    template <>
    struct LoadImpl<bool> {
      // 从给定地址加载 bool 类型数据
      __device__ static bool apply(const void *src) {
        static_assert(sizeof(bool) == sizeof(char), "");
        return LoadImpl<char>::apply(src);
      }
    };

    // 加载函数模板，用于加载指定类型的数据
    template <typename T>
    __device__ T load(const void *src) {
      return LoadImpl<T>::apply(src);
    }

    // 特化的加载函数模板，用于加载指定类型的数据
    template <typename scalar_t>
    __device__ scalar_t load(const scalar_t *src) {
      return LoadImpl<scalar_t>::apply(src);
    }
  }  // namespace c10

)ESCAPE";

// copy-pasted from c10/util/TypeCast.h and c10/core/DynamicCast.h
const std::string dynamic_cast_support_literal = R"ESCAPE(

  // 检查是否为复数结构体模板
  template <typename T>
  struct is_complex : public std::false_type {};

  // 特化模板，检查是否为复数结构体
  template <typename T>
  struct is_complex<std::complex<T>> : public std::true_type {};

  // 需要实数的结构体模板
  template <typename dest_t, typename src_t>
  struct needs_real {
    constexpr static bool value =
        (is_complex<src_t>::value && !is_complex<dest_t>::value);
  };

  // 可能为实数的结构体模板
  template <bool, typename src_t>
  struct maybe_real {
    // 应用函数，根据情况返回原始类型
    static inline src_t apply(src_t src) {
      return src;


这段代码涉及了不同的 C++ 和 CUDA 相关的语法和功能，包括条件编译、内联汇编、模板特化等，注释着重解释了每一行代码的具体作用和背景信息。
  };

// 结构体定义结束


  template <typename src_t>
  struct maybe_real<true, src_t> {
    static inline decltype(auto) apply(src_t src) {
      return src.real();
    }
  };

// 对于`maybe_real`结构体的特化，当`real`为真时，应用`src.real()`函数返回实数部分


  template <typename dest_t, typename src_t>
  struct static_cast_with_inter_type {
    static inline dest_t apply(
        src_t src) {
      constexpr bool real = needs_real<dest_t, src_t>::value;
      return static_cast<dest_t>(maybe_real<real, src_t>::apply(src));
    }
  };

// `static_cast_with_inter_type`结构体模板定义，执行类型转换，根据`needs_real`的值确定是否需要实部，然后应用`maybe_real`的结果进行转换


  template <typename src_t>
  struct static_cast_with_inter_type<uint8_t, src_t> {
    static inline uint8_t apply(
        src_t src) {
      constexpr bool real = needs_real<uint8_t, src_t>::value;
      return static_cast<uint8_t>(
          static_cast<int64_t>(maybe_real<real, src_t>::apply(src)));
    }
  };

// 特化`static_cast_with_inter_type`结构体模板，将`src_t`转换为`uint8_t`类型，包括可能的实部转换为整型后再转换为`uint8_t`


  template <>
  struct static_cast_with_inter_type<std::complex<at::Half>, at::BFloat16> {
    static inline std::complex<at::Half> apply(at::BFloat16 src) {
      return static_cast<std::complex<at::Half>>(float{src});
    }
  };

// `static_cast_with_inter_type`结构体模板的特化，将`at::BFloat16`类型转换为`std::complex<at::Half>`类型


  template <>
  struct static_cast_with_inter_type<std::complex<at::Half>, at::Half> {
    static inline std::complex<at::Half> apply(at::Half src) {
      return static_cast<std::complex<at::Half>>(float{src});
    }
  };

// `static_cast_with_inter_type`结构体模板的特化，将`at::Half`类型转换为`std::complex<at::Half>`类型


  template <>
  struct static_cast_with_inter_type<
      std::complex<at::Half>,
      std::complex<double>> {
    static inline std::complex<at::Half> apply(std::complex<double> src) {
      return static_cast<std::complex<at::Half>>(static_cast<std::complex<float>>(src));
    }
  };

// `static_cast_with_inter_type`结构体模板的特化，将`std::complex<double>`类型转换为`std::complex<at::Half>`类型，中间通过`static_cast`实现浮点数类型的转换


  // 从`ptr`获取动态类型为`src_type`的值，并将其转换为静态类型`dest_t`
  #define FETCH_AND_CAST_CASE(type, scalartype) \
    case ScalarType::scalartype:                \
      return static_cast_with_inter_type<dest_t, type>::apply(c10::load<type>(ptr));
  template<typename dest_t>
  __device__ inline dest_t fetch_and_cast(const ScalarType src_type, const void *ptr) {
    switch (src_type) {
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_QINT(FETCH_AND_CAST_CASE)
        default:
          ERROR_UNSUPPORTED_CAST
    }
    return dest_t(0); // 为了避免编译器警告而返回零值
  }

// 从`ptr`中加载数据，并根据`src_type`将其转换为`dest_t`类型，使用`static_cast_with_inter_type`进行转换，支持所有标量类型（除了QINT），默认情况下返回不支持的类型错误


  // 将静态类型为`src_t`的值转换为动态类型`dest_type`，并存储到`ptr`
  #define CAST_AND_STORE_CASE(type, scalartype)                             \
    case ScalarType::scalartype:                                            \
      *(type*)ptr = static_cast_with_inter_type<type, src_t>::apply(value); \
      return;
  template<typename src_t>
  __device__ inline void cast_and_store(const ScalarType dest_type, void *ptr, src_t value) {
    switch (dest_type) {
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_QINT(CAST_AND_STORE_CASE)
        default:;
    }
    ERROR_UNSUPPORTED_CAST
  }

// 将静态类型为`src_t`的值转换为动态类型`dest_type`，并将其存储到`ptr`中，使用`static_cast_with_inter_type`进行转换，支持所有标量类型（除了QINT），默认情况下报错不支持的类型转换错误


  template <int N>
  struct LoadWithCast {
    using array_t = Array<ScalarType, N==0? 1 : N>;
    using size_array_t = Array<uint32_t, N==0? 1: N>;

    array_t dtypes;
    size_array_t element_sizes;
    template <typename scalar_t>

// `LoadWithCast`结构体模板定义，使用`Array`模板，支持动态数组大小`N`，存储数据类型和元素大小
    // 定义一个模板结构体 StoreWithCast，支持泛型参数 N，默认为 1
    template <int N = 1>
    struct StoreWithCast {
        // 定义数组类型 array_t，元素类型为 ScalarType，长度为 N（如果 N 为 0，则长度为 1）
        using array_t = Array<ScalarType, N==0? 1 : N>;
        // 定义数组类型 size_array_t，元素类型为 uint32_t，长度为 N（如果 N 为 0，则长度为 1）
        using size_array_t = Array<uint32_t, N==0? 1: N>;
    
        // 成员变量，存储数据类型数组
        array_t dtypes;
        // 成员变量，存储元素大小数组
        size_array_t element_sizes;
    
        // 模板方法，用于将值存储到指定偏移处
        template<typename scalar_t>
        __device__ void store(scalar_t value, char *base_ptr, uint32_t offset, int arg = 0) {
            // 计算存储位置的指针，基于 base_ptr 和元素大小
            void *ptr = base_ptr + element_sizes[arg] * offset;
            // 调用 cast_and_store 函数将 value 转换并存储到 ptr 指向的位置
            cast_and_store<scalar_t>(dtypes[arg], ptr, value);
        }
    };
const std::string no_dynamic_cast_support_literal = R"ESCAPE(

// 定义字符串常量，包含不支持动态类型转换的文本信息
const std::string no_dynamic_cast_support_literal = R"ESCAPE(


  struct LoadWithoutCast {
  template <typename scalar_t>
  __device__ scalar_t load(char* base_ptr, uint32_t offset, int arg=0) {
    return c10::load(reinterpret_cast<scalar_t*>(base_ptr) + offset);
  }
  };

  // 定义 LoadWithoutCast 结构体，用于从给定偏移处加载数据，无需类型转换
  struct LoadWithoutCast {
    // 使用模板，加载类型为 scalar_t 的数据
    template <typename scalar_t>
    __device__ scalar_t load(char* base_ptr, uint32_t offset, int arg=0) {
      // 调用 c10::load 函数加载数据，基于偏移和基址指针进行解释
      return c10::load(reinterpret_cast<scalar_t*>(base_ptr) + offset);
    }
  };


  struct StoreWithoutCast {
  template<typename scalar_t>
  __device__ void store(scalar_t value, char *base_ptr, uint32_t offset, int arg=0) {
    *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
  }
  };

  // 定义 StoreWithoutCast 结构体，用于在给定偏移处存储数据，无需类型转换
  struct StoreWithoutCast {
    // 使用模板，存储类型为 scalar_t 的数据
    template<typename scalar_t>
    __device__ void store(scalar_t value, char *base_ptr, uint32_t offset, int arg=0) {
      // 将值 value 存储到基址指针 base_ptr 加上偏移 offset 处
      *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
    }
  };


const std::string offset_calc_template = R"ESCAPE(

// 定义字符串常量，包含偏移计算模板信息
const std::string offset_calc_template = R"ESCAPE(


  template <typename T>
  struct DivMod {
  T div;
  T mod;

  __device__ DivMod(T _div, T _mod) {
      div = _div;
      mod = _mod;
  }
  };

  // 定义 DivMod 结构体模板，用于保存整数除法和求余结果
  template <typename T>
  struct DivMod {
    T div;  // 商
    T mod;  // 余数

    // 构造函数，初始化 div 和 mod
    __device__ DivMod(T _div, T _mod) {
        div = _div;
        mod = _mod;
    }
  };


  //<unsigned int>
  struct IntDivider {
  IntDivider() = default;

  __device__ inline unsigned int div(unsigned int n) const {
  unsigned int t = __umulhi(n, m1);
  return (t + n) >> shift;
  }

  // 定义 IntDivider 结构体，实现整数除法操作
  //<unsigned int>
  struct IntDivider {
    // 默认构造函数
    IntDivider() = default;

    // 内联函数，实现无符号整数 n 的除法
    __device__ inline unsigned int div(unsigned int n) const {
      unsigned int t = __umulhi(n, m1);  // 使用 __umulhi 计算乘法高位
      return (t + n) >> shift;           // 返回除法结果
    }


  __device__ inline unsigned int mod(unsigned int n) const {
  return n - div(n) * divisor;
  }

    // 内联函数，实现无符号整数 n 的取模
    __device__ inline unsigned int mod(unsigned int n) const {
      return n - div(n) * divisor;  // 返回 n 对 divisor 取模的结果
    }


  __device__ inline DivMod<unsigned int> divmod(unsigned int n) const {
  unsigned int q = div(n);
  return DivMod<unsigned int>(q, n - q * divisor);
  }

    // 内联函数，同时计算无符号整数 n 的除法和取模
    __device__ inline DivMod<unsigned int> divmod(unsigned int n) const {
      unsigned int q = div(n);  // 计算除法得到商 q
      return DivMod<unsigned int>(q, n - q * divisor);  // 返回 DivMod 结构体对象
    }

    unsigned int divisor;  // d above.
    unsigned int m1;  // Magic number: m' above.
    unsigned int shift;  // Shift amounts.
  };

    unsigned int divisor;  // 除数，对应上述的 d
    unsigned int m1;       // 魔数，对应上述的 m'
    unsigned int shift;    // 移位量，用于计算除法结果
  };


  template <int NARGS>
  struct TrivialOffsetCalculator {
    // The offset for each argument. Wrapper around fixed-size array.
    // The offsets are in # of elements, not in bytes.
    Array<${index_type}, NARGS> get(${index_type} linear_idx) const {
      Array<${index_type}, NARGS> offsets;
      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] = linear_idx;
      }
      return offsets;
    }
  };

  // 定义 TrivialOffsetCalculator 结构体模板，实现简单的偏移计算器
  template <int NARGS>
  struct TrivialOffsetCalculator {
    // 每个参数的偏移量，封装在固定大小的数组中
    // 偏移量是以元素数目而不是字节为单位的
    Array<${index_type}, NARGS> get(${index_type} linear_idx) const {
      Array<${index_type}, NARGS> offsets;
      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] = linear_idx;  // 设置每个参数的偏移量为 linear_idx
      }
      return offsets;  // 返回偏移量数组
    }
  };


  template<int NARGS>
  struct OffsetCalculator {
  OffsetCalculator() = default;
  __device__ __forceinline__ Array<${index_type}, NARGS> get(${index_type} linear_idx) const {
      Array<${index_type}, NARGS> offsets;
      #pragma unroll
      for (int arg = 0; arg < NARGS; ++arg) {
      offsets[arg] = 0;
      }

  // 定义 OffsetCalculator 结构体模板，实现复杂的偏移计算器
  template<int NARGS>
  struct OffsetCalculator {
    // 默认构造函数
    OffsetCalculator() = default;

    // 内联函数，获取偏移量数组，同时考虑线性索引
    __device__ __forceinline__ Array<${index_type}, NARGS> get(${index_type} linear_idx) const {
        Array<${index_type}, NARGS> offsets;
        #pragma unroll
        for (int arg = 0; arg < NARGS; ++arg) {
            offsets[arg] = 0;  // 初始化偏移量数组为零
        }

        #pragma unroll
        for (int dim = 0; dim < MAX_DIMS; ++dim) {
            if (dim == dims) {
                break;  // 如果维度超过了设定的维度数，跳出循环
            }

            auto divmod = sizes_[dim].divmod(linear_idx);  // 计算线性索引在当前维度的商和余数
            linear_idx = divmod.div;  // 更新线性索引为商

            #pragma unroll
            for (int arg = 0; arg < NARGS; ++arg) {
                offsets[arg] += divmod.mod * strides_[dim][arg];  // 根据余数和步长计算偏移量
            }
            //printf("offset calc thread dim size stride offset %d %d %d %d %d %d %d %d\n",
            //threadIdx.x, dim, sizes_[dim].divisor, strides_[dim][0], offsets[0], linear_idx, divmod.div, divmod.mod);
        }
        return offsets;  // 返回最终计算的偏移量数组
    }

    int dims;  // 维度数
    IntDivider sizes_[MAX_DIMS];  // 各维度的大小和除法模数信息
    // 注意：此方法不支持 nInputs == 0
// 定义字符串常量，包含一个 JIT 代码模板，用于生成 CUDA 核函数代码
const std::string jit_code_template = R"ESCAPE(

  ${load_support}
  ${dynamic_casting_string}


  ${functor}

  // TODO: setup grid-stride loop
  // 声明外部 C 函数，指定为 CUDA 核函数
  extern "C" __global__
  void ${name}_kernel(
      const int numel,  // 输入参数：元素总数
      Array<char*, ${nInputs}+${nOutputs}> data, //[${nInputs}+${nOutputs}], 数据数组
      ${offset_calculator}<${nInputs}> input_calculator,  // 输入偏移计算器
      ${offset_calculator}<${nOutputs}> output_calculator,  // 输出偏移计算器
      ${loader} l,  // 加载器
      ${storer} s,  // 存储器
      ${compute_type} scalar_val${extra_params}) {  // 计算类型和额外参数
    ${declare_load_arrays}  // 声明加载数组
    ${declare_store_arrays}  // 声明存储数组

    int idx = blockIdx.x;  // 获取当前块的索引

    int remaining = numel - block_work_size * idx;  // 计算剩余元素数
    int thread_idx = threadIdx.x;  // 获取当前线程的索引

    #pragma unroll
    // 循环处理每个线程工作单元
    for (int j = 0; j < thread_work_size; j++){
        if (thread_idx >= remaining) {  // 如果线程索引超出剩余元素数，退出循环
            break;
        }

        int linear_idx = thread_idx + block_work_size * idx;  // 计算线性索引
        auto input_offsets = input_calculator.get(linear_idx);  // 获取输入偏移量
        ${load_inputs}  // 加载输入数据
        // printf(
        //    "thread %d a %f offsets %d\n", threadIdx.x, arg0[j], input_offsets[0]);
        thread_idx += num_threads;  // 更新线程索引
    }

    #pragma unroll
    // 循环处理每个线程工作单元
    for (int j = 0; j < thread_work_size; j++) {
      if ((threadIdx.x  + j*num_threads) < remaining) {
        ${call_functor}  // 调用函数对象处理数据
      }
    }

    thread_idx = threadIdx.x;  // 重新设置线程索引
    #pragma unroll
    // 循环处理每个线程工作单元
    for (int j = 0; j < thread_work_size; j++){
        if (thread_idx >= remaining) {  // 如果线程索引超出剩余元素数，退出循环
            break;
        }
        //TODO maybe think about unifying offset calculators and reuse
        //offsets computed in the load loop
        int linear_idx = thread_idx + block_work_size * idx;  // 计算线性索引
        auto output_offsets = output_calculator.get(linear_idx);  // 获取输出偏移量
        //printf("output thread %d offset %d\n", threadIdx.x, output_offsets[0]);
        ${store_outputs}  // 存储输出数据
        thread_idx += num_threads;  // 更新线程索引
    }
  }
)ESCAPE";

// 定义字符串常量，包含一个 JIT 向量化代码模板，用于生成 CUDA 向量化函数代码
const std::string jit_vectorized_code_template = R"ESCAPE(

  ${load_support}

  // 模板函数：加载标量数据
  template <typename scalar_t>
  __device__ __inline__ scalar_t load(char* base_ptr, uint32_t offset) {
      return c10::load(reinterpret_cast<scalar_t*>(base_ptr) + offset);
  }

  // 模板函数：存储标量数据
  template<typename scalar_t>
  __device__ __inline__ void store(scalar_t value, char *base_ptr, uint32_t offset) {
      *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
  }

  // 结构体模板：对齐向量，生成 CUDA 向量化加载/存储
  template<typename scalar_t, int vec_size>
  struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
    scalar_t val[vec_size];
  };

  // 模板函数：加载向量数据
  template <int vec_size, typename scalar_t>
  __device__ aligned_vector<scalar_t, vec_size> load_vector(const scalar_t *base_ptr, uint32_t offset) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    auto *from = reinterpret_cast<const vec_t *>(base_ptr);
    return from[offset];
  }

  // 特化模板函数：加载布尔向量数据
  template <int vec_size>
  __device__ aligned_vector<bool, vec_size> load_vector(const bool *base_ptr, uint32_t offset) {
    // See NOTE [Loading boolean values]
    // 加载布尔值向量数据，向量大小为 vec_size

    // 加载布尔值向量数据，向量大小为 vec_size
    // 查看说明 [Loading boolean values]
    template <int vec_size>
    __device__ aligned_vector<bool, vec_size> load_vector(const bool *base_ptr, uint32_t offset) {
    // 加载一个大小为 vec_size 的向量，并将其解释为布尔值向量
    auto tmp = load_vector<vec_size>(reinterpret_cast<const uint8_t*>(base_ptr), offset);
    // 创建一个与 tmp 向量大小相同的布尔值向量 ret
    aligned_vector<bool, vec_size> ret;
    // 遍历 tmp 向量的每个元素，将其转换为布尔值后存入 ret 向量
    for (int i = 0; i < vec_size; ++i) {
      ret.val[i] = bool(tmp.val[i]);
    }
    // 返回填充好的布尔值向量 ret
    return ret;
    }
    
    ${functor}
    
    // TODO: setup grid-stride loop
    
    // 声明全局函数 ${name}_vectorized${vec_size}_kernel，接受以下参数
    extern "C" __global__
    void ${name}_vectorized${vec_size}_kernel(
        // 输入数据总数 N
        const int N,
        // 输入输出数组，包含 ${nInputs} 个输入和 ${nOutputs} 个输出
        Array<char*, ${nInputs}+${nOutputs}> data,
        // 标量值的类型为 ${compute_type}，通常是输入数据的类型
        ${compute_type} scalar_val${extra_params}) //[${nInputs}+${nOutputs}],
        {
        // 定义向量大小为 ${vec_size}
        constexpr int vec_size = ${vec_size};
        // 声明标量类型为 ${scalar_type}
        using scalar_t = ${scalar_type};
        // 计算剩余处理的数据量
        int remaining = N - block_work_size * blockIdx.x;
        // 当前线程的索引
        int thread_idx = threadIdx.x;
        // 当前块的索引
        int idx = blockIdx.x;
        // 声明加载数组
        ${declare_load_arrays}
        // 声明存储数组
        ${declare_store_arrays}
    
        // 如果剩余数据不足一个块的处理量
        if (remaining < block_work_size) {
          // 循环处理每个线程的工作量
          #pragma unroll
          for (int j = 0; j < thread_work_size; j++){
            // 如果线程索引超过剩余数据量，则退出循环
            if (thread_idx >= remaining) {
              break;
            }
            // 计算当前线性索引
            int linear_idx = thread_idx + block_work_size * idx;
            // 加载输入数据
            ${load_unrolled_inputs}
            // 增加线程索引，准备处理下一个线程
            thread_idx += num_threads;
          }
          // 循环处理每个线程的工作量
          #pragma unroll
          for (int j = 0; j < thread_work_size; j++) {
            // 如果线程索引小于剩余数据量
            if ((threadIdx.x  + j*num_threads) < remaining) {
              // 调用函数对象处理数据
              ${call_functor}
            }
          }
          // 重置线程索引
          thread_idx = threadIdx.x;
          // 循环处理每个线程的工作量
          #pragma unroll
          for (int j = 0; j < thread_work_size; j++) {
            // 如果线程索引超过剩余数据量，则退出循环
            if (thread_idx >= remaining) {
                break;
            }
            // 计算当前线性索引
            int linear_idx = thread_idx + block_work_size * idx;
            // 存储输出数据
            ${store_unrolled_outputs}
            // 增加线程索引，准备处理下一个线程
            thread_idx += num_threads;
          }
        } else {
          // 如果剩余数据量足够一个块的处理量，则执行以下操作
          static constexpr int loop_size = thread_work_size / vec_size;
          // 加载向量化输入数据
          ${vector_inputs}
          // 循环加载向量化输入数据
          #pragma unroll
          for (int i = 0; i<loop_size; i++){
            ${load_vectorized_inputs}
            // 增加线程索引，准备处理下一个线程
            thread_idx += num_threads;
          }
    
          // 循环处理每个线程的工作量
          #pragma unroll
          for (int j = 0; j < thread_work_size; j++) {
            // 调用函数对象处理数据
            ${call_functor}
          }
    
          // 声明输出向量的类型为 aligned_vector<${result_type}, vec_size>
          using vec_t_output = aligned_vector<${result_type}, vec_size>;
          // 声明向量化输出数据
          ${vector_outputs}
          // 重置线程索引
          int thread_idx = threadIdx.x;
          // 循环处理每个线程的工作量
          #pragma unroll
          for (int i = 0; i<loop_size; i++){
            // 创建向量化输出对象 v
            vec_t_output v;
            // 存储向量化输出数据
            ${store_vectorized_outputs}
            // 增加线程索引，准备处理下一个线程
            thread_idx += num_threads;
          }
        }
    }
// 固定字符串，包含转义字符，用于替换操作
const char* const to_replace = R"ESCAPE";

// 替换字符串中所有出现的指定子串为另一字符串
static void replace_all(std::string& s, const std::string& to_replace, const std::string& replace_with) {
  std::ostringstream oss;
  std::size_t pos = 0;
  std::size_t prev_pos = pos;

  // 循环直到字符串中不再找到要替换的子串
  while (true) {
    prev_pos = pos;
    pos = s.find(to_replace, pos);
    if (pos == std::string::npos)
      break;
    // 拼接替换子串之前的部分和替换后的字符串
    oss << s.substr(prev_pos, pos - prev_pos);
    oss << replace_with;
    pos += to_replace.size();  // 移动到下一个可能的替换位置
  }

  // 拼接剩余的字符串
  oss << s.substr(prev_pos);
  s = oss.str();  // 更新原始字符串
}

// hipify 工具替换设备数学函数，例如 std::max -> ::max
// 见 torch/utils/hipify/cuda_to_hip_mappings.py。
// 这里进行反向替换，查找 " ::<name>" 避免重复替换
static std::string unhipify_math_functions(const std::string &original) {
  // 定义需要替换的映射关系
  static std::vector<std::pair<std::string,std::string>> mappings = {
    {" std::max", " ::max"},
    {" std::min", " ::min"},
    {" std::ceil", " ::ceil"},
    {" std::floor", " ::floor"},
    {" std::exp", " ::exp"},
    {" std::log", " ::log"},
    {" std::pow", " ::pow"},
    {" std::fabs", " ::fabs"},
    {" std::fmod", " ::fmod"},
    {" std::remainder", " ::remainder"},
    {" std::frexp", " ::frexp"}
  };
  std::string ret = original;  // 复制原始字符串以进行替换操作
  // 遍历映射表，依次进行反向替换操作
  for (const auto& mapping : mappings) {
    replace_all(ret, mapping.second, mapping.first);
  }
  return ret;  // 返回替换后的字符串副本
}

// 以下内容来自 fused_kernel.cpp
// TODO: 重构 codegenOutputQuery 到单独的文件中，可被两个文件引用
// 见 NOTE [ USE OF NVRTC AND DRIVER API ]
const at::cuda::NVRTC& nvrtc() {
  return at::globalContext().getNVRTC();  // 返回全局上下文的 NVRTC 对象
}

// 查询代码生成输出的架构和目标
// TODO: 重构以便该函数可以被 jit 和 aten 文件共用
void codegenOutputQuery(
    const cudaDeviceProp* const prop,  // CUDA 设备属性指针
    int& cuda_major,  // 输出 CUDA 主版本号
    int& cuda_minor,  // 输出 CUDA 次版本号
    int& nvrtc_major,  // 输出 NVRTC 主版本号
    int& nvrtc_minor,  // 输出 NVRTC 次版本号
    bool& compile_to_sass) {  // 是否编译为 SASS

#ifdef USE_ROCM
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcVersion(&nvrtc_major, &nvrtc_minor));  // 获取 NVRTC 版本信息
  cuda_major = prop->major;  // 设置 CUDA 主版本号为设备属性中的主版本号
  cuda_minor = prop->minor;  // 设置 CUDA 次版本号为设备属性中的次版本号
  compile_to_sass = false;  // 设置不编译为 SASS
#else
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcVersion(&nvrtc_major, &nvrtc_minor));  // 获取 NVRTC 版本信息
  TORCH_CHECK(
      nvrtc_major >= 6, "NVRTC versions less than 6 are not supported. Is: ", nvrtc_major);  // 检查 NVRTC 版本兼容性

  // 设备支持的版本
  // 通常低版本也能工作，但效率可能不如高版本
  using CUDAVersion = std::pair<int, int>;
  const CUDAVersion nvrtc_version{nvrtc_major, nvrtc_minor};
  const CUDAVersion dev_version{prop->major, prop->minor};
  CUDAVersion max_dev_version;  // 最大支持的驱动版本
  // 根据 NVRTC 主版本号设置最大支持的设备版本
  if (nvrtc_major <= 7) {
    max_dev_version = CUDAVersion(5, 0);
  } else if (nvrtc_major <= 8) {
    max_dev_version = CUDAVersion(6, 0);
  } else if (nvrtc_major <= 9) {
    max_dev_version = CUDAVersion(7, 2);
  } else if (nvrtc_major <= 10) {
    max_dev_version = CUDAVersion(7, 5);

    max_dev_version = CUDAVersion(7, 5);
  }

#endif
}
    max_dev_version = CUDAVersion(7, 5);
  } else if (nvrtc_version == CUDAVersion(11, 0)) { // 如果 NVRTC 版本是 11.0，支持 3-8.0
    max_dev_version = CUDAVersion(8, 0);
  } else if (nvrtc_major == 11 && nvrtc_minor < 8) {
    max_dev_version = CUDAVersion(8, 6);
  } else {
    // 如果驱动版本未知（即比当前代码更新），假设驱动支持这个设备
    max_dev_version = dev_version;
  }

  // 如果设备版本高于最大支持版本，则使用最大支持版本的主次版本号
  if (dev_version > max_dev_version) {
    cuda_major = max_dev_version.first;
    cuda_minor = max_dev_version.second;
    // 如果我们限制主次版本号，则不能编译为 sass
    compile_to_sass = false;
  } else {
    cuda_major = dev_version.first;
    cuda_minor = dev_version.second;
    compile_to_sass = true;
  }

  // 如果 CUDA 版本小于 11.1，则不允许编译为 sass
  #if defined(CUDA_VERSION) && CUDA_VERSION < 11010
    compile_to_sass = false;
  #endif
#endif
}

// TODO: another copy paste from jit, refactor so it's usable from both
// TODO: try making the CUcontext thread local to see if that improves performance - why is this slow?
// 初始化 CUDA 上下文
void initializeCudaContext() {
  // 如果上下文不存在，则懒惰地创建上下文；
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  CUcontext pctx = nullptr;
  // 获取当前 CUDA 上下文
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    // 获取 CUDA 自由锁，以确保在多线程环境中的安全性
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::getFreeMutex()));
    // 释放 CUDA 内存
    cudaFree(nullptr);
  }
}

std::string generate_code(
    const KernelDescriptor &desc,
    bool contiguous,
    bool dynamic_casting,
    BinaryFuncVariant scalar_pos,
    bool vectorized,
    int vec_size,
    bool return_by_ref) {
  // 生成额外参数的类型名列表
  c10::SmallVector<std::string> extra_args_typenames(desc.extra_args_types.size());
  for (auto i : c10::irange(extra_args_typenames.size())) {
    extra_args_typenames[i] = typeName(desc.extra_args_types[i]);
  }

  // 调用重载的 generate_code 函数，生成代码字符串
  return generate_code(
      desc.nInputs,
      desc.nOutputs,
      desc.f,
      desc.name,
      typeName(desc.f_inputs_type),
      typeName(toOpMathType(desc.f_inputs_type)),
      typeName(desc.result_type),
      contiguous,
      dynamic_casting,
      scalar_pos,
      extra_args_typenames,
      vectorized,
      vec_size,
      return_by_ref);
}

// FIXME - this are defined in Loops.cuh, but including Loops.cuh here would lead to circular includes Loops.cuh -> CUDALoops.cuh -> jit_utils.h -> Loops.cuh
// 线程工作大小定义为 4
#define THREAD_WORK_SIZE 4
constexpr int thread_work_size = THREAD_WORK_SIZE;

// 生成代码函数的重载，根据函数描述和参数生成相应的代码字符串
std::string generate_code(
    int nInputs,
    int nOutputs,
    const std::string& func_,
    const std::string& name,
    const std::string& f_inputs_type,
    const std::string& compute_type,
    const std::string& result_type,
    bool contiguous,
    bool dynamic_casting,
    BinaryFuncVariant scalar_pos,
    c10::SmallVector<std::string>& extra_args_typenames,
    bool vectorized,
    int vec_size,
    bool return_by_ref) {
  // 复制传入的函数名称到本地变量
  std::string func = func_;
  // 创建模板环境对象
  at::jit::TemplateEnv env;

  // 设置模板环境的各个变量
  env.s("index_type", "unsigned int");
  env.s("nInputs", std::to_string(nInputs));
  env.s("nOutputs", std::to_string(nOutputs));
  env.s("scalar_type", f_inputs_type);
  env.s("compute_type", compute_type);
  env.s("functor", func);
  env.s("name", name);
  env.s("cmath_string", get_cmath_string());

  // 为函数签名生成额外参数和为计算调用生成额外参数
  // 如果传入了捕获运行时状态的额外参数（例如 polygamma 函数）
  std::string extra_params = "";
  std::string extra_args = "";
  for (size_t i = 0; i < extra_args_typenames.size(); i++) {
    auto type = std::string(extra_args_typenames[i]);
    auto name = "extra_arg_" + std::to_string(i);
    extra_params += "," + type + " " + name;
    extra_args += ", " + name;
  }
  env.s("extra_params", extra_params);
  env.s("extra_args", extra_args);

  std::stringstream declare_load_arrays;
  // 为每个输入参数声明加载数组，数组类型可能不同，需要使用函数特性来确定类型
  for (int i = 0; i < nInputs; i++) {
    declare_load_arrays << f_inputs_type << " arg" << std::to_string(i)
                        << "[" << std::to_string(thread_work_size) << "];\n";
  }
  env.s("declare_load_arrays", declare_load_arrays.str());

  std::stringstream declare_store_arrays;
  // 为每个输出参数声明存储数组
  for (int i = 0; i < nOutputs; i++) {
    declare_store_arrays << result_type << " out" << std::to_string(i)
                        << "[" << std::to_string(thread_work_size) << "];\n";
  }
  env.s("declare_store_arrays", declare_store_arrays.str());

  std::stringstream functor_args;
  // 根据函数变体（是否有标量参数），生成调用函数时的参数列表
  if (scalar_pos == BinaryFuncVariant::NoScalar) {
    for (int i = 0; i < nInputs - 1; i++) {
      functor_args << "arg" << std::to_string(i) << "[j], ";
    }
    functor_args << "arg" << std::to_string(nInputs - 1) << "[j]";
  } else if (scalar_pos == BinaryFuncVariant::LhsScalar) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(nInputs == 1);
    functor_args << "scalar_val, arg0[j]";
  } else { //RhsScalar
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(nInputs == 1);
    functor_args << "arg0[j], scalar_val";
  }
  env.s("args", functor_args.str());

  std::string call_functor_template;
  // 根据是否按引用返回结果，构造调用函数的模板字符串
  if (return_by_ref) {  // 返回一个或多个输出引用
    bool need_temp_out = (compute_type != result_type);
    std::stringstream functor_outs;
    if (need_temp_out) {
      // 需要临时输出变量
      for (int i = 0; i < nOutputs - 1; i++) {
        functor_outs << "temp_out" << std::to_string(i) << ", ";
      }
      functor_outs << "temp_out" << std::to_string(nOutputs - 1);
    } else {
      // 直接使用输出数组
      for (int i = 0; i < nOutputs - 1; i++) {
        functor_outs << "out" << std::to_string(i) << "[j], ";
      }
      functor_outs << "out" << std::to_string(nOutputs - 1) << "[j]";
    }
    env.s("functor_outs", functor_outs.str());

    if (need_temp_out) {
      call_functor_template += "${compute_type} ${functor_outs};\n";
    }

    call_functor_template += "${name}<${compute_type}>(${args} ${extra_args}, ${functor_outs});\n";

    if (need_temp_out) {
      // 将临时输出变量赋值给最终的输出数组
      for (int i = 0; i < nOutputs; i++) {
        auto i_string = std::to_string(i);
        call_functor_template += "out" + i_string + "[j] = temp_out" + i_string + ";\n";
      }
    }

  } else {  // 对于单输出函数，按值返回
    call_functor_template = "out0[j] = ${name}<${compute_type}>(${args} ${extra_args});";
  }
  env.s("call_functor", at::jit::CodeTemplate(call_functor_template).format(env));

  if (f_inputs_type == "at::Half" || result_type == "at::Half" ||
      f_inputs_type == "std::complex<at::Half>" ||
      result_type == "std::complex<at::Half>" || dynamic_casting) {
    // 对于使用半精度或半精度复数的情况，进行相关处理
  // 设置环境变量 "half_string"，根据条件选择使用 jiterator_half_support_literal 或空字符串
  env.s("half_string", jiterator_half_support_literal);
} else {
  // 如果条件不满足，设置为空字符串
  env.s("half_string", "");
}
// 如果输入类型为 "at::BFloat16" 或结果类型为 "at::BFloat16" 或需要动态转换，则设置环境变量 "bfloat16_string"
if (f_inputs_type == "at::BFloat16" || result_type == "at::BFloat16" || dynamic_casting) {
  env.s("bfloat16_string", jiterator_bfloat16_support_literal);
} else {
  // 否则设置为空字符串
  env.s("bfloat16_string", "");
}
// 仅在计算类型为复数时需要定义复数数学函数
// 但是为了动态转换，即使计算类型不是复数，也需要定义 std::complex
if (f_inputs_type == "std::complex<float>" || result_type == "std::complex<float>" ||
    f_inputs_type == "std::complex<double>" || result_type == "std::complex<double>" ||
    f_inputs_type == "std::complex<at::Half>" || result_type == "std::complex<at::Half>") {
  // 设置环境变量 "traits_string"，获取复数类型的特征字符串，HIPRTC 安全
  env.s("traits_string", get_traits_string_but_hiprtc_safe());
  // 设置环境变量 "complex_body_string"，获取复数类型的主体字符串
  env.s("complex_body_string", get_complex_body_string());
  // 设置环境变量 "complex_math_string"，获取复数类型的数学函数字符串
  env.s("complex_math_string", get_complex_math_string());
  // 如果定义了 USE_ROCM 宏
  #ifdef USE_ROCM
    // 对函数进行 unhipify 转换，前提是使用了 std::complex
    func = unhipify_math_functions(func);
    // 将转换后的函数设置到环境变量中
    env.s("functor", func);
  #endif

  // 如果 dynamic_casting 为真
  } else if (dynamic_casting) {
    // 获取安全的 traits 字符串
    env.s("traits_string", get_traits_string_but_hiprtc_safe());
    // 获取复数体的字符串表示
    env.s("complex_body_string", get_complex_body_string());
    // 将复数数学字符串置空
    env.s("complex_math_string", "");
  } else {
    // 将 traits 字符串置空
    env.s("traits_string", "");
    // 将复数体字符串置空
    env.s("complex_body_string", "");
    // 将复数数学字符串置空
    env.s("complex_math_string", "");
  }

  // 如果输入类型或结果类型为 std::complex<at::Half> 或者 dynamic_casting 为真
  if (f_inputs_type == "std::complex<at::Half>" ||
      result_type == "std::complex<at::Half>" || dynamic_casting) {
    // 获取复数半体的字符串表示
    env.s("complex_half_body_string", get_complex_half_body_string());
  } else {
    // 将复数半体字符串置空
    env.s("complex_half_body_string", "");
  }

  // 设置 load_support 环境变量
  env.s("load_support", load_support_literal);

  // 如果不是向量化运算
  if (!vectorized) {
    // 如果不是 dynamic_casting
    if (!dynamic_casting) {
      // 设置 loader 为 "LoadWithoutCast"
      env.s("loader", "LoadWithoutCast");
      // 设置 storer 为 "StoreWithoutCast"
      env.s("storer", "StoreWithoutCast");
      // 设置 dynamic_casting_string 为空
      env.s("dynamic_casting_string", no_dynamic_cast_support_literal);
    } else {
      // 设置 loader 为 "LoadWithCast<nInputs>"
      env.s("loader", std::string("LoadWithCast<" + std::to_string(nInputs) + ">"));
      // 设置 storer 为 "StoreWithCast<nOutputs>"
      env.s("storer", std::string("StoreWithCast<" + std::to_string(nOutputs) + ">"));
      // 设置 dynamic_casting_string 为 dynamic_cast_support_literal
      env.s("dynamic_casting_string", dynamic_cast_support_literal);
    }

    // 如果是连续存储
    if (contiguous) {
      // 设置 offset_calculator 为 "TrivialOffsetCalculator"
      env.s("offset_calculator", "TrivialOffsetCalculator");
    } else {
      // 设置 offset_calculator 为 "OffsetCalculator"
      env.s("offset_calculator", "OffsetCalculator");
    }

    // 生成加载输入的代码块
    std::stringstream load_inputs;
    for (int i = 0; i < nInputs; i++) {
      auto i_string = std::to_string(i);
      load_inputs << "arg" << i_string << "[j] = l.load<" << f_inputs_type
                  << ">(data[" << std::to_string(i + nOutputs)
                  << "], input_offsets[" << i_string << "], " << i_string
                  << ");\n";
    }
    // 将生成的加载输入代码块设置到环境变量 load_inputs 中
    env.s("load_inputs", load_inputs.str());

    // 生成存储输出的代码块
    std::stringstream store_outputs;
    for (int i = 0; i < nOutputs; i++) {
      auto i_string = std::to_string(i);
      store_outputs << "s.store<" << result_type
                    << ">(out" << i_string << "[j], data[" << i_string
                    << "], output_offsets[" << i_string << "], " << i_string
                    << ");\n";
    }
    // 将生成的存储输出代码块设置到环境变量 store_outputs 中
    env.s("store_outputs", store_outputs.str());

    // 定义静态的 CUDA 模板
    static auto cuda_template = at::jit::CodeTemplate(
      jit_preamble + jit_common_types + offset_calc_template + jit_code_template + jit_epilogue);
    // 格式化 CUDA 模板并返回生成的代码
    const auto code = cuda_template.format(env);
    return code;
  }

  // 向量化情况下的处理
  // 设置向量大小到环境变量 vec_size 中
  env.s("vec_size", std::to_string(vec_size));
  // 设置结果类型到环境变量 result_type 中
  // 构建向量输入的字符串流，每次迭代添加一个输入的重解释转换操作
  vector_inputs << "auto * input" << i_string <<
      " = reinterpret_cast<const scalar_t*>(data[" << i_string << "+" << nOutputs << "])" <<
      " + block_work_size * idx;\n";
}
// 将构建好的字符串流存入环境中，以便后续使用
env.s("vector_inputs", vector_inputs.str());

// 构建向量输出的字符串流，每次迭代添加一个输出的重解释转换操作
std::stringstream vector_outputs;
for (const auto i : c10::irange(nOutputs)){
  auto i_string = std::to_string(i);
  vector_outputs << "vec_t_output* to_" << i_string <<
  " = reinterpret_cast<vec_t_output*>(data[" << i_string << "])" <<
  " + block_work_size / vec_size * idx;\n";
}
// 将构建好的字符串流存入环境中，以便后续使用
env.s("vector_outputs", vector_outputs.str());

// 构建加载向量化输入的字符串流
std::stringstream load_vectorized_inputs;
for (const auto i : c10::irange(nInputs)) {
  auto i_string = std::to_string(i);
  load_vectorized_inputs << "const auto vec" << i_string << " = load_vector<vec_size>("
                         << "input" << i_string << ", thread_idx);\n";
  load_vectorized_inputs << "#pragma unroll\n";
  load_vectorized_inputs << "for (int j=0; j < vec_size; j++){\n";
  load_vectorized_inputs << "  arg" << i_string << "[vec_size * i + j] = vec" << i_string << ".val[j];\n";
  load_vectorized_inputs << "}\n";
}
// 将构建好的字符串流存入环境中，以便后续使用
env.s("load_vectorized_inputs", load_vectorized_inputs.str());

// 构建存储向量化输出的字符串流
std::stringstream store_vectorized_outputs;
for (const auto i : c10::irange(nOutputs)) {
  auto i_string = std::to_string(i);
  store_vectorized_outputs << "#pragma unroll\n";
  store_vectorized_outputs << "for (int j=0; j<vec_size; j++){\n";
  store_vectorized_outputs <<   "v.val[j] = out" << i_string << "[vec_size * i + j];\n";
  store_vectorized_outputs << "}\n";
  store_vectorized_outputs << "to_"<< i_string << "[thread_idx] = v;\n";
}
// 将构建好的字符串流存入环境中，以便后续使用
env.s("store_vectorized_outputs", store_vectorized_outputs.str());

// 构建加载非向量化输入的字符串流
std::stringstream load_unrolled_inputs;
for (const auto i: c10::irange(nInputs)){
  auto i_string = std::to_string(i);
  load_unrolled_inputs << "arg" << i_string << "[j] = load<" << f_inputs_type
    << ">(data[" << std::to_string(i + nOutputs) << "], linear_idx);\n";
}
// 将构建好的字符串流存入环境中，以便后续使用
env.s("load_unrolled_inputs", load_unrolled_inputs.str());

// 构建存储非向量化输出的字符串流
std::stringstream store_unrolled_outputs;
for (const auto i : c10::irange(nOutputs)) {
  auto i_string = std::to_string(i);
  store_unrolled_outputs << "store<" << result_type << ">(out" << i_string
    << "[j], data[" << i_string << "], linear_idx);\n";
}
// 将构建好的字符串流存入环境中，以便后续使用
env.s("store_unrolled_outputs", store_unrolled_outputs.str());

// 定义 CUDA 模板，包括预言文本、公共类型、向量化代码模板和结尾语句
static auto cuda_template = at::jit::CodeTemplate(
  jit_preamble + jit_common_types + jit_vectorized_code_template + jit_epilogue);
// 根据环境变量填充 CUDA 模板并生成最终的代码
const auto code = cuda_template.format(env);
// 返回生成的 CUDA 代码
return code;
// 递归创建目录
bool _r_mkdir(const std::string& dir) {
    // 检查当前目录是否存在
    const char* p_dir = dir.c_str();
    const bool dir_exists = (access(p_dir, F_OK) == 0);
    if (dir_exists) {
        return true;
    }

    // 尝试创建当前目录
#ifdef _WIN32
    int ret = _mkdir(dir.c_str());
#else
    int ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif
    // 成功创建
    if (ret == 0) {
        return true;
    }

    // 找到目录分隔符并检查是否位于顶层目录
    auto pos = dir.find_last_of("/\\");
    if (pos == std::string::npos) {
        return false;
    }

    // 尝试创建父目录
    if (!(_r_mkdir(dir.substr(0, pos)))) {
        return false;
    }

    // 再次尝试创建完整路径
#ifdef _WIN32
    ret = _mkdir(dir.c_str());
#else
    ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif
    return ret == 0;
}

// 基于已存在的基础目录递归创建目录
bool r_mkdir_with_base(std::string& base, std::string& dir){
    const char* p_base = base.c_str();
    const bool base_exists = (access(p_base, F_OK) == 0);
    if (!base_exists) {
        return false;
    }

    // 移除末尾的 '/' 或 '\\'
    if ((base[base.size()-1]=='/') || base[base.size()-1]=='\\') {
        base.pop_back();
    }
    if ((dir[dir.size()-1]=='/') || dir[dir.size()-1]=='\\') {
        dir.pop_back();
    }

    return _r_mkdir(base+dir);
}

// 加载指定路径的代码模板文件内容
std::string load_code_template(const std::string& path) {
    std::ifstream ifs{path};
    std::string s{
        std::istreambuf_iterator<char>(ifs),
        std::istreambuf_iterator<char>()};
    return s;
}

// 生成归约操作的代码
std::string generate_reduction_code(
    const KernelDescriptor &desc,
    int vt0,
    bool contiguous,
    bool vectorized,
    int vec_size,
    int max_threads_codegen) {
    TORCH_INTERNAL_ASSERT(desc.nInputs == 1);
    TORCH_INTERNAL_ASSERT(desc.extra_args_types.size() == 0);

    return generate_reduction_code(
        desc.nOutputs,
        desc.f,
        desc.name,
        vt0,
        typeName(desc.f_inputs_type),
        typeName(toOpMathType(desc.f_inputs_type)),
        typeName(desc.result_type),
        contiguous,
        vectorized,
        vec_size,
        max_threads_codegen
    );
}

// 生成归约操作的代码，重载版本
std::string generate_reduction_code(
    int nOutputs,
    const std::string& func_,
    const std::string& name,
    const int vt0,
    const std::string& f_inputs_type,
    const std::string& reduction_accum_type,
    const std::string& result_type,
    bool contiguous,
    bool vectorized,
    int vec_size,
    int max_threads_codegen) {
    // 内部断言，确保输入和参数类型匹配
    TORCH_INTERNAL_ASSERT(nOutputs == 1); // 假设这里是为了说明
    TORCH_INTERNAL_ASSERT(nOutputs >= 1); // 假设这里是为了说明

    // 调用具体的归约代码生成函数，返回生成的代码字符串
    return generate_reduction_code(
        nOutputs,
        func_,
        name,
        vt0,
        f_inputs_type,
        reduction_accum_type,
        result_type,
        contiguous,
        vectorized,
        vec_size,
        max_threads_codegen
    );
}
    // 将函数名存储到局部变量 func 中
    std::string func = func_;
    // 创建一个模板环境对象 env，用于替换模板中的字符串变量
    at::jit::TemplateEnv env;
    // 设置模板环境中的 index_type 变量为 "unsigned int"
    env.s("index_type", "unsigned int");
    // 设置模板环境中的 scalar_type 变量为 f_inputs_type
    env.s("scalar_type", f_inputs_type);
    // 设置模板环境中的 result_type 变量为 result_type
    env.s("result_type", result_type);
    // 设置模板环境中的 reduction_accum_type 变量为 reduction_accum_type
    env.s("reduction_accum_type", reduction_accum_type);
    // 将整数 vt0 转换为字符串并设置为模板环境中的 vt0 变量
    env.s("vt0", std::to_string(vt0));
    // 将字符串 name 设置为模板环境中的 name 变量
    env.s("name", name);
    // 将最大线程数转换为字符串并设置为模板环境中的 max_threads_lb 变量
    env.s("max_threads_lb", std::to_string(max_threads_codegen));
    
    // 如果输入类型为 "at::Half" 或 "std::complex<at::Half>"
    if (f_inputs_type == "at::Half" || f_inputs_type == "std::complex<at::Half>") {
        // 设置 jiterator_half_support_literal 为模板环境中的 half_string 变量
        env.s("half_string", jiterator_half_support_literal);
    } else {
        // 否则将 half_string 设置为空字符串
        env.s("half_string", "");
    }
    
    // 如果输入类型为 "at::BFloat16"
    if (f_inputs_type == "at::BFloat16") {
        // 设置 jiterator_bfloat16_support_literal 为模板环境中的 bfloat16_string 变量
        env.s("bfloat16_string", jiterator_bfloat16_support_literal);
    } else {
        // 否则将 bfloat16_string 设置为空字符串
        env.s("bfloat16_string", "");
    }
    
    // 如果输入类型为 "std::complex<float>", "std::complex<double>", 或 "std::complex<at::Half>"
    if (f_inputs_type == "std::complex<float>" ||
        f_inputs_type == "std::complex<double>" ||
        f_inputs_type == "std::complex<at::Half>" ) {
        // 调用函数 get_traits_string_but_hiprtc_safe() 并设置返回值为 traits_string 变量
        env.s("traits_string", get_traits_string_but_hiprtc_safe());
        // 调用函数 get_complex_body_string() 并设置返回值为 complex_body_string 变量
        env.s("complex_body_string", get_complex_body_string());
        // 调用函数 get_complex_math_string() 并设置返回值为 complex_math_string 变量
        env.s("complex_math_string", get_complex_math_string());
        // 设置模板环境中的 complex 变量为整数 1 的字符串表示
        env.s("complex", std::to_string(1));
#ifdef USE_ROCM
        // 如果定义了 USE_ROCM 宏，则执行以下代码块，用于处理数学函数的适配，仅当使用 std::complex 时有效。
        func = unhipify_math_functions(func);
#endif
      } else {
        // 如果未定义 USE_ROCM 宏，则将环境变量中的以下字符串置为空。
        env.s("traits_string", "");
        env.s("complex_body_string", "");
        env.s("complex_math_string", "");
        env.s("complex", std::to_string(0));
      }
      // 如果输入类型为 "std::complex<at::Half>"，则获取半精度复数的字符串表示。
      if (f_inputs_type == "std::complex<at::Half>") {
        env.s("complex_half_body_string", get_complex_half_body_string());
      } else {
        // 否则，将半精度复数的字符串置为空。
        env.s("complex_half_body_string", "");
      }
      // 获取标准数学库的字符串表示。
      env.s("cmath_string", get_cmath_string());
      // 将函数名设为环境变量中的 "functor" 键。
      env.s("functor", func);
      // 将输出向量的大小设为环境变量中的 "output_vec_size" 键。
      env.s("output_vec_size", std::to_string(vec_size));
      // 创建静态的 CUDA 模板，格式化环境变量中的内容。
      static auto cuda_template = at::jit::CodeTemplate(
        jit_preamble + jit_common_types + offset_calc_template + get_reduction_template() + jit_epilogue);
      // 根据 CUDA 模板格式化后的内容，存储到变量 code 中。
      const auto code = cuda_template.format(env);
      // 返回生成的代码字符串。
      return code;
}

// 获取（可能创建）内核缓存目录
std::optional<std::string> get_cache_dir() {
  // 如果环境变量 USE_PYTORCH_KERNEL_CACHE 的值为 "0"，则不使用持久化缓存
  const char* uptkc = std::getenv("USE_PYTORCH_KERNEL_CACHE");
  const bool use_kernel_cache = (uptkc == nullptr) ? true : std::strcmp(uptkc, "0");

  // 如果不使用内核缓存，则返回空 optional
  if (!use_kernel_cache) {
    return {};
  }

  // 获取内核缓存路径，优先从 PYTORCH_KERNEL_CACHE_PATH 环境变量获取，然后是 TEMP（Windows）或 XDG_CACHE_HOME（Linux），最后是 HOME
  std::string cache_dir;
  char* ptkcp = std::getenv("PYTORCH_KERNEL_CACHE_PATH");
  std::string kernels_cache_dir = "";
  if (ptkcp != nullptr) {
    cache_dir = std::string(ptkcp);
  } else {
#ifdef _WIN32
    ptkcp = std::getenv("TEMP");
#else
    ptkcp = std::getenv("XDG_CACHE_HOME");
#endif
    if (ptkcp != nullptr) {
      kernels_cache_dir = "/torch/kernels";
      cache_dir = std::string(ptkcp) + kernels_cache_dir;
    } else {
      // 如果未设置 XDG_CACHE_HOME，则回退到 HOME/.cache
      ptkcp = std::getenv("HOME");
      if (ptkcp == nullptr) {
        // 如果 HOME 环境变量未设置，发出警告并返回空 optional，禁用内核缓存
        TORCH_WARN_ONCE("No PYTORCH_KERNEL_CACHE_PATH or HOME environment variable set!",
                        " This disables kernel caching.");
        return {};
      } else {
        kernels_cache_dir = "/.cache/torch/kernels";
        cache_dir = std::string(ptkcp) + kernels_cache_dir;
      }
    }
  }

  // 如果缓存目录不存在，则创建之
  const char* p_cache_dir = cache_dir.c_str();
  const bool cache_dir_exists = (access(p_cache_dir, F_OK) == 0);
  if (!cache_dir_exists) {
    std::string s_ptkcp = std::string(ptkcp);
    // 如果创建目录失败，则打印警告信息并返回空字典
    if (!r_mkdir_with_base(s_ptkcp, kernels_cache_dir)) {
      TORCH_WARN_ONCE("Specified kernel cache directory could not be created! This disables kernel caching.",
                      " Specified directory is ", cache_dir, ".",
                      " This warning will appear only once per process.");
      return {};
    }
  }

  // 检查缓存目录是否可读
  const bool cache_dir_readable = (access(p_cache_dir, R_OK) == 0);
  // 如果缓存目录不可读，则打印警告信息并返回空字典
  if (!cache_dir_readable) {
    TORCH_WARN_ONCE("Specified kernel cache directory is not readable! This disables kernel caching.",
                    " Specified directory is ", cache_dir, ".",
                    " This warning will appear only once per process.");
    return {};
  }

  // 检查缓存目录是否可写
  const bool cache_dir_writable = (access(p_cache_dir, W_OK) == 0);
  // 如果缓存目录不可写，则打印警告信息并返回空字典
  if (!cache_dir_writable) {
    TORCH_WARN_ONCE("Specified kernel cache directory is not writable! This disables kernel caching.",
                    " Specified directory is ", cache_dir, ".",
                    " This warning will appear only once per process.");
    return {};
  }

  // 如果缓存目录可读可写，则返回缓存目录路径
  return cache_dir;
}

// 编译内核，或者从缓存中获取已编译好的内核
NvrtcFunction jit_pwise_function(
    const std::string& code,
    const std::string& kernel_name) {
  // 初始化 CUDA 上下文
  initializeCudaContext();
  
  // 获取当前 CUDA 设备的属性
  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  int cuda_major = 0, cuda_minor = 0, nvrtc_major = 0, nvrtc_minor = 0;
  
  // 获取 CUDA 和 NVRTC 版本以及编译目标（ptx 或 SASS）
  bool compile_to_sass = false;
  at::cuda::jit::codegenOutputQuery(
    prop, cuda_major, cuda_minor, nvrtc_major, nvrtc_minor, compile_to_sass);

  // 获取全局的 NVRTC 上下文
  const auto& nvrtc = at::globalContext().getNVRTC();
  
  // 定义编译后的内核对象
  NvrtcFunction compiled_kernel_;

  // 构造内核函数名称
  std::string name = kernel_name + "_kernel";

  // 获取缓存目录（如果可用）
  static const std::optional<std::string> cache_dir = get_cache_dir();

  std::string file_path;

  if (cache_dir.has_value()) {
    // 尝试从缓存中读取

    // 计算代码的 SHA1 哈希值
    c10::sha1 sha1_hash{code};
    const auto hash_code = sha1_hash.str();

    // 构造缓存文件路径
    std::stringstream ss;
    ss << *cache_dir << "/";
    ss << kernel_name;
#ifdef USE_ROCM
    ss << "_arch" << prop->gcnArchName;
#else
    ss << "_arch" << cuda_major << "." << cuda_minor;
#endif
    ss << "_nvrtc" << nvrtc_major << "." << nvrtc_minor;
    ss << (compile_to_sass ? "_sass" : "_ptx");
    ss << "_" << code.length();
    ss << "_" << hash_code;
    file_path = ss.str();

    // 尝试打开文件
    std::ifstream readin{file_path, std::ios::in | std::ifstream::binary};

    if (readin.fail()) {
      // 注意：这里不会发出警告，因为文件可能不存在
      // TODO: 考虑是否应该显式检查文件是否存在，以便抛出相关警告
      readin.close();
    } else {
      // 从缓存中读取成功，直接加载并返回编译好的内核
      std::vector<char> buffer(std::istreambuf_iterator<char>(readin), {});
      AT_CUDA_DRIVER_CHECK(nvrtc.cuModuleLoadData(&(compiled_kernel_.module), buffer.data()));
      AT_CUDA_DRIVER_CHECK(
        nvrtc.cuModuleGetFunction(&(compiled_kernel_.function), compiled_kernel_.module, name.c_str()));
      readin.close();
      return compiled_kernel_;
    }
  }

  // 没有从缓存中获取到，进行即时编译

  // 创建 NVRTC 程序
  nvrtcProgram program;
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));

#ifdef USE_ROCM
  // ROCm 平台需要额外参数
  std::vector<const char*> args = {"--std=c++17"};
#else
  // 构建 nvrtc 编译参数
  // CUDA 11.1 允许直接使用 SASS（sm_）而不是 PTX（compute_）
  // 这提供了更好的向后兼容性，可以在旧驱动程序上运行（因为旧驱动程序不一定能识别新工具包发出的 PTX）；
  // 同时，为了向前兼容性（未来可能的设备 with `unsupported_arch==True`），因为 SASS 不一定兼容，
  // 我们会回退到使用 PTX。
  const std::string compute = std::string("--gpu-architecture=") +
      (compile_to_sass ? "sm_" : "compute_") + std::to_string(cuda_major) +
      std::to_string(cuda_minor);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // args 是存放编译参数的 vector
  std::vector<const char*> args = {
      "--std=c++17", compute.c_str(), "-default-device"};
#endif

#ifndef NDEBUG
  // 向生成的内核代码中添加行信息
  args.push_back("-lineinfo");
#else
  // 为了避免断言导致的寄存器使用过多
  args.push_back("-DNDEBUG");
#endif

// 使用编译后的参数调用 nvrtcCompileProgram 进行编译
const auto compilation_result =
    nvrtc.nvrtcCompileProgram(program, args.size(), args.data());

// 在编译失败时抛出错误
if (compilation_result != NVRTC_SUCCESS) {
  size_t logsize;
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetProgramLogSize(program, &logsize));
  std::string log(logsize, '\0');
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetProgramLog(program, &log[0]));
  throw std::runtime_error(code + log);
}

size_t ptx_size = 0;
std::vector<char> ptx;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11010
  // compile_to_sass 确定我们是生成 SASS 还是 PTX，因此使用不同的 API。
  const auto getSize = compile_to_sass
      ? at::globalContext().getNVRTC().nvrtcGetCUBINSize
      : at::globalContext().getNVRTC().nvrtcGetPTXSize;
  const auto getFunc = compile_to_sass
      ? at::globalContext().getNVRTC().nvrtcGetCUBIN
      : at::globalContext().getNVRTC().nvrtcGetPTX;
#else
  const auto getSize = at::globalContext().getNVRTC().nvrtcGetPTXSize;
  const auto getFunc = at::globalContext().getNVRTC().nvrtcGetPTX;
#endif

// 获取编译生成的 PTX 大小，并分配相应大小的内存
AT_CUDA_NVRTC_CHECK(getSize(program, &ptx_size));
ptx.resize(ptx_size);
// 获取编译生成的 PTX 内容
AT_CUDA_NVRTC_CHECK(getFunc(program, ptx.data()));

// 使用 cuModuleLoadData 将 PTX 数据加载为 CUDA 模块
AT_CUDA_DRIVER_CHECK(nvrtc.cuModuleLoadData(&(compiled_kernel_.module), ptx.data()));

// 获取编译后函数的句柄
AT_CUDA_DRIVER_CHECK(
    nvrtc.cuModuleGetFunction(&(compiled_kernel_.function), compiled_kernel_.module, name.c_str()));
// TODO: 使用守卫避免内存泄漏
AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcDestroyProgram(&program));

if (cache_dir.has_value()) {
  // 如果启用了缓存，将程序写入缓存
  // 注意：实际上是写入一个每个进程的临时文件，以避免多进程争用。
  //   该临时文件然后被重命名为实际文件。
  //   如果实际文件已经存在，则重命名可能失败或替换实际文件，
  //     具体行为取决于实现。
    // 使用当前进程的PID生成临时文件路径
    const auto pid = getpid();
    std::stringstream tmp_file_path_ss;
    tmp_file_path_ss << file_path << "_tmp_" << pid;
    const std::string tmp_file_path = tmp_file_path_ss.str();

    // 打开临时文件用于写入二进制数据
    std::ofstream cubin(tmp_file_path, std::ios::out | std::ofstream::binary);

    // 检查文件是否成功打开，如果失败则发出警告
    if (cubin.fail()) {
      TORCH_WARN_ONCE("Failed to write temporarily kernel cache file!",
                      " File path was ", tmp_file_path, ".",
                      " This warning will only appear once per process.");
    } else {
      // 将PTX数据拷贝到临时文件中
      std::copy(ptx.begin(), ptx.end(), std::ostreambuf_iterator<char>(cubin));

      // 尝试重命名临时文件为目标文件名，如果失败则删除临时文件
      if (std::rename(tmp_file_path.c_str(), file_path.c_str()) != 0) {
        std::remove(tmp_file_path.c_str());  // 如果重命名失败则删除临时文件
      }
    }

    // 关闭文件流
    cubin.close();
  }

  // 返回编译后的内核对象
  return compiled_kernel_;
// 初始化 CUDA 上下文，准备执行 CUDA 函数
void launch_jitted_pwise_function(
    NvrtcFunction function,         // NVRTC 编译后的 CUDA 函数
    void* args[],                   // 函数参数数组
    const dim3 nBlocks,             // CUDA 核心的块数
    const dim3 kBlockSize,          // 每个 CUDA 核心的线程块大小
    const int smem) {               // 分配给共享内存的字节数

  // 初始化 CUDA 上下文，确保 CUDA 环境准备就绪
  initializeCudaContext();

  // 获取全局的 NVRTC 编译器实例
  const auto& nvrtc = at::globalContext().getNVRTC();

  // 获取当前 CUDA 流
  auto stream = at::cuda::getCurrentCUDAStream();

  // 使用 NVRTC 接口调用 CUDA 驱动程序函数 cuLaunchKernel 启动 CUDA 核函数
  AT_CUDA_DRIVER_CHECK(nvrtc.cuLaunchKernel(
    function.function,      // 要执行的 CUDA 函数句柄
    nBlocks.x,              // 核心块的 x 维度大小
    nBlocks.y,              // 核心块的 y 维度大小
    nBlocks.z,              // 核心块的 z 维度大小
    kBlockSize.x,           // 每个核心块中线程块的 x 维度大小
    kBlockSize.y,           // 每个核心块中线程块的 y 维度大小
    kBlockSize.z,           // 每个核心块中线程块的 z 维度大小
    smem,                   // 分配给共享内存的字节数
    stream,                 // 执行核函数的 CUDA 流
    args,                   // 传递给核函数的参数数组
    nullptr));              // 不使用额外的配置选项

} // at::cuda::jit
```