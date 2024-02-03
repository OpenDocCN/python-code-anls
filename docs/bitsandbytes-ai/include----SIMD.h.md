# `bitsandbytes\include\SIMD.h`

```py
#pragma once

#include "Portable.h"

#ifdef USE_SSE42
#ifndef _MSC_VER
#include <popcntintrin.h>
#define popcnt32 _mm_popcnt_u32
#else
#include <intrin.h>
#define popcnt32 __popcnt
#endif
#else // USE_SSE42
namespace BinSearch {
FORCE_INLINE int popcnt32(int x32)
{
    // 严格来说，这不是正确的，因为它忽略了高阶位
    // 但是，这仅用于对128位寄存器上movemask的结果进行操作，最多为8位，所以可以接受
    // 对于256位寄存器，SSE42已经定义，我们不使用这个函数
    uint8 x = static_cast<uint8>(x32);
    x = (x & 0x55) + (x >> 1 & 0x55);
    x = (x & 0x33) + (x >> 2 & 0x33);
    x = (x & 0x0f) + (x >> 4 & 0x0f);
    return x;
}
} // namespace
#endif

#if defined(USE_AVX) || defined(USE_AVX2)
#include <immintrin.h>
#else
#include <emmintrin.h>
#ifdef USE_SSE41
#include <smmintrin.h>
#endif
#endif

#include "Type.h"

namespace BinSearch {
namespace Details {

template <InstrSet I, class T>
struct FVec;

template <InstrSet I, class T>
struct IVec;

template <InstrSet I, class T>
struct FVec1;

template <> struct InstrIntTraits<SSE>
{
    typedef __m128i vec_t;
};

template <> struct InstrFloatTraits<SSE, float>
{
    typedef __m128  vec_t;
};

template <> struct InstrFloatTraits<SSE, double>
{
    typedef __m128d vec_t;
};

template <> struct InstrFloatTraits<Scalar, float>
{
    typedef float  vec_t;
};

template <> struct InstrFloatTraits<Scalar, double>
{
    typedef double vec_t;
};

template <InstrSet I, typename T>
struct FTOITraits
{
    typedef IVec<SSE, float> vec_t;
};

#ifdef USE_AVX

template <>
struct FTOITraits<AVX, float>
{
    typedef IVec<AVX, float> vec_t;
};

template <> struct InstrIntTraits<AVX>
{
    typedef __m256i vec_t;
};

template <> struct InstrFloatTraits<AVX, float>
{
    typedef __m256  vec_t;
};

template <> struct InstrFloatTraits<AVX, double>
{
    typedef __m256d vec_t;
};

#endif


template <typename TR>
struct VecStorage
{
    # 定义一个类型别名，将TR::vec_t赋值给vec_t
    typedef typename TR::vec_t vec_t;

    # 强制内联转换操作符，将vec转换为vec_t的引用并返回
    FORCE_INLINE operator vec_t&() { return vec; }
    
    # 强制内联转换操作符，将vec转换为const vec_t的引用并返回
    FORCE_INLINE operator const vec_t&() const { return vec; }
protected:
    // 默认构造函数，不做任何操作
    FORCE_INLINE VecStorage() {}
    // 接受一个向量作为参数的构造函数
    FORCE_INLINE VecStorage(const vec_t& v) : vec( v ) {}

    // 存储向量数据的成员变量
    vec_t vec;
};

// 模板特化，针对 SSE 指令集
template <InstrSet>
struct IVecBase;

template <>
struct IVecBase<SSE> : VecStorage<InstrIntTraits<SSE>>
{
protected:
    // 默认构造函数，不做任何操作
    FORCE_INLINE IVecBase() {}
    // 接受一个向量作为参数的构造函数
    FORCE_INLINE IVecBase( const vec_t& v) : VecStorage<InstrIntTraits<SSE>>( v ) {}
public:
    // 返回一个全零的向量
    FORCE_INLINE static vec_t zero() { return _mm_setzero_si128(); }

    // 获取向量中的第一个元素
    FORCE_INLINE int32 get0() const { return _mm_cvtsi128_si32( vec ); }

    // 根据掩码条件，将 val 赋值给向量中的元素
    FORCE_INLINE void assignIf( const vec_t& val, const vec_t& mask )
    {
#ifdef USE_SSE41
        // 使用 SSE4.1 指令集的混合赋值操作
        vec = _mm_blendv_epi8(vec, val, mask);
#else
        // 使用 SSE2 指令集的位运算实现混合赋值操作
        vec = _mm_or_si128(_mm_andnot_si128(mask,vec), _mm_and_si128(mask,val));
#endif
    }
    // 根据掩码条件，将 val 和向量中的元素进行或操作
    FORCE_INLINE void orIf(const vec_t& val, const vec_t& mask)
    {
        vec = _mm_or_si128(vec, _mm_and_si128(val,mask));
    }
};

// 模板特化，针对 SSE 指令集和 float 类型
template <>
struct IVec<SSE, float> : IVecBase<SSE>
{
    // 默认构造函数
    FORCE_INLINE IVec() {}
    // 接受一个整数作为参数的构造函数
    FORCE_INLINE IVec( int32 i ) : IVecBase<SSE>( _mm_set1_epi32( i ) )  {}
    // 接受一个向量作为参数的构造函数
    FORCE_INLINE IVec( const vec_t& v) : IVecBase<SSE>( v )              {}
    // 接受四个整数作为参数的构造函数
    FORCE_INLINE IVec( uint32 u3, uint32 u2, uint32 u1, uint32 u0) : IVecBase<SSE>( _mm_set_epi32( u3, u2, u1, u0 ) ) {}

    // 设置向量中的所有元素为相同的值
    void setN( int32 i ) { vec = _mm_set1_epi32( i ); }

#ifdef USE_SSE41
    // 获取向量中的第二个元素
    FORCE_INLINE int32 get1() const { return _mm_extract_epi32(vec, 1); }
    // 获取向量中的第三个元素
    FORCE_INLINE int32 get2() const { return _mm_extract_epi32(vec, 2); }
    // 获取向量中的第四个元素
    FORCE_INLINE int32 get3() const { return _mm_extract_epi32(vec, 3); }
#else
    // 获取向量中的第二个元素
    FORCE_INLINE int32 get1() const { return _mm_cvtsi128_si32( _mm_shuffle_epi32( vec, 1 ) ); }
    // 获取向量中的第三个元素
    FORCE_INLINE int32 get2() const { return _mm_cvtsi128_si32( _mm_shuffle_epi32( vec, 2 ) ); }
    // 获取向量中的第四个元素
    FORCE_INLINE int32 get3() const { return _mm_cvtsi128_si32( _mm_shuffle_epi32( vec, 3 ) ); }
#endif

    // 将向量数据存储到指定的内存地址
    FORCE_INLINE void store( uint32 *pi ) const { _mm_storeu_si128( reinterpret_cast<vec_t*>(pi), vec ); }

    // 计算向量中非零元素的个数
    FORCE_INLINE int countbit()
    {
        # 将 128 位整数向量转换为 4 个单精度浮点数向量
        return popcnt32(_mm_movemask_ps(_mm_castsi128_ps(vec)));
    }
// 结构体模板定义，用于处理 SSE 指令集下的 double 类型数据
struct IVec<SSE, double> : IVecBase<SSE>
{
    // 默认构造函数
    FORCE_INLINE IVec() {}
    // 构造函数，将整数 i 转换为 SSE 数据
    FORCE_INLINE IVec( int32 i ) : IVecBase<SSE>( _mm_set1_epi64x( i ) )    {}
    // 构造函数，将 vec_t 类型数据 v 转换为 SSE 数据
    FORCE_INLINE IVec( const vec_t& v) : IVecBase<SSE>( v )                 {}
    // 构造函数，将两个 uint64 类型数据 u1, u0 转换为 SSE 数据
    FORCE_INLINE IVec( uint64 u1, uint64 u0 ) : IVecBase<SSE>( _mm_set_epi64x(u1, u0) ) {}

    // 设置 SSE 数据的值为整数 i
    void setN( int32 i ) { vec = _mm_set1_epi64x( i ); }

    // 获取 SSE 数据中的第一个整数值
    FORCE_INLINE int32 get1() const
    {
#ifdef USE_SSE41
        return _mm_extract_epi32(vec, 2);
#else
        return _mm_cvtsi128_si32( _mm_shuffle_epi32( vec, 2 ) );
#endif
    }

    // 从 SSE 数据中提取低位的两个 32 位整数，存储到 __m128i 类型数据中
    FORCE_INLINE IVec<SSE,float> extractLo32s() const
    {
        return _mm_shuffle_epi32(vec, ((2 << 2) | 0));
    }

    // 将 SSE 数据存储到 uint32 类型指针中
    FORCE_INLINE void store( uint32 *pi ) const
    {
        pi[0] = get0();
        pi[1] = get1();
    }

    // 计算 SSE 数据中非零位的个数
    FORCE_INLINE int countbit()
    {
#if 1
        // 使用 SSE 指令计算非零位的个数
        __m128i hi = _mm_shuffle_epi32(vec, 2);  // 1 cycle
        __m128i s = _mm_add_epi32(vec, hi);
        int32 x = _mm_cvtsi128_si32(s);
        return -x;
#else
        // 使用 popcnt32 函数计算非零位的个数
        return popcnt32(_mm_movemask_pd(_mm_castsi128_pd(vec)));
#endif
    }
};

// 位运算符重载
template <typename T>
FORCE_INLINE IVec<SSE,T> operator>> (const IVec<SSE,T>& a, unsigned n)            { return _mm_srli_epi32(a, n); }
template <typename T>
FORCE_INLINE IVec<SSE,T> operator<< (const IVec<SSE,T>& a, unsigned n)            { return _mm_slli_epi32(a, n); }
template <typename T>
FORCE_INLINE IVec<SSE,T> operator&  (const IVec<SSE,T>& a, const IVec<SSE,T>& b ) { return _mm_and_si128( a, b ); }
template <typename T>
FORCE_INLINE IVec<SSE,T> operator|  (const IVec<SSE,T>& a, const IVec<SSE,T>& b ) { return _mm_or_si128( a, b ); }
template <typename T>
FORCE_INLINE IVec<SSE,T> operator^  (const IVec<SSE,T>& a, const IVec<SSE,T>& b ) { return _mm_xor_si128( a, b ); }
template <typename T>
// 定义 SSE 指令集下的整型向量加法操作符重载函数
FORCE_INLINE IVec<SSE,T> operator+  (const IVec<SSE,T>& a, const IVec<SSE,T>& b ) { return _mm_add_epi32( a, b ); }

// 定义 SSE 指令集下的整型向量减法操作符重载函数
template <typename T>
FORCE_INLINE IVec<SSE,T> operator-  (const IVec<SSE,T>& a, const IVec<SSE,T>& b ) { return _mm_sub_epi32( a, b ); }

#ifdef USE_SSE41
// 当使用 SSE4.1 指令集时，定义 SSE 指令集下的整型向量最小值操作函数
template <typename T>
FORCE_INLINE IVec<SSE,T> min        (const IVec<SSE,T>& a, const IVec<SSE,T>& b ) { return _mm_min_epi32( a, b ); }
#endif

// 定义 SSE 指令集下的单精度浮点数向量存储类型
typedef VecStorage<InstrFloatTraits<SSE,float>> FVec128Float;

// 定义 SSE 指令集下的单精度浮点数向量 FVec1 结构体
template <>
struct FVec1<SSE, float> : FVec128Float
{
    FORCE_INLINE FVec1() {}
    FORCE_INLINE FVec1( float f ) : FVec128Float( _mm_load_ss( &f ) ) {}
    FORCE_INLINE FVec1( const vec_t& v ): FVec128Float( v ) {}

    FORCE_INLINE float get0() const { return _mm_cvtss_f32( vec ); }
};

// 定义 SSE 指令集下的单精度浮点数向量 FVec 结构体
template <>
struct FVec<SSE, float> : FVec128Float
{
    FORCE_INLINE FVec() {}
    FORCE_INLINE FVec( float f ) : FVec128Float( _mm_set1_ps( f ) ) {}
    FORCE_INLINE FVec( const float *v ) : FVec128Float( _mm_loadu_ps( v ) ) {}
    FORCE_INLINE FVec( const vec_t& v) : FVec128Float(v) {}
    FORCE_INLINE FVec( float f3, float f2, float f1, float f0 ) : FVec128Float( _mm_set_ps(f3, f2, f1, f0) ) {}

    void set0( float f  ) { vec = _mm_load_ss( &f ); }
    void setN( float f  ) { vec = _mm_set1_ps( f ); }

    FORCE_INLINE void setidx( const float *xi, const IVec<SSE,float>& idx )
    {
        uint32 i0 = idx.get0();
        uint32 i1 = idx.get1();
        uint32 i2 = idx.get2();
        uint32 i3 = idx.get3();
        vec = _mm_set_ps( xi[i3], xi[i2], xi[i1], xi[i0] );
    }

    FORCE_INLINE float get0() const { return _mm_cvtss_f32( vec ); }
    FORCE_INLINE float get1() const { return _mm_cvtss_f32( _mm_shuffle_ps( vec, vec, 1 ) ); }
    FORCE_INLINE float get2() const { return _mm_cvtss_f32( _mm_shuffle_ps( vec, vec, 2 ) ); }
    FORCE_INLINE float get3() const { return _mm_cvtss_f32( _mm_shuffle_ps( vec, vec, 3 ) ); }
};
# 定义 SSE 浮点数向量的加法操作符重载函数
FORCE_INLINE FVec1<SSE,float> operator+  (const FVec1<SSE,float>& a, const FVec1<SSE,float>& b) { return _mm_add_ss( a, b ); }
# 定义 SSE 浮点数向量的减法操作符重载函数
FORCE_INLINE FVec1<SSE,float> operator-  (const FVec1<SSE,float>& a, const FVec1<SSE,float>& b) { return _mm_sub_ss( a, b ); }
# 定义 SSE 浮点数向量的乘法操作符重载函数
FORCE_INLINE FVec1<SSE,float> operator*  (const FVec1<SSE,float>& a, const FVec1<SSE,float>& b) { return _mm_mul_ss( a, b ); }
# 定义 SSE 浮点数向量的除法操作符重载函数
FORCE_INLINE FVec1<SSE,float> operator/  (const FVec1<SSE,float>& a, const FVec1<SSE,float>& b) { return _mm_div_ss( a, b ); }
# 定义将 SSE 浮点数向量转换为整型的函数
FORCE_INLINE int              ftoi       (const FVec1<SSE,float>& a)                            { return _mm_cvttss_si32(a); }
# 定义 SSE 浮点数向量的大于比较操作符重载函数
FORCE_INLINE IVec<SSE,float> operator>   (const FVec1<SSE,float>& a, const FVec1<SSE,float>& b) { return _mm_castps_si128( _mm_cmpgt_ss( a, b ) ); }
# 如果定义了 USE_FMA 宏，则定义 SSE 浮点数向量的 FMA 操作函数
#ifdef USE_FMA
FORCE_INLINE FVec1<SSE, float> mulSub(const FVec1<SSE, float>& a, const FVec1<SSE, float>& b, const FVec1<SSE, float>& c) { return _mm_fmsub_ss(a, b, c); }
#endif

# 定义 SSE 浮点数向量的减法操作符重载函数
FORCE_INLINE FVec<SSE,float> operator-   (const FVec<SSE,float>& a,  const FVec<SSE,float>& b)  { return _mm_sub_ps( a, b ); }
# 定义 SSE 浮点数向量的乘法操作符重载函数
FORCE_INLINE FVec<SSE,float> operator*   (const FVec<SSE,float>& a,  const FVec<SSE,float>& b)  { return _mm_mul_ps( a, b ); }
# 定义 SSE 浮点数向量的除法操作符重载函数
FORCE_INLINE FVec<SSE,float> operator/   (const FVec<SSE,float>& a,  const FVec<SSE,float>& b)  { return _mm_div_ps( a, b ); }
# 定义将 SSE 浮点数向量转换为整型的函数
FORCE_INLINE IVec<SSE,float> ftoi        (const FVec<SSE,float>& a)                             { return _mm_cvttps_epi32(a); }
# 定义 SSE 浮点数向量的小于等于比较操作符重载函数
FORCE_INLINE IVec<SSE,float> operator<=  (const FVec<SSE,float>& a,  const FVec<SSE,float>& b)  { return _mm_castps_si128( _mm_cmple_ps( a, b ) ); }
# 定义 SSE 浮点数向量的大于等于比较操作符重载函数
FORCE_INLINE IVec<SSE,float> operator>=  (const FVec<SSE,float>& a,  const FVec<SSE,float>& b)  { return _mm_castps_si128( _mm_cmpge_ps( a, b ) ); }
# 定义 SSE 浮点数向量的小于比较操作符重载函数
FORCE_INLINE IVec<SSE,float> operator<   (const FVec<SSE,float>& a,  const FVec<SSE,float>& b)  { return _mm_castps_si128(_mm_cmplt_ps(a, b)); }
# 如果定义了 USE_FMA 宏，则继续下面的代码
#ifdef USE_FMA
// 定义 SSE 指令集下的 float 类型的 FVec 结构体的 mulSub 方法，实现向量 a 与向量 b 的乘法减法操作，结果与向量 c 相减
FORCE_INLINE FVec<SSE, float> mulSub(const FVec<SSE, float>& a, const FVec<SSE, float>& b, const FVec<SSE, float>& c) { return _mm_fmsub_ps(a, b, c); }
#endif

// 定义 SSE 指令集下的 double 类型的 FVec128Double 结构体
typedef VecStorage<InstrFloatTraits<SSE,double>> FVec128Double;

// 重载 FVec1 结构体，指定 SSE 指令集和 double 类型
template <>
struct FVec1<SSE, double> : FVec128Double
{
    // 默认构造函数
    FORCE_INLINE FVec1() {}
    // 构造函数，根据 double 类型的值 f 创建 FVec1 对象
    FORCE_INLINE FVec1( double f )       : FVec128Double( _mm_load_sd( &f ) ) {}
    // 构造函数，根据 vec_t 类型的向量 v 创建 FVec1 对象
    FORCE_INLINE FVec1( const vec_t& v ) : FVec128Double( v )                 {}

    // 获取第一个元素的值
    FORCE_INLINE double get0() const { return _mm_cvtsd_f64( vec ); }
};

// 重载 FVec 结构体，指定 SSE 指令集和 double 类型
template <>
struct FVec<SSE, double> : FVec128Double
{
    // 默认构造函数
    FORCE_INLINE FVec() {}
    // 构造函数，根据 double 类型的值 d 创建 FVec 对象
    FORCE_INLINE FVec( double d )        : FVec128Double( _mm_set1_pd( d ) )   {}
    // 构造函数，根据 double 类型的数组 v 创建 FVec 对象
    FORCE_INLINE FVec( const double *v ) : FVec128Double( _mm_loadu_pd( v ) )  {}
    // 构造函数，根据 vec_t 类型的向量 v 创建 FVec 对象
    FORCE_INLINE FVec( const vec_t& v)   : FVec128Double( v )                  {}
    // 构造函数，根据两个 double 类型的值 f1 和 f0 创建 FVec 对象
    FORCE_INLINE FVec( double f1, double f0 ) : FVec128Double( _mm_set_pd(f1, f0) ) {}

    // 设置第一个元素的值
    void set0( double f  ) { vec = _mm_load_sd( &f ); }
    // 设置所有元素的值为 f
    void setN( double f  ) { vec = _mm_set1_pd( f ); }

    // 根据索引数组 xi 和索引向量 idx 设置向量的值
    FORCE_INLINE void setidx( const double *xi, const IVec<SSE,double>& idx )
    {
        vec = _mm_set_pd( xi[idx.get1()], xi[idx.get0()] );
    }

    // 获取第一个元素的值
    FORCE_INLINE double get0() const { return _mm_cvtsd_f64( vec ); }
    // 获取第二个元素的值
    FORCE_INLINE double get1() const { return _mm_cvtsd_f64( _mm_shuffle_pd( vec, vec, 1 ) ); };
};

// 重载加法运算符，实现 FVec1 之间的加法操作
FORCE_INLINE FVec1<SSE,double> operator+   (const FVec1<SSE,double>& a, const FVec1<SSE,double>& b) { return _mm_add_sd( a, b ); }
// 重载减法运算符，实现 FVec1 之间的减法操作
FORCE_INLINE FVec1<SSE,double> operator-   (const FVec1<SSE,double>& a, const FVec1<SSE,double>& b) { return _mm_sub_sd( a, b ); }
// 重载乘法运算符，实现 FVec1 之间的乘法操作
FORCE_INLINE FVec1<SSE,double> operator*   (const FVec1<SSE,double>& a, const FVec1<SSE,double>& b) { return _mm_mul_sd( a, b ); }
// 重载除法运算符，实现 FVec1 之间的除法操作
FORCE_INLINE FVec1<SSE,double> operator/   (const FVec1<SSE,double>& a, const FVec1<SSE,double>& b) { return _mm_div_sd( a, b ); }
// 将单精度浮点数向下取整转换为整数
FORCE_INLINE int ftoi(const FVec1<SSE, double>& a) { return _mm_cvttsd_si32(a); }

// 比较两个单精度浮点数向量的大于关系，返回整数向量
FORCE_INLINE IVec<SSE, double> operator>(const FVec1<SSE, double>& a, const FVec1<SSE, double>& b) { return _mm_castpd_si128(_mm_cmpgt_sd(a, b)); }

#ifdef USE_FMA
// 使用 FMA 指令执行乘减操作
FORCE_INLINE FVec1<SSE, double> mulSub(const FVec1<SSE, double>& a, const FVec1<SSE, double>& b, const FVec1<SSE, double>& c) { return _mm_fmsub_sd(a, b, c); }
#endif

// 实现双精度浮点数向量的减法操作
FORCE_INLINE FVec<SSE, double> operator-(const FVec<SSE, double>& a, const FVec<SSE, double>& b) { return _mm_sub_pd(a, b); }

// 实现双精度浮点数向量的乘法操作
FORCE_INLINE FVec<SSE, double> operator*(const FVec<SSE, double>& a, const FVec<SSE, double>& b) { return _mm_mul_pd(a, b); }

// 实现双精度浮点数向量的除法操作
FORCE_INLINE FVec<SSE, double> operator/(const FVec<SSE, double>& a, const FVec<SSE, double>& b) { return _mm_div_pd(a, b); }

// 将双精度浮点数向下取整转换为整数向量
FORCE_INLINE IVec<SSE, float> ftoi(const FVec<SSE, double>& a) { return _mm_cvttpd_epi32(a); }

// 比较两个双精度浮点数向量的小于等于关系，返回整数向量
FORCE_INLINE IVec<SSE, double> operator<=(const FVec<SSE, double>& a, const FVec<SSE, double>& b) { return _mm_castpd_si128(_mm_cmple_pd(a, b)); }

// 比较两个双精度浮点数向量的小于关系，返回整数向量
FORCE_INLINE IVec<SSE, double> operator<(const FVec<SSE, double>& a, const FVec<SSE, double>& b) { return _mm_castpd_si128(_mm_cmplt_pd(a, b)); }

// 比较两个双精度浮点数向量的大于等于关系，返回整数向量
FORCE_INLINE IVec<SSE, double> operator>=(const FVec<SSE, double>& a, const FVec<SSE, double>& b) { return _mm_castpd_si128(_mm_cmpge_pd(a, b)); }

#ifdef USE_FMA
// 使用 FMA 指令执行乘减操作
FORCE_INLINE FVec<SSE, double> mulSub(const FVec<SSE, double>& a, const FVec<SSE, double>& b, const FVec<SSE, double>& c) { return _mm_fmsub_pd(a, b, c); }
#endif

#ifdef USE_AVX

// AVX 指令集下的整数向量基类
template <>
struct IVecBase<AVX> : VecStorage<InstrIntTraits<AVX>>
{
protected:
    FORCE_INLINE IVecBase() {}
    FORCE_INLINE IVecBase(const vec_t& v) : VecStorage<InstrIntTraits<AVX>>(v) {}
public:
    // 返回全零的 AVX 整数向量
    FORCE_INLINE static vec_t zero() { return _mm256_setzero_si256(); }
    // 返回 vec 中的第一个 32 位整数
    FORCE_INLINE int32 get0() const { return _mm_cvtsi128_si32(_mm256_castsi256_si128(vec)); }

    // 根据 mask 的条件，将 val 中的值赋给 vec
    FORCE_INLINE void assignIf( const vec_t& val, const vec_t& mask ) { vec = _mm256_blendv_epi8(vec, val, mask); }
    
    // 根据 mask 的条件，将 val 和 vec 中的值进行按位或操作
    FORCE_INLINE void orIf(const vec_t& val, const vec_t& mask)
    {
        vec = _mm256_blendv_epi8(vec, val, mask);
        //vec = _mm256_or_si256(vec, _mm256_and_si256(val,mask));
    }

    // 返回 vec 的低 128 位
    FORCE_INLINE __m128i lo128() const { return _mm256_castsi256_si128(vec); }
    
    // 返回 vec 的高 128 位
    FORCE_INLINE __m128i hi128() const { return _mm256_extractf128_si256(vec, 1); }
// AVX 模板特化，用于处理 float 类型的向量操作
template <>
struct IVec<AVX, float> : IVecBase<AVX>
{
    // 默认构造函数
    FORCE_INLINE IVec() {}
    // 构造函数，将整数 i 转换为 AVX 向量
    FORCE_INLINE IVec( int32 i ) : IVecBase<AVX>( _mm256_set1_epi32( i ) )  {}
    // 构造函数，使用给定的 AVX 向量 v
    FORCE_INLINE IVec( const vec_t& v) : IVecBase<AVX>( v )              {}
    // 构造函数，使用给定的 8 个 uint32 值初始化 AVX 向量
    FORCE_INLINE IVec(uint32 u7, uint32 u6, uint32 u5, uint32 u4, uint32 u3, uint32 u2, uint32 u1, uint32 u0) : IVecBase<AVX>(_mm256_set_epi32(u7, u6, u5, u4, u3, u2, u1, u0))          {}

    // 设置 AVX 向量的所有元素为 i
    void setN( int32 i ) { vec = _mm256_set1_epi32( i ); }

    // 获取 AVX 向量的第 1 个元素
    FORCE_INLINE int32 get1() const { return _mm256_extract_epi32(vec, 1); }
    // 获取 AVX 向量的第 2 个元素
    FORCE_INLINE int32 get2() const { return _mm256_extract_epi32(vec, 2); }
    // 获取 AVX 向量的第 3 个元素
    FORCE_INLINE int32 get3() const { return _mm256_extract_epi32(vec, 3); }
    // 获取 AVX 向量的第 4 个元素
    FORCE_INLINE int32 get4() const { return _mm256_extract_epi32(vec, 4); }
    // 获取 AVX 向量的第 5 个元素
    FORCE_INLINE int32 get5() const { return _mm256_extract_epi32(vec, 5); }
    // 获取 AVX 向量的第 6 个元素
    FORCE_INLINE int32 get6() const { return _mm256_extract_epi32(vec, 6); }
    // 获取 AVX 向量的第 7 个元素
    FORCE_INLINE int32 get7() const { return _mm256_extract_epi32(vec, 7); }

    // 根据索引数组 bi 和索引向量 idx 设置 AVX 向量
    FORCE_INLINE void setidx( const uint32 *bi, const IVec<AVX,float>& idx )
    {
        vec = _mm256_i32gather_epi32(reinterpret_cast<const int32 *>(bi), idx, sizeof(uint32));
    }

    // 将 AVX 向量存储到指针 pi 指向的内存中
    FORCE_INLINE void store( uint32 *pi ) const { _mm256_storeu_si256( reinterpret_cast<vec_t*>(pi), vec ); }

    // 计算 AVX 向量中置位位的数量
    FORCE_INLINE int countbit()
    {
        return popcnt32(_mm256_movemask_ps(_mm256_castsi256_ps(vec)));
    }
};

// AVX 模板特化，用于处理 double 类型的向量操作
template <>
struct IVec<AVX, double> : IVecBase<AVX>
{
    // 默认构造函数
    FORCE_INLINE IVec() {}
    // 构造函数，将整数 i 转换为 AVX 向量
    FORCE_INLINE IVec( int32 i ) : IVecBase<AVX>( _mm256_set1_epi64x( i ) )    {}
    // 构造函数，使用给定的 AVX 向量 v
    FORCE_INLINE IVec( const vec_t& v) : IVecBase<AVX>( v )                 {}
    // 构造函数，使用给定的 4 个 uint64 值初始化 AVX 向量
    FORCE_INLINE IVec(uint64 u3, uint64 u2, uint64 u1, uint64 u0) : IVecBase<AVX>(_mm256_set_epi64x(u3, u2, u1, u0))          {}

    // 设置 AVX 向量的所有元素为 i
    void setN( int32 i ) { vec = _mm256_set1_epi64x( i ); }

    // 提取 AVX 向量中的第 0、2、4、6 个 32 位整数并存储在 __m128i 中
    // 定义一个函数，用于提取向量中的低32位数据
    FORCE_INLINE IVec<SSE,float> extractLo32s() const
    {
      // 定义一个联合体，包含一个包含4个元素的无符号整数数组和一个__m128i类型的变量
      union {
        uint32 u32[4];
        __m128i u;
      } mask = {0,2,4,6};
      // 使用掩码对向量进行重新排列，得到一个新的__m256i类型的向量
      __m256i blend = _mm256_permutevar8x32_epi32(vec, _mm256_castsi128_si256(mask.u));
      // 将__m256i类型的向量转换为__m128i类型的向量并返回
      return _mm256_castsi256_si128(blend);
    }

    // 定义一个函数，用于获取向量中索引为2的元素
    FORCE_INLINE int32 get1() const { return _mm256_extract_epi32(vec, 2); }

    // 定义一个函数，将向量中的数据存储到指定的uint32类型数组中
    FORCE_INLINE void store( uint32 *pi ) const
    {
        // 调用extractLo32s函数将低32位数据存储到数组中
        extractLo32s().store(pi);
    }

    // 定义一个函数，用于计算向量中非零位的数量
    FORCE_INLINE int countbit()
    {
        // 使用popcnt32函数计算向量中非零位的数量
        return popcnt32(_mm256_movemask_pd(_mm256_castsi256_pd(vec)));
    }
// 右移操作符重载，将 AVX 寄存器中的整数向右移动 n 位
template <typename T>
FORCE_INLINE IVec<AVX,T> operator>> (const IVec<AVX,T>& a, unsigned n)            { return _mm256_srli_epi32(a, n); }

// 左移操作符重载，将 AVX 寄存器中的整数向左移动 n 位
template <typename T>
FORCE_INLINE IVec<AVX,T> operator<< (const IVec<AVX,T>& a, unsigned n)            { return _mm256_slli_epi32(a, n); }

// 位与操作符重载，对 AVX 寄存器中的整数进行位与操作
template <typename T>
FORCE_INLINE IVec<AVX,T> operator&  (const IVec<AVX,T>& a, const IVec<AVX,T>& b ) { return _mm256_and_si256( a, b ); }

// 位或操作符重载，对 AVX 寄存器中的整数进行位或操作
template <typename T>
FORCE_INLINE IVec<AVX,T> operator|  (const IVec<AVX,T>& a, const IVec<AVX,T>& b ) { return _mm256_or_si256( a, b ); }

// 位异或操作符重载，对 AVX 寄存器中的整数进行位异或操作
template <typename T>
FORCE_INLINE IVec<AVX,T> operator^  (const IVec<AVX,T>& a, const IVec<AVX,T>& b ) { return _mm256_xor_si256( a, b ); }

// 求最小值函数，返回 AVX 寄存器中整数的最小值
template <typename T>
FORCE_INLINE IVec<AVX,T> min        (const IVec<AVX,T>& a, const IVec<AVX,T>& b ) { return _mm256_min_epi32( a, b ); }

// 浮点数加法操作符重载，对 AVX 寄存器中的浮点数进行加法操作
FORCE_INLINE IVec<AVX,float> operator+  (const IVec<AVX,float>& a, const IVec<AVX,float>& b ) { return _mm256_add_epi32( a, b ); }

// 浮点数减法操作符重载，对 AVX 寄存器中的浮点数进行减法操作
FORCE_INLINE IVec<AVX,float> operator-  (const IVec<AVX,float>& a, const IVec<AVX,float>& b ) { return _mm256_sub_epi32( a, b ); }

// 双精度浮点数加法操作符重载，对 AVX 寄存器中的双精度浮点数进行加法操作
FORCE_INLINE IVec<AVX,double> operator+  (const IVec<AVX,double>& a, const IVec<AVX,double>& b ) { return _mm256_add_epi64( a, b ); }

// 双精度浮点数减法操作符重载，对 AVX 寄存器中的双精度浮点数进行减法操作
FORCE_INLINE IVec<AVX,double> operator-  (const IVec<AVX,double>& a, const IVec<AVX,double>& b ) { return _mm256_sub_epi64( a, b ); }

// AVX 下的单精度浮点数向量类
typedef VecStorage<InstrFloatTraits<AVX,float>> FVec256Float;

// AVX 下的单精度浮点数向量类模板特化
template <>
struct FVec<AVX, float> : FVec256Float
{
    // 默认构造函数
    FORCE_INLINE FVec() {}

    // 构造函数，将单精度浮点数转为 AVX 寄存器中的向量
    FORCE_INLINE FVec( float f ) : FVec256Float( _mm256_set1_ps( f ) ) {}

    // 构造函数，从内存中加载单精度浮点数向量到 AVX 寄存器中
    FORCE_INLINE FVec( const float *v ) : FVec256Float( _mm256_loadu_ps( v ) ) {}

    // 构造函数，从另一个向量中复制数据到 AVX 寄存器中
    FORCE_INLINE FVec( const vec_t& v) : FVec256Float(v) {}

    // 构造函数，将8个单精度浮点数依次放入 AVX 寄存器中
    FORCE_INLINE FVec(float f7, float f6, float f5, float f4, float f3, float f2, float f1, float f0) : FVec256Float(_mm256_set_ps(f7, f6, f5, f4, f3, f2, f1, f0))          {}

    // 设置 AVX 寄存器中的第一个单精度浮点数
    //void set0( float f  ) { vec = _mm256_load_ss( &f ); }
};
    // 设置 AVX 寄存器中所有元素为给定的浮点数值
    void setN( float f  ) { vec = _mm256_set1_ps( f ); }

    // 设置 AVX 寄存器中的元素为给定索引数组中对应位置的值
    FORCE_INLINE void setidx( const float *xi, const IVec<AVX,float>& idx )
    {
#if 1 // use gather primitives
        // 如果条件为真，则使用 gather primitives
        vec = _mm256_i32gather_ps (xi, idx, 4);
#elif 0
        // 否则，如果条件为假，则执行以下代码块
        // 从 idx 中获取每个索引值
        uint32 i0 = idx.get0();
        uint32 i1 = idx.get1();
        uint32 i2 = idx.get2();
        uint32 i3 = idx.get3();
        uint32 i4 = idx.get4();
        uint32 i5 = idx.get5();
        uint32 i6 = idx.get6();
        uint32 i7 = idx.get7();
        // 使用获取的索引值构建 __m256 对象 vec
        vec = _mm256_set_ps( xi[i7], xi[i6], xi[i5], xi[i4], xi[i3], xi[i2], xi[i1], xi[i0] );
#else
        // 否则，执行以下代码块
        union {
            __m256i vec;
            uint32 ui32[8];
        } i;
        // 将 idx 转换为 __m256i 类型并存储在 i 中
        i.vec = static_cast<const __m256i&>(idx);
        // 使用 i 中的索引值获取 xi 中的数据，构建 __m256 对象 vec
        vec = _mm256_set_ps(xi[i.ui32[7]], xi[i.ui32[6]], xi[i.ui32[5]], xi[i.ui32[4]], xi[i.ui32[3]], xi[i.ui32[2]], xi[i.ui32[1]], xi[i.ui32[0]]);
#endif
    }

    // 返回 vec 的低 128 位
    FORCE_INLINE FVec<SSE, float> lo128() const { return _mm256_castps256_ps128(vec); }
    // 返回 vec 的高 128 位
    FORCE_INLINE FVec<SSE, float> hi128() const { return _mm256_extractf128_ps(vec, 1); }

    // 下面的函数被注释掉，未被使用

FORCE_INLINE FVec<AVX,float> operator-   (const FVec<AVX,float>& a,  const FVec<AVX,float>& b)  { return _mm256_sub_ps( a, b ); }
FORCE_INLINE FVec<AVX,float> operator*   (const FVec<AVX,float>& a,  const FVec<AVX,float>& b)  { return _mm256_mul_ps( a, b ); }
FORCE_INLINE FVec<AVX,float> operator/   (const FVec<AVX,float>& a,  const FVec<AVX,float>& b)  { return _mm256_div_ps( a, b ); }
FORCE_INLINE IVec<AVX,float> ftoi        (const FVec<AVX,float>& a)                             { return _mm256_cvttps_epi32(a); }
FORCE_INLINE IVec<AVX,float> operator<=  (const FVec<AVX,float>& a,  const FVec<AVX,float>& b)  { return _mm256_castps_si256( _mm256_cmp_ps( a, b, _CMP_LE_OS) ); }
// 定义 AVX 指令集下的浮点向量大于等于运算符重载函数
FORCE_INLINE IVec<AVX,float> operator>=  (const FVec<AVX,float>& a,  const FVec<AVX,float>& b)  { return _mm256_castps_si256( _mm256_cmp_ps( a, b, _CMP_GE_OS ) ); }
// 定义 AVX 指令集下的浮点向量小于运算符重载函数
FORCE_INLINE IVec<AVX,float> operator<   (const FVec<AVX,float>& a,  const FVec<AVX,float>& b)  { return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LT_OS )); }
#ifdef USE_FMA
// 如果定义了 USE_FMA，定义 AVX 指令集下的浮点向量乘减法函数
FORCE_INLINE FVec<AVX, float> mulSub(const FVec<AVX, float>& a, const FVec<AVX, float>& b, const FVec<AVX, float>& c) { return _mm256_fmsub_ps(a, b, c); }
#endif

// 定义 AVX 指令集下的双精度浮点向量存储结构
typedef VecStorage<InstrFloatTraits<AVX,double>> FVec256Double;

// AVX 指令集下的双精度浮点向量结构体
template <>
struct FVec<AVX, double> : FVec256Double
{
    FORCE_INLINE FVec() {}
    // 构造函数，将双精度浮点数转换为 AVX 指令集下的双精度浮点向量
    FORCE_INLINE FVec( double d )        : FVec256Double( _mm256_set1_pd( d ) )   {}
    // 构造函数，从双精度浮点数组加载数据到 AVX 指令集下的双精度浮点向量
    FORCE_INLINE FVec( const double *v ) : FVec256Double( _mm256_loadu_pd( v ) )  {}
    // 构造函数，从 vec_t 结构体加载数据到 AVX 指令集下的双精度浮点向量
    FORCE_INLINE FVec( const vec_t& v)   : FVec256Double( v )                  {}
    // 构造函数，根据四个双精度浮点数构造 AVX 指令集下的双精度浮点向量
    FORCE_INLINE FVec(double d3, double d2, double d1, double d0) : FVec256Double(_mm256_set_pd(d3, d2, d1, d0))          {}

    // 设置向量中所有元素为指定双精度浮点数
    void setN( double f  ) { vec = _mm256_set1_pd( f ); }

    // 根据索引向量从双精度浮点数组中加载数据到 AVX 指令集下的双精度浮点向量
    FORCE_INLINE void setidx( const double *xi, const IVec<SSE,float>& idx )
    {
        vec = _mm256_i32gather_pd(xi, idx, 8);
    }

    // 根据索引向量从双精度浮点数组中加载数据到 AVX 指令集下的双精度浮点向量
    FORCE_INLINE void setidx( const double *xi, const IVec<AVX,double>& idx )
    {
        vec = _mm256_i64gather_pd(xi, idx, 8);
    }
};

// 定义 AVX 指令集下的双精度浮点向量减法运算符重载函数
FORCE_INLINE FVec<AVX,double> operator-   (const FVec<AVX,double>& a, const FVec<AVX,double>& b)    { return _mm256_sub_pd( a, b ); }
// 定义 AVX 指令集下的双精度浮点向量乘法运算符重载函数
FORCE_INLINE FVec<AVX,double> operator*   (const FVec<AVX,double>& a, const FVec<AVX,double>& b)    { return _mm256_mul_pd( a, b ); }
# 定义 AVX 下的双精度除法运算符重载函数，返回两个 AVX 双精度向量相除的结果
FORCE_INLINE FVec<AVX,double> operator/   (const FVec<AVX,double>& a, const FVec<AVX,double>& b)    { return _mm256_div_pd( a, b ); }

# 定义将 AVX 双精度向量转换为 SSE 单精度整数向量的函数
FORCE_INLINE IVec<SSE,float>  ftoi        (const FVec<AVX,double>& a)                               { return _mm256_cvttpd_epi32(a); }

# 定义 AVX 下的双精度小于等于运算符重载函数，返回两个 AVX 双精度向量每个元素比较小于等于的结果
FORCE_INLINE IVec<AVX,double> operator<=   (const FVec<AVX,double>& a, const FVec<AVX,double>& b)    { return _mm256_castpd_si256(_mm256_cmp_pd( a, b, _CMP_LE_OS ) ); }

# 定义 AVX 下的双精度小于运算符重载函数，返回两个 AVX 双精度向量每个元素比较小于的结果
FORCE_INLINE IVec<AVX,double> operator<    (const FVec<AVX,double>& a, const FVec<AVX,double>& b)    { return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OS)); }

# 定义 AVX 下的双精度大于等于运算符重载函数，返回两个 AVX 双精度向量每个元素比较大于等于的结果
FORCE_INLINE IVec<AVX,double> operator>=   (const FVec<AVX,double>& a, const FVec<AVX,double>& b)    { return _mm256_castpd_si256(_mm256_cmp_pd( a, b, _CMP_GE_OS ) ); }

# 如果定义了 USE_FMA 宏，则定义 AVX 下的双精度 FMA 减法函数，返回三个 AVX 双精度向量的 FMA 减法结果
#ifdef USE_FMA
FORCE_INLINE FVec<AVX, double> mulSub(const FVec<AVX, double>& a, const FVec<AVX, double>& b, const FVec<AVX, double>& c) { return _mm256_fmsub_pd(a, b, c); }
#endif

#endif

} // namepsace Details
} // namespace BinSearch
```