# `.\pytorch\aten\src\ATen\cuda\llvm_complex.cpp`

```
// 这段代码是从 LLVM 项目中复制并进行了修改
// 源文件位置: https://github.com/llvm/llvm-project/blob/main/libcxx/include/complex

// 包含标准字符串库
#include <string>
// 包含 ATen CUDA 的 LLVM JIT 字符串头文件
#include <ATen/cuda/llvm_jit_strings.h>

// 命名空间 at::cuda
namespace at::cuda {

// 定义一个字符串常量 complex_body，包含 C++ 原始字符串字面量 R"ESCAPE( ... )ESCAPE"
const std::string complex_body = R"ESCAPE(

namespace std {

// 前置声明 complex 模板类
template<class _Tp> class complex;

// complex 类的乘法运算符重载声明
template<class _Tp> complex<_Tp> operator*(const complex<_Tp>& __z, const complex<_Tp>& __w);
// complex 类的除法运算符重载声明
template<class _Tp> complex<_Tp> operator/(const complex<_Tp>& __x, const complex<_Tp>& __y);

// complex 模板类定义
template<class _Tp>
class complex
{
public:
    typedef _Tp value_type;
private:
    value_type __re_;
    value_type __im_;
public:
    // 构造函数，初始化实部和虚部
    constexpr
    complex(const value_type& __re = value_type(), const value_type& __im = value_type())
        : __re_(__re), __im_(__im) {}
    // 复制构造函数，从另一个复数对象复制实部和虚部
    template<class _Xp> constexpr
    complex(const complex<_Xp>& __c)
        : __re_(__c.real()), __im_(__c.imag()) {}

    // 返回实部的值
    constexpr value_type real() const {return __re_;}
    // 返回虚部的值
    constexpr value_type imag() const {return __im_;}

    // 设置实部的值
    void real(value_type __re) {__re_ = __re;}
    // 设置虚部的值
    void imag(value_type __im) {__im_ = __im;}

    // 转换为 bool 类型，判断是否非零
    constexpr operator bool() const {
        return real() || imag();
    }

    // 实部赋值运算符重载
    complex& operator= (const value_type& __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    // 实部加法赋值运算符重载
    complex& operator+=(const value_type& __re) {__re_ += __re; return *this;}
    // 实部减法赋值运算符重载
    complex& operator-=(const value_type& __re) {__re_ -= __re; return *this;}
    // 实部乘法赋值运算符重载
    complex& operator*=(const value_type& __re) {__re_ *= __re; __im_ *= __re; return *this;}
    // 实部除法赋值运算符重载
    complex& operator/=(const value_type& __re) {__re_ /= __re; __im_ /= __re; return *this;}

    // 复数赋值运算符重载，从另一个复数对象复制实部和虚部
    template<class _Xp> complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    // 复数加法赋值运算符重载
    template<class _Xp> complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    // 复数减法赋值运算符重载
    template<class _Xp> complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    // 复数乘法赋值运算符重载
    template<class _Xp> complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * complex(__c.real(), __c.imag());
            return *this;
        }
    // 复数除法赋值运算符重载
    template<class _Xp> complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / complex(__c.real(), __c.imag());
            return *this;
        }
};

// 显式实例化 complex 类模板，指定为 double 类型
template<> class complex<double>;

// 结束 std 命名空间
} // namespace std

)ESCAPE";
// 定义复数类模板 complex<float>
template<class float>
{
    // 实部和虚部的私有成员变量
    float __re_;
    float __im_;
public:
    // 定义 value_type 为 float
    typedef float value_type;

    // 构造函数，初始化实部和虚部，默认为 0.0f
    constexpr complex(float __re = 0.0f, float __im = 0.0f)
        : __re_(__re), __im_(__im) {}

    // 复制构造函数，从 complex<double> 隐式转换为 complex<float>
    explicit constexpr complex(const complex<double>& __c);

    // 返回实部和虚部的方法
    constexpr float real() const {return __re_;}
    constexpr float imag() const {return __im_;}

    // 设置实部和虚部的方法
    void real(value_type __re) {__re_ = __re;}
    void imag(value_type __im) {__im_ = __im;}

    // 隐式转换为 bool 类型，判断是否非零
    constexpr operator bool() const {
        return real() || imag();
    }

    // 赋值运算符重载，将实部设置为给定 float，虚部设为 0
    complex& operator= (float __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    // 实部加法赋值运算符重载
    complex& operator+=(float __re) {__re_ += __re; return *this;}
    // 实部减法赋值运算符重载
    complex& operator-=(float __re) {__re_ -= __re; return *this;}
    // 实部乘法赋值运算符重载，实部和虚部同时乘以给定 float
    complex& operator*=(float __re) {__re_ *= __re; __im_ *= __re; return *this;}
    // 实部除法赋值运算符重载，实部和虚部同时除以给定 float
    complex& operator/=(float __re) {__re_ /= __re; __im_ /= __re; return *this;}

    // 模板赋值运算符重载，从复数模板类型 _Xp 转换为当前类型
    template<class _Xp> complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    // 模板加法赋值运算符重载，从复数模板类型 _Xp 加到当前类型
    template<class _Xp> complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    // 模板减法赋值运算符重载，从当前类型减去复数模板类型 _Xp
    template<class _Xp> complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    // 模板乘法赋值运算符重载，当前类型乘以复数模板类型 _Xp
    template<class _Xp> complex& operator*=(const complex<_Xp>& __c)
        {
            // 使用乘法运算符重载计算乘积后赋值给当前对象
            *this = *this * complex(__c.real(), __c.imag());
            return *this;
        }
    // 模板除法赋值运算符重载，当前类型除以复数模板类型 _Xp
    template<class _Xp> complex& operator/=(const complex<_Xp>& __c)
        {
            // 使用除法运算符重载计算商后赋值给当前对象
            *this = *this / complex(__c.real(), __c.imag());
            return *this;
        }
};

// 定义复数类模板特化 complex<double>
template<>
class complex<double>
{
    // 实部和虚部的私有成员变量
    double __re_;
    double __im_;
public:
    // 定义 value_type 为 double
    typedef double value_type;

    // 构造函数，初始化实部和虚部，默认为 0.0
    constexpr complex(double __re = 0.0, double __im = 0.0)
        : __re_(__re), __im_(__im) {}

    // 复制构造函数，从 complex<float> 隐式转换为 complex<double>
    constexpr complex(const complex<float>& __c);

    // 返回实部和虚部的方法
    constexpr double real() const {return __re_;}
    constexpr double imag() const {return __im_;}

    // 设置实部和虚部的方法
    void real(value_type __re) {__re_ = __re;}
    void imag(value_type __im) {__im_ = __im;}

    // 隐式转换为 bool 类型，判断是否非零
    constexpr operator bool() const {
        return real() || imag();
    }

    // 赋值运算符重载，将实部设置为给定 double，虚部设为 0
    complex& operator= (double __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    // 实部加法赋值运算符重载
    complex& operator+=(double __re) {__re_ += __re; return *this;}
    // 实部减法赋值运算符重载
    complex& operator-=(double __re) {__re_ -= __re; return *this;}
    // 实部乘法赋值运算符重载，实部和虚部同时乘以给定 double
    complex& operator*=(double __re) {__re_ *= __re; __im_ *= __re; return *this;}
    // 实部除法赋值运算符重载，实部和虚部同时除以给定 double
    complex& operator/=(double __re) {__re_ /= __re; __im_ /= __re; return *this;}

    // 模板赋值运算符重载，从复数模板类型 _Xp 转换为当前类型
    template<class _Xp> complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator+=(const complex<_Xp>& __c)
    {
        // 将当前复数对象的实部增加上另一个复数对象的实部
        __re_ += __c.real();
        // 将当前复数对象的虚部增加上另一个复数对象的虚部
        __im_ += __c.imag();
        // 返回更新后的当前复数对象
        return *this;
    }
    template<class _Xp> complex& operator-=(const complex<_Xp>& __c)
    {
        // 将当前复数对象的实部减去另一个复数对象的实部
        __re_ -= __c.real();
        // 将当前复数对象的虚部减去另一个复数对象的虚部
        __im_ -= __c.imag();
        // 返回更新后的当前复数对象
        return *this;
    }
    template<class _Xp> complex& operator*=(const complex<_Xp>& __c)
    {
        // 将当前复数对象乘以另一个复数对象
        *this = *this * complex(__c.real(), __c.imag());
        // 返回更新后的当前复数对象
        return *this;
    }
    template<class _Xp> complex& operator/=(const complex<_Xp>& __c)
    {
        // 将当前复数对象除以另一个复数对象
        *this = *this / complex(__c.real(), __c.imag());
        // 返回更新后的当前复数对象
        return *this;
    }
// complex 类的构造函数，从复数的 double 类型复制构造到 float 类型
constexpr
complex<float>::complex(const complex<double>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}

// complex 类的构造函数，从复数的 float 类型复制构造到 double 类型
constexpr
complex<double>::complex(const complex<float>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}


// 26.3.6 operators:

// 复数加法运算符重载，接受两个相同类型的复数对象作为参数
template<class _Tp>
inline
complex<_Tp>
operator+(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    // 创建一个临时复数对象 __t，将 __x 复制到其中
    complex<_Tp> __t(__x);
    // 将 __t 加上 __y
    __t += __y;
    // 返回结果 __t
    return __t;
}

// 复数加法运算符重载，接受一个复数对象和一个 _Tp 类型的数值作为参数
template<class _Tp>
inline
complex<_Tp>
operator+(const complex<_Tp>& __x, const _Tp& __y)
{
    // 创建一个临时复数对象 __t，将 __x 复制到其中
    complex<_Tp> __t(__x);
    // 将 __t 加上 __y
    __t += __y;
    // 返回结果 __t
    return __t;
}

// 复数加法运算符重载，接受一个 _Tp 类型的数值和一个复数对象作为参数
template<class _Tp>
inline
complex<_Tp>
operator+(const _Tp& __x, const complex<_Tp>& __y)
{
    // 创建一个临时复数对象 __t，将 __y 复制到其中
    complex<_Tp> __t(__y);
    // 将 __t 加上 __x
    __t += __x;
    // 返回结果 __t
    return __t;
}

// 复数减法运算符重载，接受两个相同类型的复数对象作为参数
template<class _Tp>
inline
complex<_Tp>
operator-(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    // 创建一个临时复数对象 __t，将 __x 复制到其中
    complex<_Tp> __t(__x);
    // 将 __t 减去 __y
    __t -= __y;
    // 返回结果 __t
    return __t;
}

// 复数减法运算符重载，接受一个复数对象和一个 _Tp 类型的数值作为参数
template<class _Tp>
inline
complex<_Tp>
operator-(const complex<_Tp>& __x, const _Tp& __y)
{
    // 创建一个临时复数对象 __t，将 __x 复制到其中
    complex<_Tp> __t(__x);
    // 将 __t 减去 __y
    __t -= __y;
    // 返回结果 __t
    return __t;
}

// 复数减法运算符重载，接受一个 _Tp 类型的数值和一个复数对象作为参数
template<class _Tp>
inline
complex<_Tp>
operator-(const _Tp& __x, const complex<_Tp>& __y)
{
    // 创建一个临时复数对象 __t，将 __y 取反复制到其中
    complex<_Tp> __t(-__y);
    // 将 __t 加上 __x
    __t += __x;
    // 返回结果 __t
    return __t;
}

// 复数乘法运算符重载，接受两个相同类型的复数对象作为参数
template<class _Tp>
complex<_Tp>
operator*(const complex<_Tp>& __z, const complex<_Tp>& __w)
{
    // 获取复数 __z 和 __w 的实部和虚部
    _Tp __a = __z.real();
    _Tp __b = __z.imag();
    _Tp __c = __w.real();
    _Tp __d = __w.imag();
    // 计算乘法结果的实部和虚部
    _Tp __ac = __a * __c;
    _Tp __bd = __b * __d;
    _Tp __ad = __a * __d;
    _Tp __bc = __b * __c;
    _Tp __x = __ac - __bd;
    _Tp __y = __ad + __bc;
    // 处理特殊情况，如结果为 NaN 或者 inf
    if (isnan(__x) && isnan(__y))
    {
        bool __recalc = false;
        if (isinf(__a) || isinf(__b))
        {
            __a = copysign(isinf(__a) ? _Tp(1) : _Tp(0), __a);
            __b = copysign(isinf(__b) ? _Tp(1) : _Tp(0), __b);
            if (isnan(__c))
                __c = copysign(_Tp(0), __c);
            if (isnan(__d))
                __d = copysign(_Tp(0), __d);
            __recalc = true;
        }
        if (isinf(__c) || isinf(__d))
        {
            __c = copysign(isinf(__c) ? _Tp(1) : _Tp(0), __c);
            __d = copysign(isinf(__d) ? _Tp(1) : _Tp(0), __d);
            if (isnan(__a))
                __a = copysign(_Tp(0), __a);
            if (isnan(__b))
                __b = copysign(_Tp(0), __b);
            __recalc = true;
        }
        if (!__recalc && (isinf(__ac) || isinf(__bd) ||
                          isinf(__ad) || isinf(__bc)))
        {
            if (isnan(__a))
                __a = copysign(_Tp(0), __a);
            if (isnan(__b))
                __b = copysign(_Tp(0), __b);
            if (isnan(__c))
                __c = copysign(_Tp(0), __c);
            if (isnan(__d))
                __d = copysign(_Tp(0), __d);
            __recalc = true;
        }
        if (__recalc)
        {
            // 如果需要重新计算，则将结果设为无穷大
            __x = _Tp(INFINITY) * (__a * __c - __b * __d);
            __y = _Tp(INFINITY) * (__a * __d + __b * __c);
        }
    }
    // 返回一个复数对象，使用类型 _Tp，并初始化其实部为 __x，虚部为 __y
    return complex<_Tp>(__x, __y);
template<class _Tp>
inline constexpr
bool
operator!=(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    // 检查两个复数是否不相等，如果实部和虚部都不相等则返回 true，否则返回 false
    return !(__x == __y);
}



template<class _Tp>
inline constexpr
bool
operator!=(const complex<_Tp>& __x, const _Tp& __y)
{
    // 检查复数和一个数是否不相等，如果复数的实部不等于该数或虚部不为零，则返回 true，否则返回 false
    return !(__x == __y);
}



template<class _Tp>
inline constexpr
bool
operator!=(const _Tp& __x, const complex<_Tp>& __y)
{
    // 检查一个数和复数是否不相等，如果该数不等于复数的实部或者复数的虚部不为零，则返回 true，否则返回 false
    return !(__x == __y);
}



template<class _Tp>
inline
complex<_Tp>
operator+(const complex<_Tp>& __x)
{
    // 返回复数本身，即复数的实部和虚部不变
    return __x;
}



template<class _Tp>
inline
complex<_Tp>
operator-(const complex<_Tp>& __x)
{
    // 返回复数的相反数，实部取反，虚部取反
    return complex<_Tp>(-__x.real(), -__x.imag());
}



template<class _Tp>
inline constexpr
bool
operator==(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    // 检查两个复数是否相等，实部和虚部都相等则返回 true，否则返回 false
    return __x.real() == __y.real() && __x.imag() == __y.imag();
}



template<class _Tp>
inline constexpr
bool
operator==(const complex<_Tp>& __x, const _Tp& __y)
{
    // 检查复数和一个数是否相等，虚部必须为零，且实部等于该数才返回 true，否则返回 false
    return __x.real() == __y && __x.imag() == 0;
}



template<class _Tp>
inline constexpr
bool
operator==(const _Tp& __x, const complex<_Tp>& __y)
{
    // 检查一个数和复数是否相等，复数的虚部必须为零，且实部等于该数才返回 true，否则返回 false
    return __x == __y.real() && 0 == __y.imag();
}
    # 返回表达式 !(__x == __y) 的结果，即返回 __x 不等于 __y 的逻辑值
    return !(__x == __y);
}

// 结束了之前的命名空间或结构体的定义

template<class _Tp>
inline constexpr
bool
operator!=(const complex<_Tp>& __x, const _Tp& __y)
{
    // 返回复数对象与类型 _Tp 的不等号比较结果
    return !(__x == __y);
}

template<class _Tp>
inline constexpr
bool
operator!=(const _Tp& __x, const complex<_Tp>& __y)
{
    // 返回类型 _Tp 与复数对象的不等号比较结果
    return !(__x == __y);
}

template<class _Tp>
inline constexpr
bool
operator&&(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    // 返回两个复数对象的逻辑与操作结果
    return bool(__x) && bool(__y);
}

template<class _Tp>
inline constexpr
bool
isnan(const complex<_Tp>& __x)
{
    // 检查复数对象 __x 的实部或虚部是否为 NaN
    return isnan(__x.real()) || isnan(__x.imag());
}

template<class _Tp>
inline constexpr
bool
operator||(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    // 返回两个复数对象的逻辑或操作结果
    return bool(__x) || bool(__y);
}

// 26.3.7 values:

template <class _Tp, bool = is_integral<_Tp>::value,
                     bool = is_floating_point<_Tp>::value
                     >
struct __libcpp_complex_overload_traits {};

// Integral Types
template <class _Tp>
struct __libcpp_complex_overload_traits<_Tp, true, false>
{
    // 对于整数类型 _Tp，_ValueType 设为 double，_ComplexType 设为 complex<double>
    typedef double _ValueType;
    typedef complex<double> _ComplexType;
};

// Floating point types
template <class _Tp>
struct __libcpp_complex_overload_traits<_Tp, false, true>
{
    // 对于浮点数类型 _Tp，_ValueType 设为 _Tp，_ComplexType 设为 complex<_Tp>
    typedef _Tp _ValueType;
    typedef complex<_Tp> _ComplexType;
};

// real

template<class _Tp>
inline constexpr
_Tp
real(const complex<_Tp>& __c)
{
    // 返回复数对象 __c 的实部
    return __c.real();
}

template <class _Tp>
inline constexpr
typename __libcpp_complex_overload_traits<_Tp>::_ValueType
real(_Tp __re)
{
    // 返回类型 _Tp 的实部，对于浮点数类型 _Tp，直接返回 __re
    return __re;
}

// imag

template<class _Tp>
inline constexpr
_Tp
imag(const complex<_Tp>& __c)
{
    // 返回复数对象 __c 的虚部
    return __c.imag();
}

template <class _Tp>
inline constexpr
typename __libcpp_complex_overload_traits<_Tp>::_ValueType
imag(_Tp)
{
    // 返回类型 _Tp 的虚部，对于浮点数类型 _Tp，虚部为 0
    return 0;
}

// abs

template<class _Tp>
inline
_Tp
abs(const complex<_Tp>& __c)
{
    // 返回复数对象 __c 的绝对值
    return hypot(__c.real(), __c.imag());
}

// arg

template<class _Tp>
inline
_Tp
arg(const complex<_Tp>& __c)
{
    // 返回复数对象 __c 的幅角
    return atan2(__c.imag(), __c.real());
}

template<class _Tp>
inline
typename enable_if
<
    is_integral<_Tp>::value || is_same<_Tp, double>::value,
    double
>::type
arg(_Tp __re)
{
    // 对于整数或者双精度浮点数类型 _Tp，返回其幅角
    return atan2(0., __re);
}

template <class _Tp>
inline
typename enable_if<
    is_same<_Tp, float>::value,
    float
>::type
arg(_Tp __re)
{
    // 对于单精度浮点数类型 _Tp，返回其幅角
    return atan2f(0.F, __re);
}

}

)ESCAPE";

const std::string complex_half_body = R"ESCAPE(
namespace std {
template <>
struct alignas(2) complex<at::Half> {
  at::Half real_;
  at::Half imag_;

  // Constructors
  complex() = default;

  // implicit casting to and from `complex<float>`.
  // NOTE: computation of `complex<Half>` will occur in `complex<float>`
  __host__ __device__ inline complex(const std::complex<float>& value)
      : real_(value.real()), imag_(value.imag()) {}

  inline __host__ __device__ operator std::complex<float>() const {
    return {real_, imag_};
  }

  // 返回实部
  at::Half real() const {return real_;}
  // 返回虚部
  at::Half imag() const {return imag_;}

};
}
)ESCAPE";


const std::string &get_complex_body_string() {
  return complex_body;
}
const std::string &get_complex_half_body_string() {
  // 返回一个全局复杂字符串的引用
  return complex_half_body;
}

const std::string complex_math = R"ESCAPE(
// 定义复杂的数学函数命名空间和函数模板
namespace std {

// norm

template<class _Tp>
inline
_Tp
norm(const complex<_Tp>& __c)
{
    // 计算复数的模
    if (isinf(__c.real()))
        return abs(__c.real());
    if (isinf(__c.imag()))
        return abs(__c.imag());
    return __c.real() * __c.real() + __c.imag() * __c.imag();
}

template <class _Tp>
inline
typename __libcpp_complex_overload_traits<_Tp>::_ValueType
norm(_Tp __re)
{
    // 计算标量的模
    typedef typename __libcpp_complex_overload_traits<_Tp>::_ValueType _ValueType;
    return static_cast<_ValueType>(__re) * __re;
}

// conj

template<class _Tp>
inline
complex<_Tp>
conj(const complex<_Tp>& __c)
{
    // 计算复数的共轭
    return complex<_Tp>(__c.real(), -__c.imag());
}

template <class _Tp>
inline
typename __libcpp_complex_overload_traits<_Tp>::_ComplexType
conj(_Tp __re)
{
    // 返回标量的本身作为复数的共轭
    typedef typename __libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;
    return _ComplexType(__re);
}

// proj

template<class _Tp>
inline
complex<_Tp>
proj(const complex<_Tp>& __c)
{
    // 计算复数的投影
    complex<_Tp> __r = __c;
    if (isinf(__c.real()) || isinf(__c.imag()))
        __r = complex<_Tp>(INFINITY, copysign(_Tp(0), __c.imag()));
    return __r;
}

template <class _Tp>
inline
typename enable_if
<
    is_floating_point<_Tp>::value,
    typename __libcpp_complex_overload_traits<_Tp>::_ComplexType
>::type
proj(_Tp __re)
{
    // 计算浮点数的投影
    if (isinf(__re))
        __re = abs(__re);
    return complex<_Tp>(__re);
}

template <class _Tp>
inline
typename enable_if
<
    is_integral<_Tp>::value,
    typename __libcpp_complex_overload_traits<_Tp>::_ComplexType
>::type
proj(_Tp __re)
{
    // 计算整数的投影
    typedef typename __libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;
    return _ComplexType(__re);
}

// polar

template<class _Tp>
complex<_Tp>
polar(const _Tp& __rho, const _Tp& __theta = _Tp())
{
    // 计算极坐标到复数的转换
    if (isnan(__rho) || signbit(__rho))
        return complex<_Tp>(_Tp(NAN), _Tp(NAN));
    if (isnan(__theta))
    {
        if (isinf(__rho))
            return complex<_Tp>(__rho, __theta);
        return complex<_Tp>(__theta, __theta);
    }
    if (isinf(__theta))
    {
        if (isinf(__rho))
            return complex<_Tp>(__rho, _Tp(NAN));
        return complex<_Tp>(_Tp(NAN), _Tp(NAN));
    }
    _Tp __x = __rho * cos(__theta);
    if (isnan(__x))
        __x = 0;
    _Tp __y = __rho * sin(__theta);
    if (isnan(__y))
        __y = 0;
    return complex<_Tp>(__x, __y);
}

// log

template<class _Tp>
inline
complex<_Tp>
log(const complex<_Tp>& __x)
{
    // 计算复数的自然对数
    return complex<_Tp>(log(abs(__x)), arg(__x));
}

// log10

template<class _Tp>
inline
complex<_Tp>
log10(const complex<_Tp>& __x)
{
    // 计算复数的以10为底的对数
    return log(__x) / log(_Tp(10));
}

// log2

template<class _Tp>
inline
complex<_Tp>
log2(const complex<_Tp>& __x)
{
    // 计算复数的以2为底的对数
    return log(__x) / log(_Tp(2));
}

// sqrt

template<class _Tp>
complex<_Tp>
sqrt(const complex<_Tp>& __x)
{
    # 如果 __x.imag() 是无穷大
    if (isinf(__x.imag()))
        # 返回一个复数，实部为正无穷，虚部为 __x 的虚部
        return complex<_Tp>(_Tp(INFINITY), __x.imag());
    
    # 如果 __x.real() 是无穷大
    if (isinf(__x.real()))
    {
        # 如果 __x.real() 大于 0
        if (__x.real() > _Tp(0))
            # 返回一个复数，实部为 __x 的实部，虚部为如果 __x.imag() 是 NaN，则虚部也是 NaN，否则为 __x.imag() 的符号位
            return complex<_Tp>(__x.real(), isnan(__x.imag()) ? __x.imag() : copysign(_Tp(0), __x.imag()));
        
        # 返回一个复数，实部为如果 __x.imag() 是 NaN，则为 NaN，否则为 0，虚部为 __x 的实部的符号位
        return complex<_Tp>(isnan(__x.imag()) ? __x.imag() : _Tp(0), copysign(__x.real(), __x.imag()));
    }
    
    # 返回极坐标形式表示的复数，模为 sqrt(abs(__x))，幅角为 arg(__x) 除以 2
    return polar(sqrt(abs(__x)), arg(__x) / _Tp(2));
}

// exp

// 计算复数的指数函数 exp(z)，其中 z 是复数
template<class _Tp>
complex<_Tp>
exp(const complex<_Tp>& __x)
{
    // 获取虚部
    _Tp __i = __x.imag();
    // 如果虚部为零
    if (__i == 0) {
        // 返回指数函数的结果，实部为 exp(实部)，虚部为 copysign(0, 虚部)
        return complex<_Tp>(exp(__x.real()), copysign(_Tp(0), __x.imag()));
    }
    // 如果实部为无穷大
    if (isinf(__x.real()))
    {
        // 如果实部小于零
        if (__x.real() < _Tp(0))
        {
            // 如果虚部也非有限数，则将虚部设为 1
            if (!isfinite(__i))
                __i = _Tp(1);
        }
        // 如果虚部为零或者非有限数
        else if (__i == 0 || !isfinite(__i))
        {
            // 如果虚部为无穷大，则将虚部设为 NaN，否则返回原始复数
            if (isinf(__i))
                __i = _Tp(NAN);
            return complex<_Tp>(__x.real(), __i);
        }
    }
    // 计算实部的指数函数
    _Tp __e = exp(__x.real());
    // 返回复数的指数函数结果，实部为 exp(实部) * cos(虚部)，虚部为 exp(实部) * sin(虚部)
    return complex<_Tp>(__e * cos(__i), __e * sin(__i));
}

// pow

// 计算复数的幂函数 pow(x, y)，其中 x 和 y 均为复数
template<class _Tp>
inline
complex<_Tp>
pow(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    // 返回 exp(y * log(x))
    return exp(__y * log(__x));
}

// 通过类型推导，计算复数和复数之间的幂函数
template<class _Tp, class _Up>
inline
complex<typename __promote<_Tp, _Up>::type>
pow(const complex<_Tp>& __x, const complex<_Up>& __y)
{
    typedef complex<typename __promote<_Tp, _Up>::type> result_type;
    return std::pow(result_type(__x), result_type(__y));
}

// 通过类型推导，计算复数和数值类型之间的幂函数
template<class _Tp, class _Up>
inline
typename enable_if
<
    is_arithmetic<_Up>::value,
    complex<typename __promote<_Tp, _Up>::type>
>::type
pow(const complex<_Tp>& __x, const _Up& __y)
{
    typedef complex<typename __promote<_Tp, _Up>::type> result_type;
    return std::pow(result_type(__x), result_type(__y));
}

// 通过类型推导，计算数值类型和复数之间的幂函数
template<class _Tp, class _Up>
inline
typename enable_if
<
    is_arithmetic<_Tp>::value,
    complex<typename __promote<_Tp, _Up>::type>
>::type
pow(const _Tp& __x, const complex<_Up>& __y)
{
    typedef complex<typename __promote<_Tp, _Up>::type> result_type;
    return std::pow(result_type(__x), result_type(__y));
}

// __sqr, computes pow(x, 2)

// 计算复数的平方函数，即 pow(x, 2)
template<class _Tp>
inline
complex<_Tp>
__sqr(const complex<_Tp>& __x)
{
    // 返回复数的平方值
    return complex<_Tp>((__x.real() - __x.imag()) * (__x.real() + __x.imag()),
                        _Tp(2) * __x.real() * __x.imag());
}

// asinh

// 计算复数的反双曲正弦函数 asinh(z)，其中 z 是复数
template<class _Tp>
complex<_Tp>
asinh(const complex<_Tp>& __x)
{
    // 定义常数 π
    const _Tp __pi(atan2(+0., -0.));
    // 如果实部为无穷大
    if (isinf(__x.real()))
    {
        // 如果虚部为 NaN，返回原始复数
        if (isnan(__x.imag()))
            return __x;
        // 如果虚部为无穷大，返回复数，实部为原始实部，虚部为 copysign(π/4, 虚部)
        if (isinf(__x.imag()))
            return complex<_Tp>(__x.real(), copysign(__pi * _Tp(0.25), __x.imag()));
        // 返回复数，实部为原始实部，虚部为 copysign(0, 虚部)
        return complex<_Tp>(__x.real(), copysign(_Tp(0), __x.imag()));
    }
    // 如果实部为 NaN
    if (isnan(__x.real()))
    {
        // 如果虚部为无穷大，返回复数，实部为原始虚部，虚部为原始实部
        if (isinf(__x.imag()))
            return complex<_Tp>(__x.imag(), __x.real());
        // 如果虚部为零，返回原始复数
        if (__x.imag() == 0)
            return __x;
        // 返回复数，实部为原始实部，虚部为原始实部
        return complex<_Tp>(__x.real(), __x.real());
    }
    // 如果虚部为无穷大，返回复数，实部为 copysign(虚部, 实部)，虚部为 copysign(π/2, 虚部)
    if (isinf(__x.imag()))
        return complex<_Tp>(copysign(__x.imag(), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    // 计算 log(z + sqrt(z^2 + 1))
    complex<_Tp> __z = log(__x + sqrt(__sqr(__x) + _Tp(1)));
    // 返回复数，实部为 copysign(实部(z), 实部)，虚部为 copysign(虚部(z), 虚部)
    return complex<_Tp>(copysign(__z.real(), __x.real()), copysign(__z.imag(), __x.imag()));
}

// acosh

// 计算复数的反双曲余弦函数 acosh(z)，其中 z 是复数
template<class _Tp>
complex<_Tp>
acosh(const complex<_Tp>& __x)
{
    // 定义常数 π
    const _Tp __pi(atan2(+0., -0.));
    // 如果实部为无穷大
    if (isinf(__x.real()))
    {
        // 如果虚部为 NaN，则返回一个复数，实部为实部的绝对值，虚部保持不变
        if (isnan(__x.imag()))
            return complex<_Tp>(abs(__x.real()), __x.imag());
        
        // 如果虚部为无穷大
        if (isinf(__x.imag()))
        {
            // 如果实部大于 0，则返回一个复数，实部不变，虚部为符号为虚部的 pi/4
            if (__x.real() > 0)
                return complex<_Tp>(__x.real(), copysign(__pi * _Tp(0.25), __x.imag()));
            // 否则返回一个复数，实部为实部的相反数，虚部为符号为虚部的 3pi/4
            else
                return complex<_Tp>(-__x.real(), copysign(__pi * _Tp(0.75), __x.imag()));
        }
        
        // 如果实部小于 0，则返回一个复数，实部为实部的相反数，虚部为符号为虚部的 pi
        if (__x.real() < 0)
            return complex<_Tp>(-__x.real(), copysign(__pi, __x.imag()));
        
        // 其余情况返回一个复数，实部为实部，虚部为符号为虚部的 0
        return complex<_Tp>(__x.real(), copysign(_Tp(0), __x.imag()));
    }
    
    // 如果实部为 NaN
    if (isnan(__x.real()))
    {
        // 如果虚部为无穷大，则返回一个复数，实部为虚部的绝对值，虚部为实部
        if (isinf(__x.imag()))
            return complex<_Tp>(abs(__x.imag()), __x.real());
        
        // 其他情况返回一个复数，实部为实部，虚部为实部
        return complex<_Tp>(__x.real(), __x.real());
    }
    
    // 如果虚部为无穷大，则返回一个复数，实部为虚部的绝对值，虚部为符号为虚部的 pi/2
    if (isinf(__x.imag()))
        return complex<_Tp>(abs(__x.imag()), copysign(__pi/_Tp(2), __x.imag()));
    
    // 对于其他情况，计算 log(__x + sqrt(__sqr(__x) - 1))，返回一个复数，实部为结果实部的符号为 0，虚部为结果虚部的符号为虚部
    complex<_Tp> __z = log(__x + sqrt(__sqr(__x) - _Tp(1)));
    return complex<_Tp>(copysign(__z.real(), _Tp(0)), copysign(__z.imag(), __x.imag()));
// atanh 函数的实现，计算复数的反双曲正切值
template<class _Tp>
complex<_Tp>
atanh(const complex<_Tp>& __x)
{
    // 定义常量 pi 为 atan2 的结果，用于后续计算
    const _Tp __pi(atan2(+0., -0.));

    // 如果虚部为无穷大，返回根据实部符号构造的复数
    if (isinf(__x.imag())) {
        return complex<_Tp>(copysign(_Tp(0), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    }

    // 如果虚部为 NaN，根据实部情况返回不同的复数
    if (isnan(__x.imag())) {
        if (isinf(__x.real()) || __x.real() == 0)
            return complex<_Tp>(copysign(_Tp(0), __x.real()), __x.imag());
        return complex<_Tp>(__x.imag(), __x.imag());
    }

    // 如果实部为 NaN，返回一个复数，实部和虚部都是 NaN
    if (isnan(__x.real())) {
        return complex<_Tp>(__x.real(), __x.real());
    }

    // 如果实部为无穷大，返回根据实部符号构造的复数
    if (isinf(__x.real())) {
        return complex<_Tp>(copysign(_Tp(0), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    }

    // 如果实部的绝对值为 1，虚部为 0，则返回符号为无穷大的复数
    if (abs(__x.real()) == _Tp(1) && __x.imag() == _Tp(0)) {
        return complex<_Tp>(copysign(_Tp(INFINITY), __x.real()), copysign(_Tp(0), __x.imag()));
    }

    // 对复数 __x 进行反双曲正切运算，返回结果的复数
    complex<_Tp> __z = log((_Tp(1) + __x) / (_Tp(1) - __x)) / _Tp(2);
    return complex<_Tp>(copysign(__z.real(), __x.real()), copysign(__z.imag(), __x.imag()));
}

// sinh 函数的实现，计算复数的双曲正弦值
template<class _Tp>
complex<_Tp>
sinh(const complex<_Tp>& __x)
{
    // 如果实部是无穷大且虚部不是有限数，返回一个复数，虚部为 NaN
    if (isinf(__x.real()) && !isfinite(__x.imag()))
        return complex<_Tp>(__x.real(), _Tp(NAN));

    // 如果实部是 0 且虚部不是有限数，返回一个复数，虙部为 NaN
    if (__x.real() == 0 && !isfinite(__x.imag()))
        return complex<_Tp>(__x.real(), _Tp(NAN));

    // 如果虚部是 0 且实部不是有限数，返回原复数
    if (__x.imag() == 0 && !isfinite(__x.real()))
        return __x;

    // 计算双曲正弦的复数，返回结果
    return complex<_Tp>(sinh(__x.real()) * cos(__x.imag()), cosh(__x.real()) * sin(__x.imag()));
}

// cosh 函数的实现，计算复数的双曲余弦值
template<class _Tp>
complex<_Tp>
cosh(const complex<_Tp>& __x)
{
    // 如果实部是无穷大且虚部不是有限数，返回一个复数，虚部为 NaN
    if (isinf(__x.real()) && !isfinite(__x.imag()))
        return complex<_Tp>(abs(__x.real()), _Tp(NAN));

    // 如果实部是 0 且虚部不是有限数，返回一个复数，实部为 NaN
    if (__x.real() == 0 && !isfinite(__x.imag()))
        return complex<_Tp>(_Tp(NAN), __x.real());

    // 如果实部和虚部都是 0，返回一个复数，实部为 1，虚部为 0
    if (__x.real() == 0 && __x.imag() == 0)
        return complex<_Tp>(_Tp(1), __x.imag());

    // 如果虚部是 0 且实部不是有限数，返回一个复数，实部为实部的绝对值，虚部不变
    if (__x.imag() == 0 && !isfinite(__x.real()))
        return complex<_Tp>(abs(__x.real()), __x.imag());

    // 计算双曲余弦的复数，返回结果
    return complex<_Tp>(cosh(__x.real()) * cos(__x.imag()), sinh(__x.real()) * sin(__x.imag()));
}

// tanh 函数的实现，计算复数的双曲正切值
template<class _Tp>
complex<_Tp>
tanh(const complex<_Tp>& __x)
{
    // 如果实部是无穷大，根据虚部情况返回复数
    if (isinf(__x.real())) {
        if (!isfinite(__x.imag()))
            return complex<_Tp>(copysign(_Tp(1), __x.real()), _Tp(0));
        return complex<_Tp>(copysign(_Tp(1), __x.real()), copysign(_Tp(0), sin(_Tp(2) * __x.imag())));
    }

    // 如果实部是 NaN 且虚部是 0，返回原复数
    if (isnan(__x.real()) && __x.imag() == 0)
        return __x;

    // 计算复数 __x 的双曲正切，返回结果
    _Tp __2r(_Tp(2) * __x.real());
    _Tp __2i(_Tp(2) * __x.imag());
    _Tp __d(cosh(__2r) + cos(__2i));
    _Tp __2rsh(sinh(__2r));
    if (isinf(__2rsh) && isinf(__d))
        return complex<_Tp>(__2rsh > _Tp(0) ? _Tp(1) : _Tp(-1),
                            __2i > _Tp(0) ? _Tp(0) : _Tp(-0.));
    return  complex<_Tp>(__2rsh/__d, sin(__2i)/__d);
}

// asin 函数的实现，计算复数的反正弦值
template<class _Tp>
complex<_Tp>
asin(const complex<_Tp>& __x)
{
    // 计算复数 __x 的反双曲正弦，返回结果的复数
    complex<_Tp> __z = asinh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}
// acos

// acos 函数的模板定义，计算复数参数的反余弦值
template<class _Tp>
complex<_Tp>
acos(const complex<_Tp>& __x)
{
    // 定义常量 __pi 为 atan2(+0., -0.) 的结果
    const _Tp __pi(atan2(+0., -0.));

    // 如果实部为无穷大
    if (isinf(__x.real()))
    {
        // 如果虚部为 NaN，则返回复数 (虚部, 实部)
        if (isnan(__x.imag()))
            return complex<_Tp>(__x.imag(), __x.real());

        // 如果虚部为无穷大
        if (isinf(__x.imag()))
        {
            // 如果实部小于 0，则返回复数 (0.75 * __pi, -虚部)
            if (__x.real() < _Tp(0))
                return complex<_Tp>(_Tp(0.75) * __pi, -__x.imag());
            // 否则返回复数 (0.25 * __pi, -虚部)
            return complex<_Tp>(_Tp(0.25) * __pi, -__x.imag());
        }

        // 如果实部小于 0，则返回复数 (__pi, 如果虚部符号位为真则 -实部，否则实部)
        if (__x.real() < _Tp(0))
            return complex<_Tp>(__pi, signbit(__x.imag()) ? -__x.real() : __x.real());
        // 否则返回复数 (0, 如果虚部符号位为真则 实部，否则 -实部)
        return complex<_Tp>(_Tp(0), signbit(__x.imag()) ? __x.real() : -__x.real());
    }

    // 如果实部为 NaN
    if (isnan(__x.real()))
    {
        // 如果虚部为无穷大，则返回复数 (实部, -虚部)
        if (isinf(__x.imag()))
            return complex<_Tp>(__x.real(), -__x.imag());
        // 否则返回复数 (实部, 实部)
        return complex<_Tp>(__x.real(), __x.real());
    }

    // 如果虚部为无穷大，则返回复数 (__pi/2, -虚部)
    if (isinf(__x.imag()))
        return complex<_Tp>(__pi/_Tp(2), -__x.imag());

    // 如果实部为 0 且虚部为 0 或为 NaN，则返回复数 (__pi/2, -虚部)
    if (__x.real() == 0 && (__x.imag() == 0 || isnan(__x.imag())))
        return complex<_Tp>(__pi/_Tp(2), -__x.imag());

    // 计算复数 __z = log(__x + sqrt(__sqr(__x) - 1))
    complex<_Tp> __z = log(__x + sqrt(__sqr(__x) - _Tp(1)));

    // 如果虚部的符号位为真，则返回复数 (|__z.imag()|, |__z.real()|)
    if (signbit(__x.imag()))
        return complex<_Tp>(abs(__z.imag()), abs(__z.real()));

    // 否则返回复数 (|__z.imag()|, -|__z.real()|)
    return complex<_Tp>(abs(__z.imag()), -abs(__z.real()));
}

// atan

// atan 函数的模板定义，计算复数参数的反正切值
template<class _Tp>
complex<_Tp>
atan(const complex<_Tp>& __x)
{
    // 计算复数 __z = atanh(complex(-虚部, 实部))
    complex<_Tp> __z = atanh(complex<_Tp>(-__x.imag(), __x.real()));

    // 返回复数 (__z.imag(), -__z.real())
    return complex<_Tp>(__z.imag(), -__z.real());
}

// sin

// sin 函数的模板定义，计算复数参数的正弦值
template<class _Tp>
complex<_Tp>
sin(const complex<_Tp>& __x)
{
    // 计算复数 __z = sinh(complex(-虚部, 实部))
    complex<_Tp> __z = sinh(complex<_Tp>(-__x.imag(), __x.real()));

    // 返回复数 (__z.imag(), -__z.real())
    return complex<_Tp>(__z.imag(), -__z.real());
}

// cos

// cos 函数的模板定义，计算复数参数的余弦值
template<class _Tp>
inline
complex<_Tp>
cos(const complex<_Tp>& __x)
{
    // 返回复数 cosh(complex(-虚部, 实部))
    return cosh(complex<_Tp>(-__x.imag(), __x.real()));
}

// tan

// tan 函数的模板定义，计算复数参数的正切值
template<class _Tp>
complex<_Tp>
tan(const complex<_Tp>& __x)
{
    // 计算复数 __z = tanh(complex(-虚部, 实部))
    complex<_Tp> __z = tanh(complex<_Tp>(-__x.imag(), __x.real()));

    // 返回复数 (__z.imag(), -__z.real())
    return complex<_Tp>(__z.imag(), -__z.real());
}

// Literal suffix for complex number literals [complex.literals]
inline namespace literals
{
  inline namespace complex_literals
  {
    // 定义双精度浮点数的复数后缀字面值操作符 operator""i
    constexpr complex<double> operator""i(long double __im)
    {
        // 返回复数 (0.0, static_cast<double>(__im))
        return { 0.0, static_cast<double>(__im) };
    }

    // 定义双精度浮点数的复数后缀字面值操作符 operator""i
    constexpr complex<double> operator""i(unsigned long long __im)
    {
        // 返回复数 (0.0, static_cast<double>(__im))
        return { 0.0, static_cast<double>(__im) };
    }

    // 定义单精度浮点数的复数后缀字面值操作符 operator""if
    constexpr complex<float> operator""if(long double __im)
    {
        // 返回复数 (0.0f, static_cast<float>(__im))
        return { 0.0f, static_cast<float>(__im) };
    }

    // 定义单精度浮点数的复数后缀字面值操作符 operator""if
    constexpr complex<float> operator""if(unsigned long long __im)
    {
        // 返回复数 (0.0f, static_cast<float>(__im))
        return { 0.0f, static_cast<float>(__im) };
    }
  } // namespace complex_literals
} // namespace literals

} // namespace std

// 获取复杂数学字符串的函数，返回 complex_math
const std::string &get_complex_math_string() {
  return complex_math;
}

} // namespace at::cuda
```