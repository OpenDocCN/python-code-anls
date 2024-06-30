# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools\complex_ops.h`

```
#ifndef COMPLEX_OPS_H
#define COMPLEX_OPS_H

/*
 *  Functions to handle arithmetic operations on NumPy complex values
 */

#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include "npy_2_complexcompat.h"

// 定义一个模板类 complex_wrapper，用于处理复杂类型操作
template <class c_type, class npy_type>
class complex_wrapper {
    private:
        npy_type complex; // 私有变量，存储 NumPy 复杂数类型
        // 获取实部值的方法，返回 c_type 类型
        c_type real() const { return c_type(0); }
        // 获取虚部值的方法，返回 c_type 类型
        c_type imag() const { return c_type(0); }
        // 设置实部值的方法，参数为 c_type 类型
        void set_real(const c_type r) { }
        // 设置虚部值的方法，参数为 c_type 类型
        void set_imag(const c_type i) { }

};

// 下面是各个具体模板类型的实现

// 对于 float 类型的 complex_wrapper，实现获取实部值的方法
template <>
inline float complex_wrapper<float, npy_cfloat>::real() const {
    return npy_crealf(this->complex);
}

// 对于 float 类型的 complex_wrapper，实现设置实部值的方法
template <>
inline void complex_wrapper<float, npy_cfloat>::set_real(const float r) {
    NPY_CSETREALF(&this->complex, r);
}

// 对于 double 类型的 complex_wrapper，实现获取实部值的方法
template <>
inline double complex_wrapper<double, npy_cdouble>::real() const {
    return npy_creal(this->complex);
}

// 对于 double 类型的 complex_wrapper，实现设置实部值的方法
template <>
inline void complex_wrapper<double, npy_cdouble>::set_real(const double r) {
    NPY_CSETREAL(&this->complex, r);
}

// 对于 long double 类型的 complex_wrapper，实现获取实部值的方法
template <>
inline long double complex_wrapper<long double, npy_clongdouble>::real() const {
    return npy_creall(this->complex);
}

// 对于 long double 类型的 complex_wrapper，实现设置实部值的方法
template <>
inline void complex_wrapper<long double, npy_clongdouble>::set_real(const long double r) {
    NPY_CSETREALL(&this->complex, r);
}

// 对于 float 类型的 complex_wrapper，实现获取虚部值的方法
template <>
inline float complex_wrapper<float, npy_cfloat>::imag() const {
    return npy_cimagf(this->complex);
}

// 对于 float 类型的 complex_wrapper，实现设置虚部值的方法
template <>
inline void complex_wrapper<float, npy_cfloat>::set_imag(const float i) {
    NPY_CSETIMAGF(&this->complex, i);
}

// 对于 double 类型的 complex_wrapper，实现获取虚部值的方法
template <>
inline double complex_wrapper<double, npy_cdouble>::imag() const {
    return npy_cimag(this->complex);
}

// 对于 double 类型的 complex_wrapper，实现设置虚部值的方法
template <>
inline void complex_wrapper<double, npy_cdouble>::set_imag(const double i) {
    NPY_CSETIMAG(&this->complex, i);
}

// 对于 long double 类型的 complex_wrapper，实现获取虚部值的方法
template <>
inline long double complex_wrapper<long double, npy_clongdouble>::imag() const {
    return npy_cimagl(this->complex);
}

// 对于 long double 类型的 complex_wrapper，实现设置虚部值的方法
template <>
inline void complex_wrapper<long double, npy_clongdouble>::set_imag(const long double i) {
    NPY_CSETIMAGL(&this->complex, i);
}

// 定义 typedef 别名，简化对 complex_wrapper 模板类的使用
typedef complex_wrapper<float,npy_cfloat> npy_cfloat_wrapper;
typedef complex_wrapper<double,npy_cdouble> npy_cdouble_wrapper;
typedef complex_wrapper<long double,npy_clongdouble> npy_clongdouble_wrapper;

#endif
```