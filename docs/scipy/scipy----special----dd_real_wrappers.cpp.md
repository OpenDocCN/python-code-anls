# `D:\src\scipysrc\scipy\scipy\special\dd_real_wrappers.cpp`

```
/* These wrappers exist so that double-double extended precision arithmetic
 * now translated to in C++ in special/cephes/dd_real.h can be used in Cython.
 * The original API of the C implementation which existed prior to gh-20390 has
 * been replicated to avoid the need to modify downstream Cython files.
 */

#include "dd_real_wrappers.h"  // 包含自定义的双精度双精度类型包装器头文件
#include "special/cephes/dd_real.h"  // 包含双精度双精度算法的头文件

using special::cephes::detail::double_double;  // 使用双精度双精度命名空间中的 double_double 类型

// 创建一个双精度双精度数，仅初始化高位，低位为0
double2 dd_create_d(double x) {
    return {x, 0.0};
}

// 创建一个双精度双精度数，初始化高位和低位
double2 dd_create(double x, double y) {
    return {x, y};
}

// 双精度双精度数加法
double2 dd_add(const double2* a, const double2* b) {
    double_double dd_a(a->hi, a->lo);  // 转换为双精度双精度对象
    double_double dd_b(b->hi, b->lo);  // 转换为双精度双精度对象
    double_double result = dd_a + dd_b;  // 执行双精度双精度数加法
    return {result.hi, result.lo};  // 返回结果的高位和低位
}

// 双精度双精度数乘法
double2 dd_mul(const double2* a, const double2* b) {
    double_double dd_a(a->hi, a->lo);  // 转换为双精度双精度对象
    double_double dd_b(b->hi, b->lo);  // 转换为双精度双精度对象
    double_double result = dd_a * dd_b;  // 执行双精度双精度数乘法
    return {result.hi, result.lo};  // 返回结果的高位和低位
}

// 双精度双精度数除法
double2 dd_div(const double2* a, const double2* b) {
    double_double dd_a(a->hi, a->lo);  // 转换为双精度双精度对象
    double_double dd_b(b->hi, b->lo);  // 转换为双精度双精度对象
    double_double result = dd_a / dd_b;  // 执行双精度双精度数除法
    return {result.hi, result.lo};  // 返回结果的高位和低位
}

// 双精度双精度数的指数函数
double2 dd_exp(const double2* x) {
    double_double dd_x(x->hi, x->lo);  // 转换为双精度双精度对象
    double_double result = special::cephes::detail::exp(dd_x);  // 调用双精度双精度指数函数
    return {result.hi, result.lo};  // 返回结果的高位和低位
}

// 双精度双精度数的对数函数
double2 dd_log(const double2* x) {
    double_double dd_x(x->hi, x->lo);  // 转换为双精度双精度对象
    double_double result = special::cephes::detail::log(dd_x);  // 调用双精度双精度对数函数
    return {result.hi, result.lo};  // 返回结果的高位和低位
}

// 将双精度双精度数转换为普通双精度数
double dd_to_double(const double2* a) {
    return a->hi;  // 返回高位作为普通双精度数
}
```