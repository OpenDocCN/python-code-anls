# `D:\src\scipysrc\scikit-learn\sklearn\svm\src\liblinear\_cython_blas_helpers.h`

```
#ifndef _CYTHON_BLAS_HELPERS_H
// 如果 _CYTHON_BLAS_HELPERS_H 宏未定义，则进行条件编译
#define _CYTHON_BLAS_HELPERS_H

// 定义函数指针类型 dot_func，表示一个接受两个向量和各自的步长作为参数的函数，并返回一个 double 值
typedef double (*dot_func)(int, const double*, int, const double*, int);

// 定义函数指针类型 axpy_func，表示一个向量加法函数，接受两个向量、标量倍数和各自的步长作为参数，无返回值
typedef void (*axpy_func)(int, double, const double*, int, double*, int);

// 定义函数指针类型 scal_func，表示一个向量缩放函数，接受一个向量、标量倍数和步长作为参数，无返回值
typedef void (*scal_func)(int, double, const double*, int);

// 定义函数指针类型 nrm2_func，表示一个向量二范数计算函数，接受一个向量和步长作为参数，并返回一个 double 值
typedef double (*nrm2_func)(int, const double*, int);

// 定义结构体 BlasFunctions，用于封装 BLAS 级别的函数指针
typedef struct BlasFunctions {
    dot_func dot;   // 指向向量点积函数的指针
    axpy_func axpy; // 指向向量加法函数的指针
    scal_func scal; // 指向向量缩放函数的指针
    nrm2_func nrm2; // 指向向量二范数函数的指针
} BlasFunctions;

// 结束条件编译指令
#endif
```