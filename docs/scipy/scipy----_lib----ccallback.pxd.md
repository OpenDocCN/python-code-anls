# `D:\src\scipysrc\scipy\scipy\_lib\ccallback.pxd`

```
# -*-cython-*-  # 指定此文件使用Cython语法

# 导入所需的C语言标准库头文件和Cython的Python对象包
from libc.setjmp cimport jmp_buf  # 导入setjmp.h头文件中的jmp_buf类型
from cpython.object cimport PyObject  # 导入CPython中的PyObject类型

# 从外部头文件"ccallback.h"中声明外部C函数和结构体

cdef extern from "ccallback.h":
    # 定义ccallback_signature_t结构体，包含签名字符串和值
    ctypedef struct ccallback_signature_t:
        char *signature
        int value

    # 定义ccallback_t结构体，包含C函数指针、Python函数对象、用户数据指针、错误跳转缓冲区、前一个回调指针、签名等
    ctypedef struct ccallback_t:
        void *c_function
        PyObject *py_function
        void *user_data
        jmp_buf error_buf
        ccallback_t *prev_callback
        ccallback_signature_t *signature

        # 未使用的变量，供thunk等代码用于任何目的
        long info
        void *info_p

    # 定义CCALLBACK_DEFAULTS、CCALLBACK_OBTAIN、CCALLBACK_PARSE三个整数常量
    int CCALLBACK_DEFAULTS
    int CCALLBACK_OBTAIN
    int CCALLBACK_PARSE

    # 声明ccallback_obtain函数原型，返回ccallback_t指针，无需GIL
    ccallback_t *ccallback_obtain() nogil

    # 声明ccallback_prepare函数原型，接受ccallback_t指针、ccallback_signature_t指针数组、Python函数对象、标志位为参数，抛出-1异常
    int ccallback_prepare(ccallback_t *callback, ccallback_signature_t *sigs,
                          object func, int flags) except -1

    # 声明ccallback_release函数原型，释放ccallback_t结构体对象
    void ccallback_release(ccallback_t *callback)
```