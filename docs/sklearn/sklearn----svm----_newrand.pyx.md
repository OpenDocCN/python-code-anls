# `D:\src\scipysrc\scikit-learn\sklearn\svm\_newrand.pyx`

```
"""Wrapper for newrand.h"""

# 从 C 头文件 "newrand.h" 中导入以下两个函数
cdef extern from "newrand.h":
    # 声明一个名为 set_seed 的函数，接受一个无符号整数参数
    void set_seed(unsigned int)
    # 声明一个名为 bounded_rand_int 的函数，接受一个无符号整数参数，并返回一个无符号整数值
    unsigned int bounded_rand_int(unsigned int)


# 定义一个 Python 函数 set_seed_wrap，用于调用 C 函数 set_seed
def set_seed_wrap(unsigned int custom_seed):
    set_seed(custom_seed)


# 定义一个 Python 函数 bounded_rand_int_wrap，用于调用 C 函数 bounded_rand_int
def bounded_rand_int_wrap(unsigned int range_):
    # 返回 bounded_rand_int 函数的结果，传入 range_ 参数
    return bounded_rand_int(range_)
```