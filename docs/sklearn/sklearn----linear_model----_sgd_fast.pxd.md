# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_sgd_fast.pxd`

```
# SPDX-License-Identifier: BSD-3-Clause
"""Helper to load LossFunction from sgd_fast.pyx to sag_fast.pyx"""

# 定义一个 Cython 类 LossFunction，用于表示损失函数
cdef class LossFunction:
    # 定义损失函数方法，接受两个 double 类型的参数 y 和 p，不抛出异常，在没有全局解锁的情况下执行
    cdef double loss(self, double y, double p) noexcept nogil
    # 定义损失函数的导数方法，接受两个 double 类型的参数 y 和 p，不抛出异常，在没有全局解锁的情况下执行
    cdef double dloss(self, double y, double p) noexcept nogil


# 定义一个 Cython 类 Regression，继承自 LossFunction 类
cdef class Regression(LossFunction):
    # 重写 LossFunction 中的损失函数方法，接受两个 double 类型的参数 y 和 p，不抛出异常，在没有全局解锁的情况下执行
    cdef double loss(self, double y, double p) noexcept nogil
    # 重写 LossFunction 中的损失函数导数方法，接受两个 double 类型的参数 y 和 p，不抛出异常，在没有全局解锁的情况下执行
    cdef double dloss(self, double y, double p) noexcept nogil


# 定义一个 Cython 类 Classification，继承自 LossFunction 类
cdef class Classification(LossFunction):
    # 重写 LossFunction 中的损失函数方法，接受两个 double 类型的参数 y 和 p，不抛出异常，在没有全局解锁的情况下执行
    cdef double loss(self, double y, double p) noexcept nogil
    # 重写 LossFunction 中的损失函数导数方法，接受两个 double 类型的参数 y 和 p，不抛出异常，在没有全局解锁的情况下执行
    cdef double dloss(self, double y, double p) noexcept nogil


# 定义一个 Cython 类 Log，继承自 Classification 类
cdef class Log(Classification):
    # 重写 Classification 中的损失函数方法，接受两个 double 类型的参数 y 和 p，不抛出异常，在没有全局解锁的情况下执行
    cdef double loss(self, double y, double p) noexcept nogil
    # 重写 Classification 中的损失函数导数方法，接受两个 double 类型的参数 y 和 p，不抛出异常，在没有全局解锁的情况下执行
    cdef double dloss(self, double y, double p) noexcept nogil


# 定义一个 Cython 类 SquaredLoss，继承自 Regression 类
cdef class SquaredLoss(Regression):
    # 重写 Regression 中的损失函数方法，接受两个 double 类型的参数 y 和 p，不抛出异常，在没有全局解锁的情况下执行
    cdef double loss(self, double y, double p) noexcept nogil
    # 重写 Regression 中的损失函数导数方法，接受两个 double 类型的参数 y 和 p，不抛出异常，在没有全局解锁的情况下执行
    cdef double dloss(self, double y, double p) noexcept nogil
```