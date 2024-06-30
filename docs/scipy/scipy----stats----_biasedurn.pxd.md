# `D:\src\scipysrc\scipy\scipy\stats\_biasedurn.pxd`

```
# 声明使用外部的 C++ 头文件 "biasedurn/stocc.h"，并且不使用全局解释器锁 (nogil)
cdef extern from "biasedurn/stocc.h" nogil:
    # 声明 C++ 类 CFishersNCHypergeometric
    cdef cppclass CFishersNCHypergeometric:
        # CFishersNCHypergeometric 类的构造函数声明，接受 int, int, int, double, double 参数，不抛出异常
        CFishersNCHypergeometric(int, int, int, double, double) except +
        # 返回众数的方法声明
        int mode()
        # 返回均值的方法声明
        double mean()
        # 返回方差的方法声明
        double variance()
        # 返回指定值 x 的概率的方法声明
        double probability(int x)
        # 计算给定均值和方差的矩的方法声明，参数为双指针
        double moments(double * mean, double * var)

    # 声明 C++ 类 CWalleniusNCHypergeometric
    cdef cppclass CWalleniusNCHypergeometric:
        # CWalleniusNCHypergeometric 类的默认构造函数声明，不抛出异常
        CWalleniusNCHypergeometric() except +
        # CWalleniusNCHypergeometric 类的构造函数声明，接受 int, int, int, double, double 参数，不抛出异常
        CWalleniusNCHypergeometric(int, int, int, double, double) except +
        # 返回众数的方法声明
        int mode()
        # 返回均值的方法声明
        double mean()
        # 返回方差的方法声明
        double variance()
        # 返回指定值 x 的概率的方法声明
        double probability(int x)
        # 计算给定均值和方差的矩的方法声明，参数为双指针
        double moments(double * mean, double * var)

    # 声明 C++ 类 StochasticLib3
    cdef cppclass StochasticLib3:
        # StochasticLib3 类的构造函数声明，接受种子参数，不抛出异常
        StochasticLib3(int seed) except +
        # 返回 [0,1) 范围内随机 double 数的方法声明，不抛出异常
        double Random() except +
        # 设置精度的方法声明，接受 double 参数
        void SetAccuracy(double accur)
        # FishersNCHyp 函数声明，接受 int, int, int, double 参数，返回 int
        int FishersNCHyp (int n, int m, int N, double odds) except +
        # WalleniusNCHyp 函数声明，接受 int, int, int, double 参数，返回 int
        int WalleniusNCHyp (int n, int m, int N, double odds) except +
        # 返回 [0,1) 范围内下一个随机 double 数的函数指针声明
        double(*next_double)()
        # 返回正态分布的下一个随机数的函数指针声明，接受均值 m 和标准差 s 参数
        double(*next_normal)(const double m, const double s)
```