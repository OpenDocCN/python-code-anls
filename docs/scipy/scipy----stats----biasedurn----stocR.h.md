# `D:\src\scipysrc\scipy\scipy\stats\biasedurn\stocR.h`

```
#ifndef _STOCR_H_  // 如果_STOCR_H_未定义，则开始条件编译
#define _STOCR_H_

#include <cstddef>  // 包含标准库头文件<cstddef>，用于NULL的定义

struct StocRBase {  // 定义结构体StocRBase，用于随机数生成器的基类

  double(*next_double)();  // 函数指针成员，用于生成一个双精度浮点型随机数
  double(*next_normal)(const double m, const double s);  // 函数指针成员，用于生成一个正态分布的随机数

  StocRBase() : next_double(NULL), next_normal(NULL) {}  // 默认构造函数，初始化函数指针为NULL
  StocRBase(int seed) : next_double(NULL), next_normal(NULL) {}  // 带参构造函数，初始化函数指针为NULL

  double Random() {  // 成员函数，返回一个随机数
    return next_double();  // 调用next_double函数指针，返回生成的随机数
  }

  double Normal(double m, double s) {  // 成员函数，返回一个正态分布的随机数
    // Also see impls.cpp for the StochasticLib1 implementation
    // (should be identical to this)
    return next_normal(m, s);  // 调用next_normal函数指针，返回生成的正态分布随机数
  }
};

#endif // 如果_STOCR_H_已定义，则结束条件编译
```