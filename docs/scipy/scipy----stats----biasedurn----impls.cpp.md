# `D:\src\scipysrc\scipy\scipy\stats\biasedurn\impls.cpp`

```
// R_BUILD excludes some function implementations;
// patch them in here.

// 引入标准异常库，用于抛出异常
#include <stdexcept>
// 引入自定义的头文件 "stocc.h"，包含了一些随机数生成的函数声明
#include "stocc.h"

// 定义 StochasticLib1 命名空间中的 Normal 函数，生成一个服从正态分布的随机数
double StochasticLib1::Normal(double m, double s) {
    // 调用自定义的 next_normal 函数，生成指定均值 m 和标准差 s 的正态分布随机数
    return next_normal(m, s);
}

// 定义 FatalError 函数，用于抛出运行时异常，其参数为错误信息字符串
void FatalError(const char* msg) {
    // 抛出 std::runtime_error 异常，其错误信息为 msg
    throw std::runtime_error(msg);
}
```