# `D:\src\scipysrc\scipy\scipy\_lib\tests\test_boost_build.cpp`

```
// 测试 Boost 头文件的可用性

// 引入标准输入输出流库
#include <iostream>
// 引入 Boost 数学分布头文件
#include <boost/math/distributions.hpp>

// 定义一个测试函数
void test() {
    // 创建一个双精度浮点数二项分布对象，参数为（n=10, p=0.5）
    boost::math::binomial_distribution<double> d(10, 0.5);
}
```