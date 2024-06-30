# `D:\src\scipysrc\scipy\scipy\special\special\ellipk.h`

```
#pragma once

// 使用 `#pragma once` 预处理指令，确保头文件只被编译一次，防止多重包含问题。


#include "cephes/ellpk.h"
#include "config.h"

// 包含头文件 `"cephes/ellpk.h"` 和 `"config.h"`，用于后续函数的声明和配置信息的获取。


namespace special {

// 定义命名空间 `special`，用于封装特殊函数的实现。


SPECFUN_HOST_DEVICE inline double ellipk(double m) { return cephes::ellpk(1.0 - m); }

// 定义内联函数 `ellipk`，接受一个 `double` 类型参数 `m`，调用 `cephes::ellpk` 函数并返回其结果，该函数计算椭圆积分第一类的值。


} // namespace special

// 命名空间 `special` 的结束标记。
```