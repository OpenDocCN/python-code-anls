# `D:\src\scipysrc\scipy\scipy\special\dd_real_wrappers.h`

```
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    // 定义双精度浮点数结构体，包含高位和低位
    typedef struct double2 {
        double hi;  // 高位
        double lo;  // 低位
    } double2;

    // 创建一个双精度浮点数结构体，初始值为单个浮点数 x
    double2 dd_create_d(double x);

    // 创建一个双精度浮点数结构体，初始值为两个浮点数 x 和 y
    double2 dd_create(double x, double y);

    // 对两个双精度浮点数结构体进行加法操作，返回结果
    double2 dd_add(const double2* a, const double2* b);

    // 对两个双精度浮点数结构体进行乘法操作，返回结果
    double2 dd_mul(const double2* a, const double2* b);

    // 对两个双精度浮点数结构体进行除法操作，返回结果
    double2 dd_div(const double2* a, const double2* b);

    // 对一个双精度浮点数结构体进行指数函数操作，返回结果
    double2 dd_exp(const double2* x);

    // 对一个双精度浮点数结构体进行自然对数操作，返回结果
    double2 dd_log(const double2* x);

    // 将双精度浮点数结构体转换为普通双精度浮点数
    double dd_to_double(const double2* a);

#ifdef __cplusplus
}
#endif
```