# `.\pytorch\aten\src\ATen\div_rtn.h`

```py
#pragma once
// 声明一个只允许头文件被包含一次的预处理指令

// 定义一个模板函数，用于实现整数除法向负无穷方向取整
template <typename T>
static inline T div_rtn(T x, T y) {
    // 计算整数除法的商
    int q = x / y;
    // 计算整数除法的余数
    int r = x % y;
    // 如果余数不为零且符号不同，则向负无穷方向调整商
    if ((r != 0) && ((r < 0) != (y < 0)))
        --q;
    // 返回调整后的商
    return q;
}
```