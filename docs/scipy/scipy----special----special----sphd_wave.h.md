# `D:\src\scipysrc\scipy\scipy\special\special\sphd_wave.h`

```
#pragma once
#include "specfun.h"

namespace special {

// 计算长椭圆形的特殊函数值
template <typename T>
T prolate_segv(T m, T n, T c) {
    int kd = 1;  // 设定参数 kd 为 1，表示长椭圆形
    int int_m, int_n;  // 定义整数形式的 m 和 n
    T cv = 0.0, *eg;  // 初始化 cv 为 0，eg 为指向 T 类型数据的指针

    // 检查参数的有效性，如果不符合要求则返回 NaN
    if ((m < 0) || (n < m) || (m != floor(m)) || (n != floor(n)) || ((n - m) > 198)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    int_m = (int) m;  // 将 m 转换为整数形式
    int_n = (int) n;  // 将 n 转换为整数形式
    eg = (T *) malloc(sizeof(T) * (n - m + 2));  // 分配存储空间给 eg 数组
    if (eg == NULL) {
        set_error("prolate_segv", SF_ERROR_OTHER, "memory allocation error");  // 内存分配失败时设置错误信息
        return std::numeric_limits<T>::quiet_NaN();  // 返回 NaN
    }
    specfun::segv(int_m, int_n, c, kd, &cv, eg);  // 调用特殊函数计算 segv
    free(eg);  // 释放动态分配的内存
    return cv;  // 返回计算结果
}

// 计算短椭圆形的特殊函数值
template <typename T>
T oblate_segv(T m, T n, T c) {
    int kd = -1;  // 设定参数 kd 为 -1，表示短椭圆形
    int int_m, int_n;  // 定义整数形式的 m 和 n
    T cv = 0.0, *eg;  // 初始化 cv 为 0，eg 为指向 T 类型数据的指针

    // 检查参数的有效性，如果不符合要求则返回 NaN
    if ((m < 0) || (n < m) || (m != floor(m)) || (n != floor(n)) || ((n - m) > 198)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    int_m = (int) m;  // 将 m 转换为整数形式
    int_n = (int) n;  // 将 n 转换为整数形式
    eg = (T *) malloc(sizeof(T) * (n - m + 2));  // 分配存储空间给 eg 数组
    if (eg == NULL) {
        set_error("oblate_segv", SF_ERROR_OTHER, "memory allocation error");  // 内存分配失败时设置错误信息
        return std::numeric_limits<T>::quiet_NaN();  // 返回 NaN
    }
    specfun::segv(int_m, int_n, c, kd, &cv, eg);  // 调用特殊函数计算 segv
    free(eg);  // 释放动态分配的内存
    return cv;  // 返回计算结果
}

// 计算长椭圆形的特殊函数值并且不返回 cv
template <typename T>
void prolate_aswfa_nocv(T m, T n, T c, T x, T &s1f, T &s1d) {
    int kd = 1;  // 设定参数 kd 为 1，表示长椭圆形
    int int_m, int_n;  // 定义整数形式的 m 和 n
    T cv = 0.0, *eg;  // 初始化 cv 为 0，eg 为指向 T 类型数据的指针

    // 检查参数的有效性，如果不符合要求则设置错误信息并返回 NaN
    if ((x >= 1) || (x <= -1) || (m < 0) || (n < m) || (m != floor(m)) || (n != floor(n)) || ((n - m) > 198)) {
        set_error("prolate_aswfa_nocv", SF_ERROR_DOMAIN, NULL);
        s1d = std::numeric_limits<T>::quiet_NaN();
        s1f = std::numeric_limits<T>::quiet_NaN();
        return;
    }

    int_m = (int) m;  // 将 m 转换为整数形式
    int_n = (int) n;  // 将 n 转换为整数形式
    eg = (T *) malloc(sizeof(T) * (n - m + 2));  // 分配存储空间给 eg 数组
    if (eg == NULL) {
        set_error("prolate_aswfa_nocv", SF_ERROR_OTHER, "memory allocation error");  // 内存分配失败时设置错误信息
        s1d = std::numeric_limits<T>::quiet_NaN();
        s1f = std::numeric_limits<T>::quiet_NaN();
        return;
    }
    specfun::segv(int_m, int_n, c, kd, &cv, eg);  // 调用特殊函数计算 segv
    specfun::aswfa(x, int_m, int_n, c, kd, cv, &s1f, &s1d);  // 调用特殊函数计算 aswfa
    free(eg);  // 释放动态分配的内存
}

// 计算短椭圆形的特殊函数值并且不返回 cv
template <typename T>
void oblate_aswfa_nocv(T m, T n, T c, T x, T &s1f, T &s1d) {
    int kd = -1;  // 设定参数 kd 为 -1，表示短椭圆形
    int int_m, int_n;  // 定义整数形式的 m 和 n
    T cv = 0.0, *eg;  // 初始化 cv 为 0，eg 为指向 T 类型数据的指针

    // 检查参数的有效性，如果不符合要求则设置错误信息并返回 NaN
    if ((x >= 1) || (x <= -1) || (m < 0) || (n < m) || (m != floor(m)) || (n != floor(n)) || ((n - m) > 198)) {
        set_error("oblate_aswfa_nocv", SF_ERROR_DOMAIN, NULL);
        s1d = std::numeric_limits<T>::quiet_NaN();
        s1f = std::numeric_limits<T>::quiet_NaN();
        return;
    }

    int_m = (int) m;  // 将 m 转换为整数形式
    int_n = (int) n;  // 将 n 转换为整数形式
    eg = (T *) malloc(sizeof(T) * (n - m + 2));  // 分配存储空间给 eg 数组
    if (eg == NULL) {
        set_error("oblate_aswfa_nocv", SF_ERROR_OTHER, "memory allocation error");  // 内存分配失败时设置错误信息
        s1d = std::numeric_limits<T>::quiet_NaN();
        s1f = std::numeric_limits<T>::quiet_NaN();
        return;
    }
    specfun::segv(int_m, int_n, c, kd, &cv, eg);  // 调用特殊函数计算 segv
    specfun::aswfa(x, int_m, int_n, c, kd, cv, &s1f, &s1d);  // 调用特殊函数计算 aswfa
    free(eg);  // 释放动态分配的内存
}
    # 释放指针 eg 指向的内存空间
    free(eg);
// 以模板形式定义了一个函数 prolate_aswfa，接受多种类型的参数，并计算 prolate 椭球的相关函数值
template <typename T>
void prolate_aswfa(T m, T n, T c, T cv, T x, T &s1f, T &s1d) {
    // 检查输入参数的有效性，若不符合条件则设置错误并返回 NaN 值
    if ((x >= 1) || (x <= -1) || (m < 0) || (n < m) || (m != floor(m)) || (n != floor(n))) {
        set_error("prolate_aswfa", SF_ERROR_DOMAIN, NULL);
        s1f = std::numeric_limits<T>::quiet_NaN();
        s1d = std::numeric_limits<T>::quiet_NaN();
    } else {
        // 调用特殊函数库中的 aswfa 函数计算 prolate 椭球的函数值
        specfun::aswfa(x, static_cast<int>(m), static_cast<int>(n), c, 1, cv, &s1f, &s1d);
    }
}

// 以模板形式定义了一个函数 oblate_aswfa，接受多种类型的参数，并计算 oblate 椭球的相关函数值
template <typename T>
void oblate_aswfa(T m, T n, T c, T cv, T x, T &s1f, T &s1d) {
    // 检查输入参数的有效性，若不符合条件则设置错误并返回 NaN 值
    if ((x >= 1) || (x <= -1) || (m < 0) || (n < m) || (m != floor(m)) || (n != floor(n))) {
        set_error("oblate_aswfa", SF_ERROR_DOMAIN, NULL);
        s1f = std::numeric_limits<T>::quiet_NaN();
        s1d = std::numeric_limits<T>::quiet_NaN();
    } else {
        // 调用特殊函数库中的 aswfa 函数计算 oblate 椭球的函数值
        specfun::aswfa(x, static_cast<int>(m), static_cast<int>(n), c, -1, cv, &s1f, &s1d);
    }
}

// 以模板形式定义了一个函数 prolate_radial1_nocv，接受多种类型的参数，并计算 prolate 椭球的径向函数值（不考虑传播速度）
template <typename T>
void prolate_radial1_nocv(T m, T n, T c, T x, T &r1f, T &r1d) {
    int kf = 1, kd = 1;
    T r2f = 0.0, r2d = 0.0, cv = 0.0, *eg;
    int int_m, int_n;

    // 检查输入参数的有效性，若不符合条件则设置错误并返回 NaN 值
    if ((x <= 1.0) || (m < 0) || (n < m) || (m != floor(m)) || (n != floor(n)) || ((n - m) > 198)) {
        set_error("prolate_radial1_nocv", SF_ERROR_DOMAIN, NULL);
        r1d = std::numeric_limits<T>::quiet_NaN();
        r1f = std::numeric_limits<T>::quiet_NaN();
        return;
    }
    int_m = (int) m;
    int_n = (int) n;
    // 分配并初始化额外的内存空间
    eg = (T *) malloc(sizeof(T) * (n - m + 2));
    if (eg == NULL) {
        // 若内存分配失败，则设置错误并返回 NaN 值
        set_error("prolate_radial1_nocv", SF_ERROR_OTHER, "memory allocation error");
        r1d = std::numeric_limits<T>::quiet_NaN();
        r1f = std::numeric_limits<T>::quiet_NaN();
        return;
    }
    // 调用特殊函数库中的 segv 函数计算 prolate 椭球的参数值
    specfun::segv(int_m, int_n, c, kd, &cv, eg);
    // 调用特殊函数库中的 rswfp 函数计算 prolate 椭球的径向函数值
    specfun::rswfp(int_m, int_n, c, x, cv, kf, &r1f, &r1d, &r2f, &r2d);
    // 释放额外分配的内存空间
    free(eg);
}

// 以模板形式定义了一个函数 prolate_radial2_nocv，接受多种类型的参数，并计算 prolate 椭球的另一径向函数值（不考虑传播速度）
template <typename T>
void prolate_radial2_nocv(T m, T n, T c, T x, T &r2f, T &r2d) {
    int kf = 2, kd = 1;
    T r1f = 0.0, r1d = 0.0, cv = 0.0, *eg;
    int int_m, int_n;

    // 检查输入参数的有效性，若不符合条件则设置错误并返回 NaN 值
    if ((x <= 1.0) || (m < 0) || (n < m) || (m != floor(m)) || (n != floor(n)) || ((n - m) > 198)) {
        set_error("prolate_radial2_nocv", SF_ERROR_DOMAIN, NULL);
        r2d = std::numeric_limits<T>::quiet_NaN();
        r2f = std::numeric_limits<T>::quiet_NaN();
        return;
    }
    int_m = (int) m;
    int_n = (int) n;
    // 分配并初始化额外的内存空间
    eg = (T *) malloc(sizeof(T) * (n - m + 2));
    if (eg == NULL) {
        // 若内存分配失败，则设置错误并返回 NaN 值
        set_error("prolate_radial2_nocv", SF_ERROR_OTHER, "memory allocation error");
        r2d = std::numeric_limits<T>::quiet_NaN();
        r2f = std::numeric_limits<T>::quiet_NaN();
        return;
    }
    // 调用特殊函数库中的 segv 函数计算 prolate 椭球的参数值
    specfun::segv(int_m, int_n, c, kd, &cv, eg);
    // 调用特殊函数库中的 rswfp 函数计算 prolate 椭球的径向函数值
    specfun::rswfp(int_m, int_n, c, x, cv, kf, &r1f, &r1d, &r2f, &r2d);
    // 释放额外分配的内存空间
    free(eg);
}

// 以模板形式定义了一个函数 prolate_radial1，接受多种类型的参数，并计算 prolate 椭球的径向函数值（考虑传播速度）
template <typename T>
void prolate_radial1(T m, T n, T c, T cv, T x, T &r1f, T &r1d) {
    int kf = 1;
    T r2f = 0.0, r2d = 0.0;
    int int_m, int_n;

    // 检查输入参数的有效性，若不符合条件则设置错误并返回 NaN 值
    if ((x <= 1.0) || (m < 0) || (n < m) || (m != floor(m)) || (n != floor(n)) || ((n - m) > 198)) {
        set_error("prolate_radial1", SF_ERROR_DOMAIN, NULL);
        r1d = std::numeric_limits<T>::quiet_NaN();
        r1f = std::numeric_limits<T>::quiet_NaN();
        return;
    }
    int_m = (int) m;
    int_n = (int) n;
    // 调用特殊函数库中的 rswfp 函数计算 prolate 椭球的径向函数值
    specfun::rswfp(int_m, int_n, c, x, cv, kf, &r1f, &r1d, &r2f, &r2d);
}
    # 如果 x 小于等于 1.0 或者 m 小于 0 或者 n 小于 m 或者 m 不等于其向下取整值 或者 n 不等于其向下取整值
    if ((x <= 1.0) || (m < 0) || (n < m) || (m != floor(m)) || (n != floor(n))) {
        # 设置错误信息，指示出现在 prolate_radial1 函数中的域错误
        set_error("prolate_radial1", SF_ERROR_DOMAIN, NULL);
        # 将 r1f 和 r1d 设为 NaN
        r1f = std::numeric_limits<T>::quiet_NaN();
        r1d = std::numeric_limits<T>::quiet_NaN();
    } else {
        # 将 m 和 n 强制转换为整数
        int_m = (int) m;
        int_n = (int) n;
        # 调用 specfun 命名空间中的 rswfp 函数，计算特定参数下的一些函数值
        specfun::rswfp(int_m, int_n, c, x, cv, kf, &r1f, &r1d, &r2f, &r2d);
    }
template <typename T>
void oblate_radial2(T m, T n, T c, T cv, T x, T &r2f, T &r2d) {
    // 设置默认的 KF 和 KD 值
    int kf = 2, kd = -1;
    // 初始化 r1f 和 r1d 为 0.0
    T r1f = 0.0, r1d = 0.0, *eg;
    // 定义整数类型的变量 int_m 和 int_n
    int int_m, int_n;

    // 检查输入参数是否合法
    if ((x < 0.0) || (m < 0) || (n < m) || (m != floor(m)) || (n != floor(n)) || ((n - m) > 198)) {
        // 如果参数不合法，设置错误信息，并返回 NaN
        set_error("oblate_radial2", SF_ERROR_DOMAIN, NULL);
        r2d = std::numeric_limits<T>::quiet_NaN();
        r2f = std::numeric_limits<T>::quiet_NaN();
        return;
    }
    // 将 m 和 n 转换为整数类型
    int_m = (int) m;
    int_n = (int) n;
    // 分配内存用于存储数组 eg
    eg = (T *) malloc(sizeof(T) * (n - m + 2));
    // 检查内存分配是否成功
    if (eg == NULL) {
        // 如果分配失败，设置错误信息，并返回 NaN
        set_error("oblate_radial2", SF_ERROR_OTHER, "memory allocation error");
        r2d = std::numeric_limits<T>::quiet_NaN();
        r2f = std::numeric_limits<T>::quiet_NaN();
        return;
    }
    // 计算特殊函数 segv，并得到 cv 的值
    specfun::segv(int_m, int_n, c, kd, &cv, eg);
    // 计算特殊函数 rswfo，并得到 r1f 和 r1d 的值
    specfun::rswfo(int_m, int_n, c, x, cv, kf, &r1f, &r1d, &r2f, &r2d);
    // 释放动态分配的内存
    free(eg);
}
    # 设置 kf 的初始值为 2，这是一个整数
    int kf = 2;
    # 初始化 r1f 和 r1d 为 0.0，这两个变量用于存储浮点数值
    T r1f = 0.0, r1d = 0.0;

    # 检查参数 x, m, n 是否满足特定条件，若不满足则设置错误信息并将 r2f 和 r2d 设为 NaN
    if ((x < 0.0) || (m < 0) || (n < m) || (m != floor(m)) || (n != floor(n))) {
        # 设置错误信息为 "oblate_radial2"，错误类型为 SF_ERROR_DOMAIN，错误参数为 NULL
        set_error("oblate_radial2", SF_ERROR_DOMAIN, NULL);
        # 将 r2f 和 r2d 分别设置为 NaN
        r2f = std::numeric_limits<T>::quiet_NaN();
        r2d = std::numeric_limits<T>::quiet_NaN();
    } else {
        # 调用特殊函数库中的 rswfo 函数来计算特定参数下的结果，并存储在 r1f, r1d, r2f, r2d 中
        specfun::rswfo(static_cast<int>(m), static_cast<int>(n), c, x, cv, kf, &r1f, &r1d, &r2f, &r2d);
    }
}

// 结束 special 命名空间
} // namespace special
```