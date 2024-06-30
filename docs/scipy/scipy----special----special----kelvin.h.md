# `D:\src\scipysrc\scipy\scipy\special\special\kelvin.h`

```
#pragma once

#include "specfun.h"

namespace special {

namespace detail {

    // 定义模板类，用于处理特定的数学函数
    template <typename T>
    // 开始 ber 函数的定义
    T ber(T x) {
        // 定义复数对象 Be
        std::complex<T> Be;
        // 定义一系列局部变量 ber, bei, ger, gei, der, dei, her, hei
        T ber, bei, ger, gei, der, dei, her, hei;

        // 如果输入 x 小于 0，则取其相反数
        if (x < 0) {
            x = -x;
        }

        // 调用 detail 命名空间中的 klvna 函数，获取计算结果
        detail::klvna(x, &ber, &bei, &ger, &gei, &der, &dei, &her, &hei);
        // 设置复数 Be 的实部和虚部
        Be.real(ber);
        Be.imag(bei);
        // 调用 SPECFUN_ZCONVINF 宏，输出日志信息
        SPECFUN_ZCONVINF("ber", Be);
        // 返回复数 Be 的实部
        return Be.real();
    }

    // 开始 bei 函数的定义
    template <typename T>
    T bei(T x) {
        // 定义复数对象 Be
        std::complex<T> Be;
        // 定义一系列局部变量 ber, bei, ger, gei, der, dei, her, hei
        T ber, bei, ger, gei, der, dei, her, hei;

        // 如果输入 x 小于 0，则取其相反数
        if (x < 0) {
            x = -x;
        }

        // 调用 detail 命名空间中的 klvna 函数，获取计算结果
        detail::klvna(x, &ber, &bei, &ger, &gei, &der, &dei, &her, &hei);
        // 设置复数 Be 的实部和虚部
        Be.real(ber);
        Be.imag(bei);
        // 调用 SPECFUN_ZCONVINF 宏，输出日志信息
        SPECFUN_ZCONVINF("bei", Be);
        // 返回复数 Be 的虚部
        return Be.imag();
    }

    // 开始 ker 函数的定义
    template <typename T>
    T ker(T x) {
        // 定义复数对象 Ke
        std::complex<T> Ke;
        // 定义一系列局部变量 ber, bei, ger, gei, der, dei, her, hei
        T ber, bei, ger, gei, der, dei, her, hei;

        // 如果输入 x 小于 0，则返回 NaN
        if (x < 0) {
            return std::numeric_limits<T>::quiet_NaN();
        }

        // 调用 detail 命名空间中的 klvna 函数，获取计算结果
        detail::klvna(x, &ber, &bei, &ger, &gei, &der, &dei, &her, &hei);
        // 设置复数 Ke 的实部和虚部
        Ke.real(ger);
        Ke.imag(gei);
        // 调用 SPECFUN_ZCONVINF 宏，输出日志信息
        SPECFUN_ZCONVINF("ker", Ke);
        // 返回复数 Ke 的实部
        return Ke.real();
    }

    // 开始 kei 函数的定义
    template <typename T>
    T kei(T x) {
        // 定义复数对象 Ke
        std::complex<T> Ke;
        // 定义一系列局部变量 ber, bei, ger, gei, der, dei, her, hei
        T ber, bei, ger, gei, der, dei, her, hei;

        // 如果输入 x 小于 0，则返回 NaN
        if (x < 0) {
            return std::numeric_limits<T>::quiet_NaN();
        }

        // 调用 detail 命名空间中的 klvna 函数，获取计算结果
        detail::klvna(x, &ber, &bei, &ger, &gei, &der, &dei, &her, &hei);
        // 设置复数 Ke 的实部和虚部
        Ke.real(ger);
        Ke.imag(gei);
        // 调用 SPECFUN_ZCONVINF 宏，输出日志信息
        SPECFUN_ZCONVINF("kei", Ke);
        // 返回复数 Ke 的虚部
        return Ke.imag();
    }

    // 开始 berp 函数的定义
    template <typename T>
    T berp(T x) {
        // 定义复数对象 Bep
        std::complex<T> Bep;
        // 定义一系列局部变量 ber, bei, ger, gei, der, dei, her, hei
        T ber, bei, ger, gei, der, dei, her, hei;
        // 定义标志变量 flag，初始值为 0
        int flag = 0;

        // 如果输入 x 小于 0，则将 x 取相反数，并设置 flag 为 1
        if (x < 0) {
            x = -x;
            flag = 1;
        }

        // 调用 detail 命名空间中的 klvna 函数，获取计算结果
        detail::klvna(x, &ber, &bei, &ger, &gei, &der, &dei, &her, &hei);
        // 设置复数 Bep 的实部和虚部
        Bep.real(der);
        Bep.imag(dei);
        // 调用 SPECFUN_ZCONVINF 宏，输出日志信息
        SPECFUN_ZCONVINF("berp", Bep);
        // 如果 flag 为真，则返回 Bep 实部的相反数，否则返回 Bep 的实部
        if (flag) {
            return -Bep.real();
        }
        return Bep.real();
    }

    // 开始 beip 函数的定义
    template <typename T>
    T beip(T x) {
        // 定义复数对象 Bep
        std::complex<T> Bep;
        // 定义一系列局部变量 ber, bei, ger, gei, der, dei, her, hei
        T ber, bei, ger, gei, der, dei, her, hei;
        // 定义标志变量 flag，初始值为 0
        int flag = 0;

        // 如果输入 x 小于 0，则将 x 取相反数，并设置 flag 为 1
        if (x < 0) {
            x = -x;
            flag = 1;
        }
        // 调用 detail 命名空间中的 klvna 函数，获取计算结果
        detail::klvna(x, &ber, &bei, &ger, &gei, &der, &dei, &her, &hei);
        // 设置复数 Bep 的实部和虚部
        Bep.real(der);
        Bep.imag(dei);
        // 调用 SPECFUN_ZCONVINF 宏，输出日志信息
        SPECFUN_ZCONVINF("beip", Bep);
        // 如果 flag 为真，则返回 Bep 虚部的相反数，否则返回 Bep 的虚部
        if (flag) {
            return -Bep.imag();
        }
        return Bep.imag();
    }

    // 开始 kerp 函数的定义
    template <typename T>
    T kerp(T x) {
        // 定义复数对象 Kep
        std::complex<T> Kep;
        // 定义一系列局部变量 ber, bei, ger, gei, der, dei, her, hei

        T ber, bei, ger, gei, der, dei, her, hei;

        // 如果输入 x 小于 0，则返回 NaN
        if (x < 0) {
            return std::numeric_limits<T>::quiet_NaN();
        }
        // 调用 detail 命名空间中的 klvna 函数，获取计算结果
        detail::klvna(x, &ber, &bei, &ger, &gei, &der, &dei, &her, &hei);
        // 设置复数 Kep 的实部和虚部
        Kep.real(her);
        Kep.imag(hei);
        // 调用 SPECFUN_ZCONVINF 宏，输出日志信息
        SPECFUN_ZCONVINF("kerp", Kep);
        // 返回复数 Kep 的实部
        return Kep.real
void kelvin(T x, std::complex<T> &Be, std::complex<T> &Ke, std::complex<T> &Bep, std::complex<T> &Kep) {
    // 初始化标志位，用于处理负数情况
    int flag = 0;
    T ber, bei, ger, gei, der, dei, her, hei;
    
    // 如果输入 x 为负数，则转为正数并设置标志位为1
    if (x < 0) {
        x = -x;
        flag = 1;
    }

    // 调用 detail 命名空间中的 klvna 函数，计算 Kelvin 函数及其导数
    detail::klvna(x, &ber, &bei, &ger, &gei, &der, &dei, &her, &hei);
    
    // 将计算得到的结果赋值给复数对象 Be, Ke, Bep, Kep 的实部和虚部
    Be.real(ber);
    Be.imag(bei);
    Ke.real(ger);
    Ke.imag(gei);
    Bep.real(der);
    Bep.imag(dei);
    Kep.real(her);
    Kep.imag(hei);

    // 调用 SPECFUN_ZCONVINF 宏处理边界情况
    SPECFUN_ZCONVINF("klvna", Be);
    SPECFUN_ZCONVINF("klvna", Ke);
    SPECFUN_ZCONVINF("klvna", Bep);
    SPECFUN_ZCONVINF("klvna", Kep);
    
    // 如果标志位为真，则对 Bep 和 Ke 进行特殊处理
    if (flag) {
        Bep.real(-Bep.real());
        Bep.imag(-Bep.imag());
        Ke.real(std::numeric_limits<T>::quiet_NaN());
        Ke.imag(std::numeric_limits<T>::quiet_NaN());
        Kep.real(std::numeric_limits<T>::quiet_NaN());
        Kep.imag(std::numeric_limits<T>::quiet_NaN());
    }
}

inline void klvnzo(int nt, int kd, double *zo) {
    // ====================================================
    // Purpose: Compute the zeros of Kelvin functions
    // Input :  NT  --- Total number of zeros
    //          KD  --- Function code
    //          KD=1 to 8 for ber x, bei x, ker x, kei x,
    //                    ber'x, bei'x, ker'x and kei'x,
    //                    respectively.
    // Output:  ZO(M) --- the M-th zero of Kelvin function
    //                    for code KD
    // Routine called:
    //          KLVNA for computing Kelvin functions and
    //          their derivatives
    // ====================================================

    // 初始化变量
    double ber, bei, ger, gei, der, dei, her, hei;
    // 预设不同函数代码对应的初始猜测值
    double rt0[9] = {0.0, 2.84891, 5.02622, 1.71854, 3.91467, 6.03871, 3.77268, 2.66584, 4.93181};
    double rt = rt0[kd];

    // 循环计算每一个零点的近似值
    for (int m = 1; m <= nt; m++) {
        while (1) {
            // 调用 detail 命名空间中的 klvna 函数，计算 Kelvin 函数及其导数
            detail::klvna(rt, &ber, &bei, &ger, &gei, &der, &dei, &her, &hei);
            
            // 根据函数代码 kd 选择不同的迭代方式更新 rt 的值
            if (kd == 1) {
                rt -= ber / der;
            } else if (kd == 2) {
                rt -= bei / dei;
            } else if (kd == 3) {
                rt -= ger / her;
            } else if (kd == 4) {
                rt -= gei / hei;
            } else if (kd == 5) {
                rt -= der / (-bei - der / rt);
            } else if (kd == 6) {
                rt -= dei / (ber - dei / rt);
            } else if (kd == 7) {
                rt -= her / (-gei - her / rt);
            } else {
                rt -= hei / (ger - hei / rt);
            }

            // 如果 rt 的变化足够小，则退出循环
            if (fabs(rt - rt0[kd]) <= 5e-10) {
                break;
            } else {
                rt0[kd] = rt;
            }
        }
        // 将计算得到的零点 rt 存入 zo 数组中
        zo[m - 1] = rt;
        // 更新 rt 的初始猜测值
        rt += 4.44;
    }
}
```