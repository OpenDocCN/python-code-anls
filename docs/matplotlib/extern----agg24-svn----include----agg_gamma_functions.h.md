# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_gamma_functions.h`

```py
//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//
// Permission to copy, use, modify, sell and distribute this software 
// is granted provided this copyright notice appears in all copies. 
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------

#ifndef AGG_GAMMA_FUNCTIONS_INCLUDED
#define AGG_GAMMA_FUNCTIONS_INCLUDED

#include <math.h>
#include "agg_basics.h"

namespace agg
{
    //===============================================================gamma_none
    // 结构体 gamma_none，用于不对输入进行任何调整，直接返回输入值
    struct gamma_none
    {
        double operator()(double x) const { return x; }
    };


    //==============================================================gamma_power
    // 类 gamma_power，实现幂函数调整输入值
    class gamma_power
    {
    public:
        // 默认构造函数，设置 gamma 为 1.0
        gamma_power() : m_gamma(1.0) {}
        // 带参数的构造函数，设置 gamma 为给定值 g
        gamma_power(double g) : m_gamma(g) {}

        // 设置 gamma 的值
        void gamma(double g) { m_gamma = g; }
        // 获取当前 gamma 的值
        double gamma() const { return m_gamma; }

        // 实现函数调用操作符，返回 x 的 m_gamma 次幂
        double operator() (double x) const
        {
            return pow(x, m_gamma);
        }

    private:
        double m_gamma; // 存储 gamma 值的成员变量
    };


    //==========================================================gamma_threshold
    // 类 gamma_threshold，实现阈值函数调整输入值
    class gamma_threshold
    {
    public:
        // 默认构造函数，设置阈值为 0.5
        gamma_threshold() : m_threshold(0.5) {}
        // 带参数的构造函数，设置阈值为给定值 t
        gamma_threshold(double t) : m_threshold(t) {}

        // 设置阈值的值
        void threshold(double t) { m_threshold = t; }
        // 获取当前阈值的值
        double threshold() const { return m_threshold; }

        // 实现函数调用操作符，根据输入值 x 返回 0.0 或 1.0
        double operator() (double x) const
        {
            return (x < m_threshold) ? 0.0 : 1.0;
        }

    private:
        double m_threshold; // 存储阈值的成员变量
    };


    //============================================================gamma_linear
    // 类 gamma_linear，实现线性变换函数调整输入值
    class gamma_linear
    {
    public:
        // 默认构造函数，设置起始值为 0.0，结束值为 1.0
        gamma_linear() : m_start(0.0), m_end(1.0) {}
        // 带参数的构造函数，设置起始值为 s，结束值为 e
        gamma_linear(double s, double e) : m_start(s), m_end(e) {}

        // 设置起始和结束值
        void set(double s, double e) { m_start = s; m_end = e; }
        // 设置起始值
        void start(double s) { m_start = s; }
        // 设置结束值
        void end(double e) { m_end = e; }
        // 获取当前起始值
        double start() const { return m_start; }
        // 获取当前结束值
        double end() const { return m_end; }

        // 实现函数调用操作符，根据输入值 x 返回相应的线性变换结果
        double operator() (double x) const
        {
            if(x < m_start) return 0.0;
            if(x > m_end) return 1.0;
            return (x - m_start) / (m_end - m_start);
        }

    private:
        double m_start; // 存储起始值的成员变量
        double m_end;   // 存储结束值的成员变量
    };


    //==========================================================gamma_multiply
    // 类 gamma_multiply，实现乘法变换函数调整输入值
    class gamma_multiply
    {
    // 声明公共部分类 gamma_multiply
    public:
        // 默认构造函数，将乘数 m_mul 初始化为 1.0
        gamma_multiply() : m_mul(1.0) {}
        // 带参数的构造函数，根据给定的参数 v 初始化乘数 m_mul
        gamma_multiply(double v) : m_mul(v) {}

        // 设置乘数 m_mul 的值
        void value(double v) { m_mul = v; }
        // 返回当前乘数 m_mul 的值
        double value() const { return m_mul; }

        // 重载 () 运算符，对输入的 x 进行乘法操作，并将结果截断为最大值为 1.0 的范围内
        double operator() (double x) const
        {
            // 计算乘法结果 y
            double y = x * m_mul;
            // 如果 y 大于 1.0，则将 y 设为 1.0
            if(y > 1.0) y = 1.0;
            // 返回乘法结果 y
            return y;
        }

    // 声明私有成员变量 m_mul，用于存储乘数值
    private:
        double m_mul;
    };

    // 内联函数，将 sRGB 颜色空间值 x 转换为线性颜色空间值
    inline double sRGB_to_linear(double x)
    {
        // 根据 sRGB 转换公式判断 x 的大小，选择不同的转换方式
        return (x <= 0.04045) ? (x / 12.92) : pow((x + 0.055) / (1.055), 2.4);
    }

    // 内联函数，将线性颜色空间值 x 转换为 sRGB 颜色空间值
    inline double linear_to_sRGB(double x)
    {
        // 根据线性到 sRGB 的转换公式判断 x 的大小，选择不同的转换方式
        return (x <= 0.0031308) ? (x * 12.92) : (1.055 * pow(x, 1 / 2.4) - 0.055);
    }
}


注释：


// 结束条件：表示这是条件编译指令的结束



#endif


注释：


// 结束条件编译指令，用于结束一个条件编译块
```