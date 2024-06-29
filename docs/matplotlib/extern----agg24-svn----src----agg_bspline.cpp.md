# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_bspline.cpp`

```
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
//
// class bspline
//
//----------------------------------------------------------------------------

#include "agg_bspline.h"

namespace agg
{
    //------------------------------------------------------------------------
    // 默认构造函数，初始化成员变量
    bspline::bspline() :
        m_max(0),            // 最大点数初始化为0
        m_num(0),            // 实际点数初始化为0
        m_x(0),              // X 坐标数组初始化为nullptr
        m_y(0),              // Y 坐标数组初始化为nullptr
        m_last_idx(-1)       // 上一个索引初始化为-1
    {
    }

    //------------------------------------------------------------------------
    // 构造函数，根据给定点数初始化
    bspline::bspline(int num) :
        m_max(0),            // 最大点数初始化为0
        m_num(0),            // 实际点数初始化为0
        m_x(0),              // X 坐标数组初始化为nullptr
        m_y(0),              // Y 坐标数组初始化为nullptr
        m_last_idx(-1)       // 上一个索引初始化为-1
    {
        init(num);           // 调用初始化函数初始化数据
    }

    //------------------------------------------------------------------------
    // 构造函数，根据给定点数和坐标数组初始化
    bspline::bspline(int num, const double* x, const double* y) :
        m_max(0),            // 最大点数初始化为0
        m_num(0),            // 实际点数初始化为0
        m_x(0),              // X 坐标数组初始化为nullptr
        m_y(0),              // Y 坐标数组初始化为nullptr
        m_last_idx(-1)       // 上一个索引初始化为-1
    {
        init(num, x, y);     // 调用初始化函数根据坐标数组初始化数据
    }

    
    //------------------------------------------------------------------------
    // 初始化函数，根据给定的最大点数重新分配内存
    void bspline::init(int max)
    {
        if(max > 2 && max > m_max)
        {
            m_am.resize(max * 3);   // 调整数组 m_am 的大小以容纳足够的数据
            m_max = max;            // 更新最大点数
            m_x   = &m_am[m_max];   // 设置 X 坐标数组的起始位置
            m_y   = &m_am[m_max * 2]; // 设置 Y 坐标数组的起始位置
        }
        m_num = 0;                  // 重置实际点数为0
        m_last_idx = -1;            // 重置上一个索引为-1
    }


    //------------------------------------------------------------------------
    // 添加一个点的函数，将给定的坐标加入数组
    void bspline::add_point(double x, double y)
    {
        if(m_num < m_max)   // 如果实际点数小于最大点数
        {
            m_x[m_num] = x; // 将 X 坐标写入数组
            m_y[m_num] = y; // 将 Y 坐标写入数组
            ++m_num;        // 实际点数加一
        }
    }


    //------------------------------------------------------------------------
    // 准备函数，暂无具体实现
    void bspline::prepare()
    {
        // 如果点的数量大于2，进行以下操作
        if(m_num > 2)
        {
            int i, k, n1;
            double* temp; // 临时数组指针
            double* r;    // r 数组指针
            double* s;    // s 数组指针
            double h, p, d, f, e; // 变量声明
    
            // 将 m_am 数组初始化为 0.0
            for(k = 0; k < m_num; k++) 
            {
                m_am[k] = 0.0;
            }
    
            n1 = 3 * m_num;
    
            // 创建长度为 n1 的 pod_array<double> 对象 al，并将其首地址赋给 temp
            pod_array<double> al(n1);
            temp = &al[0];
    
            // 将 temp 数组的所有元素初始化为 0.0
            for(k = 0; k < n1; k++) 
            {
                temp[k] = 0.0;
            }
    
            // 设置 r 和 s 的指针位置
            r = temp + m_num;
            s = temp + m_num * 2;
    
            n1 = m_num - 1;
            d = m_x[1] - m_x[0];
            e = (m_y[1] - m_y[0]) / d;
    
            // 计算 al、r 和 s 的值
            for(k = 1; k < n1; k++) 
            {
                h     = d;
                d     = m_x[k + 1] - m_x[k];
                f     = e;
                e     = (m_y[k + 1] - m_y[k]) / d;
                al[k] = d / (d + h);
                r[k]  = 1.0 - al[k];
                s[k]  = 6.0 * (e - f) / (h + d);
            }
    
            // 更新 s[k] 的值
            for(k = 1; k < n1; k++) 
            {
                p = 1.0 / (r[k] * al[k - 1] + 2.0);
                al[k] *= -p;
                s[k] = (s[k] - r[k] * s[k - 1]) * p; 
            }
    
            // 设置 m_am[n1] 和 al[n1-1] 的值
            m_am[n1]     = 0.0;
            al[n1 - 1]   = s[n1 - 1];
            m_am[n1 - 1] = al[n1 - 1];
    
            // 计算 al 和 m_am 的值
            for(k = n1 - 2, i = 0; i < m_num - 2; i++, k--) 
            {
                al[k]   = al[k] * al[k + 1] + s[k];
                m_am[k] = al[k];
            }
        }
        // 重置 m_last_idx 为 -1
        m_last_idx = -1;
    }
    
    
    
    //------------------------------------------------------------------------
    // 初始化 B-spline 曲线，给定点数和点的 x、y 坐标数组
    void bspline::init(int num, const double* x, const double* y)
    {
        // 如果点的数量大于2，调用重载的 init 方法，添加每个点并准备 B-spline
        if(num > 2)
        {
            init(num); // 调用重载的 init 方法
            int i;
            for(i = 0; i < num; i++)
            {
                add_point(*x++, *y++); // 添加每个点的 x 和 y 坐标
            }
            prepare(); // 准备 B-spline 曲线
        }
        // 重置 m_last_idx 为 -1
        m_last_idx = -1;
    }
    
    
    //------------------------------------------------------------------------
    // 二分查找，查找 x0 在 x 数组中的位置
    void bspline::bsearch(int n, const double *x, double x0, int *i) 
    {
        int j = n - 1;
        int k;
          
        // 二分查找 x0 在 x 数组中的位置，并更新 i 的值
        for(*i = 0; (j - *i) > 1; ) 
        {
            if(x0 < x[k = (*i + j) >> 1]) j = k; 
            else                         *i = k;
        }
    }
    
    
    //------------------------------------------------------------------------
    // 插值计算，返回 B-spline 曲线在 x 点的插值结果
    double bspline::interpolation(double x, int i) const
    {
        int j = i + 1;
        double d = m_x[i] - m_x[j];
        double h = x - m_x[j];
        double r = m_x[i] - x;
        double p = d * d / 6.0;
        // 返回 B-spline 插值结果
        return (m_am[j] * r * r * r + m_am[i] * h * h * h) / 6.0 / d +
               ((m_y[j] - m_am[j] * p) * r + (m_y[i] - m_am[i] * p) * h) / d;
    }
    
    
    //------------------------------------------------------------------------
    // 左侧外推，返回 B-spline 曲线在 x 点的外推结果
    double bspline::extrapolation_left(double x) const
    // 根据给定的两个点的坐标和斜率差计算 B 样条插值
    {
        double d = m_x[1] - m_x[0];  // 计算两个点的 x 坐标差值
        // 返回通过 B 样条插值计算得到的值
        return (-d * m_am[1] / 6 + (m_y[1] - m_y[0]) / d) * 
               (x - m_x[0]) + 
               m_y[0];
    }

    //------------------------------------------------------------------------
    // 右侧外推函数，用于 B 样条曲线
    double bspline::extrapolation_right(double x) const
    {
        double d = m_x[m_num - 1] - m_x[m_num - 2];  // 计算最后两个点的 x 坐标差值
        // 返回右侧外推值，通过最后两个点的坐标和斜率差计算得到
        return (d * m_am[m_num - 2] / 6 + (m_y[m_num - 1] - m_y[m_num - 2]) / d) * 
               (x - m_x[m_num - 1]) + 
               m_y[m_num - 1];
    }

    //------------------------------------------------------------------------
    // 根据输入的 x 值返回 B 样条曲线的插值或外推值
    double bspline::get(double x) const
    {
        if(m_num > 2)
        {
            int i;

            // 如果 x 小于第一个点的 x 坐标，进行左侧外推
            if(x < m_x[0]) return extrapolation_left(x);

            // 如果 x 大于等于最后一个点的 x 坐标，进行右侧外推
            if(x >= m_x[m_num - 1]) return extrapolation_right(x);

            // 否则进行插值
            bsearch(m_num, m_x, x, &i);  // 通过二分搜索找到插值点的索引
            return interpolation(x, i);  // 返回插值结果
        }
        return 0.0;
    }


    //------------------------------------------------------------------------
    // 基于状态的 B 样条插值函数，维护了最后一次使用的索引
    double bspline::get_stateful(double x) const
    {
        if(m_num > 2)
        {
            // 如果 x 小于第一个点的 x 坐标，进行左侧外推
            if(x < m_x[0]) return extrapolation_left(x);

            // 如果 x 大于等于最后一个点的 x 坐标，进行右侧外推
            if(x >= m_x[m_num - 1]) return extrapolation_right(x);

            if(m_last_idx >= 0)
            {
                // 检查 x 是否不在当前范围内
                if(x < m_x[m_last_idx] || x > m_x[m_last_idx + 1])
                {
                    // 检查 x 是否在下一个点之间（最可能的情况）
                    if(m_last_idx < m_num - 2 && 
                       x >= m_x[m_last_idx + 1] &&
                       x <= m_x[m_last_idx + 2])
                    {
                        ++m_last_idx;  // 更新最后使用的索引
                    }
                    else
                    if(m_last_idx > 0 && 
                       x >= m_x[m_last_idx - 1] && 
                       x <= m_x[m_last_idx])
                    {
                        --m_last_idx;  // 更新最后使用的索引
                    }
                    else
                    {
                        // 否则进行完整搜索
                        bsearch(m_num, m_x, x, &m_last_idx);  // 使用二分搜索找到插值点的索引
                    }
                }
                return interpolation(x, m_last_idx);  // 返回插值结果
            }
            else
            {
                // 否则进行完整搜索并返回插值结果
                bsearch(m_num, m_x, x, &m_last_idx);  // 使用二分搜索找到插值点的索引
                return interpolation(x, m_last_idx);  // 返回插值结果
            }
        }
        return 0.0;
    }
}



# 这是一个代码块的结束标记，表示当前的代码块结束。
```