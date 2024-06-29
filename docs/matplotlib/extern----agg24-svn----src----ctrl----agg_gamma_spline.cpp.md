# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\ctrl\agg_gamma_spline.cpp`

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
//
// class gamma_spline
//
//----------------------------------------------------------------------------

#include "ctrl/agg_gamma_spline.h"

namespace agg
{

    //------------------------------------------------------------------------
    // gamma_spline 构造函数，默认初始化控制点和当前 x 值
    gamma_spline::gamma_spline() : 
        m_x1(0), m_y1(0), m_x2(10), m_y2(10), m_cur_x(0.0)
    {
        // 调用 values 函数初始化默认值
        values(1.0, 1.0, 1.0, 1.0);
    }


    //------------------------------------------------------------------------
    // 根据给定的 x 值返回对应的 y 值，确保 x 和 y 在 [0, 1] 范围内
    double gamma_spline::y(double x) const 
    { 
        if(x < 0.0) x = 0.0;    // 如果 x 小于 0，则设为 0
        if(x > 1.0) x = 1.0;    // 如果 x 大于 1，则设为 1
        double val = m_spline.get(x);   // 获取 Spline 曲线在 x 处的值
        if(val < 0.0) val = 0.0; // 确保 val 在 [0, 1] 范围内
        if(val > 1.0) val = 1.0;
        return val; // 返回计算得到的 y 值
    }



    //------------------------------------------------------------------------
    // 设置控制点的值，确保在指定范围内，更新内部控制点和 Spline 曲线
    void gamma_spline::values(double kx1, double ky1, double kx2, double ky2)
    {
        if(kx1 < 0.001) kx1 = 0.001;   // 如果 kx1 小于 0.001，则设为 0.001
        if(kx1 > 1.999) kx1 = 1.999;   // 如果 kx1 大于 1.999，则设为 1.999
        if(ky1 < 0.001) ky1 = 0.001;   // 如果 ky1 小于 0.001，则设为 0.001
        if(ky1 > 1.999) ky1 = 1.999;   // 如果 ky1 大于 1.999，则设为 1.999
        if(kx2 < 0.001) kx2 = 0.001;   // 如果 kx2 小于 0.001，则设为 0.001
        if(kx2 > 1.999) kx2 = 1.999;   // 如果 kx2 大于 1.999，则设为 1.999
        if(ky2 < 0.001) ky2 = 0.001;   // 如果 ky2 小于 0.001，则设为 0.001
        if(ky2 > 1.999) ky2 = 1.999;   // 如果 ky2 大于 1.999，则设为 1.999

        // 设置控制点的位置
        m_x[0] = 0.0;
        m_y[0] = 0.0;
        m_x[1] = kx1 * 0.25;
        m_y[1] = ky1 * 0.25;
        m_x[2] = 1.0 - kx2 * 0.25;
        m_y[2] = 1.0 - ky2 * 0.25;
        m_x[3] = 1.0;
        m_y[3] = 1.0;

        // 初始化 Spline 曲线
        m_spline.init(4, m_x, m_y);

        // 计算 gamma 校正表
        int i;
        for(i = 0; i < 256; i++)
        {
            m_gamma[i] = (unsigned char)(y(double(i) / 255.0) * 255.0);
        }
    }


    //------------------------------------------------------------------------
    // 获取当前设置的控制点的值
    void gamma_spline::values(double* kx1, double* ky1, double* kx2, double* ky2) const
    {
        *kx1 = m_x[1] * 4.0;    // 返回 kx1 的值
        *ky1 = m_y[1] * 4.0;    // 返回 ky1 的值
        *kx2 = (1.0 - m_x[2]) * 4.0;    // 返回 kx2 的值
        *ky2 = (1.0 - m_y[2]) * 4.0;    // 返回 ky2 的值
    }


    //------------------------------------------------------------------------
    // 设置盒子的坐标范围
    void gamma_spline::box(double x1, double y1, double x2, double y2)
    {
        m_x1 = x1;  // 设置盒子左上角 x 坐标
        m_y1 = y1;  // 设置盒子左上角 y 坐标
        m_x2 = x2;  // 设置盒子右下角 x 坐标
        m_y2 = y2;  // 设置盒子右下角 y 坐标
    }


    //------------------------------------------------------------------------
    // 重置当前 x 值
    void gamma_spline::rewind(unsigned)
    {
        m_cur_x = 0.0;  // 将当前 x 值重置为 0.0
    //------------------------------------------------------------------------
    unsigned gamma_spline::vertex(double* vx, double* vy)
    {
        // 如果当前点 m_cur_x 等于 0.0，则设定起点坐标并返回移动到指令
        if(m_cur_x == 0.0) 
        {
            *vx = m_x1;
            *vy = m_y1;
            // 增加当前点 m_cur_x 的步长，以便下一次调用时能够移动到下一个位置
            m_cur_x += 1.0 / (m_x2 - m_x1);
            return path_cmd_move_to;
        }
    
        // 如果当前点 m_cur_x 大于 1.0，则返回停止路径指令
        if(m_cur_x > 1.0) 
        {
            return path_cmd_stop;
        }
        
        // 计算当前点在曲线上的位置，并返回直线到指令
        *vx = m_x1 + m_cur_x * (m_x2 - m_x1);
        *vy = m_y1 + y(m_cur_x) * (m_y2 - m_y1);
    
        // 增加当前点 m_cur_x 的步长，以便下一次调用时能够移动到下一个位置
        m_cur_x += 1.0 / (m_x2 - m_x1);
        return path_cmd_line_to;
    }
    //------------------------------------------------------------------------
}



# 这行代码关闭了一个代码块，用于结束某个函数、循环或条件语句的定义。
```