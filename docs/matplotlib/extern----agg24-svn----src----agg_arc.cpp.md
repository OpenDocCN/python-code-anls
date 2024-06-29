# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_arc.cpp`

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
// Arc vertex generator
//
//----------------------------------------------------------------------------

#include <math.h>
#include "agg_arc.h"

// 命名空间 agg 开始
namespace agg
{
    //------------------------------------------------------------------------
    // arc 类的构造函数
    arc::arc(double x,  double y, 
             double rx, double ry, 
             double a1, double a2, 
             bool ccw) :
        m_x(x), m_y(y), m_rx(rx), m_ry(ry), m_scale(1.0)
    {
        // 规范化角度，确保角度落在合适的范围内
        normalize(a1, a2, ccw);
    }

    //------------------------------------------------------------------------
    // 初始化 arc 对象的方法
    void arc::init(double x,  double y, 
                   double rx, double ry, 
                   double a1, double a2, 
                   bool ccw)
    {
        // 设置 arc 对象的各种参数
        m_x   = x;  m_y  = y;
        m_rx  = rx; m_ry = ry; 
        normalize(a1, a2, ccw); // 规范化角度
    }
    
    //------------------------------------------------------------------------
    // 设置近似比例的方法
    void arc::approximation_scale(double s)
    {
        m_scale = s; // 设置比例
        if(m_initialized)
        {
            normalize(m_start, m_end, m_ccw); // 如果已初始化，则重新规范化角度
        }
    }

    //------------------------------------------------------------------------
    // 重置路径迭代器的方法
    void arc::rewind(unsigned)
    {
        m_path_cmd = path_cmd_move_to; // 设置路径命令为移动到
        m_angle = m_start; // 设置角度为起始角度
    }

    //------------------------------------------------------------------------
    // 获取顶点坐标的方法
    unsigned arc::vertex(double* x, double* y)
    {
        if(is_stop(m_path_cmd)) return path_cmd_stop; // 如果是停止命令，则返回停止
        if((m_angle < m_end - m_da/4) != m_ccw) // 如果当前角度加上步进角度之后与结束角度差距大于四分之一步进角度且逆时针旋转
        {
            *x = m_x + cos(m_end) * m_rx; // 计算结束点的 x 坐标
            *y = m_y + sin(m_end) * m_ry; // 计算结束点的 y 坐标
            m_path_cmd = path_cmd_stop; // 设置路径命令为停止
            return path_cmd_line_to; // 返回直线到命令
        }

        *x = m_x + cos(m_angle) * m_rx; // 计算当前点的 x 坐标
        *y = m_y + sin(m_angle) * m_ry; // 计算当前点的 y 坐标

        m_angle += m_da; // 增加角度步进量

        unsigned pf = m_path_cmd; // 保存路径命令
        m_path_cmd = path_cmd_line_to; // 设置路径命令为直线到
        return pf; // 返回保存的路径命令
    }

    //------------------------------------------------------------------------
    // 规范化角度范围的方法
    void arc::normalize(double a1, double a2, bool ccw)
    {
        // 计算半径平均值
        double ra = (fabs(m_rx) + fabs(m_ry)) / 2;
        // 计算弧度角度增量
        m_da = acos(ra / (ra + 0.125 / m_scale)) * 2;
        
        // 如果是逆时针方向
        if(ccw)
        {
            // 确保起始角度小于结束角度，使角度保持正向增长
            while(a2 < a1) a2 += pi * 2.0;
        }
        else
        {
            // 如果是顺时针方向，确保起始角度大于结束角度
            while(a1 < a2) a1 += pi * 2.0;
            // 若顺时针方向，反转角度增量
            m_da = -m_da;
        }
        
        // 设置弧线方向（逆时针或顺时针）
        m_ccw   = ccw;
        // 设置起始角度
        m_start = a1;
        // 设置结束角度
        m_end   = a2;
        // 标记已初始化
        m_initialized = true;
    }
}


注释：

# 这行代码是一个单独的右花括号 '}'，用于结束一个代码块或语句。
# 在此处，它表示一个代码块的结束，可能是函数、循环、条件语句或类定义的结束位置。
# 在给定的代码片段中，它是作为整体结构的一部分，用于保证代码的正确性和结构完整性。
```