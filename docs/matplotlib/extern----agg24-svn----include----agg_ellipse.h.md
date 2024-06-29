# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_ellipse.h`

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
// class ellipse
//
//----------------------------------------------------------------------------

#ifndef AGG_ELLIPSE_INCLUDED
#define AGG_ELLIPSE_INCLUDED

#include "agg_basics.h"
#include <math.h>

namespace agg
{

    //----------------------------------------------------------------ellipse
    // 椭圆类，用于绘制椭圆形状
    class ellipse
    {
    public:
        // 默认构造函数，初始化椭圆参数
        ellipse() : 
            m_x(0.0), m_y(0.0), m_rx(1.0), m_ry(1.0), m_scale(1.0), 
            m_num(4), m_step(0), m_cw(false) {}

        // 带参数的构造函数，初始化椭圆参数，并计算步数
        ellipse(double x, double y, double rx, double ry, 
                unsigned num_steps=0, bool cw=false) :
            m_x(x), m_y(y), m_rx(rx), m_ry(ry), m_scale(1.0), 
            m_num(num_steps), m_step(0), m_cw(cw) 
        {
            if(m_num == 0) calc_num_steps();
        }

        // 初始化函数，设置椭圆参数，并计算步数
        void init(double x, double y, double rx, double ry, 
                  unsigned num_steps=0, bool cw=false);

        // 设置近似比例尺度
        void approximation_scale(double scale);
        
        // 回到路径的起点
        void rewind(unsigned path_id);
        
        // 获取下一个顶点的坐标
        unsigned vertex(double* x, double* y);

    private:
        // 计算椭圆所需的步数
        void calc_num_steps();

        double m_x;         // 椭圆中心的 x 坐标
        double m_y;         // 椭圆中心的 y 坐标
        double m_rx;        // 椭圆的半长轴
        double m_ry;        // 椭圆的半短轴
        double m_scale;     // 近似比例尺度
        unsigned m_num;     // 步数
        unsigned m_step;    // 当前步数
        bool m_cw;          // 是否顺时针方向
    };

    //------------------------------------------------------------------------
    // 设置椭圆的参数，并计算步数
    inline void ellipse::init(double x, double y, double rx, double ry, 
                              unsigned num_steps, bool cw)
    {
        m_x = x;
        m_y = y;
        m_rx = rx;
        m_ry = ry;
        m_num = num_steps;
        m_step = 0;
        m_cw = cw;
        if(m_num == 0) calc_num_steps();
    }

    //------------------------------------------------------------------------
    // 设置椭圆的近似比例尺度
    inline void ellipse::approximation_scale(double scale)
    {   
        m_scale = scale;
        calc_num_steps();
    }

    //------------------------------------------------------------------------
    // 计算椭圆所需的步数
    inline void ellipse::calc_num_steps()
    {
        double ra = (fabs(m_rx) + fabs(m_ry)) / 2;
        double da = acos(ra / (ra + 0.125 / m_scale)) * 2;
        m_num = uround(2*pi / da);
    }

    //------------------------------------------------------------------------
    // 回到路径的起点
    inline void ellipse::rewind(unsigned)
    {
        m_step = 0;
    }
    
    
    
    //------------------------------------------------------------------------
    inline unsigned ellipse::vertex(double* x, double* y)
    {
        // 如果步骤数达到了指定的顶点数目，表示已经结束多边形的绘制
        if(m_step == m_num) 
        {
            ++m_step;
            // 返回多边形结束命令，并指定闭合路径和逆时针方向标志
            return path_cmd_end_poly | path_flags_close | path_flags_ccw;
        }
        // 如果步骤数超过了顶点数目，表示已经结束路径的绘制
        if(m_step > m_num) 
            return path_cmd_stop;
        
        // 计算当前步骤对应的角度
        double angle = double(m_step) / double(m_num) * 2.0 * pi;
        // 如果是顺时针方向，调整角度
        if(m_cw) angle = 2.0 * pi - angle;
        
        // 计算椭圆上指定角度处的顶点坐标
        *x = m_x + cos(angle) * m_rx;
        *y = m_y + sin(angle) * m_ry;
        
        // 增加步骤数
        m_step++;
        
        // 如果是第一个步骤，返回移动到指定点的命令，否则返回直线到指定点的命令
        return ((m_step == 1) ? path_cmd_move_to : path_cmd_line_to);
    }
}


注释：


// 这行代码结束了一个函数的定义或者一个代码块的结束。
// 在 C/C++ 中，大括号 {} 用于定义函数、循环、条件语句等代码块的开始和结束。
// 这里的 } 表示一个代码块的结束。
#endif



#endif


注释：


// #endif 是条件编译预处理指令的结束标志。
// 在 C/C++ 中，#ifdef 或 #ifndef 用于条件编译，检查一个标识符是否被定义。
// #endif 表示条件编译块的结束，用于结束 #ifdef、#ifndef 或 #if 的条件判断部分。
// 这里的 #endif 结束了一个条件编译块。
```