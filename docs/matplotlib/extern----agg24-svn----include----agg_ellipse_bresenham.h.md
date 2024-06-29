# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_ellipse_bresenham.h`

```py
//--------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//
// Permission to copy, use, modify, sell and distribute this software 
// is granted provided this copyright notice appears in all copies. 
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//--------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//--------------------------------------------------------------------------
//
// Simple Bresenham interpolator for ellipses
//
//--------------------------------------------------------------------------

#ifndef AGG_ELLIPSE_BRESENHAM_INCLUDED
#define AGG_ELLIPSE_BRESENHAM_INCLUDED


#include "agg_basics.h"


namespace agg
{

    //------------------------------------------ellipse_bresenham_interpolator
    class ellipse_bresenham_interpolator
    {
    public:
        // 构造函数，初始化椭圆的半长轴和半短轴平方、增量等参数
        ellipse_bresenham_interpolator(int rx, int ry) :
            m_rx2(rx * rx),         // 半长轴平方
            m_ry2(ry * ry),         // 半短轴平方
            m_two_rx2(m_rx2 << 1),  // 半长轴平方的两倍
            m_two_ry2(m_ry2 << 1),  // 半短轴平方的两倍
            m_dx(0),                // x 方向的增量
            m_dy(0),                // y 方向的增量
            m_inc_x(0),             // x 方向的增量增量
            m_inc_y(-ry * m_two_rx2), // y 方向的增量增量，初始为 -ry * 2 * rx^2
            m_cur_f(0)              // 决策参数初始值
        {}
        
        // 获取 x 方向的增量
        int dx() const { return m_dx; }
        
        // 获取 y 方向的增量
        int dy() const { return m_dy; }

        // 迭代器操作符重载，用于计算下一个点的增量
        void operator++ ()
        {
            int  mx, my, mxy, min_m;
            int  fx, fy, fxy;

            // 计算四个决策参数
            mx = fx = m_cur_f + m_inc_x + m_ry2;
            if(mx < 0) mx = -mx;

            my = fy = m_cur_f + m_inc_y + m_rx2;
            if(my < 0) my = -my;

            mxy = fxy = m_cur_f + m_inc_x + m_ry2 + m_inc_y + m_rx2;
            if(mxy < 0) mxy = -mxy;

            // 确定最小的决策参数及对应的方向
            min_m = mx; 
            bool flag = true;

            if(min_m > my)  
            { 
                min_m = my; 
                flag = false; 
            }

            // 根据最小决策参数确定下一个点的位置
            m_dx = m_dy = 0;

            if(min_m > mxy) 
            { 
                m_inc_x += m_two_ry2;
                m_inc_y += m_two_rx2;
                m_cur_f = fxy;
                m_dx = 1; 
                m_dy = 1;
                return;
            }

            if(flag) 
            {
                m_inc_x += m_two_ry2;
                m_cur_f = fx;
                m_dx = 1;
                return;
            }

            m_inc_y += m_two_rx2;
            m_cur_f = fy;
            m_dy = 1;
        }

    private:
        int m_rx2;          // 半长轴平方
        int m_ry2;          // 半短轴平方
        int m_two_rx2;      // 半长轴平方的两倍
        int m_two_ry2;      // 半短轴平方的两倍
        int m_dx;           // x 方向的增量
        int m_dy;           // y 方向的增量
        int m_inc_x;        // x 方向的增量增量
        int m_inc_y;        // y 方向的增量增量
        int m_cur_f;        // 决策参数

    };

}

#endif
```