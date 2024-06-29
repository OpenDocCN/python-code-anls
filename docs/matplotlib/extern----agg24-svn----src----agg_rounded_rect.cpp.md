# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_rounded_rect.cpp`

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
// Rounded rectangle vertex generator
//
//----------------------------------------------------------------------------

#include <math.h>
#include "agg_rounded_rect.h"

// 命名空间agg，包含了所有与rounded_rect类相关的内容
namespace agg
{
    //------------------------------------------------------------------------
    // 构造函数，初始化圆角矩形的参数
    rounded_rect::rounded_rect(double x1, double y1, double x2, double y2, double r) :
        m_x1(x1), m_y1(y1), m_x2(x2), m_y2(y2),
        m_rx1(r), m_ry1(r), m_rx2(r), m_ry2(r), 
        m_rx3(r), m_ry3(r), m_rx4(r), m_ry4(r)
    {
        // 确保矩形的左上角在左下角坐标之前
        if(x1 > x2) { m_x1 = x2; m_x2 = x1; }
        if(y1 > y2) { m_y1 = y2; m_y2 = y1; }
    }

    //--------------------------------------------------------------------
    // 设置矩形的位置
    void rounded_rect::rect(double x1, double y1, double x2, double y2)
    {
        m_x1 = x1;
        m_y1 = y1;
        m_x2 = x2;
        m_y2 = y2;
        // 确保矩形的左上角在左下角坐标之前
        if(x1 > x2) { m_x1 = x2; m_x2 = x1; }
        if(y1 > y2) { m_y1 = y2; m_y2 = y1; }
    }

    //--------------------------------------------------------------------
    // 设置所有四个角的相同半径
    void rounded_rect::radius(double r)
    {
        m_rx1 = m_ry1 = m_rx2 = m_ry2 = m_rx3 = m_ry3 = m_rx4 = m_ry4 = r; 
    }

    //--------------------------------------------------------------------
    // 设置水平和垂直方向不同的半径
    void rounded_rect::radius(double rx, double ry)
    {
        m_rx1 = m_rx2 = m_rx3 = m_rx4 = rx; 
        m_ry1 = m_ry2 = m_ry3 = m_ry4 = ry; 
    }

    //--------------------------------------------------------------------
    // 设置每个角的不同半径，分别针对底部和顶部
    void rounded_rect::radius(double rx_bottom, double ry_bottom, 
                              double rx_top,    double ry_top)
    {
        m_rx1 = m_rx2 = rx_bottom; 
        m_rx3 = m_rx4 = rx_top; 
        m_ry1 = m_ry2 = ry_bottom; 
        m_ry3 = m_ry4 = ry_top; 
    }

    //--------------------------------------------------------------------
    // 设置每个角的独立半径
    void rounded_rect::radius(double rx1, double ry1, double rx2, double ry2, 
                              double rx3, double ry3, double rx4, double ry4)
    {
        m_rx1 = rx1; m_ry1 = ry1; m_rx2 = rx2; m_ry2 = ry2; 
        m_rx3 = rx3; m_ry3 = ry3; m_rx4 = rx4; m_ry4 = ry4;
    }

    //--------------------------------------------------------------------
    // 规范化半径，确保它们不小于零
    void rounded_rect::normalize_radius()
    {
        // 计算垂直和水平方向的差值
        double dx = fabs(m_y2 - m_y1);
        double dy = fabs(m_x2 - m_x1);
    
        // 初始化比例因子为最大值
        double k = 1.0;
        double t;
        
        // 计算水平方向的比例因子并更新最小值
        t = dx / (m_rx1 + m_rx2); if(t < k) k = t; 
        t = dx / (m_rx3 + m_rx4); if(t < k) k = t; 
        
        // 计算垂直方向的比例因子并更新最小值
        t = dy / (m_ry1 + m_ry2); if(t < k) k = t; 
        t = dy / (m_ry3 + m_ry4); if(t < k) k = t; 
    
        // 如果最小比例因子小于1，则进行缩放操作
        if(k < 1.0)
        {
            m_rx1 *= k; m_ry1 *= k; m_rx2 *= k; m_ry2 *= k;
            m_rx3 *= k; m_ry3 *= k; m_rx4 *= k; m_ry4 *= k;
        }
    }
    
    //--------------------------------------------------------------------
    void rounded_rect::rewind(unsigned)
    {
        // 重置状态为0
        m_status = 0;
    }
    
    //--------------------------------------------------------------------
    unsigned rounded_rect::vertex(double* x, double* y)
    {
        unsigned cmd = path_cmd_stop;
        switch(m_status)
        {
        case 0:
            // 初始化第一个圆弧的起点和终点角度
            m_arc.init(m_x1 + m_rx1, m_y1 + m_ry1, m_rx1, m_ry1,
                       pi, pi+pi*0.5);
            m_arc.rewind(0);
            m_status++;
    
        case 1:
            // 获取圆弧上的顶点，直到到达终点
            cmd = m_arc.vertex(x, y);
            if(is_stop(cmd)) m_status++;
            else return cmd;
    
        case 2:
            // 初始化第二个圆弧的起点和终点角度
            m_arc.init(m_x2 - m_rx2, m_y1 + m_ry2, m_rx2, m_ry2,
                       pi+pi*0.5, 0.0);
            m_arc.rewind(0);
            m_status++;
    
        case 3:
            // 获取圆弧上的顶点，直到到达终点
            cmd = m_arc.vertex(x, y);
            if(is_stop(cmd)) m_status++;
            else return path_cmd_line_to;
    
        case 4:
            // 初始化第三个圆弧的起点和终点角度
            m_arc.init(m_x2 - m_rx3, m_y2 - m_ry3, m_rx3, m_ry3,
                       0.0, pi*0.5);
            m_arc.rewind(0);
            m_status++;
    
        case 5:
            // 获取圆弧上的顶点，直到到达终点
            cmd = m_arc.vertex(x, y);
            if(is_stop(cmd)) m_status++;
            else return path_cmd_line_to;
    
        case 6:
            // 初始化第四个圆弧的起点和终点角度
            m_arc.init(m_x1 + m_rx4, m_y2 - m_ry4, m_rx4, m_ry4,
                       pi*0.5, pi);
            m_arc.rewind(0);
            m_status++;
    
        case 7:
            // 获取圆弧上的顶点，直到到达终点
            cmd = m_arc.vertex(x, y);
            if(is_stop(cmd)) m_status++;
            else return path_cmd_line_to;
    
        case 8:
            // 返回多边形结束命令
            cmd = path_cmd_end_poly | path_flags_close | path_flags_ccw;
            m_status++;
            break;
        }
        return cmd;
    }
}
```