# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_bezier_arc.cpp`

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
// Arc generator. Produces at most 4 consecutive cubic bezier curves, i.e., 
// 4, 7, 10, or 13 vertices.
//
//----------------------------------------------------------------------------


#include <math.h>
#include "agg_bezier_arc.h"


namespace agg
{

    // This epsilon is used to prevent us from adding degenerate curves 
    // (converging to a single point).
    // The value isn't very critical. Function arc_to_bezier() has a limit 
    // of the sweep_angle. If fabs(sweep_angle) exceeds pi/2 the curve 
    // becomes inaccurate. But slight exceeding is quite appropriate.
    //-------------------------------------------------bezier_arc_angle_epsilon
    const double bezier_arc_angle_epsilon = 0.01;

    //------------------------------------------------------------arc_to_bezier
    // Converts an elliptical arc to a sequence of cubic bezier curves
    void arc_to_bezier(double cx, double cy, double rx, double ry, 
                       double start_angle, double sweep_angle,
                       double* curve)
    {
        // Calculate control points of the bezier curves
        double x0 = cos(sweep_angle / 2.0);
        double y0 = sin(sweep_angle / 2.0);
        double tx = (1.0 - x0) * 4.0 / 3.0;
        double ty = y0 - tx * x0 / y0;
        double px[4];
        double py[4];
        px[0] =  x0;
        py[0] = -y0;
        px[1] =  x0 + tx;
        py[1] = -ty;
        px[2] =  x0 + tx;
        py[2] =  ty;
        px[3] =  x0;
        py[3] =  y0;

        // Calculate rotation matrix components
        double sn = sin(start_angle + sweep_angle / 2.0);
        double cs = cos(start_angle + sweep_angle / 2.0);

        // Compute bezier curve points using the rotation matrix
        unsigned i;
        for(i = 0; i < 4; i++)
        {
            curve[i * 2]     = cx + rx * (px[i] * cs - py[i] * sn);
            curve[i * 2 + 1] = cy + ry * (px[i] * sn + py[i] * cs);
        }
    }



    //------------------------------------------------------------------------
    // Initialize the bezier_arc object with arc parameters
    void bezier_arc::init(double x,  double y, 
                          double rx, double ry, 
                          double start_angle, 
                          double sweep_angle)
    {
        // 将起始角度调整为 [0, 2π) 范围内的值
        start_angle = fmod(start_angle, 2.0 * pi);
        // 如果扫描角度大于等于 2π，则将其设置为 2π
        if(sweep_angle >=  2.0 * pi) sweep_angle =  2.0 * pi;
        // 如果扫描角度小于等于 -2π，则将其设置为 -2π
        if(sweep_angle <= -2.0 * pi) sweep_angle = -2.0 * pi;
    
        // 如果扫描角度的绝对值小于 1e-10
        if(fabs(sweep_angle) < 1e-10)
        {
            // 设置顶点数为 4
            m_num_vertices = 4;
            // 设置路径命令为直线到
            m_cmd = path_cmd_line_to;
            // 计算起始点和终点的顶点坐标
            m_vertices[0] = x + rx * cos(start_angle);
            m_vertices[1] = y + ry * sin(start_angle);
            m_vertices[2] = x + rx * cos(start_angle + sweep_angle);
            m_vertices[3] = y + ry * sin(start_angle + sweep_angle);
            return;
        }
    
        // 初始化总扫描角度和局部扫描角度
        double total_sweep = 0.0;
        double local_sweep = 0.0;
        double prev_sweep;
        // 设置初始顶点数为 2
        m_num_vertices = 2;
        // 设置路径命令为四次贝塞尔曲线
        m_cmd = path_cmd_curve4;
        bool done = false;
        // 进入循环，直到条件满足或顶点数达到 26
        do
        {
            // 如果扫描角度小于 0
            if(sweep_angle < 0.0)
            {
                prev_sweep  = total_sweep;
                local_sweep = -pi * 0.5;
                total_sweep -= pi * 0.5;
                // 如果总扫描角度小于等于目标扫描角度加上误差范围
                if(total_sweep <= sweep_angle + bezier_arc_angle_epsilon)
                {
                    // 计算实际的局部扫描角度并标记循环结束
                    local_sweep = sweep_angle - prev_sweep;
                    done = true;
                }
            }
            else
            {
                prev_sweep  = total_sweep;
                local_sweep =  pi * 0.5;
                total_sweep += pi * 0.5;
                // 如果总扫描角度大于等于目标扫描角度减去误差范围
                if(total_sweep >= sweep_angle - bezier_arc_angle_epsilon)
                {
                    // 计算实际的局部扫描角度并标记循环结束
                    local_sweep = sweep_angle - prev_sweep;
                    done = true;
                }
            }
    
            // 调用函数将圆弧转换为贝塞尔曲线段，并更新顶点数
            arc_to_bezier(x, y, rx, ry, 
                          start_angle, 
                          local_sweep, 
                          m_vertices + m_num_vertices - 2);
    
            // 更新顶点数和起始角度
            m_num_vertices += 6;
            start_angle += local_sweep;
        }
        while(!done && m_num_vertices < 26);
    }
    
    
    
    
    //--------------------------------------------------------------------
    // 初始化函数，用于设置圆弧参数
    void bezier_arc_svg::init(double x0, double y0, 
                              double rx, double ry, 
                              double angle,
                              bool large_arc_flag,
                              bool sweep_flag,
                              double x2, double y2)
    }
}



# 这行代码表示一个单独的右大括号 '}'，用于闭合某个代码块或数据结构。
```