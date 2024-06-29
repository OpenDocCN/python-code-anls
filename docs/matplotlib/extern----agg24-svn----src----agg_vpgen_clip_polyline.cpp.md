# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_vpgen_clip_polyline.cpp`

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

#include "agg_vpgen_clip_polyline.h"
#include "agg_clip_liang_barsky.h"

namespace agg
{
    //----------------------------------------------------------------------------
    // 重置生成的折线段的状态，将顶点数量、当前顶点和移动状态重置为初始值
    void vpgen_clip_polyline::reset()
    {
        m_vertex = 0;
        m_num_vertices = 0;
        m_move_to = false;
    }

    //----------------------------------------------------------------------------
    // 设置折线段的起点，同时重置顶点数量和当前顶点，记录起点的坐标
    void vpgen_clip_polyline::move_to(double x, double y)
    {
        m_vertex = 0;
        m_num_vertices = 0;
        m_x1 = x;
        m_y1 = y;
        m_move_to = true;
    }

    //----------------------------------------------------------------------------
    // 绘制从当前点到指定点的直线段，根据裁剪框裁剪线段，更新顶点坐标和命令
    void vpgen_clip_polyline::line_to(double x, double y)
    {
        double x2 = x;
        double y2 = y;
        // 对线段进行裁剪，返回裁剪后的状态标志
        unsigned flags = clip_line_segment(&m_x1, &m_y1, &x2, &y2, m_clip_box);

        m_vertex = 0;
        m_num_vertices = 0;
        // 如果线段未完全被裁剪掉
        if ((flags & 4) == 0)
        {
            // 如果线段是一个新的起始点或者是一个单独的移动命令
            if ((flags & 1) != 0 || m_move_to)
            {
                m_x[0] = m_x1;
                m_y[0] = m_y1;
                m_cmd[0] = path_cmd_move_to;
                m_num_vertices = 1;
            }
            // 记录线段的终点坐标和命令为直线段
            m_x[m_num_vertices] = x2;
            m_y[m_num_vertices] = y2;
            m_cmd[m_num_vertices++] = path_cmd_line_to;
            // 更新移动状态
            m_move_to = (flags & 2) != 0;
        }
        // 更新当前点坐标
        m_x1 = x;
        m_y1 = y;
    }

    //----------------------------------------------------------------------------
    // 获取生成折线段的下一个顶点坐标，并返回该顶点的绘制命令
    unsigned vpgen_clip_polyline::vertex(double* x, double* y)
    {
        // 如果还有未返回的顶点
        if (m_vertex < m_num_vertices)
        {
            *x = m_x[m_vertex];
            *y = m_y[m_vertex];
            // 返回顶点的绘制命令并将顶点索引增加
            return m_cmd[m_vertex++];
        }
        // 如果所有顶点都已经返回，则返回绘制命令停止
        return path_cmd_stop;
    }
}
```