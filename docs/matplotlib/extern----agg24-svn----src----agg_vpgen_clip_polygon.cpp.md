# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_vpgen_clip_polygon.cpp`

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

#include "agg_vpgen_clip_polygon.h"
#include "agg_clip_liang_barsky.h"

namespace agg
{

    //------------------------------------------------------------------------
    // Determine the clipping code of the vertex according to the 
    // Cyrus-Beck line clipping algorithm
    //
    //        |        |
    //  0110  |  0010  | 0011
    //        |        |
    // -------+--------+-------- clip_box.y2
    //        |        |
    //  0100  |  0000  | 0001
    //        |        |
    // -------+--------+-------- clip_box.y1
    //        |        |
    //  1100  |  1000  | 1001
    //        |        |
    //  clip_box.x1  clip_box.x2
    //
    // 
    // 根据 Cyrus-Beck 线段裁剪算法确定顶点的裁剪代码
    unsigned vpgen_clip_polygon::clipping_flags(double x, double y)
    {
        if(x < m_clip_box.x1) 
        {
            if(y > m_clip_box.y2) return 6;  // 左侧超出裁剪框，顶点在裁剪框外上方
            if(y < m_clip_box.y1) return 12; // 左侧超出裁剪框，顶点在裁剪框外下方
            return 4;                        // 左侧超出裁剪框，顶点在裁剪框内
        }

        if(x > m_clip_box.x2) 
        {
            if(y > m_clip_box.y2) return 3;  // 右侧超出裁剪框，顶点在裁剪框外上方
            if(y < m_clip_box.y1) return 9;  // 右侧超出裁剪框，顶点在裁剪框外下方
            return 1;                        // 右侧超出裁剪框，顶点在裁剪框内
        }

        if(y > m_clip_box.y2) return 2;      // 顶部超出裁剪框
        if(y < m_clip_box.y1) return 8;      // 底部超出裁剪框

        return 0;                            // 顶点在裁剪框内
    }

    //----------------------------------------------------------------------------
    // 重置顶点生成器，将顶点数量和当前顶点索引归零
    void vpgen_clip_polygon::reset()
    {
        m_vertex = 0;       // 当前顶点索引归零
        m_num_vertices = 0; // 顶点数量归零
    }

    //----------------------------------------------------------------------------
    // 移动到指定坐标，设置起始顶点，计算裁剪标志
    void vpgen_clip_polygon::move_to(double x, double y)
    {
        m_vertex = 0;                      // 当前顶点索引归零
        m_num_vertices = 0;                // 顶点数量归零
        m_clip_flags = clipping_flags(x, y); // 计算裁剪标志
        if(m_clip_flags == 0)
        {
            m_x[0] = x;                   // 如果顶点在裁剪框内，记录顶点坐标
            m_y[0] = y;
            m_num_vertices = 1;           // 设置顶点数量为1
        }
        m_x1  = x;                        // 记录起始顶点坐标
        m_y1  = y;
        m_cmd = path_cmd_move_to;          // 设置路径命令为移动到命令
    }

    //----------------------------------------------------------------------------
    // 添加直线到指定坐标，处理裁剪框外的情况
    void vpgen_clip_polygon::line_to(double x, double y)
    {
        // 设置顶点和顶点数为初始值
        m_vertex = 0;
        m_num_vertices = 0;
    
        // 调用 clipping_flags 函数计算剪裁标志位
        unsigned flags = clipping_flags(x, y);
    
        // 如果当前剪裁标志与新计算的标志相同
        if(m_clip_flags == flags)
        {
            // 如果标志为零，表示无需剪裁
            if(flags == 0)
            {
                // 将给定的 x 和 y 分别作为第一个顶点的坐标
                m_x[0] = x;
                m_y[0] = y;
                // 设置顶点数为1
                m_num_vertices = 1;
            }
        }
        else
        {
            // 使用 Liang-Barsky 算法对线段进行剪裁，更新顶点数组和顶点数
            m_num_vertices = clip_liang_barsky(m_x1, m_y1, 
                                               x, y, 
                                               m_clip_box, 
                                               m_x, m_y);
        }
    
        // 更新剪裁标志为最新计算的标志
        m_clip_flags = flags;
        // 更新上一个顶点的坐标为当前传入的 x 和 y
        m_x1 = x;
        m_y1 = y;
    }
    
    
    //----------------------------------------------------------------------------
    unsigned vpgen_clip_polygon::vertex(double* x, double* y)
    {
        // 如果当前顶点索引小于顶点数
        if(m_vertex < m_num_vertices)
        {
            // 将当前顶点的坐标赋值给指定的 x 和 y
            *x = m_x[m_vertex];
            *y = m_y[m_vertex];
            // 增加顶点索引
            ++m_vertex;
            // 将当前命令类型存储在临时变量 cmd 中，并将命令类型设置为 path_cmd_line_to
            unsigned cmd = m_cmd;
            m_cmd = path_cmd_line_to;
            // 返回之前的命令类型
            return cmd;
        }
        // 如果顶点索引不小于顶点数，则返回停止命令类型 path_cmd_stop
        return path_cmd_stop;
    }
}



# 这行代码表示一个代码块的结束，这里可能是某个函数、循环、条件语句或类定义的结束位置。
```