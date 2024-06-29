# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_vcgen_smooth_poly1.cpp`

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
// Smooth polygon generator
//
//----------------------------------------------------------------------------

#include "agg_vcgen_smooth_poly1.h"

namespace agg
{

    //------------------------------------------------------------------------
    // 构造函数，初始化成员变量
    vcgen_smooth_poly1::vcgen_smooth_poly1() :
        m_src_vertices(),                // 初始化顶点列表为空
        m_smooth_value(0.5),             // 平滑值设为0.5
        m_closed(0),                     // 初始闭合状态为0
        m_status(initial),               // 初始状态为initial
        m_src_vertex(0)                  // 初始源顶点索引为0
    {
    }


    //------------------------------------------------------------------------
    // 清空所有顶点和状态
    void vcgen_smooth_poly1::remove_all()
    {
        m_src_vertices.remove_all();      // 清空源顶点列表
        m_closed = 0;                     // 闭合状态设为0
        m_status = initial;               // 状态设为initial
    }


    //------------------------------------------------------------------------
    // 添加顶点到顶点列表
    void vcgen_smooth_poly1::add_vertex(double x, double y, unsigned cmd)
    {
        m_status = initial;               // 设定状态为initial
        if(is_move_to(cmd))               // 如果是移动到命令
        {
            m_src_vertices.modify_last(vertex_dist(x, y));  // 修改最后一个顶点坐标
        }
        else
        {
            if(is_vertex(cmd))            // 如果是顶点命令
            {
                m_src_vertices.add(vertex_dist(x, y));       // 添加新顶点到列表
            }
            else                          // 否则（假定是关闭命令）
            {
                m_closed = get_close_flag(cmd);  // 获取并设置闭合标志
            }
        }
    }


    //------------------------------------------------------------------------
    // 重置生成器状态
    void vcgen_smooth_poly1::rewind(unsigned)
    {
        if(m_status == initial)           // 如果状态为initial
        {
            m_src_vertices.close(m_closed != 0);  // 根据闭合标志关闭顶点列表
        }
        m_status = ready;                 // 设置状态为ready
        m_src_vertex = 0;                 // 源顶点索引重置为0
    }


    //------------------------------------------------------------------------
    // 计算平滑处理后的顶点
    void vcgen_smooth_poly1::calculate(const vertex_dist& v0, 
                                       const vertex_dist& v1, 
                                       const vertex_dist& v2,
                                       const vertex_dist& v3)
    {
        // 计算两条线段的插值比例系数
        double k1 = v0.dist / (v0.dist + v1.dist);
        double k2 = v1.dist / (v1.dist + v2.dist);
    
        // 计算第一个控制点的坐标，使用线性插值公式
        double xm1 = v0.x + (v2.x - v0.x) * k1;
        double ym1 = v0.y + (v2.y - v0.y) * k1;
        
        // 计算第二个控制点的坐标，使用线性插值公式
        double xm2 = v1.x + (v3.x - v1.x) * k2;
        double ym2 = v1.y + (v3.y - v1.y) * k2;
    
        // 计算平滑曲线的控制点坐标，包含平滑值 m_smooth_value 的影响
        m_ctrl1_x = v1.x + m_smooth_value * (v2.x - xm1);
        m_ctrl1_y = v1.y + m_smooth_value * (v2.y - ym1);
        m_ctrl2_x = v2.x + m_smooth_value * (v1.x - xm2);
        m_ctrl2_y = v2.y + m_smooth_value * (v1.y - ym2);
    }
    
    
    //------------------------------------------------------------------------
    unsigned vcgen_smooth_poly1::vertex(double* x, double* y)
    }
}


注释：

# 这行代码表示一个函数的结束，右花括号 '}' 通常用于标记代码块的结束。
```