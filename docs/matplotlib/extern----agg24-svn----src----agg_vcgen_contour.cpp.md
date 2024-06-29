# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_vcgen_contour.cpp`

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
// Contour generator
//
//----------------------------------------------------------------------------

#include <math.h>
#include "agg_vcgen_contour.h"

namespace agg
{

    //------------------------------------------------------------------------
    // Contour generator constructor
    vcgen_contour::vcgen_contour() :
        m_stroker(),                            // 初始化线条生成器
        m_width(1),                             // 线条宽度初始化为1
        m_src_vertices(),                       // 原始顶点序列
        m_out_vertices(),                       // 输出顶点序列
        m_status(initial),                      // 状态初始化为初始状态
        m_src_vertex(0),                        // 原始顶点索引
        m_closed(0),                            // 是否闭合的标志
        m_orientation(0),                       // 路径方向标志
        m_auto_detect(false)                    // 是否自动检测标志
    {
    }

    //------------------------------------------------------------------------
    // 清空所有数据
    void vcgen_contour::remove_all()
    {
        m_src_vertices.remove_all();            // 清空原始顶点序列
        m_closed = 0;                           // 重置闭合标志
        m_orientation = 0;                      // 重置路径方向标志
        m_status = initial;                     // 将状态设为初始状态
    }

    //------------------------------------------------------------------------
    // 添加顶点
    void vcgen_contour::add_vertex(double x, double y, unsigned cmd)
    {
        m_status = initial;                     // 将状态设为初始状态
        if(is_move_to(cmd))                     // 如果是移动到新起点命令
        {
            m_src_vertices.modify_last(vertex_dist(x, y)); // 修改最后一个顶点为新的坐标
        }
        else
        {
            if(is_vertex(cmd))                  // 如果是顶点命令
            {
                m_src_vertices.add(vertex_dist(x, y)); // 添加一个新顶点
            }
            else
            {
                if(is_end_poly(cmd))            // 如果是结束多边形命令
                {
                    m_closed = get_close_flag(cmd);  // 获取闭合标志
                    if(m_orientation == path_flags_none)  // 如果路径方向尚未设置
                    {
                        m_orientation = get_orientation(cmd); // 获取路径方向
                    }
                }
            }
        }
    }

    //------------------------------------------------------------------------
    // 重新设置到指定位置
    void vcgen_contour::rewind(unsigned)
    {
        // 如果当前状态为 initial
        if(m_status == initial)
        {
            // 关闭 m_src_vertices，并重置为初始状态
            m_src_vertices.close(true);
            
            // 如果启用自动检测
            if(m_auto_detect)
            {
                // 如果当前方向不是指定的方向，则根据多边形面积判断方向
                if(!is_oriented(m_orientation))
                {
                    m_orientation = (calc_polygon_area(m_src_vertices) > 0.0) ? 
                                    path_flags_ccw : 
                                    path_flags_cw;
                }
            }
            
            // 如果方向已经确定
            if(is_oriented(m_orientation))
            {
                // 根据方向设置描边器的宽度
                m_stroker.width(is_ccw(m_orientation) ? m_width : -m_width);
            }
        }
        
        // 设置状态为 ready
        m_status = ready;
        
        // 重置 m_src_vertex 为 0
        m_src_vertex = 0;
    }
    
    //------------------------------------------------------------------------
    unsigned vcgen_contour::vertex(double* x, double* y)
    {
        // 默认命令为线段到达
        unsigned cmd = path_cmd_line_to;
        
        // 循环直到遇到停止命令
        while(!is_stop(cmd))
        {
            // 根据当前状态执行不同的操作
            switch(m_status)
            {
            case initial:
                // 重新开始并设置为初始状态
                rewind(0);
    
            case ready:
                // 如果顶点数少于 2 加上闭合标志（unsigned(m_closed != 0) 表示是否闭合）
                if(m_src_vertices.size() < 2 + unsigned(m_closed != 0))
                {
                    // 设置命令为停止
                    cmd = path_cmd_stop;
                    break;
                }
                // 设置状态为 outline，并设置命令为移动到起始点
                m_status = outline;
                cmd = path_cmd_move_to;
                // 重置顶点索引
                m_src_vertex = 0;
                m_out_vertex = 0;
    
            case outline:
                // 如果当前顶点索引超过了顶点数组大小
                if(m_src_vertex >= m_src_vertices.size())
                {
                    // 设置状态为 end_poly
                    m_status = end_poly;
                    break;
                }
                // 计算连接点并设置状态为 out_vertices
                m_stroker.calc_join(m_out_vertices, 
                                    m_src_vertices.prev(m_src_vertex), 
                                    m_src_vertices.curr(m_src_vertex), 
                                    m_src_vertices.next(m_src_vertex), 
                                    m_src_vertices.prev(m_src_vertex).dist,
                                    m_src_vertices.curr(m_src_vertex).dist);
                // 增加顶点索引
                ++m_src_vertex;
                m_status = out_vertices;
                // 重置输出顶点索引
                m_out_vertex = 0;
    
            case out_vertices:
                // 如果当前输出顶点索引超过输出顶点数组大小
                if(m_out_vertex >= m_out_vertices.size())
                {
                    // 设置状态为 outline
                    m_status = outline;
                }
                else
                {
                    // 获取当前输出顶点坐标并返回命令为线段到达
                    const point_d& c = m_out_vertices[m_out_vertex++];
                    *x = c.x;
                    *y = c.y;
                    return cmd;
                }
                break;
    
            case end_poly:
                // 如果未闭合则返回停止命令
                if(!m_closed) return path_cmd_stop;
                // 设置状态为 stop，并返回结束多边形的命令和标志
                m_status = stop;
                return path_cmd_end_poly | path_flags_close | path_flags_ccw;
    
            case stop:
                // 返回停止命令
                return path_cmd_stop;
            }
        }
        // 返回命令
        return cmd;
    }
}



# 这行代码是一个代码块的结束标记，配对于一个以 '{' 开始的代码块，表示代码块的结束。
```