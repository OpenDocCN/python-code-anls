# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_vcgen_bspline.cpp`

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

#include "agg_vcgen_bspline.h"

namespace agg
{

    //------------------------------------------------------------------------
    // 构造函数，初始化各成员变量
    vcgen_bspline::vcgen_bspline() :
        m_src_vertices(),               // 初始化源顶点列表为空
        m_spline_x(),                   // 初始化 B 样条曲线的 X 坐标控制点为空
        m_spline_y(),                   // 初始化 B 样条曲线的 Y 坐标控制点为空
        m_interpolation_step(1.0/50.0), // 初始化插值步长为 1/50
        m_closed(0),                    // 初始化闭合标志为 false
        m_status(initial),              // 初始化状态为初始状态
        m_src_vertex(0)                 // 初始化源顶点索引为 0
    {
    }


    //------------------------------------------------------------------------
    // 清空所有数据成员
    void vcgen_bspline::remove_all()
    {
        m_src_vertices.remove_all();    // 清空源顶点列表
        m_closed = 0;                   // 设置闭合标志为 false
        m_status = initial;             // 设置状态为初始状态
        m_src_vertex = 0;               // 设置源顶点索引为 0
    }


    //------------------------------------------------------------------------
    // 添加顶点到源顶点列表
    void vcgen_bspline::add_vertex(double x, double y, unsigned cmd)
    {
        m_status = initial;             // 设置状态为初始状态
        if(is_move_to(cmd))             // 如果命令是移动到操作
        {
            m_src_vertices.modify_last(point_d(x, y));  // 修改最后一个顶点的坐标为 (x, y)
        }
        else
        {
            if(is_vertex(cmd))          // 如果命令是顶点
            {
                m_src_vertices.add(point_d(x, y));       // 将 (x, y) 添加到源顶点列表
            }
            else                        // 其他情况，如结束或关闭路径
            {
                m_closed = get_close_flag(cmd);         // 获取并设置闭合标志
            }
        }
    }


    //------------------------------------------------------------------------
    // 准备重置 B 样条生成器状态
    void vcgen_bspline::rewind(unsigned)
    {
        // 初始化当前横坐标和最大横坐标为0
        m_cur_abscissa = 0.0;
        m_max_abscissa = 0.0;
        // 设置源顶点索引为0
        m_src_vertex = 0;
        // 如果状态为initial且源顶点数量大于2
        if (m_status == initial && m_src_vertices.size() > 2)
        {
            // 如果曲线闭合
            if (m_closed)
            {
                // 初始化样条曲线的X和Y方向，增加额外的控制点
                m_spline_x.init(m_src_vertices.size() + 8);
                m_spline_y.init(m_src_vertices.size() + 8);
                // 添加控制点：0.0处为倒数第3个顶点，1.0处为倒数第3个顶点，2.0处为倒数第2个顶点，3.0处为最后一个顶点
                m_spline_x.add_point(0.0, m_src_vertices.prev(m_src_vertices.size() - 3).x);
                m_spline_y.add_point(0.0, m_src_vertices.prev(m_src_vertices.size() - 3).y);
                m_spline_x.add_point(1.0, m_src_vertices[m_src_vertices.size() - 3].x);
                m_spline_y.add_point(1.0, m_src_vertices[m_src_vertices.size() - 3].y);
                m_spline_x.add_point(2.0, m_src_vertices[m_src_vertices.size() - 2].x);
                m_spline_y.add_point(2.0, m_src_vertices[m_src_vertices.size() - 2].y);
                m_spline_x.add_point(3.0, m_src_vertices[m_src_vertices.size() - 1].x);
                m_spline_y.add_point(3.0, m_src_vertices[m_src_vertices.size() - 1].y);
            }
            else
            {
                // 初始化非闭合状态下的样条曲线
                m_spline_x.init(m_src_vertices.size());
                m_spline_y.init(m_src_vertices.size());
            }
            // 添加顶点到样条曲线
            unsigned i;
            for (i = 0; i < m_src_vertices.size(); i++)
            {
                // 如果曲线闭合，x值从4开始，否则从i开始
                double x = m_closed ? i + 4 : i;
                m_spline_x.add_point(x, m_src_vertices[i].x);
                m_spline_y.add_point(x, m_src_vertices[i].y);
            }
            // 设置当前横坐标和最大横坐标的值
            m_cur_abscissa = 0.0;
            m_max_abscissa = m_src_vertices.size() - 1;
            // 如果曲线闭合，增加额外的控制点
            if (m_closed)
            {
                m_cur_abscissa = 4.0;
                m_max_abscissa += 5.0;
                m_spline_x.add_point(m_src_vertices.size() + 4, m_src_vertices[0].x);
                m_spline_y.add_point(m_src_vertices.size() + 4, m_src_vertices[0].y);
                m_spline_x.add_point(m_src_vertices.size() + 5, m_src_vertices[1].x);
                m_spline_y.add_point(m_src_vertices.size() + 5, m_src_vertices[1].y);
                m_spline_x.add_point(m_src_vertices.size() + 6, m_src_vertices[2].x);
                m_spline_y.add_point(m_src_vertices.size() + 6, m_src_vertices[2].y);
                m_spline_x.add_point(m_src_vertices.size() + 7, m_src_vertices.next(2).x);
                m_spline_y.add_point(m_src_vertices.size() + 7, m_src_vertices.next(2).y);
            }
            // 准备样条曲线
            m_spline_x.prepare();
            m_spline_y.prepare();
        }
        // 设置状态为ready
        m_status = ready;
    }
    
    
    
    
    
    //------------------------------------------------------------------------
    // 返回当前顶点索引的X和Y坐标
    unsigned vcgen_bspline::vertex(double* x, double* y)
    {
        unsigned cmd = path_cmd_line_to;  // 设置命令初始为直线命令
        while (!is_stop(cmd))  // 循环直到命令为停止命令
        {
            switch (m_status)  // 根据状态机状态进行处理
            {
            case initial:  // 初始状态
                rewind(0);  // 重新定位到初始位置
    
            case ready:  // 准备就绪状态
                if (m_src_vertices.size() < 2)  // 如果源顶点少于2个
                {
                    cmd = path_cmd_stop;  // 设置命令为停止命令
                    break;
                }
    
                if (m_src_vertices.size() == 2)  // 如果源顶点有2个
                {
                    *x = m_src_vertices[m_src_vertex].x;  // 将x设置为第一个源顶点的x坐标
                    *y = m_src_vertices[m_src_vertex].y;  // 将y设置为第一个源顶点的y坐标
                    m_src_vertex++;  // 顶点索引加1
                    if (m_src_vertex == 1) return path_cmd_move_to;  // 如果是第一个顶点，返回移动命令
                    if (m_src_vertex == 2) return path_cmd_line_to;  // 如果是第二个顶点，返回直线命令
                    cmd = path_cmd_stop;  // 否则设置命令为停止命令
                    break;
                }
    
                cmd = path_cmd_move_to;  // 设置命令为移动命令
                m_status = polygon;  // 状态变为多边形
                m_src_vertex = 0;  // 源顶点索引重置为0
    
            case polygon:  // 多边形状态
                if (m_cur_abscissa >= m_max_abscissa)  // 如果当前横坐标超过最大横坐标
                {
                    if (m_closed)  // 如果是封闭的多边形
                    {
                        m_status = end_poly;  // 状态变为结束多边形
                        break;
                    }
                    else  // 如果不是封闭的多边形
                    {
                        *x = m_src_vertices[m_src_vertices.size() - 1].x;  // 将x设置为最后一个源顶点的x坐标
                        *y = m_src_vertices[m_src_vertices.size() - 1].y;  // 将y设置为最后一个源顶点的y坐标
                        m_status = end_poly;  // 状态变为结束多边形
                        return path_cmd_line_to;  // 返回直线命令
                    }
                }
    
                *x = m_spline_x.get_stateful(m_cur_abscissa);  // 将x设置为当前样条曲线x坐标状态
                *y = m_spline_y.get_stateful(m_cur_abscissa);  // 将y设置为当前样条曲线y坐标状态
                m_src_vertex++;  // 源顶点索引加1
                m_cur_abscissa += m_interpolation_step;  // 当前横坐标增加插值步长
                return (m_src_vertex == 1) ? path_cmd_move_to : path_cmd_line_to;  // 如果是第一个顶点返回移动命令，否则返回直线命令
    
            case end_poly:  // 结束多边形状态
                m_status = stop;  // 状态变为停止状态
                return path_cmd_end_poly | m_closed;  // 返回结束多边形命令或运算上是否封闭
    
            case stop:  // 停止状态
                return path_cmd_stop;  // 返回停止命令
            }
        }
        return cmd;  // 返回当前命令
    }
}



# 这是一个单独的右花括号，用于闭合一个代码块或函数定义。
```