# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_close_polygon.h`

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

#ifndef AGG_CONV_CLOSE_POLYGON_INCLUDED
#define AGG_CONV_CLOSE_POLYGON_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //======================================================conv_close_polygon
    // conv_close_polygon 模板类用于封闭多边形的顶点源
    template<class VertexSource> class conv_close_polygon
    {
    public:
        // 构造函数，初始化顶点源
        explicit conv_close_polygon(VertexSource& vs) : m_source(&vs) {}
        
        // 重新设置顶点源
        void attach(VertexSource& source) { m_source = &source; }

        // 重置顶点源中指定路径的迭代状态
        void rewind(unsigned path_id);
        
        // 获取下一个顶点的坐标
        unsigned vertex(double* x, double* y);

    private:
        // 禁止复制和赋值构造函数
        conv_close_polygon(const conv_close_polygon<VertexSource>&);
        const conv_close_polygon<VertexSource>& 
            operator = (const conv_close_polygon<VertexSource>&);

        // 成员变量
        VertexSource* m_source; // 指向顶点源的指针
        unsigned      m_cmd[2]; // 当前和上一个命令
        double        m_x[2];   // 当前和上一个顶点的 x 坐标
        double        m_y[2];   // 当前和上一个顶点的 y 坐标
        unsigned      m_vertex; // 当前顶点索引
        bool          m_line_to; // 是否为线段
    };



    //------------------------------------------------------------------------
    // 重置迭代器，使其从指定路径的第二个顶点开始
    template<class VertexSource> 
    void conv_close_polygon<VertexSource>::rewind(unsigned path_id)
    {
        m_source->rewind(path_id); // 调用顶点源的 rewind 方法
        m_vertex = 2;              // 设置当前顶点索引为 2（第二个顶点）
        m_line_to = false;         // 将线段标志设为 false
    }


    
    //------------------------------------------------------------------------
    // 获取下一个顶点的坐标
    template<class VertexSource> 
    unsigned conv_close_polygon<VertexSource>::vertex(double* x, double* y)
    {
        // 初始化命令为停止命令
        unsigned cmd = path_cmd_stop;
        // 无限循环，直到条件满足退出循环
        for(;;)
        {
            // 如果顶点数少于2个
            if(m_vertex < 2)
            {
                // 获取当前顶点的坐标和命令
                *x = m_x[m_vertex];
                *y = m_y[m_vertex];
                cmd = m_cmd[m_vertex];
                // 增加顶点索引，准备退出循环
                ++m_vertex;
                break;
            }
    
            // 从源对象获取顶点坐标和命令
            cmd = m_source->vertex(x, y);
    
            // 如果命令表示多边形结束
            if(is_end_poly(cmd))
            {
                // 添加闭合路径标志到命令中，然后退出循环
                cmd |= path_flags_close;
                break;
            }
    
            // 如果命令表示停止
            if(is_stop(cmd))
            {
                // 如果正在绘制直线到路径
                if(m_line_to)
                {
                    // 设定第一个命令为结束多边形并闭合，第二个命令为停止
                    m_cmd[0]  = path_cmd_end_poly | path_flags_close;
                    m_cmd[1]  = path_cmd_stop;
                    // 重置顶点索引和绘制直线到标志
                    m_vertex  = 0;
                    m_line_to = false;
                    continue; // 继续循环处理
                }
                break; // 退出循环
            }
    
            // 如果命令表示移动到新点
            if(is_move_to(cmd))
            {
                // 如果正在绘制直线到路径
                if(m_line_to)
                {
                    // 初始化第一个顶点为原点，设定第一个命令为结束多边形并闭合
                    m_x[0]    = 0.0;
                    m_y[0]    = 0.0;
                    m_cmd[0]  = path_cmd_end_poly | path_flags_close;
                    // 设定第二个顶点为当前坐标，命令为移动到命令
                    m_x[1]    = *x;
                    m_y[1]    = *y;
                    m_cmd[1]  = cmd;
                    // 重置顶点索引和绘制直线到标志
                    m_vertex  = 0;
                    m_line_to = false;
                    continue; // 继续循环处理
                }
                break; // 退出循环
            }
    
            // 如果命令表示顶点
            if(is_vertex(cmd))
            {
                // 设置绘制直线到标志，然后退出循环
                m_line_to = true;
                break;
            }
        }
        // 返回处理后的命令
        return cmd;
    }
}
// 结束条件指示符号的预处理器指令，与 #ifdef 或 #ifndef 配对使用
#endif
```