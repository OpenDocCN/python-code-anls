# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_vcgen_smooth_poly1.h`

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

#ifndef AGG_VCGEN_SMOOTH_POLY1_INCLUDED
#define AGG_VCGEN_SMOOTH_POLY1_INCLUDED

#include "agg_basics.h"
#include "agg_vertex_sequence.h"


namespace agg
{

    //======================================================vcgen_smooth_poly1
    //
    // See Implementation agg_vcgen_smooth_poly1.cpp 
    // Smooth polygon generator
    //
    //------------------------------------------------------------------------
    class vcgen_smooth_poly1
    {
        enum status_e
        {
            initial,    // 初始状态
            ready,      // 就绪状态
            polygon,    // 多边形状态
            ctrl_b,     // 控制点开始状态
            ctrl_e,     // 控制点结束状态
            ctrl1,      // 控制点1
            ctrl2,      // 控制点2
            end_poly,   // 结束多边形状态
            stop        // 停止状态
        };

    public:
        typedef vertex_sequence<vertex_dist, 6> vertex_storage;

        vcgen_smooth_poly1();   // 构造函数

        void   smooth_value(double v) { m_smooth_value = v * 0.5; }    // 设置平滑值
        double smooth_value() const { return m_smooth_value * 2.0; }   // 获取平滑值

        // Vertex Generator Interface
        void remove_all();                      // 移除所有顶点
        void add_vertex(double x, double y, unsigned cmd);   // 添加顶点

        // Vertex Source Interface
        void     rewind(unsigned path_id);      // 重置为指定路径的起点
        unsigned vertex(double* x, double* y);  // 获取下一个顶点坐标

    private:
        vcgen_smooth_poly1(const vcgen_smooth_poly1&);   // 复制构造函数（私有）
        const vcgen_smooth_poly1& operator = (const vcgen_smooth_poly1&);   // 赋值运算符重载（私有）

        void calculate(const vertex_dist& v0, 
                       const vertex_dist& v1, 
                       const vertex_dist& v2,
                       const vertex_dist& v3);   // 计算平滑控制点

        vertex_storage m_src_vertices;   // 源顶点序列
        double         m_smooth_value;   // 平滑值
        unsigned       m_closed;         // 多边形闭合状态
        status_e       m_status;         // 当前状态
        unsigned       m_src_vertex;     // 源顶点索引
        double         m_ctrl1_x;        // 控制点1的x坐标
        double         m_ctrl1_y;        // 控制点1的y坐标
        double         m_ctrl2_x;        // 控制点2的x坐标
        double         m_ctrl2_y;        // 控制点2的y坐标
    };

}


#endif
```