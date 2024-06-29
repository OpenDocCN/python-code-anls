# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_vcgen_bspline.h`

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

#ifndef AGG_VCGEN_BSPLINE_INCLUDED
#define AGG_VCGEN_BSPLINE_INCLUDED

#include "agg_basics.h"
#include "agg_array.h"
#include "agg_bspline.h"

// 命名空间agg
namespace agg
{

    //==========================================================vcgen_bspline
    // vcgen_bspline类，用于生成B样条曲线的顶点
    class vcgen_bspline
    {
        // 状态枚举
        enum status_e
        {
            initial,    // 初始状态
            ready,      // 就绪状态
            polygon,    // 多边形状态
            end_poly,   // 结束多边形状态
            stop        // 停止状态
        };

    public:
        typedef pod_bvector<point_d, 6> vertex_storage; // 使用数组存储点的类型定义

        vcgen_bspline(); // 构造函数

        // 设置/获取插值步长
        void interpolation_step(double v) { m_interpolation_step = v; }
        double interpolation_step() const { return m_interpolation_step; }

        // 顶点生成器接口
        void remove_all(); // 移除所有顶点
        void add_vertex(double x, double y, unsigned cmd); // 添加顶点

        // 顶点源接口
        void     rewind(unsigned path_id); // 重置到指定路径ID
        unsigned vertex(double* x, double* y); // 获取顶点坐标

    private:
        vcgen_bspline(const vcgen_bspline&); // 复制构造函数（禁用）
        const vcgen_bspline& operator = (const vcgen_bspline&); // 赋值运算符（禁用）

        vertex_storage m_src_vertices; // 源顶点数组
        bspline        m_spline_x; // X方向的B样条曲线
        bspline        m_spline_y; // Y方向的B样条曲线
        double         m_interpolation_step; // 插值步长
        unsigned       m_closed; // 是否闭合
        status_e       m_status; // 当前状态
        unsigned       m_src_vertex; // 源顶点索引
        double         m_cur_abscissa; // 当前横坐标
        double         m_max_abscissa; // 最大横坐标
    };

}

#endif // AGG_VCGEN_BSPLINE_INCLUDED
```