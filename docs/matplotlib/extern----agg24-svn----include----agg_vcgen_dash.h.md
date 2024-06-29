# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_vcgen_dash.h`

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
// Line dash generator
//
//----------------------------------------------------------------------------

#ifndef AGG_VCGEN_DASH_INCLUDED
#define AGG_VCGEN_DASH_INCLUDED

#include "agg_basics.h"
#include "agg_vertex_sequence.h"

namespace agg
{

    //---------------------------------------------------------------vcgen_dash
    //
    // See Implementation agg_vcgen_dash.cpp
    //
    class vcgen_dash
    {
        enum max_dashes_e
        {
            max_dashes = 32
        };

        enum status_e
        {
            initial,        // 初始状态，尚未开始生成顶点
            ready,          // 就绪状态，准备生成顶点
            polyline,       // 折线状态，正在生成折线段的顶点
            stop            // 停止状态，生成顶点结束
        };

    public:
        typedef vertex_sequence<vertex_dist, 6> vertex_storage;

        // 构造函数，初始化成员变量
        vcgen_dash();

        // 移除所有虚线段
        void remove_all_dashes();

        // 添加一个虚线段，参数为虚线段长度和间隔长度
        void add_dash(double dash_len, double gap_len);

        // 设置虚线段的起始点
        void dash_start(double ds);

        // 设置虚线段的缩短量
        void shorten(double s) { m_shorten = s; }

        // 获取虚线段的缩短量
        double shorten() const { return m_shorten; }

        // 实现顶点生成器接口，添加顶点到内部顶点序列
        void add_vertex(double x, double y, unsigned cmd);

        // 实现顶点源接口，初始化重绕操作
        void rewind(unsigned path_id);

        // 实现顶点源接口，获取下一个顶点的坐标
        unsigned vertex(double* x, double* y);

    private:
        // 禁止拷贝构造和赋值操作
        vcgen_dash(const vcgen_dash&);
        const vcgen_dash& operator = (const vcgen_dash&);

        // 计算虚线段的起始点
        void calc_dash_start(double ds);

        double             m_dashes[max_dashes];      // 虚线段长度数组
        double             m_total_dash_len;         // 总虚线段长度
        unsigned           m_num_dashes;             // 虚线段数量
        double             m_dash_start;             // 虚线段起始点
        double             m_shorten;                // 虚线段缩短量
        double             m_curr_dash_start;        // 当前虚线段的起始点
        unsigned           m_curr_dash;              // 当前处理的虚线段索引
        double             m_curr_rest;              // 当前虚线段剩余长度
        const vertex_dist* m_v1;                     // 第一个顶点
        const vertex_dist* m_v2;                     // 第二个顶点

        vertex_storage m_src_vertices;   // 源顶点序列
        unsigned       m_closed;         // 是否封闭路径
        status_e       m_status;         // 当前状态
        unsigned       m_src_vertex;     // 源顶点索引
    };


}

#endif
```