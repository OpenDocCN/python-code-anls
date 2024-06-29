# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_vcgen_markers_term.h`

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

#ifndef AGG_VCGEN_MARKERS_TERM_INCLUDED
#define AGG_VCGEN_MARKERS_TERM_INCLUDED

#include "agg_basics.h"
#include "agg_vertex_sequence.h"

namespace agg
{

    //======================================================vcgen_markers_term
    //
    // See Implemantation agg_vcgen_markers_term.cpp
    // Terminal markers generator (arrowhead/arrowtail)
    //
    //------------------------------------------------------------------------
    class vcgen_markers_term
    {
    public:
        // 默认构造函数，初始化成员变量
        vcgen_markers_term() : m_curr_id(0), m_curr_idx(0) {}

        // 以下是顶点生成器接口函数
        // 清空所有顶点
        void remove_all();
        
        // 添加顶点
        void add_vertex(double x, double y, unsigned cmd);

        // 以下是顶点源接口函数
        // 重置生成器到指定路径 ID 的状态
        void rewind(unsigned path_id);

        // 获取下一个顶点的坐标，并返回其命令
        unsigned vertex(double* x, double* y);

    private:
        // 禁止复制构造函数和赋值运算符的私有方法
        vcgen_markers_term(const vcgen_markers_term&);
        const vcgen_markers_term& operator = (const vcgen_markers_term&);

        // 内部结构体，表示坐标点的类型
        struct coord_type
        {
            double x, y;

            coord_type() {} // 默认构造函数
            coord_type(double x_, double y_) : x(x_), y(y_) {} // 初始化构造函数
        };

        // 使用 pod_bvector 存储坐标点，预分配 6 个元素的空间
        typedef pod_bvector<coord_type, 6> coord_storage; 

        coord_storage m_markers; // 存储坐标点的容器
        unsigned      m_curr_id; // 当前路径 ID
        unsigned      m_curr_idx; // 当前索引
    };


}

#endif
```