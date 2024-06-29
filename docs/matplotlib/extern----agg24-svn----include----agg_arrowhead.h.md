# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_arrowhead.h`

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
// Simple arrowhead/arrowtail generator 
//
//----------------------------------------------------------------------------
#ifndef AGG_ARROWHEAD_INCLUDED
#define AGG_ARROWHEAD_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //===============================================================arrowhead
    //
    // See implementation agg_arrowhead.cpp 
    //
    class arrowhead
    {
    public:
        // 构造函数，初始化箭头头部和尾部参数
        arrowhead();

        // 设置箭头头部参数
        void head(double d1, double d2, double d3, double d4)
        {
            m_head_d1 = d1;
            m_head_d2 = d2;
            m_head_d3 = d3;
            m_head_d4 = d4;
            m_head_flag = true;
        }

        // 启用默认箭头头部
        void head()    { m_head_flag = true; }

        // 禁用箭头头部
        void no_head() { m_head_flag = false; }

        // 设置箭头尾部参数
        void tail(double d1, double d2, double d3, double d4)
        {
            m_tail_d1 = d1;
            m_tail_d2 = d2;
            m_tail_d3 = d3;
            m_tail_d4 = d4;
            m_tail_flag = true;
        }

        // 启用默认箭头尾部
        void tail()    { m_tail_flag = true;  }

        // 禁用箭头尾部
        void no_tail() { m_tail_flag = false; }

        // 重置路径迭代器到指定的路径 ID
        void rewind(unsigned path_id);

        // 获取下一个顶点的坐标
        unsigned vertex(double* x, double* y);

    private:
        double   m_head_d1;   // 头部参数
        double   m_head_d2;
        double   m_head_d3;
        double   m_head_d4;
        double   m_tail_d1;   // 尾部参数
        double   m_tail_d2;
        double   m_tail_d3;
        double   m_tail_d4;
        bool     m_head_flag; // 是否启用头部
        bool     m_tail_flag; // 是否启用尾部
        double   m_coord[16]; // 存储坐标的数组
        unsigned m_cmd[8];    // 存储命令的数组
        unsigned m_curr_id;   // 当前路径 ID
        unsigned m_curr_coord; // 当前坐标索引
    };

}

#endif
```