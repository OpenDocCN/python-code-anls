# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_vpgen_clip_polygon.h`

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

#ifndef AGG_VPGEN_CLIP_POLYGON_INCLUDED
#define AGG_VPGEN_CLIP_POLYGON_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //======================================================vpgen_clip_polygon
    //
    // See Implementation agg_vpgen_clip_polygon.cpp
    //
    class vpgen_clip_polygon
    {
    public:
        // 构造函数，默认初始化各个成员变量
        vpgen_clip_polygon() : 
            m_clip_box(0, 0, 1, 1),      // 初始化裁剪框矩形，默认为(0,0)-(1,1)
            m_x1(0),                     // 初始化坐标 x1
            m_y1(0),                     // 初始化坐标 y1
            m_clip_flags(0),             // 初始化裁剪标志
            m_num_vertices(0),           // 初始化顶点数目
            m_vertex(0),                 // 初始化当前顶点索引
            m_cmd(path_cmd_move_to)      // 初始化当前路径命令为移动到
        {
        }

        // 设置裁剪框的位置和大小
        void clip_box(double x1, double y1, double x2, double y2)
        {
            m_clip_box.x1 = x1;          // 设置裁剪框左下角 x 坐标
            m_clip_box.y1 = y1;          // 设置裁剪框左下角 y 坐标
            m_clip_box.x2 = x2;          // 设置裁剪框右上角 x 坐标
            m_clip_box.y2 = y2;          // 设置裁剪框右上角 y 坐标
            m_clip_box.normalize();      // 标准化裁剪框，确保 x1 <= x2 和 y1 <= y2
        }

        // 返回裁剪框的左下角 x 坐标
        double x1() const { return m_clip_box.x1; }
        
        // 返回裁剪框的左下角 y 坐标
        double y1() const { return m_clip_box.y1; }
        
        // 返回裁剪框的右上角 x 坐标
        double x2() const { return m_clip_box.x2; }
        
        // 返回裁剪框的右上角 y 坐标
        double y2() const { return m_clip_box.y2; }

        // 静态成员函数，返回自动闭合路径的状态，总是返回 true
        static bool auto_close()   { return true;  }
        
        // 静态成员函数，返回自动不闭合路径的状态，总是返回 false
        static bool auto_unclose() { return false; }

        // 重置对象的状态，清空所有路径数据
        void reset();

        // 移动当前点到指定坐标 (x, y)
        void move_to(double x, double y);

        // 从当前点画一条直线到指定坐标 (x, y)
        void line_to(double x, double y);

        // 返回当前顶点的坐标，通过指针参数返回 x 和 y
        unsigned vertex(double* x, double* y);

    private:
        // 计算给定坐标是否在裁剪框内部，并返回对应的裁剪标志
        unsigned clipping_flags(double x, double y);

    private:
        rect_d        m_clip_box;      // 裁剪框矩形
        double        m_x1;            // 坐标 x1
        double        m_y1;            // 坐标 y1
        unsigned      m_clip_flags;    // 裁剪标志
        double        m_x[4];          // 存储顶点 x 坐标的数组
        double        m_y[4];          // 存储顶点 y 坐标的数组
        unsigned      m_num_vertices;  // 顶点数目
        unsigned      m_vertex;        // 当前顶点索引
        unsigned      m_cmd;           // 当前路径命令
    };

}

#endif
```