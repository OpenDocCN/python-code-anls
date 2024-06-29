# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_vpgen_clip_polyline.h`

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

#ifndef AGG_VPGEN_CLIP_POLYLINE_INCLUDED
#define AGG_VPGEN_CLIP_POLYLINE_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //======================================================vpgen_clip_polyline
    //
    // See Implementation agg_vpgen_clip_polyline.cpp
    //
    // 定义一个类 vpgen_clip_polyline，用于裁剪多边形
    class vpgen_clip_polyline
    {
    public:
        // 构造函数，初始化成员变量
        vpgen_clip_polyline() : 
            m_clip_box(0, 0, 1, 1),    // 初始化裁剪框为默认值 (0, 0, 1, 1)
            m_x1(0),                   // 初始化 m_x1 为 0
            m_y1(0),                   // 初始化 m_y1 为 0
            m_num_vertices(0),         // 初始化顶点数为 0
            m_vertex(0),               // 初始化当前顶点索引为 0
            m_move_to(false)           // 初始化移动标志为 false
        {
        }

        // 设置裁剪框的坐标
        void clip_box(double x1, double y1, double x2, double y2)
        {
            m_clip_box.x1 = x1;    // 设置裁剪框左上角 x 坐标
            m_clip_box.y1 = y1;    // 设置裁剪框左上角 y 坐标
            m_clip_box.x2 = x2;    // 设置裁剪框右下角 x 坐标
            m_clip_box.y2 = y2;    // 设置裁剪框右下角 y 坐标
            m_clip_box.normalize(); // 标准化裁剪框（确保 x1 <= x2，y1 <= y2）
        }

        // 返回裁剪框的左上角 x 坐标
        double x1() const { return m_clip_box.x1; }
        // 返回裁剪框的左上角 y 坐标
        double y1() const { return m_clip_box.y1; }
        // 返回裁剪框的右下角 x 坐标
        double x2() const { return m_clip_box.x2; }
        // 返回裁剪框的右下角 y 坐标
        double y2() const { return m_clip_box.y2; }

        // 返回是否自动闭合多边形的静态方法（不自动闭合）
        static bool auto_close()   { return false; }
        // 返回是否自动取消闭合多边形的静态方法（自动取消闭合）
        static bool auto_unclose() { return true; }

        // 重置多边形生成器状态
        void     reset();
        // 移动到指定坐标
        void     move_to(double x, double y);
        // 添加直线到指定坐标
        void     line_to(double x, double y);
        // 返回下一个顶点的坐标，并更新当前顶点索引
        unsigned vertex(double* x, double* y);

    private:
        rect_d        m_clip_box;    // 裁剪框对象
        double        m_x1;          // 起始 x 坐标
        double        m_y1;          // 起始 y 坐标
        double        m_x[2];        // x 坐标数组（用于存储直线段的两个端点）
        double        m_y[2];        // y 坐标数组（用于存储直线段的两个端点）
        unsigned      m_cmd[2];      // 命令数组（用于存储直线段的两个端点的命令）
        unsigned      m_num_vertices; // 多边形顶点数
        unsigned      m_vertex;       // 当前顶点索引
        bool          m_move_to;      // 移动标志（指示是否已经执行了 move_to 操作）
    };

}


#endif
```