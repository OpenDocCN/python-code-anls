# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_bezier_arc.h`

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
// Arc generator. Produces at most 4 consecutive cubic bezier curves, i.e., 
// 4, 7, 10, or 13 vertices.
//
//----------------------------------------------------------------------------

#ifndef AGG_BEZIER_ARC_INCLUDED
#define AGG_BEZIER_ARC_INCLUDED

#include "agg_conv_transform.h"

namespace agg
{

    //-----------------------------------------------------------------------
    // 将椭圆弧转换为最多4条连续的三次贝塞尔曲线
    void arc_to_bezier(double cx, double cy, double rx, double ry, 
                       double start_angle, double sweep_angle,
                       double* curve);


    //==============================================================bezier_arc
    // 
    // 查看实现 agg_bezier_arc.cpp
    //
    class bezier_arc
    {
    //==========================================================bezier_arc
    // 定义一个贝塞尔曲线类
    //
    // 构造函数用于初始化对象，设置顶点数组大小为 26，顶点数为 0，并设置命令为线段绘制
    bezier_arc() : m_vertex(26), m_num_vertices(0), m_cmd(path_cmd_line_to) {}

    // 构造函数，接受椭圆弧的参数并初始化
    bezier_arc(double x,  double y, 
               double rx, double ry, 
               double start_angle, 
               double sweep_angle)
    {
        // 调用初始化函数初始化椭圆弧对象
        init(x, y, rx, ry, start_angle, sweep_angle);
    }

    //--------------------------------------------------------------------
    // 初始化椭圆弧对象的函数
    void init(double x,  double y, 
              double rx, double ry, 
              double start_angle, 
              double sweep_angle);

    //--------------------------------------------------------------------
    // 重置对象状态的函数
    void rewind(unsigned)
    {
        // 将顶点索引重置为 0
        m_vertex = 0;
    }

    //--------------------------------------------------------------------
    // 返回当前顶点的坐标，并更新顶点索引
    unsigned vertex(double* x, double* y)
    {
        if(m_vertex >= m_num_vertices) return path_cmd_stop;
        // 从顶点数组中获取当前顶点的坐标
        *x = m_vertices[m_vertex];
        *y = m_vertices[m_vertex + 1];
        // 更新顶点索引，每次增加 2，因为顶点数组以 x, y 对的形式存储
        m_vertex += 2;
        // 如果是第一个顶点，则返回移动到新位置的命令，否则返回之前设置的命令
        return (m_vertex == 2) ? path_cmd_move_to : m_cmd;
    }

    // 补充函数。num_vertices() 实际上返回顶点数的两倍，因为每个顶点都有 x 和 y 坐标。
    //--------------------------------------------------------------------
    unsigned  num_vertices() const { return m_num_vertices; }
    // 返回顶点数组的常量指针
    const double* vertices() const { return m_vertices;     }
    // 返回顶点数组的非常量指针，允许修改顶点数据
    double*       vertices()       { return m_vertices;     }

private:
    unsigned m_vertex;     // 当前顶点索引
    unsigned m_num_vertices; // 顶点总数
    double   m_vertices[26]; // 存储顶点坐标的数组，大小为 26
    unsigned m_cmd;         // 当前路径命令
};



//==========================================================bezier_arc_svg
// 计算 SVG 风格的贝塞尔曲线弧
//
// 计算从 (x1, y1) 到 (x2, y2) 的椭圆弧。椭圆的大小和方向由两个半径 (rx, ry) 
// 和一个 x 轴旋转角度定义，该角度指示整个椭圆相对于当前坐标系的旋转。椭圆的中心 
// (cx, cy) 将自动计算以满足其他参数施加的约束条件。
// large-arc-flag 和 sweep-flag 有助于自动计算并确定弧是如何绘制的。
class bezier_arc_svg
{
    // 公共部分开始
    public:
        //--------------------------------------------------------------------
        // 默认构造函数，初始化成员变量 m_arc 和 m_radii_ok
        bezier_arc_svg() : m_arc(), m_radii_ok(false) {}

        // 带参数的构造函数，调用 init 方法初始化成员变量
        bezier_arc_svg(double x1, double y1, 
                       double rx, double ry, 
                       double angle,
                       bool large_arc_flag,
                       bool sweep_flag,
                       double x2, double y2) : 
            m_arc(), m_radii_ok(false)
        {
            init(x1, y1, rx, ry, angle, large_arc_flag, sweep_flag, x2, y2);
        }

        //--------------------------------------------------------------------
        // 初始化函数声明，用于设置椭圆弧的参数
        void init(double x1, double y1, 
                  double rx, double ry, 
                  double angle,
                  bool large_arc_flag,
                  bool sweep_flag,
                  double x2, double y2);

        //--------------------------------------------------------------------
        // 返回椭圆弧的半径是否有效的布尔值
        bool radii_ok() const { return m_radii_ok; }

        //--------------------------------------------------------------------
        // 重置函数，将 m_arc 对象的路径重置为初始状态
        void rewind(unsigned)
        {
            m_arc.rewind(0);
        }

        //--------------------------------------------------------------------
        // 获取下一个顶点的坐标，并返回顶点索引
        unsigned vertex(double* x, double* y)
        {
            return m_arc.vertex(x, y);
        }

        // 补充函数。num_vertices() 实际上返回顶点数的两倍。
        // 即使顶点数为1，也返回2。
        //--------------------------------------------------------------------
        // 返回顶点数的两倍
        unsigned num_vertices() const { return m_arc.num_vertices(); }
        // 返回顶点数组的常量指针
        const double* vertices() const { return m_arc.vertices();     }
        // 返回顶点数组的非常量指针
        double* vertices()       { return m_arc.vertices();     }

    private:
        // 私有成员变量，用于存储贝塞尔曲线的对象和椭圆弧半径是否有效的标志
        bezier_arc m_arc;
        bool       m_radii_ok;
    };
}



#endif



// 这里是 C/C++ 的预处理指令，用于结束条件编译指令块的定义
}



// 这里是 C/C++ 的预处理指令，用于结束条件编译指令的条件
#endif
```