# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\ctrl\agg_polygon_ctrl.h`

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
// classes polygon_ctrl_impl, polygon_ctrl
//
//----------------------------------------------------------------------------

#ifndef POLYGON_CTRL_INCLUDED
#define POLYGON_CTRL_INCLUDED

#include "agg_array.h"
#include "agg_conv_stroke.h"
#include "agg_ellipse.h"
#include "agg_color_rgba.h"
#include "agg_ctrl.h"

namespace agg
{
    // 简单多边形顶点数据源类
    class simple_polygon_vertex_source
    {
    public:
        // 构造函数，初始化多边形顶点数据源
        simple_polygon_vertex_source(const double* polygon, unsigned np, 
                                     bool roundoff = false,
                                     bool close = true) :
            m_polygon(polygon),
            m_num_points(np),
            m_vertex(0),
            m_roundoff(roundoff),
            m_close(close)
        {
        }

        // 设置是否闭合多边形
        void close(bool f) { m_close = f;    }
        
        // 获取当前多边形是否闭合
        bool close() const { return m_close; }

        // 重置多边形顶点数据源的状态
        void rewind(unsigned)
        {
            m_vertex = 0;
        }

        // 获取下一个顶点的坐标，并返回相应的路径命令
        unsigned vertex(double* x, double* y)
        {
            if(m_vertex > m_num_points) return path_cmd_stop;
            if(m_vertex == m_num_points) 
            {
                ++m_vertex;
                return path_cmd_end_poly | (m_close ? path_flags_close : 0);
            }
            *x = m_polygon[m_vertex * 2];
            *y = m_polygon[m_vertex * 2 + 1];
            if(m_roundoff)
            {
                *x = floor(*x) + 0.5;
                *y = floor(*y) + 0.5;
            }
            ++m_vertex;
            return (m_vertex == 1) ? path_cmd_move_to : path_cmd_line_to;
        }

    private:
        const double* m_polygon;  // 多边形顶点数组
        unsigned m_num_points;    // 多边形顶点数
        unsigned m_vertex;        // 当前顶点索引
        bool     m_roundoff;      // 是否四舍五入
        bool     m_close;         // 是否闭合多边形
    };

    // 多边形控件实现类，继承自控件基类
    class polygon_ctrl_impl : public ctrl
    {
    // 构造函数，初始化多边形控制器实例
    public:
        polygon_ctrl_impl(unsigned np, double point_radius=5);

        // 返回多边形顶点数量
        unsigned num_points() const { return m_num_points; }
        // 返回第n个顶点的x坐标
        double xn(unsigned n) const { return m_polygon[n * 2];     }
        // 返回第n个顶点的y坐标
        double yn(unsigned n) const { return m_polygon[n * 2 + 1]; }
        // 返回第n个顶点的x坐标的引用
        double& xn(unsigned n) { return m_polygon[n * 2];     }
        // 返回第n个顶点的y坐标的引用
        double& yn(unsigned n) { return m_polygon[n * 2 + 1]; }
    
        // 返回多边形顶点数组的指针
        const double* polygon() const { return &m_polygon[0]; }

        // 设置线条宽度
        void   line_width(double w) { m_stroke.width(w); }
        // 返回线条宽度
        double line_width() const   { return m_stroke.width(); }

        // 设置顶点半径
        void   point_radius(double r) { m_point_radius = r; }
        // 返回顶点半径
        double point_radius() const   { return m_point_radius; }

        // 设置是否进行多边形内部检查
        void in_polygon_check(bool f) { m_in_polygon_check = f; }
        // 返回是否进行多边形内部检查
        bool in_polygon_check() const { return m_in_polygon_check; }

        // 设置是否封闭多边形
        void close(bool f) { m_vs.close(f);       }
        // 返回是否封闭多边形
        bool close() const { return m_vs.close(); }

        // 顶点源接口
        // 返回路径数目，这里固定为1
        unsigned num_paths() { return 1; }
        // 重置路径到起始位置
        void     rewind(unsigned path_id);
        // 获取下一个顶点的坐标
        unsigned vertex(double* x, double* y);

        // 虚函数，检查点(x, y)是否在多边形内部
        virtual bool in_rect(double x, double y) const;
        // 虚函数，处理鼠标按下事件
        virtual bool on_mouse_button_down(double x, double y);
        // 虚函数，处理鼠标释放事件
        virtual bool on_mouse_button_up(double x, double y);
        // 虚函数，处理鼠标移动事件
        virtual bool on_mouse_move(double x, double y, bool button_flag);
        // 虚函数，处理箭头键事件
        virtual bool on_arrow_keys(bool left, bool right, bool down, bool up);


    private:
        // 检查边界，判断点(x, y)是否在第i条边上
        bool check_edge(unsigned i, double x, double y) const;
        // 判断点(x, y)是否在多边形内部
        bool point_in_polygon(double x, double y) const;

        // 存储多边形顶点坐标的数组
        pod_array<double> m_polygon;
        // 多边形顶点数量
        unsigned          m_num_points;
        int               m_node;
        int               m_edge;
        // 简单多边形顶点源
        simple_polygon_vertex_source              m_vs;
        // 线条笔刷
        conv_stroke<simple_polygon_vertex_source> m_stroke;
        // 椭圆对象
        ellipse  m_ellipse;
        // 顶点半径
        double   m_point_radius;
        // 状态变量
        unsigned m_status;
        // x方向位移
        double   m_dx;
        // y方向位移
        double   m_dy;
        // 是否进行多边形内部检查的标志
        bool     m_in_polygon_check;
    };

    //----------------------------------------------------------polygon_ctrl
    // 模板类，继承自polygon_ctrl_impl，增加颜色属性
    template<class ColorT> class polygon_ctrl : public polygon_ctrl_impl
    {
    public:
        // 构造函数，初始化多边形控制器实例，设置点的半径和颜色
        polygon_ctrl(unsigned np, double point_radius=5) :
            polygon_ctrl_impl(np, point_radius),
            m_color(rgba(0.0, 0.0, 0.0))
        {
        }
          
        // 设置线条颜色
        void line_color(const ColorT& c) { m_color = c; }
        // 返回线条颜色
        const ColorT& color(unsigned i) const { return m_color; } 

    private:
        // 拷贝构造函数（私有，禁止拷贝）
        polygon_ctrl(const polygon_ctrl<ColorT>&);
        // 赋值运算符重载（私有，禁止赋值）
        const polygon_ctrl<ColorT>& operator = (const polygon_ctrl<ColorT>&);

        // 颜色属性
        ColorT m_color;
    };
}


注释：


// 这行代码关闭了之前的 #ifdef 指令所打开的条件编译块
#endif


这段代码看起来是在C或C++中进行条件编译（conditional compilation）。在这种情况下，`#ifdef`指令用来检查某个宏是否已经定义，如果定义了，则编译其中的代码块，否则忽略。这里的 `#endif` 用于结束 `#ifdef` 开始的条件编译块。
```