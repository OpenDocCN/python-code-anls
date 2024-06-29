# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\ctrl\agg_bezier_ctrl.h`

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
// classes bezier_ctrl_impl, bezier_ctrl
//
//----------------------------------------------------------------------------

#ifndef AGG_BEZIER_CTRL_INCLUDED
#define AGG_BEZIER_CTRL_INCLUDED

#include "agg_math.h"            // 包含数学计算相关的头文件
#include "agg_ellipse.h"         // 包含椭圆绘制相关的头文件
#include "agg_trans_affine.h"    // 包含仿射变换相关的头文件
#include "agg_color_rgba.h"      // 包含 RGBA 颜色处理相关的头文件
#include "agg_conv_stroke.h"     // 包含线条转换相关的头文件
#include "agg_conv_curve.h"      // 包含曲线转换相关的头文件
#include "agg_polygon_ctrl.h"    // 包含多边形控制相关的头文件

namespace agg
{

    //--------------------------------------------------------bezier_ctrl_impl
    // bezier_ctrl_impl 类，继承自 ctrl 类
    class bezier_ctrl_impl : public ctrl
    {
    # bezier_ctrl_impl 类的公共部分
    public:
        # 默认构造函数
        bezier_ctrl_impl();

        # 设置贝塞尔曲线的控制点坐标
        void curve(double x1, double y1, 
                   double x2, double y2, 
                   double x3, double y3,
                   double x4, double y4);
        
        # 返回当前的贝塞尔曲线对象
        curve4& curve();

        # 返回第一个控制点的 x 坐标
        double x1() const { return m_poly.xn(0); }
        # 返回第一个控制点的 y 坐标
        double y1() const { return m_poly.yn(0); }
        # 返回第二个控制点的 x 坐标
        double x2() const { return m_poly.xn(1); }
        # 返回第二个控制点的 y 坐标
        double y2() const { return m_poly.yn(1); }
        # 返回第三个控制点的 x 坐标
        double x3() const { return m_poly.xn(2); }
        # 返回第三个控制点的 y 坐标
        double y3() const { return m_poly.yn(2); }
        # 返回第四个控制点的 x 坐标
        double x4() const { return m_poly.xn(3); }
        # 返回第四个控制点的 y 坐标
        double y4() const { return m_poly.yn(3); }

        # 设置第一个控制点的 x 坐标
        void x1(double x) { m_poly.xn(0) = x; }
        # 设置第一个控制点的 y 坐标
        void y1(double y) { m_poly.yn(0) = y; }
        # 设置第二个控制点的 x 坐标
        void x2(double x) { m_poly.xn(1) = x; }
        # 设置第二个控制点的 y 坐标
        void y2(double y) { m_poly.yn(1) = y; }
        # 设置第三个控制点的 x 坐标
        void x3(double x) { m_poly.xn(2) = x; }
        # 设置第三个控制点的 y 坐标
        void y3(double y) { m_poly.yn(2) = y; }
        # 设置第四个控制点的 x 坐标
        void x4(double x) { m_poly.xn(3) = x; }
        # 设置第四个控制点的 y 坐标
        void y4(double y) { m_poly.yn(3) = y; }

        # 设置线条宽度
        void   line_width(double w) { m_stroke.width(w); }
        # 返回线条宽度
        double line_width() const   { return m_stroke.width(); }

        # 设置点的半径
        void   point_radius(double r) { m_poly.point_radius(r); }
        # 返回点的半径
        double point_radius() const   { return m_poly.point_radius(); }

        # 判断指定坐标是否在对象的矩形区域内
        virtual bool in_rect(double x, double y) const;
        # 处理鼠标按下事件
        virtual bool on_mouse_button_down(double x, double y);
        # 处理鼠标释放事件
        virtual bool on_mouse_button_up(double x, double y);
        # 处理鼠标移动事件
        virtual bool on_mouse_move(double x, double y, bool button_flag);
        # 处理方向键事件
        virtual bool on_arrow_keys(bool left, bool right, bool down, bool up);

        # 实现顶点源接口，返回路径数目
        unsigned num_paths() { return 7; };
        # 实现顶点源接口，重新设置路径
        void     rewind(unsigned path_id);
        # 实现顶点源接口，返回顶点坐标
        unsigned vertex(double* x, double* y);


    private:
        # 贝塞尔曲线对象
        curve4              m_curve;
        # 椭圆对象
        ellipse             m_ellipse;
        # 线条绘制对象
        conv_stroke<curve4> m_stroke;
        # 多边形控制对象
        polygon_ctrl_impl   m_poly;
        # 索引
        unsigned            m_idx;
    };



    //----------------------------------------------------------bezier_ctrl
    # 模板类 bezier_ctrl，继承自 bezier_ctrl_impl
    template<class ColorT> class bezier_ctrl : public bezier_ctrl_impl
    {
    public:
        # 默认构造函数，设置默认颜色为黑色
        bezier_ctrl() :
            m_color(rgba(0.0, 0.0, 0.0))
        {
        }
          
        # 设置线条颜色
        void line_color(const ColorT& c) { m_color = c; }
        # 返回颜色对象
        const ColorT& color(unsigned i) const { return m_color; } 

    private:
        # 复制构造函数，禁止复制
        bezier_ctrl(const bezier_ctrl<ColorT>&);
        # 赋值运算符重载，禁止赋值
        const bezier_ctrl<ColorT>& operator = (const bezier_ctrl<ColorT>&);

        # 颜色对象
        ColorT m_color;
    };





    //--------------------------------------------------------curve3_ctrl_impl
    # curve3_ctrl_impl 类，继承自 ctrl 类
    class curve3_ctrl_impl : public ctrl
    {
    // 定义 curve3_ctrl_impl 类的公共接口和私有成员变量

    public:
        // 默认构造函数
        curve3_ctrl_impl();

        // 设置三次曲线的控制点坐标
        void curve(double x1, double y1, 
                   double x2, double y2, 
                   double x3, double y3);
        // 返回三次曲线对象的引用
        curve3& curve();

        // 获取第一个控制点的 x 坐标
        double x1() const { return m_poly.xn(0); }
        // 获取第一个控制点的 y 坐标
        double y1() const { return m_poly.yn(0); }
        // 获取第二个控制点的 x 坐标
        double x2() const { return m_poly.xn(1); }
        // 获取第二个控制点的 y 坐标
        double y2() const { return m_poly.yn(1); }
        // 获取第三个控制点的 x 坐标
        double x3() const { return m_poly.xn(2); }
        // 获取第三个控制点的 y 坐标
        double y3() const { return m_poly.yn(2); }

        // 设置第一个控制点的 x 坐标
        void x1(double x) { m_poly.xn(0) = x; }
        // 设置第一个控制点的 y 坐标
        void y1(double y) { m_poly.yn(0) = y; }
        // 设置第二个控制点的 x 坐标
        void x2(double x) { m_poly.xn(1) = x; }
        // 设置第二个控制点的 y 坐标
        void y2(double y) { m_poly.yn(1) = y; }
        // 设置第三个控制点的 x 坐标
        void x3(double x) { m_poly.xn(2) = x; }
        // 设置第三个控制点的 y 坐标
        void y3(double y) { m_poly.yn(2) = y; }

        // 设置线宽
        void   line_width(double w) { m_stroke.width(w); }
        // 获取线宽
        double line_width() const   { return m_stroke.width(); }

        // 设置点的半径
        void   point_radius(double r) { m_poly.point_radius(r); }
        // 获取点的半径
        double point_radius() const   { return m_poly.point_radius(); }

        // 虚函数，判断指定坐标是否在控件内部
        virtual bool in_rect(double x, double y) const;
        // 虚函数，处理鼠标按下事件
        virtual bool on_mouse_button_down(double x, double y);
        // 虚函数，处理鼠标释放事件
        virtual bool on_mouse_button_up(double x, double y);
        // 虚函数，处理鼠标移动事件
        virtual bool on_mouse_move(double x, double y, bool button_flag);
        // 虚函数，处理箭头键事件
        virtual bool on_arrow_keys(bool left, bool right, bool down, bool up);

        // 顶点源接口，返回路径数量
        unsigned num_paths() { return 6; };
        // 顶点源接口，重置指定路径
        void     rewind(unsigned path_id);
        // 顶点源接口，获取顶点坐标
        unsigned vertex(double* x, double* y);


    private:
        // 三次曲线对象
        curve3              m_curve;
        // 椭圆对象
        ellipse             m_ellipse;
        // 曲线描边对象
        conv_stroke<curve3> m_stroke;
        // 多边形控制对象
        polygon_ctrl_impl   m_poly;
        // 索引值
        unsigned            m_idx;
    };

//----------------------------------------------------------curve3_ctrl

    // curve3_ctrl 类的模板类，继承自 curve3_ctrl_impl
    template<class ColorT> class curve3_ctrl : public curve3_ctrl_impl
    {
    public:
        // 默认构造函数，初始化颜色为黑色
        curve3_ctrl() :
            m_color(rgba(0.0, 0.0, 0.0))
        {
        }
          
        // 设置曲线的线条颜色
        void line_color(const ColorT& c) { m_color = c; }
        // 获取指定索引的颜色
        const ColorT& color(unsigned i) const { return m_color; } 

    private:
        // 拷贝构造函数，私有化，防止拷贝
        curve3_ctrl(const curve3_ctrl<ColorT>&);
        // 赋值运算符重载，私有化，防止赋值
        const curve3_ctrl<ColorT>& operator = (const curve3_ctrl<ColorT>&);

        // 颜色对象
        ColorT m_color;
    };
}



#endif



// 这些代码行是 C/C++ 中的预处理指令，用于结束条件编译段落或关闭特定的条件编译块。
// } 是用来结束函数定义或代码块的大括号。
// #endif 是条件编译指令，用于结束条件编译块。在这里可能代表着某个条件编译的结束。
// 在 C/C++ 中，#endif 是与 #ifdef、#ifndef 或 #if 配对使用的指令，用于控制编译的条件性。
// 这些指令不产生实际的可执行代码，而是在编译预处理阶段进行处理。
```