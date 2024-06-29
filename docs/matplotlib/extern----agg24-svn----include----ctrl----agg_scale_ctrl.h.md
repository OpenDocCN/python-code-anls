# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\ctrl\agg_scale_ctrl.h`

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
// classes scale_ctrl_impl, scale_ctrl
//
//----------------------------------------------------------------------------

#ifndef AGG_SCALE_CTRL_INCLUDED
#define AGG_SCALE_CTRL_INCLUDED

#include "agg_basics.h"
#include "agg_math.h"
#include "agg_ellipse.h"
#include "agg_trans_affine.h"
#include "agg_color_rgba.h"
#include "agg_ctrl.h"

namespace agg
{

    //------------------------------------------------------------------------
    // 控件的具体实现类，继承自控件基类 ctrl
    class scale_ctrl_impl : public ctrl
    {
        // 移动状态的枚举
        enum move_e
        {
            move_nothing,
            move_value1,
            move_value2,
            move_slider
        };

    public:
        // 构造函数，初始化控件的位置和大小，是否翻转 Y 轴
        scale_ctrl_impl(double x1, double y1, double x2, double y2, bool flip_y=false);

        // 设置边框的厚度和额外的空间
        void border_thickness(double t, double extra=0.0);

        // 调整控件的大小和位置
        void resize(double x1, double y1, double x2, double y2);
        
        // 获取和设置最小增量值
        double min_delta() const { return m_min_d; }
        void min_delta(double d) { m_min_d = d; }
        
        // 获取和设置第一个值
        double value1() const { return m_value1; }
        void value1(double value);

        // 获取和设置第二个值
        double value2() const { return m_value2; }
        void value2(double value);

        // 移动控件的某个部分
        void move(double d);

        // 判断坐标 (x, y) 是否在控件内部
        virtual bool in_rect(double x, double y) const;

        // 处理鼠标按下事件，返回是否处理了该事件
        virtual bool on_mouse_button_down(double x, double y);

        // 处理鼠标释放事件，返回是否处理了该事件
        virtual bool on_mouse_button_up(double x, double y);

        // 处理鼠标移动事件，返回是否处理了该事件
        virtual bool on_mouse_move(double x, double y, bool button_flag);

        // 处理方向键事件，返回是否处理了该事件
        virtual bool on_arrow_keys(bool left, bool right, bool down, bool up);

        // 顶点源接口，返回路径数目
        unsigned num_paths() { return 5; };

        // 准备迭代指定路径的顶点
        void rewind(unsigned path_id);

        // 获取下一个顶点的坐标
        unsigned vertex(double* x, double* y);

    private:
        // 计算控件的边界框
        void calc_box();

        double   m_border_thickness;  // 边框的厚度
        double   m_border_extra;      // 边框的额外空间
        double   m_value1;            // 第一个值
        double   m_value2;            // 第二个值
        double   m_min_d;             // 最小增量
        double   m_xs1;               // 开始点 x 坐标
        double   m_ys1;               // 开始点 y 坐标
        double   m_xs2;               // 结束点 x 坐标
        double   m_ys2;               // 结束点 y 坐标
        double   m_pdx;               // 位移 x 增量
        double   m_pdy;               // 位移 y 增量
        move_e   m_move_what;         // 当前移动状态
        double   m_vx[32];            // 顶点 x 坐标数组
        double   m_vy[32];            // 顶点 y 坐标数组

        ellipse  m_ellipse;           // 椭圆对象

        unsigned m_idx;               // 索引
        unsigned m_vertex;            // 顶点

    };

} // namespace agg

#endif // AGG_SCALE_CTRL_INCLUDED
//----------------------------------------------------------------------------
    // 定义一个模板类 scale_ctrl，继承自 scale_ctrl_impl 类
    template<class ColorT> class scale_ctrl : public scale_ctrl_impl
    {
    public:
        // 构造函数，初始化控件的位置和颜色，可以选择是否翻转 y 轴
        scale_ctrl(double x1, double y1, double x2, double y2, bool flip_y=false) :
            scale_ctrl_impl(x1, y1, x2, y2, flip_y),
            // 初始化背景颜色为浅橙色
            m_background_color(rgba(1.0, 0.9, 0.8)),
            // 初始化边框颜色为黑色
            m_border_color(rgba(0.0, 0.0, 0.0)),
            // 初始化指针颜色为深红色，带有部分透明度
            m_pointers_color(rgba(0.8, 0.0, 0.0, 0.8)),
            // 初始化滑块颜色为较深的棕色，带有一定透明度
            m_slider_color(rgba(0.2, 0.1, 0.0, 0.6))
        {
            // 设置颜色数组中各个位置对应的颜色指针
            m_colors[0] = &m_background_color;
            m_colors[1] = &m_border_color;
            m_colors[2] = &m_pointers_color;
            m_colors[3] = &m_pointers_color;
            m_colors[4] = &m_slider_color;
        }
    
        // 设置背景颜色的方法
        void background_color(const ColorT& c) { m_background_color = c; }
        // 设置边框颜色的方法
        void border_color(const ColorT& c)     { m_border_color = c; }
        // 设置指针颜色的方法
        void pointers_color(const ColorT& c)   { m_pointers_color = c; }
        // 设置滑块颜色的方法
        void slider_color(const ColorT& c)     { m_slider_color = c; }
    
        // 获取指定位置颜色的方法
        const ColorT& color(unsigned i) const { return *m_colors[i]; } 
    
    private:
        // 禁止拷贝构造函数和赋值运算符的私有化
        scale_ctrl(const scale_ctrl<ColorT>&);
        const scale_ctrl<ColorT>& operator = (const scale_ctrl<ColorT>&);
    
        // 四个控件颜色成员变量
        ColorT m_background_color;
        ColorT m_border_color;
        ColorT m_pointers_color;
        ColorT m_slider_color;
        // 指向颜色数组的指针
        ColorT* m_colors[5];
    };
}

这行代码表示一个代码块的结束。在C或C++中，`}`符号用于结束一个函数或者一个代码块的定义。


#endif

这行代码通常在C或C++的头文件中使用，用于结束条件编译指令`#ifdef`或`#ifndef`。`#endif`指示条件编译的结束，之后的代码将根据条件是否定义而编译或者忽略。
```