# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\ctrl\agg_gamma_ctrl.h`

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
// class gamma_ctrl
//
//----------------------------------------------------------------------------

#ifndef AGG_GAMMA_CTRL_INCLUDED
#define AGG_GAMMA_CTRL_INCLUDED

// 包含基本定义，如宏定义和常量
#include "agg_basics.h"
// 包含用于伽马校正的样条函数
#include "agg_gamma_spline.h"
// 包含椭圆绘制相关的功能
#include "agg_ellipse.h"
// 包含用于描边转换的功能
#include "agg_conv_stroke.h"
// 包含用于文本绘制的功能
#include "agg_gsv_text.h"
// 包含仿射变换相关的功能
#include "agg_trans_affine.h"
// 包含 RGBA 颜色定义
#include "agg_color_rgba.h"
// 包含控件类的定义
#include "agg_ctrl.h"

// 命名空间 agg 中的声明
namespace agg
{
    //------------------------------------------------------------------------
    // Class that can be used to create an interactive control to set up 
    // gamma arrays.
    // 可用于创建交互式控件以设置伽马数组的类声明
    //------------------------------------------------------------------------
    class gamma_ctrl_impl : public ctrl
    {
    // 公共接口部分开始

    // 构造函数，初始化 gamma_ctrl_impl 对象，设定初始控制点坐标和是否翻转 y 轴
    gamma_ctrl_impl(double x1, double y1, double x2, double y2, bool flip_y=false);

    // 设置边框宽度，可以附加额外宽度
    void border_width(double t, double extra=0.0);

    // 设置曲线宽度
    void curve_width(double t)         { m_curve_width = t; }

    // 设置网格宽度
    void grid_width(double t)          { m_grid_width = t; }

    // 设置文本粗细
    void text_thickness(double t)      { m_text_thickness = t; }

    // 设置文本大小，可以设定高度和宽度
    void text_size(double h, double w=0.0);

    // 设置点的大小
    void point_size(double s)              { m_point_size = s; }

    // 事件处理器，如果相应事件发生则调用对应函数，返回是否需要重绘
    virtual bool in_rect(double x, double y) const;

    // 鼠标按下事件处理器，返回是否需要重绘
    virtual bool on_mouse_button_down(double x, double y);

    // 鼠标释放事件处理器，返回是否需要重绘
    virtual bool on_mouse_button_up(double x, double y);

    // 鼠标移动事件处理器，返回是否需要重绘
    virtual bool on_mouse_move(double x, double y, bool button_flag);

    // 方向键事件处理器，返回是否需要重绘
    virtual bool on_arrow_keys(bool left, bool right, bool down, bool up);

    // 切换活动控制点
    void change_active_point();

    // 设置 gamma_spline 的值
    void values(double kx1, double ky1, double kx2, double ky2);

    // 获取 gamma_spline 的值
    void values(double* kx1, double* ky1, double* kx2, double* ky2) const;

    // 获取 gamma 曲线的指针
    const unsigned char* gamma() const { return m_gamma_spline.gamma(); }

    // 计算给定 x 值对应的 y 值
    double y(double x) const { return m_gamma_spline.y(x); }

    // 重载函数调用操作符，返回给定 x 值对应的 y 值
    double operator() (double x) const { return m_gamma_spline.y(x); }

    // 获取 gamma_spline 对象的引用
    const gamma_spline& get_gamma_spline() const { return m_gamma_spline; }

    // 顶点源接口，返回路径数目
    unsigned num_paths() { return 7; }

    // 将索引重置为指定值
    void rewind(unsigned idx);

    // 返回顶点坐标
    unsigned vertex(double* x, double* y);

    // 私有方法声明部分开始

    // 计算样条框的边界
    void calc_spline_box();

    // 计算控制点
    void calc_points();

    // 计算值
    void calc_values();

    // 私有成员变量声明部分开始

    gamma_spline  m_gamma_spline;          // gamma_spline 对象
    double m_border_width;                 // 边框宽度
    double m_border_extra;                 // 额外边框宽度
    double m_curve_width;                  // 曲线宽度
    double m_grid_width;                   // 网格宽度
    double m_text_thickness;               // 文本粗细
    double m_point_size;                   // 点的大小
    double m_text_height;                  // 文本高度
    double m_text_width;                   // 文本宽度
    double m_xc1, m_yc1;                   // 控制点坐标
    double m_xc2, m_yc2;                   // 控制点坐标
    double m_xs1, m_ys1;                   // 控制点坐标
    double m_xs2, m_ys2;                   // 控制点坐标
    double m_xt1, m_yt1;                   // 控制点坐标
    double m_xt2, m_yt2;                   // 控制点坐标
    conv_stroke<gamma_spline> m_curve_poly;  // gamma_spline 的描边
    ellipse                   m_ellipse;     // 椭圆对象
    gsv_text                  m_text;        // 文本对象
    conv_stroke<gsv_text>     m_text_poly;   // 文本描边
    unsigned m_idx;                         // 索引
    unsigned m_vertex;                      // 顶点
    double   m_vx[32];                      // 顶点 x 坐标数组
    double   m_vy[32];                      // 顶点 y 坐标数组
    double   m_xp1, m_yp1;                  // 控制点坐标
    double   m_xp2, m_yp2;                  // 控制点坐标
    bool     m_p1_active;                   // 活动控制点标志
    unsigned m_mouse_point;                 // 鼠标点索引
    double   m_pdx;                         // 控制点移动增量
    double   m_pdy;                         // 控制点移动增量
};
    // 公共构造函数，初始化 gamma_ctrl 类的实例
    public:
        gamma_ctrl(double x1, double y1, double x2, double y2, bool flip_y=false) :
            // 调用 gamma_ctrl_impl 类的构造函数初始化
            gamma_ctrl_impl(x1, y1, x2, y2, flip_y),
            // 设置默认背景颜色为浅黄色
            m_background_color(rgba(1.0, 1.0, 0.9)),
            // 设置默认边框颜色为黑色
            m_border_color(rgba(0.0, 0.0, 0.0)),
            // 设置默认曲线颜色为黑色
            m_curve_color(rgba(0.0, 0.0, 0.0)),
            // 设置默认网格颜色为深黄色
            m_grid_color(rgba(0.2, 0.2, 0.0)),
            // 设置默认非活动点颜色为黑色
            m_inactive_pnt_color(rgba(0.0, 0.0, 0.0)),
            // 设置默认活动点颜色为红色
            m_active_pnt_color(rgba(1.0, 0.0, 0.0)),
            // 设置默认文本颜色为黑色
            m_text_color(rgba(0.0, 0.0, 0.0))
        {
            // 将颜色指针数组与各个颜色成员变量关联起来
            m_colors[0] = &m_background_color;
            m_colors[1] = &m_border_color;
            m_colors[2] = &m_curve_color;
            m_colors[3] = &m_grid_color;
            m_colors[4] = &m_inactive_pnt_color;
            m_colors[5] = &m_active_pnt_color;
            m_colors[6] = &m_text_color;
        }
    
        // 设置各种颜色的方法
        void background_color(const ColorT& c)   { m_background_color = c; }
        void border_color(const ColorT& c)       { m_border_color = c; }
        void curve_color(const ColorT& c)        { m_curve_color = c; }
        void grid_color(const ColorT& c)         { m_grid_color = c; }
        void inactive_pnt_color(const ColorT& c) { m_inactive_pnt_color = c; }
        void active_pnt_color(const ColorT& c)   { m_active_pnt_color = c; }
        void text_color(const ColorT& c)         { m_text_color = c; }
    
        // 获取指定索引的颜色
        const ColorT& color(unsigned i) const { return *m_colors[i]; }
    
    private:
        // 私有拷贝构造函数和赋值运算符，禁止拷贝和赋值
        gamma_ctrl(const gamma_ctrl<ColorT>&);
        const gamma_ctrl<ColorT>& operator = (const gamma_ctrl<ColorT>&);
    
        // 各种颜色成员变量
        ColorT  m_background_color;
        ColorT  m_border_color;
        ColorT  m_curve_color;
        ColorT  m_grid_color;
        ColorT  m_inactive_pnt_color;
        ColorT  m_active_pnt_color;
        ColorT  m_text_color;
    
        // 颜色指针数组，包含各种颜色的指针
        ColorT* m_colors[7];
    };
}


注释：


// 结束一个预处理器条件指令的代码块，这里匹配 #ifdef 或 #ifndef 指令



#endif


注释：


// 结束一个条件编译指令的代码块，这里匹配 #ifdef 或 #ifndef 指令的末尾
```