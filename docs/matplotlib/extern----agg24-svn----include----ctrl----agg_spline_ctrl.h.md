# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\ctrl\agg_spline_ctrl.h`

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
// classes spline_ctrl_impl, spline_ctrl
//
//----------------------------------------------------------------------------

#ifndef AGG_SPLINE_CTRL_INCLUDED
#define AGG_SPLINE_CTRL_INCLUDED

#include "agg_basics.h"          // 包含基本的 Anti-Grain Geometry 库
#include "agg_ellipse.h"         // 包含椭圆绘制相关的头文件
#include "agg_bspline.h"         // 包含 B-样条曲线相关的头文件
#include "agg_conv_stroke.h"     // 包含路径描边转换器相关的头文件
#include "agg_path_storage.h"    // 包含路径存储相关的头文件
#include "agg_trans_affine.h"    // 包含仿射变换相关的头文件
#include "agg_color_rgba.h"      // 包含 RGBA 颜色相关的头文件
#include "agg_ctrl.h"            // 包含控件基类相关的头文件

namespace agg
{

    //------------------------------------------------------------------------
    // Class that can be used to create an interactive control to set up 
    // gamma arrays.
    //------------------------------------------------------------------------
    class spline_ctrl_impl : public ctrl
    {
    // 公有成员函数：spline_ctrl_impl 类的构造函数，初始化样条控制器实现
    spline_ctrl_impl(double x1, double y1, double x2, double y2, 
                     unsigned num_pnt, bool flip_y=false);

    // 设置边界宽度
    void border_width(double t, double extra=0.0);

    // 设置曲线宽度
    void curve_width(double t) { m_curve_width = t; }

    // 设置点大小
    void point_size(double s)  { m_point_size = s; }

    // 事件处理函数：检查指定坐标是否在控件矩形内
    virtual bool in_rect(double x, double y) const;

    // 事件处理函数：处理鼠标按下事件
    virtual bool on_mouse_button_down(double x, double y);

    // 事件处理函数：处理鼠标释放事件
    virtual bool on_mouse_button_up(double x, double y);

    // 事件处理函数：处理鼠标移动事件
    virtual bool on_mouse_move(double x, double y, bool button_flag);

    // 事件处理函数：处理箭头键事件
    virtual bool on_arrow_keys(bool left, bool right, bool down, bool up);

    // 设置活动点的索引
    void active_point(int i);

    // 返回样条数组的指针
    const double* spline()  const { return m_spline_values; }

    // 返回8位样条数组的指针
    const int8u*  spline8() const { return m_spline_values8; }

    // 根据给定的 x 值计算样条的 y 值
    double value(double x) const;

    // 设置指定索引处的样条值
    void   value(unsigned idx, double y);

    // 设置指定索引处的点的坐标
    void   point(unsigned idx, double x, double y);

    // 设置指定索引处的 x 坐标
    void   x(unsigned idx, double x) { m_xp[idx] = x; }

    // 设置指定索引处的 y 坐标
    void   y(unsigned idx, double y) { m_yp[idx] = y; }

    // 获取指定索引处的 x 坐标
    double x(unsigned idx) const { return m_xp[idx]; }

    // 获取指定索引处的 y 坐标
    double y(unsigned idx) const { return m_yp[idx]; }

    // 更新样条
    void  update_spline();

    // 获取路径数量
    unsigned num_paths() { return 5; }

    // 重置路径
    void     rewind(unsigned path_id);

    // 获取路径中的顶点坐标
    unsigned vertex(double* x, double* y);

private:
    // 计算样条框
    void calc_spline_box();

    // 计算曲线
    void calc_curve();

    // 计算指定索引处的 x 坐标
    double calc_xp(unsigned idx);

    // 计算指定索引处的 y 坐标
    double calc_yp(unsigned idx);

    // 设置指定索引处的 x 坐标
    void set_xp(unsigned idx, double val);

    // 设置指定索引处的 y 坐标
    void set_yp(unsigned idx, double val);

    unsigned m_num_pnt;                      // 控制点数目
    double   m_xp[32];                       // 控制点 x 坐标数组
    double   m_yp[32];                       // 控制点 y 坐标数组
    bspline  m_spline;                       // B样条对象
    double   m_spline_values[256];           // 样条值数组
    int8u    m_spline_values8[256];          // 8位样条值数组
    double   m_border_width;                 // 边界宽度
    double   m_border_extra;                 // 额外边界
    double   m_curve_width;                  // 曲线宽度
    double   m_point_size;                   // 点大小
    double   m_xs1;                          // 起始点 x 坐标
    double   m_ys1;                          // 起始点 y 坐标
    double   m_xs2;                          // 结束点 x 坐标
    double   m_ys2;                          // 结束点 y 坐标
    path_storage              m_curve_pnt;    // 路径存储
    conv_stroke<path_storage> m_curve_poly;   // 路径描边
    ellipse                   m_ellipse;      // 椭圆对象
    unsigned m_idx;                          // 索引
    unsigned m_vertex;                       // 顶点
    double   m_vx[32];                       // 顶点 x 坐标数组
    double   m_vy[32];                       // 顶点 y 坐标数组
    int      m_active_pnt;                   // 活动点索引
    int      m_move_pnt;                     // 移动点索引
    double   m_pdx;                          // x 方向偏移量
    double   m_pdy;                          // y 方向偏移量
    const trans_affine* m_mtx;               // 变换矩阵指针
};
    // 公有构造函数，初始化样条控制器对象
    public:
        spline_ctrl(double x1, double y1, double x2, double y2, 
                    unsigned num_pnt, bool flip_y=false) :
            // 调用基类的构造函数来初始化样条控制器实现
            spline_ctrl_impl(x1, y1, x2, y2, num_pnt, flip_y),
            // 初始化背景色为浅黄色
            m_background_color(rgba(1.0, 1.0, 0.9)),
            // 初始化边框色为黑色
            m_border_color(rgba(0.0, 0.0, 0.0)),
            // 初始化曲线色为黑色
            m_curve_color(rgba(0.0, 0.0, 0.0)),
            // 初始化非活动点颜色为黑色
            m_inactive_pnt_color(rgba(0.0, 0.0, 0.0)),
            // 初始化活动点颜色为红色
            m_active_pnt_color(rgba(1.0, 0.0, 0.0))
        {
            // 将颜色对象的指针存入颜色指针数组
            m_colors[0] = &m_background_color;
            m_colors[1] = &m_border_color;
            m_colors[2] = &m_curve_color;
            m_colors[3] = &m_inactive_pnt_color;
            m_colors[4] = &m_active_pnt_color;
        }

        // 设置背景色
        void background_color(const ColorT& c)   { m_background_color = c; }
        // 设置边框色
        void border_color(const ColorT& c)       { m_border_color = c; }
        // 设置曲线色
        void curve_color(const ColorT& c)        { m_curve_color = c; }
        // 设置非活动点颜色
        void inactive_pnt_color(const ColorT& c) { m_inactive_pnt_color = c; }
        // 设置活动点颜色
        void active_pnt_color(const ColorT& c)   { m_active_pnt_color = c; }
        // 返回指定索引的颜色对象的引用
        const ColorT& color(unsigned i) const { return *m_colors[i]; } 

    private:
        // 私有复制构造函数，禁止复制构造
        spline_ctrl(const spline_ctrl<ColorT>&);
        // 私有赋值运算符重载，禁止赋值操作
        const spline_ctrl<ColorT>& operator = (const spline_ctrl<ColorT>&);

        // 背景色对象
        ColorT  m_background_color;
        // 边框色对象
        ColorT  m_border_color;
        // 曲线色对象
        ColorT  m_curve_color;
        // 非活动点颜色对象
        ColorT  m_inactive_pnt_color;
        // 活动点颜色对象
        ColorT  m_active_pnt_color;
        // 颜色指针数组，存储各颜色对象的指针
        ColorT* m_colors[5];
    };
}


注释：

// 结束一个条件编译指令或者函数定义的闭合



#endif


注释：

// 结束一个条件编译指令的闭合
```