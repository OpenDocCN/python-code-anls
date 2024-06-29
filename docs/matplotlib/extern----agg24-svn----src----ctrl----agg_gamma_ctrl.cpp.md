# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\ctrl\agg_gamma_ctrl.cpp`

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
// class gamma_ctrl_impl
//
//----------------------------------------------------------------------------

#include <stdio.h>
#include "agg_math.h"
#include "ctrl/agg_gamma_ctrl.h"

namespace agg
{

    //------------------------------------------------------------------------
    // gamma_ctrl_impl 构造函数，初始化控件的位置和属性
    gamma_ctrl_impl::gamma_ctrl_impl(double x1, double y1, double x2, double y2, bool flip_y) :
        ctrl(x1, y1, x2, y2, flip_y),  // 调用基类 ctrl 的构造函数进行初始化
        m_border_width(2.0),           // 初始化边框宽度
        m_border_extra(0.0),           // 初始化额外边框宽度
        m_curve_width(2.0),            // 初始化曲线宽度
        m_grid_width(0.2),             // 初始化网格宽度
        m_text_thickness(1.5),         // 初始化文本线宽度
        m_point_size(5.0),             // 初始化点的尺寸
        m_text_height(9.0),            // 初始化文本高度
        m_text_width(0.0),             // 初始化文本宽度
        m_xc1(x1),                     // 控制点1 x 坐标
        m_yc1(y1),                     // 控制点1 y 坐标
        m_xc2(x2),                     // 控制点2 x 坐标
        m_yc2(y2 - m_text_height * 2.0), // 控制点2 y 坐标，考虑到文本高度
        m_xt1(x1),                     // 文本点1 x 坐标
        m_yt1(y2 - m_text_height * 2.0), // 文本点1 y 坐标，考虑到文本高度
        m_xt2(x2),                     // 文本点2 x 坐标
        m_yt2(y2),                     // 文本点2 y 坐标
        m_curve_poly(m_gamma_spline),  // 初始化曲线多边形
        m_text_poly(m_text),           // 初始化文本多边形
        m_idx(0),                      // 初始化索引
        m_vertex(0),                   // 初始化顶点
        m_p1_active(true),             // 设置控制点1为活跃状态
        m_mouse_point(0),              // 鼠标点位置初始化
        m_pdx(0.0),                    // 控制点 x 方向变化量
        m_pdy(0.0)                     // 控制点 y 方向变化量
    {
        calc_spline_box();             // 计算样条框
    }


    //------------------------------------------------------------------------
    // 计算样条框的位置
    void gamma_ctrl_impl::calc_spline_box()
    {
        m_xs1 = m_xc1 + m_border_width;       // 样条框左上角 x 坐标
        m_ys1 = m_yc1 + m_border_width;       // 样条框左上角 y 坐标
        m_xs2 = m_xc2 - m_border_width;       // 样条框右下角 x 坐标
        m_ys2 = m_yc2 - m_border_width * 0.5; // 样条框右下角 y 坐标，考虑到边框宽度
    }


    //------------------------------------------------------------------------
    // 计算控制点的位置
    void gamma_ctrl_impl::calc_points()
    {
        double kx1, ky1, kx2, ky2;
        m_gamma_spline.values(&kx1, &ky1, &kx2, &ky2);  // 获取样条函数的值
        m_xp1 = m_xs1 + (m_xs2 - m_xs1) * kx1 * 0.25;   // 控制点1 x 坐标
        m_yp1 = m_ys1 + (m_ys2 - m_ys1) * ky1 * 0.25;   // 控制点1 y 坐标
        m_xp2 = m_xs2 - (m_xs2 - m_xs1) * kx2 * 0.25;   // 控制点2 x 坐标
        m_yp2 = m_ys2 - (m_ys2 - m_ys1) * ky2 * 0.25;   // 控制点2 y 坐标
    }


    //------------------------------------------------------------------------
    // 计算样条函数的值
    void gamma_ctrl_impl::calc_values()
    {
        double kx1, ky1, kx2, ky2;

        kx1 = (m_xp1 - m_xs1) * 4.0 / (m_xs2 - m_xs1);   // 样条函数参数 kx1
        ky1 = (m_yp1 - m_ys1) * 4.0 / (m_ys2 - m_ys1);   // 样条函数参数 ky1
        kx2 = (m_xs2 - m_xp2) * 4.0 / (m_xs2 - m_xs1);   // 样条函数参数 kx2
        ky2 = (m_ys2 - m_yp2) * 4.0 / (m_ys2 - m_ys1);   // 样条函数参数 ky2
        m_gamma_spline.values(kx1, ky1, kx2, ky2);        // 设置样条函数的值
    }

} // namespace agg
    //------------------------------------------------------------------------
    void gamma_ctrl_impl::text_size(double h, double w) 
    { 
        // 设置文本高度和宽度
        m_text_width = w; 
        m_text_height = h; 
        // 计算文本的垂直位置
        m_yc2 = m_y2 - m_text_height * 2.0;
        m_yt1 = m_y2 - m_text_height * 2.0;
        // 重新计算样条框
        calc_spline_box();
    }
    
    
    //------------------------------------------------------------------------
    void gamma_ctrl_impl::border_width(double t, double extra)
    { 
        // 设置边框宽度和额外空间
        m_border_width = t; 
        m_border_extra = extra;
        // 重新计算样条框
        calc_spline_box(); 
    }
    
    //------------------------------------------------------------------------
    void gamma_ctrl_impl::values(double kx1, double ky1, double kx2, double ky2)
    {
        // 将参数传递给 gamma_spline 对象的 values 方法
        m_gamma_spline.values(kx1, ky1, kx2, ky2);
    }
    
    
    //------------------------------------------------------------------------
    void gamma_ctrl_impl::values(double* kx1, double* ky1, double* kx2, double* ky2) const
    {
        // 将参数传递给 gamma_spline 对象的 values 方法（常量版本）
        m_gamma_spline.values(kx1, ky1, kx2, ky2);
    }
    
    //------------------------------------------------------------------------
    void  gamma_ctrl_impl::rewind(unsigned idx)
    {
        // 设置路径索引
        m_idx = idx;
    }
    
    
    //------------------------------------------------------------------------
    unsigned gamma_ctrl_impl::vertex(double* x, double* y)
    {
        // 初始化命令为直线到
        unsigned cmd = path_cmd_line_to;
        // 根据当前索引选择路径绘制方式
        switch(m_idx)
        {
        case 0:
            // 在第一个路径段中，根据顶点计数确定命令
            if(m_vertex == 0) cmd = path_cmd_move_to;
            if(m_vertex >= 4) cmd = path_cmd_stop;
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
            m_vertex++;
            break;
    
        case 1:
            // 在第二个路径段中，根据顶点计数确定命令
            if(m_vertex == 0 || m_vertex == 4 || m_vertex == 8) cmd = path_cmd_move_to;
            if(m_vertex >= 12) cmd = path_cmd_stop;
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
            m_vertex++;
            break;
    
        case 2:
            // 在第三个路径段中，调用曲线多边形对象的顶点方法
            cmd = m_curve_poly.vertex(x, y);
            break;
    
        case 3:
            // 在第四个路径段中，根据顶点计数确定命令
            if(m_vertex == 0  || 
               m_vertex == 4  || 
               m_vertex == 8  ||
               m_vertex == 14) cmd = path_cmd_move_to;
            if(m_vertex >= 20) cmd = path_cmd_stop;
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
            m_vertex++;
            break;
    
        case 4:                 // Point1
        case 5:                 // Point2
            // 在第五或第六个路径段中，调用椭圆对象的顶点方法
            cmd = m_ellipse.vertex(x, y);
            break;
    
        case 6:
            // 在第七个路径段中，调用文本多边形对象的顶点方法
            cmd = m_text_poly.vertex(x, y);
            break;
    
        default:
            // 其它情况下，停止路径绘制
            cmd = path_cmd_stop;
            break;
        }
    
        // 如果命令不是停止命令，进行坐标变换
        if(!is_stop(cmd))
        {
            transform_xy(x, y);
        }
    
        return cmd;
    }
    
    
    
    //------------------------------------------------------------------------
    bool gamma_ctrl_impl::on_arrow_keys(bool left, bool right, bool down, bool up)
    {
        // 处理箭头键事件
        // 省略具体实现细节
    }
    {
        // 定义用于存储曲线控制点的增量值的变量
        double kx1, ky1, kx2, ky2;
        // 初始化返回值为 false
        bool ret = false;
        // 从 gamma_spline 对象获取当前控制点的值
        m_gamma_spline.values(&kx1, &ky1, &kx2, &ky2);
        // 如果第一个控制点处于活动状态
        if (m_p1_active)
        {
            // 如果 left 按钮按下，将第一个控制点的 x 值减小 0.005，并设置返回值为 true
            if (left)  { kx1 -= 0.005; ret = true; }
            // 如果 right 按钮按下，将第一个控制点的 x 值增加 0.005，并设置返回值为 true
            if (right) { kx1 += 0.005; ret = true; }
            // 如果 down 按钮按下，将第一个控制点的 y 值减小 0.005，并设置返回值为 true
            if (down)  { ky1 -= 0.005; ret = true; }
            // 如果 up 按钮按下，将第一个控制点的 y 值增加 0.005，并设置返回值为 true
            if (up)    { ky1 += 0.005; ret = true; }
        }
        else // 如果第二个控制点处于活动状态
        {
            // 如果 left 按钮按下，将第二个控制点的 x 值增加 0.005，并设置返回值为 true
            if (left)  { kx2 += 0.005; ret = true; }
            // 如果 right 按钮按下，将第二个控制点的 x 值减小 0.005，并设置返回值为 true
            if (right) { kx2 -= 0.005; ret = true; }
            // 如果 down 按钮按下，将第二个控制点的 y 值增加 0.005，并设置返回值为 true
            if (down)  { ky2 += 0.005; ret = true; }
            // 如果 up 按钮按下，将第二个控制点的 y 值减小 0.005，并设置返回值为 true
            if (up)    { ky2 -= 0.005; ret = true; }
        }
        // 如果有控制点发生变化，则更新 gamma_spline 对象中的控制点值
        if (ret)
        {
            m_gamma_spline.values(kx1, ky1, kx2, ky2);
        }
        // 返回操作是否有变化的标志
        return ret;
    }
    
    
    
    //------------------------------------------------------------------------
    void gamma_ctrl_impl::change_active_point()
    {
        // 切换活动控制点的状态
        m_p1_active = m_p1_active ? false : true;
    }
    
    
    
    
    //------------------------------------------------------------------------
    bool gamma_ctrl_impl::in_rect(double x, double y) const
    {
        // 将坐标进行反向转换
        inverse_transform_xy(&x, &y);
        // 检查坐标是否在矩形区域内
        return x >= m_x1 && x <= m_x2 && y >= m_y1 && y <= m_y2;
    }
    
    
    //------------------------------------------------------------------------
    bool gamma_ctrl_impl::on_mouse_button_down(double x, double y)
    {
        // 将坐标进行反向转换
        inverse_transform_xy(&x, &y);
        // 计算并更新控制点的位置
        calc_points();
    
        // 如果鼠标按下的位置在第一个控制点附近
        if (calc_distance(x, y, m_xp1, m_yp1) <= m_point_size + 1)
        {
            // 设置鼠标操作的控制点为第一个，并记录偏移量
            m_mouse_point = 1;
            m_pdx = m_xp1 - x;
            m_pdy = m_yp1 - y;
            // 激活第一个控制点
            m_p1_active = true;
            return true;
        }
    
        // 如果鼠标按下的位置在第二个控制点附近
        if (calc_distance(x, y, m_xp2, m_yp2) <= m_point_size + 1)
        {
            // 设置鼠标操作的控制点为第二个，并记录偏移量
            m_mouse_point = 2;
            m_pdx = m_xp2 - x;
            m_pdy = m_yp2 - y;
            // 激活第二个控制点
            m_p1_active = false;
            return true;
        }
    
        // 如果鼠标按下的位置不在任何控制点附近，返回 false
        return false;
    }
    
    
    //------------------------------------------------------------------------
    bool gamma_ctrl_impl::on_mouse_button_up(double, double)
    {
        // 如果鼠标按钮释放，将鼠标操作的控制点标志重置为 0，并返回 true
        if (m_mouse_point)
        {
            m_mouse_point = 0;
            return true;
        }
        // 否则返回 false
        return false;
    }
    
    
    //------------------------------------------------------------------------
    bool gamma_ctrl_impl::on_mouse_move(double x, double y, bool button_flag)
    {
        // 将坐标进行反向转换
        inverse_transform_xy(&x, &y);
        // 如果鼠标按钮未按下，执行鼠标按钮释放操作
        if (!button_flag)
        {
            return on_mouse_button_up(x, y);
        }
    
        // 如果鼠标正在操作第一个控制点
        if (m_mouse_point == 1)
        {
            // 更新第一个控制点的位置，并重新计算数值
            m_xp1 = x + m_pdx;
            m_yp1 = y + m_pdy;
            calc_values();
            return true;
        }
    
        // 如果鼠标正在操作第二个控制点
        if (m_mouse_point == 2)
        {
            // 更新第二个控制点的位置，并重新计算数值
            m_xp2 = x + m_pdx;
            m_yp2 = y + m_pdy;
            calc_values();
            return true;
        }
    
        // 如果鼠标未操作任何控制点，返回 false
        return false;
    }
}


注释：


# 结束一个代码块或函数的定义
```