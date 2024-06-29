# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\ctrl\agg_bezier_ctrl.cpp`

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
// classes bezier_ctrl_impl, bezier_ctrl
//
//----------------------------------------------------------------------------

#include <string.h>
#include <stdio.h>
#include "ctrl/agg_bezier_ctrl.h"

namespace agg
{

    //------------------------------------------------------------------------
    // bezier_ctrl_impl 类的构造函数，初始化成员变量和多边形控制点
    bezier_ctrl_impl::bezier_ctrl_impl() :
        ctrl(0,0,1,1,false),          // 调用基类 ctrl 的构造函数，设置默认参数
        m_stroke(m_curve),            // 用 m_curve 初始化 m_stroke
        m_poly(4, 5.0),               // 创建一个 4 个点的多边形，边宽为 5.0
        m_idx(0)                      // 初始化索引 m_idx 为 0
    {
        m_poly.in_polygon_check(false); // 设置多边形的内部检查为 false
        // 设置多边形的四个顶点坐标
        m_poly.xn(0) = 100.0;
        m_poly.yn(0) =   0.0;
        m_poly.xn(1) = 100.0;
        m_poly.yn(1) =  50.0;
        m_poly.xn(2) =  50.0;
        m_poly.yn(2) = 100.0;
        m_poly.xn(3) =   0.0;
        m_poly.yn(3) = 100.0;
    }


    //------------------------------------------------------------------------
    // 设置 Bezier 曲线的控制点坐标
    void bezier_ctrl_impl::curve(double x1, double y1, 
                                 double x2, double y2, 
                                 double x3, double y3,
                                 double x4, double y4)
    {
        // 分别设置四个控制点的坐标
        m_poly.xn(0) = x1;
        m_poly.yn(0) = y1;
        m_poly.xn(1) = x2;
        m_poly.yn(1) = y2;
        m_poly.xn(2) = x3;
        m_poly.yn(2) = y3;
        m_poly.xn(3) = x4;
        m_poly.yn(3) = y4;
        // 调用无参数的 curve() 函数进行曲线更新
        curve();
    }

    //------------------------------------------------------------------------
    // 返回当前的 Bezier 曲线对象 m_curve
    curve4& bezier_ctrl_impl::curve()
    {
        // 使用多边形的四个顶点初始化 m_curve 对象
        m_curve.init(m_poly.xn(0), m_poly.yn(0),
                     m_poly.xn(1), m_poly.yn(1),
                     m_poly.xn(2), m_poly.yn(2),
                     m_poly.xn(3), m_poly.yn(3));
        // 返回初始化后的 m_curve 曲线对象的引用
        return m_curve;
    }

    //------------------------------------------------------------------------
    // 设置重新访问指定索引的状态，这里未完全提供代码
    void bezier_ctrl_impl::rewind(unsigned idx)
    {
        // 将当前索引复制给成员变量 m_idx
        m_idx = idx;
    
        // 设置曲线的近似比例为当前比例
        m_curve.approximation_scale(scale());
    
        // 根据索引选择执行相应的操作
        switch(idx)
        {
        default:
        case 0:                 // 控制线 1
            // 初始化曲线对象，使用前两个控制点
            m_curve.init(m_poly.xn(0),  m_poly.yn(0), 
                        (m_poly.xn(0) + m_poly.xn(1)) * 0.5,
                        (m_poly.yn(0) + m_poly.yn(1)) * 0.5,
                        (m_poly.xn(0) + m_poly.xn(1)) * 0.5,
                        (m_poly.yn(0) + m_poly.yn(1)) * 0.5,
                         m_poly.xn(1),  m_poly.yn(1));
            // 重置描边对象
            m_stroke.rewind(0);
            break;
    
        case 1:                 // 控制线 2
            // 初始化曲线对象，使用第三四个控制点
            m_curve.init(m_poly.xn(2),  m_poly.yn(2), 
                        (m_poly.xn(2) + m_poly.xn(3)) * 0.5,
                        (m_poly.yn(2) + m_poly.yn(3)) * 0.5,
                        (m_poly.xn(2) + m_poly.xn(3)) * 0.5,
                        (m_poly.yn(2) + m_poly.yn(3)) * 0.5,
                         m_poly.xn(3),  m_poly.yn(3));
            // 重置描边对象
            m_stroke.rewind(0);
            break;
    
        case 2:                 // 曲线本身
            // 初始化曲线对象，使用所有四个控制点
            m_curve.init(m_poly.xn(0), m_poly.yn(0), 
                         m_poly.xn(1), m_poly.yn(1),
                         m_poly.xn(2), m_poly.yn(2),
                         m_poly.xn(3), m_poly.yn(3));
            // 重置描边对象
            m_stroke.rewind(0);
            break;
    
        case 3:                 // 点 1
            // 初始化椭圆对象，使用第一个控制点
            m_ellipse.init(m_poly.xn(0), m_poly.yn(0), point_radius(), point_radius(), 20);
            // 重置椭圆对象
            m_ellipse.rewind(0);
            break;
    
        case 4:                 // 点 2
            // 初始化椭圆对象，使用第二个控制点
            m_ellipse.init(m_poly.xn(1), m_poly.yn(1), point_radius(), point_radius(), 20);
            // 重置椭圆对象
            m_ellipse.rewind(0);
            break;
    
        case 5:                 // 点 3
            // 初始化椭圆对象，使用第三个控制点
            m_ellipse.init(m_poly.xn(2), m_poly.yn(2), point_radius(), point_radius(), 20);
            // 重置椭圆对象
            m_ellipse.rewind(0);
            break;
    
        case 6:                 // 点 4
            // 初始化椭圆对象，使用第四个控制点
            m_ellipse.init(m_poly.xn(3), m_poly.yn(3), point_radius(), point_radius(), 20);
            // 重置椭圆对象
            m_ellipse.rewind(0);
            break;
        }
    }
    
    
    //------------------------------------------------------------------------
    unsigned bezier_ctrl_impl::vertex(double* x, double* y)
    {
        // 初始化路径命令为停止状态
        unsigned cmd = path_cmd_stop;
    
        // 根据索引选择执行相应的操作
        switch(m_idx)
        {
        case 0:
        case 1:
        case 2:
            // 获取曲线对象的顶点坐标
            cmd = m_stroke.vertex(x, y);
            break;
    
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
            // 获取椭圆对象的顶点坐标
            cmd = m_ellipse.vertex(x, y);
            break;
        }
    
        // 如果命令不是停止状态，则对顶点坐标进行变换
        if(!is_stop(cmd))
        {
            transform_xy(x, y);
        }
        return cmd;
    }
    
    
    
    //------------------------------------------------------------------------
    bool bezier_ctrl_impl::in_rect(double x, double y) const
    {
        // 始终返回 false，表示点不在矩形内
        return false;
    }
    
    
    //------------------------------------------------------------------------
    // 当鼠标按下时的回调函数，转换鼠标坐标后调用曲线对象的鼠标按下事件处理函数
    bool bezier_ctrl_impl::on_mouse_button_down(double x, double y)
    {
        // 反转换坐标系中的鼠标坐标
        inverse_transform_xy(&x, &y);
        // 调用曲线对象的鼠标按下事件处理函数，并返回结果
        return m_poly.on_mouse_button_down(x, y);
    }
    
    
    //------------------------------------------------------------------------
    // 当鼠标移动时的回调函数，转换鼠标坐标后调用曲线对象的鼠标移动事件处理函数
    bool bezier_ctrl_impl::on_mouse_move(double x, double y, bool button_flag)
    {
        // 反转换坐标系中的鼠标坐标
        inverse_transform_xy(&x, &y);
        // 调用曲线对象的鼠标移动事件处理函数，并返回结果
        return m_poly.on_mouse_move(x, y, button_flag);
    }
    
    
    //------------------------------------------------------------------------
    // 当鼠标释放按键时的回调函数，直接调用曲线对象的鼠标释放按键事件处理函数
    bool bezier_ctrl_impl::on_mouse_button_up(double x, double y)
    {
        // 直接调用曲线对象的鼠标释放按键事件处理函数，并返回结果
        return m_poly.on_mouse_button_up(x, y);
    }
    
    
    //------------------------------------------------------------------------
    // 处理箭头键事件的回调函数，直接调用曲线对象的箭头键事件处理函数
    bool bezier_ctrl_impl::on_arrow_keys(bool left, bool right, bool down, bool up)
    {
        // 直接调用曲线对象的箭头键事件处理函数，并返回结果
        return m_poly.on_arrow_keys(left, right, down, up);
    }
    
    
    //------------------------------------------------------------------------
    // 曲线控制器的构造函数，初始化成员变量和曲线形状
    curve3_ctrl_impl::curve3_ctrl_impl() :
        ctrl(0,0,1,1,false),        // 调用基类的构造函数进行初始化
        m_stroke(m_curve),          // 初始化曲线的描绘对象
        m_poly(3, 5.0),             // 初始化曲线的多边形对象，设置边数和缩放因子
        m_idx(0)                    // 初始化索引变量
    {
        // 关闭多边形对象的多边形检查功能
        m_poly.in_polygon_check(false);
        // 设置多边形对象的顶点坐标
        m_poly.xn(0) = 100.0;
        m_poly.yn(0) =   0.0;
        m_poly.xn(1) = 100.0;
        m_poly.yn(1) =  50.0;
        m_poly.xn(2) =  50.0;
        m_poly.yn(2) = 100.0;
    }
    
    
    //------------------------------------------------------------------------
    // 设置曲线的控制点坐标并调用曲线绘制函数
    void curve3_ctrl_impl::curve(double x1, double y1, 
                                 double x2, double y2, 
                                 double x3, double y3)
    {
        // 设置多边形对象的控制点坐标
        m_poly.xn(0) = x1;
        m_poly.yn(0) = y1;
        m_poly.xn(1) = x2;
        m_poly.yn(1) = y2;
        m_poly.xn(2) = x3;
        m_poly.yn(2) = y3;
        // 调用曲线对象的绘制函数
        curve();
    }
    
    //------------------------------------------------------------------------
    // 获取当前曲线对象的引用并初始化曲线参数
    curve3& curve3_ctrl_impl::curve()
    {
        // 初始化曲线对象的参数
        m_curve.init(m_poly.xn(0), m_poly.yn(0),
                     m_poly.xn(1), m_poly.yn(1),
                     m_poly.xn(2), m_poly.yn(2));
        // 返回曲线对象的引用
        return m_curve;
    }
    
    //------------------------------------------------------------------------
    // 处理重置操作的函数，重置曲线的索引值
    void curve3_ctrl_impl::rewind(unsigned idx)
    {
        // 重置曲线对象的索引值
        m_idx = idx;
    }
    {
        m_idx = idx;  // 将参数 idx 的值赋给成员变量 m_idx
    
        switch(idx)
        {
        default:
        case 0:                 // 控制线
            // 初始化曲线对象 m_curve，使用第一个控制点和中间点的坐标作为起点和控制点，第二个控制点作为终点
            m_curve.init(m_poly.xn(0),  m_poly.yn(0), 
                        (m_poly.xn(0) + m_poly.xn(1)) * 0.5,
                        (m_poly.yn(0) + m_poly.yn(1)) * 0.5,
                         m_poly.xn(1),  m_poly.yn(1));
            m_stroke.rewind(0);  // 重置 m_stroke 对象
            break;
    
        case 1:                 // 第二个控制线
            // 初始化曲线对象 m_curve，使用第二个控制点和中间点的坐标作为起点和控制点，第三个控制点作为终点
            m_curve.init(m_poly.xn(1),  m_poly.yn(1), 
                        (m_poly.xn(1) + m_poly.xn(2)) * 0.5,
                        (m_poly.yn(1) + m_poly.yn(2)) * 0.5,
                         m_poly.xn(2),  m_poly.yn(2));
            m_stroke.rewind(0);  // 重置 m_stroke 对象
            break;
    
        case 2:                 // 曲线本身
            // 初始化曲线对象 m_curve，使用第一个、第二个和第三个控制点的坐标
            m_curve.init(m_poly.xn(0), m_poly.yn(0), 
                         m_poly.xn(1), m_poly.yn(1),
                         m_poly.xn(2), m_poly.yn(2));
            m_stroke.rewind(0);  // 重置 m_stroke 对象
            break;
    
        case 3:                 // 点1
            // 初始化椭圆对象 m_ellipse，使用第一个控制点的坐标，并设置半径和细分数
            m_ellipse.init(m_poly.xn(0), m_poly.yn(0), point_radius(), point_radius(), 20);
            m_ellipse.rewind(0);  // 重置 m_ellipse 对象
            break;
    
        case 4:                 // 点2
            // 初始化椭圆对象 m_ellipse，使用第二个控制点的坐标，并设置半径和细分数
            m_ellipse.init(m_poly.xn(1), m_poly.yn(1), point_radius(), point_radius(), 20);
            m_ellipse.rewind(0);  // 重置 m_ellipse 对象
            break;
    
        case 5:                 // 点3
            // 初始化椭圆对象 m_ellipse，使用第三个控制点的坐标，并设置半径和细分数
            m_ellipse.init(m_poly.xn(2), m_poly.yn(2), point_radius(), point_radius(), 20);
            m_ellipse.rewind(0);  // 重置 m_ellipse 对象
            break;
        }
    }
    
    
    //------------------------------------------------------------------------
    unsigned curve3_ctrl_impl::vertex(double* x, double* y)
    {
        unsigned cmd = path_cmd_stop;  // 设置默认命令为停止
        switch(m_idx)
        {
        case 0:
        case 1:
        case 2:
            // 根据当前索引 m_idx 调用 m_stroke 对象的 vertex 方法获取顶点命令
            cmd = m_stroke.vertex(x, y);
            break;
    
        case 3:
        case 4:
        case 5:
        case 6:
            // 根据当前索引 m_idx 调用 m_ellipse 对象的 vertex 方法获取顶点命令
            cmd = m_ellipse.vertex(x, y);
            break;
        }
    
        // 如果命令不是停止命令，则对顶点坐标进行变换
        if(!is_stop(cmd))
        {
            transform_xy(x, y);  // 调用 transform_xy 方法进行坐标变换
        }
        return cmd;  // 返回顶点命令
    }
    
    
    
    //------------------------------------------------------------------------
    bool curve3_ctrl_impl::in_rect(double x, double y) const
    {
        return false;  // 无条件返回假，表示点 (x, y) 不在矩形内
    }
    
    
    //------------------------------------------------------------------------
    bool curve3_ctrl_impl::on_mouse_button_down(double x, double y)
    {
        inverse_transform_xy(&x, &y);  // 调用 inverse_transform_xy 方法反向变换鼠标点击坐标
        return m_poly.on_mouse_button_down(x, y);  // 调用 m_poly 对象的 on_mouse_button_down 方法处理鼠标按下事件
    }
    
    
    //------------------------------------------------------------------------
    bool curve3_ctrl_impl::on_mouse_move(double x, double y, bool button_flag)
    {
        inverse_transform_xy(&x, &y);  // 调用 inverse_transform_xy 方法反向变换鼠标移动坐标
        return m_poly.on_mouse_move(x, y, button_flag);  // 调用 m_poly 对象的 on_mouse_move 方法处理鼠标移动事件
    }
    
    
    //------------------------------------------------------------------------
    bool curve3_ctrl_impl::on_mouse_button_up(double x, double y)
    {
        // 调用 m_poly 对象的 on_mouse_button_up 方法，并返回其结果
        return m_poly.on_mouse_button_up(x, y);
    }


    //------------------------------------------------------------------------
    bool curve3_ctrl_impl::on_arrow_keys(bool left, bool right, bool down, bool up)
    {
        // 调用 m_poly 对象的 on_arrow_keys 方法，并返回其结果
        return m_poly.on_arrow_keys(left, right, down, up);
    }
}



# 这行代码关闭了一个代码块，与某个前面的 "{" 对应，用于结束一个代码块的定义或范围。
```