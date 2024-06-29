# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\ctrl\agg_scale_ctrl.cpp`

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

#include "ctrl/agg_scale_ctrl.h"

namespace agg
{

    //------------------------------------------------------------------------
    // scale_ctrl_impl 构造函数
    // 参数 x1, y1, x2, y2: 控件的边界框坐标
    // 参数 flip_y: 是否翻转 y 轴
    scale_ctrl_impl::scale_ctrl_impl(double x1, double y1, 
                                     double x2, double y2, bool flip_y) :
        ctrl(x1, y1, x2, y2, flip_y),  // 调用基类 ctrl 的构造函数
        m_border_thickness(1.0),      // 初始化边框厚度
        // 根据控件边界框长宽比选择合适的边框额外距离
        m_border_extra((fabs(x2 - x1) > fabs(y2 - y1)) ? (y2 - y1) / 2 : (x2 - x1) / 2),
        m_pdx(0.0),                   // 初始化平移量 x
        m_pdy(0.0),                   // 初始化平移量 y
        m_move_what(move_nothing),    // 初始化移动状态为无
        m_value1(0.3),                // 初始化值1为0.3
        m_value2(0.7),                // 初始化值2为0.7
        m_min_d(0.01)                 // 初始化最小增量为0.01
    {
        calc_box();  // 计算控件内部框的边界
    }


    //------------------------------------------------------------------------
    // 计算控件内部框的边界
    void scale_ctrl_impl::calc_box()
    {
        m_xs1 = m_x1 + m_border_thickness;  // 左边界 x 坐标
        m_ys1 = m_y1 + m_border_thickness;  // 上边界 y 坐标
        m_xs2 = m_x2 - m_border_thickness;  // 右边界 x 坐标
        m_ys2 = m_y2 - m_border_thickness;  // 下边界 y 坐标
    }


    //------------------------------------------------------------------------
    // 设置边框厚度和额外距离
    // 参数 t: 边框厚度
    // 参数 extra: 额外距离
    void scale_ctrl_impl::border_thickness(double t, double extra)
    { 
        m_border_thickness = t;  // 设置边框厚度
        m_border_extra = extra;  // 设置额外距离
        calc_box();              // 重新计算控件内部框的边界
    }


    //------------------------------------------------------------------------
    // 调整控件大小
    // 参数 x1, y1, x2, y2: 控件的新边界框坐标
    void scale_ctrl_impl::resize(double x1, double y1, double x2, double y2)
    {
        m_x1 = x1;  // 更新控件左边界 x 坐标
        m_y1 = y1;  // 更新控件上边界 y 坐标
        m_x2 = x2;  // 更新控件右边界 x 坐标
        m_y2 = y2;  // 更新控件下边界 y 坐标
        calc_box();  // 重新计算控件内部框的边界
        // 根据新边界框长宽比选择合适的边框额外距离
        m_border_extra = (fabs(x2 - x1) > fabs(y2 - y1)) ? 
                            (y2 - y1) / 2 : 
                            (x2 - x1) / 2;
    }


    //------------------------------------------------------------------------
    // 设置值1，确保在 0 到 1 之间，并且与值2之间的差距至少为最小增量
    // 参数 value: 设置的值
    void scale_ctrl_impl::value1(double value) 
    { 
        if(value < 0.0) value = 0.0;                // 如果值小于0，则设置为0
        if(value > 1.0) value = 1.0;                // 如果值大于1，则设置为1
        if(m_value2 - value < m_min_d) value = m_value2 - m_min_d;  // 如果与值2的差距小于最小增量，则设置为合适的值
        m_value1 = value;                          // 设置值1
    }
    {
        // 对传入的值进行范围限制，确保其在 [0.0, 1.0] 之间
        if(value < 0.0) value = 0.0;
        if(value > 1.0) value = 1.0;
        // 如果加上当前值后小于最小差值，则调整为当前值加上最小差值
        if(m_value1 + value < m_min_d) value = m_value1 + m_min_d;
        // 更新 m_value2 的值为传入的 value
        m_value2 = value;
    }
    
    
    //------------------------------------------------------------------------
    void scale_ctrl_impl::move(double d)
    {
        // 将 m_value1 和 m_value2 增加相同的增量 d
        m_value1 += d;
        m_value2 += d;
        // 如果 m_value1 小于 0.0，则调整 m_value1 和 m_value2
        if(m_value1 < 0.0)
        {
            m_value2 -= m_value1;  // 减去 m_value1，以确保 m_value2 保持在 [0.0, 1.0] 范围内
            m_value1 = 0.0;  // 将 m_value1 设置为 0.0
        }
        // 如果 m_value2 大于 1.0，则调整 m_value1 和 m_value2
        if(m_value2 > 1.0)
        {
            m_value1 -= m_value2 - 1.0;  // 减去超出的部分，以确保 m_value1 和 m_value2 保持在 [0.0, 1.0] 范围内
            m_value2 = 1.0;  // 将 m_value2 设置为 1.0
        }
    }
    
    
    //------------------------------------------------------------------------
    void scale_ctrl_impl::rewind(unsigned idx)
    {
        // 略，该方法似乎缺少实现，应该是错误的或者未完整的方法定义
    }
    
    
    //------------------------------------------------------------------------
    unsigned scale_ctrl_impl::vertex(double* x, double* y)
    {
        // 初始化命令为直线路径
        unsigned cmd = path_cmd_line_to;
        // 根据 m_idx 的不同情况进行处理
        switch(m_idx)
        {
        case 0:
        case 4:
            // 对于 m_idx 为 0 或 4 时的处理
            if(m_vertex == 0) cmd = path_cmd_move_to;  // 如果是第一个顶点，设置为移动到命令
            if(m_vertex >= 4) cmd = path_cmd_stop;  // 如果超过了顶点数量，设置为停止命令
            *x = m_vx[m_vertex];  // 获取当前顶点的 x 坐标
            *y = m_vy[m_vertex];  // 获取当前顶点的 y 坐标
            m_vertex++;  // 增加顶点计数
            break;
    
        case 1:
            // 对于 m_idx 为 1 时的处理
            if(m_vertex == 0 || m_vertex == 4) cmd = path_cmd_move_to;  // 如果是第一个或第五个顶点，设置为移动到命令
            if(m_vertex >= 8) cmd = path_cmd_stop;  // 如果超过了顶点数量，设置为停止命令
            *x = m_vx[m_vertex];  // 获取当前顶点的 x 坐标
            *y = m_vy[m_vertex];  // 获取当前顶点的 y 坐标
            m_vertex++;  // 增加顶点计数
            break;
    
        case 2:
        case 3:
            // 对于 m_idx 为 2 或 3 时的处理，调用椭圆对象的顶点生成方法
            cmd = m_ellipse.vertex(x, y);
            break;
    
        default:
            // 其他情况，设置为停止命令
            cmd = path_cmd_stop;
            break;
        }
    
        // 如果命令不是停止命令，则进行坐标变换
        if(!is_stop(cmd))
        {
            transform_xy(x, y);
        }
    
        return cmd;  // 返回生成的命令
    }
    
    
    //------------------------------------------------------------------------
    bool scale_ctrl_impl::in_rect(double x, double y) const
    {
        inverse_transform_xy(&x, &y);  // 反向变换坐标
        // 判断坐标是否在指定的矩形范围内
        return x >= m_x1 && x <= m_x2 && y >= m_y1 && y <= m_y2;
    }
    
    
    //------------------------------------------------------------------------
    bool scale_ctrl_impl::on_mouse_button_down(double x, double y)
    {
        // 略，该方法似乎缺少实现，应该是错误的或者未完整的方法定义
    }
    {
        // 使用逆变换函数更新坐标 x 和 y
        inverse_transform_xy(&x, &y);
    
        // 定义变量
        double xp1;
        double xp2;
        double ys1;
        double ys2;
        double xp;
        double yp;
    
        // 判断横向距离与纵向距离的绝对值，确定计算方式
        if(fabs(m_x2 - m_x1) > fabs(m_y2 - m_y1))
        {
            // 计算滑块位置和范围
            xp1 = m_xs1 + (m_xs2 - m_xs1) * m_value1;
            xp2 = m_xs1 + (m_xs2 - m_xs1) * m_value2;
            ys1 = m_y1  - m_border_extra / 2.0;
            ys2 = m_y2  + m_border_extra / 2.0;
            yp = (m_ys1 + m_ys2) / 2.0;
    
            // 检查是否在范围内，并设置滑块移动操作
            if(x > xp1 && y > ys1 && x < xp2 && y < ys2)
            {
                m_pdx = xp1 - x;
                m_move_what = move_slider;
                return true;
            }
    
            // 检查是否靠近第一个值，并设置相关操作
            if(calc_distance(x, y, xp1, yp) <= m_y2 - m_y1)
            {
                m_pdx = xp1 - x;
                m_move_what = move_value1;
                return true;
            }
    
            // 检查是否靠近第二个值，并设置相关操作
            if(calc_distance(x, y, xp2, yp) <= m_y2 - m_y1)
            {
                m_pdx = xp2 - x;
                m_move_what = move_value2;
                return true;
            }
        }
        else
        {
            // 计算滑块位置和范围（纵向）
            xp1 = m_x1  - m_border_extra / 2.0;
            xp2 = m_x2  + m_border_extra / 2.0;
            ys1 = m_ys1 + (m_ys2 - m_ys1) * m_value1;
            ys2 = m_ys1 + (m_ys2 - m_ys1) * m_value2;
            xp = (m_xs1 + m_xs2) / 2.0;
    
            // 检查是否在范围内，并设置滑块移动操作
            if(x > xp1 && y > ys1 && x < xp2 && y < ys2)
            {
                m_pdy = ys1 - y;
                m_move_what = move_slider;
                return true;
            }
    
            // 检查是否靠近第一个值，并设置相关操作
            if(calc_distance(x, y, xp, ys1) <= m_x2 - m_x1)
            {
                m_pdy = ys1 - y;
                m_move_what = move_value1;
                return true;
            }
    
            // 检查是否靠近第二个值，并设置相关操作
            if(calc_distance(x, y, xp, ys2) <= m_x2 - m_x1)
            {
                m_pdy = ys2 - y;
                m_move_what = move_value2;
                return true;
            }
        }
    
        // 如果未触发任何操作，返回 false
        return false;
    }
    
    
    
    //------------------------------------------------------------------------
    bool scale_ctrl_impl::on_mouse_move(double x, double y, bool button_flag)
    {
        // 对给定的坐标进行逆变换处理，修改传入的 x 和 y 值
        inverse_transform_xy(&x, &y);
        // 如果按钮标志位为假
        if(!button_flag)
        {
            // 调用 on_mouse_button_up 函数并返回其结果
            return on_mouse_button_up(x, y);
        }
    
        // 计算新的坐标位置
        double xp = x + m_pdx;
        double yp = y + m_pdy;
        double dv;
    
        // 根据移动类型进行不同的操作
        switch(m_move_what)
        {
        case move_value1:
            // 根据横纵坐标差异判断赋值方式
            if(fabs(m_x2 - m_x1) > fabs(m_y2 - m_y1))
            {
                m_value1 = (xp - m_xs1) / (m_xs2 - m_xs1);
            }
            else
            {
                m_value1 = (yp - m_ys1) / (m_ys2 - m_ys1);
            }
            // 确保 m_value1 在合法范围内
            if(m_value1 < 0.0) m_value1 = 0.0;
            if(m_value1 > m_value2 - m_min_d) m_value1 = m_value2 - m_min_d;
            // 返回操作成功标志
            return true;
    
        case move_value2:
            // 根据横纵坐标差异判断赋值方式
            if(fabs(m_x2 - m_x1) > fabs(m_y2 - m_y1))
            {
                m_value2 = (xp - m_xs1) / (m_xs2 - m_xs1);
            }
            else
            {
                m_value2 = (yp - m_ys1) / (m_ys2 - m_ys1);
            }
            // 确保 m_value2 在合法范围内
            if(m_value2 > 1.0) m_value2 = 1.0;
            if(m_value2 < m_value1 + m_min_d) m_value2 = m_value1 + m_min_d;
            // 返回操作成功标志
            return true;
    
        case move_slider:
            // 计算当前区间长度
            dv = m_value2 - m_value1;
            // 根据横纵坐标差异判断赋值方式
            if(fabs(m_x2 - m_x1) > fabs(m_y2 - m_y1))
            {
                m_value1 = (xp - m_xs1) / (m_xs2 - m_xs1);
            }
            else
            {
                m_value1 = (yp - m_ys1) / (m_ys2 - m_ys1);
            }
            // 根据计算结果调整 m_value2
            m_value2 = m_value1 + dv;
            // 确保 m_value1 在合法范围内，若不合法则调整区间长度 dv
            if(m_value1 < 0.0)
            {
                dv = m_value2 - m_value1;
                m_value1 = 0.0;
                m_value2 = m_value1 + dv;
            }
            // 确保 m_value2 在合法范围内，若不合法则调整区间长度 dv
            if(m_value2 > 1.0)
            {
                dv = m_value2 - m_value1;
                m_value2 = 1.0;
                m_value1 = m_value2 - dv;
            }
            // 返回操作成功标志
            return true;
        }
    
        // 若未匹配到任何移动类型，则返回操作失败标志
        return false;
    }
    
    
    //------------------------------------------------------------------------
    bool scale_ctrl_impl::on_mouse_button_up(double, double)
    {
        // 重置移动状态
        m_move_what = move_nothing;
        // 返回操作成功标志
        return false;
    }
    
    
    //------------------------------------------------------------------------
    bool scale_ctrl_impl::on_arrow_keys(bool left, bool right, bool down, bool up)
    {
/*
        如果 `right` 或 `up` 为真，则执行以下操作：
            - 增加 `m_value` 的值 0.005
            - 如果 `m_value` 超过 1.0，则将其设为 1.0
            - 返回 true 表示操作成功
        如果 `left` 或 `down` 为真，则执行以下操作：
            - 减少 `m_value` 的值 0.005
            - 如果 `m_value` 小于 0.0，则将其设为 0.0
            - 返回 true 表示操作成功
        如果以上条件都不满足，则返回 false
*/
        return false;
    }
}
```