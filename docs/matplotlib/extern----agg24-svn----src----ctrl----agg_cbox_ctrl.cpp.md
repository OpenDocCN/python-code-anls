# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\ctrl\agg_cbox_ctrl.cpp`

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
// classes rbox_ctrl_impl, rbox_ctrl
//
//----------------------------------------------------------------------------

#include <string.h>
#include "ctrl/agg_cbox_ctrl.h"

// 命名空间 agg
namespace agg
{

    //------------------------------------------------------------------------
    // 实现 cbox_ctrl_impl 类的构造函数
    cbox_ctrl_impl::cbox_ctrl_impl(double x, double y, 
                                   const char* l, 
                                   bool flip_y) :
        ctrl(x, y, x + 9.0 * 1.5, y + 9.0 * 1.5, flip_y),  // 调用基类 ctrl 的构造函数，设定控件的初始位置和大小
        m_text_thickness(1.5),                             // 设定文本的线条粗细
        m_text_height(9.0),                                // 设定文本的高度
        m_text_width(0.0),                                 // 设定文本的宽度
        m_status(false),                                   // 初始化状态为 false，表示未选中
        m_text_poly(m_text)                                // 初始化文本多边形
    {
        label(l);  // 调用 label 方法，设置标签文本
    }


    //------------------------------------------------------------------------
    // 设置文本的大小
    void cbox_ctrl_impl::text_size(double h, double w)
    {
        m_text_width = w;  // 设置文本的宽度
        m_text_height = h; // 设置文本的高度
    }

    //------------------------------------------------------------------------
    // 设置标签文本
    void cbox_ctrl_impl::label(const char* l)
    {
        unsigned len = strlen(l);         // 计算标签文本的长度
        if(len > 127) len = 127;          // 限制标签文本长度最大为 127 字符
        memcpy(m_label, l, len);          // 复制标签文本到 m_label
        m_label[len] = 0;                 // 在末尾添加字符串结束符
    }


    //------------------------------------------------------------------------
    // 处理鼠标按下事件
    bool cbox_ctrl_impl::on_mouse_button_down(double x, double y)
    {
        inverse_transform_xy(&x, &y);     // 将鼠标坐标反向变换
        if(x >= m_x1 && y >= m_y1 && x <= m_x2 && y <= m_y2)
        {
            m_status = !m_status;         // 切换状态（选中/未选中）
            return true;                  // 返回 true 表示事件已处理
        }
        return false;                     // 返回 false 表示事件未处理
    }


    //------------------------------------------------------------------------
    // 处理鼠标移动事件
    bool cbox_ctrl_impl::on_mouse_move(double, double, bool)
    {
        return false;                     // 返回 false 表示事件未处理
    }

    //------------------------------------------------------------------------
    // 检查给定坐标是否在控件的范围内
    bool cbox_ctrl_impl::in_rect(double x, double y) const
    {
        inverse_transform_xy(&x, &y);     // 将坐标反向变换
        return x >= m_x1 && y >= m_y1 && x <= m_x2 && y <= m_y2;  // 返回是否在控件范围内的判断结果
    }

    //------------------------------------------------------------------------
    // 处理鼠标释放事件
    bool cbox_ctrl_impl::on_mouse_button_up(double, double)
    {
        return false;                     // 返回 false 表示事件未处理
    }

    //------------------------------------------------------------------------
    // 处理方向键事件
    bool cbox_ctrl_impl::on_arrow_keys(bool, bool, bool, bool)
    {
        return false;                     // 返回 false 表示事件未处理
    }

} // namespace agg
    //------------------------------------------------------------------------
    void cbox_ctrl_impl::rewind(unsigned idx)
    {
        // 设置当前索引
        m_idx = idx;
    
        // 声明局部变量
        double d2;
        double t;
    
        // 根据索引选择执行不同的分支
        switch(idx)
        {
        default:
        case 0:                 // Border
            // 初始化顶点计数器
            m_vertex = 0;
            // 设置边框顶点坐标
            m_vx[0] = m_x1; 
            m_vy[0] = m_y1;
            m_vx[1] = m_x2;
            m_vy[1] = m_y1;
            m_vx[2] = m_x2;
            m_vy[2] = m_y2;
            m_vx[3] = m_x1; 
            m_vy[3] = m_y2;
            // 设置边框文本相关顶点坐标
            m_vx[4] = m_x1 + m_text_thickness; 
            m_vy[4] = m_y1 + m_text_thickness; 
            m_vx[5] = m_x1 + m_text_thickness; 
            m_vy[5] = m_y2 - m_text_thickness;
            m_vx[6] = m_x2 - m_text_thickness;
            m_vy[6] = m_y2 - m_text_thickness;
            m_vx[7] = m_x2 - m_text_thickness;
            m_vy[7] = m_y1 + m_text_thickness; 
            break;
    
        case 1:                 // Text
            // 设置文本内容
            m_text.text(m_label);
            // 设置文本起始点和大小
            m_text.start_point(m_x1 + m_text_height * 2.0, m_y1 + m_text_height / 5.0);
            m_text.size(m_text_height, m_text_width);
            // 设置文本多边形线宽和样式
            m_text_poly.width(m_text_thickness);
            m_text_poly.line_join(round_join);
            m_text_poly.line_cap(round_cap);
            // 重置文本多边形路径
            m_text_poly.rewind(0);
            break;
    
        case 2:                 // Active item
            // 初始化顶点计数器
            m_vertex = 0;
            // 计算局部变量值
            d2 = (m_y2 - m_y1) / 2.0;
            t = m_text_thickness * 1.5;
            // 设置活动项相关顶点坐标
            m_vx[0] = m_x1 + m_text_thickness;
            m_vy[0] = m_y1 + m_text_thickness;
            m_vx[1] = m_x1 + d2;
            m_vy[1] = m_y1 + d2 - t;
            m_vx[2] = m_x2 - m_text_thickness;
            m_vy[2] = m_y1 + m_text_thickness;
            m_vx[3] = m_x1 + d2 + t;
            m_vy[3] = m_y1 + d2;
            m_vx[4] = m_x2 - m_text_thickness;
            m_vy[4] = m_y2 - m_text_thickness;
            m_vx[5] = m_x1 + d2;
            m_vy[5] = m_y1 + d2 + t;
            m_vx[6] = m_x1 + m_text_thickness;
            m_vy[6] = m_y2 - m_text_thickness;
            m_vx[7] = m_x1 + d2 - t;
            m_vy[7] = m_y1 + d2;
            break;
        }
    }
    //------------------------------------------------------------------------
    unsigned cbox_ctrl_impl::vertex(double* x, double* y)
    
    
    这段代码是一个 C++ 类 `cbox_ctrl_impl` 的方法实现，主要用于根据给定的索引 `idx` 初始化对象的顶点坐标。根据不同的索引值，执行不同的初始化操作，包括设置边框的顶点坐标、设置文本的位置和样式，以及设置活动项的相关顶点坐标。
    {
        // 初始化命令为 path_cmd_line_to
        unsigned cmd = path_cmd_line_to;
    
        // 根据 m_idx 的值进行不同的操作选择
        switch(m_idx)
        {
        // 如果 m_idx 为 0 时的情况
        case 0:
            // 如果 m_vertex 为 0 或 4，则设置命令为 path_cmd_move_to
            if(m_vertex == 0 || m_vertex == 4) cmd = path_cmd_move_to;
            // 如果 m_vertex 大于等于 8，则设置命令为 path_cmd_stop
            if(m_vertex >= 8) cmd = path_cmd_stop;
            // 将当前顶点的坐标存入 *x 和 *y 中
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
            // 增加顶点索引 m_vertex
            m_vertex++;
            break;
    
        // 如果 m_idx 为 1 时的情况
        case 1:
            // 调用 m_text_poly 对象的 vertex 方法获取顶点坐标，并将结果作为命令返回
            cmd = m_text_poly.vertex(x, y);
            break;
    
        // 如果 m_idx 为 2 时的情况
        case 2:
            // 如果 m_status 为真
            if(m_status)
            {
                // 如果 m_vertex 为 0 时，设置命令为 path_cmd_move_to
                if(m_vertex == 0) cmd = path_cmd_move_to;
                // 如果 m_vertex 大于等于 8，则设置命令为 path_cmd_stop
                if(m_vertex >= 8) cmd = path_cmd_stop;
                // 将当前顶点的坐标存入 *x 和 *y 中
                *x = m_vx[m_vertex];
                *y = m_vy[m_vertex];
                // 增加顶点索引 m_vertex
                m_vertex++;
            }
            else
            {
                // 如果 m_status 为假，则设置命令为 path_cmd_stop
                cmd = path_cmd_stop;
            }
            break;
    
        // 默认情况下
        default:
            // 设置命令为 path_cmd_stop
            cmd = path_cmd_stop;
            break;
        }
    
        // 如果命令不是停止命令，则对 *x 和 *y 进行坐标变换
        if(!is_stop(cmd))
        {
            transform_xy(x, y);
        }
    
        // 返回确定的命令
        return cmd;
    }
}
```