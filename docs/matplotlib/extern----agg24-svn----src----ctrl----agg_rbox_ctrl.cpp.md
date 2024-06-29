# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\ctrl\agg_rbox_ctrl.cpp`

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
// classes rbox_ctrl_impl, rbox_ctrl
//
//----------------------------------------------------------------------------

#include <string.h>    // 导入字符串处理函数
#include "ctrl/agg_rbox_ctrl.h"  // 导入自定义的agg_rbox_ctrl头文件

namespace agg
{
  
    //------------------------------------------------------------------------
    // rbox_ctrl_impl 类的构造函数，初始化各成员变量并计算rbox的边界框
    rbox_ctrl_impl::rbox_ctrl_impl(double x1, double y1, 
                                   double x2, double y2, bool flip_y) :
        ctrl(x1, y1, x2, y2, flip_y),      // 调用基类ctrl的构造函数进行初始化
        m_border_width(1.0),               // 初始化边框宽度
        m_border_extra(0.0),               // 初始化边框额外空白
        m_text_thickness(1.5),             // 初始化文本线条粗细
        m_text_height(9.0),                // 初始化文本高度
        m_text_width(0.0),                 // 初始化文本宽度
        m_num_items(0),                    // 初始化条目数量
        m_cur_item(-1),                    // 初始化当前条目索引
        m_ellipse_poly(m_ellipse),         // 使用m_ellipse初始化椭圆多边形
        m_text_poly(m_text),               // 使用m_text初始化文本多边形
        m_idx(0),                          // 初始化索引
        m_vertex(0)                        // 初始化顶点
    {
        calc_rbox();  // 计算rbox的边界框
    }


    //------------------------------------------------------------------------
    // 计算rbox的边界框坐标
    void rbox_ctrl_impl::calc_rbox()
    {
        m_xs1 = m_x1 + m_border_width;    // 计算左上角x坐标
        m_ys1 = m_y1 + m_border_width;    // 计算左上角y坐标
        m_xs2 = m_x2 - m_border_width;    // 计算右下角x坐标
        m_ys2 = m_y2 - m_border_width;    // 计算右下角y坐标
    }


    //------------------------------------------------------------------------
    // 向rbox_ctrl_impl对象添加条目，最多32个
    void rbox_ctrl_impl::add_item(const char* text)
    {
        if(m_num_items < 32)               // 如果条目数量小于32
        {
            m_items[m_num_items].resize(strlen(text) + 1);  // 调整字符串存储空间大小
            strcpy(&m_items[m_num_items][0], text);         // 复制文本内容到条目数组
            m_num_items++;                                  // 增加条目数量计数器
        }
    }


    //------------------------------------------------------------------------
    // 设置边框宽度和额外空白，并重新计算rbox的边界框
    void rbox_ctrl_impl::border_width(double t, double extra)
    { 
        m_border_width = t;        // 设置边框宽度
        m_border_extra = extra;    // 设置边框额外空白
        calc_rbox();              // 重新计算rbox的边界框
    }


    //------------------------------------------------------------------------
    // 设置文本大小
    void rbox_ctrl_impl::text_size(double h, double w) 
    { 
        m_text_width = w;    // 设置文本宽度
        m_text_height = h;   // 设置文本高度
    }

    //------------------------------------------------------------------------
    // 重置索引并准备进行迭代操作
    {
        // 设置当前索引
        m_idx = idx;
        // 计算每个条目的垂直偏移量
        m_dy = m_text_height * 2.0;
        // 初始化绘制项目计数器
        m_draw_item = 0;
    
        // 根据索引选择操作
        switch(idx)
        {
        default:
    
        case 0:                 // 背景
            // 设置顶点数为4
            m_vertex = 0;
            // 定义背景矩形的顶点坐标
            m_vx[0] = m_x1 - m_border_extra; 
            m_vy[0] = m_y1 - m_border_extra;
            m_vx[1] = m_x2 + m_border_extra; 
            m_vy[1] = m_y1 - m_border_extra;
            m_vx[2] = m_x2 + m_border_extra; 
            m_vy[2] = m_y2 + m_border_extra;
            m_vx[3] = m_x1 - m_border_extra; 
            m_vy[3] = m_y2 + m_border_extra;
            break;
    
        case 1:                 // 边框
            // 设置顶点数为8
            m_vertex = 0;
            // 定义边框矩形的顶点坐标，包括边框宽度的调整
            m_vx[0] = m_x1; 
            m_vy[0] = m_y1;
            m_vx[1] = m_x2; 
            m_vy[1] = m_y1;
            m_vx[2] = m_x2; 
            m_vy[2] = m_y2;
            m_vx[3] = m_x1; 
            m_vy[3] = m_y2;
            m_vx[4] = m_x1 + m_border_width; 
            m_vy[4] = m_y1 + m_border_width; 
            m_vx[5] = m_x1 + m_border_width; 
            m_vy[5] = m_y2 - m_border_width; 
            m_vx[6] = m_x2 - m_border_width; 
            m_vy[6] = m_y2 - m_border_width; 
            m_vx[7] = m_x2 - m_border_width; 
            m_vy[7] = m_y1 + m_border_width; 
            break;
    
        case 2:                 // 文本
            // 设置文本内容
            m_text.text(&m_items[0][0]);
            // 设置文本起始点
            m_text.start_point(m_xs1 + m_dy * 1.5, m_ys1 + m_dy / 2.0);
            // 设置文本大小
            m_text.size(m_text_height, m_text_width);
            // 设置文本多边形的线宽
            m_text_poly.width(m_text_thickness);
            // 设置文本多边形的线连接方式
            m_text_poly.line_join(round_join);
            // 设置文本多边形的线端点样式
            m_text_poly.line_cap(round_cap);
            // 重置文本多边形的绘制路径
            m_text_poly.rewind(0);
            break;
    
        case 3:                 // 非活动条目
            // 初始化椭圆对象，用于非活动条目的绘制
            m_ellipse.init(m_xs1 + m_dy / 1.3, 
                           m_ys1 + m_dy / 1.3,
                           m_text_height / 1.5, 
                           m_text_height / 1.5, 32);
            // 设置椭圆多边形的线宽
            m_ellipse_poly.width(m_text_thickness);
            // 重置椭圆多边形的绘制路径
            m_ellipse_poly.rewind(0);
            break;
    
        case 4:                 // 活动条目
            // 如果当前条目有效，初始化椭圆对象用于活动条目的绘制
            if(m_cur_item >= 0)
            {
                m_ellipse.init(m_xs1 + m_dy / 1.3, 
                               m_ys1 + m_dy * m_cur_item + m_dy / 1.3,
                               m_text_height / 2.0, 
                               m_text_height / 2.0, 32);
                // 重置椭圆绘制路径
                m_ellipse.rewind(0);
            }
            break;
    
        }
    }
    
    
    //------------------------------------------------------------------------
    unsigned rbox_ctrl_impl::vertex(double* x, double* y)
    {
        // 初始化路径命令为 path_cmd_line_to
        unsigned cmd = path_cmd_line_to;
        // 根据 m_idx 的值执行不同的操作
        switch(m_idx)
        {
        case 0:
            // 如果 m_vertex 等于 0，则命令为 path_cmd_move_to
            if(m_vertex == 0) cmd = path_cmd_move_to;
            // 如果 m_vertex 大于等于 4，则命令为 path_cmd_stop
            if(m_vertex >= 4) cmd = path_cmd_stop;
            // 设置输出参数 *x 和 *y 为 m_vx[m_vertex] 和 m_vy[m_vertex] 的值
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
            // m_vertex 自增
            m_vertex++;
            break;
    
        case 1:
            // 如果 m_vertex 等于 0 或者等于 4，则命令为 path_cmd_move_to
            if(m_vertex == 0 || m_vertex == 4) cmd = path_cmd_move_to;
            // 如果 m_vertex 大于等于 8，则命令为 path_cmd_stop
            if(m_vertex >= 8) cmd = path_cmd_stop;
            // 设置输出参数 *x 和 *y 为 m_vx[m_vertex] 和 m_vy[m_vertex] 的值
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
            // m_vertex 自增
            m_vertex++;
            break;
    
        case 2:
            // 调用 m_text_poly.vertex(x, y)，获取路径命令
            cmd = m_text_poly.vertex(x, y);
            // 如果路径命令表示结束
            if(is_stop(cmd))
            {
                // 增加绘制项目计数器
                m_draw_item++;
                // 如果绘制项目计数器大于等于项目数，结束分支
                if(m_draw_item >= m_num_items)
                {
                    break;
                }
                else
                {
                    // 设置文本内容和起始点
                    m_text.text(&m_items[m_draw_item][0]);
                    m_text.start_point(m_xs1 + m_dy * 1.5, 
                                       m_ys1 + m_dy * (m_draw_item + 1) - m_dy / 2.0);
                    // 重置 m_text_poly
                    m_text_poly.rewind(0);
                    // 再次调用 m_text_poly.vertex(x, y)，获取新的路径命令
                    cmd = m_text_poly.vertex(x, y);
                }
            }
            break;
    
        case 3:
            // 调用 m_ellipse_poly.vertex(x, y)，获取路径命令
            cmd = m_ellipse_poly.vertex(x, y);
            // 如果路径命令表示结束
            if(is_stop(cmd))
            {
                // 增加绘制项目计数器
                m_draw_item++;
                // 如果绘制项目计数器大于等于项目数，结束分支
                if(m_draw_item >= m_num_items)
                {
                    break;
                }
                else
                {
                    // 初始化椭圆对象并设置路径
                    m_ellipse.init(m_xs1 + m_dy / 1.3, 
                                   m_ys1 + m_dy * m_draw_item + m_dy / 1.3,
                                   m_text_height / 1.5, 
                                   m_text_height / 1.5, 32);
                    // 重置 m_ellipse_poly
                    m_ellipse_poly.rewind(0);
                    // 再次调用 m_ellipse_poly.vertex(x, y)，获取新的路径命令
                    cmd = m_ellipse_poly.vertex(x, y);
                }
            }
            break;
    
        case 4:
            // 如果 m_cur_item 大于等于 0，则调用 m_ellipse.vertex(x, y)
            if(m_cur_item >= 0)
            {
                cmd = m_ellipse.vertex(x, y);
            }
            else
            {
                // 否则，命令为 path_cmd_stop
                cmd = path_cmd_stop;
            }
            break;
    
        default:
            // 默认情况下，命令为 path_cmd_stop
            cmd = path_cmd_stop;
            break;
        }
    
        // 如果命令不表示结束，调用 transform_xy(x, y)
        if(!is_stop(cmd))
        {
            transform_xy(x, y);
        }
    
        // 返回命令
        return cmd;
    }
    
    
    //------------------------------------------------------------------------
    // 检查点 (x, y) 是否在矩形内部
    bool rbox_ctrl_impl::in_rect(double x, double y) const
    {
        // 反向转换坐标 (x, y)
        inverse_transform_xy(&x, &y);
        // 返回是否在矩形内部的判断结果
        return x >= m_x1 && x <= m_x2 && y >= m_y1 && y <= m_y2;
    }
    {
        // 调用 inverse_transform_xy 函数将 x 和 y 坐标反向变换
        inverse_transform_xy(&x, &y);
        // 声明无符号整型变量 i
        unsigned i;
        // 循环遍历 m_num_items 次
        for(i = 0; i < m_num_items; i++)  
        {
            // 计算 xp 和 yp 的值
            double xp = m_xs1 + m_dy / 1.3;
            double yp = m_ys1 + m_dy * i + m_dy / 1.3;
            // 如果点 (x, y) 到 (xp, yp) 的距离小于等于 m_text_height / 1.5
            if(calc_distance(x, y, xp, yp) <= m_text_height / 1.5)
            {
                // 设置当前项 m_cur_item 为 i，并返回 true
                m_cur_item = int(i);
                return true;
            }
        }
        // 如果未找到符合条件的项，返回 false
        return false;
    }

    //------------------------------------------------------------------------
    bool rbox_ctrl_impl::on_mouse_move(double, double, bool)
    {
        // 当鼠标移动时，默认返回 false
        return false;
    }

    //------------------------------------------------------------------------
    bool rbox_ctrl_impl::on_mouse_button_up(double, double)
    {
        // 当鼠标按钮抬起时，默认返回 false
        return false;
    }

    //------------------------------------------------------------------------
    bool rbox_ctrl_impl::on_arrow_keys(bool left, bool right, bool down, bool up)
    {
        // 如果当前项 m_cur_item 大于等于 0
        if(m_cur_item >= 0)
        {
            // 如果按下了上键或右键
            if(up || right) 
            {
                // 将当前项 m_cur_item 增加 1
                m_cur_item++;
                // 如果增加后超过了项目总数，则回到第一个项目
                if(m_cur_item >= int(m_num_items))
                {
                    m_cur_item = 0;
                }
                return true;
            }

            // 如果按下了下键或左键
            if(down || left) 
            {
                // 将当前项 m_cur_item 减少 1
                m_cur_item--;
                // 如果减少后小于 0，则回到最后一个项目
                if(m_cur_item < 0)
                {
                    m_cur_item = m_num_items - 1;
                }
                return true;
            }
        }
        // 如果 m_cur_item 小于 0，或者没有按下箭头键，则返回 false
        return false;
    }
}


注释：


# 关闭当前的代码块或函数定义
```