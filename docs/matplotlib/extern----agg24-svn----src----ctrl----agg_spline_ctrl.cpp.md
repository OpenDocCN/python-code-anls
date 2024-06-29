# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\ctrl\agg_spline_ctrl.cpp`

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

#include "ctrl/agg_spline_ctrl.h"

// 引入 agg 命名空间
namespace agg
{

    //------------------------------------------------------------------------
    // spline_ctrl_impl 类的构造函数，初始化控制点和参数
    spline_ctrl_impl::spline_ctrl_impl(double x1, double y1, double x2, double y2, 
                                       unsigned num_pnt, bool flip_y) :
        ctrl(x1, y1, x2, y2, flip_y),       // 调用基类 ctrl 的构造函数初始化控制点范围
        m_num_pnt(num_pnt),                // 初始化控制点数量
        m_border_width(1.0),               // 边框宽度初始化为 1.0
        m_border_extra(0.0),               // 边框额外值初始化为 0.0
        m_curve_width(1.0),                // 曲线宽度初始化为 1.0
        m_point_size(3.0),                 // 点大小初始化为 3.0
        m_curve_poly(m_curve_pnt),         // 使用 m_curve_pnt 初始化 m_curve_poly
        m_idx(0),                          // 索引初始化为 0
        m_vertex(0),                       // 顶点数初始化为 0
        m_active_pnt(-1),                  // 活动点初始化为 -1
        m_move_pnt(-1),                    // 移动点初始化为 -1
        m_pdx(0.0),                        // 水平偏移初始化为 0.0
        m_pdy(0.0)                         // 垂直偏移初始化为 0.0
    {
        if(m_num_pnt < 4)  m_num_pnt = 4;   // 如果控制点数量小于 4，则设置为 4
        if(m_num_pnt > 32) m_num_pnt = 32;  // 如果控制点数量大于 32，则设置为 32

        unsigned i;
        // 初始化控制点数组 m_xp 和 m_yp
        for(i = 0; i < m_num_pnt; i++)
        {
            m_xp[i] = double(i) / double(m_num_pnt - 1);  // 水平位置均匀分布
            m_yp[i] = 0.5;                                // 垂直位置初始化为 0.5
        }
        calc_spline_box();    // 计算曲线框的位置和大小
        update_spline();      // 更新样条曲线
    }


    //------------------------------------------------------------------------
    // 设置边框宽度和额外边框
    void spline_ctrl_impl::border_width(double t, double extra)
    { 
        m_border_width = t;       // 设置边框宽度
        m_border_extra = extra;   // 设置额外边框
        calc_spline_box();       // 重新计算曲线框
    }


    //------------------------------------------------------------------------
    // 计算曲线框的位置和大小
    void spline_ctrl_impl::calc_spline_box()
    {
        m_xs1 = m_x1 + m_border_width;   // 计算曲线框左上角 x 坐标
        m_ys1 = m_y1 + m_border_width;   // 计算曲线框左上角 y 坐标
        m_xs2 = m_x2 - m_border_width;   // 计算曲线框右下角 x 坐标
        m_ys2 = m_y2 - m_border_width;   // 计算曲线框右下角 y 坐标
    }


    //------------------------------------------------------------------------
    // 更新样条曲线的值
    void spline_ctrl_impl::update_spline()
    {
        int i;
        m_spline.init(m_num_pnt, m_xp, m_yp);  // 使用控制点初始化样条曲线

        // 计算曲线上每个点的值
        for(i = 0; i < 256; i++)
        {
            m_spline_values[i] = m_spline.get(double(i) / 255.0);  // 获取样条曲线在 [0,1] 区间的值
            if(m_spline_values[i] < 0.0) m_spline_values[i] = 0.0;  // 确保值在合理范围内
            if(m_spline_values[i] > 1.0) m_spline_values[i] = 1.0;
            m_spline_values8[i] = (int8u)(m_spline_values[i] * 255.0);  // 转换为 [0, 255] 范围内的整数值
        }
    }

    // 以下函数未完全显示，需要继续注释...
    {
        // 定义整型变量 i
        int i;
        // 清空 m_curve_pnt 对象中的所有点
        m_curve_pnt.remove_all();
        // 将当前绘图位置移动到起始点 (m_xs1, m_ys1)，根据样条值调整纵坐标位置
        m_curve_pnt.move_to(m_xs1, m_ys1 + (m_ys2 - m_ys1) * m_spline_values[0]);
        // 循环绘制曲线
        for(i = 1; i < 256; i++)
        {
            // 绘制直线到下一个点，横坐标根据样条曲线计算
            m_curve_pnt.line_to(m_xs1 + (m_xs2 - m_xs1) * double(i) / 255.0, 
                                m_ys1 + (m_ys2 - m_ys1) * m_spline_values[i]);
        }
    }
    
    
    //------------------------------------------------------------------------
    // 计算给定索引下的横坐标位置
    double spline_ctrl_impl::calc_xp(unsigned idx)
    {
        return m_xs1 + (m_xs2 - m_xs1) * m_xp[idx];
    }
    
    
    //------------------------------------------------------------------------
    // 计算给定索引下的纵坐标位置
    double spline_ctrl_impl::calc_yp(unsigned idx)
    {
        return m_ys1 + (m_ys2 - m_ys1) * m_yp[idx];
    }
    
    
    //------------------------------------------------------------------------
    // 设置给定索引下的横坐标位置，确保在有效范围内
    void spline_ctrl_impl::set_xp(unsigned idx, double val)
    {
        if(val < 0.0) val = 0.0;
        if(val > 1.0) val = 1.0;
    
        // 对第一个和最后一个点的特殊处理
        if(idx == 0)
        {
            val = 0.0;
        }
        else if(idx == m_num_pnt - 1)
        {
            val = 1.0;
        }
        else
        {
            // 在相邻点之间设置横坐标值，保证不会太接近
            if(val < m_xp[idx - 1] + 0.001) val = m_xp[idx - 1] + 0.001;
            if(val > m_xp[idx + 1] - 0.001) val = m_xp[idx + 1] - 0.001;
        }
        // 将计算后的值赋给数组中的对应位置
        m_xp[idx] = val;
    }
    
    //------------------------------------------------------------------------
    // 设置给定索引下的纵坐标位置，确保在有效范围内
    void spline_ctrl_impl::set_yp(unsigned idx, double val)
    {
        if(val < 0.0) val = 0.0;
        if(val > 1.0) val = 1.0;
        // 将计算后的值赋给数组中的对应位置
        m_yp[idx] = val;
    }
    
    
    //------------------------------------------------------------------------
    // 设置给定索引下的点的横坐标和纵坐标位置
    void spline_ctrl_impl::point(unsigned idx, double x, double y)
    {
        if(idx < m_num_pnt) 
        {
            // 设置横坐标和纵坐标
            set_xp(idx, x);
            set_yp(idx, y);
        }
    }
    
    
    //------------------------------------------------------------------------
    // 设置给定索引下的点的纵坐标位置
    void spline_ctrl_impl::value(unsigned idx, double y)
    {
        if(idx < m_num_pnt) 
        {
            // 设置纵坐标
            set_yp(idx, y);
        }
    }
    
    //------------------------------------------------------------------------
    // 根据输入的横坐标计算对应的纵坐标值
    double spline_ctrl_impl::value(double x) const
    { 
        // 根据样条函数获取纵坐标值
        x = m_spline.get(x);
        // 确保返回值在有效范围内
        if(x < 0.0) x = 0.0;
        if(x > 1.0) x = 1.0;
        return x;
    }
    
    
    //------------------------------------------------------------------------
    // 重新设置给定索引下的参数
    void spline_ctrl_impl::rewind(unsigned idx)
    {
        unsigned i;  // 声明一个无符号整型变量 i
    
        m_idx = idx;  // 将参数 idx 的值赋给成员变量 m_idx
    
        switch(idx)  // 开始根据 idx 的值进行 switch-case 分支选择
        {
        default:  // 默认分支，如果 idx 的值不在 case 中指定的范围内
    
        case 0:  // 当 idx 的值为 0 时，执行以下代码（背景）
    
            m_vertex = 0;  // 设置 m_vertex 为 0
            m_vx[0] = m_x1 - m_border_extra;   // 设置 m_vx 数组的第一个元素
            m_vy[0] = m_y1 - m_border_extra;   // 设置 m_vy 数组的第一个元素
            m_vx[1] = m_x2 + m_border_extra;   // 设置 m_vx 数组的第二个元素
            m_vy[1] = m_y1 - m_border_extra;   // 设置 m_vy 数组的第二个元素
            m_vx[2] = m_x2 + m_border_extra;   // 设置 m_vx 数组的第三个元素
            m_vy[2] = m_y2 + m_border_extra;   // 设置 m_vy 数组的第三个元素
            m_vx[3] = m_x1 - m_border_extra;   // 设置 m_vx 数组的第四个元素
            m_vy[3] = m_y2 + m_border_extra;   // 设置 m_vy 数组的第四个元素
            break;  // 结束 case 0
    
        case 1:  // 当 idx 的值为 1 时，执行以下代码（边框）
    
            m_vertex = 0;  // 设置 m_vertex 为 0
            m_vx[0] = m_x1;   // 设置 m_vx 数组的第一个元素
            m_vy[0] = m_y1;   // 设置 m_vy 数组的第一个元素
            m_vx[1] = m_x2;   // 设置 m_vx 数组的第二个元素
            m_vy[1] = m_y1;   // 设置 m_vy 数组的第二个元素
            m_vx[2] = m_x2;   // 设置 m_vx 数组的第三个元素
            m_vy[2] = m_y2;   // 设置 m_vy 数组的第三个元素
            m_vx[3] = m_x1;   // 设置 m_vx 数组的第四个元素
            m_vy[3] = m_y2;   // 设置 m_vy 数组的第四个元素
            m_vx[4] = m_x1 + m_border_width;   // 设置 m_vx 数组的第五个元素
            m_vy[4] = m_y1 + m_border_width;   // 设置 m_vy 数组的第五个元素
            m_vx[5] = m_x1 + m_border_width;   // 设置 m_vx 数组的第六个元素
            m_vy[5] = m_y2 - m_border_width;   // 设置 m_vy 数组的第六个元素
            m_vx[6] = m_x2 - m_border_width;   // 设置 m_vx 数组的第七个元素
            m_vy[6] = m_y2 - m_border_width;   // 设置 m_vy 数组的第七个元素
            m_vx[7] = m_x2 - m_border_width;   // 设置 m_vx 数组的第八个元素
            m_vy[7] = m_y1 + m_border_width;   // 设置 m_vy 数组的第八个元素
            break;  // 结束 case 1
    
        case 2:  // 当 idx 的值为 2 时，执行以下代码（曲线）
    
            calc_curve();  // 调用 calc_curve() 函数计算曲线
            m_curve_poly.width(m_curve_width);  // 设置曲线的宽度
            m_curve_poly.rewind(0);  // 重置 m_curve_poly 对象
            break;  // 结束 case 2
    
        case 3:  // 当 idx 的值为 3 时，执行以下代码（非活动点）
    
            m_curve_pnt.remove_all();  // 清空 m_curve_pnt 对象
            for(i = 0; i < m_num_pnt; i++)  // 遍历非活动点的索引
            {
                if(int(i) != m_active_pnt)  // 如果当前索引不是活动点索引
                {
                    m_ellipse.init(calc_xp(i), calc_yp(i),  // 初始化椭圆对象
                                   m_point_size, m_point_size, 32);
                    m_curve_pnt.concat_path(m_ellipse);  // 将椭圆路径添加到 m_curve_pnt 中
                }
            }
            m_curve_poly.rewind(0);  // 重置 m_curve_poly 对象
            break;  // 结束 case 3
    
        case 4:  // 当 idx 的值为 4 时，执行以下代码（活动点）
    
            m_curve_pnt.remove_all();  // 清空 m_curve_pnt 对象
            if(m_active_pnt >= 0)  // 如果活动点索引大于等于 0
            {
                m_ellipse.init(calc_xp(m_active_pnt), calc_yp(m_active_pnt),  // 初始化椭圆对象
                               m_point_size, m_point_size, 32);
    
                m_curve_pnt.concat_path(m_ellipse);  // 将椭圆路径添加到 m_curve_pnt 中
            }
            m_curve_poly.rewind(0);  // 重置 m_curve_poly 对象
            break;  // 结束 case 4
    
        }
    }
    
    
    //------------------------------------------------------------------------
    unsigned spline_ctrl_impl::vertex(double* x, double* y)
    {
        // 设置默认的命令为 path_cmd_line_to
        unsigned cmd = path_cmd_line_to;
    
        // 根据 m_idx 的值进行不同的处理
        switch(m_idx)
        {
        case 0:
            // 当 m_vertex 等于 0 时，将命令设置为 path_cmd_move_to
            if(m_vertex == 0) cmd = path_cmd_move_to;
            // 当 m_vertex 大于等于 4 时，将命令设置为 path_cmd_stop
            if(m_vertex >= 4) cmd = path_cmd_stop;
            // 将 m_vx 和 m_vy 中的值赋给 *x 和 *y
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
            // m_vertex 加一
            m_vertex++;
            break;
    
        case 1:
            // 当 m_vertex 等于 0 或者等于 4 时，将命令设置为 path_cmd_move_to
            if(m_vertex == 0 || m_vertex == 4) cmd = path_cmd_move_to;
            // 当 m_vertex 大于等于 8 时，将命令设置为 path_cmd_stop
            if(m_vertex >= 8) cmd = path_cmd_stop;
            // 将 m_vx 和 m_vy 中的值赋给 *x 和 *y
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
            // m_vertex 加一
            m_vertex++;
            break;
    
        case 2:
            // 调用 m_curve_poly 对象的 vertex 方法，并将结果赋给 cmd
            cmd = m_curve_poly.vertex(x, y);
            break;
    
        case 3:
        case 4:
            // 调用 m_curve_pnt 对象的 vertex 方法，并将结果赋给 cmd
            cmd = m_curve_pnt.vertex(x, y);
            break;
    
        default:
            // 默认情况下将命令设置为 path_cmd_stop
            cmd = path_cmd_stop;
            break;
        }
    
        // 如果 cmd 不是停止命令，则调用 transform_xy 方法
        if(!is_stop(cmd))
        {
            transform_xy(x, y);
        }
    
        // 返回计算出的命令
        return cmd;
    }
    // 定义一个布尔类型的函数，处理箭头键的输入，控制样条曲线的移动
    bool spline_ctrl_impl::on_arrow_keys(bool left, bool right, bool down, bool up)
    {
        // 初始化 x 和 y 的增量为 0
        double kx = 0.0;
        double ky = 0.0;
        // 初始化返回值为 false
        bool ret = false;
        
        // 如果当前活动点的索引大于等于 0
        if(m_active_pnt >= 0)
        {
            // 获取当前活动点的 x 和 y 坐标
            kx = m_xp[m_active_pnt];
            ky = m_yp[m_active_pnt];
            
            // 根据箭头键更新 x 和 y 坐标，并设置返回值为 true
            if(left)  { kx -= 0.001; ret = true; }
            if(right) { kx += 0.001; ret = true; }
            if(down)  { ky -= 0.001; ret = true; }
            if(up)    { ky += 0.001; ret = true; }
        }
        
        // 如果 ret 为 true，则更新当前活动点的 x 和 y 坐标
        if(ret)
        {
            set_xp(m_active_pnt, kx);
            set_yp(m_active_pnt, ky);
            // 更新样条曲线
            update_spline();
        }
        
        // 返回处理结果
        return ret;
    }
}



# 这行代码表示一个代码块的结束，即函数或条件语句的结束
```