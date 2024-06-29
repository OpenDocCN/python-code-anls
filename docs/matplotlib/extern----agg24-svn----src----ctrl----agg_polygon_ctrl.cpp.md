# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\ctrl\agg_polygon_ctrl.cpp`

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
// classes polygon_ctrl_impl
//
//----------------------------------------------------------------------------

#include "ctrl/agg_polygon_ctrl.h"   // 包含 polygon 控制类的头文件

namespace agg
{

    polygon_ctrl_impl::polygon_ctrl_impl(unsigned np, double point_radius) :
        ctrl(0, 0, 1, 1, false),   // 调用基类 ctrl 的构造函数
        m_polygon(np * 2),         // 初始化 m_polygon 为长度为 np*2 的向量
        m_num_points(np),          // 设置多边形控制点的数量为 np
        m_node(-1),                // 初始化 m_node 为 -1
        m_edge(-1),                // 初始化 m_edge 为 -1
        m_vs(&m_polygon[0], m_num_points, false),   // 使用 m_polygon 初始化 m_vs
        m_stroke(m_vs),            // 使用 m_vs 初始化 m_stroke
        m_point_radius(point_radius),   // 设置点的半径为 point_radius
        m_status(0),               // 初始化 m_status 为 0
        m_dx(0.0),                 // 初始化 m_dx 为 0.0
        m_dy(0.0),                 // 初始化 m_dy 为 0.0
        m_in_polygon_check(true)   // 将 m_in_polygon_check 设置为 true
    {
        m_stroke.width(1.0);      // 设置 m_stroke 的宽度为 1.0
    }


    void polygon_ctrl_impl::rewind(unsigned)
    {
        m_status = 0;              // 重置 m_status 为 0
        m_stroke.rewind(0);        // 重置 m_stroke 的状态为初始状态
    }

    unsigned polygon_ctrl_impl::vertex(double* x, double* y)
    {
        unsigned cmd = path_cmd_stop;   // 初始化 cmd 为 path_cmd_stop
        double r = m_point_radius;      // 将 r 设置为 m_point_radius 的值
        if(m_status == 0)               // 如果 m_status 为 0
        {
            cmd = m_stroke.vertex(x, y);   // 获取 m_stroke 的下一个顶点坐标
            if(!is_stop(cmd))              // 如果 cmd 不是停止命令
            {
                transform_xy(x, y);        // 对顶点坐标进行转换
                return cmd;                 // 返回命令
            }
            if(m_node >= 0 && m_node == int(m_status)) r *= 1.2;   // 如果 m_node 大于等于 0 且等于 m_status，则 r 扩大 1.2 倍
            m_ellipse.init(xn(m_status), yn(m_status), r, r, 32);  // 初始化椭圆
            ++m_status;                   // 增加 m_status
        }
        cmd = m_ellipse.vertex(x, y);     // 获取椭圆的下一个顶点坐标
        if(!is_stop(cmd))                 // 如果 cmd 不是停止命令
        {
            transform_xy(x, y);           // 对顶点坐标进行转换
            return cmd;                    // 返回命令
        }
        if(m_status >= m_num_points) return path_cmd_stop;   // 如果 m_status 大于等于 m_num_points，返回停止命令
        if(m_node >= 0 && m_node == int(m_status)) r *= 1.2;   // 如果 m_node 大于等于 0 且等于 m_status，则 r 扩大 1.2 倍
        m_ellipse.init(xn(m_status), yn(m_status), r, r, 32);  // 初始化椭圆
        ++m_status;                     // 增加 m_status
        cmd = m_ellipse.vertex(x, y);   // 获取椭圆的下一个顶点坐标
        if(!is_stop(cmd))               // 如果 cmd 不是停止命令
        {
            transform_xy(x, y);         // 对顶点坐标进行转换
        }
        return cmd;                     // 返回命令
    }


    bool polygon_ctrl_impl::check_edge(unsigned i, double x, double y) const
    {
       // 初始化返回值为 false
       bool ret = false;
    
       // 获取当前点和前一点的索引
       unsigned n1 = i;
       unsigned n2 = (i + m_num_points - 1) % m_num_points;
    
       // 获取当前点和前一点的坐标
       double x1 = xn(n1);
       double y1 = yn(n1);
       double x2 = xn(n2);
       double y2 = yn(n2);
    
       // 计算当前点和前一点的坐标差
       double dx = x2 - x1;
       double dy = y2 - y1;
    
       // 如果当前点和前一点距离大于阈值 0.0000001，则执行以下操作
       if(sqrt(dx*dx + dy*dy) > 0.0000001)
       {
          // 计算垂直于当前边的向量坐标
          double x3 = x;
          double y3 = y;
          double x4 = x3 - dy;
          double y4 = y3 + dx;
    
          // 计算交点的分母
          double den = (y4-y3) * (x2-x1) - (x4-x3) * (y2-y1);
    
          // 计算交点的参数 u1
          double u1 = ((x4-x3) * (y1-y3) - (y4-y3) * (x1-x3)) / den;
    
          // 计算交点的坐标
          double xi = x1 + u1 * (x2 - x1);
          double yi = y1 + u1 * (y2 - y1);
    
          // 计算鼠标点击点到交点的距离
          dx = xi - x;
          dy = yi - y;
    
          // 如果参数 u1 在 (0, 1) 之间且鼠标点击点到交点的距离小于等于点的半径，则设置返回值为 true
          if (u1 > 0.0 && u1 < 1.0 && sqrt(dx*dx + dy*dy) <= m_point_radius)
          {
             ret = true;
          }
       }
       // 返回结果
       return ret;
    }
    
    
    
    bool polygon_ctrl_impl::in_rect(double x, double y) const
    {
        // 始终返回 false，表示点不在矩形内
        return false;
    }
    
    
    bool polygon_ctrl_impl::on_mouse_button_down(double x, double y)
    {
        // 初始化返回值为 false
        bool ret = false;
    
        // 重置选中的节点和边索引
        m_node = -1;
        m_edge = -1;
    
        // 将坐标 x, y 转换到逆变换后的坐标空间
        inverse_transform_xy(&x, &y);
    
        // 遍历多边形的所有节点
        for (unsigned i = 0; i < m_num_points; i++)
        {
            // 如果鼠标点击点到节点的距离小于点的半径，则选中该节点
            if(sqrt( (x-xn(i)) * (x-xn(i)) + (y-yn(i)) * (y-yn(i)) ) < m_point_radius)
            {
                // 记录鼠标点击点与节点的距离差
                m_dx = x - xn(i);
                m_dy = y - yn(i);
                // 设置选中的节点索引
                m_node = int(i);
                ret = true;
                break;
            }
        }
    
        // 如果没有选中节点，则检查是否选中了边
        if(!ret)
        {
            for (unsigned i = 0; i < m_num_points; i++)
            {
                if(check_edge(i, x, y))
                {
                    // 记录鼠标点击点的坐标
                    m_dx = x;
                    m_dy = y;
                    // 设置选中的边索引
                    m_edge = int(i);
                    ret = true;
                    break;
                }
            }
        }
    
        // 如果仍未选中节点或边，则检查鼠标点击点是否在多边形内部
        if(!ret)
        {
            if(point_in_polygon(x, y))
            {
                // 记录鼠标点击点的坐标
                m_dx = x;
                m_dy = y;
                // 设置选中的节点索引为多边形节点数（表示在多边形内部）
                m_node = int(m_num_points);
                ret = true;
            }
        }
        // 返回结果
        return ret;
    }
    
    
    bool polygon_ctrl_impl::on_mouse_move(double x, double y, bool button_flag)
    {
        // 该函数暂未提供代码，需要在实现时进行注释
    }
    {
        // 初始化返回值为false
        bool ret = false;
        // 定义变量，用于存储坐标差值
        double dx;
        double dy;
        // 调用函数将坐标进行逆变换
        inverse_transform_xy(&x, &y);
        // 如果当前节点索引等于点的总数
        if(m_node == int(m_num_points))
        {
            // 计算当前点与上一次操作点的坐标差值
            dx = x - m_dx;
            dy = y - m_dy;
            // 遍历所有点，更新它们的坐标
            unsigned i;
            for(i = 0; i < m_num_points; i++)
            {
                xn(i) += dx;
                yn(i) += dy;
            }
            // 更新上一次操作点的坐标
            m_dx = x;
            m_dy = y;
            // 设置返回值为true
            ret = true;
        }
        else
        {
            // 如果当前边索引大于等于0
            if(m_edge >= 0)
            {
                // 计算当前点与上一次操作点的坐标差值
                unsigned n1 = m_edge;
                unsigned n2 = (n1 + m_num_points - 1) % m_num_points;
                dx = x - m_dx;
                dy = y - m_dy;
                // 更新两个相关点的坐标
                xn(n1) += dx;
                yn(n1) += dy;
                xn(n2) += dx;
                yn(n2) += dy;
                // 更新上一次操作点的坐标
                m_dx = x;
                m_dy = y;
                // 设置返回值为true
                ret = true;
            }
            else
            {
                // 如果当前节点索引大于等于0
                if(m_node >= 0)
                {
                    // 更新当前节点的坐标
                    xn(m_node) = x - m_dx;
                    yn(m_node) = y - m_dy;
                    // 设置返回值为true
                    ret = true;
                }
            }
        }
        // 返回操作结果
        return ret;
    }
    
    bool polygon_ctrl_impl::on_mouse_button_up(double x, double y)
    {
        // 返回值为true如果当前有节点或边处于活动状态，否则为false
        bool ret = (m_node >= 0) || (m_edge >= 0);
        // 重置节点和边的状态
        m_node = -1;
        m_edge = -1;
        // 返回操作结果
        return ret;
    }
    
    bool polygon_ctrl_impl::on_arrow_keys(bool left, bool right, bool down, bool up)
    {
        // 箭头键操作默认返回false
        return false;
    }
    
    
    
    //======= Crossings Multiply algorithm of InsideTest ======================== 
    //
    // By Eric Haines, 3D/Eye Inc, erich@eye.com
    //
    // This version is usually somewhat faster than the original published in
    // Graphics Gems IV; by turning the division for testing the X axis crossing
    // into a tricky multiplication test this part of the test became faster,
    // which had the additional effect of making the test for "both to left or
    // both to right" a bit slower for triangles than simply computing the
    // intersection each time.  The main increase is in triangle testing speed,
    // which was about 15% faster; all other polygon complexities were pretty much
    // the same as before.  On machines where division is very expensive (not the
    // case on the HP 9000 series on which I tested) this test should be much
    // faster overall than the old code.  Your mileage may (in fact, will) vary,
    // depending on the machine and the test data, but in general I believe this
    // code is both shorter and faster.  This test was inspired by unpublished
    // Graphics Gems submitted by Joseph Samosky and Mark Haigh-Hutchinson.
    // Related work by Samosky is in:
    //
    // Samosky, Joseph, "SectionView: A system for interactively specifying and
    // visualizing sections through three-dimensional medical image data",
    // M.S. Thesis, Department of Electrical Engineering and Computer Science,
    // Massachusetts Institute of Technology, 1993.
    //
    // 射出一个测试光线沿着 +X 轴。策略是将顶点的 Y 值与测试点的 Y 值进行比较，
    // 并且快速地丢弃完全在测试光线一侧的边。注意，凸多边形和环绕多边形的代码可以
    // 像 CrossingsTest() 代码一样添加，这里为了清晰起见略去了。
    //
    // 输入具有 numverts 个顶点的二维多边形 _pgon_ 和测试点 _point_，如果在内部返回1，如果在外部返回0。
    bool polygon_ctrl_impl::point_in_polygon(double tx, double ty) const
    {
        // 如果顶点少于3个，返回false
        if(m_num_points < 3) return false;
        // 如果不需要进行多边形检查，返回false
        if(!m_in_polygon_check) return false;
    
        unsigned j;
        int yflag0, yflag1, inside_flag;
        double vtx0, vty0, vtx1, vty1;
    
        // 最后一个顶点作为起始点
        vtx0 = xn(m_num_points - 1);
        vty0 = yn(m_num_points - 1);
    
        // 获取位于 X 轴上方/下方的测试标志位
        yflag0 = (vty0 >= ty);
    
        // 第一个顶点作为终点
        vtx1 = xn(0);
        vty1 = yn(0);
    
        inside_flag = 0;
        for (j = 1; j <= m_num_points; ++j) 
        {
            // 当前顶点的位于 X 轴上方/下方的测试标志位
            yflag1 = (vty1 >= ty);
    
            // 检查端点是否跨越 X 轴（即 Y 值不同）；
            // 如果是，+X 光线可能与此边相交。
            // 旧的测试还检查端点是否同时位于测试点左侧或右侧。
            // 然而，由于下面使用的更快的交点计算，对于大多数多边形来说，这个测试被发现是一个平衡点，
            // 对于三角形来说则是一个失败的测试（如果超过50%的边通过此测试，则会跨越象限，因此必须计算 X 交点）。
            // 我要感谢 Joseph Samosky 激发我尝试删除代码中的“同时左侧或同时右侧”部分。
            if (yflag0 != yflag1) 
            {
                // 检查多边形段与 +X 光线的交点。
                // 如果 >= 测试点的 X，则光线与之相交。
                // 通过检查第一个顶点相对于测试点的符号来避免除法运算的 ">=" 测试；
                // 这个想法受到 Joseph Samosky 和 Mark Haigh-Hutchinson 的不同多边形包含测试的启发。
                if ( ((vty1-ty) * (vtx0-vtx1) >=
                      (vtx1-tx) * (vty0-vty1)) == yflag1 ) 
                {
                    inside_flag ^= 1; // 切换内部标志
                }
            }
    
            // 移动到下一对顶点，并尽可能保留信息。
            yflag0 = yflag1;
            vtx0 = vtx1;
            vty0 = vty1;
    
            // 计算下一个顶点的索引
            unsigned k = (j >= m_num_points) ? j - m_num_points : j;
            vtx1 = xn(k);
            vty1 = yn(k);
        }
    
        // 如果内部标志不为0，则在多边形内部；否则在外部。
        return inside_flag != 0;
    }
}



# 这行代码表示一个函数定义的结束，对应于之前的函数定义的开始位置。
```