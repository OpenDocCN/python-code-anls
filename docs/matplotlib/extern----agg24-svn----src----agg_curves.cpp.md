# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_curves.cpp`

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

#include <math.h>               // 引入数学函数库，包括 sqrt() 等
#include "agg_curves.h"         // 引入曲线处理头文件
#include "agg_math.h"           // 引入数学计算头文件

namespace agg
{

    //------------------------------------------------------------------------
    const double curve_distance_epsilon                  = 1e-30; // 曲线距离的极小值
    const double curve_collinearity_epsilon              = 1e-30; // 曲线共线性的极小值
    const double curve_angle_tolerance_epsilon           = 0.01;  // 曲线角度容差
    enum curve_recursion_limit_e { curve_recursion_limit = 32 };  // 曲线递归深度限制


    //------------------------------------------------------------------------
    void curve3_inc::approximation_scale(double s) 
    { 
        m_scale = s;    // 设置曲线的近似比例
    }

    //------------------------------------------------------------------------
    double curve3_inc::approximation_scale() const 
    { 
        return m_scale; // 返回当前曲线的近似比例
    }

    //------------------------------------------------------------------------
    void curve3_inc::init(double x1, double y1, 
                          double x2, double y2, 
                          double x3, double y3)
    {
        m_start_x = x1; // 设置曲线起始点的 x 坐标
        m_start_y = y1; // 设置曲线起始点的 y 坐标
        m_end_x   = x3; // 设置曲线终点的 x 坐标
        m_end_y   = y3; // 设置曲线终点的 y 坐标

        double dx1 = x2 - x1; // 计算控制点 1 到起始点的 x 方向距离
        double dy1 = y2 - y1; // 计算控制点 1 到起始点的 y 方向距离
        double dx2 = x3 - x2; // 计算终点到控制点 2 的 x 方向距离
        double dy2 = y3 - y2; // 计算终点到控制点 2 的 y 方向距离

        double len = sqrt(dx1 * dx1 + dy1 * dy1) + sqrt(dx2 * dx2 + dy2 * dy2); // 计算控制点到起始点和终点到控制点的总长度

        m_num_steps = uround(len * 0.25 * m_scale); // 根据长度和近似比例计算步数

        if(m_num_steps < 4)
        {
            m_num_steps = 4;   // 如果步数少于4，则设置为4步
        }

        double subdivide_step  = 1.0 / m_num_steps; // 计算步长的倒数
        double subdivide_step2 = subdivide_step * subdivide_step; // 步长的平方

        double tmpx = (x1 - x2 * 2.0 + x3) * subdivide_step2; // 计算 x 方向上的临时变量
        double tmpy = (y1 - y2 * 2.0 + y3) * subdivide_step2; // 计算 y 方向上的临时变量

        m_saved_fx = m_fx = x1; // 保存起始点的 x 坐标
        m_saved_fy = m_fy = y1; // 保存起始点的 y 坐标
        
        m_saved_dfx = m_dfx = tmpx + (x2 - x1) * (2.0 * subdivide_step); // 计算起始点到控制点 2 的 x 方向导数
        m_saved_dfy = m_dfy = tmpy + (y2 - y1) * (2.0 * subdivide_step); // 计算起始点到控制点 2 的 y 方向导数

        m_ddfx = tmpx * 2.0; // 计算 x 方向的二阶导数
        m_ddfy = tmpy * 2.0; // 计算 y 方向的二阶导数

        m_step = m_num_steps; // 设置当前步数
    }

    //------------------------------------------------------------------------
    void curve3_inc::rewind(unsigned)
    {
        // 如果步数为0，则将步骤设置为-1，并返回
        if(m_num_steps == 0)
        {
            m_step = -1;
            return;
        }
        // 将当前步数设置为总步数
        m_step = m_num_steps;
        // 恢复保存的函数值和导数值
        m_fx   = m_saved_fx;
        m_fy   = m_saved_fy;
        m_dfx  = m_saved_dfx;
        m_dfy  = m_saved_dfy;
    }

    //------------------------------------------------------------------------
    // 获取贝塞尔曲线的顶点坐标
    unsigned curve3_inc::vertex(double* x, double* y)
    {
        // 如果步骤小于0，则返回停止路径指令
        if(m_step < 0) return path_cmd_stop;
        // 如果当前步数等于总步数，设置起始点坐标并返回移动到指令
        if(m_step == m_num_steps)
        {
            *x = m_start_x;
            *y = m_start_y;
            --m_step;
            return path_cmd_move_to;
        }
        // 如果当前步数为0，设置结束点坐标并返回直线到指令
        if(m_step == 0)
        {
            *x = m_end_x;
            *y = m_end_y;
            --m_step;
            return path_cmd_line_to;
        }
        // 计算下一个顶点坐标，并更新函数值和导数值
        m_fx  += m_dfx; 
        m_fy  += m_dfy;
        m_dfx += m_ddfx; 
        m_dfy += m_ddfy; 
        *x = m_fx;
        *y = m_fy;
        --m_step;
        return path_cmd_line_to;
    }

    //------------------------------------------------------------------------
    // 初始化三次贝塞尔曲线分割
    void curve3_div::init(double x1, double y1, 
                          double x2, double y2, 
                          double x3, double y3)
    {
        // 清空点集合
        m_points.remove_all();
        // 设置距离误差的平方为相对于近似比例的值
        m_distance_tolerance_square = 0.5 / m_approximation_scale;
        m_distance_tolerance_square *= m_distance_tolerance_square;
        // 计算贝塞尔曲线
        bezier(x1, y1, x2, y2, x3, y3);
        // 初始化计数器
        m_count = 0;
    }

    //------------------------------------------------------------------------
    // 递归计算三次贝塞尔曲线
    void curve3_div::recursive_bezier(double x1, double y1, 
                                      double x2, double y2, 
                                      double x3, double y3,
                                      unsigned level)
    {
        // 如果递归层级超过曲线递归限制，则直接返回，不再继续分割
        if(level > curve_recursion_limit) 
        {
            return;
        }
    
        // 计算所有线段的中点
        //----------------------
        double x12   = (x1 + x2) / 2;                
        double y12   = (y1 + y2) / 2;
        double x23   = (x2 + x3) / 2;
        double y23   = (y2 + y3) / 2;
        double x123  = (x12 + x23) / 2;
        double y123  = (y12 + y23) / 2;
    
        double dx = x3-x1;
        double dy = y3-y1;
        double d = fabs(((x2 - x3) * dy - (y2 - y3) * dx));
        double da;
    
        // 如果曲线的曲率大于曲线共线性阈值，则继续分割
        if(d > curve_collinearity_epsilon)
        { 
            // 普通情况
            //-----------------
            if(d * d <= m_distance_tolerance_square * (dx*dx + dy*dy))
            {
                // 如果曲率不超过距离容差值，则趋向于完成子分割
                //----------------------
                if(m_angle_tolerance < curve_angle_tolerance_epsilon)
                {
                    // 如果角度容差小于角度阈值极小值，则添加中点并返回
                    //----------------------
                    m_points.add(point_d(x123, y123));
                    return;
                }
    
                // 角度与尖峰条件
                //----------------------
                da = fabs(atan2(y3 - y2, x3 - x2) - atan2(y2 - y1, x2 - x1));
                if(da >= pi) da = 2*pi - da;
    
                if(da < m_angle_tolerance)
                {
                    // 最终可以停止递归
                    //----------------------
                    m_points.add(point_d(x123, y123));
                    return;                 
                }
            }
        }
        else
        {
            // 共线情况
            //------------------
            da = dx*dx + dy*dy;
            if(da == 0)
            {
                d = calc_sq_distance(x1, y1, x2, y2);
            }
            else
            {
                d = ((x2 - x1)*dx + (y2 - y1)*dy) / da;
                if(d > 0 && d < 1)
                {
                    // 简单的共线情况，1---2---3
                    // 可以仅保留两个端点
                    return;
                }
                if(d <= 0) d = calc_sq_distance(x2, y2, x1, y1);
                else if(d >= 1) d = calc_sq_distance(x2, y2, x3, y3);
                else            d = calc_sq_distance(x2, y2, x1 + d*dx, y1 + d*dy);
            }
            if(d < m_distance_tolerance_square)
            {
                // 如果距离小于距离容差的平方，则添加端点并返回
                m_points.add(point_d(x2, y2));
                return;
            }
        }
    
        // 继续分割曲线
        //----------------------
        recursive_bezier(x1, y1, x12, y12, x123, y123, level + 1); 
        recursive_bezier(x123, y123, x23, y23, x3, y3, level + 1); 
    }
    
    //------------------------------------------------------------------------
    void curve3_div::bezier(double x1, double y1, 
                            double x2, double y2, 
                            double x3, double y3)
    {
        // 将起始点添加到点集合中
        m_points.add(point_d(x1, y1));
        // 递归计算贝塞尔曲线的控制点和终点
        recursive_bezier(x1, y1, x2, y2, x3, y3, 0);
        // 将终点添加到点集合中
        m_points.add(point_d(x3, y3));
    }
    
    
    
    
    
    //------------------------------------------------------------------------
    void curve4_inc::approximation_scale(double s) 
    { 
        // 设置近似比例的私有成员变量
        m_scale = s;
    }
    
    //------------------------------------------------------------------------
    double curve4_inc::approximation_scale() const 
    { 
        // 返回当前的近似比例
        return m_scale;
    }
#if defined(_MSC_VER) && _MSC_VER <= 1200
//------------------------------------------------------------------------
static double MSC60_fix_ICE(double v) { return v; }
#endif



// 如果编译器是 Visual Studio 2008 或更早版本，则定义并实现了一个静态函数 MSC60_fix_ICE，
// 该函数返回其参数值，无变化。



//------------------------------------------------------------------------
void curve4_inc::init(double x1, double y1,
                      double x2, double y2,
                      double x3, double y3,
                      double x4, double y4)
{
    m_start_x = x1;
    m_start_y = y1;
    m_end_x   = x4;
    m_end_y   = y4;

    double dx1 = x2 - x1;
    double dy1 = y2 - y1;
    double dx2 = x3 - x2;
    double dy2 = y3 - y2;
    double dx3 = x4 - x3;
    double dy3 = y4 - y3;

    double len = (sqrt(dx1 * dx1 + dy1 * dy1) + 
                  sqrt(dx2 * dx2 + dy2 * dy2) + 
                  sqrt(dx3 * dx3 + dy3 * dy3)) * 0.25 * m_scale;

#if defined(_MSC_VER) && _MSC_VER <= 1200
    m_num_steps = uround(MSC60_fix_ICE(len));
#else
    m_num_steps = uround(len);
#endif

    if(m_num_steps < 4)
    {
        m_num_steps = 4;   
    }

    double subdivide_step  = 1.0 / m_num_steps;
    double subdivide_step2 = subdivide_step * subdivide_step;
    double subdivide_step3 = subdivide_step * subdivide_step * subdivide_step;

    double pre1 = 3.0 * subdivide_step;
    double pre2 = 3.0 * subdivide_step2;
    double pre4 = 6.0 * subdivide_step2;
    double pre5 = 6.0 * subdivide_step3;

    double tmp1x = x1 - x2 * 2.0 + x3;
    double tmp1y = y1 - y2 * 2.0 + y3;

    double tmp2x = (x2 - x3) * 3.0 - x1 + x4;
    double tmp2y = (y2 - y3) * 3.0 - y1 + y4;

    m_saved_fx = m_fx = x1;
    m_saved_fy = m_fy = y1;

    m_saved_dfx = m_dfx = (x2 - x1) * pre1 + tmp1x * pre2 + tmp2x * subdivide_step3;
    m_saved_dfy = m_dfy = (y2 - y1) * pre1 + tmp1y * pre2 + tmp2y * subdivide_step3;

    m_saved_ddfx = m_ddfx = tmp1x * pre4 + tmp2x * pre5;
    m_saved_ddfy = m_ddfy = tmp1y * pre4 + tmp2y * pre5;

    m_dddfx = tmp2x * pre5;
    m_dddfy = tmp2y * pre5;

    m_step = m_num_steps;
}



// 初始化曲线的控制点和计算相关的步数和预处理值。
// 设置曲线的起始点和结束点。
// 计算控制点之间的距离总和乘以缩放因子的四分之一。
// 根据编译器版本选择使用 MSC60_fix_ICE 函数或直接取整 len 来设置 m_num_steps。
// 如果计算得到的步数小于4，则设置为最小步数4。
// 计算曲线细分的步长及其平方和立方。
// 计算并设置曲线初始化时需要保存的各个导数和步长值。



//------------------------------------------------------------------------
void curve4_inc::rewind(unsigned)
{
    if(m_num_steps == 0)
    {
        m_step = -1;
        return;
    }
    m_step = m_num_steps;
    m_fx   = m_saved_fx;
    m_fy   = m_saved_fy;
    m_dfx  = m_saved_dfx;
    m_dfy  = m_saved_dfy;
    m_ddfx = m_saved_ddfx;
    m_ddfy = m_saved_ddfy;
}



// 重置曲线的迭代状态，以便重新开始绘制。
// 如果步数为0，则设置 m_step 为-1，表示曲线已结束。
// 否则，恢复保存的曲线状态，包括位置和导数信息。



//------------------------------------------------------------------------
unsigned curve4_inc::vertex(double* x, double* y)



// 提取当前曲线上的顶点坐标到提供的指针中。
    {
        // 如果步数小于0，返回路径命令停止
        if(m_step < 0) return path_cmd_stop;
        // 如果步数等于总步数，将起始点坐标赋给输出的x和y，减少步数并返回移动命令
        if(m_step == m_num_steps)
        {
            *x = m_start_x;
            *y = m_start_y;
            --m_step;
            return path_cmd_move_to;
        }
    
        // 如果步数为0，将结束点坐标赋给输出的x和y，减少步数并返回直线命令
        if(m_step == 0)
        {
            *x = m_end_x;
            *y = m_end_y;
            --m_step;
            return path_cmd_line_to;
        }
    
        // 更新当前点的坐标和速度增量
        m_fx   += m_dfx;
        m_fy   += m_dfy;
        m_dfx  += m_ddfx; 
        m_dfy  += m_ddfy; 
        m_ddfx += m_dddfx; 
        m_ddfy += m_dddfy; 
    
        // 将更新后的当前点坐标赋给输出的x和y，减少步数并返回直线命令
        *x = m_fx;
        *y = m_fy;
        --m_step;
        return path_cmd_line_to;
    }
    
    
    
    
    //------------------------------------------------------------------------
    // 初始化四次贝塞尔曲线的控制点和距离容差
    void curve4_div::init(double x1, double y1, 
                          double x2, double y2, 
                          double x3, double y3,
                          double x4, double y4)
    {
        // 清空点集合
        m_points.remove_all();
        // 根据近似尺度计算距离容差的平方值
        m_distance_tolerance_square = 0.5 / m_approximation_scale;
        m_distance_tolerance_square *= m_distance_tolerance_square;
        // 调用bezier函数计算贝塞尔曲线
        bezier(x1, y1, x2, y2, x3, y3, x4, y4);
        // 初始化点计数器
        m_count = 0;
    }
    
    //------------------------------------------------------------------------
    // 递归计算四次贝塞尔曲线
    void curve4_div::recursive_bezier(double x1, double y1, 
                                      double x2, double y2, 
                                      double x3, double y3, 
                                      double x4, double y4,
                                      unsigned level)
    }
    
    //------------------------------------------------------------------------
    // 计算四次贝塞尔曲线的点集合
    void curve4_div::bezier(double x1, double y1, 
                            double x2, double y2, 
                            double x3, double y3, 
                            double x4, double y4)
    {
        // 添加起始点到点集合
        m_points.add(point_d(x1, y1));
        // 递归计算贝塞尔曲线
        recursive_bezier(x1, y1, x2, y2, x3, y3, x4, y4, 0);
        // 添加结束点到点集合
        m_points.add(point_d(x4, y4));
    }
}



# 这是一个单独的右大括号 '}'，用于结束某个代码块或函数的定义。
```