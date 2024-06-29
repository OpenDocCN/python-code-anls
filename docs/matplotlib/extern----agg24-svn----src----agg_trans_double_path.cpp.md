# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_trans_double_path.cpp`

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

#include "agg_math.h"
#include "agg_trans_double_path.h"

namespace agg
{

    //------------------------------------------------------------------------
    // Constructor initializing member variables.
    trans_double_path::trans_double_path() :
        m_kindex1(0.0),                          // Initialize m_kindex1 to 0.0
        m_kindex2(0.0),                          // Initialize m_kindex2 to 0.0
        m_base_length(0.0),                       // Initialize m_base_length to 0.0
        m_base_height(1.0),                       // Initialize m_base_height to 1.0
        m_status1(initial),                       // Initialize m_status1 to initial
        m_status2(initial),                       // Initialize m_status2 to initial
        m_preserve_x_scale(true)                  // Initialize m_preserve_x_scale to true
    {
    }


    //------------------------------------------------------------------------
    // Reset function clears vertex lists and resets status variables.
    void trans_double_path::reset()
    {
        m_src_vertices1.remove_all();             // Clear vertices list 1
        m_src_vertices2.remove_all();             // Clear vertices list 2
        m_kindex1 = 0.0;                          // Reset m_kindex1 to 0.0
        m_kindex1 = 0.0;                          // Reset m_kindex2 to 0.0 (should be m_kindex2)
        m_status1 = initial;                      // Reset m_status1 to initial
        m_status2 = initial;                      // Reset m_status2 to initial
    }


    //------------------------------------------------------------------------
    // Function to initiate a new path in the first set of vertices.
    void trans_double_path::move_to1(double x, double y)
    {
        if(m_status1 == initial)
        {
            m_src_vertices1.modify_last(vertex_dist(x, y));  // Modify last vertex in list 1
            m_status1 = making_path;                         // Set status1 to making_path
        }
        else
        {
            line_to1(x, y);                                  // Call line_to1 if not initial status
        }
    }


    //------------------------------------------------------------------------
    // Function to add a line segment to the first set of vertices.
    void trans_double_path::line_to1(double x, double y)
    {
        if(m_status1 == making_path)
        {
            m_src_vertices1.add(vertex_dist(x, y));          // Add vertex to list 1
        }
    }


    //------------------------------------------------------------------------
    // Function to initiate a new path in the second set of vertices.
    void trans_double_path::move_to2(double x, double y)
    {
        if(m_status2 == initial)
        {
            m_src_vertices2.modify_last(vertex_dist(x, y));  // Modify last vertex in list 2
            m_status2 = making_path;                         // Set status2 to making_path
        }
        else
        {
            line_to2(x, y);                                  // Call line_to2 if not initial status
        }
    }


    //------------------------------------------------------------------------
    // Function to add a line segment to the second set of vertices.
    void trans_double_path::line_to2(double x, double y)
    {
        if(m_status2 == making_path)
        {
            m_src_vertices2.add(vertex_dist(x, y));          // Add vertex to list 2
        }
    }


    //------------------------------------------------------------------------
    // Function to finalize the path by copying vertices to the provided storage.
    double trans_double_path::finalize_path(vertex_storage& vertices)
    {
        unsigned i;  // 无符号整数变量 i，用于循环索引
        double dist;  // 双精度浮点数变量 dist，用于存储距离值
        double d;  // 双精度浮点数变量 d
    
        vertices.close(false);  // 调用 vertices 对象的 close 方法，参数为 false
    
        if(vertices.size() > 2)  // 如果 vertices 容器中的元素个数大于 2
        {
            // 检查倒数第二个顶点的距离是否小于倒数第三个顶点的距离的十倍
            if(vertices[vertices.size() - 2].dist * 10.0 < vertices[vertices.size() - 3].dist)
            {
                d = vertices[vertices.size() - 3].dist + vertices[vertices.size() - 2].dist;  // 计算 d 的值为倒数第三个顶点距离加上倒数第二个顶点距离
    
                vertices[vertices.size() - 2] = vertices[vertices.size() - 1];  // 将倒数第一个顶点复制到倒数第二个顶点位置
                vertices.remove_last();  // 移除最后一个顶点
                vertices[vertices.size() - 2].dist = d;  // 将倒数第二个顶点的距离设为 d
            }
        }
    
        dist = 0;  // 初始化 dist 为 0
    
        // 遍历 vertices 容器中的所有顶点
        for(i = 0; i < vertices.size(); i++)
        {
            vertex_dist& v = vertices[i];  // 获取当前顶点 v 的引用
            d = v.dist;  // 将当前顶点的距离赋给 d
            v.dist = dist;  // 将当前顶点的距离设为累计距离 dist
            dist += d;  // 更新累计距离为原始累计距离加上当前顶点的距离
        }
    
        return (vertices.size() - 1) / dist;  // 返回顶点数减一除以累计距离
    }
    
    
    //------------------------------------------------------------------------
    void trans_double_path::finalize_paths()
    {
        // 如果两条路径均处于 making_path 状态且都有超过一个顶点
        if(m_status1 == making_path && m_src_vertices1.size() > 1 &&
           m_status2 == making_path && m_src_vertices2.size() > 1)
        {
            m_kindex1 = finalize_path(m_src_vertices1);  // 调用 finalize_path 方法处理第一条路径并赋值给 m_kindex1
            m_kindex2 = finalize_path(m_src_vertices2);  // 调用 finalize_path 方法处理第二条路径并赋值给 m_kindex2
            m_status1 = ready;  // 将第一条路径状态设置为 ready
            m_status2 = ready;  // 将第二条路径状态设置为 ready
        }
    }
    
    
    //------------------------------------------------------------------------
    double trans_double_path::total_length1() const
    {
        // 如果基础长度大于等于 1e-10，则返回基础长度
        if(m_base_length >= 1e-10) return m_base_length;
        // 否则，如果第一条路径状态为 ready，则返回第一条路径最后一个顶点的距离；否则返回 0.0
        return (m_status1 == ready) ? m_src_vertices1[m_src_vertices1.size() - 1].dist : 0.0;
    }
    
    
    //------------------------------------------------------------------------
    double trans_double_path::total_length2() const
    {
        // 如果基础长度大于等于 1e-10，则返回基础长度
        if(m_base_length >= 1e-10) return m_base_length;
        // 否则，如果第二条路径状态为 ready，则返回第二条路径最后一个顶点的距离；否则返回 0.0
        return (m_status2 == ready) ? m_src_vertices2[m_src_vertices2.size() - 1].dist : 0.0;
    }
    
    
    //------------------------------------------------------------------------
    void trans_double_path::transform1(const vertex_storage& vertices, 
                                       double kindex, double kx, 
                                       double *x, double* y) const
    {
        // 此方法实现的具体功能需要参考其定义的具体内容，这里不做进一步的注释
    }
    {
        // 初始化变量，定义起始点坐标和增量
        double x1 = 0.0;
        double y1 = 0.0;
        double dx = 1.0;
        double dy = 1.0;
        double d  = 0.0;
        double dd = 1.0;
        
        // 对输入的 *x 进行乘以 kx 的操作
        *x *= kx;
        
        // 如果 *x 小于 0.0，执行左侧外推
        if (*x < 0.0)
        {
            // 左侧外推
            //--------------------------
            // 获取顶点集合中第一个顶点的坐标
            x1 = vertices[0].x;
            y1 = vertices[0].y;
            // 计算第一个和第二个顶点的坐标差值
            dx = vertices[1].x - x1;
            dy = vertices[1].y - y1;
            // 获取第一个和第二个顶点之间的距离差值
            dd = vertices[1].dist - vertices[0].dist;
            // 设置 d 为 *x 的值
            d  = *x;
        }
        else if (*x > vertices[vertices.size() - 1].dist)
        {
            // 右侧外推
            //--------------------------
            // 获取顶点集合中倒数第二个和最后一个顶点的索引
            unsigned i = vertices.size() - 2;
            unsigned j = vertices.size() - 1;
            // 获取最后一个顶点的坐标
            x1 = vertices[j].x;
            y1 = vertices[j].y;
            // 计算倒数第二个和最后一个顶点的坐标差值
            dx = x1 - vertices[i].x;
            dy = y1 - vertices[i].y;
            // 获取倒数第二个和最后一个顶点之间的距离差值
            dd = vertices[j].dist - vertices[i].dist;
            // 设置 d 为 *x 减去最后一个顶点的距离
            d  = *x - vertices[j].dist;
        }
        else
        {
            // 内插
            //--------------------------
            unsigned i = 0;
            unsigned j = vertices.size() - 1;
            
            // 如果保持 x 比例尺，则进行二分查找
            if (m_preserve_x_scale)
            {
                unsigned k;
                for (i = 0; (j - i) > 1; ) 
                {
                    // 二分查找 *x 在顶点集合中的位置
                    if (*x < vertices[k = (i + j) >> 1].dist) 
                    {
                        j = k; 
                    }
                    else 
                    {
                        i = k;
                    }
                }
                // 获取起始距离和结束距离
                d  = vertices[i].dist;
                dd = vertices[j].dist - d;
                // 设置 d 为 *x 减去起始距离
                d  = *x - d;
            }
            else
            {
                // 否则直接基于 *x 计算索引
                i = unsigned(*x * kindex);
                j = i + 1;
                // 获取起始距离和结束距离
                dd = vertices[j].dist - vertices[i].dist;
                // 设置 d 为 ((*x * kindex) - i) * dd
                d = ((*x * kindex) - i) * dd;
            }
            
            // 获取起始点坐标和增量
            x1 = vertices[i].x;
            y1 = vertices[i].y;
            dx = vertices[j].x - x1;
            dy = vertices[j].y - y1;
        }
        
        // 计算最终的 *x 和 *y 值
        *x = x1 + dx * d / dd;
        *y = y1 + dy * d / dd;
    }
    
    
    
    //------------------------------------------------------------------------
    void trans_double_path::transform(double *x, double *y) const
    {
        // 检查两个状态是否都准备好
        if(m_status1 == ready && m_status2 == ready)
        {
            // 如果基础长度大于 1e-10
            if(m_base_length > 1e-10)
            {
                // 根据源顶点数组的最后一个顶点的距离与基础长度调整 *x 的值
                *x *= m_src_vertices1[m_src_vertices1.size() - 1].dist / m_base_length;
            }
    
            // 复制 *x 和 *y 到局部变量 x1 和 y1
            double x1 = *x;
            double y1 = *y;
            // 复制 *x 和 *y 到局部变量 x2 和 y2
            double x2 = *x;
            double y2 = *y;
            // 计算两个顶点数组最后一个顶点的距离比率
            double dd = m_src_vertices2[m_src_vertices2.size() - 1].dist /
                        m_src_vertices1[m_src_vertices1.size() - 1].dist;
    
            // 对第一个顶点数组应用变换函数，更新 x1 和 y1
            transform1(m_src_vertices1, m_kindex1, 1.0, &x1, &y1);
            // 对第二个顶点数组应用变换函数，更新 x2 和 y2
            transform1(m_src_vertices2, m_kindex2, dd,  &x2, &y2);
    
            // 更新 *x，使用插值计算基于基础高度的新 x 值
            *x = x1 + *y * (x2 - x1) / m_base_height;
            // 更新 *y，使用插值计算基于基础高度的新 y 值
            *y = y1 + *y * (y2 - y1) / m_base_height;
        }
    }
}



# 这行代码表示一个代码块的结束，通常用于结束一个函数、循环、条件语句或类定义。
```