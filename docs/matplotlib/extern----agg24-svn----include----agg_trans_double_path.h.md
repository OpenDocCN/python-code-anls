# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_trans_double_path.h`

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

#ifndef AGG_TRANS_DOUBLE_PATH_INCLUDED
#define AGG_TRANS_DOUBLE_PATH_INCLUDED

#include "agg_basics.h"
#include "agg_vertex_sequence.h"

namespace agg
{
    
    // See also: agg_trans_double_path.cpp
    //
    //-------------------------------------------------------trans_double_path
    // trans_double_path 类定义，用于处理双路径变换
    class trans_double_path
    {
        // 内部状态枚举
        enum status_e
        {
            initial,      // 初始状态
            making_path,  // 正在构造路径
            ready         // 就绪状态
        };
    // 定义公共部分，包含顶点序列的类型定义
    public:
        typedef vertex_sequence<vertex_dist, 6> vertex_storage;

        // 默认构造函数
        trans_double_path();

        //--------------------------------------------------------------------
        // 设置或获取基础长度
        void   base_length(double v)  { m_base_length = v; }
        double base_length() const { return m_base_length; }

        //--------------------------------------------------------------------
        // 设置或获取基础高度
        void   base_height(double v)  { m_base_height = v; }
        double base_height() const { return m_base_height; }

        //--------------------------------------------------------------------
        // 设置或获取是否保持 x 缩放
        void preserve_x_scale(bool f) { m_preserve_x_scale = f;    }
        bool preserve_x_scale() const { return m_preserve_x_scale; }

        //--------------------------------------------------------------------
        // 重置路径
        void reset();
        // 移动到指定点（路径1）
        void move_to1(double x, double y);
        // 添加直线到指定点（路径1）
        void line_to1(double x, double y);
        // 移动到指定点（路径2）
        void move_to2(double x, double y);
        // 添加直线到指定点（路径2）
        void line_to2(double x, double y);
        // 完成路径添加操作
        void finalize_paths();

        //--------------------------------------------------------------------
        // 添加两个路径到当前对象
        template<class VertexSource1, class VertexSource2> 
        void add_paths(VertexSource1& vs1, VertexSource2& vs2, 
                       unsigned path1_id=0, unsigned path2_id=0)
        {
            double x;
            double y;

            unsigned cmd;

            // 处理第一个顶点源的路径
            vs1.rewind(path1_id);
            while(!is_stop(cmd = vs1.vertex(&x, &y)))
            {
                if(is_move_to(cmd)) 
                {
                    move_to1(x, y);
                }
                else 
                {
                    if(is_vertex(cmd))
                    {
                        line_to1(x, y);
                    }
                }
            }

            // 处理第二个顶点源的路径
            vs2.rewind(path2_id);
            while(!is_stop(cmd = vs2.vertex(&x, &y)))
            {
                if(is_move_to(cmd)) 
                {
                    move_to2(x, y);
                }
                else 
                {
                    if(is_vertex(cmd))
                    {
                        line_to2(x, y);
                    }
                }
            }
            // 完成路径添加操作
            finalize_paths();
        }

        //--------------------------------------------------------------------
        // 计算路径1的总长度
        double total_length1() const;
        // 计算路径2的总长度
        double total_length2() const;
        // 对给定坐标进行变换
        void transform(double *x, double *y) const;
    // 声明私有方法 finalize_path，用于计算路径的最终值，并返回该值的双精度浮点数
    private:
        double finalize_path(vertex_storage& vertices);
        
        // 声明私有方法 transform1，用给定的顶点存储对象和系数进行变换，
        // 将变换后的结果写入提供的指针 x 和 y 所指向的位置
        void transform1(const vertex_storage& vertices, 
                        double kindex, double kx,
                        double *x, double* y) const;

        // 声明私有成员变量
        vertex_storage m_src_vertices1;  // 源顶点存储对象1
        vertex_storage m_src_vertices2;  // 源顶点存储对象2
        double         m_base_length;    // 基础长度
        double         m_base_height;    // 基础高度
        double         m_kindex1;        // 系数1
        double         m_kindex2;        // 系数2
        status_e       m_status1;        // 状态1
        status_e       m_status2;        // 状态2
        bool           m_preserve_x_scale; // 是否保留 X 轴比例
    };
}



#endif


注释：


// 这里是 C/C++ 中的预处理指令结束符号
}



// 这是条件编译的结束指令，用于结束 #ifdef 或 #ifndef 块
#endif
```