# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_trans_single_path.h`

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

#ifndef AGG_TRANS_SINGLE_PATH_INCLUDED
#define AGG_TRANS_SINGLE_PATH_INCLUDED

#include "agg_basics.h"
#include "agg_vertex_sequence.h"

namespace agg
{

    // See also: agg_trans_single_path.cpp
    //
    //-------------------------------------------------------trans_single_path
    class trans_single_path
    {
        // 状态枚举，表示路径转换对象的状态
        enum status_e
        {
            initial,        // 初始状态
            making_path,    // 正在生成路径中
            ready           // 准备完毕
        };

    public:
        // 定义顶点存储类型为 vertex_sequence<vertex_dist, 6>
        typedef vertex_sequence<vertex_dist, 6> vertex_storage;

        // 默认构造函数
        trans_single_path();

        //--------------------------------------------------------------------
        // 设置基础长度
        void   base_length(double v)  { m_base_length = v; }

        // 获取基础长度
        double base_length() const { return m_base_length; }

        //--------------------------------------------------------------------
        // 设置是否保持 x 轴比例
        void preserve_x_scale(bool f) { m_preserve_x_scale = f;    }

        // 查询是否保持 x 轴比例
        bool preserve_x_scale() const { return m_preserve_x_scale; }

        //--------------------------------------------------------------------
        // 重置路径转换对象
        void reset();

        // 移动到指定坐标
        void move_to(double x, double y);

        // 添加直线到指定坐标
        void line_to(double x, double y);

        // 完成路径的最终化
        void finalize_path();

        //--------------------------------------------------------------------
        // 添加路径到当前对象
        template<class VertexSource> 
        void add_path(VertexSource& vs, unsigned path_id=0)
        {
            double x;
            double y;
            unsigned cmd;

            // 重置顶点源并遍历顶点
            vs.rewind(path_id);
            while (!is_stop(cmd = vs.vertex(&x, &y)))
            {
                if (is_move_to(cmd)) 
                {
                    move_to(x, y);
                }
                else 
                {
                    if (is_vertex(cmd))
                    {
                        line_to(x, y);
                    }
                }
            }
            // 完成路径的最终化
            finalize_path();
        }

        //--------------------------------------------------------------------
        // 计算路径的总长度
        double total_length() const;

        // 对给定坐标进行变换
        void transform(double *x, double *y) const;

    private:
        vertex_storage m_src_vertices;   // 源顶点序列
        double         m_base_length;    // 基础长度
        double         m_kindex;         // 系数索引
        status_e       m_status;         // 当前状态
        bool           m_preserve_x_scale;  // 是否保持 x 轴比例
    };

}

#endif


这段代码是一个 C++ 的头文件，定义了一个路径转换类 `trans_single_path`，用于处理和转换图形路径。
```