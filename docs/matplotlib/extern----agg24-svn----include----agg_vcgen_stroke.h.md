# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_vcgen_stroke.h`

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

// 如果AGG_VCGEN_STROKE_INCLUDED未定义，则定义它，避免重复包含
#ifndef AGG_VCGEN_STROKE_INCLUDED
#define AGG_VCGEN_STROKE_INCLUDED

// 包含数学路径描边相关的头文件
#include "agg_math_stroke.h"

// 使用agg命名空间
namespace agg
{

    //============================================================vcgen_stroke
    //
    // See Implementation agg_vcgen_stroke.cpp
    // Stroke generator
    //
    //------------------------------------------------------------------------
    // vcgen_stroke类，路径描边生成器
    class vcgen_stroke
    {
        // 定义状态枚举
        enum status_e
        {
            initial,      // 初始状态
            ready,        // 准备就绪状态
            cap1,         // 终点样式1
            cap2,         // 终点样式2
            outline1,     // 轮廓1
            close_first,  // 关闭第一个
            outline2,     // 轮廓2
            out_vertices, // 输出顶点
            end_poly1,    // 结束多边形1
            end_poly2,    // 结束多边形2
            stop          // 停止状态
        };
    // 定义了一个公有类 vcgen_stroke，包含了几个类型别名和成员变量
    public:
        // 使用 vertex_dist 类型和容量为 6 的顶点序列作为 vertex_storage 类型
        typedef vertex_sequence<vertex_dist, 6> vertex_storage;
        // 使用 point_d 类型和容量为 6 的 POD（Plain Old Data）位向量作为 coord_storage 类型
        typedef pod_bvector<point_d, 6>         coord_storage;
    
        // 默认构造函数，不执行任何操作
        vcgen_stroke();
    
        // 设置线段端点样式
        void line_cap(line_cap_e lc)     { m_stroker.line_cap(lc); }
        // 设置线段连接样式
        void line_join(line_join_e lj)   { m_stroker.line_join(lj); }
        // 设置内部连接样式
        void inner_join(inner_join_e ij) { m_stroker.inner_join(ij); }
    
        // 获取当前线段端点样式
        line_cap_e   line_cap()   const { return m_stroker.line_cap(); }
        // 获取当前线段连接样式
        line_join_e  line_join()  const { return m_stroker.line_join(); }
        // 获取当前内部连接样式
        inner_join_e inner_join() const { return m_stroker.inner_join(); }
    
        // 设置线段宽度
        void width(double w) { m_stroker.width(w); }
        // 设置斜接限制
        void miter_limit(double ml) { m_stroker.miter_limit(ml); }
        // 设置斜接角度限制
        void miter_limit_theta(double t) { m_stroker.miter_limit_theta(t); }
        // 设置内部斜接限制
        void inner_miter_limit(double ml) { m_stroker.inner_miter_limit(ml); }
        // 设置近似比例尺
        void approximation_scale(double as) { m_stroker.approximation_scale(as); }
    
        // 获取当前线段宽度
        double width() const { return m_stroker.width(); }
        // 获取当前斜接限制
        double miter_limit() const { return m_stroker.miter_limit(); }
        // 获取当前内部斜接限制
        double inner_miter_limit() const { return m_stroker.inner_miter_limit(); }
        // 获取当前近似比例尺
        double approximation_scale() const { return m_stroker.approximation_scale(); }
    
        // 设置线段的短缩距离
        void shorten(double s) { m_shorten = s; }
        // 获取当前线段的短缩距离
        double shorten() const { return m_shorten; }
    
        // 顶点生成器接口
    
        // 清空所有顶点数据
        void remove_all();
        // 添加顶点，包括 x 坐标、y 坐标和命令（unsigned cmd）
        void add_vertex(double x, double y, unsigned cmd);
    
        // 顶点源接口
    
        // 重置到指定路径 ID 的顶点数据
        void rewind(unsigned path_id);
        // 获取下一个顶点的坐标，返回值为顶点命令
        unsigned vertex(double* x, double* y);
    
    private:
        // 禁止拷贝构造函数和赋值运算符重载
        vcgen_stroke(const vcgen_stroke&);
        const vcgen_stroke& operator = (const vcgen_stroke&);
    
        // 线段生成器使用 coord_storage 类型的数学线段对象
        math_stroke<coord_storage> m_stroker;
        // 源顶点数据存储
        vertex_storage             m_src_vertices;
        // 输出顶点数据存储
        coord_storage              m_out_vertices;
        // 短缩距离
        double                     m_shorten;
        // 标志路径是否闭合
        unsigned                   m_closed;
        // 当前状态
        status_e                   m_status;
        // 前一个状态
        status_e                   m_prev_status;
        // 源顶点索引
        unsigned                   m_src_vertex;
        // 输出顶点索引
        unsigned                   m_out_vertex;
    };
}

#endif



}



#endif



// 结束一个 C++ 的函数定义或者一个 C++ 头文件的条件编译指令
}
// 结束函数定义
#endif
// 结束条件编译指令



// 这段代码是 C++ 中的结尾部分，用于结束一个函数的定义或条件编译指令
```