# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_math_stroke.h`

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
// Stroke math
//
//----------------------------------------------------------------------------

#ifndef AGG_STROKE_MATH_INCLUDED
#define AGG_STROKE_MATH_INCLUDED

#include "agg_math.h"
#include "agg_vertex_sequence.h"

namespace agg
{
    //-------------------------------------------------------------line_cap_e
    enum line_cap_e
    {
        butt_cap,           // 末端为平直
        square_cap,         // 末端为方形
        round_cap           // 末端为圆形
    };

    //------------------------------------------------------------line_join_e
    enum line_join_e
    {
        miter_join         = 0, // 尖角连接
        miter_join_revert  = 1, // 反向尖角连接
        round_join         = 2, // 圆角连接
        bevel_join         = 3, // 斜角连接
        miter_join_round   = 4  // 圆角尖角连接
    };


    //-----------------------------------------------------------inner_join_e
    enum inner_join_e
    {
        inner_bevel,        // 内部斜角连接
        inner_miter,        // 内部尖角连接
        inner_jag,          // 内部齿状连接
        inner_round         // 内部圆角连接
    };

    //------------------------------------------------------------math_stroke
    // 数学描边类模板，用于生成描边形状的顶点序列
    template<class VertexConsumer> class math_stroke
    {
    // 公共部分开始
    public:
        // 为了简化类型调用，定义顶点消费者的值类型为坐标类型
        typedef typename VertexConsumer::value_type coord_type;

        // 默认构造函数，初始化线段属性
        math_stroke();

        // 设置线段端点样式
        void line_cap(line_cap_e lc)     { m_line_cap = lc; }
        // 设置线段连接处样式
        void line_join(line_join_e lj)   { m_line_join = lj; }
        // 设置内部连接样式
        void inner_join(inner_join_e ij) { m_inner_join = ij; }

        // 获取线段端点样式
        line_cap_e   line_cap()   const { return m_line_cap; }
        // 获取线段连接处样式
        line_join_e  line_join()  const { return m_line_join; }
        // 获取内部连接样式
        inner_join_e inner_join() const { return m_inner_join; }

        // 设置线段宽度
        void width(double w);
        // 设置斜接限制
        void miter_limit(double ml) { m_miter_limit = ml; }
        // 根据斜接角度设置斜接限制
        void miter_limit_theta(double t);
        // 设置内部斜接限制
        void inner_miter_limit(double ml) { m_inner_miter_limit = ml; }
        // 设置近似比例尺度
        void approximation_scale(double as) { m_approx_scale = as; }

        // 获取线段宽度
        double width() const { return m_width * 2.0; }
        // 获取斜接限制
        double miter_limit() const { return m_miter_limit; }
        // 获取内部斜接限制
        double inner_miter_limit() const { return m_inner_miter_limit; }
        // 获取近似比例尺度
        double approximation_scale() const { return m_approx_scale; }

        // 计算线段端点处的处理
        void calc_cap(VertexConsumer& vc,
                      const vertex_dist& v0, 
                      const vertex_dist& v1, 
                      double len);

        // 计算线段连接处的处理
        void calc_join(VertexConsumer& vc,
                       const vertex_dist& v0, 
                       const vertex_dist& v1, 
                       const vertex_dist& v2,
                       double len1, 
                       double len2);

    private:
        // 内联函数，添加顶点到顶点消费者
        AGG_INLINE void add_vertex(VertexConsumer& vc, double x, double y)
        {
            vc.add(coord_type(x, y));
        }

        // 计算圆弧的顶点
        void calc_arc(VertexConsumer& vc,
                      double x,   double y, 
                      double dx1, double dy1, 
                      double dx2, double dy2);

        // 计算斜接
        void calc_miter(VertexConsumer& vc,
                        const vertex_dist& v0, 
                        const vertex_dist& v1, 
                        const vertex_dist& v2,
                        double dx1, double dy1, 
                        double dx2, double dy2,
                        line_join_e lj,
                        double mlimit,
                        double dbevel);

        // 线段宽度
        double       m_width;
        // 绝对线段宽度
        double       m_width_abs;
        // 线段宽度的精度
        double       m_width_eps;
        // 线段宽度的符号
        int          m_width_sign;
        // 斜接限制
        double       m_miter_limit;
        // 内部斜接限制
        double       m_inner_miter_limit;
        // 近似比例尺度
        double       m_approx_scale;
        // 线段端点样式
        line_cap_e   m_line_cap;
        // 线段连接处样式
        line_join_e  m_line_join;
        // 内部连接样式
        inner_join_e m_inner_join;
    };

    //-----------------------------------------------------------------------
    // 模板类 math_stroke 的默认构造函数实现
    template<class VC> math_stroke<VC>::math_stroke() :
        // 初始化线段属性的默认值
        m_width(0.5),
        m_width_abs(0.5),
        m_width_eps(0.5/1024.0),
        m_width_sign(1),
        m_miter_limit(4.0),
        m_inner_miter_limit(1.01),
        m_approx_scale(1.0),
        m_line_cap(butt_cap),
        m_line_join(miter_join),
        m_inner_join(inner_miter)
    {
    }
    //-----------------------------------------------------------------------
    // 设置线条宽度，根据给定的宽度值 w 调整线条宽度的内部变量。宽度为 w 的一半。
    template<class VC> void math_stroke<VC>::width(double w)
    { 
        m_width = w * 0.5;  // 将输入的宽度值乘以0.5，存储在 m_width 中
        if(m_width < 0)
        {
            m_width_abs  = -m_width;   // 如果宽度为负数，取其绝对值并存储在 m_width_abs 中
            m_width_sign = -1;         // 设置线条宽度的符号为负数
        }
        else
        {
            m_width_abs  = m_width;    // 如果宽度为非负数，直接存储在 m_width_abs 中
            m_width_sign = 1;          // 设置线条宽度的符号为正数
        }
        m_width_eps = m_width / 1024.0;  // 计算一个小的宽度值，存储在 m_width_eps 中
    }
    
    //-----------------------------------------------------------------------
    // 设置斜接限制角度，根据给定的角度 t 计算斜接限制值并存储在 m_miter_limit 中
    template<class VC> void math_stroke<VC>::miter_limit_theta(double t)
    { 
        m_miter_limit = 1.0 / sin(t * 0.5) ;  // 根据角度 t 计算斜接限制，存储在 m_miter_limit 中
    }
    
    //-----------------------------------------------------------------------
    // 计算弧线，根据给定的参数计算弧线的顶点，并添加到顶点集合 vc 中
    template<class VC> 
    void math_stroke<VC>::calc_arc(VC& vc,
                                   double x,   double y, 
                                   double dx1, double dy1, 
                                   double dx2, double dy2)
    {
        double a1 = atan2(dy1 * m_width_sign, dx1 * m_width_sign);  // 计算起始角度 a1
        double a2 = atan2(dy2 * m_width_sign, dx2 * m_width_sign);  // 计算结束角度 a2
        double da = a1 - a2;  // 计算角度差 da
    
        // 根据当前线条宽度计算一个角度值，并将其存储在 da 中
        da = acos(m_width_abs / (m_width_abs + 0.125 / m_approx_scale)) * 2;
    
        // 添加起始顶点到顶点集合 vc 中
        add_vertex(vc, x + dx1, y + dy1);
    
        if(m_width_sign > 0)
        {
            if(a1 > a2) a2 += 2 * pi;  // 调整角度范围，确保顺时针方向
            n = int((a2 - a1) / da);   // 计算顶点数目 n
            da = (a2 - a1) / (n + 1);  // 计算角度步长 da
            a1 += da;                 // 调整起始角度 a1
            for(i = 0; i < n; i++)
            {
                // 计算新的顶点位置，并添加到顶点集合 vc 中
                add_vertex(vc, x + cos(a1) * m_width, y + sin(a1) * m_width);
                a1 += da;  // 更新角度 a1
            }
        }
        else
        {
            if(a1 < a2) a2 -= 2 * pi;  // 调整角度范围，确保逆时针方向
            n = int((a1 - a2) / da);   // 计算顶点数目 n
            da = (a1 - a2) / (n + 1);  // 计算角度步长 da
            a1 -= da;                 // 调整起始角度 a1
            for(i = 0; i < n; i++)
            {
                // 计算新的顶点位置，并添加到顶点集合 vc 中
                add_vertex(vc, x + cos(a1) * m_width, y + sin(a1) * m_width);
                a1 -= da;  // 更新角度 a1
            }
        }
    
        // 添加结束顶点到顶点集合 vc 中
        add_vertex(vc, x + dx2, y + dy2);
    }
    {
        // 清空顶点缓冲区，准备接收新的顶点数据
        vc.remove_all();
    
        // 计算第一条线段的方向向量的单位法向量
        double dx1 = (v1.y - v0.y) / len;
        double dy1 = (v1.x - v0.x) / len;
    
        // 计算第二条线段的单位法向量，初始为零向量
        double dx2 = 0;
        double dy2 = 0;
    
        // 将第一条线段的单位法向量乘以线宽，得到实际的偏移量
        dx1 *= m_width;
        dy1 *= m_width;
    
        // 根据线帽类型，计算第二条线段的单位法向量
        if(m_line_cap != round_cap)
        {
            if(m_line_cap == square_cap)
            {
                dx2 = dy1 * m_width_sign;
                dy2 = dx1 * m_width_sign;
            }
            // 添加两个端点到顶点缓冲区
            add_vertex(vc, v0.x - dx1 - dx2, v0.y + dy1 - dy2);
            add_vertex(vc, v0.x + dx1 - dx2, v0.y - dy1 - dy2);
        }
        else
        {
            // 计算夹角以确定连接点的位置
            double da = acos(m_width_abs / (m_width_abs + 0.125 / m_approx_scale)) * 2;
            double a1;
            int i;
            int n = int(pi / da);
    
            // 计算每段连接点之间的夹角
            da = pi / (n + 1);
            // 添加起始连接点到顶点缓冲区
            add_vertex(vc, v0.x - dx1, v0.y + dy1);
    
            // 根据线宽正负号，确定连接点的位置
            if(m_width_sign > 0)
            {
                a1 = atan2(dy1, -dx1);
                a1 += da;
                // 添加每个连接点到顶点缓冲区
                for(i = 0; i < n; i++)
                {
                    add_vertex(vc, v0.x + cos(a1) * m_width, 
                                   v0.y + sin(a1) * m_width);
                    a1 += da;
                }
            }
            else
            {
                a1 = atan2(-dy1, dx1);
                a1 -= da;
                // 添加每个连接点到顶点缓冲区
                for(i = 0; i < n; i++)
                {
                    add_vertex(vc, v0.x + cos(a1) * m_width, 
                                   v0.y + sin(a1) * m_width);
                    a1 -= da;
                }
            }
            // 添加结束连接点到顶点缓冲区
            add_vertex(vc, v0.x + dx1, v0.y - dy1);
        }
    }
    
    //-----------------------------------------------------------------------
    template<class VC> 
    void math_stroke<VC>::calc_join(VC& vc,
                                    const vertex_dist& v0, 
                                    const vertex_dist& v1, 
                                    const vertex_dist& v2,
                                    double len1, 
                                    double len2)
    {
        // 此处为计算连接点的函数模板，根据给定的参数计算连接处的顶点
    }
}


这行代码是一个闭合大括号 `}`，用于结束一个代码块或者控制流结构，但是在这个片段中看不到前面的代码，因此具体它结束的是什么结构无法确定。


#endif


这行代码是C/C++预处理器指令，用于条件编译。`#endif` 用于结束一个条件编译块，配合 `#ifdef` 或者 `#ifndef` 来控制在特定条件下编译代码段。
```