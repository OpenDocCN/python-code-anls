# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_smooth_poly1.h`

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
// Smooth polygon generator
//
//----------------------------------------------------------------------------
#ifndef AGG_CONV_SMOOTH_POLY1_INCLUDED
#define AGG_CONV_SMOOTH_POLY1_INCLUDED

#include "agg_basics.h"
#include "agg_vcgen_smooth_poly1.h"
#include "agg_conv_adaptor_vcgen.h"
#include "agg_conv_curve.h"


namespace agg
{

    //-------------------------------------------------------conv_smooth_poly1
    // conv_smooth_poly1 结构模板定义，继承自 conv_adaptor_vcgen 模板，用于平滑多边形生成
    template<class VertexSource> 
    struct conv_smooth_poly1 : 
    public conv_adaptor_vcgen<VertexSource, vcgen_smooth_poly1>
    {
        typedef conv_adaptor_vcgen<VertexSource, vcgen_smooth_poly1> base_type;

        // 构造函数，初始化基类 conv_adaptor_vcgen
        conv_smooth_poly1(VertexSource& vs) : 
            conv_adaptor_vcgen<VertexSource, vcgen_smooth_poly1>(vs)
        {
        }

        // 设置平滑值的方法
        void   smooth_value(double v) { base_type::generator().smooth_value(v); }
        // 获取当前平滑值的方法
        double smooth_value() const { return base_type::generator().smooth_value(); }

    private:
        // 禁止复制构造函数和赋值运算符
        conv_smooth_poly1(const conv_smooth_poly1<VertexSource>&);
        const conv_smooth_poly1<VertexSource>& 
            operator = (const conv_smooth_poly1<VertexSource>&);
    };



    //-------------------------------------------------conv_smooth_poly1_curve
    // conv_smooth_poly1_curve 结构模板定义，继承自 conv_curve，用于平滑曲线生成
    template<class VertexSource> 
    struct conv_smooth_poly1_curve : 
    public conv_curve<conv_smooth_poly1<VertexSource> >
    {
        // 构造函数，初始化成员变量和基类 conv_curve
        conv_smooth_poly1_curve(VertexSource& vs) :
            conv_curve<conv_smooth_poly1<VertexSource> >(m_smooth),
            m_smooth(vs)
        {
        }

        // 设置平滑值的方法
        void   smooth_value(double v) { m_smooth.generator().smooth_value(v); }
        // 获取当前平滑值的方法
        double smooth_value() const { return m_smooth.generator().smooth_value(); }

    private:
        // 禁止复制构造函数和赋值运算符
        conv_smooth_poly1_curve(const conv_smooth_poly1_curve<VertexSource>&);
        const conv_smooth_poly1_curve<VertexSource>& 
            operator = (const conv_smooth_poly1_curve<VertexSource>&);

        conv_smooth_poly1<VertexSource> m_smooth; // 内部平滑多边形生成器对象
    };

}

#endif // AGG_CONV_SMOOTH_POLY1_INCLUDED
```