# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_bspline.h`

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

// 防止头文件重复包含
#ifndef AGG_CONV_BSPLINE_INCLUDED
#define AGG_CONV_BSPLINE_INCLUDED

// 包含基本的 AGG 库文件和相关头文件
#include "agg_basics.h"
#include "agg_vcgen_bspline.h"
#include "agg_conv_adaptor_vcgen.h"

// 命名空间 agg 开始
namespace agg
{

    //---------------------------------------------------------conv_bspline
    // 模板类定义，继承自 conv_adaptor_vcgen 模板类，使用 VertexSource 和 vcgen_bspline
    template<class VertexSource> 
    struct conv_bspline : public conv_adaptor_vcgen<VertexSource, vcgen_bspline>
    {
        typedef conv_adaptor_vcgen<VertexSource, vcgen_bspline> base_type;

        // 构造函数，初始化基类
        conv_bspline(VertexSource& vs) : 
            conv_adaptor_vcgen<VertexSource, vcgen_bspline>(vs) {}

        // 设置插值步长的方法
        void   interpolation_step(double v) { base_type::generator().interpolation_step(v); }
        
        // 获取当前插值步长的方法
        double interpolation_step() const { return base_type::generator().interpolation_step(); }

    private:
        // 禁止拷贝构造函数和赋值运算符
        conv_bspline(const conv_bspline<VertexSource>&);
        const conv_bspline<VertexSource>& 
            operator = (const conv_bspline<VertexSource>&);
    };

} // 命名空间 agg 结束

// 结束防止头文件重复包含的条件
#endif
```