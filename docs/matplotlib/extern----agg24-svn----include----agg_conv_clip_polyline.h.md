# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_clip_polyline.h`

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
// polyline clipping converter
// There an optimized Liang-Barsky algorithm is used. 
// The algorithm doesn't optimize the degenerate edges, i.e. it will never
// break a closed polyline into two or more ones, instead, there will be 
// degenerate edges coinciding with the respective clipping boundaries.
// This is a sub-optimal solution, because that optimization would require 
// extra, rather expensive math while the rasterizer tolerates it quite well, 
// without any considerable overhead.
//
//----------------------------------------------------------------------------

#ifndef AGG_CONV_CLIP_polyline_INCLUDED
#define AGG_CONV_CLIP_polyline_INCLUDED

#include "agg_basics.h"
#include "agg_conv_adaptor_vpgen.h"
#include "agg_vpgen_clip_polyline.h"

namespace agg
{

    //=======================================================conv_clip_polyline
    // 模板类，继承自 conv_adaptor_vpgen，用于多边形线段裁剪
    template<class VertexSource> 
    struct conv_clip_polyline : public conv_adaptor_vpgen<VertexSource, vpgen_clip_polyline>
    {
        typedef conv_adaptor_vpgen<VertexSource, vpgen_clip_polyline> base_type;

        // 构造函数，初始化基类
        conv_clip_polyline(VertexSource& vs) : 
            conv_adaptor_vpgen<VertexSource, vpgen_clip_polyline>(vs) {}

        // 设定裁剪框的范围
        void clip_box(double x1, double y1, double x2, double y2)
        {
            base_type::vpgen().clip_box(x1, y1, x2, y2);
        }

        // 获取裁剪框的左上角和右下角的 x、y 坐标
        double x1() const { return base_type::vpgen().x1(); }
        double y1() const { return base_type::vpgen().y1(); }
        double x2() const { return base_type::vpgen().x2(); }
        double y2() const { return base_type::vpgen().y2(); }

    private:
        // 禁止拷贝构造和赋值操作，保证唯一性
        conv_clip_polyline(const conv_clip_polyline<VertexSource>&);
        const conv_clip_polyline<VertexSource>& 
            operator = (const conv_clip_polyline<VertexSource>&);
    };

}

#endif
```