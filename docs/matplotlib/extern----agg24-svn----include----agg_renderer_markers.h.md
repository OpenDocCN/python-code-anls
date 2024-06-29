# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_renderer_markers.h`

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
// class renderer_markers
//
//----------------------------------------------------------------------------

#ifndef AGG_RENDERER_MARKERS_INCLUDED
#define AGG_RENDERER_MARKERS_INCLUDED

#include "agg_basics.h"
#include "agg_renderer_primitives.h"

namespace agg
{

    //---------------------------------------------------------------marker_e
    // 枚举类型，定义了不同的标记类型，用于标识不同的绘制符号
    enum marker_e
    {
        marker_square,
        marker_diamond,
        marker_circle,
        marker_crossed_circle,
        marker_semiellipse_left,
        marker_semiellipse_right,
        marker_semiellipse_up,
        marker_semiellipse_down,
        marker_triangle_left,
        marker_triangle_right,
        marker_triangle_up,
        marker_triangle_down,
        marker_four_rays,
        marker_cross,
        marker_x,
        marker_dash,
        marker_dot,
        marker_pixel,
        
        end_of_markers
    };



    //--------------------------------------------------------renderer_markers
    // 模板类，继承自renderer_primitives<BaseRenderer>
    template<class BaseRenderer> class renderer_markers :
    public renderer_primitives<BaseRenderer>
    {
    };

}

#endif
```