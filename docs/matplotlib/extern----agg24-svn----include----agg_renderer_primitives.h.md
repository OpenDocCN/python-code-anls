# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_renderer_primitives.h`

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
// class renderer_primitives
//
//----------------------------------------------------------------------------

#ifndef AGG_RENDERER_PRIMITIVES_INCLUDED
#define AGG_RENDERER_PRIMITIVES_INCLUDED

#include "agg_basics.h"
#include "agg_renderer_base.h"
#include "agg_dda_line.h"
#include "agg_ellipse_bresenham.h"

namespace agg
{
    //-----------------------------------------------------renderer_primitives
    // 渲染器基本图元类模板
    template<class BaseRenderer> class renderer_primitives
    {
    private:
        // 基础渲染器类型指针
        base_ren_type* m_ren;
        // 填充颜色
        color_type m_fill_color;
        // 线条颜色
        color_type m_line_color;
        // 当前 x 坐标
        int m_curr_x;
        // 当前 y 坐标
        int m_curr_y;
    };

}

#endif
```