# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_rasterizer_outline.h`

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

#ifndef AGG_RASTERIZER_OUTLINE_INCLUDED
#define AGG_RASTERIZER_OUTLINE_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //======================================================rasterizer_outline
    // rasterizer_outline 模板类定义
    template<class Renderer> class rasterizer_outline
    {
    private:
        Renderer* m_ren;     // 指向渲染器的指针，用于渲染图形
        int       m_start_x; // 起始 x 坐标
        int       m_start_y; // 起始 y 坐标
        unsigned  m_vertices; // 顶点数量统计
    };

}

#endif
```