# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_renderer_base.h`

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
// class renderer_base
//
//----------------------------------------------------------------------------

#ifndef AGG_RENDERER_BASE_INCLUDED
#define AGG_RENDERER_BASE_INCLUDED

#include "agg_basics.h"
#include "agg_rendering_buffer.h"

namespace agg
{

    //-----------------------------------------------------------renderer_base
    // 模板类 renderer_base，用于渲染器基类
    template<class PixelFormat> class renderer_base
    {
    private:
        // 指向像素格式的指针，用于渲染操作
        pixfmt_type* m_ren;
        // 定义一个矩形整数区域，表示渲染的裁剪框
        rect_i       m_clip_box;
    };

}

#endif
```