# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_renderer_mclip.h`

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
// class renderer_mclip
//
//----------------------------------------------------------------------------

#ifndef AGG_RENDERER_MCLIP_INCLUDED
#define AGG_RENDERER_MCLIP_INCLUDED

// 包含基本的渲染器和数组支持
#include "agg_basics.h"
#include "agg_array.h"
#include "agg_renderer_base.h"

namespace agg
{

    //----------------------------------------------------------renderer_mclip
    // 模板类定义，用于剪辑渲染器操作
    template<class PixelFormat> class renderer_mclip
    {
    private:
        // 禁止拷贝构造和赋值操作
        renderer_mclip(const renderer_mclip<PixelFormat>&);
        const renderer_mclip<PixelFormat>& 
            operator = (const renderer_mclip<PixelFormat>&);

        // 基本渲染器类型和剪辑区域数组
        base_ren_type          m_ren;
        pod_bvector<rect_i, 4> m_clip;
        // 当前剪辑区域的索引和整体边界框
        unsigned               m_curr_cb;
        rect_i                 m_bounds;
    };

}

#endif
```