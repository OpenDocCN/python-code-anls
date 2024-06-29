# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_solid.h`

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
// span_solid_rgba8
//
//----------------------------------------------------------------------------

#ifndef AGG_SPAN_SOLID_INCLUDED
#define AGG_SPAN_SOLID_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //--------------------------------------------------------------span_solid
    // span_solid 类模板，用于生成指定颜色的水平线段
    template<class ColorT> class span_solid
    {
    public:
        typedef ColorT color_type; // 定义 ColorT 为颜色类型

        //--------------------------------------------------------------------
        // 设置颜色
        void color(const color_type& c) { m_color = c; }
        
        // 获取颜色
        const color_type& color() const { return m_color; }

        //--------------------------------------------------------------------
        // 准备生成线段，此处无需实际操作，因此为空函数
        void prepare() {}

        //--------------------------------------------------------------------
        // 生成指定长度的线段，每个像素点填充为当前设置的颜色
        void generate(color_type* span, int x, int y, unsigned len)
        {   
            do { *span++ = m_color; } while(--len);
        }

    private:
        color_type m_color; // 私有成员变量，存储当前颜色
    };


}

#endif
```