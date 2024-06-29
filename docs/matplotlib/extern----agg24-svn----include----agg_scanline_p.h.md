# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_scanline_p.h`

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
// Class scanline_p - a general purpose scanline container with packed spans.
//
//----------------------------------------------------------------------------

#ifndef AGG_SCANLINE_P_INCLUDED
#define AGG_SCANLINE_P_INCLUDED

#include "agg_array.h"

namespace agg
{

    //=============================================================scanline_p8
    // 
    // This is a general purpose scanline container which supports the interface 
    // used in the rasterizer::render(). See description of scanline_u8
    // for details.
    // 
    //------------------------------------------------------------------------
    class scanline_p8
    {
    private:
        // 禁止拷贝构造函数和赋值操作符的使用
        scanline_p8(const self_type&);
        const self_type& operator = (const self_type&);

        // 上一个 x 坐标位置和当前 y 坐标位置
        int                   m_last_x;
        int                   m_y;

        // 保存覆盖值的数组和当前覆盖指针
        pod_array<cover_type> m_covers;
        cover_type*           m_cover_ptr;

        // 保存跨度对象的数组和当前跨度指针
        pod_array<span>       m_spans;
        span*                 m_cur_span;
    };








    //==========================================================scanline32_p8
    class scanline32_p8
    {
    private:
        // 禁止拷贝构造函数和赋值操作符的使用
        scanline32_p8(const self_type&);
        const self_type& operator = (const self_type&);

        // 最大长度、上一个 x 坐标位置和当前 y 坐标位置
        unsigned              m_max_len;
        int                   m_last_x;
        int                   m_y;

        // 保存覆盖值的数组和当前覆盖指针
        pod_array<cover_type> m_covers;
        cover_type*           m_cover_ptr;

        // 保存跨度对象的数组
        span_array_type       m_spans;
    };


}


#endif
```