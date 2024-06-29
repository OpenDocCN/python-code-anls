# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_glyph_raster_bin.h`

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

#ifndef AGG_GLYPH_RASTER_BIN_INCLUDED
#define AGG_GLYPH_RASTER_BIN_INCLUDED

#include <string.h>
#include "agg_basics.h"

namespace agg
{

    //========================================================glyph_raster_bin
    template<class ColorT> class glyph_raster_bin
    {
    private:
        //--------------------------------------------------------------------
        // 函数：value
        // 描述：从指定地址解析两个字节，根据系统字节序返回16位无符号整数
        int16u value(const int8u* p) const
        {
            int16u v;
            if(m_big_endian)
            {
                 *(int8u*)&v      = p[1];      // 大端序：高字节在前，低字节在后
                *((int8u*)&v + 1) = p[0];
            }
            else
            {
                 *(int8u*)&v      = p[0];      // 小端序：低字节在前，高字节在后
                *((int8u*)&v + 1) = p[1];
            }
            return v;
        }


        //--------------------------------------------------------------------
        const int8u* m_font;                // 指向字形数据的指针
        bool m_big_endian;                  // 标识当前系统的字节序是否为大端
        cover_type m_span[32];              // 用于存储渲染结果的覆盖数组
        const int8u* m_bits;                // 指向字形位图数据的指针
        unsigned m_glyph_width;             // 字形的宽度（像素）
        unsigned m_glyph_byte_width;        // 字形的字节宽度（字节）
    };


}

#endif
```