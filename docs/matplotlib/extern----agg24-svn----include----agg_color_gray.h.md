# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_color_gray.h`

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
// Adaptation for high precision colors has been sponsored by
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
//
//----------------------------------------------------------------------------
//
// color types gray8, gray16
//
//----------------------------------------------------------------------------

#ifndef AGG_COLOR_GRAY_INCLUDED
#define AGG_COLOR_GRAY_INCLUDED

#include "agg_basics.h"
#include "agg_color_rgba.h"

namespace agg
{

    //===================================================================gray8
    // 灰度颜色类型 gray8T 模板
    template<class Colorspace>
    struct gray8T
    {
    };

    // 灰度颜色类型 gray8 的别名，使用线性色彩空间
    typedef gray8T<linear> gray8;
    
    // 灰度颜色类型 sgray8 的别名，使用 sRGB 色彩空间
    typedef gray8T<sRGB> sgray8;


    //==================================================================gray16
    // 灰度颜色类型 gray16 结构
    struct gray16
    {
    };


    //===================================================================gray32
    // 灰度颜色类型 gray32 结构
    struct gray32
    {
    };
}

#endif
```