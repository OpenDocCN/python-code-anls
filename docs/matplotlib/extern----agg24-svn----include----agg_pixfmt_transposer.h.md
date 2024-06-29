# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_pixfmt_transposer.h`

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

#ifndef AGG_PIXFMT_TRANSPOSER_INCLUDED
#define AGG_PIXFMT_TRANSPOSER_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //=======================================================pixfmt_transposer
    // 模板类 pixfmt_transposer，用于对像素格式进行转置
    template<class PixFmt> class pixfmt_transposer
    {
    private:
        // 指向像素格式对象的指针
        pixfmt_type* m_pixf;
    };
}

#endif
```