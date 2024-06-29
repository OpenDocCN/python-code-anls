# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_interpolator_persp.h`

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

#ifndef AGG_SPAN_INTERPOLATOR_PERSP_INCLUDED
#define AGG_SPAN_INTERPOLATOR_PERSP_INCLUDED

// 包含透视变换类和 DDA 线性插值器类的头文件
#include "agg_trans_perspective.h"
#include "agg_dda_line.h"

namespace agg
{

    //===========================================span_interpolator_persp_exact
    // 透视精确插值器类模板定义，使用指定的子像素位移精度
    template<unsigned SubpixelShift = 8> 
    class span_interpolator_persp_exact
    {
    private:
        // 透视变换类型对象，用于直接变换和反变换
        trans_type             m_trans_dir;
        trans_type             m_trans_inv;
        // 迭代器类型对象
        iterator_type          m_iterator;
        // DDA 线性插值器对象，用于 X 和 Y 方向的缩放
        dda2_line_interpolator m_scale_x;
        dda2_line_interpolator m_scale_y;
    };

    //============================================span_interpolator_persp_lerp
    // 透视线性插值器类模板定义，使用指定的子像素位移精度
    template<unsigned SubpixelShift = 8> 
    class span_interpolator_persp_lerp
    {
    private:
        // 透视变换类型对象，用于直接变换和反变换
        trans_type             m_trans_dir;
        trans_type             m_trans_inv;
        // DDA 线性插值器对象，用于 X 和 Y 方向的坐标和缩放
        dda2_line_interpolator m_coord_x;
        dda2_line_interpolator m_coord_y;
        dda2_line_interpolator m_scale_x;
        dda2_line_interpolator m_scale_y;
    };

}

#endif
```