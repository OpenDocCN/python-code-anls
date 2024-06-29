# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_subdiv_adaptor.h`

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

// 如果未定义 AGG_SPAN_SUBDIV_ADAPTOR_INCLUDED，定义它，以避免重复包含
#ifndef AGG_SPAN_SUBDIV_ADAPTOR_INCLUDED
#define AGG_SPAN_SUBDIV_ADAPTOR_INCLUDED

// 包含基本的 AGG 库文件，确保这些基本功能可用
#include "agg_basics.h"

// 命名空间 agg 的开始
namespace agg
{

    //=================================================span_subdiv_adaptor
    // span_subdiv_adaptor 类模板的定义
    template<class Interpolator, unsigned SubpixelShift = 8> 
    class span_subdiv_adaptor
    {
    private:
        // 私有成员变量
        unsigned m_subdiv_shift;        // 子像素移位量
        unsigned m_subdiv_size;         // 子像素大小
        unsigned m_subdiv_mask;         // 子像素掩码
        interpolator_type* m_interpolator;  // 插值器类型指针
        int      m_src_x;               // 源 x 坐标
        double   m_src_y;               // 源 y 坐标
        unsigned m_pos;                 // 当前位置
        unsigned m_len;                 // 长度
    };

}

// 结束命名空间 agg
#endif
```