# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_rendering_buffer.h`

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
// class rendering_buffer
//
//----------------------------------------------------------------------------

#ifndef AGG_RENDERING_BUFFER_INCLUDED
#define AGG_RENDERING_BUFFER_INCLUDED

#include "agg_array.h"

namespace agg
{

    //===========================================================row_accessor
    // 模板类，提供访问渲染缓冲区中行数据的功能
    template<class T> class row_accessor
    {
    private:
        //--------------------------------------------------------------------
        T*            m_buf;    // 指向渲染缓冲区的指针
        T*            m_start;  // 指向第一个像素的指针，取决于跨度
        unsigned      m_width;  // 宽度，以像素为单位
        unsigned      m_height; // 高度，以像素为单位
        int           m_stride; // 每行的字节数，可以是负数
    };




    //==========================================================row_ptr_cache
    // 模板类，提供缓存行指针的功能
    template<class T> class row_ptr_cache
    {
    private:
        //--------------------------------------------------------------------
        T*            m_buf;        // 指向渲染缓冲区的指针
        pod_array<T*> m_rows;       // 缓存渲染缓冲区每行的指针
        unsigned      m_width;      // 宽度，以像素为单位
        unsigned      m_height;     // 高度，以像素为单位
        int           m_stride;     // 每行的字节数，可以是负数
    };




    //========================================================rendering_buffer
    // 
    // 主要用于访问帧缓冲区中行的主类型定义。它提供了在矩形矩阵中从顶部到底部或从底部到顶部导航的功能，具体取决于跨度。
    //
    // row_accessor 创建和销毁廉价，但在调用 row_ptr() 时执行一次乘法运算。
    // 
    // row_ptr_cache 创建一个指向行的指针数组，因此通过 row_ptr() 的访问可能更快。但是在创建时需要内存分配。
    // 例如，在典型的 Intel Pentium 硬件上，row_ptr_cache 可以提高 span_image_filter_rgb_nn 的速度达到 10%。
    //
    // 它仅在像 pixfmt_rgba32 这样的简写 typedef 中使用，并且可以在 agg_config.h 中重新定义。
    // 在实际应用中，您可以根据需要同时使用这两种方法。
    //------------------------------------------------------------------------
    
    
    注释：
    
    
    # 这是一条注释，用于在代码中划分不同部分或者标记重要的区域
#ifdef AGG_RENDERING_BUFFER
    // 如果定义了 AGG_RENDERING_BUFFER 宏，则使用 AGG_RENDERING_BUFFER 作为 rendering_buffer 的类型别名
    typedef AGG_RENDERING_BUFFER rendering_buffer;
#else
    // 如果未定义 AGG_RENDERING_BUFFER 宏，则使用 row_accessor<int8u> 作为 rendering_buffer 的类型别名
//  typedef row_ptr_cache<int8u> rendering_buffer;
    typedef row_accessor<int8u> rendering_buffer;
#endif
```