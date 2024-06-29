# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_pattern_rgb.h`

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
// Adaptation for high precision colors has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------


#ifndef AGG_SPAN_PATTERN_RGB_INCLUDED
#define AGG_SPAN_PATTERN_RGB_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //========================================================span_pattern_rgb
    // 模板类定义，用于处理 RGB 模式的图案填充
    template<class Source> class span_pattern_rgb
    {
    // 定义公共部分：指定源类型、颜色类型、排序类型等
    public:
        typedef Source source_type;                    // 源类型定义
        typedef typename source_type::color_type color_type;  // 颜色类型定义
        typedef typename source_type::order_type order_type;  // 排序类型定义
        typedef typename color_type::value_type value_type;   // 值类型定义
        typedef typename color_type::calc_type calc_type;     // 计算类型定义

        //--------------------------------------------------------------------
        // 默认构造函数，未做任何操作
        span_pattern_rgb() {}

        // 构造函数，初始化源、偏移量和透明度
        span_pattern_rgb(source_type& src,
                         unsigned offset_x, unsigned offset_y) :
            m_src(&src),
            m_offset_x(offset_x),
            m_offset_y(offset_y),
            m_alpha(color_type::base_mask)
        {}

        //--------------------------------------------------------------------
        // 将新的源对象附加到当前模式
        void   attach(source_type& v)      { m_src = &v; }
        
        // 返回当前源对象的引用
        source_type& source()       { return *m_src; }
        
        // 返回当前源对象的常量引用
        const  source_type& source() const { return *m_src; }

        //--------------------------------------------------------------------
        // 设置/获取 X 轴偏移量
        void       offset_x(unsigned v) { m_offset_x = v; }
        unsigned   offset_x() const { return m_offset_x; }

        // 设置/获取 Y 轴偏移量
        void       offset_y(unsigned v) { m_offset_y = v; }
        unsigned   offset_y() const { return m_offset_y; }

        // 设置/获取透明度
        void       alpha(value_type v) { m_alpha = v; }
        value_type alpha() const { return m_alpha; }

        //--------------------------------------------------------------------
        // 准备函数，目前为空实现
        void prepare() {}

        // 生成函数，从源中生成颜色 span，加上偏移量和透明度
        void generate(color_type* span, int x, int y, unsigned len)
        {   
            x += m_offset_x;                    // 根据 X 偏移量调整 x 坐标
            y += m_offset_y;                    // 根据 Y 偏移量调整 y 坐标
            const value_type* p = (const value_type*)m_src->span(x, y, len);  // 获取源中指定区域的颜色数据指针
            do
            {
                span->r = p[order_type::R];     // 设置 span 的红色分量
                span->g = p[order_type::G];     // 设置 span 的绿色分量
                span->b = p[order_type::B];     // 设置 span 的蓝色分量
                span->a = m_alpha;              // 设置 span 的透明度分量
                p = m_src->next_x();            // 获取下一个 X 坐标的颜色数据指针
                ++span;                         // 移动到下一个颜色 span
            }
            while(--len);                       // 重复直到处理完指定长度的 span
        }

    private:
        source_type* m_src;         // 源对象指针
        unsigned     m_offset_x;    // X 轴偏移量
        unsigned     m_offset_y;    // Y 轴偏移量
        value_type   m_alpha;       // 透明度值
    };
}


注释：


// 结束一个 C/C++ 的预处理条件指令块，配合 #ifdef 或 #ifndef 使用



#endif


注释：


// 结束一个 C/C++ 的条件编译指令块，配合 #ifdef 或 #ifndef 使用，表示条件编译的结束
```