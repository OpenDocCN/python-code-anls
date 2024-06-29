# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_pattern_gray.h`

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

// 如果 AGG_SPAN_PATTERN_GRAY_INCLUDED 未定义，则定义 AGG_SPAN_PATTERN_GRAY_INCLUDED
#ifndef AGG_SPAN_PATTERN_GRAY_INCLUDED
#define AGG_SPAN_PATTERN_GRAY_INCLUDED

// 包含 AGG 库的基础定义
#include "agg_basics.h"

// 使用 agg 命名空间
namespace agg
{

    //=======================================================span_pattern_gray
    // 定义一个模板类 span_pattern_gray，模板参数为 Source
    template<class Source> class span_pattern_gray
    {
    # 定义一个模板类 span_pattern_gray，公开以下类型定义：source_type、color_type、value_type 和 calc_type
    public:
        typedef Source source_type;               # 定义类型 source_type 为模板参数 Source
        typedef typename source_type::color_type color_type;    # 定义类型 color_type 为 source_type 中的 color_type
        typedef typename color_type::value_type value_type;     # 定义类型 value_type 为 color_type 中的 value_type
        typedef typename color_type::calc_type calc_type;       # 定义类型 calc_type 为 color_type 中的 calc_type
    
        //--------------------------------------------------------------------
        # 默认构造函数 span_pattern_gray，无参数
        span_pattern_gray() {}
    
        # 带参数的构造函数 span_pattern_gray，接受一个 source_type 的引用 src，以及两个无符号整数 offset_x 和 offset_y
        span_pattern_gray(source_type& src, 
                          unsigned offset_x, unsigned offset_y) :
            m_src(&src),                   # 初始化 m_src 指针为 src 的地址
            m_offset_x(offset_x),          # 初始化 m_offset_x 为 offset_x
            m_offset_y(offset_y),          # 初始化 m_offset_y 为 offset_y
            m_alpha(color_type::base_mask) # 初始化 m_alpha 为 color_type::base_mask
        {}
    
        //--------------------------------------------------------------------
        # 成员函数 attach，接受一个 source_type 的引用 v，用于将 m_src 指针指向 v
        void   attach(source_type& v)      { m_src = &v; }
               source_type& source()       { return *m_src; }       # 返回 m_src 指向的 source_type 引用
        const  source_type& source() const { return *m_src; }       # 返回 const 的 m_src 指向的 source_type 引用
    
        //--------------------------------------------------------------------
        # 成员函数 offset_x，设置 m_offset_x 的值
        void       offset_x(unsigned v) { m_offset_x = v; }
        # 成员函数 offset_y，设置 m_offset_y 的值
        void       offset_y(unsigned v) { m_offset_y = v; }
        # 成员函数 offset_x，返回 m_offset_x 的值
        unsigned   offset_x() const { return m_offset_x; }
        # 成员函数 offset_y，返回 m_offset_y 的值
        unsigned   offset_y() const { return m_offset_y; }
        # 成员函数 alpha，设置 m_alpha 的值
        void       alpha(value_type v) { m_alpha = v; }
        # 成员函数 alpha，返回 m_alpha 的值
        value_type alpha() const { return m_alpha; }
    
        //--------------------------------------------------------------------
        # 成员函数 prepare，空实现，无操作
        void prepare() {}
    
        # 成员函数 generate，生成 span，接受 color_type 指针 span、整数 x 和 y，以及无符号整数 len
        void generate(color_type* span, int x, int y, unsigned len)
        {   
            x += m_offset_x;   # 将 x 增加 m_offset_x
            y += m_offset_y;   # 将 y 增加 m_offset_y
            const value_type* p = (const value_type*)m_src->span(x, y, len);   # 获取从 m_src 获取的 span 数据的指针，并将其转换为 value_type*
            do
            {
                span->v = *p;    # 将 span->v 设置为指针 p 所指向的值
                span->a = m_alpha;   # 将 span->a 设置为 m_alpha 的值
                p = m_src->next_x();   # 将 p 设置为 m_src 的下一个 x 值
                ++span;         # 将 span 指针向前移动一个位置
            }
            while(--len);   # 循环 len 次，递减 len 直到为 0
        }
    
    private:
        source_type* m_src;       # 私有成员变量 m_src，指向 source_type 的指针
        unsigned     m_offset_x;  # 私有成员变量 m_offset_x，偏移量 x
        unsigned     m_offset_y;  # 私有成员变量 m_offset_y，偏移量 y
        value_type   m_alpha;     # 私有成员变量 m_alpha，alpha 值
    };
}


注释：


// 结束一个条件编译块或头文件的尾部



#endif


注释：


// 如果前面有 #ifdef 或 #ifndef，则用来结束条件编译指令
```