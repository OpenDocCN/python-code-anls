# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_font_cache_manager.h`

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

#ifndef AGG_FONT_CACHE_MANAGER_INCLUDED
#define AGG_FONT_CACHE_MANAGER_INCLUDED

#include <string.h>         // Include the C string manipulation library for using functions like memcpy
#include "agg_array.h"      // Include the array header from the AGG library

namespace agg
{

    //---------------------------------------------------------glyph_data_type
    enum glyph_data_type
    {
        glyph_data_invalid = 0,     // Invalid glyph data type
        glyph_data_mono    = 1,     // Monochrome glyph data type
        glyph_data_gray8   = 2,     // 8-bit grayscale glyph data type
        glyph_data_outline = 3      // Outline glyph data type
    };


    //-------------------------------------------------------------glyph_cache
    struct glyph_cache
    {
        unsigned        glyph_index;    // Index of the glyph
        int8u*          data;           // Pointer to the glyph data
        unsigned        data_size;      // Size of the glyph data in bytes
        glyph_data_type data_type;      // Type of the glyph data (mono, gray8, outline)
        rect_i          bounds;         // Bounding rectangle of the glyph
        double          advance_x;      // Horizontal advance of the glyph
        double          advance_y;      // Vertical advance of the glyph
    };


    //--------------------------------------------------------------font_cache
    class font_cache
    {
    // 公共部分开始
    public:
        // 定义块大小的枚举常量 block_size，并初始化为 16384-16
        enum block_size_e { block_size = 16384-16 };

        //--------------------------------------------------------------------
        // 构造函数，初始化字体缓存对象
        font_cache() : 
            // 使用 block_allocator 初始化 m_allocator
            m_allocator(block_size),
            // 初始化 m_font_signature 为 0
            m_font_signature(0)
        {}

        //--------------------------------------------------------------------
        // 设置字体签名的函数
        void signature(const char* font_signature)
        {
            // 分配内存给 m_font_signature，长度为 font_signature 的长度加 1
            m_font_signature = (char*)m_allocator.allocate(strlen(font_signature) + 1);
            // 将 font_signature 复制到 m_font_signature
            strcpy(m_font_signature, font_signature);
            // 初始化 m_glyphs 数组为 0
            memset(m_glyphs, 0, sizeof(m_glyphs));
        }

        //--------------------------------------------------------------------
        // 检查字体签名是否匹配的函数
        bool font_is(const char* font_signature) const
        {
            // 比较输入的 font_signature 和 m_font_signature 是否相等
            return strcmp(font_signature, m_font_signature) == 0;
        }

        //--------------------------------------------------------------------
        // 根据字形代码查找字形缓存的函数
        const glyph_cache* find_glyph(unsigned glyph_code) const
        {
            // 计算字形代码的最高字节
            unsigned msb = (glyph_code >> 8) & 0xFF;
            // 如果 m_glyphs[msb] 存在，则返回相应的字形缓存；否则返回 0
            if(m_glyphs[msb]) 
            {
                return m_glyphs[msb][glyph_code & 0xFF];
            }
            return 0;
        }

        //--------------------------------------------------------------------
        // 缓存字形数据的函数
        glyph_cache* cache_glyph(unsigned        glyph_code, 
                                 unsigned        glyph_index,
                                 unsigned        data_size,
                                 glyph_data_type data_type,
                                 const rect_i&   bounds,
                                 double          advance_x,
                                 double          advance_y)
        {
            // 计算字形代码的最高字节
            unsigned msb = (glyph_code >> 8) & 0xFF;
            // 如果 m_glyphs[msb] 不存在，则分配内存给它
            if(m_glyphs[msb] == 0)
            {
                m_glyphs[msb] = 
                    (glyph_cache**)m_allocator.allocate(sizeof(glyph_cache*) * 256, 
                                                        sizeof(glyph_cache*));
                // 将 m_glyphs[msb] 初始化为 0
                memset(m_glyphs[msb], 0, sizeof(glyph_cache*) * 256);
            }

            // 计算字形代码的最低字节
            unsigned lsb = glyph_code & 0xFF;
            // 如果 m_glyphs[msb][lsb] 已经存在，则返回 0（不覆盖）
            if(m_glyphs[msb][lsb]) return 0;

            // 分配内存给新的字形缓存对象 glyph_cache
            glyph_cache* glyph = 
                (glyph_cache*)m_allocator.allocate(sizeof(glyph_cache),
                                                   sizeof(double));

            // 设置字形缓存对象的属性
            glyph->glyph_index        = glyph_index;
            glyph->data               = m_allocator.allocate(data_size);
            glyph->data_size          = data_size;
            glyph->data_type          = data_type;
            glyph->bounds             = bounds;
            glyph->advance_x          = advance_x;
            glyph->advance_y          = advance_y;
            // 将新的字形缓存对象添加到 m_glyphs 中，并返回它
            return m_glyphs[msb][lsb] = glyph;
        }

    private:
        // 块分配器对象 m_allocator
        block_allocator m_allocator;
        // 字形缓存指针数组 m_glyphs，256 个元素
        glyph_cache**   m_glyphs[256];
        // 字体签名字符串指针 m_font_signature
        char*           m_font_signature;
    };
    //---------------------------------------------------------font_cache_pool
    class font_cache_pool
    {
    private:
        font_cache** m_fonts;     // 指向字体缓存数组的指针
        unsigned     m_max_fonts; // 最大字体数量
        unsigned     m_num_fonts; // 当前字体数量
        font_cache*  m_cur_font;  // 当前选定的字体对象指针
    };
    
    
    
    //------------------------------------------------------------------------
    enum glyph_rendering
    {
        glyph_ren_native_mono,   // 原生单色渲染方式
        glyph_ren_native_gray8,  // 原生灰度8位渲染方式
        glyph_ren_outline,       // 轮廓渲染方式
        glyph_ren_agg_mono,      // AGG库单色渲染方式
        glyph_ren_agg_gray8      // AGG库灰度8位渲染方式
    };
    
    
    
    //------------------------------------------------------font_cache_manager
    template<class FontEngine> class font_cache_manager
    {
    private:
        //--------------------------------------------------------------------
        font_cache_manager(const self_type&);  // 复制构造函数（私有化，禁止复制）
        const self_type& operator = (const self_type&);  // 赋值运算符重载（私有化，禁止赋值）
    
        //--------------------------------------------------------------------
        void synchronize()
        {
            // 如果引擎的变动标志与当前对象的不一致
            if(m_change_stamp != m_engine.change_stamp())
            {
                // 更新字体缓存为引擎当前的字体签名
                m_fonts.font(m_engine.font_signature());
                // 更新变动标志为引擎当前的变动标志
                m_change_stamp = m_engine.change_stamp();
                // 重置上一个和最后一个字形缓存指针
                m_prev_glyph = m_last_glyph = 0;
            }
        }
    
        font_cache_pool     m_fonts;            // 字体缓存池对象
        font_engine_type&   m_engine;           // 字体引擎对象的引用
        int                 m_change_stamp;     // 变动标志
        double              m_dx;               // X方向位移
        double              m_dy;               // Y方向位移
        const glyph_cache*  m_prev_glyph;       // 前一个字形缓存指针
        const glyph_cache*  m_last_glyph;       // 最后一个字形缓存指针
        path_adaptor_type   m_path_adaptor;     // 路径适配器对象
        gray8_adaptor_type  m_gray8_adaptor;    // 灰度8位适配器对象
        gray8_scanline_type m_gray8_scanline;   // 灰度8位扫描线对象
        mono_adaptor_type   m_mono_adaptor;     // 单色适配器对象
        mono_scanline_type  m_mono_scanline;    // 单色扫描线对象
    };
}


注释：


// 这是一个预处理器指令，表示结束一个条件编译块，对应于上面的 #ifdef 或 #if 预处理器指令。
// 在条件编译中，当满足指定条件时，编译器会编译位于 #ifdef 和 #endif 之间的代码块。
// 如果条件不满足，那么这段代码块将被忽略，直到遇到 #endif 结束。



#endif


注释：


// 这是一个预处理器指令，用于结束一个条件编译块的另一种形式。
// 在条件编译中，通常与 #ifdef、#ifndef 或 #if 配合使用，用于标记条件编译块的结束。
// 如果在代码中找不到与 #ifdef、#ifndef 或 #if 相对应的条件，那么 #endif 将会导致编译错误。
```