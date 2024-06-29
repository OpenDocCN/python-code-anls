# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_converter.h`

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

#ifndef AGG_SPAN_CONVERTER_INCLUDED
#define AGG_SPAN_CONVERTER_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //----------------------------------------------------------span_converter
    // span_converter 模板类的定义，用于将一个 SpanGenerator 生成的数据转换为另一种格式
    template<class SpanGenerator, class SpanConverter> class span_converter
    {
    public:
        // 定义 color_type 类型为 SpanGenerator 类中的 color_type 类型
        typedef typename SpanGenerator::color_type color_type;

        // 构造函数，初始化 SpanGenerator 和 SpanConverter 对象的指针
        span_converter(SpanGenerator& span_gen, SpanConverter& span_cnv) : 
            m_span_gen(&span_gen), m_span_cnv(&span_cnv) {}

        // 重新绑定 SpanGenerator 对象的方法
        void attach_generator(SpanGenerator& span_gen) { m_span_gen = &span_gen; }
        // 重新绑定 SpanConverter 对象的方法
        void attach_converter(SpanConverter& span_cnv) { m_span_cnv = &span_cnv; }

        //--------------------------------------------------------------------
        // 准备生成转换后数据的方法，调用 SpanGenerator 和 SpanConverter 对象的准备方法
        void prepare() 
        { 
            m_span_gen->prepare(); 
            m_span_cnv->prepare();
        }

        //--------------------------------------------------------------------
        // 生成经过转换的颜色数据的方法
        void generate(color_type* span, int x, int y, unsigned len)
        {
            // 调用 SpanGenerator 对象的 generate 方法生成原始数据
            m_span_gen->generate(span, x, y, len);
            // 调用 SpanConverter 对象的 generate 方法进行数据转换
            m_span_cnv->generate(span, x, y, len);
        }

    private:
        // SpanGenerator 对象的指针
        SpanGenerator* m_span_gen;
        // SpanConverter 对象的指针
        SpanConverter* m_span_cnv;
    };

}

#endif
```