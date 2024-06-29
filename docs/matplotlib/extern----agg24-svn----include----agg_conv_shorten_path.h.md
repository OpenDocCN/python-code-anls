# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_shorten_path.h`

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

#ifndef AGG_CONV_SHORTEN_PATH_INCLUDED
#define AGG_CONV_SHORTEN_PATH_INCLUDED

#include "agg_basics.h"
#include "agg_conv_adaptor_vcgen.h"
#include "agg_vcgen_vertex_sequence.h"

namespace agg
{

    //=======================================================conv_shorten_path
    // conv_shorten_path 模板类，继承自 conv_adaptor_vcgen 模板类
    template<class VertexSource>  class conv_shorten_path : 
    public conv_adaptor_vcgen<VertexSource, vcgen_vertex_sequence>
    {
    public:
        // 基类类型定义
        typedef conv_adaptor_vcgen<VertexSource, vcgen_vertex_sequence> base_type;

        // 构造函数，初始化基类
        conv_shorten_path(VertexSource& vs) : 
            conv_adaptor_vcgen<VertexSource, vcgen_vertex_sequence>(vs)
        {
        }

        // 缩短路径的方法，调用基类的生成器(generator)的对应方法
        void shorten(double s) { base_type::generator().shorten(s); }
        // 获取当前路径缩短的长度，调用基类的生成器(generator)的对应方法
        double shorten() const { return base_type::generator().shorten(); }

    private:
        // 拷贝构造函数和赋值运算符都被声明为私有，禁止使用
        conv_shorten_path(const conv_shorten_path<VertexSource>&);
        const conv_shorten_path<VertexSource>& 
            operator = (const conv_shorten_path<VertexSource>&);
    };


}

#endif
```