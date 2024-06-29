# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_marker_adaptor.h`

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

#ifndef AGG_CONV_MARKER_ADAPTOR_INCLUDED
#define AGG_CONV_MARKER_ADAPTOR_INCLUDED

#include "agg_basics.h"  // 引入基础头文件
#include "agg_conv_adaptor_vcgen.h"  // 引入转换适配器和顶点生成器头文件
#include "agg_vcgen_vertex_sequence.h"  // 引入顶点序列生成器头文件

namespace agg
{

    //=====================================================conv_marker_adaptor
    // 基于顶点源的标记适配器
    template<class VertexSource, class Markers=null_markers>
    struct conv_marker_adaptor : 
    public conv_adaptor_vcgen<VertexSource, vcgen_vertex_sequence, Markers>
    {
        typedef Markers marker_type;  // 标记类型定义
        typedef conv_adaptor_vcgen<VertexSource, vcgen_vertex_sequence, Markers> base_type;  // 基础类型定义

        // 构造函数，初始化基类的适配器
        conv_marker_adaptor(VertexSource& vs) : 
            conv_adaptor_vcgen<VertexSource, vcgen_vertex_sequence, Markers>(vs)
        {
        }

        // 缩短路径函数，调用基类的生成器缩短路径方法
        void shorten(double s) { base_type::generator().shorten(s); }
        
        // 获取当前缩短路径值，调用基类的生成器获取缩短路径方法
        double shorten() const { return base_type::generator().shorten(); }

    private:
        // 禁用拷贝构造函数和赋值操作符
        conv_marker_adaptor(const conv_marker_adaptor<VertexSource, Markers>&);
        const conv_marker_adaptor<VertexSource, Markers>& 
            operator = (const conv_marker_adaptor<VertexSource, Markers>&);
    };


}

#endif
```