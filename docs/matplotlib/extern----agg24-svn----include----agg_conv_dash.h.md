# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_dash.h`

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

// 防止重复包含头文件
#ifndef AGG_CONV_DASH_INCLUDED
#define AGG_CONV_DASH_INCLUDED

// 包含基础头文件及相关类声明
#include "agg_basics.h"
#include "agg_vcgen_dash.h"
#include "agg_conv_adaptor_vcgen.h"

namespace agg
{

    //---------------------------------------------------------------conv_dash
    // conv_dash 类的模板定义，继承自 conv_adaptor_vcgen，用于生成虚线效果
    template<class VertexSource, class Markers=null_markers> 
    struct conv_dash : public conv_adaptor_vcgen<VertexSource, vcgen_dash, Markers>
    {
        // 定义类型别名
        typedef Markers marker_type;
        typedef conv_adaptor_vcgen<VertexSource, vcgen_dash, Markers> base_type;

        // 构造函数，初始化基类
        conv_dash(VertexSource& vs) : 
            conv_adaptor_vcgen<VertexSource, vcgen_dash, Markers>(vs)
        {
        }

        // 移除所有虚线段
        void remove_all_dashes() 
        { 
            base_type::generator().remove_all_dashes(); 
        }

        // 添加虚线段
        void add_dash(double dash_len, double gap_len) 
        { 
            base_type::generator().add_dash(dash_len, gap_len); 
        }

        // 设置虚线起始偏移
        void dash_start(double ds) 
        { 
            base_type::generator().dash_start(ds); 
        }

        // 缩短虚线长度
        void shorten(double s) { base_type::generator().shorten(s); }

        // 获取当前虚线长度
        double shorten() const { return base_type::generator().shorten(); }

    private:
        // 禁止拷贝构造函数和赋值操作符
        conv_dash(const conv_dash<VertexSource, Markers>&);
        const conv_dash<VertexSource, Markers>& 
            operator = (const conv_dash<VertexSource, Markers>&);
    };

}

// 结束 ifndef AGG_CONV_DASH_INCLUDED
#endif
```