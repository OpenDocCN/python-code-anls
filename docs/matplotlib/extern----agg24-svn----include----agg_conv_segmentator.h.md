# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_segmentator.h`

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

#ifndef AGG_CONV_SEGMENTATOR_INCLUDED
#define AGG_CONV_SEGMENTATOR_INCLUDED

#include "agg_basics.h"
#include "agg_conv_adaptor_vpgen.h"
#include "agg_vpgen_segmentator.h"

namespace agg
{

    //========================================================conv_segmentator
    // conv_segmentator 模板结构体继承自 conv_adaptor_vpgen 类和 vpgen_segmentator 类
    template<class VertexSource> 
    struct conv_segmentator : public conv_adaptor_vpgen<VertexSource, vpgen_segmentator>
    {
        typedef conv_adaptor_vpgen<VertexSource, vpgen_segmentator> base_type;

        // 构造函数，初始化基类 conv_adaptor_vpgen，传入参数 VertexSource 的引用
        conv_segmentator(VertexSource& vs) : 
            conv_adaptor_vpgen<VertexSource, vpgen_segmentator>(vs) {}

        // 设置近似比例的方法，调用基类的 vpgen().approximation_scale(s) 方法
        void approximation_scale(double s) { base_type::vpgen().approximation_scale(s);        }
        
        // 获取当前近似比例的方法，调用基类的 vpgen().approximation_scale() 方法
        double approximation_scale() const { return base_type::vpgen().approximation_scale();  }

    private:
        // 拷贝构造函数，被声明为 private，禁止对象的拷贝构造
        conv_segmentator(const conv_segmentator<VertexSource>&);
        
        // 赋值运算符重载，被声明为 private，禁止对象的赋值操作
        const conv_segmentator<VertexSource>& 
            operator = (const conv_segmentator<VertexSource>&);
    };


}

#endif
//----------------------------------------------------------------------------
```