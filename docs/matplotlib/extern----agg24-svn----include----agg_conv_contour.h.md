# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_contour.h`

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
// conv_stroke
//
//----------------------------------------------------------------------------
#ifndef AGG_CONV_CONTOUR_INCLUDED
#define AGG_CONV_CONTOUR_INCLUDED

#include "agg_basics.h"                    // 包含基本的 AGG 库头文件
#include "agg_vcgen_contour.h"             // 包含 AGG 轮廓生成器头文件
#include "agg_conv_adaptor_vcgen.h"        // 包含 AGG 转换器适配器头文件

namespace agg
{

    //-----------------------------------------------------------conv_contour
    template<class VertexSource>            // 使用模板定义 conv_contour 结构体，继承自 conv_adaptor_vcgen
    struct conv_contour : public conv_adaptor_vcgen<VertexSource, vcgen_contour>
    {
        typedef conv_adaptor_vcgen<VertexSource, vcgen_contour> base_type;  // 定义基类类型为 conv_adaptor_vcgen

        conv_contour(VertexSource& vs) :    // 构造函数，接受 VertexSource 类型的引用参数 vs
            conv_adaptor_vcgen<VertexSource, vcgen_contour>(vs)  // 调用基类的构造函数初始化
        {
        }

        void line_join(line_join_e lj) { base_type::generator().line_join(lj); }  // 设置线段连接方式
        void inner_join(inner_join_e ij) { base_type::generator().inner_join(ij); }  // 设置内部连接方式
        void width(double w) { base_type::generator().width(w); }  // 设置线段宽度
        void miter_limit(double ml) { base_type::generator().miter_limit(ml); }  // 设置斜角限制
        void miter_limit_theta(double t) { base_type::generator().miter_limit_theta(t); }  // 设置斜角限制角度
        void inner_miter_limit(double ml) { base_type::generator().inner_miter_limit(ml); }  // 设置内部斜角限制
        void approximation_scale(double as) { base_type::generator().approximation_scale(as); }  // 设置近似比例尺度
        void auto_detect_orientation(bool v) { base_type::generator().auto_detect_orientation(v); }  // 自动检测方向

        line_join_e line_join() const { return base_type::generator().line_join(); }  // 获取当前线段连接方式
        inner_join_e inner_join() const { return base_type::generator().inner_join(); }  // 获取当前内部连接方式
        double width() const { return base_type::generator().width(); }  // 获取当前线段宽度
        double miter_limit() const { return base_type::generator().miter_limit(); }  // 获取当前斜角限制
        double inner_miter_limit() const { return base_type::generator().inner_miter_limit(); }  // 获取当前内部斜角限制
        double approximation_scale() const { return base_type::generator().approximation_scale(); }  // 获取当前近似比例尺度
        bool auto_detect_orientation() const { return base_type::generator().auto_detect_orientation(); }  // 获取当前自动检测方向

    private:
        conv_contour(const conv_contour<VertexSource>&);  // 私有拷贝构造函数，禁止复制对象
        const conv_contour<VertexSource>& 
            operator = (const conv_contour<VertexSource>&);  // 私有赋值运算符重载，禁止赋值操作
    };

}

#endif
```