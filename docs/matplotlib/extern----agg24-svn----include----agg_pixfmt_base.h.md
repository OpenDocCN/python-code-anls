# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_pixfmt_base.h`

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

// 如果 AGG_PIXFMT_BASE_INCLUDED 宏未定义，则定义 AGG_PIXFMT_BASE_INCLUDED 宏
#ifndef AGG_PIXFMT_BASE_INCLUDED
#define AGG_PIXFMT_BASE_INCLUDED

// 包含基础头文件和颜色处理头文件
#include "agg_basics.h"
#include "agg_color_gray.h"
#include "agg_color_rgba.h"

// 命名空间 agg 开始
namespace agg
{
    // 定义像素格式标签结构体 pixfmt_gray_tag
    struct pixfmt_gray_tag
    {
    };

    // 定义像素格式标签结构体 pixfmt_rgb_tag
    struct pixfmt_rgb_tag
    {
    };

    // 定义像素格式标签结构体 pixfmt_rgba_tag
    struct pixfmt_rgba_tag
    {
    };

    // 定义颜色混合基础模板类 blender_base
    //--------------------------------------------------------------blender_base
    template<class ColorT, class Order = void> 
    struct blender_base
    {
        // 定义颜色类型、顺序类型及值类型
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;

        // 获取带有覆盖率的颜色值
        static rgba get(value_type r, value_type g, value_type b, value_type a, cover_type cover = cover_full)
        {
            // 如果覆盖率大于无覆盖，则计算新的颜色值
            if (cover > cover_none)
            {
                rgba c(
                    color_type::to_double(r), 
                    color_type::to_double(g), 
                    color_type::to_double(b), 
                    color_type::to_double(a));

                // 如果覆盖率小于全覆盖，则按比例缩放颜色值
                if (cover < cover_full)
                {
                    double x = double(cover) / cover_full;
                    c.r *= x;
                    c.g *= x;
                    c.b *= x;
                    c.a *= x;
                }

                return c;
            }
            else 
            {
                // 否则返回无颜色状态
                return rgba::no_color();
            }
        }

        // 通过指针获取带有覆盖率的颜色值
        static rgba get(const value_type* p, cover_type cover = cover_full)
        {
            return get(
                p[order_type::R], 
                p[order_type::G], 
                p[order_type::B], 
                p[order_type::A], 
                cover);
        }

        // 设置像素值的颜色分量
        static void set(value_type* p, value_type r, value_type g, value_type b, value_type a)
        {
            p[order_type::R] = r;
            p[order_type::G] = g;
            p[order_type::B] = b;
            p[order_type::A] = a;
        }

        // 通过 rgba 结构设置像素值的颜色分量
        static void set(value_type* p, const rgba& c)
        {
            p[order_type::R] = color_type::from_double(c.r);
            p[order_type::G] = color_type::from_double(c.g);
            p[order_type::B] = color_type::from_double(c.b);
            p[order_type::A] = color_type::from_double(c.a);
        }
    };
}

// 结束命名空间 agg
#endif // AGG_PIXFMT_BASE_INCLUDED
```