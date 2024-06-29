# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_pattern_filters_rgba.h`

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

#ifndef AGG_PATTERN_FILTERS_RGBA8_INCLUDED
#define AGG_PATTERN_FILTERS_RGBA8_INCLUDED

#include "agg_basics.h"
#include "agg_line_aa_basics.h"
#include "agg_color_rgba.h"

namespace agg
{

    //=======================================================pattern_filter_nn
    // 模板结构体，实现最近邻插值的图案过滤器
    template<class ColorT> struct pattern_filter_nn
    {
        typedef ColorT color_type;
        
        // 返回膨胀值，这里为0
        static unsigned dilation() { return 0; }

        // 低分辨率像素获取函数，从二维数组 buf 中获取像素值，存入 p 中，位置为 (x, y)
        static void AGG_INLINE pixel_low_res(color_type const* const* buf, 
                                             color_type* p, int x, int y)
        {
            *p = buf[y][x];
        }

        // 高分辨率像素获取函数，从二维数组 buf 中获取像素值，存入 p 中，位置为 (x, y)
        // 使用位移运算进行像素位置计算
        static void AGG_INLINE pixel_high_res(color_type const* const* buf, 
                                              color_type* p, int x, int y)
        {
            *p = buf[y >> line_subpixel_shift]
                    [x >> line_subpixel_shift];
        }
    };

    // 定义 rgba8 类型的最近邻插值图案过滤器
    typedef pattern_filter_nn<rgba8>  pattern_filter_nn_rgba8;
    // 定义 rgba16 类型的最近邻插值图案过滤器
    typedef pattern_filter_nn<rgba16> pattern_filter_nn_rgba16;


    //===========================================pattern_filter_bilinear_rgba
    // 模板结构体，实现双线性插值的 rgba 图案过滤器
    template<class ColorT> struct pattern_filter_bilinear_rgba
    {
        // 定义模板类，模板参数为 ColorT 类型
        typedef ColorT color_type;
        // 嵌套类型定义，value_type 是 color_type 的值类型
        typedef typename color_type::value_type value_type;
        // 嵌套类型定义，calc_type 是 color_type 的计算类型
        typedef typename color_type::calc_type calc_type;
    
        // 静态成员函数，返回固定的膨胀值为 1
        static unsigned dilation() { return 1; }
    
        // 内联函数，低分辨率像素访问器，从 buf 数组中读取像素数据并赋值给 p
        static AGG_INLINE void pixel_low_res(color_type const* const* buf, 
                                             color_type* p, int x, int y)
        {
            *p = buf[y][x];
        }
    
        // 内联函数，高分辨率像素访问器，通过双线性插值计算像素值并赋值给 p
        static AGG_INLINE void pixel_high_res(color_type const* const* buf, 
                                              color_type* p, int x, int y)
        {
            // 定义用于双线性插值的变量
            calc_type r, g, b, a;
            r = g = b = a = 0;
    
            calc_type weight;
            // 计算低分辨率像素坐标
            int x_lr = x >> line_subpixel_shift;
            int y_lr = y >> line_subpixel_shift;
    
            // 计算在低分辨率像素格子中的偏移量
            x &= line_subpixel_mask;
            y &= line_subpixel_mask;
            // 指向 buf 中对应像素的指针
            const color_type* ptr = buf[y_lr] + x_lr;
    
            // 计算插值权重和像素分量
            weight = (line_subpixel_scale - x) * 
                     (line_subpixel_scale - y);
            r += weight * ptr->r;
            g += weight * ptr->g;
            b += weight * ptr->b;
            a += weight * ptr->a;
    
            ++ptr;
    
            weight = x * (line_subpixel_scale - y);
            r += weight * ptr->r;
            g += weight * ptr->g;
            b += weight * ptr->b;
            a += weight * ptr->a;
    
            ptr = buf[y_lr + 1] + x_lr;
    
            weight = (line_subpixel_scale - x) * y;
            r += weight * ptr->r;
            g += weight * ptr->g;
            b += weight * ptr->b;
            a += weight * ptr->a;
    
            ++ptr;
    
            weight = x * y;
            r += weight * ptr->r;
            g += weight * ptr->g;
            b += weight * ptr->b;
            a += weight * ptr->a;
    
            // 将计算得到的像素值降采样并赋值给输出像素 p
            p->r = (value_type)color_type::downshift(r, line_subpixel_shift * 2);
            p->g = (value_type)color_type::downshift(g, line_subpixel_shift * 2);
            p->b = (value_type)color_type::downshift(b, line_subpixel_shift * 2);
            p->a = (value_type)color_type::downshift(a, line_subpixel_shift * 2);
        }
    };
    
    // 定义具体像素类型 rgba8 的双线性插值滤波器模板实例
    typedef pattern_filter_bilinear_rgba<rgba8>  pattern_filter_bilinear_rgba8;
    // 定义具体像素类型 rgba16 的双线性插值滤波器模板实例
    typedef pattern_filter_bilinear_rgba<rgba16> pattern_filter_bilinear_rgba16;
    // 定义具体像素类型 rgba32 的双线性插值滤波器模板实例
    typedef pattern_filter_bilinear_rgba<rgba32> pattern_filter_bilinear_rgba32;
}


注释：


// 这是一个 C/C++ 预处理器指令，用于结束条件编译部分



#endif


注释：


// 这是一个 C/C++ 预处理器指令，用于结束条件编译指令块
```