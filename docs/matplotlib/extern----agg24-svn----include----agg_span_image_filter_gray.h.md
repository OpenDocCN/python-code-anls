# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_image_filter_gray.h`

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
#ifndef AGG_SPAN_IMAGE_FILTER_GRAY_INCLUDED
#define AGG_SPAN_IMAGE_FILTER_GRAY_INCLUDED

#include "agg_basics.h"
#include "agg_color_gray.h"
#include "agg_span_image_filter.h"

// 命名空间agg的声明
namespace agg
{

    //==============================================span_image_filter_gray_nn
    // 模板类span_image_filter_gray_nn，继承自span_image_filter
    template<class Source, class Interpolator> 
    class span_image_filter_gray_nn : 
    public span_image_filter<Source, Interpolator>
    {
    public:
        // 类型定义
        typedef Source source_type;                      // 源类型
        typedef typename source_type::color_type color_type;  // 颜色类型
        typedef Interpolator interpolator_type;          // 插值器类型
        typedef span_image_filter<source_type, interpolator_type> base_type;  // 基类类型
        typedef typename color_type::value_type value_type;  // 值类型
        typedef typename color_type::calc_type calc_type;      // 计算类型
        typedef typename color_type::long_type long_type;      // 长整型类型

        //--------------------------------------------------------------------
        // 默认构造函数
        span_image_filter_gray_nn() {}
        
        // 构造函数，初始化基类和成员变量
        span_image_filter_gray_nn(source_type& src, 
                                  interpolator_type& inter) :
            base_type(src, inter, 0) 
        {}

        //--------------------------------------------------------------------
        // 生成函数，根据指定位置和长度生成灰度图像数据
        void generate(color_type* span, int x, int y, unsigned len)
        {
            // 初始化插值器，开始生成过程
            base_type::interpolator().begin(x + base_type::filter_dx_dbl(), 
                                            y + base_type::filter_dy_dbl(), len);
            do
            {
                // 获取当前插值后的坐标
                base_type::interpolator().coordinates(&x, &y);
                // 从源图像中获取灰度值并存入span
                span->v = *(const value_type*)
                    base_type::source().span(x >> image_subpixel_shift, 
                                             y >> image_subpixel_shift, 
                                             1);
                // 设置alpha通道为最大值
                span->a = color_type::full_value();
                ++span;             // 移动到下一个像素
                ++base_type::interpolator();  // 移动插值器到下一个位置
            } while(--len);         // 循环直到生成完所有像素
        }
    };
    //=========================================span_image_filter_gray_bilinear
    template<class Source, class Interpolator> 
    class span_image_filter_gray_bilinear : 
    public span_image_filter<Source, Interpolator>
    {
    public:
        typedef Source source_type;  // 定义源类型为 Source
        typedef typename source_type::color_type color_type;  // 定义颜色类型为 Source 类型中的 color_type
        typedef Interpolator interpolator_type;  // 定义插值器类型为 Interpolator
        typedef span_image_filter<source_type, interpolator_type> base_type;  // 定义基类类型为 span_image_filter<source_type, interpolator_type>
        typedef typename color_type::value_type value_type;  // 定义值类型为颜色类型的值类型
        typedef typename color_type::calc_type calc_type;  // 定义计算类型为颜色类型的计算类型
        typedef typename color_type::long_type long_type;  // 定义长整型类型为颜色类型的长整型
    
        //--------------------------------------------------------------------
        span_image_filter_gray_bilinear() {}  // 默认构造函数
    
        // 构造函数，初始化基类和插值器
        span_image_filter_gray_bilinear(source_type& src, interpolator_type& inter) :
            base_type(src, inter, 0) 
        {}
    
        //--------------------------------------------------------------------
        // 生成函数，计算灰度双线性滤波后的结果
        void generate(color_type* span, int x, int y, unsigned len)
        {
            base_type::interpolator().begin(x + base_type::filter_dx_dbl(), 
                                            y + base_type::filter_dy_dbl(), len);  // 初始化插值器
    
            long_type fg;  // 前景色值
            const value_type *fg_ptr;  // 前景色指针
            do
            {
                int x_hr;
                int y_hr;
    
                base_type::interpolator().coordinates(&x_hr, &y_hr);  // 获取插值后的坐标
    
                x_hr -= base_type::filter_dx_int();  // 考虑滤波器的 X 方向偏移
                y_hr -= base_type::filter_dy_int();  // 考虑滤波器的 Y 方向偏移
    
                int x_lr = x_hr >> image_subpixel_shift;  // 将高分辨率坐标右移得到低分辨率坐标
                int y_lr = y_hr >> image_subpixel_shift;
    
                fg = 0;  // 初始化前景色值
    
                x_hr &= image_subpixel_mask;  // 获取低位分数部分
                y_hr &= image_subpixel_mask;
    
                fg_ptr = (const value_type*)base_type::source().span(x_lr, y_lr, 2);  // 获取左上角像素的值
                fg    += *fg_ptr * (image_subpixel_scale - x_hr) * (image_subpixel_scale - y_hr);  // 计算双线性插值后的值
    
                fg_ptr = (const value_type*)base_type::source().next_x();  // 获取 X 方向下一个像素的值
                fg    += *fg_ptr * x_hr * (image_subpixel_scale - y_hr);  // 计算双线性插值后的值
    
                fg_ptr = (const value_type*)base_type::source().next_y();  // 获取 Y 方向下一个像素的值
                fg    += *fg_ptr * (image_subpixel_scale - x_hr) * y_hr;  // 计算双线性插值后的值
    
                fg_ptr = (const value_type*)base_type::source().next_x();  // 获取 X、Y 方向下一个像素的值
                fg    += *fg_ptr * x_hr * y_hr;  // 计算双线性插值后的值
    
                span->v = color_type::downshift(fg, image_subpixel_shift * 2);  // 将计算结果向下移位并存入 span 的亮度分量
                span->a = color_type::full_value();  // 设置 span 的 alpha 值为最大
                ++span;  // 移动到下一个 span
                ++base_type::interpolator();  // 移动到下一个插值点
    
            } while(--len);  // 循环直到处理完所有长度
        }
    };
    
    
    //====================================span_image_filter_gray_bilinear_clip
    template<class Source, class Interpolator> 
    class span_image_filter_gray_bilinear_clip : 
    public span_image_filter<Source, Interpolator>
    {
    private:
        color_type m_back_color;  // 定义私有成员变量 m_back_color
    };
    // 定义一个模板类 span_image_filter_gray_2x2，用于处理灰度图像的滤波操作，继承自 span_image_filter 类
    template<class Source, class Interpolator> 
    class span_image_filter_gray_2x2 : 
    public span_image_filter<Source, Interpolator>
    {
    };
    
    
    // 定义一个模板类 span_image_filter_gray，用于处理灰度图像的滤波操作，继承自 span_image_filter 类
    template<class Source, class Interpolator> 
    class span_image_filter_gray : 
    public span_image_filter<Source, Interpolator>
    {
    };
    
    
    // 定义一个模板类 span_image_resample_gray_affine，用于灰度图像的仿射重采样操作，继承自 span_image_resample_affine 类
    template<class Source> 
    class span_image_resample_gray_affine : 
    public span_image_resample_affine<Source>
    {
    };
    
    
    // 定义一个模板类 span_image_resample_gray，用于灰度图像的重采样操作，继承自 span_image_resample 类
    template<class Source, class Interpolator>
    class span_image_resample_gray : 
    public span_image_resample<Source, Interpolator>
    {
    };
}



#endif
```