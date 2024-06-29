# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\util\agg_color_conv_rgb16.h`

```
```cpp`
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
// This part of the library has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------
//
// A set of functors used with color_conv(). See file agg_color_conv.h
// These functors can convert images with up to 8 bits per component.
// Use convertors in the following way:
//
// agg::color_conv(dst, src, agg::color_conv_XXXX_to_YYYY());
//----------------------------------------------------------------------------

#ifndef AGG_COLOR_CONV_RGB16_INCLUDED
#define AGG_COLOR_CONV_RGB16_INCLUDED

#include "agg_basics.h"
#include "agg_color_conv.h"

namespace agg
{

    //-------------------------------------------------color_conv_gray16_to_gray8
    // Functor to convert 16-bit grayscale to 8-bit grayscale
    class color_conv_gray16_to_gray8
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            int16u* s = (int16u*)src;
            // Loop through each pixel and convert 16-bit to 8-bit grayscale
            do
            {
                *dst++ = *s++ >> 8;
            }
            while(--width);
        }
    };


    //-----------------------------------------------------color_conv_rgb24_rgb48
    // Functor to convert 24-bit RGB to 48-bit RGB
    template<int I1, int I3> class color_conv_rgb24_rgb48
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            int16u* d = (int16u*)dst;
            // Loop through each pixel and convert 24-bit RGB to 48-bit RGB
            do
            {
                *d++ = (src[I1] << 8) | src[I1];
                *d++ = (src[1]  << 8) | src[1] ;
                *d++ = (src[I3] << 8) | src[I3];
                src += 3;
            }
            while(--width);
        }
    };

    // Aliases for different RGB to 48-bit RGB conversions
    typedef color_conv_rgb24_rgb48<0,2> color_conv_rgb24_to_rgb48;
    typedef color_conv_rgb24_rgb48<0,2> color_conv_bgr24_to_bgr48;
    typedef color_conv_rgb24_rgb48<2,0> color_conv_rgb24_to_bgr48;
    typedef color_conv_rgb24_rgb48<2,0> color_conv_bgr24_to_rgb48;

    //-----------------------------------------------------color_conv_rgb24_rgb48
    // Functor to convert 48-bit RGB to 24-bit RGB
    template<int I1, int I3> class color_conv_rgb48_rgb24
    {
    //----------------------------------------------color_conv_rgbAAA_rgb24
    // 模板类定义，将特定格式的RGBAAA颜色转换为RGB24格式
    template<int R, int B> class color_conv_rgbAAA_rgb24
    {
    public:
        // 转换运算符重载，将输入的RGBAAA格式数据转换为RGB24格式数据
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                // 从源数据中读取一个32位RGBAAA值
                int32u rgb = *(int32u*)src;
                // 将RGBAAA格式中的R通道数据转换并写入目标RGB24格式中
                dst[R] = int8u(rgb >> 22);
                // 将RGBAAA格式中的G通道数据转换并写入目标RGB24格式中
                dst[1] = int8u(rgb >> 12);
                // 将RGBAAA格式中的B通道数据转换并写入目标RGB24格式中
                dst[B] = int8u(rgb >> 2);
                // 源数据指针移动到下一个32位值
                src += 4;
                // 目标数据指针移动到下一个像素的起始位置
                dst += 3;
            }
            // 按像素宽度递减循环，直到处理完所有像素
            while(--width);
        }
    };

    // 将RGBAAA到RGB24的转换定义为不同的类型别名
    typedef color_conv_rgbAAA_rgb24<0,2> color_conv_rgbAAA_to_rgb24;
    typedef color_conv_rgbAAA_rgb24<2,0> color_conv_rgbAAA_to_bgr24;
    typedef color_conv_rgbAAA_rgb24<2,0> color_conv_bgrAAA_to_rgb24;
    typedef color_conv_rgbAAA_rgb24<0,2> color_conv_bgrAAA_to_bgr24;


    //----------------------------------------------color_conv_rgbBBA_rgb24
    // 模板类定义，将特定格式的RGBBBA颜色转换为RGB24格式
    template<int R, int B> class color_conv_rgbBBA_rgb24
    {
    public:
        // 转换运算符重载，将输入的RGBBBA格式数据转换为RGB24格式数据
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                // 从源数据中读取一个32位RGBBBA值
                int32u rgb = *(int32u*)src;
                // 将RGBBBA格式中的R通道数据转换并写入目标RGB24格式中
                dst[R] = int8u(rgb >> 24);
                // 将RGBBBA格式中的G通道数据转换并写入目标RGB24格式中
                dst[1] = int8u(rgb >> 13);
                // 将RGBBBA格式中的B通道数据转换并写入目标RGB24格式中
                dst[B] = int8u(rgb >> 2);
                // 源数据指针移动到下一个32位值
                src += 4;
                // 目标数据指针移动到下一个像素的起始位置
                dst += 3;
            }
            // 按像素宽度递减循环，直到处理完所有像素
            while(--width);
        }
    };

    // 将RGBBBA到RGB24的转换定义为不同的类型别名
    typedef color_conv_rgbBBA_rgb24<0,2> color_conv_rgbBBA_to_rgb24;
    typedef color_conv_rgbBBA_rgb24<2,0> color_conv_rgbBBA_to_bgr24;


    //----------------------------------------------color_conv_bgrABB_rgb24
    // 模板类定义，将特定格式的BGRABB颜色转换为RGB24格式
    template<int B, int R> class color_conv_bgrABB_rgb24
    {
    public:
        // 转换运算符重载，将输入的BGRABB格式数据转换为RGB24格式数据
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                // 从源数据中读取一个32位BGRABB值
                int32u bgr = *(int32u*)src;
                // 将BGRABB格式中的R通道数据转换并写入目标RGB24格式中
                dst[R] = int8u(bgr >> 3);
                // 将BGRABB格式中的G通道数据转换并写入目标RGB24格式中
                dst[1] = int8u(bgr >> 14);
                // 将BGRABB格式中的B通道数据转换并写入目标RGB24格式中
                dst[B] = int8u(bgr >> 24);
                // 源数据指针移动到下一个32位值
                src += 4;
                // 目标数据指针移动到下一个像素的起始位置
                dst += 3;
            }
            // 按像素宽度递减循环，直到处理完所有像素
            while(--width);
        }
    };

    // 将BGRABB到RGB24的转换定义为不同的类型别名
    typedef color_conv_bgrABB_rgb24<2,0> color_conv_bgrABB_to_rgb24;
    typedef color_conv_bgrABB_rgb24<0,2> color_conv_bgrABB_to_bgr24;
    // 定义一个类型别名，将 color_conv_bgrABB_rgb24 模板化，用参数 0 和 2 实例化为 color_conv_bgrABB_to_bgr24。
    
    //-------------------------------------------------color_conv_rgba64_rgba32
    // color_conv_rgba64_rgba32 模板类的声明，该类用于将 RGBA64 格式的颜色转换为 RGBA32 格式。
    
    template<int I1, int I2, int I3, int I4> class color_conv_rgba64_rgba32
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                // 将 RGBA64 格式的颜色数据转换为 RGBA32 格式
                *dst++ = int8u(((int16u*)src)[I1] >> 8);
                *dst++ = int8u(((int16u*)src)[I2] >> 8);
                *dst++ = int8u(((int16u*)src)[I3] >> 8);
                *dst++ = int8u(((int16u*)src)[I4] >> 8); 
                src += 8; // 指向下一个 RGBA64 数据块
            }
            while(--width); // 处理每一行的像素数据，直到 width 为 0
        }
    };
    
    //------------------------------------------------------------------------
    // 将 color_conv_rgba64_rgba32 模板化，参数为 0, 1, 2, 3，形成 color_conv_rgba64_to_rgba32 类型别名。
    
    typedef color_conv_rgba64_rgba32<0,1,2,3> color_conv_rgba64_to_rgba32; //----color_conv_rgba64_to_rgba32
    typedef color_conv_rgba64_rgba32<0,1,2,3> color_conv_argb64_to_argb32; //----color_conv_argb64_to_argb32
    typedef color_conv_rgba64_rgba32<0,1,2,3> color_conv_bgra64_to_bgra32; //----color_conv_bgra64_to_bgra32
    typedef color_conv_rgba64_rgba32<0,1,2,3> color_conv_abgr64_to_abgr32; //----color_conv_abgr64_to_abgr32
    typedef color_conv_rgba64_rgba32<0,3,2,1> color_conv_argb64_to_abgr32; //----color_conv_argb64_to_abgr32
    typedef color_conv_rgba64_rgba32<3,2,1,0> color_conv_argb64_to_bgra32; //----color_conv_argb64_to_bgra32
    typedef color_conv_rgba64_rgba32<1,2,3,0> color_conv_argb64_to_rgba32; //----color_conv_argb64_to_rgba32
    typedef color_conv_rgba64_rgba32<3,0,1,2> color_conv_bgra64_to_abgr32; //----color_conv_bgra64_to_abgr32
    typedef color_conv_rgba64_rgba32<3,2,1,0> color_conv_bgra64_to_argb32; //----color_conv_bgra64_to_argb32
    typedef color_conv_rgba64_rgba32<2,1,0,3> color_conv_bgra64_to_rgba32; //----color_conv_bgra64_to_rgba32
    typedef color_conv_rgba64_rgba32<3,2,1,0> color_conv_rgba64_to_abgr32; //----color_conv_rgba64_to_abgr32
    typedef color_conv_rgba64_rgba32<3,0,1,2> color_conv_rgba64_to_argb32; //----color_conv_rgba64_to_argb32
    typedef color_conv_rgba64_rgba32<2,1,0,3> color_conv_rgba64_to_bgra32; //----color_conv_rgba64_to_bgra32
    typedef color_conv_rgba64_rgba32<0,3,2,1> color_conv_abgr64_to_argb32; //----color_conv_abgr64_to_argb32
    typedef color_conv_rgba64_rgba32<1,2,3,0> color_conv_abgr64_to_bgra32; //----color_conv_abgr64_to_bgra32
    typedef color_conv_rgba64_rgba32<3,2,1,0> color_conv_abgr64_to_rgba32; //----color_conv_abgr64_to_rgba32
    
    
    //--------------------------------------------color_conv_rgb24_rgba64
    // color_conv_rgb24_rgba64 模板类的声明，该类用于将 RGB24 格式的颜色转换为 RGBA64 格式。
    
    template<int I1, int I2, int I3, int A> class color_conv_rgb24_rgba64
    {
    template<int R, int B> class color_conv_rgb24_gray16
    {
    public:
        // 将 RGB24 格式转换为 Gray16 格式的颜色空间转换器
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            int16u* d = (int16u*)dst;
            do
            {
                // 根据给定的 R 和 B 分量权重，将 RGB24 格式转换为 Gray16 格式
                *d++ = src[R]*77 + src[1]*150 + src[B]*29;
                // 指针移动到下一个像素
                src += 3;
            }
            // 处理完所有像素后结束循环
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    // 定义一系列的 RGB24 到 RGBA64 颜色空间转换器模板实例

    // RGB24 到 ARGB64 格式的颜色空间转换器模板实例
    typedef color_conv_rgb24_rgba64<1,2,3,0> color_conv_rgb24_to_argb64; //----color_conv_rgb24_to_argb64

    // RGB24 到 ABGR64 格式的颜色空间转换器模板实例
    typedef color_conv_rgb24_rgba64<3,2,1,0> color_conv_rgb24_to_abgr64; //----color_conv_rgb24_to_abgr64

    // RGB24 到 BGRA64 格式的颜色空间转换器模板实例
    typedef color_conv_rgb24_rgba64<2,1,0,3> color_conv_rgb24_to_bgra64; //----color_conv_rgb24_to_bgra64

    // RGB24 到 RGBA64 格式的颜色空间转换器模板实例
    typedef color_conv_rgb24_rgba64<0,1,2,3> color_conv_rgb24_to_rgba64; //----color_conv_rgb24_to_rgba64

    // BGR24 到 ARGB64 格式的颜色空间转换器模板实例
    typedef color_conv_rgb24_rgba64<3,2,1,0> color_conv_bgr24_to_argb64; //----color_conv_bgr24_to_argb64

    // BGR24 到 ABGR64 格式的颜色空间转换器模板实例
    typedef color_conv_rgb24_rgba64<1,2,3,0> color_conv_bgr24_to_abgr64; //----color_conv_bgr24_to_abgr64

    // BGR24 到 BGRA64 格式的颜色空间转换器模板实例
    typedef color_conv_rgb24_rgba64<0,1,2,3> color_conv_bgr24_to_bgra64; //----color_conv_bgr24_to_bgra64

    // BGR24 到 RGBA64 格式的颜色空间转换器模板实例
    typedef color_conv_rgb24_rgba64<2,1,0,3> color_conv_bgr24_to_rgba64; //----color_conv_bgr24_to_rgba64

    // RGB24 到 Gray16 格式的颜色空间转换器模板实例
    typedef color_conv_rgb24_gray16<0,2> color_conv_rgb24_to_gray16;

    // BGR24 到 Gray16 格式的颜色空间转换器模板实例
    typedef color_conv_rgb24_gray16<2,0> color_conv_bgr24_to_gray16;
}


注释：

// 这是 C/C++ 中的预处理指令，表示条件编译结束的标记，对应于 #ifdef 或 #if 的开始部分



#endif


注释：

// 这是 C/C++ 中的预处理指令，用于结束条件编译块，对应于 #ifdef 或 #if 的结束部分
```