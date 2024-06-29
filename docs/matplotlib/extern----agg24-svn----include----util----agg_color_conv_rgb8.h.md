# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\util\agg_color_conv_rgb8.h`

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
//
// A set of functors used with color_conv(). See file agg_color_conv.h
// These functors can convert images with up to 8 bits per component.
// Use convertors in the following way:
//
// agg::color_conv(dst, src, agg::color_conv_XXXX_to_YYYY());
// whare XXXX and YYYY can be any of:
//  rgb24
//  bgr24
//  rgba32
//  abgr32
//  argb32
//  bgra32
//  rgb555
//  rgb565
//----------------------------------------------------------------------------

#ifndef AGG_COLOR_CONV_RGB8_INCLUDED
#define AGG_COLOR_CONV_RGB8_INCLUDED

#include "agg_basics.h"
#include "agg_color_conv.h"

namespace agg
{

    //-----------------------------------------------------color_conv_rgb24
    // Functor for converting RGB24 to BGR24 format
    class color_conv_rgb24
    {
    public:
        // Operator function for converting RGB24 to BGR24
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                // Temporary array to hold RGB components
                int8u tmp[3];
                tmp[0] = *src++; // Red component
                tmp[1] = *src++; // Green component
                tmp[2] = *src++; // Blue component
                *dst++ = tmp[2]; // Store Blue
                *dst++ = tmp[1]; // Store Green
                *dst++ = tmp[0]; // Store Red
            }
            while(--width); // Loop until width is decremented to zero
        }
    };

    // Typedefs for other conversions using the same functor
    typedef color_conv_rgb24 color_conv_rgb24_to_bgr24;
    typedef color_conv_rgb24 color_conv_bgr24_to_rgb24;

    // Typedefs for same format conversions
    typedef color_conv_same<3> color_conv_bgr24_to_bgr24;
    typedef color_conv_same<3> color_conv_rgb24_to_rgb24;



    //------------------------------------------------------color_conv_rgba32
    // Template class for RGBA32 color conversion
    template<int I1, int I2, int I3, int I4> class color_conv_rgba32
    {
    public:
        // Operator function for converting RGBA32 formats
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                // Temporary array to hold RGBA components
                int8u tmp[4];
                tmp[0] = *src++; // Red component
                tmp[1] = *src++; // Green component
                tmp[2] = *src++; // Blue component
                tmp[3] = *src++; // Alpha component
                *dst++ = tmp[I1]; // Store component based on template parameter I1
                *dst++ = tmp[I2]; // Store component based on template parameter I2
                *dst++ = tmp[I3]; // Store component based on template parameter I3
                *dst++ = tmp[I4]; // Store component based on template parameter I4
            }
            while(--width); // Loop until width is decremented to zero
        }
    };


    //------------------------------------------------------------------------
    typedef color_conv_rgba32<0,3,2,1> color_conv_argb32_to_abgr32; // 定义颜色转换器，将 ARGB32 格式转换为 ABGR32 格式
    typedef color_conv_rgba32<3,2,1,0> color_conv_argb32_to_bgra32; // 定义颜色转换器，将 ARGB32 格式转换为 BGRA32 格式
    typedef color_conv_rgba32<1,2,3,0> color_conv_argb32_to_rgba32; // 定义颜色转换器，将 ARGB32 格式转换为 RGBA32 格式
    typedef color_conv_rgba32<3,0,1,2> color_conv_bgra32_to_abgr32; // 定义颜色转换器，将 BGRA32 格式转换为 ABGR32 格式
    typedef color_conv_rgba32<3,2,1,0> color_conv_bgra32_to_argb32; // 定义颜色转换器，将 BGRA32 格式转换为 ARGB32 格式
    typedef color_conv_rgba32<2,1,0,3> color_conv_bgra32_to_rgba32; // 定义颜色转换器，将 BGRA32 格式转换为 RGBA32 格式
    typedef color_conv_rgba32<3,2,1,0> color_conv_rgba32_to_abgr32; // 定义颜色转换器，将 RGBA32 格式转换为 ABGR32 格式
    typedef color_conv_rgba32<3,0,1,2> color_conv_rgba32_to_argb32; // 定义颜色转换器，将 RGBA32 格式转换为 ARGB32 格式
    typedef color_conv_rgba32<2,1,0,3> color_conv_rgba32_to_bgra32; // 定义颜色转换器，将 RGBA32 格式转换为 BGRA32 格式
    typedef color_conv_rgba32<0,3,2,1> color_conv_abgr32_to_argb32; // 定义颜色转换器，将 ABGR32 格式转换为 ARGB32 格式
    typedef color_conv_rgba32<1,2,3,0> color_conv_abgr32_to_bgra32; // 定义颜色转换器，将 ABGR32 格式转换为 BGRA32 格式
    typedef color_conv_rgba32<3,2,1,0> color_conv_abgr32_to_rgba32; // 定义颜色转换器，将 ABGR32 格式转换为 RGBA32 格式
    
    //------------------------------------------------------------------------
    
    typedef color_conv_same<4> color_conv_rgba32_to_rgba32; // 定义相同颜色转换器，保持 RGBA32 格式不变
    typedef color_conv_same<4> color_conv_argb32_to_argb32; // 定义相同颜色转换器，保持 ARGB32 格式不变
    typedef color_conv_same<4> color_conv_bgra32_to_bgra32; // 定义相同颜色转换器，保持 BGRA32 格式不变
    typedef color_conv_same<4> color_conv_abgr32_to_abgr32; // 定义相同颜色转换器，保持 ABGR32 格式不变
    
    //--------------------------------------------color_conv_rgb24_rgba32
    
    // 定义从 RGB24 格式到 RGBA32 格式的颜色转换器模板
    template<int I1, int I2, int I3, int A> class color_conv_rgb24_rgba32
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            // 进行颜色转换，将 RGB24 格式的像素数据转换为 RGBA32 格式
            do
            {
                dst[I1] = *src++;
                dst[I2] = *src++;
                dst[I3] = *src++;
                dst[A]  = 255; // 设置 alpha 通道为不透明
                dst += 4; // 移动到下一个像素
            }
            while(--width); // 处理完所有像素
        }
    };
    
    //------------------------------------------------------------------------
    
    typedef color_conv_rgb24_rgba32<1,2,3,0> color_conv_rgb24_to_argb32; // 定义颜色转换器，将 RGB24 格式转换为 ARGB32 格式
    typedef color_conv_rgb24_rgba32<3,2,1,0> color_conv_rgb24_to_abgr32; // 定义颜色转换器，将 RGB24 格式转换为 ABGR32 格式
    typedef color_conv_rgb24_rgba32<2,1,0,3> color_conv_rgb24_to_bgra32; // 定义颜色转换器，将 RGB24 格式转换为 BGRA32 格式
    typedef color_conv_rgb24_rgba32<0,1,2,3> color_conv_rgb24_to_rgba32; // 定义颜色转换器，将 RGB24 格式转换为 RGBA32 格式
    typedef color_conv_rgb24_rgba32<3,2,1,0> color_conv_bgr24_to_argb32; // 定义颜色转换器，将 BGR24 格式转换为 ARGB32 格式
    typedef color_conv_rgb24_rgba32<1,2,3,0> color_conv_bgr24_to_abgr32; // 定义颜色转换器，将 BGR24 格式转换为 ABGR32 格式
    // 定义模板别名，将 BGR24 转换为 BGRA32 格式的颜色转换器
    typedef color_conv_rgb24_rgba32<0,1,2,3> color_conv_bgr24_to_bgra32; //----color_conv_bgr24_to_bgra32

    // 定义模板别名，将 BGR24 转换为 RGBA32 格式的颜色转换器
    typedef color_conv_rgb24_rgba32<2,1,0,3> color_conv_bgr24_to_rgba32; //----color_conv_bgr24_to_rgba32

    

    //-------------------------------------------------color_conv_rgba32_rgb24
    // RGBA32 到 RGB24 格式的颜色转换模板
    template<int I1, int I2, int I3> class color_conv_rgba32_rgb24
    {
    public:
        // 转换操作符，将 RGBA32 格式的像素数据转换为 RGB24 格式
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                // 按照指定的顺序复制源像素的指定通道到目标像素
                *dst++ = src[I1];
                *dst++ = src[I2];
                *dst++ = src[I3];
                // 跳过 Alpha 通道，移动到下一个像素
                src += 4;
            }
            while(--width);
        }
    };



    //------------------------------------------------------------------------
    // 定义模板别名，将 ARGB32 转换为 RGB24 格式的颜色转换器
    typedef color_conv_rgba32_rgb24<1,2,3> color_conv_argb32_to_rgb24; //----color_conv_argb32_to_rgb24

    // 定义模板别名，将 ABGR32 转换为 RGB24 格式的颜色转换器
    typedef color_conv_rgba32_rgb24<3,2,1> color_conv_abgr32_to_rgb24; //----color_conv_abgr32_to_rgb24

    // 定义模板别名，将 BGRA32 转换为 RGB24 格式的颜色转换器
    typedef color_conv_rgba32_rgb24<2,1,0> color_conv_bgra32_to_rgb24; //----color_conv_bgra32_to_rgb24

    // 定义模板别名，将 RGBA32 转换为 RGB24 格式的颜色转换器
    typedef color_conv_rgba32_rgb24<0,1,2> color_conv_rgba32_to_rgb24; //----color_conv_rgba32_to_rgb24

    // 定义模板别名，将 ARGB32 转换为 BGR24 格式的颜色转换器
    typedef color_conv_rgba32_rgb24<3,2,1> color_conv_argb32_to_bgr24; //----color_conv_argb32_to_bgr24

    // 定义模板别名，将 ABGR32 转换为 BGR24 格式的颜色转换器
    typedef color_conv_rgba32_rgb24<1,2,3> color_conv_abgr32_to_bgr24; //----color_conv_abgr32_to_bgr24

    // 定义模板别名，将 BGRA32 转换为 BGR24 格式的颜色转换器
    typedef color_conv_rgba32_rgb24<0,1,2> color_conv_bgra32_to_bgr24; //----color_conv_bgra32_to_bgr24

    // 定义模板别名，将 RGBA32 转换为 BGR24 格式的颜色转换器
    typedef color_conv_rgba32_rgb24<2,1,0> color_conv_rgba32_to_bgr24; //----color_conv_rgba32_to_bgr24


    //------------------------------------------------color_conv_rgb555_rgb24
    // RGB555 到 RGB24 格式的颜色转换模板
    template<int R, int B> class color_conv_rgb555_rgb24
    {
    public:
        // 转换操作符，将 RGB555 格式的像素数据转换为 RGB24 格式
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                // 从源数据中读取 RGB555 值
                unsigned rgb = *(int16u*)src;
                // 将 RGB555 值按照指定的顺序转换为 RGB24 格式
                dst[R] = (int8u)((rgb >> 7) & 0xF8);
                dst[1] = (int8u)((rgb >> 2) & 0xF8);
                dst[B] = (int8u)((rgb << 3) & 0xF8);
                // 移动到下一个像素
                src += 2;
                dst += 3;
            }
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    // 定义模板别名，将 RGB555 转换为 BGR24 格式的颜色转换器
    typedef color_conv_rgb555_rgb24<2,0> color_conv_rgb555_to_bgr24; //----color_conv_rgb555_to_bgr24

    // 定义模板别名，将 RGB555 转换为 RGB24 格式的颜色转换器
    typedef color_conv_rgb555_rgb24<0,2> color_conv_rgb555_to_rgb24; //----color_conv_rgb555_to_rgb24


    //-------------------------------------------------color_conv_rgb24_rgb555
    // RGB24 到 RGB555 格式的颜色转换模板
    template<int R, int B> class color_conv_rgb24_rgb555
    {
    //----color_conv_rgb555_rgba32
    template<int R, int G, int B, int A> class color_conv_rgb555_rgba32
    {
    public:
        // 将 RGB555 格式的颜色数据转换为 RGBA32 格式
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                // 从 src 中读取 RGB555 格式的颜色数据
                unsigned rgb = *(int16u*)src;
                // 将 RGB555 格式转换为 RGBA32 格式
                dst[R] = (rgb >> 7) & 0xF8;    // 提取红色分量并扩展至 8 位
                dst[G] = (rgb >> 2) & 0xF8;    // 提取绿色分量并扩展至 8 位
                dst[B] = (rgb << 3) & 0xF8;    // 提取蓝色分量并扩展至 8 位
                dst[A] = 0xFF;                 // 设置 alpha 通道为不透明
                // 移动指针到下一个像素
                src += 2;
                dst += 4;                      // RGBA32 格式每像素占 4 字节
            }
            while(--width);
        }
    };
    //------------------------------------------------color_conv_rgb555_rgba32
    // 定义一个模板类 color_conv_rgb555_rgba32，将 RGB555 转换为 RGBA32 格式
    template<int R, int G, int B, int A> class color_conv_rgb555_rgba32
    {
    public:
        // 重载操作符()，将 RGB555 格式的颜色数据转换为 RGBA32 格式
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                // 从源数据中读取 RGB555 格式的整数值
                int rgb = *(int16*)src;
                // 计算目标数据的 R 通道值，将 RGB555 中的 R 转换为 RGBA32 的 R
                dst[R] = (int8u)((rgb >> 7) & 0xF8);
                // 计算目标数据的 G 通道值，将 RGB555 中的 G 转换为 RGBA32 的 G
                dst[G] = (int8u)((rgb >> 2) & 0xF8);
                // 计算目标数据的 B 通道值，将 RGB555 中的 B 转换为 RGBA32 的 B
                dst[B] = (int8u)((rgb << 3) & 0xF8);
                // 计算目标数据的 A 通道值，将 RGB555 中的 A 转换为 RGBA32 的 A
                dst[A] = (int8u)(rgb >> 15);
                // 更新源数据和目标数据的指针位置，以处理下一个像素
                src += 2;
                dst += 4;
            }
            while(--width); // 继续处理下一个像素，直到处理完所有宽度
        }
    };
    
    
    //------------------------------------------------------------------------
    // 将不同排列顺序的 RGB555 格式转换为不同排列顺序的 RGBA32 格式
    typedef color_conv_rgb555_rgba32<1,2,3,0> color_conv_rgb555_to_argb32; //----color_conv_rgb555_to_argb32
    typedef color_conv_rgb555_rgba32<3,2,1,0> color_conv_rgb555_to_abgr32; //----color_conv_rgb555_to_abgr32
    typedef color_conv_rgb555_rgba32<2,1,0,3> color_conv_rgb555_to_bgra32; //----color_conv_rgb555_to_bgra32
    typedef color_conv_rgb555_rgba32<0,1,2,3> color_conv_rgb555_to_rgba32; //----color_conv_rgb555_to_rgba32
    
    
    //------------------------------------------------color_conv_rgba32_rgb555
    // 定义一个模板类 color_conv_rgba32_rgb555，将 RGBA32 转换为 RGB555 格式
    template<int R, int G, int B, int A> class color_conv_rgba32_rgb555
    {
    public:
        // 重载操作符()，将 RGBA32 格式的颜色数据转换为 RGB555 格式
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                // 从源数据中读取 RGBA32 格式的整数值
                *(int16u*)dst = (int16u)(((unsigned(src[R]) << 7) & 0x7C00) | 
                                         ((unsigned(src[G]) << 2) & 0x3E0)  |
                                         ((unsigned(src[B]) >> 3)) |
                                         ((unsigned(src[A]) << 8) & 0x8000));
                // 更新源数据和目标数据的指针位置，以处理下一个像素
                src += 4;
                dst += 2;
            }
            while(--width); // 继续处理下一个像素，直到处理完所有宽度
        }
    };
    
    
    //------------------------------------------------------------------------
    // 将不同排列顺序的 RGBA32 格式转换为 RGB555 格式
    typedef color_conv_rgba32_rgb555<1,2,3,0> color_conv_argb32_to_rgb555; //----color_conv_argb32_to_rgb555
    typedef color_conv_rgba32_rgb555<3,2,1,0> color_conv_abgr32_to_rgb555; //----color_conv_abgr32_to_rgb555
    typedef color_conv_rgba32_rgb555<2,1,0,3> color_conv_bgra32_to_rgb555; //----color_conv_bgra32_to_rgb555
    typedef color_conv_rgba32_rgb555<0,1,2,3> color_conv_rgba32_to_rgb555; //----color_conv_rgba32_to_rgb555
    
    
    //------------------------------------------------color_conv_rgb565_rgba32
    // 定义一个模板类 color_conv_rgb565_rgba32，将 RGB565 格式转换为 RGBA32 格式
    template<int R, int G, int B, int A> class color_conv_rgb565_rgba32
    {
    public:
        // 重载操作符()，将 RGB565 格式的颜色数据转换为 RGBA32 格式
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                // 从源数据中读取 RGB565 格式的整数值
                int rgb = *(int16*)src;
                // 计算目标数据的 R 通道值，将 RGB565 中的 R 转换为 RGBA32 的 R
                dst[R] = (rgb >> 8) & 0xF8;
                // 计算目标数据的 G 通道值，将 RGB565 中的 G 转换为 RGBA32 的 G
                dst[G] = (rgb >> 3) & 0xFC;
                // 计算目标数据的 B 通道值，将 RGB565 中的 B 转换为 RGBA32 的 B
                dst[B] = (rgb << 3) & 0xF8;
                // 设置目标数据的 A 通道值为 255，即不透明
                dst[A] = 255;
                // 更新源数据和目标数据的指针位置，以处理下一个像素
                src += 2;
                dst += 4;
            }
            while(--width); // 继续处理下一个像素，直到处理完所有宽度
        }
    };
    //------------------------------------------------------------------------
    typedef color_conv_rgb565_rgba32<1,2,3,0> color_conv_rgb565_to_argb32; //----color_conv_rgb565_to_argb32
    typedef color_conv_rgb565_rgba32<3,2,1,0> color_conv_rgb565_to_abgr32; //----color_conv_rgb565_to_abgr32
    typedef color_conv_rgb565_rgba32<2,1,0,3> color_conv_rgb565_to_bgra32; //----color_conv_rgb565_to_bgra32
    typedef color_conv_rgb565_rgba32<0,1,2,3> color_conv_rgb565_to_rgba32; //----color_conv_rgb565_to_rgba32
    
    //------------------------------------------------color_conv_rgba32_rgb565
    template<int R, int G, int B> class color_conv_rgba32_rgb565
    {
    public:
        // 将 RGBA32 格式的颜色转换为 RGB565 格式
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                // 从源数据中提取 R、G、B 分量，并将其转换为 RGB565 格式
                *(int16u*)dst = (int16u)(((unsigned(src[R]) << 8) & 0xF800) | 
                                         ((unsigned(src[G]) << 3) & 0x7E0)  |
                                         ((unsigned(src[B]) >> 3)));
                src += 4;
                dst += 2;
            }
            while(--width);
        }
    };
    
    //------------------------------------------------------------------------
    typedef color_conv_rgba32_rgb565<1,2,3> color_conv_argb32_to_rgb565; //----color_conv_argb32_to_rgb565
    typedef color_conv_rgba32_rgb565<3,2,1> color_conv_abgr32_to_rgb565; //----color_conv_abgr32_to_rgb565
    typedef color_conv_rgba32_rgb565<2,1,0> color_conv_bgra32_to_rgb565; //----color_conv_bgra32_to_rgb565
    typedef color_conv_rgba32_rgb565<0,1,2> color_conv_rgba32_to_rgb565; //----color_conv_rgba32_to_rgb565
    
    //---------------------------------------------color_conv_rgb555_to_rgb565
    class color_conv_rgb555_to_rgb565
    {
    public:
        // 将 RGB555 格式的颜色转换为 RGB565 格式
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                unsigned rgb = *(int16u*)src;
                // 将 RGB555 格式的颜色数据转换为 RGB565 格式
                *(int16u*)dst = (int16u)(((rgb << 1) & 0xFFC0) | (rgb & 0x1F));
                src += 2;
                dst += 2;
            }
            while(--width);
        }
    };
    
    //----------------------------------------------color_conv_rgb565_to_rgb555
    class color_conv_rgb565_to_rgb555
    {
    public:
        // 将 RGB565 格式的颜色转换为 RGB555 格式
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                unsigned rgb = *(int16u*)src;
                // 将 RGB565 格式的颜色数据转换为 RGB555 格式
                *(int16u*)dst = (int16u)(((rgb >> 1) & 0x7FE0) | (rgb & 0x1F));
                src += 2;
                dst += 2;
            }
            while(--width);
        }
    };
    
    //------------------------------------------------------------------------
    typedef color_conv_same<2> color_conv_rgb555_to_rgb555; //----color_conv_rgb555_to_rgb555
    // 定义一个模板类 color_conv_rgb24_gray8，用于将 RGB24 格式的颜色转换为灰度值
    template<int R, int B> class color_conv_rgb24_gray8
    {
    public:
        // 重载函数调用操作符，将 RGB24 格式的颜色转换为灰度值
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            // 使用加权平均方法将 RGB24 格式的颜色值转换为灰度值
            do
            {
                *dst++ = (src[R]*77 + src[1]*150 + src[B]*29) >> 8;
                src += 3;
            }
            while(--width);
        }
    };
    
    // 定义模板别名 color_conv_rgb24_to_gray8，使用 color_conv_rgb24_gray8 模板将 RGB24 格式转换为灰度值
    typedef color_conv_rgb24_gray8<0,2> color_conv_rgb24_to_gray8; //----color_conv_rgb24_to_gray8
    
    // 定义模板别名 color_conv_bgr24_to_gray8，使用 color_conv_rgb24_gray8 模板将 BGR24 格式转换为灰度值
    typedef color_conv_rgb24_gray8<2,0> color_conv_bgr24_to_gray8; //----color_conv_bgr24_to_gray8
}
// 结束条件：关闭前述的条件编译指令段落

#endif
// 若定义了条件编译指令，则包含该段代码块在内，否则排除
```