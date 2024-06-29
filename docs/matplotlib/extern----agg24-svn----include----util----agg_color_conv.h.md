# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\util\agg_color_conv.h`

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
// Conversion from one colorspace/pixel format to another
//
//----------------------------------------------------------------------------

#ifndef AGG_COLOR_CONV_INCLUDED
#define AGG_COLOR_CONV_INCLUDED

#include <string.h>
#include "agg_basics.h"
#include "agg_rendering_buffer.h"

// 命名空间 agg 开始
namespace agg
{

    //--------------------------------------------------------------color_conv
    // 模板函数 color_conv 用于颜色空间/像素格式的转换
    template<class RenBuf, class CopyRow> 
    void color_conv(RenBuf* dst, const RenBuf* src, CopyRow copy_row_functor)
    {
        // 获取源图像和目标图像的宽度和高度
        unsigned width = src->width();
        unsigned height = src->height();

        // 如果目标图像的尺寸小于源图像，则调整处理的宽度和高度
        if(dst->width()  < width)  width  = dst->width();
        if(dst->height() < height) height = dst->height();

        // 如果处理的宽度大于 0
        if(width)
        {
            unsigned y;
            // 遍历每一行图像数据
            for(y = 0; y < height; y++)
            {
                // 调用复制行函数，将源图像的行复制到目标图像
                copy_row_functor(dst->row_ptr(0, y, width), 
                                 src->row_ptr(y), 
                                 width);
            }
        }
    }


    //---------------------------------------------------------color_conv_row
    // 函数 color_conv_row 用于单独的行转换操作
    template<class CopyRow> 
    void color_conv_row(int8u* dst, 
                        const int8u* src,
                        unsigned width,
                        CopyRow copy_row_functor)
    {
        // 调用复制行函数，将源图像的行复制到目标图像
        copy_row_functor(dst, src, width);
    }


    //---------------------------------------------------------color_conv_same
    // 类模板 color_conv_same 用于相同像素格式的行数据拷贝
    template<int BPP> class color_conv_same
    {
    public:
        // 重载函数调用操作符，使用 memmove 完成数据拷贝
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            memmove(dst, src, width*BPP);
        }
    };


    // Generic pixel converter.
    // 通用像素转换器，用于从源格式读取像素并写入目标格式
    template<class DstFormat, class SrcFormat>
    struct conv_pixel
    {
        void operator()(void* dst, const void* src) const
        {
            // 从源格式读取像素，并写入目标格式
            DstFormat::write_plain_color(dst, SrcFormat::read_plain_color(src));
        }
    };

    // Generic row converter. Uses conv_pixel to convert individual pixels.
    // 通用行转换器，使用 conv_pixel 转换每个像素
    template<class DstFormat, class SrcFormat>
    struct conv_row
    // 定义一个函数对象结构体 conv_row，用于像素格式转换的行处理
    {
        void operator()(void* dst, const void* src, unsigned width) const
        {
            // 创建一个 conv_pixel 对象，用于格式转换
            conv_pixel<DstFormat, SrcFormat> conv;
            do
            {
                // 对单个像素进行格式转换
                conv(dst, src);
                // 更新目标和源指针以处理下一个像素
                dst = (int8u*)dst + DstFormat::pix_width;
                src = (int8u*)src + SrcFormat::pix_width;
            }
            while (--width); // 循环直到处理完所有像素
        }
    };

    // 当源格式和目标格式相同时的特化版本
    template<class Format>
    struct conv_row<Format, Format>
    {
        void operator()(void* dst, const void* src, unsigned width) const
        {
            // 直接使用 memmove 进行内存拷贝，因为源和目标格式相同
            memmove(dst, src, width * Format::pix_width);
        }
    };

    // 顶层转换函数，将一个像素格式转换为另一个
    template<class DstFormat, class SrcFormat, class RenBuf>
    void convert(RenBuf* dst, const RenBuf* src)
    {
        // 调用 color_conv 函数，传入转换器 conv_row<DstFormat, SrcFormat>() 进行颜色转换
        color_conv(dst, src, conv_row<DstFormat, SrcFormat>());
    }
}
# 结束一个预处理器条件编译指令的定义部分

#endif
# 结束一个条件编译指令的整体范围
```