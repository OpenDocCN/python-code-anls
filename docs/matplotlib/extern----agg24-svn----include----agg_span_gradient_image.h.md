# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_gradient_image.h`

```py
//----------------------------------------------------------------------------
// AGG Contribution Pack - Gradients 1 (AGG CP - Gradients 1)
// http://milan.marusinec.sk/aggcp
//
// For Anti-Grain Geometry - Version 2.4 
// http://www.antigrain.org
//
// Contribution Created By:
//  Milan Marusinec alias Milano
//  milan@marusinec.sk
//  Copyright (c) 2007-2008
//
// Permission to copy, use, modify, sell and distribute this software
// is granted provided this copyright notice appears in all copies.
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
// [History] -----------------------------------------------------------------
//
// 03.02.2008-Milano: Ported from Object Pascal code of AggPas
//

// 防止重复包含定义
#ifndef AGG_SPAN_GRADIENT_IMAGE_INCLUDED
// 定义头文件宏，标记为包含渐变图像跨度的文件
#define AGG_SPAN_GRADIENT_IMAGE_INCLUDED

// 包含基本的 AGG 头文件
#include "agg_basics.h"
// 包含渐变图像跨度的声明
#include "agg_span_gradient.h"
// 包含 RGBA 颜色类型声明
#include "agg_color_rgba.h"
// 包含渲染缓冲区声明
#include "agg_rendering_buffer.h"
// 包含 RGBA32 位像素格式声明
#include "agg_pixfmt_rgba.h"

// 命名空间开始
namespace agg
{

    //==========================================================one_color_function
    // 模板类定义，用于返回单一颜色
    template<class ColorT> class one_color_function
    {
    public:
        typedef ColorT color_type;

        color_type m_color; // 颜色对象

        // 构造函数，初始化颜色
        one_color_function() :
            m_color()
        {
        }

        // 静态函数，返回颜色数目（这里为1）
        static unsigned size() { return 1; }

        // 返回颜色的常量引用
        const color_type& operator [] (unsigned i) const 
        {
            return m_color;
        }

        // 返回颜色的指针
        color_type* operator [] (unsigned i)
        {
            return &m_color;
        }            
    };

    //==========================================================gradient_image
    // 模板类定义，表示渐变图像
    template<class ColorT> class gradient_image
    {
    private:
        //------------ fields
        typedef ColorT color_type; // 颜色类型定义
        typedef agg::pixfmt_rgba32 pixfmt_type; // RGBA32 像素格式类型

        agg::rgba8* m_buffer; // RGBA8 缓冲区指针

        int m_alocdx; // X 方向的分配大小
        int m_alocdy; // Y 方向的分配大小
        int m_width; // 图像宽度
        int m_height; // 图像高度

        color_type* m_color; // 颜色对象指针

        one_color_function<color_type> m_color_function; // 单一颜色函数对象


这段代码是C++语言的头文件部分，用于定义渐变图像处理的相关类和结构。注释中描述了文件的作用、来源、版权信息以及历史记录。每个类和函数的作用也都有相应的注释说明，以便于阅读和理解代码的功能和结构。
    // 定义一个公共类 gradient_image
    public:
        // 默认构造函数，初始化对象的成员变量
        gradient_image() :
            m_color_function(),   // 调用默认构造函数初始化 m_color_function
            m_buffer(NULL),       // 将 m_buffer 初始化为 NULL
            m_alocdx(0),          // 将 m_alocdx 初始化为 0
            m_alocdy(0),          // 将 m_alocdy 初始化为 0
            m_width(0),           // 将 m_width 初始化为 0
            m_height(0)           // 将 m_height 初始化为 0
        {
            m_color = m_color_function[0 ];  // 初始化 m_color 为 m_color_function 的第一个元素
        }
    
        // 析构函数，释放动态分配的内存
        ~gradient_image()
        {
            // 如果 m_buffer 不为 NULL，则释放其指向的内存
            if (m_buffer) { delete [] m_buffer; }
        }
    
        // 创建图像函数，返回指向图像数据的指针
        void* image_create(int width, int height )
        {
            void* result = NULL;  // 初始化返回结果指针为 NULL
    
            // 如果指定的宽度或高度大于当前已分配的宽度或高度
            if (width > m_alocdx || height > m_alocdy)
            {
                // 如果 m_buffer 不为 NULL，则释放其指向的内存
                if (m_buffer) { delete [] m_buffer; }
    
                m_buffer = NULL;  // 将 m_buffer 置为 NULL
                // 分配新的大小为 width * height 的 rgba8 类型内存块，并将其赋给 m_buffer
                m_buffer = new agg::rgba8[width * height];
    
                // 如果成功分配了内存
                if (m_buffer)
                {
                    m_alocdx = width;   // 更新 m_alocdx 为新分配的宽度
                    m_alocdy = height;  // 更新 m_alocdy 为新分配的高度
                }
                else
                {
                    m_alocdx = 0;  // 分配失败，将 m_alocdx 置为 0
                    m_alocdy = 0;  // 分配失败，将 m_alocdy 置为 0
                };
            };
    
            // 如果 m_buffer 不为 NULL
            if (m_buffer)
            {
                m_width  = width;   // 更新 m_width 为指定的宽度
                m_height = height;  // 更新 m_height 为指定的高度
    
                // 遍历每一行，将每行像素的 rgba 值设为 0
                for (int rows = 0; rows < height; rows++)
                {
                    agg::rgba8* row = &m_buffer[rows * m_alocdx ];
                    memset(row ,0 ,m_width * 4 );
                };
    
                result = m_buffer;  // 将 result 指向 m_buffer
            };
            return result;  // 返回图像数据的指针
        }
    
        // 返回 m_buffer 的指针
        void* image_buffer() { return m_buffer; }
        // 返回图像的宽度
        int   image_width()  { return m_width; }
        // 返回图像的高度
        int   image_height() { return m_height; }
        // 返回图像的跨度（每行像素占用的字节数）
        int   image_stride() { return m_alocdx * 4; }
    
        // 计算函数，根据指定的 x、y 坐标和 d 值计算颜色
        int calculate(int x, int y, int d) const
        {
            // 如果 m_buffer 不为 NULL
            if (m_buffer)
            {
                // 将 x、y 坐标右移 agg::gradient_subpixel_shift 位，得到像素坐标 px、py
                int px = x >> agg::gradient_subpixel_shift;
                int py = y >> agg::gradient_subpixel_shift;
    
                px %= m_width;  // 取余确保 px 在图像宽度范围内
    
                // 如果 px 小于 0，则加上图像宽度
                if (px < 0)
                {
                    px += m_width;
                }
    
                py %= m_height;  // 取余确保 py 在图像高度范围内
    
                // 如果 py 小于 0，则加上图像高度
                if (py < 0 )
                {
                    py += m_height;
                }
    
                // 找到指定位置像素的 rgba8 结构体指针
                rgba8* pixel = &m_buffer[py * m_alocdx + px ];
    
                // 将 m_color 设置为像素的 rgba 值
                m_color->r = pixel->r;
                m_color->g = pixel->g;
                m_color->b = pixel->b;
                m_color->a = pixel->a;
    
            }
            else
            {
                // 如果 m_buffer 为 NULL，则将 m_color 的 rgba 值设为 0
                m_color->r = 0;
                m_color->g = 0;
                m_color->b = 0;
                m_color->a = 0;
            }
            return 0;  // 返回计算结果，此处返回固定值 0
        }
    
        // 返回 m_color_function 对象的常引用
        const one_color_function<color_type>& color_function() const
        {
            return m_color_function;  // 返回 m_color_function 对象
        }
    
    };
}

#endif



// 结束一个 C++ 的条件编译指令块
}
// 结束一个 C++ 的预处理器指令 #endif
```