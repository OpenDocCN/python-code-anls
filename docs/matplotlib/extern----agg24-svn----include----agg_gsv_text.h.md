# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_gsv_text.h`

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
// Class gsv_text
//
//----------------------------------------------------------------------------

#ifndef AGG_GSV_TEXT_INCLUDED
#define AGG_GSV_TEXT_INCLUDED

#include "agg_array.h"
#include "agg_conv_stroke.h"
#include "agg_conv_transform.h"

namespace agg
{


    //---------------------------------------------------------------gsv_text
    //
    // See Implementation agg_gsv_text.cpp 
    //
    class gsv_text
    {
        enum status
        {
            initial,        // 初始状态
            next_char,      // 下一个字符状态
            start_glyph,    // 开始字形状态
            glyph           // 字形状态
        };

    public:
        // 构造函数
        gsv_text();

        // 设置字体
        void font(const void* font);

        // 设置是否垂直翻转
        void flip(bool flip_y) { m_flip = flip_y; }

        // 加载字体文件
        void load_font(const char* file);

        // 设置文本大小
        void size(double height, double width=0.0);

        // 设置字符间距
        void space(double space);

        // 设置行间距
        void line_space(double line_space);

        // 设置文本起始点坐标
        void start_point(double x, double y);

        // 设置要绘制的文本内容
        void text(const char* text);
        
        // 计算文本的宽度
        double text_width();

        // 重置状态
        void rewind(unsigned path_id);

        // 获取顶点坐标
        unsigned vertex(double* x, double* y);

    private:
        // 不允许拷贝
        gsv_text(const gsv_text&);
        const gsv_text& operator = (const gsv_text&);

        // 获取16位无符号整数值，处理字节序
        int16u value(const int8u* p) const
        {
            int16u v;
            if(m_big_endian)
            {
                 *(int8u*)&v      = p[1];
                *((int8u*)&v + 1) = p[0];
            }
            else
            {
                 *(int8u*)&v      = p[0];
                *((int8u*)&v + 1) = p[1];
            }
            return v;
        }
    // 定义一个私有类 gsv_text，包含多个双精度浮点型成员变量和几个字符数组指针成员
    private:
        double          m_x;              // X 坐标
        double          m_y;              // Y 坐标
        double          m_start_x;        // 起始 X 坐标
        double          m_width;          // 宽度
        double          m_height;         // 高度
        double          m_space;          // 空间
        double          m_line_space;     // 行间距
        char            m_chr[2];         // 字符数组，最多两个字符
        char*           m_text;           // 字符指针
        pod_array<char> m_text_buf;       // 字符缓冲区，使用模板 pod_array
        char*           m_cur_chr;        // 当前字符指针
        const void*     m_font;           // 字体指针
        pod_array<char> m_loaded_font;    // 加载的字体数据缓冲区
        status          m_status;         // 状态
        bool            m_big_endian;     // 大端序标志
        bool            m_flip;           // 翻转标志
        int8u*          m_indices;        // 索引数组
        int8*           m_glyphs;         // 字形数组
        int8*           m_bglyph;         // 背景字形数组
        int8*           m_eglyph;         // 结束字形数组
        double          m_w;              // 宽度
        double          m_h;              // 高度
    };



    //--------------------------------------------------------gsv_text_outline
    // 定义一个模板类 gsv_text_outline，模板参数为 Transformer，默认为 trans_affine
    template<class Transformer = trans_affine> class gsv_text_outline
    {
    public:
        // 构造函数，接受 gsv_text 对象和 Transformer 对象的引用作为参数
        gsv_text_outline(gsv_text& text, Transformer& trans) :
          // 使用 text 初始化 m_polyline 成员
          m_polyline(text),
          // 使用 m_polyline 和 trans 初始化 m_trans 成员
          m_trans(m_polyline, trans)
        {
        }

        // 设置线条宽度的方法
        void width(double w) 
        { 
            m_polyline.width(w); 
        }

        // 设置转换器的方法，接受指向 Transformer 对象的指针
        void transformer(const Transformer* trans) 
        {
            m_trans->transformer(trans);
        }

        // 重置方法，接受路径 ID 作为参数
        void rewind(unsigned path_id) 
        { 
            // 调用 m_trans 的 rewind 方法，传入 path_id
            m_trans.rewind(path_id); 
            // 设置 m_polyline 的线段连接方式为圆角连接
            m_polyline.line_join(round_join);
            // 设置 m_polyline 的线段端点方式为圆角端点
            m_polyline.line_cap(round_cap);
        }

        // 获取顶点坐标的方法，接受指向 x 和 y 坐标的指针
        unsigned vertex(double* x, double* y)
        {
            return m_trans.vertex(x, y);
        }

    private:
        // 带转换器的描边转换器，模板参数为 gsv_text
        conv_stroke<gsv_text> m_polyline;
        // 描边转换器的转换对象，模板参数为 conv_stroke<gsv_text> 和 Transformer
        conv_transform<conv_stroke<gsv_text>, Transformer> m_trans;
    };
}
// 结束条件判断，关闭上一个预处理指令的块结构

#endif
// 结束当前条件编译块，指示预处理器结束条件编译部分
```