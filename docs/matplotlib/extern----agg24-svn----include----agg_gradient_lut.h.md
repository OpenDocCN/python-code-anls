# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_gradient_lut.h`

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

#ifndef AGG_GRADIENT_LUT_INCLUDED
#define AGG_GRADIENT_LUT_INCLUDED

#include "agg_array.h"
#include "agg_dda_line.h"
#include "agg_color_rgba.h"
#include "agg_color_gray.h"

namespace agg
{

    //======================================================color_interpolator
    // 颜色插值器模板类，用于在两种颜色之间进行插值
    template<class ColorT> struct color_interpolator
    {
    public:
        typedef ColorT color_type;

        // 构造函数，初始化插值器
        color_interpolator(const color_type& c1, 
                           const color_type& c2, 
                           unsigned len) :
            m_c1(c1),
            m_c2(c2),
            m_len(len),
            m_count(0)
        {}

        // 前缀递增操作符，用于推进插值器的状态
        void operator ++ ()
        {
            ++m_count;
        }

        // 获取当前插值状态下的颜色值
        color_type color() const
        {
            return m_c1.gradient(m_c2, double(m_count) / m_len);
        }

    private:
        color_type m_c1;  // 起始颜色
        color_type m_c2;  // 终止颜色
        unsigned   m_len; // 插值长度
        unsigned   m_count; // 当前插值计数
    };

    //========================================================================
    // Fast specialization for rgba8
    // rgba8 类型的快速特化，优化颜色插值操作
    template<> struct color_interpolator<rgba8>
    {
    public:
        typedef rgba8 color_type;

        // 构造函数，初始化 rgba8 类型的插值器
        color_interpolator(const color_type& c1, 
                           const color_type& c2, 
                           unsigned len) :
            r(c1.r, c2.r, len),
            g(c1.g, c2.g, len),
            b(c1.b, c2.b, len),
            a(c1.a, c2.a, len)
        {}

        // 前缀递增操作符，推进插值器状态的各个通道
        void operator ++ ()
        {
            ++r; ++g; ++b; ++a;
        }

        // 获取当前插值状态下的 rgba8 类型的颜色值
        color_type color() const
        {
            return color_type(r.y(), g.y(), b.y(), a.y());
        }

    private:
        agg::dda_line_interpolator<14> r, g, b, a; // 使用 dda_line_interpolator 优化的各通道插值器
    };

    //========================================================================
    // Fast specialization for gray8
    // gray8 类型的快速特化，优化颜色插值操作
    template<> struct color_interpolator<gray8>
    {
    //============================================================gradient_lut
    // gradient_lut 类模板的定义开始

    template<class ColorInterpolator, 
             unsigned ColorLutSize=256> class gradient_lut
    {
    public:
        // 类型定义
        typedef ColorInterpolator interpolator_type; // 定义插值器类型
        typedef typename interpolator_type::color_type color_type; // 定义颜色类型
        enum { color_lut_size = ColorLutSize }; // 颜色查找表大小常量

        //--------------------------------------------------------------------
        // 构造函数：初始化颜色查找表
        gradient_lut() : m_color_lut(color_lut_size) {}
        // 构造颜色查找表

        // Build Gradient Lut
        // 构建渐变查找表的方法说明
        // 首先调用 remove_all() 清空之前的设置，
        // 然后至少调用两次 add_color() 添加颜色点，
        // 最后调用 build_lut() 构建查找表。
        // add_color() 方法中的参数 "offset" 必须在 [0...1] 范围内，
        // 它定义了一个颜色停止点，就像 SVG 规范中渐变和图案部分所描述的那样。
        // 最简单的线性渐变是：
        //    gradient_lut.add_color(0.0, start_color);
        //    gradient_lut.add_color(1.0, end_color);
        //--------------------------------------------------------------------
        void remove_all(); // 清空颜色查找表
        void add_color(double offset, const color_type& color); // 添加颜色到查找表
        void build_lut(); // 构建颜色查找表

        // Size-index Interface. This class can be used directly as the 
        // ColorF in span_gradient. All it needs is two access methods 
        // size() and operator [].
        // 大小-索引接口：这个类可以直接作为 span_gradient 中的 ColorF 使用。
        // 它只需要两个访问方法 size() 和 operator []。
        //--------------------------------------------------------------------
        static unsigned size() // 返回颜色查找表的大小
        { 
            return color_lut_size; 
        }
        const color_type& operator [] (unsigned i) const // 访问颜色查找表中的元素
        { 
            return m_color_lut[i]; 
        }
        
    private:
        std::vector<color_type> m_color_lut; // 存储颜色查找表的向量
    };
    private:
        //--------------------------------------------------------------------
        // 结构体定义：表示颜色点，包含偏移量和颜色信息
        struct color_point
        {
            double     offset;   // 偏移量
            color_type color;    // 颜色

            // 默认构造函数
            color_point() {}

            // 带参数的构造函数，初始化偏移量和颜色
            color_point(double off, const color_type& c) : 
                offset(off), color(c)
            {
                // 确保偏移量在 [0.0, 1.0] 范围内
                if(offset < 0.0) offset = 0.0;
                if(offset > 1.0) offset = 1.0;
            }
        };
        
        // 使用 pod_bvector 定义颜色点向量的类型
        typedef agg::pod_bvector<color_point, 4> color_profile_type;

        // 使用 pod_array 定义颜色查找表的类型
        typedef agg::pod_array<color_type>       color_lut_type;

        // 静态方法：比较两个颜色点的偏移量大小
        static bool offset_less(const color_point& a, const color_point& b)
        {
            return a.offset < b.offset;
        }

        // 静态方法：比较两个颜色点的偏移量是否相等
        static bool offset_equal(const color_point& a, const color_point& b)
        {
            return a.offset == b.offset;
        }

        //--------------------------------------------------------------------
        // 颜色点向量，存储颜色点的序列
        color_profile_type  m_color_profile;

        // 颜色查找表，存储插值后的颜色值序列
        color_lut_type      m_color_lut;
    };

//------------------------------------------------------------------------
// 模板函数实现：移除所有颜色点
template<class T, unsigned S>
void gradient_lut<T,S>::remove_all()
{ 
    // 调用颜色点向量的移除所有元素的方法
    m_color_profile.remove_all(); 
}

//------------------------------------------------------------------------
// 模板函数实现：添加颜色点
template<class T, unsigned S>
void gradient_lut<T,S>::add_color(double offset, const color_type& color)
{
    // 向颜色点向量中添加新的颜色点
    m_color_profile.add(color_point(offset, color));
}

//------------------------------------------------------------------------
// 模板函数实现：构建颜色查找表
template<class T, unsigned S>
void gradient_lut<T,S>::build_lut()
{
    // 对颜色点向量按偏移量进行快速排序
    quick_sort(m_color_profile, offset_less);

    // 去除重复的颜色点
    m_color_profile.cut_at(remove_duplicates(m_color_profile, offset_equal));

    // 如果颜色点数量大于等于2，则开始构建颜色查找表
    if(m_color_profile.size() >= 2)
    {
        unsigned i;
        unsigned start = uround(m_color_profile[0].offset * color_lut_size);
        unsigned end;
        color_type c = m_color_profile[0].color;

        // 初始化查找表的前部分
        for(i = 0; i < start; i++) 
        {
            m_color_lut[i] = c;
        }

        // 插值生成中间部分的颜色值
        for(i = 1; i < m_color_profile.size(); i++)
        {
            end  = uround(m_color_profile[i].offset * color_lut_size);
            interpolator_type ci(m_color_profile[i-1].color, 
                                 m_color_profile[i  ].color, 
                                 end - start + 1);
            while(start < end)
            {
                m_color_lut[start] = ci.color();
                ++ci;
                ++start;
            }
        }

        // 填充查找表的后部分
        c = m_color_profile.last().color;
        for(; end < m_color_lut.size(); end++)
        {
            m_color_lut[end] = c;
        }
    }
}
}


这行代码表示一个 C/C++ 程序中的结束大括号，用于结束某个代码块或函数。


#endif


这行代码通常出现在 C/C++ 的预处理器指令中，用于条件编译。`#endif` 用于结束条件编译指令 `#if`, `#ifdef`, 或 `#ifndef` 的作用范围，确保只有在特定条件满足时才包含其中的代码。
```