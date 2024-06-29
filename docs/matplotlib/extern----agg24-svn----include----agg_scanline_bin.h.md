# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_scanline_bin.h`

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
// Class scanline_bin - binary scanline.
//
//----------------------------------------------------------------------------
//
// Adaptation for 32-bit screen coordinates (scanline32_bin) has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------

#ifndef AGG_SCANLINE_BIN_INCLUDED
#define AGG_SCANLINE_BIN_INCLUDED

#include "agg_array.h"  // 包含 agg_array 头文件

namespace agg
{

    //=============================================================scanline_bin
    // 
    // This is binary scaline container which supports the interface 
    // used in the rasterizer::render(). See description of agg_scanline_u8 
    // for details.
    // 
    //------------------------------------------------------------------------
    class scanline_bin
    {
    //===========================================================scanline32_bin
    // scanline32_bin 类定义

    class scanline32_bin
    {
    public:
        // 定义坐标类型为 int32
        typedef int32 coord_type;

        // 定义 span 结构体，包含 x 坐标和长度 len
        struct span
        {
            int16 x;    // x 坐标
            int16 len;  // 长度
        };

        // 定义 const_iterator 类型为 const span*，用于迭代 span 结构体
        typedef const span* const_iterator;

        //--------------------------------------------------------------------
        // 构造函数，初始化 m_last_x 为 0x7FFFFFF0，初始化 m_spans 和 m_cur_span
        scanline32_bin() :
            m_last_x(0x7FFFFFF0),
            m_spans(),
            m_cur_span(0)
        {
        }

        //--------------------------------------------------------------------
        // 重置函数，根据给定的 min_x 和 max_x 初始化 spans 的大小，并重置 m_last_x 和 m_cur_span
        void reset(int min_x, int max_x)
        {
            unsigned max_len = max_x - min_x + 3;
            if(max_len > m_spans.size())
            {
                m_spans.resize(max_len);  // 调整 m_spans 的大小
            }
            m_last_x   = 0x7FFFFFF0;    // 重置 m_last_x
            m_cur_span = &m_spans[0];   // 设置 m_cur_span 指向 m_spans 的首元素
        }

        //--------------------------------------------------------------------
        // 添加单个 cell 的函数，根据当前的 m_last_x 和 x 判断是否合并到当前 span 或者新建一个 span
        void add_cell(int x, unsigned)
        {
            if(x == m_last_x+1)
            {
                m_cur_span->len++;      // 合并到当前 span
            }
            else
            {
                ++m_cur_span;           // 移动到下一个 span
                m_cur_span->x = (int16)x;   // 设置新 span 的起始 x 坐标
                m_cur_span->len = 1;        // 设置新 span 的长度为 1
            }
            m_last_x = x;               // 更新 m_last_x
        }

        //--------------------------------------------------------------------
        // 添加跨越多个 cell 的 span 的函数，根据当前的 m_last_x 和 x 判断是否合并到当前 span 或者新建一个 span
        void add_span(int x, unsigned len, unsigned)
        {
            if(x == m_last_x+1)
            {
                m_cur_span->len = (int16)(m_cur_span->len + len);  // 合并到当前 span
            }
            else
            {
                ++m_cur_span;           // 移动到下一个 span
                m_cur_span->x = (int16)x;   // 设置新 span 的起始 x 坐标
                m_cur_span->len = (int16)len;   // 设置新 span 的长度
            }
            m_last_x = x + len - 1;     // 更新 m_last_x
        }

        //--------------------------------------------------------------------
        // 添加多个 cell 的函数，直接调用 add_span，不使用传入的额外参数
        void add_cells(int x, unsigned len, const void*)
        {
            add_span(x, len, 0);    // 调用 add_span
        }

        //--------------------------------------------------------------------
        // 完成当前扫描线的处理，设置当前扫描线的 y 坐标为 y
        void finalize(int y) 
        { 
            m_y = y; 
        }

        //--------------------------------------------------------------------
        // 重置 spans 的状态，设置 m_last_x 和 m_cur_span
        void reset_spans()
        {
            m_last_x    = 0x7FFFFFF0;    // 重置 m_last_x
            m_cur_span  = &m_spans[0];   // 设置 m_cur_span 指向 m_spans 的首元素
        }

        //--------------------------------------------------------------------
        // 返回当前扫描线的 y 坐标
        int            y()         const { return m_y; }

        // 返回当前扫描线包含的 span 的数量
        unsigned       num_spans() const { return unsigned(m_cur_span - &m_spans[0]); }

        // 返回指向第一个 span 的迭代器
        const_iterator begin()     const { return &m_spans[1]; }

    private:
        // 禁止复制构造函数和赋值运算符
        scanline32_bin(const scanline32_bin&);
        const scanline32_bin operator = (const scanline32_bin&);

        int             m_last_x;   // 上一个处理的 x 坐标
        int             m_y;        // 当前扫描线的 y 坐标
        pod_array<span> m_spans;    // 存储 span 的数组
        span*           m_cur_span; // 当前操作的 span
    };
    public:
        typedef int32 coord_type;  # 定义坐标类型为32位整数

        //--------------------------------------------------------------------
        struct span
        {
            span() {}  # 默认构造函数
            span(coord_type x_, coord_type len_) : x(x_), len(len_) {}  # 带参构造函数，初始化x和len

            coord_type x;  # 范围起始位置
            coord_type len;  # 范围长度
        };
        typedef pod_bvector<span, 4> span_array_type;  # 使用pod_bvector定义存储span结构的数组类型

        //--------------------------------------------------------------------
        class const_iterator
        {
        public:
            const_iterator(const span_array_type& spans) :  # 构造函数，接受span数组类型的引用
                m_spans(spans),  # 初始化成员变量m_spans为参数spans的引用
                m_span_idx(0)  # 初始化迭代器的当前索引为0
            {}

            const span& operator*()  const { return m_spans[m_span_idx];  }  # 解引用操作符，返回当前索引位置的span对象的引用
            const span* operator->() const { return &m_spans[m_span_idx]; }  # 成员访问操作符，返回指向当前索引位置span对象的指针

            void operator ++ () { ++m_span_idx; }  # 前置递增操作符，使迭代器向前移动一个位置

        private:
            const span_array_type& m_spans;  # 引用类型成员变量，存储span数组
            unsigned               m_span_idx;  # 当前迭代器索引
        };

        //--------------------------------------------------------------------
        scanline32_bin() : m_max_len(0), m_last_x(0x7FFFFFF0) {}  # 默认构造函数，初始化m_max_len为0，m_last_x为0x7FFFFFF0

        //--------------------------------------------------------------------
        void reset(int min_x, int max_x)
        {
            m_last_x = 0x7FFFFFF0;  # 重置m_last_x为0x7FFFFFF0，表示最后处理的x坐标
            m_spans.remove_all();  # 清空span数组
        }

        //--------------------------------------------------------------------
        void add_cell(int x, unsigned)
        {
            if(x == m_last_x+1)  # 如果当前x与上一个处理的x相邻
            {
                m_spans.last().len++;  # 在最后一个span的基础上增加长度
            }
            else
            {
                m_spans.add(span(coord_type(x), 1));  # 否则添加一个新的span，长度为1
            }
            m_last_x = x;  # 更新最后处理的x坐标
        }

        //--------------------------------------------------------------------
        void add_span(int x, unsigned len, unsigned)
        {
            if(x == m_last_x+1)  # 如果当前x与上一个处理的x相邻
            {
                m_spans.last().len += coord_type(len);  # 在最后一个span的基础上增加长度
            }
            else
            {
                m_spans.add(span(coord_type(x), coord_type(len)));  # 否则添加一个新的span，长度为len
            }
            m_last_x = x + len - 1;  # 更新最后处理的x坐标
        }

        //--------------------------------------------------------------------
        void add_cells(int x, unsigned len, const void*)
        {
            add_span(x, len, 0);  # 添加一段连续的span，长度为len
        }

        //--------------------------------------------------------------------
        void finalize(int y) 
        { 
            m_y = y;  # 最终处理的y坐标
        }

        //--------------------------------------------------------------------
        void reset_spans()
        {
            m_last_x = 0x7FFFFFF0;  # 重置m_last_x为0x7FFFFFF0，表示最后处理的x坐标
            m_spans.remove_all();  # 清空span数组
        }

        //--------------------------------------------------------------------
        int            y()         const { return m_y; }  # 返回y坐标
        unsigned       num_spans() const { return m_spans.size(); }  # 返回span数组的大小
        const_iterator begin()     const { return const_iterator(m_spans); }  # 返回span数组的常量迭代器
    # 禁止复制构造函数的私有声明，防止对象通过复制构造函数进行复制
    private:
        scanline32_bin(const scanline32_bin&);
        
        # 禁止赋值运算符的私有声明，防止对象通过赋值运算符进行赋值
        const scanline32_bin operator = (const scanline32_bin&);

        # 最大长度限制成员变量，用于存储扫描线的最大长度
        unsigned        m_max_len;
        
        # 上一次处理的 x 坐标成员变量，用于跟踪上一个处理的像素位置
        int             m_last_x;
        
        # 当前扫描线的 y 坐标成员变量，表示当前处理的扫描线的位置
        int             m_y;
        
        # 存储扫描线跨度的容器类型，可能是一个数组或者其他数据结构
        span_array_type m_spans;
    };
}
# 结束条件：结束预处理指令的定义，匹配 #ifdef 或 #ifndef 开始的条件编译指令的结束位置
#endif
# 结束条件：结束条件编译指令块，表示条件编译的结束位置
```