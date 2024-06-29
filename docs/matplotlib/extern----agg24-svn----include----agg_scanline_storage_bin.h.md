# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_scanline_storage_bin.h`

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
// Adaptation for 32-bit screen coordinates has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------


#ifndef AGG_SCANLINE_STORAGE_BIN_INCLUDED
#define AGG_SCANLINE_STORAGE_BIN_INCLUDED

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "agg_array.h"


namespace agg
{

    //-----------------------------------------------scanline_storage_bin
    // 二进制扫描线存储类
    class scanline_storage_bin
    {
    private:
        // 用于存储跨度数据的向量
        pod_bvector<span_data, 10>    m_spans;
        // 用于存储扫描线数据的向量
        pod_bvector<scanline_data, 8> m_scanlines;
        // 虚假跨度对象，用于占位
        span_data     m_fake_span;
        // 虚假扫描线对象，用于占位
        scanline_data m_fake_scanline;
        // 最小 x 坐标
        int           m_min_x;
        // 最小 y 坐标
        int           m_min_y;
        // 最大 x 坐标
        int           m_max_x;
        // 最大 y 坐标
        int           m_max_y;
        // 当前扫描线索引
        unsigned      m_cur_scanline;
    };













    //---------------------------------------serialized_scanlines_adaptor_bin
    // 二进制序列化扫描线适配器类
    class serialized_scanlines_adaptor_bin
    {
        public:
            // 定义 cover_type 为布尔类型
            typedef bool cover_type;

            //--------------------------------------------------------------------
            // 嵌套类 embedded_scanline
            class embedded_scanline
            {
            public:

                //----------------------------------------------------------------
                // 嵌套类 const_iterator
                class const_iterator
                {
                public:
                    // 定义 span 结构体
                    struct span
                    {
                        int32 x;   // 起始位置 x
                        int32 len; // 长度 len
                    };

                    // 构造函数，初始化为默认值
                    const_iterator() : m_ptr(0) {}
                    // 构造函数，初始化迭代器，并读取数据
                    const_iterator(const embedded_scanline* sl) :
                        m_ptr(sl->m_ptr),
                        m_dx(sl->m_dx)
                    {
                        // 读取 x 偏移量并加上 m_dx
                        m_span.x   = read_int32() + m_dx;
                        // 读取 span 长度
                        m_span.len = read_int32();
                    }

                    // 返回当前 span 引用
                    const span& operator*()  const { return m_span;  }
                    // 返回当前 span 指针
                    const span* operator->() const { return &m_span; }

                    // 前进操作，读取下一个 span 的 x 和 len
                    void operator ++ ()
                    {
                        m_span.x   = read_int32() + m_dx;
                        m_span.len = read_int32();
                    }

                private:
                    // 读取 int32 类型的数据
                    int read_int32()
                    {
                        int32 val;
                        ((int8u*)&val)[0] = *m_ptr++;
                        ((int8u*)&val)[1] = *m_ptr++;
                        ((int8u*)&val)[2] = *m_ptr++;
                        ((int8u*)&val)[3] = *m_ptr++;
                        return val;
                    }

                    const int8u* m_ptr; // 指向数据的指针
                    span         m_span; // 当前 span
                    int          m_dx;   // x 方向的偏移量
                };

                friend class const_iterator;


                //----------------------------------------------------------------
                // embedded_scanline 构造函数，初始化成员变量
                embedded_scanline() : m_ptr(0), m_y(0), m_num_spans(0) {}

                //----------------------------------------------------------------
                // 重置函数，设置参数为给定值
                void     reset(int, int)     {}
                // 返回 span 的数量
                unsigned num_spans()   const { return m_num_spans;  }
                // 返回 y 方向的位置
                int      y()           const { return m_y;          }
                // 返回迭代器的起始位置
                const_iterator begin() const { return const_iterator(this); }


            private:
                //----------------------------------------------------------------
                // 内部函数，读取 int32 类型的数据
                int read_int32()
                {
                    int32 val;
                    ((int8u*)&val)[0] = *m_ptr++;
                    ((int8u*)&val)[1] = *m_ptr++;
                    ((int8u*)&val)[2] = *m_ptr++;
                    ((int8u*)&val)[3] = *m_ptr++;
                    return val;
                }

            public:
                //----------------------------------------------------------------
                // 初始化函数，设置指针和偏移量
                void init(const int8u* ptr, int dx, int dy)
                {
                    m_ptr       = ptr;
                    m_y         = read_int32() + dy; // 读取并设置 y 方向的位置
                    m_num_spans = unsigned(read_int32()); // 读取并设置 span 的数量
                    m_dx        = dx; // 设置 x 方向的偏移量
                }

            private:
                const int8u* m_ptr;   // 指向数据的指针
                int          m_y;     // y 方向的位置
                unsigned     m_num_spans; // span 的数量
                int          m_dx;    // x 方向的偏移量
            };
    // 公共默认构造函数，初始化所有成员变量为零或最大/最小可能的值
    public:
        //--------------------------------------------------------------------
        serialized_scanlines_adaptor_bin() :
            m_data(0),                           // 数据指针初始化为零
            m_end(0),                            // 结束指针初始化为零
            m_ptr(0),                            // 当前指针初始化为零
            m_dx(0),                             // 水平偏移量初始化为零
            m_dy(0),                             // 垂直偏移量初始化为零
            m_min_x(0x7FFFFFFF),                  // 最小 X 坐标初始化为最大可能整数
            m_min_y(0x7FFFFFFF),                  // 最小 Y 坐标初始化为最大可能整数
            m_max_x(-0x7FFFFFFF),                 // 最大 X 坐标初始化为最小可能整数
            m_max_y(-0x7FFFFFFF)                  // 最大 Y 坐标初始化为最小可能整数
        {}

        //--------------------------------------------------------------------
        // 带参数的构造函数，初始化成员变量使用给定的数据、大小、偏移量
        serialized_scanlines_adaptor_bin(const int8u* data, unsigned size,
                                         double dx, double dy) :
            m_data(data),                        // 数据指针初始化为给定的数据
            m_end(data + size),                  // 结束指针初始化为数据末尾
            m_ptr(data),                         // 当前指针初始化为数据起始处
            m_dx(iround(dx)),                    // 水平偏移量初始化为舍入后的给定 dx
            m_dy(iround(dy)),                    // 垂直偏移量初始化为舍入后的给定 dy
            m_min_x(0x7FFFFFFF),                  // 最小 X 坐标初始化为最大可能整数
            m_min_y(0x7FFFFFFF),                  // 最小 Y 坐标初始化为最大可能整数
            m_max_x(-0x7FFFFFFF),                 // 最大 X 坐标初始化为最小可能整数
            m_max_y(-0x7FFFFFFF)                  // 最大 Y 坐标初始化为最小可能整数
        {}

        //--------------------------------------------------------------------
        // 初始化函数，用给定的数据、大小、偏移量更新成员变量
        void init(const int8u* data, unsigned size, double dx, double dy)
        {
            m_data  = data;                      // 数据指针更新为给定的数据
            m_end   = data + size;               // 结束指针更新为数据末尾
            m_ptr   = data;                      // 当前指针更新为数据起始处
            m_dx    = iround(dx);                // 水平偏移量更新为舍入后的给定 dx
            m_dy    = iround(dy);                // 垂直偏移量更新为舍入后的给定 dy
            m_min_x = 0x7FFFFFFF;                 // 最小 X 坐标更新为最大可能整数
            m_min_y = 0x7FFFFFFF;                 // 最小 Y 坐标更新为最大可能整数
            m_max_x = -0x7FFFFFFF;                // 最大 X 坐标更新为最小可能整数
            m_max_y = -0x7FFFFFFF;                // 最大 Y 坐标更新为最小可能整数
        }

    private:
        //--------------------------------------------------------------------
        // 读取四字节整数并返回
        int read_int32()
        {
            int32 val;
            ((int8u*)&val)[0] = *m_ptr++;        // 将当前指针指向的字节存入 val 的第一个字节
            ((int8u*)&val)[1] = *m_ptr++;        // 将当前指针指向的字节存入 val 的第二个字节
            ((int8u*)&val)[2] = *m_ptr++;        // 将当前指针指向的字节存入 val 的第三个字节
            ((int8u*)&val)[3] = *m_ptr++;        // 将当前指针指向的字节存入 val 的第四个字节
            return val;                          // 返回读取到的整数值
        }
    public:
        // 迭代扫描线接口

        //--------------------------------------------------------------------
        // 将扫描线指针重置到起始位置
        bool rewind_scanlines()
        {
            m_ptr = m_data;  // 将指针指向数据的起始位置
            if(m_ptr < m_end)  // 如果指针未超出数据结尾
            {
                m_min_x = read_int32() + m_dx;  // 读取并计算最小 x 值
                m_min_y = read_int32() + m_dy;  // 读取并计算最小 y 值
                m_max_x = read_int32() + m_dx;  // 读取并计算最大 x 值
                m_max_y = read_int32() + m_dy;  // 读取并计算最大 y 值
            }
            return m_ptr < m_end;  // 返回是否还有更多扫描线可读取
        }

        //--------------------------------------------------------------------
        // 返回最小 x 值
        int min_x() const { return m_min_x; }

        // 返回最小 y 值
        int min_y() const { return m_min_y; }

        // 返回最大 x 值
        int max_x() const { return m_max_x; }

        // 返回最大 y 值
        int max_y() const { return m_max_y; }

        //--------------------------------------------------------------------
        // 模板函数，处理扫描线
        template<class Scanline> bool sweep_scanline(Scanline& sl)
        {
            sl.reset_spans();  // 重置扫描线的跨度

            for(;;)
            {
                if(m_ptr >= m_end) return false;  // 如果指针超出数据结尾，返回 false

                int y = read_int32() + m_dy;  // 读取并计算 y 坐标
                unsigned num_spans = read_int32();  // 读取跨度的数量

                do
                {
                    int x = read_int32() + m_dx;  // 读取并计算 x 坐标
                    int len = read_int32();  // 读取跨度的长度

                    if(len < 0) len = -len;  // 处理负长度情况
                    sl.add_span(x, unsigned(len), cover_full);  // 添加跨度到扫描线对象
                }
                while(--num_spans > 0);  // 处理所有跨度

                if(sl.num_spans())  // 如果扫描线有有效跨度
                {
                    sl.finalize(y);  // 完成当前扫描线的处理
                    break;
                }
            }
            return true;  // 返回处理成功
        }

        //--------------------------------------------------------------------
        // 特化函数，处理嵌入式扫描线
        bool sweep_scanline(embedded_scanline& sl)
        {
            do
            {
                if(m_ptr >= m_end) return false;  // 如果指针超出数据结尾，返回 false

                sl.init(m_ptr, m_dx, m_dy);  // 初始化嵌入式扫描线对象

                // 跳转到下一个扫描线
                //--------------------------
                read_int32();                    // 跳过 Y 坐标
                int num_spans = read_int32();    // 读取跨度数量
                m_ptr += num_spans * sizeof(int32) * 2;  // 跳过整个扫描线的数据块
            }
            while(sl.num_spans() == 0);  // 如果扫描线无有效跨度，继续处理下一扫描线

            return true;  // 返回处理成功
        }

    private:
        const int8u* m_data;
        const int8u* m_end;
        const int8u* m_ptr;
        int          m_dx;
        int          m_dy;
        int          m_min_x;
        int          m_min_y;
        int          m_max_x;
        int          m_max_y;
    };
}



#endif



// 这些代码片段是 C 或 C++ 中的预处理器指令，用于条件编译。
// `}` 闭合了一个代码块。
// `#endif` 结束了一个条件编译块，用于匹配前面的 `#ifdef` 或 `#ifndef`。
```