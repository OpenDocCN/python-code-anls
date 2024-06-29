# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_interpolator_trans.h`

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
// Horizontal span interpolator for use with an arbitrary transformer
// The efficiency highly depends on the operations done in the transformer
//
//----------------------------------------------------------------------------

#ifndef AGG_SPAN_INTERPOLATOR_TRANS_INCLUDED
#define AGG_SPAN_INTERPOLATOR_TRANS_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //=================================================span_interpolator_trans
    // 水平跨度插值器，用于与任意变换器一起使用
    template<class Transformer, unsigned SubpixelShift = 8> 
    class span_interpolator_trans
    {
    public:
        typedef Transformer trans_type;
        enum subpixel_scale_e
        {
            subpixel_shift = SubpixelShift,
            subpixel_scale = 1 << subpixel_shift
        };

        //--------------------------------------------------------------------
        // 构造函数，默认构造
        span_interpolator_trans() {}

        // 构造函数，接受一个变换器作为参数
        span_interpolator_trans(trans_type& trans) : m_trans(&trans) {}

        // 构造函数，接受变换器和初始坐标作为参数
        span_interpolator_trans(trans_type& trans,
                                double x, double y, unsigned) :
            m_trans(&trans)
        {
            begin(x, y, 0);
        }

        //----------------------------------------------------------------
        // 获取当前使用的变换器
        const trans_type& transformer() const { return *m_trans; }

        // 设置新的变换器
        void transformer(const trans_type& trans) { m_trans = &trans; }

        //----------------------------------------------------------------
        // 设置起始坐标并进行变换，计算子像素精度的坐标
        void begin(double x, double y, unsigned)
        {
            m_x = x;
            m_y = y;
            m_trans->transform(&x, &y);
            m_ix = iround(x * subpixel_scale);
            m_iy = iround(y * subpixel_scale);
        }

        //----------------------------------------------------------------
        // 水平移动操作符重载，增加 x 坐标并重新计算变换后的坐标
        void operator++()
        {
            m_x += 1.0;
            double x = m_x;
            double y = m_y;
            m_trans->transform(&x, &y);
            m_ix = iround(x * subpixel_scale);
            m_iy = iround(y * subpixel_scale);
        }

        //----------------------------------------------------------------
        // 获取当前坐标（以整数表示）
        void coordinates(int* x, int* y) const
        {
            *x = m_ix;
            *y = m_iy;
        }
    private:
        // 指向 trans_type 类型的指针，用于存储某种类型的转换数据
        trans_type*       m_trans;
        // 存储一个 double 类型的值，表示 x 坐标
        double            m_x;
        // 存储一个 double 类型的值，表示 y 坐标
        double            m_y;
        // 存储一个 int 类型的值，表示 x 坐标的整数部分
        int               m_ix;
        // 存储一个 int 类型的值，表示 y 坐标的整数部分
        int               m_iy;
    };
}


注释：


// 结束一个条件编译指令的标记块



#endif


注释：


// 条件编译指令的结束，用于结束一个条件编译段落
```