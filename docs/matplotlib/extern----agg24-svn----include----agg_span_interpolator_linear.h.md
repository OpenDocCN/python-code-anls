# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_interpolator_linear.h`

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

#ifndef AGG_SPAN_INTERPOLATOR_LINEAR_INCLUDED
#define AGG_SPAN_INTERPOLATOR_LINEAR_INCLUDED

#include "agg_basics.h"
#include "agg_dda_line.h"
#include "agg_trans_affine.h"

namespace agg
{

    //================================================span_interpolator_linear
    // 模板类定义：span_interpolator_linear，用于线性插值的跨度插值器
    // Transformer：默认为 trans_affine 类型的转换器
    // SubpixelShift：默认为 8，表示次像素级别的位移
    template<class Transformer = trans_affine, unsigned SubpixelShift = 8> 
    class span_interpolator_linear
    {
    public:
        // 定义一个类型别名，将 Transformer 命名为 trans_type
        typedef Transformer trans_type;

        // 枚举类型 subpixel_scale_e，包含 subpixel_shift 和 subpixel_scale 常量
        enum subpixel_scale_e
        {
            // subpixel_shift 值来自于模板参数 SubpixelShift
            subpixel_shift = SubpixelShift,
            // subpixel_scale 值为 subpixel_shift 左移的结果
            subpixel_scale  = 1 << subpixel_shift
        };

        //--------------------------------------------------------------------
        // 默认构造函数，无操作
        span_interpolator_linear() {}

        // 带参数的构造函数，初始化 m_trans 指向给定的 trans 对象
        span_interpolator_linear(trans_type& trans) : m_trans(&trans) {}

        // 带参数的构造函数，初始化 m_trans 指向给定的 trans 对象，并调用 begin 函数初始化
        span_interpolator_linear(trans_type& trans,
                                 double x, double y, unsigned len) :
            m_trans(&trans)
        {
            begin(x, y, len);
        }

        //----------------------------------------------------------------
        // 返回当前的 trans_type 引用
        const trans_type& transformer() const { return *m_trans; }

        // 设置新的 trans_type 对象
        void transformer(trans_type& trans) { m_trans = &trans; }

        //----------------------------------------------------------------
        // 初始化函数，根据给定的 x, y 和 len 计算起始点和结束点的插值
        void begin(double x, double y, unsigned len)
        {
            double tx;
            double ty;

            tx = x;
            ty = y;
            m_trans->transform(&tx, &ty);  // 使用 trans 对象进行坐标变换
            int x1 = iround(tx * subpixel_scale);  // 计算起始点 x 坐标
            int y1 = iround(ty * subpixel_scale);  // 计算起始点 y 坐标

            tx = x + len;
            ty = y;
            m_trans->transform(&tx, &ty);  // 使用 trans 对象进行坐标变换
            int x2 = iround(tx * subpixel_scale);  // 计算结束点 x 坐标
            int y2 = iround(ty * subpixel_scale);  // 计算结束点 y 坐标

            // 使用 DDA2 线性插值器初始化 m_li_x 和 m_li_y
            m_li_x = dda2_line_interpolator(x1, x2, len);
            m_li_y = dda2_line_interpolator(y1, y2, len);
        }

        //----------------------------------------------------------------
        // 重新同步函数，根据给定的 xe, ye 和 len 进行坐标变换和插值器更新
        void resynchronize(double xe, double ye, unsigned len)
        {
            m_trans->transform(&xe, &ye);  // 使用 trans 对象进行坐标变换
            // 更新 m_li_x 和 m_li_y 的值
            m_li_x = dda2_line_interpolator(m_li_x.y(), iround(xe * subpixel_scale), len);
            m_li_y = dda2_line_interpolator(m_li_y.y(), iround(ye * subpixel_scale), len);
        }
    
        //----------------------------------------------------------------
        // 前缀递增运算符重载，递增 m_li_x 和 m_li_y
        void operator++()
        {
            ++m_li_x;
            ++m_li_y;
        }

        //----------------------------------------------------------------
        // 获取当前坐标函数，将当前的 m_li_x 和 m_li_y 的值通过指针返回
        void coordinates(int* x, int* y) const
        {
            *x = m_li_x.y();
            *y = m_li_y.y();
        }

    private:
        trans_type* m_trans;  // 指向 trans_type 对象的指针
        dda2_line_interpolator m_li_x;  // DDA2 线性插值器，用于 x 坐标
        dda2_line_interpolator m_li_y;  // DDA2 线性插值器，用于 y 坐标
    };






    //=====================================span_interpolator_linear_subdiv
    // span_interpolator_linear_subdiv 类模板的定义
    template<class Transformer = trans_affine, unsigned SubpixelShift = 8> 
    class span_interpolator_linear_subdiv
    {
    private:
        unsigned m_subdiv_shift;  // 子像素移位量
        unsigned m_subdiv_size;  // 子像素大小
        unsigned m_subdiv_mask;  // 子像素掩码
        trans_type* m_trans;  // 指向 trans_type 对象的指针
        dda2_line_interpolator m_li_x;  // DDA2 线性插值器，用于 x 坐标
        dda2_line_interpolator m_li_y;  // DDA2 线性插值器，用于 y 坐标
        int      m_src_x;  // 源 x 坐标
        double   m_src_y;  // 源 y 坐标
        unsigned m_pos;  // 当前位置
        unsigned m_len;  // 长度
    };
}

# 这里是 C/C++ 中的预处理器指令，用于结束一个条件编译区段，对应于 #ifdef 或 #ifndef 的开始部分
#endif
```