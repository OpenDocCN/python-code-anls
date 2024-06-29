# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_scanline_boolean_algebra.h`

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

#ifndef AGG_SCANLINE_BOOLEAN_ALGEBRA_INCLUDED
#define AGG_SCANLINE_BOOLEAN_ALGEBRA_INCLUDED

#include <stdlib.h>
#include <math.h>
#include "agg_basics.h"

// 命名空间 agg 开始
namespace agg
{

    //-----------------------------------------------sbool_combine_spans_bin
    // Functor.
    // Combine two binary encoded spans, i.e., when we don't have any
    // anti-aliasing information, but only X and Length. The function
    // is compatible with any type of scanlines.
    //----------------
    template<class Scanline1, 
             class Scanline2, 
             class Scanline> 
    struct sbool_combine_spans_bin
    {
        // 将两个二进制编码的跨度合并，适用于没有抗锯齿信息的情况
        void operator () (const typename Scanline1::const_iterator&, 
                          const typename Scanline2::const_iterator&, 
                          int x, unsigned len, 
                          Scanline& sl) const
        {
            // 向扫描线对象添加一个跨度，跨度覆盖整个区域
            sl.add_span(x, len, cover_full);
        }
    };



    //---------------------------------------------sbool_combine_spans_empty
    // Functor.
    // Combine two spans as empty ones. The functor does nothing
    // and is used to XOR binary spans.
    //----------------
    template<class Scanline1, 
             class Scanline2, 
             class Scanline> 
    struct sbool_combine_spans_empty
    {
        // 将两个空跨度合并，这个函数对象不执行任何操作，用于异或二进制跨度
        void operator () (const typename Scanline1::const_iterator&, 
                          const typename Scanline2::const_iterator&, 
                          int, unsigned, 
                          Scanline&) const
        {}
    };



    //--------------------------------------------------sbool_add_span_empty
    // Functor.
    // Add nothing. Used in conbine_shapes_sub
    //----------------
    template<class Scanline1, 
             class Scanline> 
    struct sbool_add_span_empty
    {
        // 不添加任何内容，用于 conbine_shapes_sub
        void operator () (const typename Scanline1::const_iterator&, 
                          int, unsigned, 
                          Scanline&) const
        {}
    };


    //----------------------------------------------------sbool_add_span_bin
    // Functor.
    // Add a binary span
    //----------------
    template<class Scanline1, 
             class Scanline> 
    struct sbool_add_span_bin


注释结束。
    // Functor 结构体，用于在扫描线对象中添加一个不带抗锯齿信息的跨度
    {
        void operator () (const typename Scanline1::const_iterator&, 
                          int x, unsigned len, 
                          Scanline& sl) const
        {
            // 调用扫描线对象的 add_span 方法，添加一个全覆盖的跨度
            sl.add_span(x, len, cover_full);
        }
    };

    


    //-----------------------------------------------------sbool_add_span_aa
    // Functor 结构体。
    // 添加一个带有抗锯齿效果的跨度
    // 仅使用 X 和 Length，忽略抗锯齿信息。此函数兼容任何类型的扫描线。
    //----------------
    template<class Scanline1, 
             class Scanline> 
    struct sbool_add_span_aa
    {
        void operator () (const typename Scanline1::const_iterator& span, 
                          int x, unsigned len, 
                          Scanline& sl) const
        {
            if(span->len < 0)
            {
                // 如果跨度长度小于零，则添加一个使用跨度覆盖信息的跨度
                sl.add_span(x, len, *span->covers);
            }
            else
            if(span->len > 0)
            {
                // 否则，根据跨度的位置和长度，添加一组单元格及其覆盖信息
                const typename Scanline1::cover_type* covers = span->covers;
                if(span->x < x) covers += x - span->x;
                sl.add_cells(x, len, covers);
            }
        }
    };




    //----------------------------------------------sbool_intersect_spans_aa
    // Functor 结构体。
    // 交集两个跨度，并保留抗锯齿信息。
    // 将结果添加到 "sl" 扫描线对象中。
    //------------------
    template<class Scanline1, 
             class Scanline2, 
             class Scanline, 
             unsigned CoverShift = cover_shift> 
    struct sbool_intersect_spans_aa
    };






    //--------------------------------------------------sbool_unite_spans_aa
    // Functor 结构体。
    // 合并两个跨度，并保留抗锯齿信息。
    // 将结果添加到 "sl" 扫描线对象中。
    //------------------
    template<class Scanline1, 
             class Scanline2, 
             class Scanline, 
             unsigned CoverShift = cover_shift> 
    struct sbool_unite_spans_aa
    };


    //---------------------------------------------sbool_xor_formula_linear
    // Functor 结构体。
    // 线性异或运算的计算公式。
    //------------------
    template<unsigned CoverShift = cover_shift> 
    struct sbool_xor_formula_linear
    {
        enum cover_scale_e
        {
            // 定义覆盖信息的位移量、大小和掩码
            cover_shift = CoverShift,
            cover_size  = 1 << cover_shift,
            cover_mask  = cover_size - 1
        };

        // 计算线性异或的结果
        static AGG_INLINE unsigned calculate(unsigned a, unsigned b)
        {
            unsigned cover = a + b;
            if(cover > cover_mask) cover = cover_mask + cover_mask - cover;
            return cover;
        }
    };


    //---------------------------------------------sbool_xor_formula_saddle
    // Functor 结构体。
    // 鞍点异或运算的计算公式。
    //------------------
    template<unsigned CoverShift = cover_shift> 
    struct sbool_xor_formula_saddle
    {
        // 定义枚举 `cover_scale_e`，包含覆盖相关的常量和掩码
        enum cover_scale_e
        {
            cover_shift = CoverShift,       // 覆盖位移，通过参数 `CoverShift` 设置
            cover_size  = 1 << cover_shift, // 覆盖大小，通过位移运算得出
            cover_mask  = cover_size - 1    // 覆盖掩码，通过覆盖大小计算得出
        };
    
        // 定义静态内联函数 `calculate`，计算两个无符号整数的特定值
        static AGG_INLINE unsigned calculate(unsigned a, unsigned b)
        {
            unsigned k = a * b;
            // 如果计算结果等于覆盖掩码的平方，则返回零
            if (k == cover_mask * cover_mask) return 0;
    
            // 使用特定算法计算新的值 `a` 和 `b`
            a = (cover_mask * cover_mask - (a << cover_shift) + k) >> cover_shift;
            b = (cover_mask * cover_mask - (b << cover_shift) + k) >> cover_shift;
            // 返回特定计算结果
            return cover_mask - ((a * b) >> cover_shift);
        }
    };
    
    
    //-------------------------------------------sbool_xor_formula_abs_diff
    // 结构体 `sbool_xor_formula_abs_diff`
    struct sbool_xor_formula_abs_diff
    {
        // 静态内联函数 `calculate`，计算两个无符号整数的绝对值差
        static AGG_INLINE unsigned calculate(unsigned a, unsigned b)
        {
            // 返回两个整数转换为有符号整数后的绝对值
            return unsigned(abs(int(a) - int(b)));
        }
    };
    
    
    
    //----------------------------------------------------sbool_xor_spans_aa
    // 结构体 `sbool_xor_spans_aa`
    // 函数对象。
    // 对两个区间进行异或操作，保留抗锯齿信息。
    // 结果添加到扫描线 `sl` 中。
    //------------------
    template<class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class XorFormula,
             unsigned CoverShift = cover_shift> 
    struct sbool_xor_spans_aa
    {
        // 枚举类型定义，定义了覆盖度缩放的相关常量
        enum cover_scale_e
        {
            cover_shift = CoverShift,    // 覆盖度移位量，用于计算覆盖度大小
            cover_size  = 1 << cover_shift,  // 覆盖度大小，2 的 cover_shift 次方
            cover_mask  = cover_size - 1,    // 覆盖度掩码，用于限制覆盖度大小
            cover_full  = cover_mask         // 完整的覆盖度，与掩码相同
        };
    
        // 函数调用运算符重载，用于将两个扫描线合并，并将结果添加到给定的扫描线中
        void operator () (const typename Scanline1::const_iterator& span1, 
                          const typename Scanline2::const_iterator& span2, 
                          int x, unsigned len, 
                          Scanline& sl) const
        {
            unsigned cover;  // 用于存储覆盖度信息的变量
            const typename Scanline1::cover_type* covers1;  // 指向第一个扫描线覆盖度数据的指针
            const typename Scanline2::cover_type* covers2;  // 指向第二个扫描线覆盖度数据的指针
    
            // 根据扫描线的类型计算操作码，并选择合适的组合算法
            // 0 = 两个扫描线都是 AA 类型
            // 1 = span1 是实心的，span2 是 AA 类型
            // 2 = span1 是 AA 类型，span2 是实心的
            // 3 = 两个扫描线都是实心的
            switch((span1->len < 0) | ((span2->len < 0) << 1))
            {
            case 0:      // 两个都是 AA 类型的扫描线
                covers1 = span1->covers;  // 获取 span1 的覆盖度数据指针
                covers2 = span2->covers;  // 获取 span2 的覆盖度数据指针
                if(span1->x < x) covers1 += x - span1->x;  // 调整 covers1 指针位置
                if(span2->x < x) covers2 += x - span2->x;  // 调整 covers2 指针位置
                do
                {
                    cover = XorFormula::calculate(*covers1++, *covers2++);  // 计算覆盖度值
                    if(cover) sl.add_cell(x, cover);  // 如果覆盖度非零，将单元格添加到扫描线中
                    ++x;  // 移动 x 坐标
                }
                while(--len);  // 递减长度直到为零
                break;
    
            case 1:      // span1 是实心的，span2 是 AA 类型的扫描线
                covers2 = span2->covers;  // 获取 span2 的覆盖度数据指针
                if(span2->x < x) covers2 += x - span2->x;  // 调整 covers2 指针位置
                do
                {
                    cover = XorFormula::calculate(*(span1->covers), *covers2++);  // 计算覆盖度值
                    if(cover) sl.add_cell(x, cover);  // 如果覆盖度非零，将单元格添加到扫描线中
                    ++x;  // 移动 x 坐标
                }
                while(--len);  // 递减长度直到为零
                break;
    
            case 2:      // span1 是 AA 类型的扫描线，span2 是实心的
                covers1 = span1->covers;  // 获取 span1 的覆盖度数据指针
                if(span1->x < x) covers1 += x - span1->x;  // 调整 covers1 指针位置
                do
                {
                    cover = XorFormula::calculate(*covers1++, *(span2->covers));  // 计算覆盖度值
                    if(cover) sl.add_cell(x, cover);  // 如果覆盖度非零，将单元格添加到扫描线中
                    ++x;  // 移动 x 坐标
                }
                while(--len);  // 递减长度直到为零
                break;
    
            case 3:      // 两个都是实心的扫描线
                cover = XorFormula::calculate(*(span1->covers), *(span2->covers));  // 计算覆盖度值
                if(cover) sl.add_span(x, len, cover);  // 如果覆盖度非零，将跨度添加到扫描线中
                break;
    
            }
        }
    };
    //--------------------------------------------sbool_add_spans_and_render
    // 定义了一个模板函数 sbool_add_spans_and_render，用于将扫描线 sl1 的内容添加到 sl 中并进行渲染
    template<class Scanline1, 
             class Scanline, 
             class Renderer, 
             class AddSpanFunctor>
    void sbool_add_spans_and_render(const Scanline1& sl1, 
                                    Scanline& sl, 
                                    Renderer& ren, 
                                    AddSpanFunctor add_span)
    {
        // 重置扫描线 sl 的跨度
        sl.reset_spans();
        // 初始化扫描线 sl1 的迭代器 span
        typename Scanline1::const_iterator span = sl1.begin();
        // 获取扫描线 sl1 中的跨度数目
        unsigned num_spans = sl1.num_spans();
        // 循环处理扫描线 sl1 中的每一个跨度
        for(;;)
        {
            // 将当前跨度 span 添加到扫描线 sl 中，使用 add_span 函数对象
            add_span(span, span->x, abs((int)span->len), sl);
            // 减少剩余要处理的跨度数目
            if(--num_spans == 0) break;
            // 移动到下一个跨度
            ++span;
        }
        // 完成扫描线 sl 的最终化，设置其 y 坐标
        sl.finalize(sl1.y());
        // 调用渲染器 ren 对最终化的扫描线 sl 进行渲染
        ren.render(sl);
    }


    //---------------------------------------------sbool_intersect_scanlines
    // 对两个扫描线 sl1 和 sl2 进行求交，并生成新的扫描线 sl
    // combine_spans 函数对象可以是 sbool_combine_spans_bin 或 sbool_intersect_spans_aa 的类型
    // 前者是一般的用于合并两个跨度的函数对象，不包含反走样功能；后者保留了反走样信息，但处理速度较慢
    template<class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class CombineSpansFunctor>
    void sbool_intersect_scanlines(const Scanline1& sl1, 
                                   const Scanline2& sl2, 
                                   Scanline& sl, 
                                   CombineSpansFunctor combine_spans)
    {
        // 重置扫描线容器的状态，准备接收新的扫描线数据
        sl.reset_spans();
    
        // 获取第一个扫描线生成器中的扫描线数量
        unsigned num1 = sl1.num_spans();
        // 如果第一个扫描线生成器中没有扫描线，则直接返回
        if(num1 == 0) return;
    
        // 获取第二个扫描线生成器中的扫描线数量
        unsigned num2 = sl2.num_spans();
        // 如果第二个扫描线生成器中没有扫描线，则直接返回
        if(num2 == 0) return;
    
        // 定义第一个扫描线生成器的常量迭代器
        typename Scanline1::const_iterator span1 = sl1.begin();
        // 定义第二个扫描线生成器的常量迭代器
        typename Scanline2::const_iterator span2 = sl2.begin();
    
        // 当两个生成器中都还有扫描线时执行以下循环
        while(num1 && num2)
        {
            // 获取第一个扫描线的起始和结束坐标
            int xb1 = span1->x;
            int xe1 = xb1 + abs((int)span1->len) - 1;
    
            // 获取第二个扫描线的起始和结束坐标
            int xb2 = span2->x;
            int xe2 = xb2 + abs((int)span2->len) - 1;
    
            // 确定下一步应该推进哪些扫描线段
            // 选择结束坐标较小的扫描线段进行推进
            bool advance_span1 = xe1 <  xe2;
            // 当两个扫描线段的结束坐标相同时，可以同时推进两个扫描线段
            bool advance_both  = xe1 == xe2;
    
            // 计算两个扫描线段的交集，并检查它们是否相交
            if(xb1 < xb2) xb1 = xb2;
            if(xe1 > xe2) xe1 = xe2;
            // 如果两个扫描线段有交集，则合并它们
            if(xb1 <= xe1)
            {
                combine_spans(span1, span2, xb1, xe1 - xb1 + 1, sl);
            }
    
            // 推进扫描线段的位置
            if(advance_both)
            {
                // 同时推进两个扫描线段
                --num1;
                --num2;
                if(num1) ++span1;
                if(num2) ++span2;
            }
            else
            {
                // 根据结束坐标较小的扫描线段推进对应的生成器
                if(advance_span1)
                {
                    --num1;
                    if(num1) ++span1;
                }
                else
                {
                    --num2;
                    if(num2) ++span2;
                }
            }
        }
    }
    
    //------------------------------------------------sbool_intersect_shapes
    // 对扫描线形状进行交集操作。这里使用了"扫描线生成器"抽象概念。
    // ScanlineGen1 和 ScanlineGen2 可能是 rasterizer_scanline_aa<> 类型的生成器。
    // 函数需要三个扫描线容器，它们可以是不同的类型。
    // "sl1" 和 "sl2" 用于从生成器中获取扫描线，"sl" 用作结果扫描线以进行渲染。
    // 外部的 "sl1" 和 "sl2" 仅用于优化和重用扫描线对象。
    // 函数调用 sbool_intersect_scanlines，并将 CombineSpansFunctor 作为最后一个参数传入。详细信息请参见 sbool_intersect_scanlines。
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer,
             class CombineSpansFunctor>
    void sbool_intersect_shapes(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                Scanline1& sl1, Scanline2& sl2,
                                Scanline& sl, Renderer& ren, 
                                CombineSpansFunctor combine_spans)
    {
        // 准备扫描线生成器。
        // 如果任何一个生成器不包含扫描线，则返回。
        //-----------------
        if(!sg1.rewind_scanlines()) return;
        if(!sg2.rewind_scanlines()) return;
    
        // 获取边界框
        //----------------
        rect_i r1(sg1.min_x(), sg1.min_y(), sg1.max_x(), sg1.max_y());
        rect_i r2(sg2.min_x(), sg2.min_y(), sg2.max_x(), sg2.max_y());
    
        // 计算边界框的交集，如果它们不相交，则返回。
        //-----------------
        rect_i ir = intersect_rectangles(r1, r2);
        if(!ir.is_valid()) return;
    
        // 重置扫描线并获取两个第一个扫描线
        //-----------------
        sl.reset(ir.x1, ir.x2);
        sl1.reset(sg1.min_x(), sg1.max_x());
        sl2.reset(sg2.min_x(), sg2.max_x());
        if(!sg1.sweep_scanline(sl1)) return;
        if(!sg2.sweep_scanline(sl2)) return;
    
        ren.prepare();
    
        // 主循环
        // 在这里，我们将具有相同 Y 坐标的扫描线同步，忽略所有其他扫描线。
        // 只有具有相同 Y 坐标的扫描线才会被合并。
        //-----------------
        for(;;)
        {
            while(sl1.y() < sl2.y())
            {
                if(!sg1.sweep_scanline(sl1)) return;
            }
            while(sl2.y() < sl1.y())
            {
                if(!sg2.sweep_scanline(sl2)) return;
            }
    
            if(sl1.y() == sl2.y())
            {
                // Y 坐标相同。
                // 合并扫描线，如果包含任何跨度则进行渲染，并将两个生成器推进到下一个扫描线。
                //----------------------
                sbool_intersect_scanlines(sl1, sl2, sl, combine_spans);
                if(sl.num_spans())
                {
                    sl.finalize(sl1.y());
                    ren.render(sl);
                }
                if(!sg1.sweep_scanline(sl1)) return;
                if(!sg2.sweep_scanline(sl2)) return;
            }
        }
    }
    //----------------------------------------------------sbool_unite_shapes
    // 合并扫描线形状。这里使用了“扫描线生成器”抽象化。
    // ScanlineGen1 和 ScanlineGen2 是生成器，可以是 rasterizer_scanline_aa<> 类型。
    // 函数需要三个扫描线容器，可以是不同类型。
    // "sl1" 和 "sl2" 用于从生成器中检索扫描线，
    // "sl" 用作生成的扫描线以进行渲染。
    // 外部的 "sl1" 和 "sl2" 仅用于优化和重用扫描线对象。
    // 函数调用 sbool_unite_scanlines，最后一个参数为 CombineSpansFunctor。详见 sbool_unite_scanlines。
    //----------
    
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer,
             class AddSpanFunctor1,
             class AddSpanFunctor2,
             class CombineSpansFunctor>
    void sbool_unite_shapes(ScanlineGen1& sg1, ScanlineGen2& sg2,
                            Scanline1& sl1, Scanline2& sl2,
                            Scanline& sl, Renderer& ren, 
                            AddSpanFunctor1 add_span1,
                            AddSpanFunctor2 add_span2,
                            CombineSpansFunctor combine_spans)
    }
    
    
    //-------------------------------------------------sbool_subtract_shapes
    // 减去扫描线形状 "sg1-sg2"。这里使用了“扫描线生成器”抽象化。
    // ScanlineGen1 和 ScanlineGen2 是生成器，可以是 rasterizer_scanline_aa<> 类型。
    // 函数需要三个扫描线容器，可以是不同类型。
    // "sl1" 和 "sl2" 用于从生成器中检索扫描线，
    // "sl" 用作生成的扫描线以进行渲染。
    // 外部的 "sl1" 和 "sl2" 仅用于优化和重用扫描线对象。
    // 函数调用 sbool_intersect_scanlines，最后一个参数为 CombineSpansFunctor。详见 combine_scanlines_sub。
    //----------
    // 定义函数模板 sbool_subtract_shapes，用于计算两个形状的布尔减法
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer,
             class AddSpanFunctor1,
             class CombineSpansFunctor>
    void sbool_subtract_shapes(ScanlineGen1& sg1, ScanlineGen2& sg2,
                               Scanline1& sl1, Scanline2& sl2,
                               Scanline& sl, Renderer& ren, 
                               AddSpanFunctor1 add_span1,
                               CombineSpansFunctor combine_spans)
    {
        // 准备扫描线生成器
        // 这里，sg1 是主扫描线生成器，sg2 是从扫描线生成器
        //-----------------
        if(!sg1.rewind_scanlines()) return;
        bool flag2 = sg2.rewind_scanlines();
    
        // 获取边界框
        //----------------
        rect_i r1(sg1.min_x(), sg1.min_y(), sg1.max_x(), sg1.max_y());
    
        // 重置扫描线并获取前两个扫描线
        //-----------------
        sl.reset(sg1.min_x(), sg1.max_x());
        sl1.reset(sg1.min_x(), sg1.max_x());
        sl2.reset(sg2.min_x(), sg2.max_x());
        if(!sg1.sweep_scanline(sl1)) return;
    
        if(flag2) flag2 = sg2.sweep_scanline(sl2);
    
        ren.prepare();
    
        // 一个虚拟的 span2 处理器
        sbool_add_span_empty<Scanline2, Scanline> add_span2;
    
        // 主循环
        // 在这里，我们同步具有相同 Y 坐标的扫描线，忽略所有其他扫描线。
        // 只有具有相同 Y 坐标的扫描线才需要合并。
        //-----------------
        bool flag1 = true;
        do
        {
            // 同步 "从" 扫描线生成器到 "主" 扫描线生成器
            //-----------------
            while(flag2 && sl2.y() < sl1.y())
            {
                flag2 = sg2.sweep_scanline(sl2);
            }
    
            if(flag2 && sl2.y() == sl1.y())
            {
                // Y 坐标相同
                // 合并扫描线并在包含任何 span 的情况下进行渲染
                //----------------------
                sbool_unite_scanlines(sl1, sl2, sl, add_span1, add_span2, combine_spans);
                if(sl.num_spans())
                {
                    sl.finalize(sl1.y());
                    ren.render(sl);
                }
            }
            else
            {
                // 添加 span 并进行渲染
                sbool_add_spans_and_render(sl1, sl, ren, add_span1);
            }
    
            // 推进 "主" 扫描线生成器
            flag1 = sg1.sweep_scanline(sl1);
        }
        while(flag1);
    }
    
    //---------------------------------------------sbool_intersect_shapes_aa
    // 计算两个抗锯齿扫描线形状的交集。
    // 这里使用了 "扫描线生成器" 抽象。ScanlineGen1 和 ScanlineGen2 是生成器，可以是 rasterizer_scanline_aa<> 类型。
    // 它们的函数需要三个可以是不同类型的扫描线容器。
    //--------------------------------------------sbool_intersect_shapes_aa
    // 求两个具有反锯齿效果的扫描线形状的交集
    // 详见 intersect_shapes_aa 的更多注释
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_intersect_shapes_aa(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                   Scanline1& sl1, Scanline2& sl2,
                                   Scanline& sl, Renderer& ren)
    {
        // 创建一个用于合并扫描线的函数对象
        sbool_intersect_spans_aa<Scanline1, Scanline2, Scanline> combine_functor;
        // 调用 sbool_intersect_shapes 函数进行实际的形状交集计算
        sbool_intersect_shapes(sg1, sg2, sl1, sl2, sl, ren, combine_functor);
    }





    //-------------------------------------------------sbool_intersect_shapes_bin
    // 求两个二进制扫描线形状的交集（不考虑反锯齿效果）
    // 详见 intersect_shapes_aa 的更多注释
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_intersect_shapes_bin(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                    Scanline1& sl1, Scanline2& sl2,
                                    Scanline& sl, Renderer& ren)
    {
        // 创建一个用于合并扫描线的函数对象
        sbool_combine_spans_bin<Scanline1, Scanline2, Scanline> combine_functor;
        // 调用 sbool_intersect_shapes 函数进行实际的形状交集计算
        sbool_intersect_shapes(sg1, sg2, sl1, sl2, sl, ren, combine_functor);
    }





    //-------------------------------------------------sbool_unite_shapes_aa
    // 合并两个具有反锯齿效果的扫描线形状
    // 详见 intersect_shapes_aa 的更多注释
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_unite_shapes_aa(ScanlineGen1& sg1, ScanlineGen2& sg2,
                               Scanline1& sl1, Scanline2& sl2,
                               Scanline& sl, Renderer& ren)
    {
        // 创建两个用于添加扫描线段的函数对象
        sbool_add_span_aa<Scanline1, Scanline> add_functor1;
        sbool_add_span_aa<Scanline2, Scanline> add_functor2;
        // 创建一个用于合并扫描线的函数对象
        sbool_unite_spans_aa<Scanline1, Scanline2, Scanline> combine_functor;
        // 调用 sbool_unite_shapes 函数进行实际的形状合并计算
        sbool_unite_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                           add_functor1, add_functor2, combine_functor);
    }





    //------------------------------------------------sbool_unite_shapes_bin
    // 合并两个二进制扫描线形状（不考虑反锯齿效果）
    // 详见 intersect_shapes_aa 的更多注释
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    // 定义一个函数，用于合并两个二进制扫描线生成器的形状
    void sbool_unite_shapes_bin(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                Scanline1& sl1, Scanline2& sl2,
                                Scanline& sl, Renderer& ren)
    {
        // 创建用于二进制扫描线的添加函数对象
        sbool_add_span_bin<Scanline1, Scanline> add_functor1;
        sbool_add_span_bin<Scanline2, Scanline> add_functor2;
        // 创建用于合并二进制扫描线形状的函数对象
        sbool_combine_spans_bin<Scanline1, Scanline2, Scanline> combine_functor;
        // 调用实际的合并函数来合并形状
        sbool_unite_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                           add_functor1, add_functor2, combine_functor);
    }


    //---------------------------------------------------sbool_xor_shapes_aa
    // 对两个反锯齿扫描线形状应用异或操作。这里使用了修改过的“Linear”异或，
    // 而不是经典的“Saddle”异或。原因是确保结果与扫描线光栅化器产生的完全一致。
    // 有关更多注释，请参见intersect_shapes_aa。
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_xor_shapes_aa(ScanlineGen1& sg1, ScanlineGen2& sg2,
                             Scanline1& sl1, Scanline2& sl2,
                             Scanline& sl, Renderer& ren)
    {
        // 创建用于反锯齿扫描线的添加函数对象
        sbool_add_span_aa<Scanline1, Scanline> add_functor1;
        sbool_add_span_aa<Scanline2, Scanline> add_functor2;
        // 创建用于异或反锯齿扫描线形状的函数对象
        sbool_xor_spans_aa<Scanline1, Scanline2, Scanline, 
                           sbool_xor_formula_linear<> > combine_functor;
        // 调用实际的合并函数来执行异或操作
        sbool_unite_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                           add_functor1, add_functor2, combine_functor);
    }


    //------------------------------------------sbool_xor_shapes_saddle_aa
    // 对两个反锯齿扫描线形状应用经典的“Saddle”异或操作，计算出反锯齿值。
    // 异或公式为：a XOR b : 1-((1-a+a*b)*(1-b+a*b))
    // 有关更多注释，请参见intersect_shapes_aa。
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_xor_shapes_saddle_aa(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                    Scanline1& sl1, Scanline2& sl2,
                                    Scanline& sl, Renderer& ren)
    {
        // 创建用于反锯齿扫描线的添加函数对象
        sbool_add_span_aa<Scanline1, Scanline> add_functor1;
        sbool_add_span_aa<Scanline2, Scanline> add_functor2;
        // 创建用于经典"Saddle"异或操作的函数对象
        sbool_xor_spans_aa<Scanline1, 
                           Scanline2, 
                           Scanline, 
                           sbool_xor_formula_saddle<> > combine_functor;
        // 调用实际的合并函数来执行经典"Saddle"异或操作
        sbool_unite_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                           add_functor1, add_functor2, combine_functor);
    }
    // 对两个抗锯齿扫描线形状应用异或运算。此处计算抗锯齿值的绝对差：
    // a XOR b : abs(a-b)
    // 更多注释请参见 intersect_shapes_aa 函数
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_xor_shapes_abs_diff_aa(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                      Scanline1& sl1, Scanline2& sl2,
                                      Scanline& sl, Renderer& ren)
    {
        // 创建用于添加抗锯齿范围的函数对象
        sbool_add_span_aa<Scanline1, Scanline> add_functor1;
        sbool_add_span_aa<Scanline2, Scanline> add_functor2;
        
        // 创建用于异或操作的抗锯齿范围函数对象，使用绝对差异公式
        sbool_xor_spans_aa<Scanline1, 
                           Scanline2, 
                           Scanline, 
                           sbool_xor_formula_abs_diff> combine_functor;
        
        // 调用 sbool_unite_shapes 函数，将两个形状合并并应用上述函数对象
        sbool_unite_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                           add_functor1, add_functor2, combine_functor);
    }
    
    
    
    //--------------------------------------------------sbool_xor_shapes_bin
    // 对两个二进制扫描线形状应用异或运算（无抗锯齿效果）。
    // 更多注释请参见 intersect_shapes_aa 函数
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_xor_shapes_bin(ScanlineGen1& sg1, ScanlineGen2& sg2,
                              Scanline1& sl1, Scanline2& sl2,
                              Scanline& sl, Renderer& ren)
    {
        // 创建用于添加二进制范围的函数对象
        sbool_add_span_bin<Scanline1, Scanline> add_functor1;
        sbool_add_span_bin<Scanline2, Scanline> add_functor2;
        
        // 创建用于组合二进制范围的空函数对象
        sbool_combine_spans_empty<Scanline1, Scanline2, Scanline> combine_functor;
        
        // 调用 sbool_unite_shapes 函数，将两个形状合并并应用上述函数对象
        sbool_unite_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                           add_functor1, add_functor2, combine_functor);
    }
    
    
    
    
    
    //----------------------------------------------sbool_subtract_shapes_aa
    // 对带有抗锯齿效果的形状 "sg1-sg2" 进行减法操作
    // 更多注释请参见 intersect_shapes_aa 函数
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_subtract_shapes_aa(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                  Scanline1& sl1, Scanline2& sl2,
                                  Scanline& sl, Renderer& ren)
    {
        // 创建用于添加抗锯齿范围的函数对象
        sbool_add_span_aa<Scanline1, Scanline> add_functor;
        
        // 创建用于减法操作的抗锯齿范围函数对象
        sbool_subtract_spans_aa<Scanline1, Scanline2, Scanline> combine_functor;
        
        // 调用 sbool_subtract_shapes 函数，对两个形状进行减法操作并应用上述函数对象
        sbool_subtract_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                              add_functor, combine_functor);
    }
    //---------------------------------------------sbool_subtract_shapes_bin
    // 从二进制形状中减去形状 "sg1-sg2"，不进行抗锯齿处理
    // 更多注释请参见 intersect_shapes_aa
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_subtract_shapes_bin(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                   Scanline1& sl1, Scanline2& sl2,
                                   Scanline& sl, Renderer& ren)
    {
        // 创建一个用于添加的二进制形状处理函数对象
        sbool_add_span_bin<Scanline1, Scanline> add_functor;
        // 创建一个用于组合空间的二进制形状处理函数对象
        sbool_combine_spans_empty<Scanline1, Scanline2, Scanline> combine_functor;
        // 调用函数 sbool_subtract_shapes 进行二进制形状的减法操作
        sbool_subtract_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                              add_functor, combine_functor);
    }
    
    
    
    
    
    //------------------------------------------------------------sbool_op_e
    enum sbool_op_e
    {
        sbool_or,            //----sbool_or
        sbool_and,           //----sbool_and
        sbool_xor,           //----sbool_xor
        sbool_xor_saddle,    //----sbool_xor_saddle
        sbool_xor_abs_diff,  //----sbool_xor_abs_diff
        sbool_a_minus_b,     //----sbool_a_minus_b
        sbool_b_minus_a      //----sbool_b_minus_a
    };
    
    
    
    
    
    //----------------------------------------------sbool_combine_shapes_bin
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_combine_shapes_bin(sbool_op_e op,
                                  ScanlineGen1& sg1, ScanlineGen2& sg2,
                                  Scanline1& sl1, Scanline2& sl2,
                                  Scanline& sl, Renderer& ren)
    {
        switch(op)
        {
        case sbool_or          : sbool_unite_shapes_bin    (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_and         : sbool_intersect_shapes_bin(sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_xor         :
        case sbool_xor_saddle  : 
        case sbool_xor_abs_diff: sbool_xor_shapes_bin      (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_a_minus_b   : sbool_subtract_shapes_bin (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_b_minus_a   : sbool_subtract_shapes_bin (sg2, sg1, sl2, sl1, sl, ren); break;
        }
    }
    
    
    
    
    
    //-----------------------------------------------sbool_combine_shapes_aa
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_combine_shapes_aa(sbool_op_e op,
                                 ScanlineGen1& sg1, ScanlineGen2& sg2,
                                 Scanline1& sl1, Scanline2& sl2,
                                 Scanline& sl, Renderer& ren)
    {
        // 根据操作符选择相应的布尔运算函数并执行，根据不同的操作符调用不同的函数进行形状计算
        switch(op)
        {
        case sbool_or          : sbool_unite_shapes_aa       (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_and         : sbool_intersect_shapes_aa   (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_xor         : sbool_xor_shapes_aa         (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_xor_saddle  : sbool_xor_shapes_saddle_aa  (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_xor_abs_diff: sbool_xor_shapes_abs_diff_aa(sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_a_minus_b   : sbool_subtract_shapes_aa    (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_b_minus_a   : sbool_subtract_shapes_aa    (sg2, sg1, sl2, sl1, sl, ren); break;
        }
    }
}


注释：

// 这是一个预处理指令的结束符号 '}'，用于结束一个条件编译块或函数定义。



#endif


注释：

// 这是一个预处理指令，用于结束一个条件编译块。#endif 与 #ifdef 或 #ifndef 配对使用，用来判断是否定义了某个宏，如果定义了，则编译下面的代码；如果没有定义，则忽略下面的代码。
```