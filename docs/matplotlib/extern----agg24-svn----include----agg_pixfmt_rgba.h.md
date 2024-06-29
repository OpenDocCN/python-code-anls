# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_pixfmt_rgba.h`

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
// Adaptation for high precision colors has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------

#ifndef AGG_PIXFMT_RGBA_INCLUDED
#define AGG_PIXFMT_RGBA_INCLUDED

#include <string.h>
#include <math.h>
#include "agg_pixfmt_base.h"
#include "agg_rendering_buffer.h"

namespace agg
{
    // 定义一个模板函数，用于返回两个值中的最小值
    template<class T> inline T sd_min(T a, T b) { return (a < b) ? a : b; }
    
    // 定义一个模板函数，用于返回两个值中的最大值
    template<class T> inline T sd_max(T a, T b) { return (a > b) ? a : b; }

    // 对 rgba 颜色进行裁剪，确保各个分量在合理范围内
    inline rgba & clip(rgba & c)
    {
        if (c.a > 1) c.a = 1; else if (c.a < 0) c.a = 0;
        if (c.r > c.a) c.r = c.a; else if (c.r < 0) c.r = 0;
        if (c.g > c.a) c.g = c.a; else if (c.g < 0) c.g = 0;
        if (c.b > c.a) c.b = c.a; else if (c.b < 0) c.b = 0;
        return c;
    }

    //=========================================================multiplier_rgba
    // rgba 颜色乘法结构体模板
    template<class ColorT, class Order> 
    struct multiplier_rgba
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;

        //--------------------------------------------------------------------
        // 预乘函数，将颜色分量乘以 alpha 通道值
        static AGG_INLINE void premultiply(value_type* p)
        {
            value_type a = p[Order::A];
            p[Order::R] = color_type::multiply(p[Order::R], a);
            p[Order::G] = color_type::multiply(p[Order::G], a);
            p[Order::B] = color_type::multiply(p[Order::B], a);
        }

        //--------------------------------------------------------------------
        // 反乘函数，将颜色分量除以 alpha 通道值
        static AGG_INLINE void demultiply(value_type* p)
        {
            value_type a = p[Order::A];
            p[Order::R] = color_type::demultiply(p[Order::R], a);
            p[Order::G] = color_type::demultiply(p[Order::G], a);
            p[Order::B] = color_type::demultiply(p[Order::B], a);
        }
    };

    //=====================================================apply_gamma_dir_rgba
    // 应用 gamma 校正的 RGBA 颜色类模板
    template<class ColorT, class Order, class GammaLut> 
    class apply_gamma_dir_rgba
    {
    //=====================================================apply_gamma_dir_rgba
    template<class ColorT, class Order, class GammaLut> class apply_gamma_dir_rgba
    {
    public:
        typedef ColorT color_type;                       // 定义颜色类型
        typedef typename color_type::value_type value_type;  // 定义值类型

        apply_gamma_dir_rgba(const GammaLut& gamma) : m_gamma(gamma) {}  // 构造函数，初始化 m_gamma

        AGG_INLINE void operator () (value_type* p)
        {
            p[Order::R] = m_gamma.dir(p[Order::R]);    // 对红色通道应用 gamma 正向调整
            p[Order::G] = m_gamma.dir(p[Order::G]);    // 对绿色通道应用 gamma 正向调整
            p[Order::B] = m_gamma.dir(p[Order::B]);    // 对蓝色通道应用 gamma 正向调整
        }

    private:
        const GammaLut& m_gamma;   // 存储 gamma 查找表的引用
    };

    //=====================================================apply_gamma_inv_rgba
    template<class ColorT, class Order, class GammaLut> class apply_gamma_inv_rgba
    {
    public:
        typedef ColorT color_type;                       // 定义颜色类型
        typedef typename color_type::value_type value_type;  // 定义值类型

        apply_gamma_inv_rgba(const GammaLut& gamma) : m_gamma(gamma) {}  // 构造函数，初始化 m_gamma

        AGG_INLINE void operator () (value_type* p)
        {
            p[Order::R] = m_gamma.inv(p[Order::R]);    // 对红色通道应用 gamma 反向调整
            p[Order::G] = m_gamma.inv(p[Order::G]);    // 对绿色通道应用 gamma 反向调整
            p[Order::B] = m_gamma.inv(p[Order::B]);    // 对蓝色通道应用 gamma 反向调整
        }

    private:
        const GammaLut& m_gamma;   // 存储 gamma 查找表的引用
    };


    template<class ColorT, class Order> 
    struct conv_rgba_pre
    {
        typedef ColorT color_type;                       // 定义颜色类型
        typedef Order order_type;                        // 定义顺序类型
        typedef typename color_type::value_type value_type;  // 定义值类型

        //--------------------------------------------------------------------
        static AGG_INLINE void set_plain_color(value_type* p, color_type c)
        {
            c.premultiply();                            // 对颜色进行预乘处理
            p[Order::R] = c.r;                          // 将预乘后的红色通道赋给 p 数组
            p[Order::G] = c.g;                          // 将预乘后的绿色通道赋给 p 数组
            p[Order::B] = c.b;                          // 将预乘后的蓝色通道赋给 p 数组
            p[Order::A] = c.a;                          // 将预乘后的透明度通道赋给 p 数组
        }

        //--------------------------------------------------------------------
        static AGG_INLINE color_type get_plain_color(const value_type* p)
        {
            return color_type(
                p[Order::R],                            // 从 p 数组获取红色通道值
                p[Order::G],                            // 从 p 数组获取绿色通道值
                p[Order::B],                            // 从 p 数组获取蓝色通道值
                p[Order::A]).demultiply();              // 从 p 数组获取透明度通道值，并进行反预乘处理
        }
    };

    template<class ColorT, class Order> 
    struct conv_rgba_plain
    {
        typedef ColorT color_type;                       // 定义颜色类型
        typedef Order order_type;                        // 定义顺序类型
        typedef typename color_type::value_type value_type;  // 定义值类型

        //--------------------------------------------------------------------
        static AGG_INLINE void set_plain_color(value_type* p, color_type c)
        {
            p[Order::R] = c.r;                          // 将红色通道值赋给 p 数组
            p[Order::G] = c.g;                          // 将绿色通道值赋给 p 数组
            p[Order::B] = c.b;                          // 将蓝色通道值赋给 p 数组
            p[Order::A] = c.a;                          // 将透明度通道值赋给 p 数组
        }

        //--------------------------------------------------------------------
        static AGG_INLINE color_type get_plain_color(const value_type* p)
        {
            return color_type(
                p[Order::R],                            // 从 p 数组获取红色通道值
                p[Order::G],                            // 从 p 数组获取绿色通道值
                p[Order::B],                            // 从 p 数组获取蓝色通道值
                p[Order::A]);                           // 从 p 数组获取透明度通道值
        }
    };

    //=============================================================blender_rgba
    // Blends "plain" (i.e. non-premultiplied) colors into a premultiplied buffer.
    // 定义模板结构体 blender_rgba，继承自 conv_rgba_pre<ColorT, Order>
    template<class ColorT, class Order> 
    struct blender_rgba : conv_rgba_pre<ColorT, Order>
    {
        // 声明类型别名
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        // 使用 Alvy-Ray Smith 的非预乘合成函数混合像素。
        // 由于渲染缓冲区实际上是预乘的，因此我们省略了初始的预乘和最终的解预乘过程。
        
        //--------------------------------------------------------------------
        // 静态内联函数，混合像素
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha, cover_type cover)
        {
            // 调用另一个 blend_pix 函数，用 alpha 和 cover 混合像素
            blend_pix(p, cr, cg, cb, color_type::mult_cover(alpha, cover));
        }
        
        //--------------------------------------------------------------------
        // 静态内联函数，混合像素
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha)
        {
            // 在每个颜色通道上使用线性插值，混合像素
            p[Order::R] = color_type::lerp(p[Order::R], cr, alpha);
            p[Order::G] = color_type::lerp(p[Order::G], cg, alpha);
            p[Order::B] = color_type::lerp(p[Order::B], cb, alpha);
            p[Order::A] = color_type::prelerp(p[Order::A], alpha, alpha);
        }
    };
    {
        // 定义类型别名，从 ColorT 中获取颜色类型和顺序类型
        typedef ColorT color_type;
        typedef Order order_type;
        // 从颜色类型中获取值类型、计算类型和长整型类型
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
    
        // 使用 Alvy-Ray Smith 的预乘形式混合像素的合成函数进行像素混合
    
        //--------------------------------------------------------------------
        // 内联函数：将像素按预乘形式混合，使用给定的颜色和覆盖参数
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha, cover_type cover)
        {
            // 调用重载的 blend_pix 函数，对颜色进行覆盖处理后进行混合
            blend_pix(p, 
                color_type::mult_cover(cr, cover), 
                color_type::mult_cover(cg, cover), 
                color_type::mult_cover(cb, cover), 
                color_type::mult_cover(alpha, cover));
        }
        
        //--------------------------------------------------------------------
        // 内联函数：将像素按预乘形式混合，使用给定的颜色和不透明度
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha)
        {
            // 根据颜色顺序将给定颜色按预乘形式与像素进行混合
            p[Order::R] = color_type::prelerp(p[Order::R], cr, alpha);
            p[Order::G] = color_type::prelerp(p[Order::G], cg, alpha);
            p[Order::B] = color_type::prelerp(p[Order::B], cb, alpha);
            p[Order::A] = color_type::prelerp(p[Order::A], alpha, alpha);
        }
    };
    
    //======================================================blender_rgba_plain
    // 将“普通”（非预乘）颜色混合到普通（非预乘）缓冲区中。
    template<class ColorT, class Order> 
    struct blender_rgba_plain : conv_rgba_plain<ColorT, Order>
    {
        // 定义类型别名，用于颜色和顺序操作
        typedef ColorT color_type;
        typedef Order order_type;
        // 确定值类型，计算类型和长整型
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        // 使用非预乘形式的 Alvy-Ray Smith 混合函数混合像素。

        //--------------------------------------------------------------------
        // 内联函数：混合像素，考虑覆盖度
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha, cover_type cover)
        {
            // 调用重载的 blend_pix 函数，将 alpha 和覆盖度乘起来作为混合因子
            blend_pix(p, cr, cg, cb, color_type::mult_cover(alpha, cover));
        }
        
        //--------------------------------------------------------------------
        // 内联函数：混合像素，不考虑覆盖度
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha)
        {
            // 如果 alpha 大于空值（即完全透明），则进行混合操作
            if (alpha > color_type::empty_value())
            {
                calc_type a = p[Order::A]; // 获取像素中 Alpha 通道的值
                // 分别计算每个通道的混合结果
                calc_type r = color_type::multiply(p[Order::R], a);
                calc_type g = color_type::multiply(p[Order::G], a);
                calc_type b = color_type::multiply(p[Order::B], a);
                // 使用线性插值函数计算新的像素值
                p[Order::R] = color_type::lerp(r, cr, alpha);
                p[Order::G] = color_type::lerp(g, cg, alpha);
                p[Order::B] = color_type::lerp(b, cb, alpha);
                // 更新 Alpha 通道值
                p[Order::A] = color_type::prelerp(a, alpha, alpha);
                // 反向处理像素，将预乘颜色还原为非预乘颜色
                multiplier_rgba<ColorT, Order>::demultiply(p);
            }
        }
    };

    // SVG 混合操作。
    // 规范详见 http://www.w3.org/TR/SVGCompositing/

    //=========================================================comp_op_rgba_clear
    // RGBA 清除操作
    template<class ColorT, class Order> 
    struct comp_op_rgba_clear : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = 0
        // Da'  = 0
        // 内联函数：混合像素，根据覆盖度操作
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 如果覆盖度足够大，完全清除像素
            if (cover >= cover_full)
            {
                p[0] = p[1] = p[2] = p[3] = color_type::empty_value(); 
            }
            // 如果覆盖度在完全透明和完全不透明之间，则根据覆盖度进行混合操作
            else if (cover > cover_none)
            {
                set(p, get(p, cover_full - cover));
            }
        }
    };

    //===========================================================comp_op_rgba_src
    // RGBA 源操作
    template<class ColorT, class Order> 
    struct comp_op_rgba_src : blender_base<ColorT, Order>
    {
        // 定义 ColorT 为 Color 类型，color_type 为 ColorT 的别名
        typedef ColorT color_type;
        // 定义 value_type 为 color_type 类型的值类型
        typedef typename color_type::value_type value_type;
        // 使用 blender_base 类模板中的 get 和 set 方法
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Sca
        // Da'  = Sa
        // 混合像素函数，根据给定的像素数据和覆盖度进行混合操作
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 如果覆盖度大于等于完全覆盖
            if (cover >= cover_full)
            {
                // 设置像素 p 的颜色为 (r, g, b, a)
                set(p, r, g, b, a);
            }
            else
            {
                // 使用覆盖度为 cover 的像素数据进行混合计算
                rgba s = get(r, g, b, a, cover);
                // 获取像素 p 的当前颜色数据
                rgba d = get(p, cover_full - cover);
                // 将源像素 s 的颜色分量加到目标像素 d 上
                d.r += s.r;
                d.g += s.g;
                d.b += s.b;
                d.a += s.a;
                // 将混合后的颜色数据设置回像素 p
                set(p, d);
            }
        }
    };

    //===========================================================comp_op_rgba_dst
    // RGBA 目标混合操作结构体
    template<class ColorT, class Order> 
    struct comp_op_rgba_dst : blender_base<ColorT, Order>
    {
        // 定义 ColorT 为 Color 类型，color_type 为 ColorT 的别名
        typedef ColorT color_type;
        // 定义 value_type 为 color_type 类型的值类型
        typedef typename color_type::value_type value_type;

        // Dca' = Dca.Sa + Dca.(1 - Sa) = Dca
        // Da'  = Da.Sa + Da.(1 - Sa) = Da
        // 目标混合像素函数，不进行实际的混合计算
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 这太简单了！
        }
    };

    //======================================================comp_op_rgba_src_over
    // RGBA 源覆盖混合操作结构体
    template<class ColorT, class Order> 
    struct comp_op_rgba_src_over : blender_base<ColorT, Order>
    {
        // 定义 ColorT 为 Color 类型，color_type 为 ColorT 的别名
        typedef ColorT color_type;
        // 定义 value_type 为 color_type 类型的值类型
        typedef typename color_type::value_type value_type;
        // 使用 blender_base 类模板中的 get 和 set 方法
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Sca + Dca.(1 - Sa) = Dca + Sca - Dca.Sa
        // Da'  = Sa + Da - Sa.Da 
        // 源覆盖混合像素函数，根据给定的像素数据和覆盖度进行混合操作
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
#if 1
            // 如果条件为真，则调用 blender_rgba_pre 的 blend_pix 方法
            blender_rgba_pre<ColorT, Order>::blend_pix(p, r, g, b, a, cover);
#else
            // 如果条件为假，则执行以下代码块
            // 获取源像素和目标像素的 rgba 值
            rgba s = get(r, g, b, a, cover);
            rgba d = get(p);
            // 根据 Porter-Duff 合成规则计算新的目标像素值
            d.r += s.r - d.r * s.a;
            d.g += s.g - d.g * s.a;
            d.b += s.b - d.b * s.a;
            d.a += s.a - d.a * s.a;
            // 将计算后的值设置为目标像素的值
            set(p, d);
#endif
        }
    };

    //======================================================comp_op_rgba_dst_over
    // 定义 RGBA 混合模式 dst_over
    template<class ColorT, class Order> 
    struct comp_op_rgba_dst_over : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Porter-Duff 合成公式：Dca' = Dca + Sca.(1 - Da), Da' = Da + Sa - Sa.Da
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取源像素和目标像素的 rgba 值
            rgba s = get(r, g, b, a, cover);
            rgba d = get(p);
            double d1a = 1 - d.a;
            // 根据合成公式计算新的目标像素值
            d.r += s.r * d1a;
            d.g += s.g * d1a;
            d.b += s.b * d1a;
            d.a += s.a * d1a;
            // 将计算后的值设置为目标像素的值
            set(p, d);
        }
    };

    //======================================================comp_op_rgba_src_in
    // 定义 RGBA 混合模式 src_in
    template<class ColorT, class Order> 
    struct comp_op_rgba_src_in : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Porter-Duff 合成公式：Dca' = Sca.Da, Da' = Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取目标像素的 alpha 值
            double da = ColorT::to_double(p[Order::A]);
            // 如果目标像素的 alpha 大于 0，则执行混合操作
            if (da > 0)
            {
                // 获取源像素的 rgba 值
                rgba s = get(r, g, b, a, cover);
                // 根据合成公式计算新的目标像素值
                rgba d = get(p, cover_full - cover);
                d.r += s.r * da;
                d.g += s.g * da;
                d.b += s.b * da;
                d.a += s.a * da;
                // 将计算后的值设置为目标像素的值
                set(p, d);
            }
        }
    };

    //======================================================comp_op_rgba_dst_in
    // 定义 RGBA 混合模式 dst_in
    template<class ColorT, class Order> 
    struct comp_op_rgba_dst_in : blender_base<ColorT, Order>
    {
        // 定义颜色类型为 ColorT
        typedef ColorT color_type;
        // 定义值类型为 color_type 的值类型
        typedef typename color_type::value_type value_type;
        // 使用 blender_base<ColorT, Order> 的 get 函数
        using blender_base<ColorT, Order>::get;
        // 使用 blender_base<ColorT, Order> 的 set 函数
    
        // 混合像素函数，用于混合 RGBA 颜色值
        // 根据公式 Dca' = Dca.Sa 和 Da' = Sa.Da 计算混合后的像素值
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 将输入的 alpha 值转换为双精度浮点数
            double sa = ColorT::to_double(a);
            // 获取目标像素 p 的值，考虑覆盖度
            rgba d = get(p, cover_full - cover);
            // 获取目标像素 p 的完整值，考虑完整的覆盖度
            rgba d2 = get(p, cover);
            // 根据混合公式计算新的 RGBA 值
            d.r += d2.r * sa;
            d.g += d2.g * sa;
            d.b += d2.b * sa;
            d.a += d2.a * sa;
            // 将计算得到的新值设置回目标像素 p
            set(p, d);
        }
    };
    
    //======================================================comp_op_rgba_src_out
    // RGBA 源透明度目标混合操作结构体
    template<class ColorT, class Order> 
    struct comp_op_rgba_src_out : blender_base<ColorT, Order>
    {
        // 定义颜色类型为 ColorT
        typedef ColorT color_type;
        // 定义值类型为 color_type 的值类型
        typedef typename color_type::value_type value_type;
        // 使用 blender_base<ColorT, Order> 的 get 函数
        using blender_base<ColorT, Order>::get;
        // 使用 blender_base<ColorT, Order> 的 set 函数
    
        // 混合像素函数，用于混合 RGBA 颜色值
        // 根据公式 Dca' = Sca.(1 - Da) 和 Da' = Sa.(1 - Da) 计算混合后的像素值
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取源像素的 RGBA 值
            rgba s = get(r, g, b, a, cover);
            // 获取目标像素 p 的值，考虑完整的覆盖度
            rgba d = get(p, cover_full - cover);
            // 计算目标像素 p 的 alpha 值的补数
            double d1a = 1 - ColorT::to_double(p[Order::A]);
            // 根据混合公式计算新的 RGBA 值
            d.r += s.r * d1a;
            d.g += s.g * d1a;
            d.b += s.b * d1a;
            d.a += s.a * d1a;
            // 将计算得到的新值设置回目标像素 p
            set(p, d);
        }
    };
    
    //======================================================comp_op_rgba_dst_out
    // RGBA 目标透明度目标混合操作结构体
    template<class ColorT, class Order> 
    struct comp_op_rgba_dst_out : blender_base<ColorT, Order>
    {
        // 定义颜色类型为 ColorT
        typedef ColorT color_type;
        // 定义值类型为 color_type 的值类型
        typedef typename color_type::value_type value_type;
        // 使用 blender_base<ColorT, Order> 的 get 函数
        using blender_base<ColorT, Order>::get;
        // 使用 blender_base<ColorT, Order> 的 set 函数
    
        // 混合像素函数，用于混合 RGBA 颜色值
        // 根据公式 Dca' = Dca.(1 - Sa) 和 Da' = Da.(1 - Sa) 计算混合后的像素值
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取目标像素 p 的值，考虑完整的覆盖度
            rgba d = get(p, cover_full - cover);
            // 获取目标像素 p 的值，考虑覆盖度
            rgba dc = get(p, cover);
            // 计算源像素 alpha 值的补数
            double s1a = 1 - ColorT::to_double(a);
            // 根据混合公式计算新的 RGBA 值
            d.r += dc.r * s1a;
            d.g += dc.g * s1a;
            d.b += dc.b * s1a;
            d.a += dc.a * s1a;
            // 将计算得到的新值设置回目标像素 p
            set(p, d);
        }
    };
    
    //=====================================================comp_op_rgba_src_atop
    // RGBA 源在顶部混合操作结构体
    template<class ColorT, class Order> 
    struct comp_op_rgba_src_atop : blender_base<ColorT, Order>
    {
        // 定义颜色类型为 ColorT，值类型为 color_type 的 value_type
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        // 使用 blender_base 模板中的 get 和 set 函数
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;
    
        // 混合像素函数，用于实现 Dca' = Sca.Da + Dca.(1 - Sa) 和 Da' = Da 的混合操作
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取源像素 rgba 表示
            rgba s = get(r, g, b, a, cover);
            // 获取目标像素 rgba 表示
            rgba d = get(p);
            // 计算补色因子 (1 - Sa)
            double s1a = 1 - s.a;
            // 执行混合计算
            d.r = s.r * d.a + d.r * s1a;
            d.g = s.g * d.a + d.g * s1a;
            d.b = s.b * d.a + d.g * s1a;
            // 将混合后的像素值设置回目标像素
            set(p, d);
        }
    };
    
    //=====================================================comp_op_rgba_dst_atop
    template<class ColorT, class Order> 
    struct comp_op_rgba_dst_atop : blender_base<ColorT, Order>
    {
        // 定义颜色类型为 ColorT，值类型为 color_type 的 value_type
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        // 使用 blender_base 模板中的 get 和 set 函数
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;
    
        // 混合像素函数，用于实现 Dca' = Dca.Sa + Sca.(1 - Da) 和 Da' = Sa 的混合操作
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取源像素 rgba 表示
            rgba sc = get(r, g, b, a, cover);
            // 获取目标像素 rgba 表示
            rgba dc = get(p, cover);
            // 获取目标像素的完全覆盖 rgba 表示
            rgba d = get(p, cover_full - cover);
            // 计算 Sa 和 (1 - Da) 的补色因子
            double sa = ColorT::to_double(a);
            double d1a = 1 - ColorT::to_double(p[Order::A]);
            // 执行混合计算
            d.r += dc.r * sa + sc.r * d1a;
            d.g += dc.g * sa + sc.g * d1a;
            d.b += dc.b * sa + sc.b * d1a;
            d.a += sc.a;
            // 将混合后的像素值设置回目标像素
            set(p, d);
        }
    };
    
    //=========================================================comp_op_rgba_xor
    template<class ColorT, class Order> 
    struct comp_op_rgba_xor : blender_base<ColorT, Order>
    {
        // 定义颜色类型为 ColorT，值类型为 color_type 的 value_type
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        // 使用 blender_base 模板中的 get 和 set 函数
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;
    
        // 混合像素函数，用于实现 Dca' = Sca.(1 - Da) + Dca.(1 - Sa) 和 Da' = Sa + Da - 2.Sa.Da 的混合操作
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取源像素 rgba 表示
            rgba s = get(r, g, b, a, cover);
            // 获取目标像素 rgba 表示
            rgba d = get(p);
            // 计算 Sa 和 (1 - Da) 的补色因子
            double s1a = 1 - s.a;
            double d1a = 1 - ColorT::to_double(p[Order::A]);
            // 执行混合计算
            d.r = s.r * d1a + d.r * s1a;
            d.g = s.g * d1a + d.g * s1a;
            d.b = s.b * d1a + d.b * s1a;
            d.a = s.a + d.a - 2 * s.a * d.a;
            // 将混合后的像素值设置回目标像素
            set(p, d);
        }
    };
    {
        // 定义颜色类型为 ColorT 和像素顺序类型为 Order
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        // 使用 blender_base 类的 get 和 set 方法
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;
    
        // Dca' = Sca + Dca
        // Da'  = Sa + Da 
        // 像素混合函数，将颜色 r, g, b, a 混合到像素 p 上，根据覆盖度 cover
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取混合源像素 s
            rgba s = get(r, g, b, a, cover);
            // 如果混合源像素的 alpha 大于 0
            if (s.a > 0)
            {
                // 获取目标像素 d
                rgba d = get(p);
                // 计算新的 alpha 值
                d.a = sd_min(d.a + s.a, 1.0);
                // 计算新的颜色通道值，保证不超过 alpha
                d.r = sd_min(d.r + s.r, d.a);
                d.g = sd_min(d.g + s.g, d.a);
                d.b = sd_min(d.b + s.b, d.a);
                // 将新的颜色值设置到像素 p 上，同时进行裁剪
                set(p, clip(d));
            }
        }
    };
    
    //========================================================comp_op_rgba_minus
    // 注意：不包含在 SVG 规范中。
    // RGBA 减法混合模式
    template<class ColorT, class Order> 
    struct comp_op_rgba_minus : blender_base<ColorT, Order>
    {
        // 定义颜色类型为 ColorT 和像素顺序类型为 Order
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        // 使用 blender_base 类的 get 和 set 方法
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;
    
        // Dca' = Dca - Sca
        // Da' = 1 - (1 - Sa).(1 - Da) = Da + Sa - Sa.Da
        // 像素混合函数，将颜色 r, g, b, a 混合到像素 p 上，根据覆盖度 cover
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取混合源像素 s
            rgba s = get(r, g, b, a, cover);
            // 如果混合源像素的 alpha 大于 0
            if (s.a > 0)
            {
                // 获取目标像素 d
                rgba d = get(p);
                // 计算新的 alpha 值
                d.a += s.a - s.a * d.a;
                // 计算新的颜色通道值，保证不低于 0
                d.r = sd_max(d.r - s.r, 0.0);
                d.g = sd_max(d.g - s.g, 0.0);
                d.b = sd_max(d.b - s.b, 0.0);
                // 将新的颜色值设置到像素 p 上，同时进行裁剪
                set(p, clip(d));
            }
        }
    };
    
    //=====================================================comp_op_rgba_multiply
    // RGBA 乘法混合模式
    template<class ColorT, class Order> 
    struct comp_op_rgba_multiply : blender_base<ColorT, Order>
    {
        // 定义颜色类型为 ColorT 和像素顺序类型为 Order
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        // 使用 blender_base 类的 get 和 set 方法
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;
    
        // Dca' = Sca.Dca + Sca.(1 - Da) + Dca.(1 - Sa)
        // Da'  = Sa + Da - Sa.Da 
        // 像素混合函数，将颜色 r, g, b, a 混合到像素 p 上，根据覆盖度 cover
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取混合源像素 s
            rgba s = get(r, g, b, a, cover);
            // 如果混合源像素的 alpha 大于 0
            if (s.a > 0)
            {
                // 获取目标像素 d
                rgba d = get(p);
                // 计算中间值
                double s1a = 1 - s.a;
                double d1a = 1 - d.a;
                // 计算新的颜色通道值
                d.r = s.r * d.r + s.r * d1a + d.r * s1a;
                d.g = s.g * d.g + s.g * d1a + d.g * s1a;
                d.b = s.b * d.b + s.b * d1a + d.b * s1a;
                // 计算新的 alpha 值
                d.a += s.a - s.a * d.a;
                // 将新的颜色值设置到像素 p 上，同时进行裁剪
                set(p, clip(d));
            }
        }
    };
    
    //=====================================================comp_op_rgba_screen
    template<class ColorT, class Order> 
    // 定义模板结构体 comp_op_rgba_screen，继承自 blender_base<ColorT, Order>
    struct comp_op_rgba_screen : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;  // 使用 blender_base 的 get 方法
        using blender_base<ColorT, Order>::set;  // 使用 blender_base 的 set 方法

        // 混合像素的屏幕模式操作
        // Dca' = Sca + Dca - Sca.Dca
        // Da'  = Sa + Da - Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取源像素 s
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)  // 如果源像素不完全透明
            {
                // 获取目标像素 d
                rgba d = get(p);
                // 执行屏幕混合操作
                d.r += s.r - s.r * d.r;
                d.g += s.g - s.g * d.g;
                d.b += s.b - s.b * d.b;
                d.a += s.a - s.a * d.a;
                // 将处理后的像素值设置回目标像素
                set(p, clip(d));
            }
        }
    };

    //=====================================================comp_op_rgba_overlay
    // 定义模板结构体 comp_op_rgba_overlay，继承自 blender_base<ColorT, Order>
    template<class ColorT, class Order> 
    struct comp_op_rgba_overlay : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;  // 使用 blender_base 的 get 方法
        using blender_base<ColorT, Order>::set;  // 使用 blender_base 的 set 方法

        // 混合像素的叠加模式操作
        // if 2.Dca <= Da
        //   Dca' = 2.Sca.Dca + Sca.(1 - Da) + Dca.(1 - Sa)
        // otherwise
        //   Dca' = Sa.Da - 2.(Da - Dca).(Sa - Sca) + Sca.(1 - Da) + Dca.(1 - Sa)
        // 
        // Da' = Sa + Da - Sa.Da
        static AGG_INLINE double calc(double dca, double sca, double da, double sa, double sada, double d1a, double s1a)
        {
            return (2 * dca <= da) ? 
                2 * sca * dca + sca * d1a + dca * s1a : 
                sada - 2 * (da - dca) * (sa - sca) + sca * d1a + dca * s1a;
        }

        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取源像素 s
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)  // 如果源像素不完全透明
            {
                // 获取目标像素 d
                rgba d = get(p);
                // 计算所需的参数值
                double d1a = 1 - d.a;
                double s1a = 1 - s.a;
                double sada = s.a * d.a;
                // 执行叠加混合操作
                d.r = calc(d.r, s.r, d.a, s.a, sada, d1a, s1a);
                d.g = calc(d.g, s.g, d.a, s.a, sada, d1a, s1a);
                d.b = calc(d.b, s.b, d.a, s.a, sada, d1a, s1a);
                d.a += s.a - s.a * d.a;
                // 将处理后的像素值设置回目标像素
                set(p, clip(d));
            }
        }
    };

    //=====================================================comp_op_rgba_darken
    // 定义模板结构体 comp_op_rgba_darken，继承自 blender_base<ColorT, Order>
    template<class ColorT, class Order> 
    struct comp_op_rgba_darken : blender_base<ColorT, Order>
    {
        // 定义颜色类型为 ColorT，值类型为其 value_type
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        // 使用 blender_base<ColorT, Order> 中的 get 和 set 方法
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;
    
        // 混合模式公式：Dca' = min(Sca.Da, Dca.Sa) + Sca.(1 - Da) + Dca.(1 - Sa)
        // Da'  = Sa + Da - Sa.Da 
        // 对象的静态方法，混合像素值
        static AGG_INLINE void blend_pix(value_type* p,
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取源像素 rgba 表示
            rgba s = get(r, g, b, a, cover);
            // 如果源像素的 alpha 大于 0
            if (s.a > 0)
            {
                // 获取目标像素 rgba 表示
                rgba d = get(p);
                // 计算补色
                double d1a = 1 - d.a;
                double s1a = 1 - s.a;
                // 根据混合公式计算混合后的颜色值
                d.r = sd_min(s.r * d.a, d.r * s.a) + s.r * d1a + d.r * s1a;
                d.g = sd_min(s.g * d.a, d.g * s.a) + s.g * d1a + d.g * s1a;
                d.b = sd_min(s.b * d.a, d.b * s.a) + s.b * d1a + d.b * s1a;
                // 更新混合后的 alpha 值
                d.a += s.a - s.a * d.a;
                // 将混合后的颜色值设置回目标像素
                set(p, clip(d));
            }
        }
    };
    
    //=====================================================comp_op_rgba_lighten
    // 定义颜色混合模式类 comp_op_rgba_lighten，继承自 blender_base<ColorT, Order>
    template<class ColorT, class Order> 
    struct comp_op_rgba_lighten : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        // 使用 blender_base<ColorT, Order> 中的 get 和 set 方法
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;
    
        // 混合模式公式：Dca' = max(Sca.Da, Dca.Sa) + Sca.(1 - Da) + Dca.(1 - Sa)
        // Da'  = Sa + Da - Sa.Da 
        // 对象的静态方法，混合像素值
        static AGG_INLINE void blend_pix(value_type* p,
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取源像素 rgba 表示
            rgba s = get(r, g, b, a, cover);
            // 如果源像素的 alpha 大于 0
            if (s.a > 0)
            {
                // 获取目标像素 rgba 表示
                rgba d = get(p);
                // 计算补色
                double d1a = 1 - d.a;
                double s1a = 1 - s.a;
                // 根据混合公式计算混合后的颜色值
                d.r = sd_max(s.r * d.a, d.r * s.a) + s.r * d1a + d.r * s1a;
                d.g = sd_max(s.g * d.a, d.g * s.a) + s.g * d1a + d.g * s1a;
                d.b = sd_max(s.b * d.a, d.b * s.a) + s.b * d1a + d.b * s1a;
                // 更新混合后的 alpha 值
                d.a += s.a - s.a * d.a;
                // 将混合后的颜色值设置回目标像素
                set(p, clip(d));
            }
        }
    };
    
    //=====================================================comp_op_rgba_color_dodge
    // 接着定义颜色混合模式类 comp_op_rgba_color_dodge，继承自 blender_base<ColorT, Order>
    template<class ColorT, class Order> 
    struct comp_op_rgba_color_dodge : blender_base<ColorT, Order>
    // 定义颜色类型为 ColorT
    typedef ColorT color_type;
    // 获取颜色值类型，并命名为 value_type
    typedef typename color_type::value_type value_type;
    // 使用 blender_base 类模板中的 get 和 set 方法
    using blender_base<ColorT, Order>::get;
    using blender_base<ColorT, Order>::set;

    // 混合模式计算函数，根据不同情况计算输出像素的颜色分量
    // 如果 Sca == Sa 且 Dca == 0，则 Dca' = Sca.(1 - Da)
    // 否则如果 Sca == Sa，则 Dca' = Sa.Da + Sca.(1 - Da) + Dca.(1 - Sa)
    // 否则如果 Sca < Sa，则 Dca' = Sa.Da.min(1, Dca/Da.Sa/(Sa - Sca)) + Sca.(1 - Da) + Dca.(1 - Sa)
    //
    // 输出像素的 alpha 通道计算公式为 Da' = Sa + Da - Sa.Da
    static AGG_INLINE double calc(double dca, double sca, double da, double sa, double sada, double d1a, double s1a)
    {
        if (sca < sa) return sada * sd_min(1.0, (dca / da) * sa / (sa - sca)) + sca * d1a + dca * s1a;
        if (dca > 0) return sada + sca * d1a + dca * s1a;
        return sca * d1a;
    }

    // 像素混合函数，根据源像素和目标像素的 alpha 值进行混合
    static AGG_INLINE void blend_pix(value_type* p, 
        value_type r, value_type g, value_type b, value_type a, cover_type cover)
    {
        // 获取源像素的 RGBA 颜色值
        rgba s = get(r, g, b, a, cover);
        // 如果源像素的 alpha 大于 0
        if (s.a > 0)
        {
            // 获取目标像素的 RGBA 颜色值
            rgba d = get(p);
            // 如果目标像素的 alpha 大于 0
            if (d.a > 0)
            {
                // 计算源像素与目标像素的混合结果
                double sada = s.a * d.a;
                double s1a = 1 - s.a;
                double d1a = 1 - d.a;
                // 计算混合后的红色分量
                d.r = calc(d.r, s.r, d.a, s.a, sada, d1a, s1a);
                // 计算混合后的绿色分量
                d.g = calc(d.g, s.g, d.a, s.a, sada, d1a, s1a);
                // 计算混合后的蓝色分量
                d.b = calc(d.b, s.b, d.a, s.a, sada, d1a, s1a);
                // 更新目标像素的 alpha 值
                d.a += s.a - s.a * d.a;
                // 将混合后的颜色值设置回目标像素
                set(p, clip(d));
            }
            else 
            {
                // 如果目标像素的 alpha 为 0，则直接将源像素设置为目标像素
                set(p, s);
            }
        }
    }
};
// RGBA 颜色混合模式 - 颜色加深（color burn）
template<class ColorT, class Order> 
struct comp_op_rgba_color_burn : blender_base<ColorT, Order>
    {
        // 定义别名以简化类型引用
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        // 使用基类中的成员函数
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;
    
        // 根据不同的混合条件计算像素混合后的结果
        static AGG_INLINE double calc(double dca, double sca, double da, double sa, double sada, double d1a, double s1a)
        {
            // 如果 Sca > 0，则根据以下公式计算混合后的颜色分量
            if (sca > 0)
                return sada * (1 - sd_min(1.0, (1 - dca / da) * sa / sca)) + sca * d1a + dca * s1a;
            
            // 如果 Dca > Da，则应用以下混合公式
            if (dca > da)
                return sada + dca * s1a;
            
            // 否则，只应用以下简化混合公式
            return dca * s1a;
        }
    
        // 对像素进行混合操作，根据当前像素值和传入的颜色进行处理
        static AGG_INLINE void blend_pix(value_type* p,
                                         value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取传入颜色的 RGBA 值及其覆盖度
            rgba s = get(r, g, b, a, cover);
            
            // 如果传入颜色的 alpha 值大于 0，则进行混合处理
            if (s.a > 0)
            {
                // 获取当前像素的 RGBA 值
                rgba d = get(p);
                
                // 如果当前像素的 alpha 值大于 0，则执行混合计算
                if (d.a > 0)
                {
                    double sada = s.a * d.a;
                    double s1a = 1 - s.a;
                    double d1a = 1 - d.a;
                    
                    // 分别对 RGB 通道执行混合计算
                    d.r = calc(d.r, s.r, d.a, s.a, sada, d1a, s1a);
                    d.g = calc(d.g, s.g, d.a, s.a, sada, d1a, s1a);
                    d.b = calc(d.b, s.b, d.a, s.a, sada, d1a, s1a);
                    
                    // 更新混合后的 alpha 值
                    d.a += s.a - sada;
                    
                    // 将混合后的像素值写回到当前像素
                    set(p, clip(d));
                }
                else
                {
                    // 如果当前像素的 alpha 值为 0，则直接将传入的颜色值设置为当前像素值
                    set(p, s);
                }
            }
        }
    };
    
    
    
    //=====================================================comp_op_rgba_hard_light
    // 定义 RGBA 像素的硬光混合操作
    template<class ColorT, class Order> 
    struct comp_op_rgba_hard_light : blender_base<ColorT, Order>
    {
        // 定义颜色类型为 ColorT，并定义值类型为 color_type 的 value_type
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        // 使用 blender_base 模板类中的 get 和 set 函数
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;
    
        // 根据软光混合模式计算像素颜色混合结果
        // 如果 2 * Sca < Sa，则使用以下计算公式：
        //    Dca' = 2 * Sca * Dca + Sca * (1 - Da) + Dca * (1 - Sa)
        // 否则使用以下计算公式：
        //    Dca' = Sa * Da - 2 * (Da - Dca) * (Sa - Sca) + Sca * (1 - Da) + Dca * (1 - Sa)
        // 
        // 计算新的 alpha 值：
        //    Da' = Sa + Da - Sa * Da
        static AGG_INLINE double calc(double dca, double sca, double da, double sa, double sada, double d1a, double s1a)
        {
            return (2 * sca < sa) ? 
                2 * sca * dca + sca * d1a + dca * s1a : 
                sada - 2 * (da - dca) * (sa - sca) + sca * d1a + dca * s1a;
        }
    
        // 对单个像素进行软光混合操作
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取源像素 rgba
            rgba s = get(r, g, b, a, cover);
            // 如果源像素 alpha 大于 0，则进行混合计算
            if (s.a > 0)
            {
                // 获取目标像素 rgba
                rgba d = get(p);
                // 计算 1 - 目标像素 alpha 和 1 - 源像素 alpha
                double d1a = 1 - d.a;
                double s1a = 1 - s.a;
                // 计算源像素 alpha 乘目标像素 alpha
                double sada = s.a * d.a;
                // 对目标像素的 RGB 通道分别进行软光混合计算
                d.r = calc(d.r, s.r, d.a, s.a, sada, d1a, s1a);
                d.g = calc(d.g, s.g, d.a, s.a, sada, d1a, s1a);
                d.b = calc(d.b, s.b, d.a, s.a, sada, d1a, s1a);
                // 更新目标像素 alpha 值
                d.a += s.a - sada;
                // 将处理后的目标像素 rgba 设置回去
                set(p, clip(d));
            }
        }
    };
    
    //=====================================================comp_op_rgba_soft_light
    // RGBA 软光混合模式结构体
    template<class ColorT, class Order> 
    struct comp_op_rgba_soft_light : blender_base<ColorT, Order>
    {
        // 定义 ColorT 为颜色类型，value_type 为其值类型
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        // 使用 blender_base 模板中的 get 和 set 函数
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;
    
        // 根据不同条件计算像素混合结果
        // 如果 2 * Sca <= Sa
        //   Dca' = Dca * Sa - (Sa * Da - 2 * Sca * Da) * Dca * Sa * (Sa * Da - Dca * Sa) + Sca * (1 - Da) + Dca * (1 - Sa)
        // 否则如果 2 * Sca > Sa 并且 4 * Dca <= Da
        //   Dca' = Dca * Sa + (2 * Sca * Da - Sa * Da) * (((((16 * Dca * Sa - 12) * Dca * Sa + 4) * Dca * Da) - Dca * Da)) + Sca * (1 - Da) + Dca * (1 - Sa)
        // 否则如果 2 * Sca > Sa 并且 4 * Dca > Da
        //   Dca' = Dca * Sa + (2 * Sca * Da - Sa * Da) * ((sqrt(Dca * Sa) - Dca * Sa)) + Sca * (1 - Da) + Dca * (1 - Sa)
        // 
        // Da' = Sa + Da - Sa * Da
        static AGG_INLINE double calc(double dca, double sca, double da, double sa, double sada, double d1a, double s1a)
        {
            double dcasa = dca * sa;
            if (2 * sca <= sa) return dcasa - (sada - 2 * sca * da) * dcasa * (sada - dcasa) + sca * d1a + dca * s1a;
            if (4 * dca <= da) return dcasa + (2 * sca * da - sada) * ((((16 * dcasa - 12) * dcasa + 4) * dca * da) - dca * da) + sca * d1a + dca * s1a;
            return dcasa + (2 * sca * da - sada) * (sqrt(dcasa) - dcasa) + sca * d1a + dca * s1a;
        }
    
        // 像素混合函数，将 s 混合到目标像素 p 上
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取源像素 s
            rgba s = get(r, g, b, a, cover);
            // 如果源像素的 alpha 大于 0
            if (s.a > 0)
            {
                // 获取目标像素 d
                rgba d = get(p);
                // 如果目标像素的 alpha 大于 0
                if (d.a > 0)
                {
                    double sada = s.a * d.a;
                    double s1a = 1 - s.a;
                    double d1a = 1 - d.a;
                    // 计算每个通道的混合结果
                    d.r = calc(d.r, s.r, d.a, s.a, sada, d1a, s1a);
                    d.g = calc(d.g, s.g, d.a, s.a, sada, d1a, s1a);
                    d.b = calc(d.b, s.b, d.a, s.a, sada, d1a, s1a);
                    // 更新目标像素的 alpha
                    d.a += s.a - sada;
                    // 将混合结果设置回目标像素 p，并进行范围裁剪
                    set(p, clip(d));
                }
                else
                {
                    // 如果目标像素的 alpha 不大于 0，则直接将源像素设置到目标像素 p 上
                    set(p, s);
                }
            }
        }
    };
    
    //=====================================================comp_op_rgba_difference
    template<class ColorT, class Order> 
    struct comp_op_rgba_difference : blender_base<ColorT, Order>
    {
        // 定义颜色类型为 ColorT
        typedef ColorT color_type;
        // 定义值类型为 color_type 的值类型
        typedef typename color_type::value_type value_type;
        // 使用 blender_base 模板中的 get 函数
        using blender_base<ColorT, Order>::get;
        // 使用 blender_base 模板中的 set 函数
        using blender_base<ColorT, Order>::set;
    
        // 混合像素函数，根据指定的 RGBA 值和覆盖度进行像素混合
        // 使用公式：Dca' = Sca + Dca - 2.min(Sca.Da, Dca.Sa)
        //          Da'  = Sa + Da - Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取源像素 s 的 RGBA 值和覆盖度
            rgba s = get(r, g, b, a, cover);
            // 如果源像素的 alpha 值大于 0
            if (s.a > 0)
            {
                // 获取目标像素 d 的 RGBA 值
                rgba d = get(p);
                // 根据混合公式更新目标像素的 RGBA 值
                d.r += s.r - 2 * sd_min(s.r * d.a, d.r * s.a);
                d.g += s.g - 2 * sd_min(s.g * d.a, d.g * s.a);
                d.b += s.b - 2 * sd_min(s.b * d.a, d.b * s.a);
                d.a += s.a - s.a * d.a;
                // 将更新后的目标像素值设置回 p 指向的位置，确保值在合理范围内
                set(p, clip(d));
            }
        }
    };
    
    //=====================================================comp_op_rgba_exclusion
    template<class ColorT, class Order> 
    struct comp_op_rgba_exclusion : blender_base<ColorT, Order>
    {
        // 定义颜色类型为 ColorT
        typedef ColorT color_type;
        // 定义值类型为 color_type 的值类型
        typedef typename color_type::value_type value_type;
        // 使用 blender_base 模板中的 get 函数
        using blender_base<ColorT, Order>::get;
        // 使用 blender_base 模板中的 set 函数
        using blender_base<ColorT, Order>::set;
    
        // 混合像素函数，根据指定的 RGBA 值和覆盖度进行像素混合
        // 使用公式：Dca' = (Sca.Da + Dca.Sa - 2.Sca.Dca) + Sca.(1 - Da) + Dca.(1 - Sa)
        //          Da'  = Sa + Da - Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取源像素 s 的 RGBA 值和覆盖度
            rgba s = get(r, g, b, a, cover);
            // 如果源像素的 alpha 值大于 0
            if (s.a > 0)
            {
                // 获取目标像素 d 的 RGBA 值
                rgba d = get(p);
                // 计算混合后的颜色分量值，并更新目标像素的 RGBA 值
                double d1a = 1 - d.a;
                double s1a = 1 - s.a;
                d.r = (s.r * d.a + d.r * s.a - 2 * s.r * d.r) + s.r * d1a + d.r * s1a;
                d.g = (s.g * d.a + d.g * s.a - 2 * s.g * d.g) + s.g * d1a + d.g * s1a;
                d.b = (s.b * d.a + d.b * s.a - 2 * s.b * d.b) + s.b * d1a + d.b * s1a;
                d.a += s.a - s.a * d.a;
                // 将更新后的目标像素值设置回 p 指向的位置，确保值在合理范围内
                set(p, clip(d));
            }
        }
    };
//=====================================================comp_op_rgba_contrast
template<class ColorT, class Order> struct comp_op_rgba_contrast
{
    typedef ColorT color_type;  // 定义颜色类型
    typedef Order order_type;  // 定义顺序类型
    typedef typename color_type::value_type value_type;  // 颜色值类型
    typedef typename color_type::calc_type calc_type;  // 计算类型
    typedef typename color_type::long_type long_type;  // 长整型类型
    enum base_scale_e
    { 
        base_shift = color_type::base_shift,  // 基础位移
        base_mask  = color_type::base_mask    // 基础掩码
    };


    static AGG_INLINE void blend_pix(value_type* p, 
                                     unsigned sr, unsigned sg, unsigned sb, 
                                     unsigned sa, unsigned cover)
    {
        if (cover < 255)
        {
            sr = (sr * cover + 255) >> 8;  // 计算红色通道
            sg = (sg * cover + 255) >> 8;  // 计算绿色通道
            sb = (sb * cover + 255) >> 8;  // 计算蓝色通道
            sa = (sa * cover + 255) >> 8;  // 计算透明度通道
        }
        long_type dr = p[Order::R];  // 目标像素红色分量
        long_type dg = p[Order::G];  // 目标像素绿色分量
        long_type db = p[Order::B];  // 目标像素蓝色分量
        int       da = p[Order::A];  // 目标像素透明度分量
        long_type d2a = da >> 1;  // 目标像素透明度一半
        unsigned s2a = sa >> 1;   // 源像素透明度一半

        // 计算混合后的红、绿、蓝色分量
        int r = (int)((((dr - d2a) * int((sr - s2a)*2 + base_mask)) >> base_shift) + d2a); 
        int g = (int)((((dg - d2a) * int((sg - s2a)*2 + base_mask)) >> base_shift) + d2a); 
        int b = (int)((((db - d2a) * int((sb - s2a)*2 + base_mask)) >> base_shift) + d2a); 

        // 确保颜色值在合理范围内
        r = (r < 0) ? 0 : r;
        g = (g < 0) ? 0 : g;
        b = (b < 0) ? 0 : b;

        // 更新目标像素的红、绿、蓝色分量
        p[Order::R] = (value_type)((r > da) ? da : r);
        p[Order::G] = (value_type)((g > da) ? da : g);
        p[Order::B] = (value_type)((b > da) ? da : b);
    }
};
    {
        // 定义颜色类型为 ColorT，顺序类型为 Order
        typedef ColorT color_type;
        typedef Order order_type;
        // 定义值类型为 color_type 的值类型
        typedef typename color_type::value_type value_type;
        // 定义计算类型为 color_type 的计算类型
        typedef typename color_type::calc_type calc_type;
        // 定义长整型类型为 color_type 的长整型类型
        typedef typename color_type::long_type long_type;
        // 定义基本比例枚举
        enum base_scale_e
        { 
            // 基本移位量为 color_type::base_shift
            base_shift = color_type::base_shift,
            // 基本掩码为 color_type::base_mask
            base_mask  = color_type::base_mask
        };
    
        // 混合像素的静态内联函数
        // 根据 Alpha 混合公式计算像素颜色值
        static AGG_INLINE void blend_pix(value_type* p, 
                                         unsigned sr, unsigned sg, unsigned sb, 
                                         unsigned sa, unsigned cover)
        {
            // 计算实际的 sa 值，考虑覆盖率 cover
            sa = (sa * cover + 255) >> 8;
            // 如果 sa 不为零
            if (sa)
            {
                // 计算当前像素的 Alpha 值
                calc_type da = p[Order::A];
                // 计算红色通道的混合值
                calc_type dr = ((da - p[Order::R]) * sa + base_mask) >> base_shift;
                // 计算绿色通道的混合值
                calc_type dg = ((da - p[Order::G]) * sa + base_mask) >> base_shift;
                // 计算蓝色通道的混合值
                calc_type db = ((da - p[Order::B]) * sa + base_mask) >> base_shift;
                // 计算反转后的 sa
                calc_type s1a = base_mask - sa;
                // 更新红色通道的值
                p[Order::R] = (value_type)(dr + ((p[Order::R] * s1a + base_mask) >> base_shift));
                // 更新绿色通道的值
                p[Order::G] = (value_type)(dg + ((p[Order::G] * s1a + base_mask) >> base_shift));
                // 更新蓝色通道的值
                p[Order::B] = (value_type)(db + ((p[Order::B] * s1a + base_mask) >> base_shift));
                // 更新 Alpha 通道的值
                p[Order::A] = (value_type)(sa + da - ((sa * da + base_mask) >> base_shift));
            }
        }
    };
    
    //=================================================comp_op_rgba_invert_rgb
    // RGBA 到反转 RGB 的混合操作模板结构
    template<class ColorT, class Order> struct comp_op_rgba_invert_rgb
    {
        // 定义类型别名，使用 ColorT 作为颜色类型，Order 作为顺序类型
        typedef ColorT color_type;
        typedef Order order_type;
        // 定义值类型、计算类型和长整型类型，根据颜色类型的成员类型推导
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
        // 枚举基本比例常量
        enum base_scale_e
        { 
            // 从颜色类型获取基本移位和掩码值
            base_shift = color_type::base_shift,
            base_mask  = color_type::base_mask
        };
    
        // 混合像素的静态内联函数
        // 根据给定的像素和混合参数，进行像素混合操作
        static AGG_INLINE void blend_pix(value_type* p, 
                                         unsigned sr, unsigned sg, unsigned sb, 
                                         unsigned sa, unsigned cover)
        {
            // 如果覆盖度小于255，则进行覆盖度调整
            if (cover < 255)
            {
                sr = (sr * cover + 255) >> 8;
                sg = (sg * cover + 255) >> 8;
                sb = (sb * cover + 255) >> 8;
                sa = (sa * cover + 255) >> 8;
            }
            // 如果 alpha 值不为0，则执行像素混合计算
            if (sa)
            {
                calc_type da = p[Order::A];  // 获取目标像素的 alpha 值
                // 计算每个颜色通道的混合结果
                calc_type dr = ((da - p[Order::R]) * sr + base_mask) >> base_shift;
                calc_type dg = ((da - p[Order::G]) * sg + base_mask) >> base_shift;
                calc_type db = ((da - p[Order::B]) * sb + base_mask) >> base_shift;
                calc_type s1a = base_mask - sa;  // 计算 1 - Sa
    
                // 更新目标像素的 RGB 值
                p[Order::R] = (value_type)(dr + ((p[Order::R] * s1a + base_mask) >> base_shift));
                p[Order::G] = (value_type)(dg + ((p[Order::G] * s1a + base_mask) >> base_shift));
                p[Order::B] = (value_type)(db + ((p[Order::B] * s1a + base_mask) >> base_shift));
                // 更新目标像素的 alpha 值
                p[Order::A] = (value_type)(sa + da - ((sa * da + base_mask) >> base_shift));
            }
        }
    };
#endif


//======================================================comp_op_table_rgba
// 定义模板结构体 comp_op_table_rgba，用于存储 RGBA 颜色操作函数表
template<class ColorT, class Order> struct comp_op_table_rgba
{
    typedef typename ColorT::value_type value_type; // 定义颜色值类型
    typedef typename ColorT::calc_type calc_type; // 定义计算类型
    // 定义操作函数指针类型，接受颜色值和覆盖度作为参数
    typedef void (*comp_op_func_type)(value_type* p, 
                                      value_type cr, 
                                      value_type cg, 
                                      value_type cb,
                                      value_type ca,
                                      cover_type cover);
    static comp_op_func_type g_comp_op_func[]; // 声明操作函数指针数组
};

//==========================================================g_comp_op_func
// 定义 comp_op_table_rgba 结构体中的静态成员 g_comp_op_func
template<class ColorT, class Order> 
typename comp_op_table_rgba<ColorT, Order>::comp_op_func_type
comp_op_table_rgba<ColorT, Order>::g_comp_op_func[] = 
{
    comp_op_rgba_clear      <ColorT,Order>::blend_pix, // 清除混合模式的像素混合函数
    comp_op_rgba_src        <ColorT,Order>::blend_pix, // 源像素混合模式的像素混合函数
    comp_op_rgba_dst        <ColorT,Order>::blend_pix, // 目标像素混合模式的像素混合函数
    comp_op_rgba_src_over   <ColorT,Order>::blend_pix, // 源覆盖目标混合模式的像素混合函数
    comp_op_rgba_dst_over   <ColorT,Order>::blend_pix, // 目标覆盖源混合模式的像素混合函数
    comp_op_rgba_src_in     <ColorT,Order>::blend_pix, // 源在目标内混合模式的像素混合函数
    comp_op_rgba_dst_in     <ColorT,Order>::blend_pix, // 目标在源内混合模式的像素混合函数
    comp_op_rgba_src_out    <ColorT,Order>::blend_pix, // 源在目标外混合模式的像素混合函数
    comp_op_rgba_dst_out    <ColorT,Order>::blend_pix, // 目标在源外混合模式的像素混合函数
    comp_op_rgba_src_atop   <ColorT,Order>::blend_pix, // 源覆盖目标顶部混合模式的像素混合函数
    comp_op_rgba_dst_atop   <ColorT,Order>::blend_pix, // 目标覆盖源顶部混合模式的像素混合函数
    comp_op_rgba_xor        <ColorT,Order>::blend_pix, // 异或混合模式的像素混合函数
    comp_op_rgba_plus       <ColorT,Order>::blend_pix, // 加法混合模式的像素混合函数
    //comp_op_rgba_minus    <ColorT,Order>::blend_pix, // 减法混合模式的像素混合函数（已注释）
    comp_op_rgba_multiply   <ColorT,Order>::blend_pix, // 乘法混合模式的像素混合函数
    comp_op_rgba_screen     <ColorT,Order>::blend_pix, // 屏幕混合模式的像素混合函数
    comp_op_rgba_overlay    <ColorT,Order>::blend_pix, // 叠加混合模式的像素混合函数
    comp_op_rgba_darken     <ColorT,Order>::blend_pix, // 变暗混合模式的像素混合函数
    comp_op_rgba_lighten    <ColorT,Order>::blend_pix, // 变亮混合模式的像素混合函数
    comp_op_rgba_color_dodge<ColorT,Order>::blend_pix, // 颜色减淡混合模式的像素混合函数
    comp_op_rgba_color_burn <ColorT,Order>::blend_pix, // 颜色加深混合模式的像素混合函数
    comp_op_rgba_hard_light <ColorT,Order>::blend_pix, // 强光混合模式的像素混合函数
    comp_op_rgba_soft_light <ColorT,Order>::blend_pix, // 柔光混合模式的像素混合函数
    comp_op_rgba_difference <ColorT,Order>::blend_pix, // 差值混合模式的像素混合函数
    comp_op_rgba_exclusion  <ColorT,Order>::blend_pix, // 排除混合模式的像素混合函数
    //comp_op_rgba_contrast <ColorT,Order>::blend_pix, // 对比度混合模式的像素混合函数（已注释）
    //comp_op_rgba_invert   <ColorT,Order>::blend_pix, // 反转混合模式的像素混合函数（已注释）
    //comp_op_rgba_invert_rgb<ColorT,Order>::blend_pix, // RGB反转混合模式的像素混合函数（已注释）
    0 // 结束标志，表示函数指针数组的结束
};


//==============================================================comp_op_e
// 定义混合操作枚举 comp_op_e
enum comp_op_e
    {
        comp_op_clear,         // 清除混合操作，将目标完全透明化
        comp_op_src,           // 将源覆盖到目标，忽略目标
        comp_op_dst,           // 将目标复制到结果，忽略源
        comp_op_src_over,      // 将源覆盖到目标，保留目标的内容
        comp_op_dst_over,      // 将目标覆盖到源，保留源的内容
        comp_op_src_in,        // 只保留源和目标重叠部分，源的内容
        comp_op_dst_in,        // 只保留源和目标重叠部分，目标的内容
        comp_op_src_out,       // 只保留源和目标不重叠部分，源的内容
        comp_op_dst_out,       // 只保留源和目标不重叠部分，目标的内容
        comp_op_src_atop,      // 将源覆盖到目标，保留源和目标重叠部分
        comp_op_dst_atop,      // 将目标覆盖到源，保留源和目标重叠部分
        comp_op_xor,           // 异或操作，保留源和目标不重叠部分
        comp_op_plus,          // 颜色值相加，饱和处理
        //comp_op_minus,         // 颜色值相减，饱和处理（已注释掉）
        comp_op_multiply,      // 颜色值相乘，饱和处理
        comp_op_screen,        // 屏幕模式混合
        comp_op_overlay,       // 叠加模式混合
        comp_op_darken,        // 变暗模式混合
        comp_op_lighten,       // 变亮模式混合
        comp_op_color_dodge,   // 颜色减淡模式混合
        comp_op_color_burn,    // 颜色加深模式混合
        comp_op_hard_light,    // 强光模式混合
        comp_op_soft_light,    // 柔光模式混合
        comp_op_difference,    // 差值模式混合
        comp_op_exclusion,     // 排除模式混合
        //comp_op_contrast,      // 对比度模式混合（已注释掉）
        //comp_op_invert,        // 反相模式混合（已注释掉）
        //comp_op_invert_rgb,    // RGB反相模式混合（已注释掉）
    
        end_of_comp_op_e       // 混合操作枚举结束标记
    };
    
    //====================================================comp_op_adaptor_rgba
    template<class ColorT, class Order> 
    struct comp_op_adaptor_rgba
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
    
        // RGBA颜色混合像素函数适配器
        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 调用 RGBA 混合操作函数表中的特定混合函数
            comp_op_table_rgba<ColorT, Order>::g_comp_op_func[op](p, 
                color_type::multiply(r, a), 
                color_type::multiply(g, a), 
                color_type::multiply(b, a), 
                a, cover);
        }
    };
    
    //=========================================comp_op_adaptor_clip_to_dst_rgba
    template<class ColorT, class Order> 
    struct comp_op_adaptor_clip_to_dst_rgba
    {
        // 定义颜色类型、顺序类型和相关值类型
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
    
        // 定义混合像素函数，根据给定的操作符混合指定颜色和透明度到像素数组中
        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 对颜色进行乘法混合，将结果赋值给 r, g, b
            r = color_type::multiply(r, a);
            g = color_type::multiply(g, a);
            b = color_type::multiply(b, a);
            
            // 获取目标像素的透明度
            value_type da = p[Order::A];
            
            // 调用 RGBA 混合操作函数，将乘法混合后的颜色和透明度混合到目标像素上
            comp_op_table_rgba<ColorT, Order>::g_comp_op_func[op](p, 
                color_type::multiply(r, da), 
                color_type::multiply(g, da), 
                color_type::multiply(b, da), 
                color_type::multiply(a, da), cover);
        }
    };
    
    //================================================comp_op_adaptor_rgba_pre
    // RGBA 预处理混合操作适配器
    template<class ColorT, class Order> 
    struct comp_op_adaptor_rgba_pre
    {
        // 定义颜色类型、顺序类型和相关值类型
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
    
        // 定义混合像素函数，直接调用 RGBA 混合操作函数
        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 调用 RGBA 混合操作函数，将给定的颜色和透明度混合到目标像素上
            comp_op_table_rgba<ColorT, Order>::g_comp_op_func[op](p, r, g, b, a, cover);
        }
    };
    
    //=====================================comp_op_adaptor_clip_to_dst_rgba_pre
    // 剪切到目标 RGBA 预处理混合操作适配器
    template<class ColorT, class Order> 
    struct comp_op_adaptor_clip_to_dst_rgba_pre
    {
        // 定义颜色类型、顺序类型和相关值类型
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
    
        // 定义混合像素函数，对给定颜色乘以目标像素的透明度后调用 RGBA 混合操作函数
        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取目标像素的透明度
            value_type da = p[Order::A];
            
            // 调用 RGBA 混合操作函数，将乘法混合后的颜色和透明度混合到目标像素上
            comp_op_table_rgba<ColorT, Order>::g_comp_op_func[op](p, 
                color_type::multiply(r, da), 
                color_type::multiply(g, da), 
                color_type::multiply(b, da), 
                color_type::multiply(a, da), cover);
        }
    };
    
    //====================================================comp_op_adaptor_rgba_plain
    // RGBA 普通混合操作适配器
    template<class ColorT, class Order> 
    struct comp_op_adaptor_rgba_plain
    {
        // 声明颜色类型和顺序类型
        typedef ColorT color_type;
        typedef Order order_type;
        // 声明值类型、计算类型和长整型类型，这些类型均从颜色类型中获取
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
    
        // 像素混合函数，接受操作码、像素指针 p 和 RGBA 颜色分量及覆盖度参数
        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 对输入的像素值进行预乘处理
            multiplier_rgba<ColorT, Order>::premultiply(p);
            // 使用 RGBA 混合适配器执行像素混合操作
            comp_op_adaptor_rgba<ColorT, Order>::blend_pix(op, p, r, g, b, a, cover);
            // 对像素值进行反向处理以恢复未预乘状态
            multiplier_rgba<ColorT, Order>::demultiply(p);
        }
    };
    
    //===================================================comp_op_adaptor_clip_to_dst_rgba_plain
    template<class ColorT, class Order> 
    struct comp_op_adaptor_clip_to_dst_rgba_plain
    {
        // 声明颜色类型和顺序类型
        typedef ColorT color_type;
        typedef Order order_type;
        // 声明值类型、计算类型和长整型类型，这些类型均从颜色类型中获取
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
    
        // 像素混合函数，接受操作码、像素指针 p 和 RGBA 颜色分量及覆盖度参数
        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 对输入的像素值进行预乘处理
            multiplier_rgba<ColorT, Order>::premultiply(p);
            // 使用裁剪到目标的 RGBA 混合适配器执行像素混合操作
            comp_op_adaptor_clip_to_dst_rgba<ColorT, Order>::blend_pix(op, p, r, g, b, a, cover);
            // 对像素值进行反向处理以恢复未预乘状态
            multiplier_rgba<ColorT, Order>::demultiply(p);
        }
    };
    
    //==============================================================comp_adaptor_rgba
    template<class BlenderPre> 
    struct comp_adaptor_rgba
    {
        // 声明颜色类型、顺序类型和值类型、计算类型以及长整型类型，这些类型均从 BlenderPre 类中获取
        typedef typename BlenderPre::color_type color_type;
        typedef typename BlenderPre::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
    
        // 像素混合函数，接受操作码、像素指针 p 和 RGBA 颜色分量及覆盖度参数
        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 使用 BlenderPre 类的 blend_pix 方法，对颜色分量进行预乘和混合操作
            BlenderPre::blend_pix(p, 
                color_type::multiply(r, a), 
                color_type::multiply(g, a), 
                color_type::multiply(b, a), 
                a, cover);
        }
    };
    
    //================================================comp_adaptor_clip_to_dst_rgba
    template<class BlenderPre> 
    struct comp_adaptor_clip_to_dst_rgba
    {
    {
        // 定义使用 BlenderPre 的颜色类型、顺序类型、值类型、计算类型和长整型
        typedef typename BlenderPre::color_type color_type;
        typedef typename BlenderPre::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
    
        // 混合像素的静态内联方法，接受操作码、像素指针 p，以及 RGBA 颜色分量和覆盖率作为参数
        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 根据像素的 alpha 值 r、g、b 进行乘法混合
            r = color_type::multiply(r, a);
            g = color_type::multiply(g, a);
            b = color_type::multiply(b, a);
            // 获取目标像素的 alpha 值 da
            value_type da = p[order_type::A];
            // 使用 BlenderPre 定义的 blend_pix 方法进行像素混合，输入经过乘法混合后的颜色分量
            BlenderPre::blend_pix(p, 
                color_type::multiply(r, da), 
                color_type::multiply(g, da), 
                color_type::multiply(b, da), 
                color_type::multiply(a, da), cover);
        }
    };
    
    //=======================================================comp_adaptor_rgba_pre
    // 使用 BlenderPre 的适配器结构，适用于 RGBA 预乘的情况
    template<class BlenderPre> 
    struct comp_adaptor_rgba_pre
    {
        // 定义使用 BlenderPre 的颜色类型、顺序类型、值类型、计算类型和长整型
        typedef typename BlenderPre::color_type color_type;
        typedef typename BlenderPre::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
    
        // 混合像素的静态内联方法，接受操作码、像素指针 p，以及 RGBA 颜色分量和覆盖率作为参数
        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 直接使用 BlenderPre 定义的 blend_pix 方法，输入未经处理的 RGBA 颜色分量
            BlenderPre::blend_pix(p, r, g, b, a, cover);
        }
    };
    
    //======================================comp_adaptor_clip_to_dst_rgba_pre
    // 使用 BlenderPre 的适配器结构，适用于 RGBA 预乘并进行目标像素裁剪的情况
    template<class BlenderPre> 
    struct comp_adaptor_clip_to_dst_rgba_pre
    {
        // 定义使用 BlenderPre 的颜色类型、顺序类型、值类型、计算类型和长整型
        typedef typename BlenderPre::color_type color_type;
        typedef typename BlenderPre::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
    
        // 混合像素的静态内联方法，接受操作码、像素指针 p，以及 RGBA 颜色分量和覆盖率作为参数
        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 获取目标像素的 alpha 值 da
            unsigned da = p[order_type::A];
            // 使用 BlenderPre 定义的 blend_pix 方法进行像素混合，输入经过乘法混合后的颜色分量和覆盖率
            BlenderPre::blend_pix(p, 
                color_type::multiply(r, da), 
                color_type::multiply(g, da), 
                color_type::multiply(b, da), 
                color_type::multiply(a, da), 
                cover);
        }
    };
    
    //=======================================================comp_adaptor_rgba_plain
    // 使用 BlenderPre 的适配器结构，适用于 RGBA 普通混合（非预乘）的情况
    template<class BlenderPre> 
    struct comp_adaptor_rgba_plain
    {
        // 定义命名空间 BlenderPre 中的类型别名
        typedef typename BlenderPre::color_type color_type;
        typedef typename BlenderPre::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
    
        // 像素混合函数，将指定颜色和透明度按照给定操作 op 混合到像素 p 上
        static AGG_INLINE void blend_pix(unsigned op, value_type* p,
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 对目标像素 p 进行预乘操作
            multiplier_rgba<color_type, order_type>::premultiply(p);
            // 使用 comp_adaptor_rgba<BlenderPre> 混合器混合像素
            comp_adaptor_rgba<BlenderPre>::blend_pix(op, p, r, g, b, a, cover);
            // 对混合后的像素 p 进行去乘操作
            multiplier_rgba<color_type, order_type>::demultiply(p);
        }
    };
    
    //==========================================comp_adaptor_clip_to_dst_rgba_plain
    // 使用 BlenderPre 类型的颜色和顺序类型，定义的像素混合适配器
    template<class BlenderPre> 
    struct comp_adaptor_clip_to_dst_rgba_plain
    {
        // 定义命名空间 BlenderPre 中的类型别名
        typedef typename BlenderPre::color_type color_type;
        typedef typename BlenderPre::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
    
        // 像素混合函数，将指定颜色和透明度按照给定操作 op 混合到像素 p 上
        static AGG_INLINE void blend_pix(unsigned op, value_type* p,
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // 对目标像素 p 进行预乘操作
            multiplier_rgba<color_type, order_type>::premultiply(p);
            // 使用 comp_adaptor_clip_to_dst_rgba<BlenderPre> 混合器混合像素
            comp_adaptor_clip_to_dst_rgba<BlenderPre>::blend_pix(op, p, r, g, b, a, cover);
            // 对混合后的像素 p 进行去乘操作
            multiplier_rgba<color_type, order_type>::demultiply(p);
        }
    };
    
    
    //=================================================pixfmt_alpha_blend_rgba
    // 使用 Blender 和 RenBuf 类型，定义的 RGBA 像素格式，支持透明度混合
    template<class Blender, class RenBuf> 
    class pixfmt_alpha_blend_rgba
    {
    // 定义公共成员（public），表示这些成员可以在类外部访问
    public:
        // 声明像素格式为 RGBA
        typedef pixfmt_rgba_tag pixfmt_category;
        // 使用 RenBuf 类型作为渲染缓冲区类型
        typedef RenBuf   rbuf_type;
        // 声明行数据类型为 rbuf_type 的行数据类型
        typedef typename rbuf_type::row_data row_data;
        // 使用 Blender 类型作为混合器类型
        typedef Blender  blender_type;
        // 声明颜色类型为 blender_type 的颜色类型
        typedef typename blender_type::color_type color_type;
        // 声明颜色通道顺序类型为 blender_type 的顺序类型
        typedef typename blender_type::order_type order_type;
        // 声明值类型为颜色类型的值类型
        typedef typename color_type::value_type value_type;
        // 声明计算类型为颜色类型的计算类型
        typedef typename color_type::calc_type calc_type;
        
        // 枚举常量定义
        enum 
        {
            // 像素步长为 4（即 RGBA 四个通道）
            pix_step = 4,
            // 像素宽度为值类型大小乘以像素步长
            pix_width = sizeof(value_type) * pix_step,
        };

        // 定义像素类型结构体
        struct pixel_type
        {
            // 像素颜色通道数组
            value_type c[pix_step];

            // 设置像素颜色值
            void set(value_type r, value_type g, value_type b, value_type a)
            {
                c[order_type::R] = r;  // 设置红色通道值
                c[order_type::G] = g;  // 设置绿色通道值
                c[order_type::B] = b;  // 设置蓝色通道值
                c[order_type::A] = a;  // 设置透明度通道值
            }

            // 使用颜色类型设置像素颜色值
            void set(const color_type& color)
            {
                set(color.r, color.g, color.b, color.a);  // 调用上面的 set 方法
            }

            // 获取像素颜色值
            void get(value_type& r, value_type& g, value_type& b, value_type& a) const
            {
                r = c[order_type::R];  // 获取红色通道值
                g = c[order_type::G];  // 获取绿色通道值
                b = c[order_type::B];  // 获取蓝色通道值
                a = c[order_type::A];  // 获取透明度通道值
            }

            // 获取像素颜色类型
            color_type get() const
            {
                return color_type(
                    c[order_type::R],    // 使用红色通道值
                    c[order_type::G],    // 使用绿色通道值
                    c[order_type::B],    // 使用蓝色通道值
                    c[order_type::A]);   // 使用透明度通道值
            }

            // 返回下一个像素的指针
            pixel_type* next()
            {
                return this + 1;  // 返回当前像素类型结构体的下一个实例的指针
            }

            // 返回下一个像素的常量指针
            const pixel_type* next() const
            {
                return this + 1;  // 返回当前像素类型结构体的下一个实例的常量指针
            }

            // 前进 n 步像素，返回指定步数后的像素指针
            pixel_type* advance(int n)
            {
                return this + n;  // 返回当前像素类型结构体前进 n 步后的像素指针
            }

            // 前进 n 步像素，返回指定步数后的常量像素指针
            const pixel_type* advance(int n) const
            {
                return this + n;  // 返回当前像素类型结构体前进 n 步后的常量像素指针
            }
        };
        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c, unsigned cover)
        {
            // 使用 Blender 对象调用 blend_pix 方法，混合当前像素 p 的颜色值与给定的颜色 c
            m_blender.blend_pix(p->c, c.r, c.g, c.b, c.a, cover);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c)
        {
            // 使用 Blender 对象调用 blend_pix 方法，混合当前像素 p 的颜色值与给定的不带覆盖的颜色 c
            m_blender.blend_pix(p->c, c.r, c.g, c.b, c.a);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c, unsigned cover)
        {
            // 如果颜色 c 不透明
            if (!c.is_transparent())
            {
                // 如果颜色 c 是不透明的并且覆盖率等于完全覆盖
                if (c.is_opaque() && cover == cover_mask)
                {
                    // 直接设置像素 p 的颜色为给定的颜色 c
                    p->set(c.r, c.g, c.b, c.a);
                }
                else
                {
                    // 使用 Blender 对象调用 blend_pix 方法，混合当前像素 p 的颜色值与给定的颜色 c，并考虑覆盖率
                    m_blender.blend_pix(p->c, c.r, c.g, c.b, c.a, cover);
                }
            }
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c)
        {
            // 如果颜色 c 不透明
            if (!c.is_transparent())
            {
                // 如果颜色 c 是不透明的
                if (c.is_opaque())
                {
                    // 直接设置像素 p 的颜色为给定的颜色 c
                    p->set(c.r, c.g, c.b, c.a);
                }
                else
                {
                    // 使用 Blender 对象调用 blend_pix 方法，混合当前像素 p 的颜色值与给定的颜色 c
                    m_blender.blend_pix(p->c, c.r, c.g, c.b, c.a);
                }
            }
        }
    public:
        typedef pixfmt_rgba_tag pixfmt_category;
        typedef RenBuf   rbuf_type;
        typedef typename rbuf_type::row_data row_data;
        typedef Blender  blender_type;
        typedef typename blender_type::color_type color_type;
        typedef typename blender_type::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        
        enum 
        {
            pix_step = 4,   // 定义每个像素的步长为4，即每个像素有四个通道
            pix_width  = sizeof(value_type) * pix_step,   // 计算像素的宽度，单位是字节
        };
        
        struct pixel_type
        {
            value_type c[pix_step];   // 存储像素值的数组，长度为pix_step

            void set(value_type r, value_type g, value_type b, value_type a)
            {
                c[order_type::R] = r;   // 设置红色通道的值
                c[order_type::G] = g;   // 设置绿色通道的值
                c[order_type::B] = b;   // 设置蓝色通道的值
                c[order_type::A] = a;   // 设置透明度通道的值
            }

            void set(const color_type& color)
            {
                set(color.r, color.g, color.b, color.a);   // 根据颜色对象设置像素值
            }

            void get(value_type& r, value_type& g, value_type& b, value_type& a) const
            {
                r = c[order_type::R];   // 获取红色通道的值
                g = c[order_type::G];   // 获取绿色通道的值
                b = c[order_type::B];   // 获取蓝色通道的值
                a = c[order_type::A];   // 获取透明度通道的值
            }

            color_type get() const
            {
                return color_type(
                    c[order_type::R],    // 返回颜色对象，使用像素的红色通道值
                    c[order_type::G],    // 使用像素的绿色通道值
                    c[order_type::B],    // 使用像素的蓝色通道值
                    c[order_type::A]);   // 使用像素的透明度通道值
            }

            pixel_type* next()
            {
                return this + 1;   // 返回下一个像素的指针
            }

            const pixel_type* next() const
            {
                return this + 1;   // 返回下一个像素的指针（常量版本）
            }

            pixel_type* advance(int n)
            {
                return this + n;   // 返回向前移动n个像素后的指针
            }

            const pixel_type* advance(int n) const
            {
                return this + n;   // 返回向前移动n个像素后的指针（常量版本）
            }
        };


    private:
        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c, unsigned cover = cover_full)
        {
            m_blender.blend_pix(m_comp_op, p->c, c.r, c.g, c.b, c.a, cover);
            // 使用混合器对像素p进行颜色混合操作，参数包括混合操作类型、当前像素的颜色通道值和给定颜色c的通道值及覆盖度cover
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c, unsigned cover = cover_full)
        {
            if (!c.is_transparent())   // 如果颜色c不是完全透明的
            {
                if (c.is_opaque() && cover == cover_mask)   // 如果颜色c是不透明且覆盖模式为掩码模式
                {
                    p->set(c.r, c.g, c.b, c.a);   // 直接设置像素p的颜色值为c的颜色值
                }
                else
                {
                    blend_pix(p, c, cover);   // 否则进行颜色混合操作
                }
            }
        }

    private:
        rbuf_type* m_rbuf;   // 渲染缓冲区指针
        Blender m_blender;   // 混合器对象
        unsigned m_comp_op;  // 混合操作类型
    };
    // 定义 RGBA 颜色混合器模板，使用 rgba8 数据格式和指定的像素顺序 order_rgba
    typedef blender_rgba<rgba8, order_rgba> blender_rgba32;
    // 定义 RGBA 颜色混合器模板，使用 rgba8 数据格式和指定的像素顺序 order_argb
    typedef blender_rgba<rgba8, order_argb> blender_argb32;
    // 定义 RGBA 颜色混合器模板，使用 rgba8 数据格式和指定的像素顺序 order_abgr
    typedef blender_rgba<rgba8, order_abgr> blender_abgr32;
    // 定义 RGBA 颜色混合器模板，使用 rgba8 数据格式和指定的像素顺序 order_bgra
    typedef blender_rgba<rgba8, order_bgra> blender_bgra32;
    
    // 定义 sRGBA 颜色混合器模板，使用 srgba8 数据格式和指定的像素顺序 order_rgba
    typedef blender_rgba<srgba8, order_rgba> blender_srgba32;
    // 定义 sRGBA 颜色混合器模板，使用 srgba8 数据格式和指定的像素顺序 order_argb
    typedef blender_rgba<srgba8, order_argb> blender_sargb32;
    // 定义 sRGBA 颜色混合器模板，使用 srgba8 数据格式和指定的像素顺序 order_abgr
    typedef blender_rgba<srgba8, order_abgr> blender_sabgr32;
    // 定义 sRGBA 颜色混合器模板，使用 srgba8 数据格式和指定的像素顺序 order_bgra
    typedef blender_rgba<srgba8, order_bgra> blender_sbgra32;
    
    // 定义预乘 RGBA 颜色混合器模板，使用 rgba8 数据格式和指定的像素顺序 order_rgba
    typedef blender_rgba_pre<rgba8, order_rgba> blender_rgba32_pre;
    // 定义预乘 RGBA 颜色混合器模板，使用 rgba8 数据格式和指定的像素顺序 order_argb
    typedef blender_rgba_pre<rgba8, order_argb> blender_argb32_pre;
    // 定义预乘 RGBA 颜色混合器模板，使用 rgba8 数据格式和指定的像素顺序 order_abgr
    typedef blender_rgba_pre<rgba8, order_abgr> blender_abgr32_pre;
    // 定义预乘 RGBA 颜色混合器模板，使用 rgba8 数据格式和指定的像素顺序 order_bgra
    typedef blender_rgba_pre<rgba8, order_bgra> blender_bgra32_pre;
    
    // 定义预乘 sRGBA 颜色混合器模板，使用 srgba8 数据格式和指定的像素顺序 order_rgba
    typedef blender_rgba_pre<srgba8, order_rgba> blender_srgba32_pre;
    // 定义预乘 sRGBA 颜色混合器模板，使用 srgba8 数据格式和指定的像素顺序 order_argb
    typedef blender_rgba_pre<srgba8, order_argb> blender_sargb32_pre;
    // 定义预乘 sRGBA 颜色混合器模板，使用 srgba8 数据格式和指定的像素顺序 order_abgr
    typedef blender_rgba_pre<srgba8, order_abgr> blender_sabgr32_pre;
    // 定义预乘 sRGBA 颜色混合器模板，使用 srgba8 数据格式和指定的像素顺序 order_bgra
    typedef blender_rgba_pre<srgba8, order_bgra> blender_sbgra32_pre;
    
    // 定义平面 RGBA 颜色混合器模板，使用 rgba8 数据格式和指定的像素顺序 order_rgba
    typedef blender_rgba_plain<rgba8, order_rgba> blender_rgba32_plain;
    // 定义平面 RGBA 颜色混合器模板，使用 rgba8 数据格式和指定的像素顺序 order_argb
    typedef blender_rgba_plain<rgba8, order_argb> blender_argb32_plain;
    // 定义平面 RGBA 颜色混合器模板，使用 rgba8 数据格式和指定的像素顺序 order_abgr
    typedef blender_rgba_plain<rgba8, order_abgr> blender_abgr32_plain;
    // 定义平面 RGBA 颜色混合器模板，使用 rgba8 数据格式和指定的像素顺序 order_bgra
    typedef blender_rgba_plain<rgba8, order_bgra> blender_bgra32_plain;
    
    // 定义平面 sRGBA 颜色混合器模板，使用 srgba8 数据格式和指定的像素顺序 order_rgba
    typedef blender_rgba_plain<srgba8, order_rgba> blender_srgba32_plain;
    // 定义平面 sRGBA 颜色混合器模板，使用 srgba8 数据格式和指定的像素顺序 order_argb
    typedef blender_rgba_plain<srgba8, order_argb> blender_sargb32_plain;
    // 定义平面 sRGBA 颜色混合器模板，使用 srgba8 数据格式和指定的像素顺序 order_abgr
    typedef blender_rgba_plain<srgba8, order_abgr> blender_sabgr32_plain;
    // 定义平面 sRGBA 颜色混合器模板，使用 srgba8 数据格式和指定的像素顺序 order_bgra
    typedef blender_rgba_plain<srgba8, order_bgra> blender_sbgra32_plain;
    
    // 定义 RGBA 颜色混合器模板，使用 rgba16 数据格式和指定的像素顺序 order_rgba
    typedef blender_rgba<rgba16, order_rgba> blender_rgba64;
    // 定义 RGBA 颜色混合器模板，使用 rgba16 数据格式和指定的像素顺序 order_argb
    typedef blender_rgba<rgba16, order_argb> blender_argb64;
    // 定义 RGBA 颜色混合器模板，使用 rgba16 数据格式和指定的像素顺序 order_abgr
    typedef blender_rgba<rgba16, order_abgr> blender_abgr64;
    // 定义 RGBA 颜色混合器模板，使用 rgba16 数据格式和指定的像素顺序 order_bgra
    typedef blender_rgba<rgba16, order_bgra> blender_bgra64;
    
    // 定义预乘 RGBA 颜色混合器模板，使用 rgba16 数据格式和指定的像素顺序 order_rgba
    typedef blender_rgba_pre<rgba16, order_rgba> blender_rgba64_pre;
    // 定义预乘 RGBA 颜色混合器模板，使用 rgba16 数据格式和指定的像素顺序 order_argb
    typedef blender_rgba_pre<rgba16, order_argb> blender_argb64_pre;
    // 定义预乘 RGBA 颜色混合器模板，使用 rgba16 数据格式和指定的像素顺序 order_abgr
    typedef blender_rgba_pre<rgba16, order_abgr> blender_abgr64_pre;
    // 定义预乘 RGBA 颜色混合器模板，使用 rgba16 数据格式和指定的像素顺序 order_bgra
    typedef blender_rgba_pre<rgba16, order_bgra> blender_bgra64_pre;
    
    // 定义平面 RGBA 颜色混合器模板，使用 rgba16 数据格式和指定的像素顺序 order_rgba
    typedef blender_rgba_plain<rgba16, order_rgba> blender_rgba64_plain;
    // 定义平面 RGBA 颜色混合器模板，使用 rgba16 数据格式和指定的像素顺序 order_argb
    typedef blender_rgba_plain<rgba16, order_argb> blender_argb64_plain;
    // 定义平面 RGBA 颜色混合器模板，使用 rgba16 数据格式和指定的像素顺序 order_abgr
    typedef blender_rgba_plain<rgba16, order_abgr> blender_abgr64_plain;
    // 定义平面 RGBA 颜色混合器模板，使用 rgba16 数据格式和指定的像素顺序 order_bgra
    typedef blender_rgba_plain<rgba16, order_bgra> blender_bgra64_plain;
    
    // 定义 RGBA 颜色混合器模板，使用 rgba32 数据格式和指定的像素顺序 order_rgba
    typedef blender_rgba<rgba32, order_rgba> blender_rgba128;
    // 定义 RGBA 颜色混合器模板，使用 rgba32 数据格式和指定的像素顺序 order_argb
    typedef blender_rgba<rgba32, order_argb> blender_argb128;
    // 定义 RGBA 颜色混合器模板，使用 rgba32 数据格式和指定的像素顺序 order_abgr
    // 定义四种不同的混合器类型，每种混合器类型对应不同的颜色顺序和数据格式
    typedef blender_rgba_plain<rgba32, order_rgba> blender_rgba128_plain;
    typedef blender_rgba_plain<rgba32, order_argb> blender_argb128_plain;
    typedef blender_rgba_plain<rgba32, order_abgr> blender_abgr128_plain;
    typedef blender_rgba_plain<rgba32, order_bgra> blender_bgra128_plain;

    //-----------------------------------------------------------------------
    // 定义基于 RGBA32 格式和指定混合器类型的像素格式类型
    typedef pixfmt_alpha_blend_rgba<blender_rgba32, rendering_buffer> pixfmt_rgba32;
    typedef pixfmt_alpha_blend_rgba<blender_argb32, rendering_buffer> pixfmt_argb32;
    typedef pixfmt_alpha_blend_rgba<blender_abgr32, rendering_buffer> pixfmt_abgr32;
    typedef pixfmt_alpha_blend_rgba<blender_bgra32, rendering_buffer> pixfmt_bgra32;

    typedef pixfmt_alpha_blend_rgba<blender_srgba32, rendering_buffer> pixfmt_srgba32;
    typedef pixfmt_alpha_blend_rgba<blender_sargb32, rendering_buffer> pixfmt_sargb32;
    typedef pixfmt_alpha_blend_rgba<blender_sabgr32, rendering_buffer> pixfmt_sabgr32;
    typedef pixfmt_alpha_blend_rgba<blender_sbgra32, rendering_buffer> pixfmt_sbgra32;

    typedef pixfmt_alpha_blend_rgba<blender_rgba32_pre, rendering_buffer> pixfmt_rgba32_pre;
    typedef pixfmt_alpha_blend_rgba<blender_argb32_pre, rendering_buffer> pixfmt_argb32_pre;
    typedef pixfmt_alpha_blend_rgba<blender_abgr32_pre, rendering_buffer> pixfmt_abgr32_pre;
    typedef pixfmt_alpha_blend_rgba<blender_bgra32_pre, rendering_buffer> pixfmt_bgra32_pre;

    typedef pixfmt_alpha_blend_rgba<blender_srgba32_pre, rendering_buffer> pixfmt_srgba32_pre;
    typedef pixfmt_alpha_blend_rgba<blender_sargb32_pre, rendering_buffer> pixfmt_sargb32_pre;
    typedef pixfmt_alpha_blend_rgba<blender_sabgr32_pre, rendering_buffer> pixfmt_sabgr32_pre;
    typedef pixfmt_alpha_blend_rgba<blender_sbgra32_pre, rendering_buffer> pixfmt_sbgra32_pre;

    typedef pixfmt_alpha_blend_rgba<blender_rgba32_plain, rendering_buffer> pixfmt_rgba32_plain;
    typedef pixfmt_alpha_blend_rgba<blender_argb32_plain, rendering_buffer> pixfmt_argb32_plain;
    typedef pixfmt_alpha_blend_rgba<blender_abgr32_plain, rendering_buffer> pixfmt_abgr32_plain;
    typedef pixfmt_alpha_blend_rgba<blender_bgra32_plain, rendering_buffer> pixfmt_bgra32_plain;

    typedef pixfmt_alpha_blend_rgba<blender_srgba32_plain, rendering_buffer> pixfmt_srgba32_plain;
    typedef pixfmt_alpha_blend_rgba<blender_sargb32_plain, rendering_buffer> pixfmt_sargb32_plain;
    typedef pixfmt_alpha_blend_rgba<blender_sabgr32_plain, rendering_buffer> pixfmt_sabgr32_plain;
    typedef pixfmt_alpha_blend_rgba<blender_sbgra32_plain, rendering_buffer> pixfmt_sbgra32_plain;

    // 定义基于 RGBA64 格式和指定混合器类型的像素格式类型
    typedef pixfmt_alpha_blend_rgba<blender_rgba64, rendering_buffer> pixfmt_rgba64;
    typedef pixfmt_alpha_blend_rgba<blender_argb64, rendering_buffer> pixfmt_argb64;
    typedef pixfmt_alpha_blend_rgba<blender_abgr64, rendering_buffer> pixfmt_abgr64;
    typedef pixfmt_alpha_blend_rgba<blender_bgra64, rendering_buffer> pixfmt_bgra64;
    # 定义基于不同混合模式和渲染缓冲区的 RGBA64 颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_rgba64_pre, rendering_buffer> pixfmt_rgba64_pre;
    # 定义基于不同混合模式和渲染缓冲区的 ARGB64 颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_argb64_pre, rendering_buffer> pixfmt_argb64_pre;
    # 定义基于不同混合模式和渲染缓冲区的 ABGR64 颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_abgr64_pre, rendering_buffer> pixfmt_abgr64_pre;
    # 定义基于不同混合模式和渲染缓冲区的 BGRA64 颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_bgra64_pre, rendering_buffer> pixfmt_bgra64_pre;
    
    # 定义基于不同混合模式和渲染缓冲区的 RGBA64 平面颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_rgba64_plain, rendering_buffer> pixfmt_rgba64_plain;
    # 定义基于不同混合模式和渲染缓冲区的 ARGB64 平面颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_argb64_plain, rendering_buffer> pixfmt_argb64_plain;
    # 定义基于不同混合模式和渲染缓冲区的 ABGR64 平面颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_abgr64_plain, rendering_buffer> pixfmt_abgr64_plain;
    # 定义基于不同混合模式和渲染缓冲区的 BGRA64 平面颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_bgra64_plain, rendering_buffer> pixfmt_bgra64_plain;
    
    # 定义基于不同混合模式和渲染缓冲区的 RGBA128 颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_rgba128, rendering_buffer> pixfmt_rgba128;
    # 定义基于不同混合模式和渲染缓冲区的 ARGB128 颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_argb128, rendering_buffer> pixfmt_argb128;
    # 定义基于不同混合模式和渲染缓冲区的 ABGR128 颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_abgr128, rendering_buffer> pixfmt_abgr128;
    # 定义基于不同混合模式和渲染缓冲区的 BGRA128 颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_bgra128, rendering_buffer> pixfmt_bgra128;
    
    # 定义基于不同混合模式和渲染缓冲区的 RGBA128 平面颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_rgba128_plain, rendering_buffer> pixfmt_rgba128_plain;
    # 定义基于不同混合模式和渲染缓冲区的 ARGB128 平面颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_argb128_plain, rendering_buffer> pixfmt_argb128_plain;
    # 定义基于不同混合模式和渲染缓冲区的 ABGR128 平面颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_abgr128_plain, rendering_buffer> pixfmt_abgr128_plain;
    # 定义基于不同混合模式和渲染缓冲区的 BGRA128 平面颜色格式处理器类型
    typedef pixfmt_alpha_blend_rgba<blender_bgra128_plain, rendering_buffer> pixfmt_bgra128_plain;
}


注释：

// 这是一个 C/C++ 的预处理器指令，用于结束一个条件编译块，与 `#ifdef` 或 `#ifndef` 配合使用。
// 在条件满足时（如宏已定义或未定义），编译器将包括在 `#ifdef` 或 `#ifndef` 之后的代码。
// 该行标志着条件编译块的结束。
#endif
```