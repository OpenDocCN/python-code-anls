# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_pixfmt_rgb.h`

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

#ifndef AGG_PIXFMT_RGB_INCLUDED
#define AGG_PIXFMT_RGB_INCLUDED

#include <string.h>  // 包含字符串处理相关的标准头文件
#include "agg_pixfmt_base.h"  // 包含抽象基类的头文件
#include "agg_rendering_buffer.h"  // 包含渲染缓冲相关的头文件

namespace agg
{

    //=====================================================apply_gamma_dir_rgb
    // 模板类 apply_gamma_dir_rgb，用于执行 RGB 颜色的正向 gamma 校正
    template<class ColorT, class Order, class GammaLut> class apply_gamma_dir_rgb
    {
    public:
        typedef typename ColorT::value_type value_type;  // 定义数值类型

        // 构造函数，初始化 gamma 表
        apply_gamma_dir_rgb(const GammaLut& gamma) : m_gamma(gamma) {}

        // 重载操作符，执行正向 gamma 校正
        AGG_INLINE void operator () (value_type* p)
        {
            p[Order::R] = m_gamma.dir(p[Order::R]);  // 对红色通道执行正向 gamma 校正
            p[Order::G] = m_gamma.dir(p[Order::G]);  // 对绿色通道执行正向 gamma 校正
            p[Order::B] = m_gamma.dir(p[Order::B]);  // 对蓝色通道执行正向 gamma 校正
        }

    private:
        const GammaLut& m_gamma;  // gamma 表的常引用
    };



    //=====================================================apply_gamma_inv_rgb
    // 模板类 apply_gamma_inv_rgb，用于执行 RGB 颜色的反向 gamma 校正
    template<class ColorT, class Order, class GammaLut> class apply_gamma_inv_rgb
    {
    public:
        typedef typename ColorT::value_type value_type;  // 定义数值类型

        // 构造函数，初始化 gamma 表
        apply_gamma_inv_rgb(const GammaLut& gamma) : m_gamma(gamma) {}

        // 重载操作符，执行反向 gamma 校正
        AGG_INLINE void operator () (value_type* p)
        {
            p[Order::R] = m_gamma.inv(p[Order::R]);  // 对红色通道执行反向 gamma 校正
            p[Order::G] = m_gamma.inv(p[Order::G]);  // 对绿色通道执行反向 gamma 校正
            p[Order::B] = m_gamma.inv(p[Order::B]);  // 对蓝色通道执行反向 gamma 校正
        }

    private:
        const GammaLut& m_gamma;  // gamma 表的常引用
    };


    //=========================================================blender_rgb
    // 结构体 blender_rgb，用于 RGB 颜色的混合操作
    template<class ColorT, class Order> 
    struct blender_rgb
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

        // 使用 Alvy-Ray Smith 的非预乘形式混合像素的合成函数进行像素混合。
        // 由于渲染缓冲区是不透明的，我们跳过初始预乘和最终去预乘。

        //--------------------------------------------------------------------
        // 静态内联函数：混合像素
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha, cover_type cover)
        {
            // 调用另一个 blend_pix 函数，将 alpha 和 cover 乘法混合后的结果传递给它
            blend_pix(p, cr, cg, cb, color_type::mult_cover(alpha, cover));
        }
        
        //--------------------------------------------------------------------
        // 静态内联函数：混合像素
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha)
        {
            // 对每个颜色分量进行线性插值混合
            p[Order::R] = color_type::lerp(p[Order::R], cr, alpha);
            p[Order::G] = color_type::lerp(p[Order::G], cg, alpha);
            p[Order::B] = color_type::lerp(p[Order::B], cb, alpha);
        }
    };

    //======================================================blender_rgb_pre
    // RGB 预乘混合器
    template<class ColorT, class Order> 
    struct blender_rgb_pre
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        // 使用 Alvy-Ray Smith 的预乘形式混合像素的合成函数。

        //--------------------------------------------------------------------
        // 静态内联函数：预乘形式混合像素
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha, cover_type cover)
        {
            // 调用另一个 blend_pix 函数，将每个颜色分量乘以 cover 后传递给它
            blend_pix(p, 
                color_type::mult_cover(cr, cover), 
                color_type::mult_cover(cg, cover), 
                color_type::mult_cover(cb, cover), 
                color_type::mult_cover(alpha, cover));
        }

        //--------------------------------------------------------------------
        // 静态内联函数：预乘形式混合像素
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha)
        {
            // 对每个颜色分量进行预乘线性插值混合
            p[Order::R] = color_type::prelerp(p[Order::R], cr, alpha);
            p[Order::G] = color_type::prelerp(p[Order::G], cg, alpha);
            p[Order::B] = color_type::prelerp(p[Order::B], cb, alpha);
        }
    };

    //===================================================blender_rgb_gamma
    // RGB 伽马校正混合器，继承自 blender_base<ColorT, Order>
    template<class ColorT, class Order, class Gamma> 
    class blender_rgb_gamma : public blender_base<ColorT, Order>
    {
    // 公共部分开始
    public:
        // 定义颜色类型、顺序类型、伽马类型及其相关类型
        typedef ColorT color_type;
        typedef Order order_type;
        typedef Gamma gamma_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        //--------------------------------------------------------------------
        // 默认构造函数，初始化 m_gamma 为 nullptr
        blender_rgb_gamma() : m_gamma(0) {}

        // 设置伽马类型对象的方法
        void gamma(const gamma_type& g) { m_gamma = &g; }

        //--------------------------------------------------------------------
        // 内联函数，混合像素的方法
        AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha, cover_type cover)
        {
            // 调用下面的 blend_pix 方法，处理 alpha 和 cover
            blend_pix(p, cr, cg, cb, color_type::mult_cover(alpha, cover));
        }
        
        //--------------------------------------------------------------------
        // 内联函数，混合像素的方法
        AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha)
        {
            // 计算经过伽马校正后的当前像素的 R、G、B 分量
            calc_type r = m_gamma->dir(p[Order::R]);
            calc_type g = m_gamma->dir(p[Order::G]);
            calc_type b = m_gamma->dir(p[Order::B]);
            
            // 混合并应用 alpha 通道的颜色校正
            p[Order::R] = m_gamma->inv(color_type::downscale((m_gamma->dir(cr) - r) * alpha) + r);
            p[Order::G] = m_gamma->inv(color_type::downscale((m_gamma->dir(cg) - g) * alpha) + g);
            p[Order::B] = m_gamma->inv(color_type::downscale((m_gamma->dir(cb) - b) * alpha) + b);
        }
        
    private:
        // 私有成员变量，指向伽马类型对象的常量指针
        const gamma_type* m_gamma;
    };
    // 定义公共成员：
    public:
        // 像素格式类型为 RGB
        typedef pixfmt_rgb_tag pixfmt_category;
        // 渲染缓冲区类型为 RenBuf
        typedef RenBuf   rbuf_type;
        // 混合器类型为 Blender
        typedef Blender  blender_type;
        // 行数据类型为渲染缓冲区类型的行数据
        typedef typename rbuf_type::row_data row_data;
        // 颜色类型为混合器类型的颜色类型
        typedef typename blender_type::color_type color_type;
        // 像素顺序类型为混合器类型的顺序类型
        typedef typename blender_type::order_type order_type;
        // 像素值类型为颜色类型的值类型
        typedef typename color_type::value_type value_type;
        // 计算类型为颜色类型的计算类型
        typedef typename color_type::calc_type calc_type;

        // 枚举常量
        enum 
        {
            // 像素步长为 Step
            pix_step = Step,
            // 像素偏移为 Offset
            pix_offset = Offset,
            // 像素宽度为值类型大小乘以像素步长
            pix_width = sizeof(value_type) * pix_step
        };

        // 像素类型结构体
        struct pixel_type
        {
            // 像素通道数组
            value_type c[pix_step];

            // 设置像素值为指定的 RGB 颜色
            void set(value_type r, value_type g, value_type b)
            {
                c[order_type::R] = r;
                c[order_type::G] = g;
                c[order_type::B] = b;
            }

            // 根据给定的颜色类型设置像素值
            void set(const color_type& color)
            {
                set(color.r, color.g, color.b);
            }

            // 获取像素值的 RGB 分量
            void get(value_type& r, value_type& g, value_type& b) const
            {
                r = c[order_type::R];
                g = c[order_type::G];
                b = c[order_type::B];
            }

            // 获取像素值的完整颜色类型
            color_type get() const
            {
                return color_type(
                    c[order_type::R], 
                    c[order_type::G], 
                    c[order_type::B]);
            }

            // 返回下一个像素的指针
            pixel_type* next()
            {
                return this + 1;
            }

            // 返回下一个像素的常量指针
            const pixel_type* next() const
            {
                return this + 1;
            }

            // 返回当前像素向前移动 n 个像素后的指针
            pixel_type* advance(int n)
            {
                return this + n;
            }

            // 返回当前像素向前移动 n 个像素后的常量指针
            const pixel_type* advance(int n) const
            {
                return this + n;
            }
        };
        //--------------------------------------------------------------------
        // 使用指定的颜色和覆盖度混合像素数据
        AGG_INLINE void blend_pix(pixel_type* p, 
            value_type r, value_type g, value_type b, value_type a, 
            unsigned cover)
        {
            m_blender.blend_pix(p->c, r, g, b, a, cover);
        }

        //--------------------------------------------------------------------
        // 使用指定的颜色混合像素数据（不考虑覆盖度）
        AGG_INLINE void blend_pix(pixel_type* p, 
            value_type r, value_type g, value_type b, value_type a)
        {
            m_blender.blend_pix(p->c, r, g, b, a);
        }

        //--------------------------------------------------------------------
        // 使用指定的颜色和覆盖度混合像素数据（颜色由 color_type 对象提供）
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c, unsigned cover)
        {
            m_blender.blend_pix(p->c, c.r, c.g, c.b, c.a, cover);
        }

        //--------------------------------------------------------------------
        // 使用指定的颜色混合像素数据（颜色由 color_type 对象提供，不考虑覆盖度）
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c)
        {
            m_blender.blend_pix(p->c, c.r, c.g, c.b, c.a);
        }

        //--------------------------------------------------------------------
        // 复制或混合像素数据，根据颜色的透明度和覆盖度进行判断处理
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c, unsigned cover)
        {
            if (!c.is_transparent())  // 如果颜色不是全透明
            {
                if (c.is_opaque() && cover == cover_mask)  // 如果颜色是完全不透明且覆盖度为最大值
                {
                    p->set(c);  // 直接设置像素颜色为给定颜色
                }
                else
                {
                    blend_pix(p, c, cover);  // 否则进行颜色混合处理
                }
            }
        }

        //--------------------------------------------------------------------
        // 复制或混合像素数据，根据颜色的透明度进行判断处理
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c)
        {
            if (!c.is_transparent())  // 如果颜色不是全透明
            {
                if (c.is_opaque())  // 如果颜色是完全不透明
                {
                    p->set(c);  // 直接设置像素颜色为给定颜色
                }
                else
                {
                    blend_pix(p, c);  // 否则进行颜色混合处理
                }
            }
        }

    private:
        rbuf_type* m_rbuf;  // 渲染缓冲区指针
        Blender    m_blender;  // 混合器对象
    };
    
    //-----------------------------------------------------------------------
    typedef blender_rgb<rgba8, order_rgb> blender_rgb24;  // RGBA8 像素格式的 RGB 混合器
    typedef blender_rgb<rgba8, order_bgr> blender_bgr24;  // RGBA8 像素格式的 BGR 混合器
    typedef blender_rgb<srgba8, order_rgb> blender_srgb24;  // sRGBA8 像素格式的 RGB 混合器
    typedef blender_rgb<srgba8, order_bgr> blender_sbgr24;  // sRGBA8 像素格式的 BGR 混合器
    typedef blender_rgb<rgba16, order_rgb> blender_rgb48;  // RGBA16 像素格式的 RGB 混合器
    typedef blender_rgb<rgba16, order_bgr> blender_bgr48;  // RGBA16 像素格式的 BGR 混合器
    typedef blender_rgb<rgba32, order_rgb> blender_rgb96;  // RGBA32 像素格式的 RGB 混合器
    typedef blender_rgb<rgba32, order_bgr> blender_bgr96;  // RGBA32 像素格式的 BGR 混合器

    typedef blender_rgb_pre<rgba8, order_rgb> blender_rgb24_pre;  // 预乘后的 RGBA8 像素格式的 RGB 混合器
    typedef blender_rgb_pre<rgba8, order_bgr> blender_bgr24_pre;  // 预乘后的 RGBA8 像素格式的 BGR 混合器
    typedef blender_rgb_pre<srgba8, order_rgb> blender_srgb24_pre;  // 预乘后的 sRGBA8 像素格式的 RGB 混合器
    typedef blender_rgb_pre<srgba8, order_bgr> blender_sbgr24_pre;  // 预乘后的 sRGBA8 像素格式的 BGR 混合器
    typedef blender_rgb_pre<rgba16, order_rgb> blender_rgb48_pre;  // 预乘后的 RGBA16 像素格式的 RGB 混合器


这些注释描述了每个函数的作用，以及类型定义的用途和含义。
    typedef blender_rgb_pre<rgba16, order_bgr> blender_bgr48_pre;
    typedef blender_rgb_pre<rgba32, order_rgb> blender_rgb96_pre;
    typedef blender_rgb_pre<rgba32, order_bgr> blender_bgr96_pre;
    
    定义了三个模板类型别名，分别是不同像素格式和颜色顺序的预混合 RGB 混合器。
    
    
    typedef pixfmt_alpha_blend_rgb<blender_rgb24, rendering_buffer, 3> pixfmt_rgb24;
    typedef pixfmt_alpha_blend_rgb<blender_bgr24, rendering_buffer, 3> pixfmt_bgr24;
    typedef pixfmt_alpha_blend_rgb<blender_srgb24, rendering_buffer, 3> pixfmt_srgb24;
    typedef pixfmt_alpha_blend_rgb<blender_sbgr24, rendering_buffer, 3> pixfmt_sbgr24;
    typedef pixfmt_alpha_blend_rgb<blender_rgb48, rendering_buffer, 3> pixfmt_rgb48;
    typedef pixfmt_alpha_blend_rgb<blender_bgr48, rendering_buffer, 3> pixfmt_bgr48;
    typedef pixfmt_alpha_blend_rgb<blender_rgb96, rendering_buffer, 3> pixfmt_rgb96;
    typedef pixfmt_alpha_blend_rgb<blender_bgr96, rendering_buffer, 3> pixfmt_bgr96;
    
    定义了八个像素格式和混合方式的类型别名，使用了预定义的 RGB 混合器和渲染缓冲区。
    
    
    typedef pixfmt_alpha_blend_rgb<blender_rgb24_pre, rendering_buffer, 3> pixfmt_rgb24_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr24_pre, rendering_buffer, 3> pixfmt_bgr24_pre;
    typedef pixfmt_alpha_blend_rgb<blender_srgb24_pre, rendering_buffer, 3> pixfmt_srgb24_pre;
    typedef pixfmt_alpha_blend_rgb<blender_sbgr24_pre, rendering_buffer, 3> pixfmt_sbgr24_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb48_pre, rendering_buffer, 3> pixfmt_rgb48_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr48_pre, rendering_buffer, 3> pixfmt_bgr48_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb96_pre, rendering_buffer, 3> pixfmt_rgb96_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr96_pre, rendering_buffer, 3> pixfmt_bgr96_pre;
    
    定义了八个预混合的像素格式和混合方式的类型别名，使用了预定义的 RGB 混合器和渲染缓冲区。
    
    
    typedef pixfmt_alpha_blend_rgb<blender_rgb24, rendering_buffer, 4, 0> pixfmt_rgbx32;
    typedef pixfmt_alpha_blend_rgb<blender_rgb24, rendering_buffer, 4, 1> pixfmt_xrgb32;
    typedef pixfmt_alpha_blend_rgb<blender_bgr24, rendering_buffer, 4, 1> pixfmt_xbgr32;
    typedef pixfmt_alpha_blend_rgb<blender_bgr24, rendering_buffer, 4, 0> pixfmt_bgrx32;
    typedef pixfmt_alpha_blend_rgb<blender_srgb24, rendering_buffer, 4, 0> pixfmt_srgbx32;
    typedef pixfmt_alpha_blend_rgb<blender_srgb24, rendering_buffer, 4, 1> pixfmt_sxrgb32;
    typedef pixfmt_alpha_blend_rgb<blender_sbgr24, rendering_buffer, 4, 1> pixfmt_sxbgr32;
    typedef pixfmt_alpha_blend_rgb<blender_sbgr24, rendering_buffer, 4, 0> pixfmt_sbgrx32;
    typedef pixfmt_alpha_blend_rgb<blender_rgb48, rendering_buffer, 4, 0> pixfmt_rgbx64;
    typedef pixfmt_alpha_blend_rgb<blender_rgb48, rendering_buffer, 4, 1> pixfmt_xrgb64;
    typedef pixfmt_alpha_blend_rgb<blender_bgr48, rendering_buffer, 4, 1> pixfmt_xbgr64;
    typedef pixfmt_alpha_blend_rgb<blender_bgr48, rendering_buffer, 4, 0> pixfmt_bgrx64;
    typedef pixfmt_alpha_blend_rgb<blender_rgb96, rendering_buffer, 4, 0> pixfmt_rgbx128;
    typedef pixfmt_alpha_blend_rgb<blender_rgb96, rendering_buffer, 4, 1> pixfmt_xrgb128;
    typedef pixfmt_alpha_blend_rgb<blender_bgr96, rendering_buffer, 4, 1> pixfmt_xbgr128;
    
    定义了多种带有 alpha 通道混合方式的像素格式和类型别名，使用了不同的 RGB 混合器、渲染缓冲区和 alpha 通道配置。
    // 定义一个像素格式，用于将带有 alpha 混合的 RGB 像素渲染到渲染缓冲区，每个像素由4个字节组成，没有 alpha 通道
    typedef pixfmt_alpha_blend_rgb<blender_bgr96, rendering_buffer, 4, 0> pixfmt_bgrx128;

    // 定义不同的像素格式，每个格式都是带有 alpha 混合的 RGB 像素，用于不同的渲染缓冲区，每个像素由4个字节组成
    typedef pixfmt_alpha_blend_rgb<blender_rgb24_pre, rendering_buffer, 4, 0> pixfmt_rgbx32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb24_pre, rendering_buffer, 4, 1> pixfmt_xrgb32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr24_pre, rendering_buffer, 4, 1> pixfmt_xbgr32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr24_pre, rendering_buffer, 4, 0> pixfmt_bgrx32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_srgb24_pre, rendering_buffer, 4, 0> pixfmt_srgbx32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_srgb24_pre, rendering_buffer, 4, 1> pixfmt_sxrgb32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_sbgr24_pre, rendering_buffer, 4, 1> pixfmt_sxbgr32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_sbgr24_pre, rendering_buffer, 4, 0> pixfmt_sbgrx32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb48_pre, rendering_buffer, 4, 0> pixfmt_rgbx64_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb48_pre, rendering_buffer, 4, 1> pixfmt_xrgb64_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr48_pre, rendering_buffer, 4, 1> pixfmt_xbgr64_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr48_pre, rendering_buffer, 4, 0> pixfmt_bgrx64_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb96_pre, rendering_buffer, 4, 0> pixfmt_rgbx128_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb96_pre, rendering_buffer, 4, 1> pixfmt_xrgb128_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr96_pre, rendering_buffer, 4, 1> pixfmt_xbgr128_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr96_pre, rendering_buffer, 4, 0> pixfmt_bgrx128_pre;

    //-----------------------------------------------------pixfmt_rgb24_gamma
    // 使用 gamma 校正的 RGB24 像素格式，继承自带 alpha 混合的 RGB 像素格式，并指定每个像素由3个字节组成
    template<class Gamma> class pixfmt_rgb24_gamma : 
    public pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba8, order_rgb, Gamma>, rendering_buffer, 3>
    {
    public:
        // 构造函数，接受渲染缓冲区和 Gamma 校正器对象作为参数
        pixfmt_rgb24_gamma(rendering_buffer& rb, const Gamma& g) :
            // 调用基类构造函数初始化
            pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba8, order_rgb, Gamma>, rendering_buffer, 3>(rb) 
        {
            // 设置像素格式的 gamma 校正器
            this->blender().gamma(g);
        }
    };
        
    //-----------------------------------------------------pixfmt_srgb24_gamma
    // 使用 gamma 校正的 sRGB24 像素格式，继承自带 alpha 混合的 RGB 像素格式，并指定每个像素由3个字节组成
    template<class Gamma> class pixfmt_srgb24_gamma : 
    public pixfmt_alpha_blend_rgb<blender_rgb_gamma<srgba8, order_rgb, Gamma>, rendering_buffer, 3>
    {
    public:
        // 构造函数，接受渲染缓冲区和 Gamma 校正器对象作为参数
        pixfmt_srgb24_gamma(rendering_buffer& rb, const Gamma& g) :
            // 调用基类构造函数初始化
            pixfmt_alpha_blend_rgb<blender_rgb_gamma<srgba8, order_rgb, Gamma>, rendering_buffer, 3>(rb) 
        {
            // 设置像素格式的 gamma 校正器
            this->blender().gamma(g);
        }
    };
        
    //-----------------------------------------------------pixfmt_bgr24_gamma
    // 使用 gamma 校正的 BGR24 像素格式，继承自带 alpha 混合的 RGB 像素格式，并指定每个像素由3个字节组成
    template<class Gamma> class pixfmt_bgr24_gamma : 
    public pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba8, order_bgr, Gamma>, rendering_buffer, 3>
    {
    // 定义一个模板类 pixfmt_bgr24_gamma，继承自 pixfmt_alpha_blend_rgb 类模板，用于处理像素格式为 BGR24 且支持 Gamma 校正的情况
    template<class Gamma> class pixfmt_bgr24_gamma : 
    public pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba8, order_bgr, Gamma>, rendering_buffer, 3>
    {
    public:
        // 构造函数，接受渲染缓冲区和 Gamma 校正器作为参数，初始化基类 pixfmt_alpha_blend_rgb
        pixfmt_bgr24_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba8, order_bgr, Gamma>, rendering_buffer, 3>(rb) 
        {
            // 设置基类中的混合器（blender）使用给定的 Gamma 校正器
            this->blender().gamma(g);
        }
    };
    
    //-----------------------------------------------------pixfmt_sbgr24_gamma
    // 定义一个模板类 pixfmt_sbgr24_gamma，继承自 pixfmt_alpha_blend_rgb 类模板，用于处理像素格式为 sBGR24（srgba8）且支持 Gamma 校正的情况
    template<class Gamma> class pixfmt_sbgr24_gamma : 
    public pixfmt_alpha_blend_rgb<blender_rgb_gamma<srgba8, order_bgr, Gamma>, rendering_buffer, 3>
    {
    public:
        // 构造函数，接受渲染缓冲区和 Gamma 校正器作为参数，初始化基类 pixfmt_alpha_blend_rgb
        pixfmt_sbgr24_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb<blender_rgb_gamma<srgba8, order_bgr, Gamma>, rendering_buffer, 3>(rb) 
        {
            // 设置基类中的混合器（blender）使用给定的 Gamma 校正器
            this->blender().gamma(g);
        }
    };
    
    //-----------------------------------------------------pixfmt_rgb48_gamma
    // 定义一个模板类 pixfmt_rgb48_gamma，继承自 pixfmt_alpha_blend_rgb 类模板，用于处理像素格式为 RGB48 且支持 Gamma 校正的情况
    template<class Gamma> class pixfmt_rgb48_gamma : 
    public pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba16, order_rgb, Gamma>, rendering_buffer, 3>
    {
    public:
        // 构造函数，接受渲染缓冲区和 Gamma 校正器作为参数，初始化基类 pixfmt_alpha_blend_rgb
        pixfmt_rgb48_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba16, order_rgb, Gamma>, rendering_buffer, 3>(rb) 
        {
            // 设置基类中的混合器（blender）使用给定的 Gamma 校正器
            this->blender().gamma(g);
        }
    };
    
    //-----------------------------------------------------pixfmt_bgr48_gamma
    // 定义一个模板类 pixfmt_bgr48_gamma，继承自 pixfmt_alpha_blend_rgb 类模板，用于处理像素格式为 BGR48 且支持 Gamma 校正的情况
    template<class Gamma> class pixfmt_bgr48_gamma : 
    public pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba16, order_bgr, Gamma>, rendering_buffer, 3>
    {
    public:
        // 构造函数，接受渲染缓冲区和 Gamma 校正器作为参数，初始化基类 pixfmt_alpha_blend_rgb
        pixfmt_bgr48_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba16, order_bgr, Gamma>, rendering_buffer, 3>(rb) 
        {
            // 设置基类中的混合器（blender）使用给定的 Gamma 校正器
            this->blender().gamma(g);
        }
    };
}


注释：


// 这是 C/C++ 的预处理器指令，用于结束条件编译块



#endif


注释：


// 这是 C/C++ 的预处理器指令，用于结束条件编译指令的范围
```