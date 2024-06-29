# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_pixfmt_gray.h`

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
// Adaptation for high precision colors has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------

#ifndef AGG_PIXFMT_GRAY_INCLUDED
#define AGG_PIXFMT_GRAY_INCLUDED

#include <string.h>
#include "agg_pixfmt_base.h"
#include "agg_rendering_buffer.h"

namespace agg
{
 
    //============================================================blender_gray
    // 灰度颜色混合器模板类
    template<class ColorT> struct blender_gray
    {
        typedef ColorT color_type;                // 定义颜色类型
        typedef typename color_type::value_type value_type;     // 值类型
        typedef typename color_type::calc_type calc_type;       // 计算类型
        typedef typename color_type::long_type long_type;       // 长整型类型

        // 使用 Alvy-Ray Smith 的非预乘形式的混合函数来混合像素。
        // 因为渲染缓冲是不透明的，所以跳过初始预乘和最终去预乘的步骤。
        
        // 静态成员函数：blend_pix
        // 参数：
        //   - p: 指向像素值的指针
        //   - cv: 要混合的灰度值
        //   - alpha: 混合的透明度
        //   - cover: 覆盖度
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cv, value_type alpha, cover_type cover)
        {
            blend_pix(p, cv, color_type::mult_cover(alpha, cover));   // 调用另一个 blend_pix 函数进行实际的混合
        }

        // 静态成员函数：blend_pix
        // 参数：
        //   - p: 指向像素值的指针
        //   - cv: 要混合的灰度值
        //   - alpha: 混合的透明度
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cv, value_type alpha)
        {
            *p = color_type::lerp(*p, cv, alpha);     // 使用线性插值函数 lerp 进行混合操作
        }
    };


    //======================================================blender_gray_pre
    // 预乘灰度颜色混合器模板类
    template<class ColorT> struct blender_gray_pre
    {
        // 定义颜色类型为 ColorT
        typedef ColorT color_type;
        // 定义值类型为 color_type 的值类型
        typedef typename color_type::value_type value_type;
        // 定义计算类型为 color_type 的计算类型
        typedef typename color_type::calc_type calc_type;
        // 定义长整型类型为 color_type 的长整型类型
        typedef typename color_type::long_type long_type;
    
        // 使用 Alvy-Ray Smith 的预乘形式的混合像素的混合函数进行像素混合。
    
        // 像素混合函数，使用预乘形式的 Alvy-Ray Smith 的混合算法
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cv, value_type alpha, cover_type cover)
        {
            // 调用 blend_pix 函数，传入经过覆盖乘法后的 cv 和 alpha 值
            blend_pix(p, color_type::mult_cover(cv, cover), color_type::mult_cover(alpha, cover));
        }
    
        // 像素混合函数，使用预乘形式进行像素混合
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cv, value_type alpha)
        {
            // 对 p 指向的像素应用预乘形式的混合
            *p = color_type::prelerp(*p, cv, alpha);
        }
    };
    
    
    
    //=====================================================apply_gamma_dir_gray
    template<class ColorT, class GammaLut> class apply_gamma_dir_gray
    {
    public:
        // 定义值类型为 ColorT 的值类型
        typedef typename ColorT::value_type value_type;
    
        // 构造函数，接受 GammaLut 类型的 gamma 参数
        apply_gamma_dir_gray(const GammaLut& gamma) : m_gamma(gamma) {}
    
        // 操作符重载函数，对灰度值应用正向 gamma 校正
        AGG_INLINE void operator () (value_type* p)
        {
            // 使用 m_gamma 对 *p 进行正向 gamma 校正
            *p = m_gamma.dir(*p);
        }
    
    private:
        const GammaLut& m_gamma;
    };
    
    
    
    //=====================================================apply_gamma_inv_gray
    template<class ColorT, class GammaLut> class apply_gamma_inv_gray
    {
    public:
        // 定义值类型为 ColorT 的值类型
        typedef typename ColorT::value_type value_type;
    
        // 构造函数，接受 GammaLut 类型的 gamma 参数
        apply_gamma_inv_gray(const GammaLut& gamma) : m_gamma(gamma) {}
    
        // 操作符重载函数，对灰度值应用反向 gamma 校正
        AGG_INLINE void operator () (value_type* p)
        {
            // 使用 m_gamma 对 *p 进行反向 gamma 校正
            *p = m_gamma.inv(*p);
        }
    
    private:
        const GammaLut& m_gamma;
    };
    
    
    
    //=================================================pixfmt_alpha_blend_gray
    template<class Blender, class RenBuf, unsigned Step = 1, unsigned Offset = 0>
    class pixfmt_alpha_blend_gray
    {
    // 定义公共部分，以下是像素格式标签为灰度的定义
    public:
        typedef pixfmt_gray_tag pixfmt_category;
        // 定义渲染缓冲类型为 RenBuf
        typedef RenBuf   rbuf_type;
        // 定义渲染缓冲的行数据类型为 row_data
        typedef typename rbuf_type::row_data row_data;
        // 定义混合器类型为 Blender
        typedef Blender  blender_type;
        // 定义混合器的颜色类型为 color_type
        typedef typename blender_type::color_type color_type;
        // 定义一个假的排序类型
        typedef int                               order_type; // A fake one
        // 定义颜色值类型为 value_type
        typedef typename color_type::value_type   value_type;
        // 定义计算类型为 calc_type
        typedef typename color_type::calc_type    calc_type;
        
        // 像素的宽度，以字节为单位，为值类型大小乘以 Step
        enum 
        {
            pix_width = sizeof(value_type) * Step,
            // 像素步长为 Step
            pix_step = Step,
            // 像素偏移为 Offset
            pix_offset = Offset,
        };
        
        // 定义像素类型结构体
        struct pixel_type
        {
            // 像素值数组
            value_type c[pix_step];

            // 设置像素值为 v
            void set(value_type v)
            {
                c[0] = v;
            }

            // 设置像素值为 color 的值
            void set(const color_type& color)
            {
                set(color.v);
            }

            // 获取像素值存入 v 中
            void get(value_type& v) const
            {
                v = c[0];
            }

            // 获取像素的颜色值
            color_type get() const
            {
                return color_type(c[0]);
            }

            // 返回下一个像素的指针
            pixel_type* next()
            {
                return this + 1;
            }

            // 返回下一个像素的指针（常量成员函数版本）
            const pixel_type* next() const
            {
                return this + 1;
            }

            // 前进 n 步后的像素指针
            pixel_type* advance(int n)
            {
                return this + n;
            }

            // 前进 n 步后的像素指针（常量成员函数版本）
            const pixel_type* advance(int n) const
            {
                return this + n;
            }
        };
    private:
        //--------------------------------------------------------------------
        // 使用混合器类型对像素进行混合操作，带有覆盖参数
        AGG_INLINE void blend_pix(pixel_type* p, 
            value_type v, value_type a, 
            unsigned cover)
        {
            // 调用混合器的混合函数来处理像素
            blender_type::blend_pix(p->c, v, a, cover);
        }

        //--------------------------------------------------------------------
        // 使用混合器类型对像素进行混合操作，不带覆盖参数
        AGG_INLINE void blend_pix(pixel_type* p, value_type v, value_type a)
        {
            // 调用混合器的混合函数来处理像素
            blender_type::blend_pix(p->c, v, a);
        }

        //--------------------------------------------------------------------
        // 使用混合器类型对像素进行混合操作，带有颜色和覆盖参数
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c, unsigned cover)
        {
            // 调用混合器的混合函数来处理像素
            blender_type::blend_pix(p->c, c.v, c.a, cover);
        }

        //--------------------------------------------------------------------
        // 使用混合器类型对像素进行混合操作，带有颜色但不带覆盖参数
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c)
        {
            // 调用混合器的混合函数来处理像素
            blender_type::blend_pix(p->c, c.v, c.a);
        }

        //--------------------------------------------------------------------
        // 复制或混合像素，根据像素是否透明和是否不透明选择操作
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c, unsigned cover)
        {
            // 如果像素不透明
            if (!c.is_transparent())
            {
                // 如果像素完全不透明并且覆盖率达到最大
                if (c.is_opaque() && cover == cover_mask)
                {
                    // 直接设置像素值为给定颜色
                    p->set(c);
                }
                else
                {
                    // 否则使用混合器对像素进行混合操作
                    blend_pix(p, c, cover);
                }
            }
        }

        //--------------------------------------------------------------------
        // 复制或混合像素，根据像素是否透明选择操作
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c)
        {
            // 如果像素不透明
            if (!c.is_transparent())
            {
                // 如果像素完全不透明
                if (c.is_opaque())
                {
                    // 直接设置像素值为给定颜色
                    p->set(c);
                }
                else
                {
                    // 否则使用混合器对像素进行混合操作
                    blend_pix(p, c);
                }
            }
        }

    private:
        // 渲染缓冲区指针
        rbuf_type* m_rbuf;
    };

typedef blender_gray<gray8> blender_gray8;
typedef blender_gray<sgray8> blender_sgray8;
typedef blender_gray<gray16> blender_gray16;
typedef blender_gray<gray32> blender_gray32;

typedef blender_gray_pre<gray8> blender_gray8_pre;
typedef blender_gray_pre<sgray8> blender_sgray8_pre;
typedef blender_gray_pre<gray16> blender_gray16_pre;
typedef blender_gray_pre<gray32> blender_gray32_pre;

typedef pixfmt_alpha_blend_gray<blender_gray8, rendering_buffer> pixfmt_gray8;
typedef pixfmt_alpha_blend_gray<blender_sgray8, rendering_buffer> pixfmt_sgray8;
typedef pixfmt_alpha_blend_gray<blender_gray16, rendering_buffer> pixfmt_gray16;
typedef pixfmt_alpha_blend_gray<blender_gray32, rendering_buffer> pixfmt_gray32;

typedef pixfmt_alpha_blend_gray<blender_gray8_pre, rendering_buffer> pixfmt_gray8_pre;
typedef pixfmt_alpha_blend_gray<blender_sgray8_pre, rendering_buffer> pixfmt_sgray8_pre;
typedef pixfmt_alpha_blend_gray<blender_gray16_pre, rendering_buffer> pixfmt_gray16_pre;
    // 定义了名为 pixfmt_gray32_pre 的类型别名，使用了 pixfmt_alpha_blend_gray 模板，
    // 其中 blender_gray32_pre 是混合器，rendering_buffer 是渲染缓冲器类型的参数。
    typedef pixfmt_alpha_blend_gray<blender_gray32_pre, rendering_buffer> pixfmt_gray32_pre;
}


注释：


// 结束了一个预处理器指令块，用于条件编译的结束标志



#endif


注释：


// 如果定义了条件编译指令 #ifdef 或者 #ifndef，这是条件编译的结束标志
```