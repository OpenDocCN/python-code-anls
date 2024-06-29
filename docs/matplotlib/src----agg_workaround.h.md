# `D:\src\scipysrc\matplotlib\src\agg_workaround.h`

```
#ifndef MPL_AGG_WORKAROUND_H
#define MPL_AGG_WORKAROUND_H

#include "agg_pixfmt_rgba.h"

/**********************************************************************
 WORKAROUND: This class is to workaround a bug in Agg SVN where the
 blending of RGBA32 pixels does not preserve enough precision
*/

// 定义模板结构体 fixed_blender_rgba_pre，继承自 agg::conv_rgba_pre<ColorT, Order>
template<class ColorT, class Order>
struct fixed_blender_rgba_pre : agg::conv_rgba_pre<ColorT, Order>
{
    typedef ColorT color_type;  // 定义颜色类型
    typedef Order order_type;   // 定义顺序类型
    typedef typename color_type::value_type value_type;    // 定义值类型
    typedef typename color_type::calc_type calc_type;      // 定义计算类型
    typedef typename color_type::long_type long_type;      // 定义长整型类型
    enum base_scale_e
    {
        base_shift = color_type::base_shift,  // 枚举基础位移量
        base_mask  = color_type::base_mask    // 枚举基础掩码值
    };

    //--------------------------------------------------------------------
    // 使用给定的颜色、透明度和覆盖度混合像素
    static AGG_INLINE void blend_pix(value_type* p,
                                     value_type cr, value_type cg, value_type cb,
                                     value_type alpha, agg::cover_type cover)
    {
        blend_pix(p,
                  color_type::mult_cover(cr, cover),    // 混合红色通道
                  color_type::mult_cover(cg, cover),    // 混合绿色通道
                  color_type::mult_cover(cb, cover),    // 混合蓝色通道
                  color_type::mult_cover(alpha, cover)); // 混合透明度通道
    }

    //--------------------------------------------------------------------
    // 使用给定的颜色和透明度混合像素
    static AGG_INLINE void blend_pix(value_type* p,
                                     value_type cr, value_type cg, value_type cb,
                                     value_type alpha)
    {
        alpha = base_mask - alpha; // 计算补码透明度
        p[Order::R] = (value_type)(((p[Order::R] * alpha) >> base_shift) + cr); // 混合红色通道
        p[Order::G] = (value_type)(((p[Order::G] * alpha) >> base_shift) + cg); // 混合绿色通道
        p[Order::B] = (value_type)(((p[Order::B] * alpha) >> base_shift) + cb); // 混合蓝色通道
        p[Order::A] = (value_type)(base_mask - ((alpha * (base_mask - p[Order::A])) >> base_shift)); // 混合透明度通道
    }
};

// 定义模板结构体 fixed_blender_rgba_plain，继承自 agg::conv_rgba_plain<ColorT, Order>
template<class ColorT, class Order>
struct fixed_blender_rgba_plain : agg::conv_rgba_plain<ColorT, Order>
{
    typedef ColorT color_type;  // 定义颜色类型
    typedef Order order_type;   // 定义顺序类型
    typedef typename color_type::value_type value_type;    // 定义值类型
    typedef typename color_type::calc_type calc_type;      // 定义计算类型
    typedef typename color_type::long_type long_type;      // 定义长整型类型
    enum base_scale_e { base_shift = color_type::base_shift }; // 枚举基础位移量

    //--------------------------------------------------------------------
    // 使用给定的颜色、透明度和覆盖度混合像素
    static AGG_INLINE void blend_pix(value_type* p,
                                     value_type cr, value_type cg, value_type cb, value_type alpha, agg::cover_type cover)
    {
        blend_pix(p, cr, cg, cb, color_type::mult_cover(alpha, cover));
    }

    //--------------------------------------------------------------------
    // 使用给定的颜色和透明度混合像素
    static AGG_INLINE void blend_pix(value_type* p,
                                     value_type cr, value_type cg, value_type cb, value_type alpha)
    {
        // 如果 alpha 等于 0，则直接返回，不进行计算
        if(alpha == 0) return;
        
        // 获取当前像素 p 中 Order::A 对应的值，并存储到变量 a 中
        calc_type a = p[Order::A];
        
        // 计算调整后的红色分量 r，使用 p 中 Order::R 对应的值和 a 的乘积
        calc_type r = p[Order::R] * a;
        
        // 计算调整后的绿色分量 g，使用 p 中 Order::G 对应的值和 a 的乘积
        calc_type g = p[Order::G] * a;
        
        // 计算调整后的蓝色分量 b，使用 p 中 Order::B 对应的值和 a 的乘积
        calc_type b = p[Order::B] * a;
        
        // 计算新的 alpha 值，并按位左移 base_shift 位，再减去原 alpha 乘以 a 的结果
        a = ((alpha + a) << base_shift) - alpha * a;
        
        // 更新像素 p 中 Order::A 对应的值，转换为 value_type 类型
        p[Order::A] = (value_type)(a >> base_shift);
        
        // 更新像素 p 中 Order::R 对应的值，根据公式计算新的红色分量
        p[Order::R] = (value_type)((((cr << base_shift) - r) * alpha + (r << base_shift)) / a);
        
        // 更新像素 p 中 Order::G 对应的值，根据公式计算新的绿色分量
        p[Order::G] = (value_type)((((cg << base_shift) - g) * alpha + (g << base_shift)) / a);
        
        // 更新像素 p 中 Order::B 对应的值，根据公式计算新的蓝色分量
        p[Order::B] = (value_type)((((cb << base_shift) - b) * alpha + (b << base_shift)) / a);
    }
};

#endif



// 结束预处理指令的条件编译区块，对应于 #ifdef 或 #ifndef
};
// 结束当前的命名空间或类定义块
#endif
// 结束条件编译指令块
```