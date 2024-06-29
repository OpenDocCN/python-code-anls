# `D:\src\scipysrc\matplotlib\src\_image_resample.h`

```
/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_RESAMPLE_H
#define MPL_RESAMPLE_H

#include "agg_image_accessors.h"
#include "agg_path_storage.h"
#include "agg_pixfmt_gray.h"
#include "agg_pixfmt_rgba.h"
#include "agg_renderer_base.h"
#include "agg_renderer_scanline.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_scanline_u.h"
#include "agg_span_allocator.h"
#include "agg_span_converter.h"
#include "agg_span_image_filter_gray.h"
#include "agg_span_image_filter_rgba.h"
#include "agg_span_interpolator_adaptor.h"
#include "agg_span_interpolator_linear.h"

#include "agg_workaround.h"

// Based on:

//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://antigrain.com/)
//
// Permission to copy, use, modify, sell and distribute this software
// is granted provided this copyright notice appears in all copies.
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://antigrain.com/
//----------------------------------------------------------------------------
//
// Adaptation for high precision colors has been sponsored by
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
//

//===================================================================gray64
namespace agg
{
    // 定义 gray64 结构体
    struct gray64
    {
    };


    //====================================================================rgba32
    // 定义 rgba64 结构体
    struct rgba64
    {
    };
}


// 定义插值方法的枚举类型
typedef enum {
    NEAREST,     // 最近邻插值
    BILINEAR,    // 双线性插值
    BICUBIC,     // 双三次插值
    SPLINE16,    // 16 点样条插值
    SPLINE36,    // 36 点样条插值
    HANNING,     // Hanning 插值
    HAMMING,     // Hamming 插值
    HERMITE,     // Hermite 插值
    KAISER,      // Kaiser 插值
    QUADRIC,     // Quadric 插值
    CATROM,      // Catmull-Rom 插值
    GAUSSIAN,    // Gaussian 插值
    BESSEL,      // Bessel 插值
    MITCHELL,    // Mitchell 插值
    SINC,        // Sinc 插值
    LANCZOS,     // Lanczos 插值
    BLACKMAN,    // Blackman 插值
} interpolation_e;


// T is rgba if and only if it has an T::r field.
// 定义用于检查颜色类型是否为灰度的模板
template<typename T, typename = void> struct is_grayscale : std::true_type {};
template<typename T> struct is_grayscale<T, decltype(T::r, void())> : std::false_type {};


// 定义颜色类型映射结构体
template<typename color_type>
struct type_mapping
{
    // 定义混合器类型
    using blender_type = typename std::conditional<
        is_grayscale<color_type>::value,  // 如果颜色类型为灰度
        agg::blender_gray<color_type>,   // 使用灰度混合器
        typename std::conditional<
            std::is_same<color_type, agg::rgba8>::value,  // 如果颜色类型为 rgba8
            fixed_blender_rgba_plain<color_type, agg::order_rgba>,  // 使用固定 rgba 混合器
            agg::blender_rgba_plain<color_type, agg::order_rgba>     // 否则使用普通 rgba 混合器
        >::type
    >::type;

    // 定义像素格式类型
    using pixfmt_type = typename std::conditional<
        is_grayscale<color_type>::value,  // 如果颜色类型为灰度
        agg::pixfmt_alpha_blend_gray<blender_type, agg::rendering_buffer>,  // 使用灰度像素格式
        agg::pixfmt_alpha_blend_rgba<blender_type, agg::rendering_buffer>   // 否则使用 rgba 像素格式
    >::type;
};

#endif // MPL_RESAMPLE_H
    // 定义一个类型别名 pixfmt_pre_type，根据颜色类型是否为灰度，选择不同的像素格式类型
    using pixfmt_pre_type = typename std::conditional<
        is_grayscale<color_type>::value,  // 如果颜色类型是灰度
        pixfmt_type,  // 使用普通像素格式类型
        agg::pixfmt_alpha_blend_rgba<  // 否则，使用带有 alpha 混合的 RGBA 像素格式类型
            typename std::conditional<
                std::is_same<color_type, agg::rgba8>::value,  // 如果颜色类型是 RGBA8
                fixed_blender_rgba_pre<color_type, agg::order_rgba>,  // 使用固定的 RGBA8 预混合器
                agg::blender_rgba_pre<color_type, agg::order_rgba>  // 否则，使用一般的 RGBA 预混合器
            >::type,
            agg::rendering_buffer>  // 渲染缓冲区类型
    >::type;
    // 定义一个模板类型别名 span_gen_affine_type，根据颜色类型是否为灰度，选择不同的仿射变换的跨度类型
    template<typename A> using span_gen_affine_type = typename std::conditional<
        is_grayscale<color_type>::value,  // 如果颜色类型是灰度
        agg::span_image_resample_gray_affine<A>,  // 使用灰度图像的仿射变换跨度类型
        agg::span_image_resample_rgba_affine<A>  // 否则，使用 RGBA 图像的仿射变换跨度类型
    >::type;
    // 定义一个模板类型别名 span_gen_filter_type，根据颜色类型是否为灰度，选择不同的图像过滤器的跨度类型
    template<typename A, typename B> using span_gen_filter_type = typename std::conditional<
        is_grayscale<color_type>::value,  // 如果颜色类型是灰度
        agg::span_image_filter_gray<A, B>,  // 使用灰度图像的过滤器跨度类型
        agg::span_image_filter_rgba<A, B>  // 否则，使用 RGBA 图像的过滤器跨度类型
    >::type;
    // 定义一个模板类型别名 span_gen_nn_type，根据颜色类型是否为灰度，选择不同的最近邻图像过滤器的跨度类型
    template<typename A, typename B> using span_gen_nn_type = typename std::conditional<
        is_grayscale<color_type>::value,  // 如果颜色类型是灰度
        agg::span_image_filter_gray_nn<A, B>,  // 使用灰度图像的最近邻过滤器跨度类型
        agg::span_image_filter_rgba_nn<A, B>  // 否则，使用 RGBA 图像的最近邻过滤器跨度类型
    >::type;
// 结束span_conv_alpha类定义

/* 用于处理带alpha通道的像素数据的类，根据给定的alpha值进行预处理和生成 */
template<typename color_type>
class span_conv_alpha
{
public:
    // 构造函数，初始化alpha值
    span_conv_alpha(const double alpha) :
        m_alpha(alpha)
    {
    }

    // 准备函数，当前为空实现
    void prepare() {}

    // 生成函数，根据alpha值对span中的像素进行处理，修改alpha通道值
    void generate(color_type* span, int x, int y, unsigned len) const
    {
        if (m_alpha != 1.0) {
            do {
                span->a *= m_alpha; // 修改像素的alpha通道值
                ++span;            // 指向下一个像素
            } while (--len);       // 继续直到处理完所有像素
        }
    }
private:
    const double m_alpha;  // 存储alpha值的常量成员变量
};


/* 用于使用查找表进行转换的类 */
class lookup_distortion
{
public:
    // 构造函数，初始化转换所需的参数
    lookup_distortion(const double *mesh, int in_width, int in_height,
                      int out_width, int out_height) :
        m_mesh(mesh),
        m_in_width(in_width),
        m_in_height(in_height),
        m_out_width(out_width),
        m_out_height(out_height)
    {}

    // 计算函数，根据给定的转换网格计算新的坐标
    void calculate(int* x, int* y) {
        if (m_mesh) {
            // 将像素坐标转换为浮点坐标
            double dx = double(*x) / agg::image_subpixel_scale;
            double dy = double(*y) / agg::image_subpixel_scale;
            // 检查坐标是否在有效范围内
            if (dx >= 0 && dx < m_out_width &&
                dy >= 0 && dy < m_out_height) {
                // 计算新的坐标位置
                const double *coord = m_mesh + (int(dy) * m_out_width + int(dx)) * 2;
                *x = int(coord[0] * agg::image_subpixel_scale); // 更新x坐标
                *y = int(coord[1] * agg::image_subpixel_scale); // 更新y坐标
            }
        }
    }

protected:
    const double *m_mesh;   // 存储转换网格的指针
    int m_in_width;         // 输入图像宽度
    int m_in_height;        // 输入图像高度
    int m_out_width;        // 输出图像宽度
    int m_out_height;       // 输出图像高度
};


// 结构体，存储重采样参数
struct resample_params_t {
    interpolation_e interpolation;   // 插值类型
    bool is_affine;                  // 是否为仿射变换
    agg::trans_affine affine;        // 仿射变换矩阵
    const double *transform_mesh;    // 变换网格
    bool resample;                   // 是否重采样
    bool norm;                       // 是否进行归一化
    double radius;                   // 半径
    double alpha;                    // alpha值
};


// 静态函数，根据给定的重采样参数设置滤波器
static void get_filter(const resample_params_t &params,
                       agg::image_filter_lut &filter)
{
    switch (params.interpolation) {
    case NEAREST:
        // 永远不应该执行到这里，仅用于消除编译器警告
        break;

    case HANNING:
        filter.calculate(agg::image_filter_hanning(), params.norm);  // 使用汉宁窗口函数设置滤波器
        break;

    case HAMMING:
        filter.calculate(agg::image_filter_hamming(), params.norm);  // 使用哈明窗口函数设置滤波器
        break;

    case HERMITE:
        filter.calculate(agg::image_filter_hermite(), params.norm);  // 使用Hermite插值设置滤波器
        break;

    case BILINEAR:
        filter.calculate(agg::image_filter_bilinear(), params.norm);  // 使用双线性插值设置滤波器
        break;

    case BICUBIC:
        filter.calculate(agg::image_filter_bicubic(), params.norm);   // 使用双三次插值设置滤波器
        break;

    case SPLINE16:
        filter.calculate(agg::image_filter_spline16(), params.norm);  // 使用16次样条插值设置滤波器
        break;

    case SPLINE36:
        filter.calculate(agg::image_filter_spline36(), params.norm);  // 使用36次样条插值设置滤波器
        break;

    case KAISER:
        filter.calculate(agg::image_filter_kaiser(), params.norm);    // 使用Kaiser窗口函数设置滤波器
        break;

    case QUADRIC:
        filter.calculate(agg::image_filter_quadric(), params.norm);   // 使用二次插值设置滤波器
        break;

    case CATROM:
        filter.calculate(agg::image_filter_catrom(), params.norm);    // 使用Catmull-Rom插值设置滤波器
        break;
    }
    // 对于 GAUSSIAN 情况，使用高斯滤波器计算滤波器参数，并应用到图像滤波器上
    case GAUSSIAN:
        filter.calculate(agg::image_filter_gaussian(), params.norm);
        break;

    // 对于 BESSEL 情况，使用贝塞尔滤波器计算滤波器参数，并应用到图像滤波器上
    case BESSEL:
        filter.calculate(agg::image_filter_bessel(), params.norm);
        break;

    // 对于 MITCHELL 情况，使用米切尔滤波器计算滤波器参数，并应用到图像滤波器上
    case MITCHELL:
        filter.calculate(agg::image_filter_mitchell(), params.norm);
        break;

    // 对于 SINC 情况，使用 SINC 滤波器计算滤波器参数，并应用到图像滤波器上，使用给定的半径参数
    case SINC:
        filter.calculate(agg::image_filter_sinc(params.radius), params.norm);
        break;

    // 对于 LANCZOS 情况，使用 LANCZOS 滤波器计算滤波器参数，并应用到图像滤波器上，使用给定的半径参数
    case LANCZOS:
        filter.calculate(agg::image_filter_lanczos(params.radius), params.norm);
        break;

    // 对于 BLACKMAN 情况，使用 BLACKMAN 滤波器计算滤波器参数，并应用到图像滤波器上，使用给定的半径参数
    case BLACKMAN:
        filter.calculate(agg::image_filter_blackman(params.radius), params.norm);
        break;
    // 模板函数定义，对输入图像进行重新采样
template<typename color_type>
void resample(
    // 输入参数: 输入图像数据指针、输入图像宽度、高度
    const void *input, int in_width, int in_height,
    // 输出参数: 输出图像数据指针、输出图像宽度、高度
    void *output, int out_width, int out_height,
    // 重采样参数的引用
    resample_params_t &params)
{
    // 定义类型映射
    using type_mapping_t = type_mapping<color_type>;

    // 定义输入像素格式和输出像素格式
    using input_pixfmt_t = typename type_mapping_t::pixfmt_type;
    using output_pixfmt_t = typename type_mapping_t::pixfmt_type;

    // 渲染器类型和扫描线光栅化器类型
    using renderer_t = agg::renderer_base<output_pixfmt_t>;
    using rasterizer_t = agg::rasterizer_scanline_aa<agg::rasterizer_sl_clip_dbl>;

    // 反射模式类型和图像访问器类型
    using reflect_t = agg::wrap_mode_reflect;
    using image_accessor_t = agg::image_accessor_wrap<input_pixfmt_t, reflect_t, reflect_t>;

    // 跨度分配器类型和 alpha 通道转换器类型
    using span_alloc_t = agg::span_allocator<color_type>;
    using span_conv_alpha_t = span_conv_alpha<color_type>;

    // 仿射插值器类型和任意插值器类型
    using affine_interpolator_t = agg::span_interpolator_linear<>;
    using arbitrary_interpolator_t =
        agg::span_interpolator_adaptor<agg::span_interpolator_linear<>, lookup_distortion>;

    // 计算每个像素的字节数
    size_t itemsize = sizeof(color_type);
    if (is_grayscale<color_type>::value) {
        itemsize /= 2;  // agg::grayXX 包含一个我们没有的 alpha 通道
    }

    // 如果插值类型不是最近邻且仿射变换参数符合条件，则设置为最近邻插值
    if (params.interpolation != NEAREST &&
        params.is_affine &&
        fabs(params.affine.sx) == 1.0 &&
        fabs(params.affine.sy) == 1.0 &&
        params.affine.shx == 0.0 &&
        params.affine.shy == 0.0) {
        params.interpolation = NEAREST;
    }

    // 分配跨度和扫描线光栅化器对象
    span_alloc_t span_alloc;
    rasterizer_t rasterizer;
    agg::scanline_u8 scanline;

    // 创建 alpha 通道转换对象
    span_conv_alpha_t conv_alpha(params.alpha);

    // 设置输入图像的渲染缓冲区和像素格式
    agg::rendering_buffer input_buffer;
    input_buffer.attach(
        (unsigned char *)input, in_width, in_height, in_width * itemsize);
    input_pixfmt_t input_pixfmt(input_buffer);
    image_accessor_t input_accessor(input_pixfmt);

    // 设置输出图像的渲染缓冲区和像素格式
    agg::rendering_buffer output_buffer;
    output_buffer.attach(
        (unsigned char *)output, out_width, out_height, out_width * itemsize);
    output_pixfmt_t output_pixfmt(output_buffer);
    renderer_t renderer(output_pixfmt);

    // 创建反转的仿射变换对象
    agg::trans_affine inverted = params.affine;
    inverted.invert();

    // 设置光栅化器的剪裁框
    rasterizer.clip_box(0, 0, out_width, out_height);

    // 创建路径存储对象
    agg::path_storage path;

    // 如果使用仿射变换，则创建矩形路径
    if (params.is_affine) {
        path.move_to(0, 0);
        path.line_to(in_width, 0);
        path.line_to(in_width, in_height);
        path.line_to(0, in_height);
        path.close_polygon();
        agg::conv_transform<agg::path_storage> rectangle(path, params.affine);
        rasterizer.add_path(rectangle);
    } else {  // 否则创建矩形路径
        path.move_to(0, 0);
        path.line_to(out_width, 0);
        path.line_to(out_width, out_height);
        path.line_to(0, out_height);
        path.close_polygon();
        rasterizer.add_path(path);
    }
    // 如果插值方式为最近邻
    if (params.interpolation == NEAREST) {
        // 如果使用仿射插值
        if (params.is_affine) {
            // 定义仿射插值器类型和对应的跨度生成器类型
            using span_gen_t = typename type_mapping_t::template span_gen_nn_type<image_accessor_t, affine_interpolator_t>;
            // 定义跨度转换器类型
            using span_conv_t = agg::span_converter<span_gen_t, span_conv_alpha_t>;
            // 定义最近邻渲染器类型
            using nn_renderer_t = agg::renderer_scanline_aa<renderer_t, span_alloc_t, span_conv_t>;
            
            // 创建仿射插值器对象
            affine_interpolator_t interpolator(inverted);
            // 创建跨度生成器对象
            span_gen_t span_gen(input_accessor, interpolator);
            // 创建跨度转换器对象
            span_conv_t span_conv(span_gen, conv_alpha);
            // 创建最近邻渲染器对象
            nn_renderer_t nn_renderer(renderer, span_alloc, span_conv);
            // 使用 AGG 库渲染扫描线
            agg::render_scanlines(rasterizer, scanline, nn_renderer);
        } else {
            // 定义任意插值器类型和对应的跨度生成器类型
            using span_gen_t = typename type_mapping_t::template span_gen_nn_type<image_accessor_t, arbitrary_interpolator_t>;
            // 定义跨度转换器类型
            using span_conv_t = agg::span_converter<span_gen_t, span_conv_alpha_t>;
            // 定义最近邻渲染器类型
            using nn_renderer_t = agg::renderer_scanline_aa<renderer_t, span_alloc_t, span_conv_t>;
            
            // 创建查找畸变对象
            lookup_distortion dist(
                params.transform_mesh, in_width, in_height, out_width, out_height);
            // 创建任意插值器对象
            arbitrary_interpolator_t interpolator(inverted, dist);
            // 创建跨度生成器对象
            span_gen_t span_gen(input_accessor, interpolator);
            // 创建跨度转换器对象
            span_conv_t span_conv(span_gen, conv_alpha);
            // 创建最近邻渲染器对象
            nn_renderer_t nn_renderer(renderer, span_alloc, span_conv);
            // 使用 AGG 库渲染扫描线
            agg::render_scanlines(rasterizer, scanline, nn_renderer);
        }
    }
    } else {
        // 定义图像滤镜对象
        agg::image_filter_lut filter;
        // 从参数中获取并设置滤镜
        get_filter(params, filter);

        // 如果参数指定了仿射变换并启用了重采样
        if (params.is_affine && params.resample) {
            // 定义仿射变换的数据类型生成器
            using span_gen_t = typename type_mapping_t::template span_gen_affine_type<image_accessor_t>;
            // 定义数据类型转换器，将生成的数据类型转换为带透明度的类型
            using span_conv_t = agg::span_converter<span_gen_t, span_conv_alpha_t>;
            // 定义抗锯齿扫描线渲染器
            using int_renderer_t = agg::renderer_scanline_aa<renderer_t, span_alloc_t, span_conv_t>;
            // 创建仿射插值器对象
            affine_interpolator_t interpolator(inverted);
            // 创建数据类型生成器对象
            span_gen_t span_gen(input_accessor, interpolator, filter);
            // 创建数据类型转换器对象
            span_conv_t span_conv(span_gen, conv_alpha);
            // 创建抗锯齿扫描线渲染器对象
            int_renderer_t int_renderer(renderer, span_alloc, span_conv);
            // 使用渲染器渲染扫描线
            agg::render_scanlines(rasterizer, scanline, int_renderer);
        } else {
            // 定义特定滤镜类型的数据类型生成器
            using span_gen_t = typename type_mapping_t::template span_gen_filter_type<image_accessor_t, arbitrary_interpolator_t>;
            // 定义数据类型转换器，将生成的数据类型转换为带透明度的类型
            using span_conv_t = agg::span_converter<span_gen_t, span_conv_alpha_t>;
            // 定义抗锯齿扫描线渲染器
            using int_renderer_t = agg::renderer_scanline_aa<renderer_t, span_alloc_t, span_conv_t>;
            // 创建查找失真对象
            lookup_distortion dist(params.transform_mesh, in_width, in_height, out_width, out_height);
            // 创建任意插值器对象
            arbitrary_interpolator_t interpolator(inverted, dist);
            // 创建数据类型生成器对象
            span_gen_t span_gen(input_accessor, interpolator, filter);
            // 创建数据类型转换器对象
            span_conv_t span_conv(span_gen, conv_alpha);
            // 创建抗锯齿扫描线渲染器对象
            int_renderer_t int_renderer(renderer, span_alloc, span_conv);
            // 使用渲染器渲染扫描线
            agg::render_scanlines(rasterizer, scanline, int_renderer);
        }
    }
}

#endif /* MPL_RESAMPLE_H */


注释：


// 结束了一个 C/C++ 的条件编译部分，关闭了一个预处理器的条件编译指令
#endif /* MPL_RESAMPLE_H */
// 关闭 #ifdef MPL_RESAMPLE_H 预处理条件，标志着条件编译块的结束


这段代码是在 C/C++ 中常见的条件编译指令结尾部分。`#endif` 用于关闭之前的 `#ifdef` 或 `#ifndef` 预处理指令，这些指令通常用来根据预定义的宏来选择性地包含或排除代码段。注释 `/* MPL_RESAMPLE_H */` 通常用于说明与 `#ifdef` 匹配的宏名称，以提供代码的可读性和维护性。
```