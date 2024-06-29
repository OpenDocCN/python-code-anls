# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_image_filter.h`

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
// Image transformations with filtering. Span generator base class
//
//----------------------------------------------------------------------------

#ifndef AGG_SPAN_IMAGE_FILTER_INCLUDED
#define AGG_SPAN_IMAGE_FILTER_INCLUDED

#include "agg_basics.h"                      // 引入基本的 AGG 函数和宏定义
#include "agg_image_filters.h"               // 引入 AGG 图像滤波器
#include "agg_span_interpolator_linear.h"    // 引入线性插值器的 span 类

namespace agg
{

    //-------------------------------------------------------span_image_filter
    template<class Source, class Interpolator> class span_image_filter
    {
    // 公共部分开始

    // 声明类型别名：源类型和插值器类型
    public:
        typedef Source source_type;
        typedef Interpolator interpolator_type;

        // 默认构造函数
        //--------------------------------------------------------------------
        span_image_filter() {}

        // 带参数的构造函数，初始化成员变量
        span_image_filter(source_type& src, 
                          interpolator_type& interpolator,
                          image_filter_lut* filter) : 
            m_src(&src),                          // 初始化源对象指针
            m_interpolator(&interpolator),        // 初始化插值器对象指针
            m_filter(filter),                     // 初始化滤波器对象指针
            m_dx_dbl(0.5),                        // 双精度水平滤波偏移量初始化
            m_dy_dbl(0.5),                        // 双精度垂直滤波偏移量初始化
            m_dx_int(image_subpixel_scale / 2),   // 整数型水平滤波偏移量初始化
            m_dy_int(image_subpixel_scale / 2)    // 整数型垂直滤波偏移量初始化
        {}

        // 设置源对象的方法
        void attach(source_type& v) { m_src = &v; }

        //--------------------------------------------------------------------
        // 获取源对象的方法（可变版本）
        source_type& source()            { return *m_src; }
        // 获取源对象的方法（常量版本）
        const source_type& source() const { return *m_src; }
        // 获取滤波器对象的方法（常量版本）
        const image_filter_lut& filter() const { return *m_filter; }
        // 获取水平滤波偏移量的整数部分的方法
        int filter_dx_int() const { return m_dx_int; }
        // 获取垂直滤波偏移量的整数部分的方法
        int filter_dy_int() const { return m_dy_int; }
        // 获取水平滤波偏移量的双精度版本的方法
        double filter_dx_dbl() const { return m_dx_dbl; }
        // 获取垂直滤波偏移量的双精度版本的方法
        double filter_dy_dbl() const { return m_dy_dbl; }

        //--------------------------------------------------------------------
        // 设置插值器对象的方法
        void interpolator(interpolator_type& v) { m_interpolator = &v; }
        // 设置滤波器对象的方法
        void filter(image_filter_lut& v) { m_filter = &v; }
        
        // 设置滤波偏移量的方法，根据给定的双精度偏移量计算整数偏移量
        void filter_offset(double dx, double dy)
        {
            m_dx_dbl = dx;                              // 设置双精度水平滤波偏移量
            m_dy_dbl = dy;                              // 设置双精度垂直滤波偏移量
            m_dx_int = iround(dx * image_subpixel_scale);   // 计算并设置整数型水平滤波偏移量
            m_dy_int = iround(dy * image_subpixel_scale);   // 计算并设置整数型垂直滤波偏移量
        }

        // 设置滤波偏移量的方法，水平和垂直偏移量相同
        void filter_offset(double d) { filter_offset(d, d); }

        //--------------------------------------------------------------------
        // 获取插值器对象的方法（可变版本）
        interpolator_type& interpolator() { return *m_interpolator; }

        //--------------------------------------------------------------------
        // 准备方法，空实现
        void prepare() {}

        //--------------------------------------------------------------------
    // 私有部分开始

    private:
        source_type*      m_src;         // 源对象指针
        interpolator_type* m_interpolator;  // 插值器对象指针
        image_filter_lut*  m_filter;     // 滤波器对象指针
        double   m_dx_dbl;               // 双精度水平滤波偏移量
        double   m_dy_dbl;               // 双精度垂直滤波偏移量
        unsigned m_dx_int;               // 整数型水平滤波偏移量
        unsigned m_dy_int;               // 整数型垂直滤波偏移量
    };
    //=====================================================span_image_resample_affine
    // span_image_resample_affine 类的定义，继承自 span_image_filter<Source, interpolator_type>
    template<class Source, class Interpolator> 
    class span_image_resample_affine : 
    public span_image_filter<Source, Interpolator>
    {
    public:
        // 声明类型别名
        typedef Source source_type;
        typedef span_interpolator_linear<trans_affine> interpolator_type;
        typedef span_image_filter<source_type, interpolator_type> base_type;

        //--------------------------------------------------------------------
        // 默认构造函数，初始化成员变量 m_scale_limit, m_blur_x, m_blur_y
        span_image_resample_affine() : 
            m_scale_limit(200.0),
            m_blur_x(1.0),
            m_blur_y(1.0)
        {}

        //--------------------------------------------------------------------
        // 构造函数，初始化基类 base_type 和成员变量 m_scale_limit, m_blur_x, m_blur_y
        span_image_resample_affine(source_type& src, 
                                   interpolator_type& inter,
                                   image_filter_lut& filter) :
            base_type(src, inter, &filter),
            m_scale_limit(200.0),
            m_blur_x(1.0),
            m_blur_y(1.0)
        {}

        //--------------------------------------------------------------------
        // 返回 m_scale_limit 的整数值
        int  scale_limit() const { return uround(m_scale_limit); }
        // 设置 m_scale_limit 的值
        void scale_limit(int v)  { m_scale_limit = v; }

        //--------------------------------------------------------------------
        // 返回 m_blur_x 的值
        double blur_x() const { return m_blur_x; }
        // 返回 m_blur_y 的值
        double blur_y() const { return m_blur_y; }
        // 设置 m_blur_x 的值
        void blur_x(double v) { m_blur_x = v; }
        // 设置 m_blur_y 的值
        void blur_y(double v) { m_blur_y = v; }
        // 同时设置 m_blur_x 和 m_blur_y 的值
        void blur(double v) { m_blur_x = m_blur_y = v; }

        //--------------------------------------------------------------------
        // 准备函数，计算缩放比例，并根据 m_scale_limit 和 m_blur_x, m_blur_y 进行调整
        void prepare() 
        {
            double scale_x;
            double scale_y;

            // 获取插值器的绝对缩放比例
            base_type::interpolator().transformer().scaling_abs(&scale_x, &scale_y);

            // 如果缩放比例乘积超过 m_scale_limit，则按比例缩小
            if(scale_x * scale_y > m_scale_limit)
            {
                scale_x = scale_x * m_scale_limit / (scale_x * scale_y);
                scale_y = scale_y * m_scale_limit / (scale_x * scale_y);
            }

            // 缩放比例不能小于 1
            if(scale_x < 1) scale_x = 1;
            if(scale_y < 1) scale_y = 1;

            // 缩放比例不能超过 m_scale_limit
            if(scale_x > m_scale_limit) scale_x = m_scale_limit;
            if(scale_y > m_scale_limit) scale_y = m_scale_limit;

            // 根据 m_blur_x 和 m_blur_y 调整缩放比例
            scale_x *= m_blur_x;
            scale_y *= m_blur_y;

            // 缩放比例不能小于 1
            if(scale_x < 1) scale_x = 1;
            if(scale_y < 1) scale_y = 1;

            // 计算最终的缩放像素值，并进行四舍五入
            m_rx     = uround(    scale_x * double(image_subpixel_scale));
            m_rx_inv = uround(1.0/scale_x * double(image_subpixel_scale));

            m_ry     = uround(    scale_y * double(image_subpixel_scale));
            m_ry_inv = uround(1.0/scale_y * double(image_subpixel_scale));
        }

    protected:
        // 像素缩放值和其倒数，用于插值
        int m_rx;
        int m_ry;
        int m_rx_inv;
        int m_ry_inv;

    private:
        // 缩放和模糊限制参数
        double m_scale_limit;
        double m_blur_x;
        double m_blur_y;
    };
    // 公共部分：
    // 定义别名，分别表示源类型和插值器类型，基类为 span_image_filter
    typedef Source source_type;
    typedef Interpolator interpolator_type;
    typedef span_image_filter<source_type, interpolator_type> base_type;

    //--------------------------------------------------------------------
    // 默认构造函数，初始化 m_scale_limit 为 20，模糊参数为 image_subpixel_scale
    span_image_resample() : 
        m_scale_limit(20),
        m_blur_x(image_subpixel_scale),
        m_blur_y(image_subpixel_scale)
    {}

    //--------------------------------------------------------------------
    // 构造函数，接受源对象、插值器和滤波器的引用，并初始化基类和成员变量
    span_image_resample(source_type& src, 
                        interpolator_type& inter,
                        image_filter_lut& filter) :
        base_type(src, inter, &filter),
        m_scale_limit(20),
        m_blur_x(image_subpixel_scale),
        m_blur_y(image_subpixel_scale)
    {}

    //--------------------------------------------------------------------
    // 返回当前的尺度限制 m_scale_limit
    int scale_limit() const { return m_scale_limit; }
    
    // 设置尺度限制为 v
    void scale_limit(int v)  { m_scale_limit = v; }

    //--------------------------------------------------------------------
    // 返回水平方向的模糊值（以像素为单位）
    double blur_x() const { return double(m_blur_x) / double(image_subpixel_scale); }
    
    // 返回垂直方向的模糊值（以像素为单位）
    double blur_y() const { return double(m_blur_y) / double(image_subpixel_scale); }
    
    // 设置水平方向的模糊值（以像素为单位）
    void blur_x(double v) { m_blur_x = uround(v * double(image_subpixel_scale)); }
    
    // 设置垂直方向的模糊值（以像素为单位）
    void blur_y(double v) { m_blur_y = uround(v * double(image_subpixel_scale)); }
    
    // 同时设置水平和垂直方向的模糊值（以像素为单位）
    void blur(double v)   { m_blur_x = 
                            m_blur_y = uround(v * double(image_subpixel_scale)); }

protected:
    //--------------------------------------------------------------------
    // 内联函数，用于调整尺度参数，确保在合理范围内并应用模糊效果
    AGG_INLINE void adjust_scale(int* rx, int* ry)
    {
        // 确保尺度不小于 image_subpixel_scale
        if(*rx < image_subpixel_scale) *rx = image_subpixel_scale;
        if(*ry < image_subpixel_scale) *ry = image_subpixel_scale;
        
        // 确保尺度不超过 image_subpixel_scale * m_scale_limit
        if(*rx > image_subpixel_scale * m_scale_limit) 
        {
            *rx = image_subpixel_scale * m_scale_limit;
        }
        if(*ry > image_subpixel_scale * m_scale_limit) 
        {
            *ry = image_subpixel_scale * m_scale_limit;
        }
        
        // 应用模糊效果
        *rx = (*rx * m_blur_x) >> image_subpixel_shift;
        *ry = (*ry * m_blur_y) >> image_subpixel_shift;
        
        // 再次确保尺度不小于 image_subpixel_scale
        if(*rx < image_subpixel_scale) *rx = image_subpixel_scale;
        if(*ry < image_subpixel_scale) *ry = image_subpixel_scale;
    }

    int m_scale_limit;   // 尺度限制
    int m_blur_x;        // 水平方向模糊值（以 image_subpixel_scale 为单位）
    int m_blur_y;        // 垂直方向模糊值（以 image_subpixel_scale 为单位）
};
}



#endif



// 这里是 C/C++ 预处理器条件编译的结束部分，用于结束一个条件编译块
```