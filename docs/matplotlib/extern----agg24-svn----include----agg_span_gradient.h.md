# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_gradient.h`

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

#ifndef AGG_SPAN_GRADIENT_INCLUDED
#define AGG_SPAN_GRADIENT_INCLUDED

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "agg_basics.h"
#include "agg_math.h"
#include "agg_array.h"

// 命名空间agg，定义了一些渐变相关的类和枚举
namespace agg
{

    // 渐变子像素处理的枚举类型
    enum gradient_subpixel_scale_e
    {
        gradient_subpixel_shift = 4,                              // 渐变子像素位移量
        gradient_subpixel_scale = 1 << gradient_subpixel_shift,    // 渐变子像素比例
        gradient_subpixel_mask  = gradient_subpixel_scale - 1      // 渐变子像素掩码
    };

    //==========================================================span_gradient
    // span_gradient类模板，用于生成渐变色的跨距（span）
    template<class ColorT,
             class Interpolator,
             class GradientF, 
             class ColorF>
    class span_gradient
    {
    // 定义公共成员类型别名：插值器类型和颜色类型
    public:
        typedef Interpolator interpolator_type;
        typedef ColorT color_type;

        // 定义枚举类型，表示降采样的位移量
        enum downscale_shift_e
        {
            // 计算降采样位移量，使用插值器的子像素位移减去梯度子像素位移
            downscale_shift = interpolator_type::subpixel_shift - 
                              gradient_subpixel_shift
        };

        //--------------------------------------------------------------------
        // 默认构造函数
        span_gradient() {}

        //--------------------------------------------------------------------
        // 构造函数，初始化成员变量
        span_gradient(interpolator_type& inter,
                      GradientF& gradient_function,
                      ColorF& color_function,
                      double d1, double d2) : 
            m_interpolator(&inter),
            m_gradient_function(&gradient_function),
            m_color_function(&color_function),
            // 将浮点数转换为整数，乘以梯度子像素比例，存储到成员变量 m_d1 和 m_d2
            m_d1(iround(d1 * gradient_subpixel_scale)),
            m_d2(iround(d2 * gradient_subpixel_scale))
        {}

        //--------------------------------------------------------------------
        // 返回插值器对象的引用
        interpolator_type& interpolator() { return *m_interpolator; }
        // 返回梯度函数对象的常量引用
        const GradientF& gradient_function() const { return *m_gradient_function; }
        // 返回颜色函数对象的常量引用
        const ColorF& color_function() const { return *m_color_function; }
        // 返回经过比例转换的 m_d1 值
        double d1() const { return double(m_d1) / gradient_subpixel_scale; }
        // 返回经过比例转换的 m_d2 值
        double d2() const { return double(m_d2) / gradient_subpixel_scale; }

        //--------------------------------------------------------------------
        // 设置插值器对象的新引用
        void interpolator(interpolator_type& i) { m_interpolator = &i; }
        // 设置梯度函数对象的新引用
        void gradient_function(GradientF& gf) { m_gradient_function = &gf; }
        // 设置颜色函数对象的新引用
        void color_function(ColorF& cf) { m_color_function = &cf; }
        // 设置经过比例转换的 m_d1 值
        void d1(double v) { m_d1 = iround(v * gradient_subpixel_scale); }
        // 设置经过比例转换的 m_d2 值
        void d2(double v) { m_d2 = iround(v * gradient_subpixel_scale); }

        //--------------------------------------------------------------------
        // 准备方法，当前没有实现任何操作
        void prepare() {}

        //--------------------------------------------------------------------
        // 生成方法，用插值器生成颜色跨度
        void generate(color_type* span, int x, int y, unsigned len)
        {   
            // 计算 d2 和 d1 的差值
            int dd = m_d2 - m_d1;
            if(dd < 1) dd = 1;
            // 使用插值器开始生成颜色跨度
            m_interpolator->begin(x+0.5, y+0.5, len);
            do
            {
                // 获取当前插值器的坐标
                m_interpolator->coordinates(&x, &y);
                // 计算梯度函数的值，右移 downscale_shift 位
                int d = m_gradient_function->calculate(x >> downscale_shift, 
                                                       y >> downscale_shift, m_d2);
                // 根据计算的梯度值调整颜色函数的索引
                d = ((d - m_d1) * (int)m_color_function->size()) / dd;
                if(d < 0) d = 0;
                if(d >= (int)m_color_function->size()) d = m_color_function->size() - 1;
                // 将计算出的颜色写入 span 数组
                *span++ = (*m_color_function)[d];
                // 递增插值器以处理下一个位置
                ++(*m_interpolator);
            }
            while(--len);
        }

    private:
        // 插值器类型指针
        interpolator_type* m_interpolator;
        // 梯度函数指针
        GradientF*         m_gradient_function;
        // 颜色函数指针
        ColorF*            m_color_function;
        // 梯度范围的整数表示，经过梯度子像素比例缩放
        int                m_d1;
        int                m_d2;
    //=====================================================gradient_linear_color
    template<class ColorT> 
    struct gradient_linear_color
    {
        typedef ColorT color_type;
    
        gradient_linear_color() {}
        gradient_linear_color(const color_type& c1, const color_type& c2, 
                              unsigned size = 256) :
            m_c1(c1), m_c2(c2), m_size(size)
                // VFALCO 4/28/09
                ,m_mult(1/(double(size)-1))
                // VFALCO
            {}
    
        // 返回渐变色条的尺寸
        unsigned size() const { return m_size; }
        
        // 返回指定位置的颜色
        color_type operator [] (unsigned v) const 
        {
            // VFALCO 4/28/09 
            //return m_c1.gradient(m_c2, double(v) / double(m_size - 1));
            return m_c1.gradient(m_c2, double(v) * m_mult );
            // VFALCO
        }
    
        // 设置渐变色条的两个颜色和尺寸
        void colors(const color_type& c1, const color_type& c2, unsigned size = 256)
        {
            m_c1 = c1;
            m_c2 = c2;
            m_size = size;
            // VFALCO 4/28/09
            // 计算倍数以便优化颜色插值
            m_mult=1/(double(size)-1);
            // VFALCO
        }
    
        color_type m_c1;     // 渐变起始颜色
        color_type m_c2;     // 渐变结束颜色
        unsigned m_size;     // 渐变条尺寸
        // VFALCO 4/28/09
        double m_mult;      // 尺寸倍数，用于优化颜色插值
        // VFALCO
    };
    
    //==========================================================gradient_circle
    class gradient_circle
    {
        // 实际上与radial相同，只是为了兼容性而已
    public:
        // 计算圆形渐变的值
        static AGG_INLINE int calculate(int x, int y, int)
        {
            return int(fast_sqrt(x*x + y*y));
        }
    };
    
    
    //==========================================================gradient_radial
    class gradient_radial
    {
    public:
        // 计算径向渐变的值
        static AGG_INLINE int calculate(int x, int y, int)
        {
            return int(fast_sqrt(x*x + y*y));
        }
    };
    
    //========================================================gradient_radial_d
    class gradient_radial_d
    {
    public:
        // 计算双精度径向渐变的值
        static AGG_INLINE int calculate(int x, int y, int)
        {
            return uround(sqrt(double(x)*double(x) + double(y)*double(y)));
        }
    };
    
    //====================================================gradient_radial_focus
    class gradient_radial_focus
    {
    public:
        //---------------------------------------------------------------------
        // 默认构造函数，初始化半径为100 * gradient_subpixel_scale，焦点坐标为(0, 0)
        gradient_radial_focus() : 
            m_r(100 * gradient_subpixel_scale), 
            m_fx(0), 
            m_fy(0)
        {
            // 更新计算值
            update_values();
        }

        //---------------------------------------------------------------------
        // 带参数的构造函数，初始化半径为r * gradient_subpixel_scale，焦点坐标为(fx, fy)
        gradient_radial_focus(double r, double fx, double fy) : 
            m_r (iround(r  * gradient_subpixel_scale)), 
            m_fx(iround(fx * gradient_subpixel_scale)), 
            m_fy(iround(fy * gradient_subpixel_scale))
        {
            // 更新计算值
            update_values();
        }

        //---------------------------------------------------------------------
        // 初始化函数，设置半径为r * gradient_subpixel_scale，焦点坐标为(fx, fy)
        void init(double r, double fx, double fy)
        {
            m_r  = iround(r  * gradient_subpixel_scale);
            m_fx = iround(fx * gradient_subpixel_scale);
            m_fy = iround(fy * gradient_subpixel_scale);
            // 更新计算值
            update_values();
        }

        //---------------------------------------------------------------------
        // 返回半径的实际值（除以gradient_subpixel_scale）
        double radius()  const { return double(m_r)  / gradient_subpixel_scale; }
        
        // 返回焦点x坐标的实际值（除以gradient_subpixel_scale）
        double focus_x() const { return double(m_fx) / gradient_subpixel_scale; }
        
        // 返回焦点y坐标的实际值（除以gradient_subpixel_scale）
        double focus_y() const { return double(m_fy) / gradient_subpixel_scale; }

        //---------------------------------------------------------------------
        // 计算梯度的函数，返回计算结果的整数值
        int calculate(int x, int y, int) const
        {
            double dx = x - m_fx;
            double dy = y - m_fy;
            double d2 = dx * m_fy - dy * m_fx;
            double d3 = m_r2 * (dx * dx + dy * dy) - d2 * d2;
            // 返回计算结果的四舍五入值
            return iround((dx * m_fx + dy * m_fy + sqrt(fabs(d3))) * m_mul);
        }

    private:
        //---------------------------------------------------------------------
        // 更新计算值的私有函数
        void update_values()
        {
            // 计算不变值。如果焦点恰好在梯度圆上，则分母为零。
            // 在这种情况下，将焦点移动一个亚像素单位，可能朝向原点(0,0)的方向，并重新计算值。
            m_r2  = double(m_r)  * double(m_r);
            m_fx2 = double(m_fx) * double(m_fx);
            m_fy2 = double(m_fy) * double(m_fy);
            double d = (m_r2 - (m_fx2 + m_fy2));
            if(d == 0)
            {
                if(m_fx) { if(m_fx < 0) ++m_fx; else --m_fx; }
                if(m_fy) { if(m_fy < 0) ++m_fy; else --m_fy; }
                m_fx2 = double(m_fx) * double(m_fx);
                m_fy2 = double(m_fy) * double(m_fy);
                d = (m_r2 - (m_fx2 + m_fy2));
            }
            // 计算乘数值
            m_mul = m_r / d;
        }

        int    m_r;     // 半径的整数值
        int    m_fx;    // 焦点x坐标的整数值
        int    m_fy;    // 焦点y坐标的整数值
        double m_r2;    // 半径的平方
        double m_fx2;   // 焦点x坐标的平方
        double m_fy2;   // 焦点y坐标的平方
        double m_mul;   // 计算用的乘数值
    };
    //==============================================================gradient_x
    class gradient_x
    {
    public:
        // 返回 x 的值作为梯度计算结果
        static int calculate(int x, int, int) { return x; }
    };
    
    
    //==============================================================gradient_y
    class gradient_y
    {
    public:
        // 返回 y 的值作为梯度计算结果
        static int calculate(int, int y, int) { return y; }
    };
    
    //========================================================gradient_diamond
    class gradient_diamond
    {
    public:
        // 计算钻石形梯度，返回 x 和 y 的绝对值中较大的一个
        static AGG_INLINE int calculate(int x, int y, int) 
        { 
            int ax = abs(x);
            int ay = abs(y);
            return ax > ay ? ax : ay; 
        }
    };
    
    //=============================================================gradient_xy
    class gradient_xy
    {
    public:
        // 计算 x 和 y 的绝对值乘积再除以 d，作为梯度计算结果
        static AGG_INLINE int calculate(int x, int y, int d) 
        { 
            return abs(x) * abs(y) / d; 
        }
    };
    
    //========================================================gradient_sqrt_xy
    class gradient_sqrt_xy
    {
    public:
        // 计算 x 和 y 的绝对值乘积的平方根，作为梯度计算结果
        static AGG_INLINE int calculate(int x, int y, int) 
        { 
            return fast_sqrt(abs(x) * abs(y)); 
        }
    };
    
    //==========================================================gradient_conic
    class gradient_conic
    {
    public:
        // 计算 x 和 y 的反正切值的绝对值乘以 d 除以 pi 的结果，作为梯度计算结果
        static AGG_INLINE int calculate(int x, int y, int d) 
        { 
            return uround(fabs(atan2(double(y), double(x))) * double(d) / pi);
        }
    };
    
    //=================================================gradient_repeat_adaptor
    template<class GradientF> class gradient_repeat_adaptor
    {
    public:
        // 使用给定的梯度函数对象初始化重复梯度适配器
        gradient_repeat_adaptor(const GradientF& gradient) : 
            m_gradient(&gradient) {}
    
        // 调用内部梯度函数对象的 calculate 方法，并对结果取模 d，保证结果非负
        AGG_INLINE int calculate(int x, int y, int d) const
        {
            int ret = m_gradient->calculate(x, y, d) % d;
            if(ret < 0) ret += d;
            return ret;
        }
    
    private:
        const GradientF* m_gradient;
    };
    
    //================================================gradient_reflect_adaptor
    template<class GradientF> class gradient_reflect_adaptor
    {
    public:
        // 使用给定的梯度函数对象初始化反射梯度适配器
        gradient_reflect_adaptor(const GradientF& gradient) : 
            m_gradient(&gradient) {}
    
        // 调用内部梯度函数对象的 calculate 方法，并对结果取模 d*2，保证结果在 [0, d*2) 范围内
        AGG_INLINE int calculate(int x, int y, int d) const
        {
            int d2 = d << 1;
            int ret = m_gradient->calculate(x, y, d) % d2;
            if(ret <  0) ret += d2;
            if(ret >= d) ret  = d2 - ret;
            return ret;
        }
    
    private:
        const GradientF* m_gradient;
    };
}


注释：


// 结束条件为匹配的条件预处理命令的末尾，这里是一个预处理命令的结束标志



#endif


注释：


// 预处理命令的结束，用于结束条件编译指令，对应于条件编译指令 #ifdef 或 #ifndef 的结束
```