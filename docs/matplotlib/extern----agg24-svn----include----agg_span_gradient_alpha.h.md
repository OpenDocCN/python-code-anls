# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_gradient_alpha.h`

```
//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// 版权所有 2002-2005 年 Maxim Shemanarev（http://www.antigrain.com）
//
// 允许复制、使用、修改、销售和分发本软件，前提是所有副本中包含此版权声明。
// 本软件按原样提供，不提供明示或暗示的任何保证，也不对适用于任何特定目的做任何声明。
//
//----------------------------------------------------------------------------
// 联系方式：mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------

#ifndef AGG_SPAN_GRADIENT_ALPHA_INCLUDED
#define AGG_SPAN_GRADIENT_ALPHA_INCLUDED

#include "agg_span_gradient.h"  // 包含另一个头文件 "agg_span_gradient.h"

namespace agg
{
    //======================================================span_gradient_alpha
    // span_gradient_alpha 类模板定义，用于处理具有渐变和透明度的区间渲染
    template<class ColorT, 
             class Interpolator,
             class GradientF, 
             class AlphaF>
    class span_gradient_alpha
    {
        // 定义公共成员类型别名：插值器类型、颜色类型、alpha 值类型
        typedef Interpolator interpolator_type;
        typedef ColorT color_type;
        typedef typename color_type::value_type alpha_type;

        // 定义缩放枚举常量，通过插值器和梯度的子像素偏移计算得出
        enum downscale_shift_e
        {
            downscale_shift = interpolator_type::subpixel_shift - gradient_subpixel_shift
        };


        //--------------------------------------------------------------------
        // 默认构造函数，无操作
        span_gradient_alpha() {}

        //--------------------------------------------------------------------
        // 构造函数，初始化成员变量，使用插值器、梯度函数、alpha 函数和距离值
        span_gradient_alpha(interpolator_type& inter,
                            GradientF& gradient_function,
                            AlphaF& alpha_function,
                            double d1, double d2) : 
            m_interpolator(&inter),
            m_gradient_function(&gradient_function),
            m_alpha_function(&alpha_function),
            m_d1(iround(d1 * gradient_subpixel_scale)),
            m_d2(iround(d2 * gradient_subpixel_scale))
        {}

        //--------------------------------------------------------------------
        // 返回插值器的引用
        interpolator_type& interpolator() { return *m_interpolator; }
        // 返回梯度函数的常量引用
        const GradientF& gradient_function() const { return *m_gradient_function; }
        // 返回alpha 函数的常量引用
        const AlphaF& alpha_function() const { return *m_alpha_function; }
        // 返回距离值 d1，转换为实际距离
        double d1() const { return double(m_d1) / gradient_subpixel_scale; }
        // 返回距离值 d2，转换为实际距离
        double d2() const { return double(m_d2) / gradient_subpixel_scale; }

        //--------------------------------------------------------------------
        // 设置新的插值器
        void interpolator(interpolator_type& i) { m_interpolator = &i; }
        // 设置新的梯度函数
        void gradient_function(const GradientF& gf) { m_gradient_function = &gf; }
        // 设置新的alpha 函数
        void alpha_function(const AlphaF& af) { m_alpha_function = &af; }
        // 设置新的距离值 d1，并转换为内部使用的整数形式
        void d1(double v) { m_d1 = iround(v * gradient_subpixel_scale); }
        // 设置新的距离值 d2，并转换为内部使用的整数形式
        void d2(double v) { m_d2 = iround(v * gradient_subpixel_scale); }

        //--------------------------------------------------------------------
        // 准备函数，无操作
        void prepare() {}

        //--------------------------------------------------------------------
        // 生成函数，根据插值器、梯度函数和alpha 函数生成颜色值
        void generate(color_type* span, int x, int y, unsigned len)
        {   
            // 计算距离差
            int dd = m_d2 - m_d1;
            if(dd < 1) dd = 1;  // 如果距离差小于1，则设为1，避免除零错误

            // 初始化插值器，并开始生成
            m_interpolator->begin(x+0.5, y+0.5, len);
            do
            {
                m_interpolator->coordinates(&x, &y);  // 获取当前坐标
                // 计算梯度函数的值，根据位移进行调整
                int d = m_gradient_function->calculate(x >> downscale_shift, 
                                                       y >> downscale_shift, m_d2);
                // 根据距离范围计算alpha 函数的索引
                d = ((d - m_d1) * (int)m_alpha_function->size()) / dd;
                if(d < 0) d = 0;  // 确保索引不小于0
                if(d >= (int)m_alpha_function->size()) d = m_alpha_function->size() - 1;  // 确保索引不超出alpha 函数的范围
                span->a = (*m_alpha_function)[d];  // 设置颜色的alpha 值
                ++span;  // 指向下一个颜色
                ++(*m_interpolator);  // 更新插值器
            }
            while(--len);  // 执行直到生成完所有长度的颜色
        }
    // 定义一个私有类，包含插值器、渐变函数、透明度函数以及两个整型成员变量
    private:
        interpolator_type* m_interpolator;        // 插值器指针
        GradientF*         m_gradient_function;   // 渐变函数指针
        AlphaF*            m_alpha_function;      // 透明度函数指针
        int                m_d1;                  // 第一个整型成员变量
        int                m_d2;                  // 第二个整型成员变量
    };
    
    
    //=======================================================gradient_alpha_x
    // 定义一个模板结构体，用于提取给定类型 ColorT 的 alpha 值
    template<class ColorT> struct gradient_alpha_x
    {
        typedef typename ColorT::value_type alpha_type;  // 定义 alpha_type 为 ColorT 的 value_type
        alpha_type operator [] (alpha_type x) const { return x; }  // 返回输入的 alpha 值
    };
    
    //====================================================gradient_alpha_x_u8
    // 定义一个结构体，用于提取 alpha 值，适用于 uint8 类型
    struct gradient_alpha_x_u8
    {
        typedef int8u alpha_type;  // 定义 alpha_type 为 int8u（即 uint8）
        alpha_type operator [] (alpha_type x) const { return x; }  // 返回输入的 alpha 值
    };
    
    //==========================================gradient_alpha_one_munus_x_u8
    // 定义一个结构体，用于计算 alpha 的补值（255 - x），适用于 uint8 类型
    struct gradient_alpha_one_munus_x_u8
    {
        typedef int8u alpha_type;  // 定义 alpha_type 为 int8u（即 uint8）
        alpha_type operator [] (alpha_type x) const { return 255 - x; }  // 返回 255 减去输入的 alpha 值的结果
    };
}



#endif



// 这两行代码是 C/C++ 中的预处理器指令，用于条件编译和头文件保护。`#endif` 结束条件编译指令块，
// 对应于前面的 `#ifdef` 或 `#ifndef`，用于确保头文件只被包含一次。
```