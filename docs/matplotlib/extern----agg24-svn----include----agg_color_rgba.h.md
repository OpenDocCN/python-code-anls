# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_color_rgba.h`

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
//
// Adaptation for high precision colors has been sponsored by
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------

#ifndef AGG_COLOR_RGBA_INCLUDED
#define AGG_COLOR_RGBA_INCLUDED

#include <math.h>            // 引入数学函数库
#include "agg_basics.h"      // 引入基础函数库
#include "agg_gamma_lut.h"   // 引入伽马校正查找表

namespace agg
{
    // Supported component orders for RGB and RGBA pixel formats
    //=======================================================================
    struct order_rgb  { enum rgb_e  { R=0, G=1, B=2, N=3 }; };   // RGB 像素格式的组成顺序
    struct order_bgr  { enum bgr_e  { B=0, G=1, R=2, N=3 }; };   // BGR 像素格式的组成顺序
    struct order_rgba { enum rgba_e { R=0, G=1, B=2, A=3, N=4 }; };  // RGBA 像素格式的组成顺序
    struct order_argb { enum argb_e { A=0, R=1, G=2, B=3, N=4 }; };  // ARGB 像素格式的组成顺序
    struct order_abgr { enum abgr_e { A=0, B=1, G=2, R=3, N=4 }; };  // ABGR 像素格式的组成顺序
    struct order_bgra { enum bgra_e { B=0, G=1, R=2, A=3, N=4 }; };  // BGRA 像素格式的组成顺序

    // Colorspace tag types.
    struct linear {};   // 线性色彩空间标签
    struct sRGB {};     // sRGB 色彩空间标签

    //====================================================================rgba
    struct rgba         // RGBA 颜色结构体
    };

    inline rgba operator+(const rgba& a, const rgba& b)   // 重载加法运算符
    {
        return rgba(a) += b;   // 返回两个 rgba 对象相加的结果
    }

    inline rgba operator*(const rgba& a, double b)   // 重载乘法运算符
    {
        return rgba(a) *= b;   // 返回 rgba 对象与数值相乘的结果
    }

    //------------------------------------------------------------------------
    inline rgba rgba::from_wavelength(double wl, double gamma)
    // 定义一个 RGBA 颜色结构体 t，初始化为全黑
    {
        rgba t(0.0, 0.0, 0.0);
    
        // 根据波长 wl 的范围设置 t 的红、绿、蓝通道值
        if (wl >= 380.0 && wl <= 440.0)
        {
            // 设置红通道值
            t.r = -1.0 * (wl - 440.0) / (440.0 - 380.0);
            // 设置蓝通道值
            t.b = 1.0;
        }
        else if (wl >= 440.0 && wl <= 490.0)
        {
            // 设置绿通道值
            t.g = (wl - 440.0) / (490.0 - 440.0);
            // 设置蓝通道值
            t.b = 1.0;
        }
        else if (wl >= 490.0 && wl <= 510.0)
        {
            // 设置绿通道值
            t.g = 1.0;
            // 设置蓝通道值
            t.b = -1.0 * (wl - 510.0) / (510.0 - 490.0);
        }
        else if (wl >= 510.0 && wl <= 580.0)
        {
            // 设置红通道值
            t.r = (wl - 510.0) / (580.0 - 510.0);
            // 设置绿通道值
            t.g = 1.0;
        }
        else if (wl >= 580.0 && wl <= 645.0)
        {
            // 设置红通道值
            t.r = 1.0;
            // 设置绿通道值
            t.g = -1.0 * (wl - 645.0) / (645.0 - 580.0);
        }
        else if (wl >= 645.0 && wl <= 780.0)
        {
            // 设置红通道值
            t.r = 1.0;
        }
    
        // 设置饱和度 s，默认为 1.0
        double s = 1.0;
        // 根据波长 wl 调整饱和度 s
        if (wl > 700.0)
            s = 0.3 + 0.7 * (780.0 - wl) / (780.0 - 700.0);
        else if (wl < 420.0)
            s = 0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0);
    
        // 根据 gamma 值对 t 的每个通道进行 gamma 校正
        t.r = pow(t.r * s, gamma);
        t.g = pow(t.g * s, gamma);
        t.b = pow(t.b * s, gamma);
        // 返回最终计算得到的颜色 t
        return t;
    }
    
    // 定义一个函数，返回经预乘处理后的 RGBA 颜色
    inline rgba rgba_pre(double r, double g, double b, double a)
    {
        // 调用 rgba 结构体的构造函数，创建颜色对象，再进行预乘处理
        return rgba(r, g, b, a).premultiply();
    }
    
    
    //===================================================================rgba8
    // 定义一个模板结构体 rgba8T，使用线性色彩空间
    template<class Colorspace>
    struct rgba8T
    };
    
    // 定义别名 rgba8 和 srgba8，分别使用 linear 和 sRGB 色彩空间
    typedef rgba8T<linear> rgba8;
    typedef rgba8T<sRGB> srgba8;
    
    
    //-------------------------------------------------------------rgb8_packed
    // 定义一个内联函数，根据给定的无符号整数 v，返回对应的 rgba8 颜色
    inline rgba8 rgb8_packed(unsigned v)
    {
        return rgba8((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
    }
    
    //-------------------------------------------------------------bgr8_packed
    // 定义一个内联函数，根据给定的无符号整数 v，返回对应的 bgr8 颜色
    inline rgba8 bgr8_packed(unsigned v)
    {
        return rgba8(v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF);
    }
    
    //------------------------------------------------------------argb8_packed
    // 定义一个内联函数，根据给定的无符号整数 v，返回对应的 argb8 颜色
    inline rgba8 argb8_packed(unsigned v)
    {
        return rgba8((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF, v >> 24);
    }
    
    //---------------------------------------------------------rgba8_gamma_dir
    // 定义一个模板函数，根据给定的 GammaLUT 对象 gamma，对 rgba8 颜色 c 进行 gamma 正向处理
    template<class GammaLUT>
    rgba8 rgba8_gamma_dir(rgba8 c, const GammaLUT& gamma)
    {
        return rgba8(gamma.dir(c.r), gamma.dir(c.g), gamma.dir(c.b), c.a);
    }
    
    //---------------------------------------------------------rgba8_gamma_inv
    // 定义一个模板函数，根据给定的 GammaLUT 对象 gamma，对 rgba8 颜色 c 进行 gamma 反向处理
    template<class GammaLUT>
    rgba8 rgba8_gamma_inv(rgba8 c, const GammaLUT& gamma)
    {
        return rgba8(gamma.inv(c.r), gamma.inv(c.g), gamma.inv(c.b), c.a);
    }
    
    
    
    //==================================================================rgba16
    // 定义一个结构体 rgba16，表示 16 位 RGBA 颜色
    struct rgba16
    };
    
    
    //------------------------------------------------------rgba16_gamma_dir
    // 定义一个模板函数，根据给定的 GammaLUT 对象 gamma，对 rgba16 颜色 c 进行 gamma 正向处理
    template<class GammaLUT>
    rgba16 rgba16_gamma_dir(rgba16 c, const GammaLUT& gamma)
    {
        return rgba16(gamma.dir(c.r), gamma.dir(c.g), gamma.dir(c.b), c.a);
    }
    //------------------------------------------------------rgba16_gamma_inv
    // 使用模板函数定义，接受一个 rgba16 类型的颜色 c 和一个 GammaLUT 类型的 gamma 参数，
    // 返回一个新的 rgba16 类型的颜色，通过 gamma 对象的 inv 方法分别逆向处理颜色的 r、g、b 通道，保持 a 通道不变。
    template<class GammaLUT>
    rgba16 rgba16_gamma_inv(rgba16 c, const GammaLUT& gamma)
    {
        return rgba16(gamma.inv(c.r), gamma.inv(c.g), gamma.inv(c.b), c.a);
    }
    
    //====================================================================rgba32
    // 定义一个名为 rgba32 的结构体
    struct rgba32
    {
    };
}


注释：

// 关闭一个 C 语言的 #ifdef 或 #ifndef 块
// 该行关闭了之前开始的一个条件编译块，对应于 #ifdef 或 #ifndef 的开始
// 这里是代码结构的结束标记，用来匹配之前的条件编译指令开始的地方
// 在编译时根据条件编译的定义情况来确定是否编译这部分代码
// 在这里，没有提供任何特定的条件编译指令，因此这个块可能是作为示例中的一部分
// 通常情况下，会根据条件来选择是否包含或者排除某些代码段
#endif



#endif


注释：

// 结束一个条件编译块的另一种方式
// 与示例中相似，用于关闭之前的条件编译块
// 这里可能是另一个条件编译块的结束标记，但没有提供其开头的指令
// 在实际代码中，这种结构允许根据编译时的不同条件选择性地包含或排除某些代码
// 再次强调，条件编译的目的是根据预定义的条件来决定是否编译特定的代码段
// 这里的 #endif 表示前面有对应的 #ifdef 或 #ifndef 语句在控制条件编译
// 这些指令在 C 和 C++ 等语言中非常常见，用于跨平台开发和调试时的不同配置
#endif
```