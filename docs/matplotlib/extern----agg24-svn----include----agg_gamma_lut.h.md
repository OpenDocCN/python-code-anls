# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_gamma_lut.h`

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

#ifndef AGG_GAMMA_LUT_INCLUDED
#define AGG_GAMMA_LUT_INCLUDED

#include <math.h>               // 包含数学函数库，提供数学函数的定义和声明
#include "agg_basics.h"         // 包含基本的几何和图形操作定义
#include "agg_gamma_functions.h"// 包含伽马校正函数的定义

namespace agg
{
    template<class LoResT=int8u, 
             class HiResT=int8u, 
             unsigned GammaShift=8, 
             unsigned HiResShift=8> class gamma_lut
    {
    // 公共部分开始

    // gamma_lut 类型的 typedef，定义了模板参数和常量 GammaShift、HiResShift
    typedef gamma_lut<LoResT, HiResT, GammaShift, HiResShift> self_type;

    // gamma_scale_e 枚举类型，定义了与 GammaShift 相关的常量
    enum gamma_scale_e
    {
        gamma_shift = GammaShift,   // GammaShift 的值
        gamma_size  = 1 << gamma_shift,  // 2 的 GammaShift 次方
        gamma_mask  = gamma_size - 1    // gamma_size - 1
    };

    // hi_res_scale_e 枚举类型，定义了与 HiResShift 相关的常量
    enum hi_res_scale_e
    {
        hi_res_shift = HiResShift,   // HiResShift 的值
        hi_res_size  = 1 << hi_res_shift,  // 2 的 HiResShift 次方
        hi_res_mask  = hi_res_size - 1    // hi_res_size - 1
    };

    // gamma_lut 类的析构函数，释放内存
    ~gamma_lut()
    {
        pod_allocator<LoResT>::deallocate(m_inv_gamma, hi_res_size);  // 释放 m_inv_gamma 内存
        pod_allocator<HiResT>::deallocate(m_dir_gamma, gamma_size);  // 释放 m_dir_gamma 内存
    }

    // gamma_lut 类的默认构造函数
    gamma_lut() : 
        m_gamma(1.0),    // 设置 m_gamma 初始值为 1.0
        m_dir_gamma(pod_allocator<HiResT>::allocate(gamma_size)),  // 分配 m_dir_gamma 内存
        m_inv_gamma(pod_allocator<LoResT>::allocate(hi_res_size))  // 分配 m_inv_gamma 内存
    {
        unsigned i;
        // 初始化 m_dir_gamma 数组
        for(i = 0; i < gamma_size; i++)
        {
            m_dir_gamma[i] = HiResT(i << (hi_res_shift - gamma_shift));  // 计算并赋值给 m_dir_gamma[i]
        }

        // 初始化 m_inv_gamma 数组
        for(i = 0; i < hi_res_size; i++)
        {
            m_inv_gamma[i] = LoResT(i >> (hi_res_shift - gamma_shift));  // 计算并赋值给 m_inv_gamma[i]
        }
    }

    // gamma_lut 类的带参数构造函数
    gamma_lut(double g) :
        m_gamma(1.0),   // 设置 m_gamma 初始值为 1.0
        m_dir_gamma(pod_allocator<HiResT>::allocate(gamma_size)),  // 分配 m_dir_gamma 内存
        m_inv_gamma(pod_allocator<LoResT>::allocate(hi_res_size))  // 分配 m_inv_gamma 内存
    {
        gamma(g);  // 调用 gamma 函数设置 gamma 值
    }

    // 设置 gamma 值的函数
    void gamma(double g) 
    {
        m_gamma = g;  // 设置 m_gamma 为传入的 g 值

        unsigned i;
        // 计算并设置 m_dir_gamma 数组的值
        for(i = 0; i < gamma_size; i++)
        {
            m_dir_gamma[i] = (HiResT)
                uround(pow(i / double(gamma_mask), m_gamma) * double(hi_res_mask));  // 计算并赋值给 m_dir_gamma[i]
        }

        double inv_g = 1.0 / g;  // 计算 g 的倒数
        // 计算并设置 m_inv_gamma 数组的值
        for(i = 0; i < hi_res_size; i++)
        {
            m_inv_gamma[i] = (LoResT)
                uround(pow(i / double(hi_res_mask), inv_g) * double(gamma_mask));  // 计算并赋值给 m_inv_gamma[i]
        }
    }

    // 返回 m_gamma 值的函数
    double gamma() const
    {
        return m_gamma;  // 返回 m_gamma
    }

    // 返回 m_dir_gamma 数组中指定索引 v 对应的值的函数
    HiResT dir(LoResT v) const 
    { 
        return m_dir_gamma[unsigned(v)];  // 返回 m_dir_gamma 数组中索引为 v 的值
    }

    // 返回 m_inv_gamma 数组中指定索引 v 对应的值的函数
    LoResT inv(HiResT v) const 
    { 
        return m_inv_gamma[unsigned(v)];  // 返回 m_inv_gamma 数组中索引为 v 的值
    }

private:
    // 禁止拷贝构造函数和赋值操作符
    gamma_lut(const self_type&);
    const self_type& operator = (const self_type&);

    double m_gamma;   // 存储 gamma 值的变量
    HiResT* m_dir_gamma;  // 指向存储 HiResT 类型数据的数组的指针
    LoResT* m_inv_gamma;  // 指向存储 LoResT 类型数据的数组的指针
};
    public:
        // 返回 m_dir_table 中索引为 v 的元素
        LinearType dir(int8u v) const
        {
            return m_dir_table[v];
        }

        // 返回 m_inv_table 中与 v 相对应的元素
        int8u inv(LinearType v) const
        {
            // Unrolled binary search.
            int8u x = 0;
            if (v > m_inv_table[128]) x = 128;  // 如果 v 大于 m_inv_table[128]，则 x 设置为 128
            if (v > m_inv_table[x + 64]) x += 64;  // 如果 v 大于 m_inv_table[x + 64]，则 x 加 64
            if (v > m_inv_table[x + 32]) x += 32;  // 如果 v 大于 m_inv_table[x + 32]，则 x 加 32
            if (v > m_inv_table[x + 16]) x += 16;  // 如果 v 大于 m_inv_table[x + 16]，则 x 加 16
            if (v > m_inv_table[x + 8]) x += 8;  // 如果 v 大于 m_inv_table[x + 8]，则 x 加 8
            if (v > m_inv_table[x + 4]) x += 4;  // 如果 v 大于 m_inv_table[x + 4]，则 x 加 4
            if (v > m_inv_table[x + 2]) x += 2;  // 如果 v 大于 m_inv_table[x + 2]，则 x 加 2
            if (v > m_inv_table[x + 1]) x += 1;  // 如果 v 大于 m_inv_table[x + 1]，则 x 加 1
            return x;  // 返回找到的 x 值
        }

    protected:
        LinearType m_dir_table[256];  // 存储 LinearType 类型的数组，长度为 256
        LinearType m_inv_table[256];  // 存储 LinearType 类型的数组，长度为 256

        // 只允许派生类实例化。
        sRGB_lut_base() 
        {
        }
    };

    // sRGB_lut - 实现各种类型的 sRGB 转换。
    // 基本模板未定义，下面提供了特化版本。
    template<class LinearType>
    class sRGB_lut;

    template<>
    class sRGB_lut<float> : public sRGB_lut_base<float>
    {
    public:
        sRGB_lut()
        {
            // 生成查找表。
            m_dir_table[0] = 0;  // 将 m_dir_table[0] 设置为 0
            m_inv_table[0] = 0;  // 将 m_inv_table[0] 设置为 0
            for (unsigned i = 1; i <= 255; ++i)
            {
                // 浮点 RGB 在区间 [0,1] 内。
                m_dir_table[i] = float(sRGB_to_linear(i / 255.0));  // 将 m_dir_table[i] 设置为 sRGB 线性化后的浮点数
                m_inv_table[i] = float(sRGB_to_linear((i - 0.5) / 255.0));  // 将 m_inv_table[i] 设置为 sRGB 线性化后的浮点数
            }
        }
    };

    template<>
    class sRGB_lut<int16u> : public sRGB_lut_base<int16u>
    {
    public:
        sRGB_lut()
        {
            // 生成查找表。
            m_dir_table[0] = 0;  // 将 m_dir_table[0] 设置为 0
            m_inv_table[0] = 0;  // 将 m_inv_table[0] 设置为 0
            for (unsigned i = 1; i <= 255; ++i)
            {
                // 16 位 RGB 在区间 [0,65535] 内。
                m_dir_table[i] = uround(65535.0 * sRGB_to_linear(i / 255.0));  // 将 m_dir_table[i] 设置为 sRGB 线性化后的 16 位整数
                m_inv_table[i] = uround(65535.0 * sRGB_to_linear((i - 0.5) / 255.0));  // 将 m_inv_table[i] 设置为 sRGB 线性化后的 16 位整数
            }
        }
    };

    template<>
    class sRGB_lut<int8u> : public sRGB_lut_base<int8u>
    {
    public:
        sRGB_lut()
        {
            // 生成查找表。
            m_dir_table[0] = 0;  // 将 m_dir_table[0] 设置为 0
            m_inv_table[0] = 0;  // 将 m_inv_table[0] 设置为 0
            for (unsigned i = 1; i <= 255; ++i)
            {
                // 8 位 RGB 使用简单的双向查找表处理。
                m_dir_table[i] = uround(255.0 * sRGB_to_linear(i / 255.0));  // 将 m_dir_table[i] 设置为 sRGB 线性化后的 8 位整数
                m_inv_table[i] = uround(255.0 * linear_to_sRGB(i / 255.0));  // 将 m_inv_table[i] 设置为 sRGB 反线性化后的 8 位整数
            }
        }

        // 在这种情况下，逆变换是一个简单的查找。
        int8u inv(int8u v) const
        {
            return m_inv_table[v];  // 返回 m_inv_table 中与 v 相对应的元素
        }
    };

    // sRGB_conv 对象的通用基类。定义一个内部的 sRGB_lut 对象，以便用户不必自行创建。
    template<class T>
    class sRGB_conv_base
    {
    // 定义一个模板类 sRGB_conv_base，包含了一些用于 sRGB 和线性 RGB 之间转换的基本函数
    template<class T>
    class sRGB_conv_base {
    public:
        // 将 sRGB 转换为 RGB，使用查找表进行转换
        static T rgb_from_sRGB(int8u x)
        {
            return lut.dir(x);
        }

        // 将 RGB 转换为 sRGB，使用查找表进行转换
        static int8u rgb_to_sRGB(T x)
        {
            return lut.inv(x);
        }

    private:
        // 声明静态成员变量 lut，用于存储 sRGB 和 RGB 之间转换的查找表
        static sRGB_lut<T> lut;
    };

    // 定义 sRGB_conv_base 类模板的静态成员变量 lut
    // 由于这是一个模板，因此不需要将其定义放在 cpp 文件中
    template<class T>
    sRGB_lut<T> sRGB_conv_base<T>::lut;

    // sRGB 和线性 RGB 转换的包装类模板 sRGB_conv 的前向声明
    // 基础模板未定义，下面提供了特化实现
    template<class T>
    class sRGB_conv;

    // 特化模板，用于 float 类型的 sRGB 和线性 RGB 转换
    template<>
    class sRGB_conv<float> : public sRGB_conv_base<float>
    {
    public:
        // 将 sRGB 中的 alpha 值转换为 float 类型的值
        static float alpha_from_sRGB(int8u x)
        {
            return float(x / 255.0);
        }

        // 将 float 类型的 alpha 值转换为 sRGB 中的值
        static int8u alpha_to_sRGB(float x)
        {
            if (x <= 0) return 0;
            else if (x >= 1) return 255;
            else return int8u(0.5 + x * 255);
        }
    };

    // 特化模板，用于 int16u 类型的 sRGB 和线性 RGB 转换
    template<>
    class sRGB_conv<int16u> : public sRGB_conv_base<int16u>
    {
    public:
        // 将 sRGB 中的 alpha 值转换为 int16u 类型的值
        static int16u alpha_from_sRGB(int8u x)
        {
            return (x << 8) | x;
        }

        // 将 int16u 类型的 alpha 值转换为 sRGB 中的值
        static int8u alpha_to_sRGB(int16u x)
        {
            return x >> 8;
        }
    };

    // 特化模板，用于 int8u 类型的 sRGB 和线性 RGB 转换
    template<>
    class sRGB_conv<int8u> : public sRGB_conv_base<int8u>
    {
    public:
        // 将 sRGB 中的 alpha 值转换为 int8u 类型的值
        static int8u alpha_from_sRGB(int8u x)
        {
            return x;
        }

        // 将 int8u 类型的 alpha 值转换为 sRGB 中的值
        static int8u alpha_to_sRGB(int8u x)
        {
            return x;
        }
    };
}
// 结束条件预处理指令

#endif
// 结束预处理指令段落
```