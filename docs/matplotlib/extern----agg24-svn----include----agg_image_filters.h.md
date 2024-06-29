# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_image_filters.h`

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
// Image transformation filters,
// Filtering classes (image_filter_lut, image_filter),
// Basic filter shape classes
//----------------------------------------------------------------------------

#ifndef AGG_IMAGE_FILTERS_INCLUDED
#define AGG_IMAGE_FILTERS_INCLUDED

#include "agg_array.h"
#include "agg_math.h"

namespace agg
{

    // See Implementation agg_image_filters.cpp 

    // 枚举定义图像滤波器的尺度相关常量
    enum image_filter_scale_e
    {
        image_filter_shift = 14,                      // 图像滤波器移位值
        image_filter_scale = 1 << image_filter_shift, // 图像滤波器尺度
        image_filter_mask  = image_filter_scale - 1   // 图像滤波器掩码
    };

    // 枚举定义图像亚像素尺度相关常量
    enum image_subpixel_scale_e
    {
        image_subpixel_shift = 8,                         // 图像亚像素移位值
        image_subpixel_scale = 1 << image_subpixel_shift, // 图像亚像素尺度
        image_subpixel_mask  = image_subpixel_scale - 1   // 图像亚像素掩码
    };


    //-----------------------------------------------------image_filter_lut
    // 图像滤波器查找表类的定义
    class image_filter_lut
    {
    public:
        // 定义模板成员函数 calculate，根据给定的滤波器函数计算权重数组
        template<class FilterF> void calculate(const FilterF& filter,
                                               bool normalization=true)
        {
            // 获取滤波器的半径
            double r = filter.radius();
            // 根据半径重新分配查找表（Lookup Table）的大小
            realloc_lut(r);
            unsigned i;
            // 计算像素的直径，并左移相应位数以获得像素直径的倍数
            unsigned pivot = diameter() << (image_subpixel_shift - 1);
            for(i = 0; i < pivot; i++)
            {
                // 将像素位置转换为浮点数 x，通过滤波器计算权重 y
                double x = double(i) / double(image_subpixel_scale);
                double y = filter.calc_weight(x);
                // 将计算得到的权重乘以比例尺并四舍五入，存入权重数组
                m_weight_array[pivot + i] = 
                m_weight_array[pivot - i] = (int16)iround(y * image_filter_scale);
            }
            // 设置权重数组的第一个和最后一个元素相等
            unsigned end = (diameter() << image_subpixel_shift) - 1;
            m_weight_array[0] = m_weight_array[end];
            // 如果需要归一化，调用 normalize 方法
            if(normalization) 
            {
                normalize();
            }
        }

        // 默认构造函数初始化成员变量
        image_filter_lut() : m_radius(0), m_diameter(0), m_start(0) {}

        // 模板构造函数，根据给定的滤波器函数进行初始化
        template<class FilterF> image_filter_lut(const FilterF& filter, 
                                                 bool normalization=true)
        {
            calculate(filter, normalization);
        }

        // 返回滤波器的半径
        double       radius()       const { return m_radius;   }
        // 返回滤波器的直径
        unsigned     diameter()     const { return m_diameter; }
        // 返回查找表的起始位置
        int          start()        const { return m_start;    }
        // 返回权重数组的指针
        const int16* weight_array() const { return &m_weight_array[0]; }
        // 声明归一化方法
        void         normalize();

    private:
        // 重新分配查找表的大小，根据给定的半径
        void realloc_lut(double radius);
        // 禁用拷贝构造函数和赋值运算符重载
        image_filter_lut(const image_filter_lut&);
        const image_filter_lut& operator = (const image_filter_lut&);

        // 成员变量：滤波器半径、直径、起始位置、权重数组
        double           m_radius;
        unsigned         m_diameter;
        int              m_start;
        pod_array<int16> m_weight_array;
    };

    //--------------------------------------------------------image_filter
    // 模板类 image_filter 继承自 image_filter_lut
    template<class FilterF> class image_filter : public image_filter_lut
    {
    public:
        // 默认构造函数，调用基类的 calculate 方法初始化
        image_filter()
        {
            calculate(m_filter_function);
        }
    private:
        // 成员变量：滤波器函数
        FilterF m_filter_function;
    };


    //-----------------------------------------------image_filter_bilinear
    // 结构体 image_filter_bilinear，实现静态方法 radius 和 calc_weight
    struct image_filter_bilinear
    {
        // 返回固定的半径值
        static double radius() { return 1.0; }
        // 计算线性插值权重，返回 1.0 - x
        static double calc_weight(double x)
        {
            return 1.0 - x;
        }
    };


    //-----------------------------------------------image_filter_hanning
    // 结构体 image_filter_hanning，实现静态方法 radius 和 calc_weight
    struct image_filter_hanning
    {
        // 返回固定的半径值
        static double radius() { return 1.0; }
        // 计算汉宁窗口权重，返回 0.5 + 0.5 * cos(pi * x)
        static double calc_weight(double x)
        {
            return 0.5 + 0.5 * cos(pi * x);
        }
    };


    //-----------------------------------------------image_filter_hamming
    // 结构体 image_filter_hamming，实现静态方法 radius 和 calc_weight
    struct image_filter_hamming
    {
        // 返回固定的半径值
        static double radius() { return 1.0; }
        // 计算海明窗口权重，返回 0.54 + 0.46 * cos(pi * x)
        static double calc_weight(double x)
        {
            return 0.54 + 0.46 * cos(pi * x);
        }
    };

    //-----------------------------------------------image_filter_hermite
    // 定义 Hermite 图像滤波器结构体，包含静态方法和静态成员函数
    struct image_filter_hermite
    {
        // 返回 Hermite 滤波器的半径
        static double radius() { return 1.0; }
        
        // 计算 Hermite 权重函数的值
        static double calc_weight(double x)
        {
            return (2.0 * x - 3.0) * x * x + 1.0;
        }
    };

    //------------------------------------------------image_filter_quadric
    // 定义 Quadric 图像滤波器结构体，包含静态方法和静态成员函数
    struct image_filter_quadric
    {
        // 返回 Quadric 滤波器的半径
        static double radius() { return 1.5; }
        
        // 计算 Quadric 权重函数的值
        static double calc_weight(double x)
        {
            double t;
            if(x <  0.5) return 0.75 - x * x;
            if(x <  1.5) {t = x - 1.5; return 0.5 * t * t;}
            return 0.0;
        }
    };

    //------------------------------------------------image_filter_bicubic
    // 定义 Bicubic 图像滤波器类，包含静态方法和静态成员函数
    class image_filter_bicubic
    {
        // 辅助函数，计算 x 的立方
        static double pow3(double x)
        {
            return (x <= 0.0) ? 0.0 : x * x * x;
        }

    public:
        // 返回 Bicubic 滤波器的半径
        static double radius() { return 2.0; }
        
        // 计算 Bicubic 权重函数的值
        static double calc_weight(double x)
        {
            return
                (1.0/6.0) * 
                (pow3(x + 2) - 4 * pow3(x + 1) + 6 * pow3(x) - 4 * pow3(x - 1));
        }
    };

    //-------------------------------------------------image_filter_kaiser
    // 定义 Kaiser 图像滤波器类，包含私有成员和公共方法
    class image_filter_kaiser
    {
        double a;       // 参数 a
        double i0a;     // 参数 i0a
        double epsilon; // 极小值

    public:
        // 构造函数，初始化 Kaiser 滤波器对象
        image_filter_kaiser(double b = 6.33) :
            a(b), epsilon(1e-12)
        {
            // 计算并存储参数 i0a
            i0a = 1.0 / bessel_i0(b);
        }

        // 返回 Kaiser 滤波器的半径
        static double radius() { return 1.0; }
        
        // 计算 Kaiser 权重函数的值
        double calc_weight(double x) const
        {
            return bessel_i0(a * sqrt(1. - x * x)) * i0a;
        }

    private:
        // 私有函数，计算修正的 Bessel 函数 I0
        double bessel_i0(double x) const
        {
            int i;
            double sum, y, t;

            sum = 1.;
            y = x * x / 4.;
            t = y;
        
            // 迭代计算修正的 Bessel 函数 I0 的近似值
            for(i = 2; t > epsilon; i++)
            {
                sum += t;
                t *= (double)y / (i * i);
            }
            return sum;
        }
    };

    //----------------------------------------------image_filter_catrom
    // 定义 Catmull-Rom 图像滤波器结构体，包含静态方法和静态成员函数
    struct image_filter_catrom
    {
        // 返回 Catmull-Rom 滤波器的半径
        static double radius() { return 2.0; }
        
        // 计算 Catmull-Rom 权重函数的值
        static double calc_weight(double x)
        {
            if(x <  1.0) return 0.5 * (2.0 + x * x * (-5.0 + x * 3.0));
            if(x <  2.0) return 0.5 * (4.0 + x * (-8.0 + x * (5.0 - x)));
            return 0.;
        }
    };

    //---------------------------------------------image_filter_mitchell
    // 定义 Mitchell 图像滤波器类，包含私有成员和公共方法
    class image_filter_mitchell
    {
        double p0, p2, p3;  // 参数 p0, p2, p3
        double q0, q1, q2, q3; // 参数 q0, q1, q2, q3
    //----------------------------------------------image_filter_mitchell
    // Mitchell 过滤器类，用于图像处理，提供计算权重和半径的静态方法和实例方法
    class image_filter_mitchell
    {
    public:
        // 构造函数，初始化参数b和c，默认值分别为1/3和1/3
        image_filter_mitchell(double b = 1.0/3.0, double c = 1.0/3.0) :
            // 初始化参数p0、p2、p3、q0、q1、q2、q3，这些值被用于计算权重
            p0((6.0 - 2.0 * b) / 6.0),   // p0的计算
            p2((-18.0 + 12.0 * b + 6.0 * c) / 6.0),  // p2的计算
            p3((12.0 - 9.0 * b - 6.0 * c) / 6.0),    // p3的计算
            q0((8.0 * b + 24.0 * c) / 6.0),          // q0的计算
            q1((-12.0 * b - 48.0 * c) / 6.0),        // q1的计算
            q2((6.0 * b + 30.0 * c) / 6.0),          // q2的计算
            q3((-b - 6.0 * c) / 6.0)                 // q3的计算
        {}

        // 静态方法，返回 Mitchell 过滤器的半径，值为2.0
        static double radius() { return 2.0; }

        // 实例方法，计算给定x值对应的权重
        double calc_weight(double x) const
        {
            // 根据输入的x值计算并返回对应的权重
            if(x < 1.0) return p0 + x * x * (p2 + x * p3);  // x < 1.0 的权重计算
            if(x < 2.0) return q0 + x * (q1 + x * (q2 + x * q3));  // 1.0 <= x < 2.0 的权重计算
            return 0.0;  // 对于其他情况，返回0.0，表示权重为0
        }
    };


    //----------------------------------------------image_filter_spline16
    // Spline16 过滤器类，静态方法提供半径和权重的计算
    struct image_filter_spline16
    {
        // 静态方法，返回 Spline16 过滤器的半径，值为2.0
        static double radius() { return 2.0; }

        // 静态方法，根据输入的x值计算并返回对应的权重
        static double calc_weight(double x)
        {
            // 根据输入的x值判断并计算对应的权重
            if(x < 1.0)
            {
                return ((x - 9.0/5.0 ) * x - 1.0/5.0 ) * x + 1.0;  // x < 1.0 条件下的权重计算
            }
            return ((-1.0/3.0 * (x-1) + 4.0/5.0) * (x-1) - 7.0/15.0 ) * (x-1);  // 其他情况下的权重计算
        }
    };


    //---------------------------------------------image_filter_spline36
    // Spline36 过滤器类，静态方法提供半径和权重的计算
    struct image_filter_spline36
    {
        // 静态方法，返回 Spline36 过滤器的半径，值为3.0
        static double radius() { return 3.0; }

        // 静态方法，根据输入的x值计算并返回对应的权重
        static double calc_weight(double x)
        {
           // 根据输入的x值判断并计算对应的权重
           if(x < 1.0)
           {
              return ((13.0/11.0 * x - 453.0/209.0) * x - 3.0/209.0) * x + 1.0;  // x < 1.0 条件下的权重计算
           }
           if(x < 2.0)
           {
              return ((-6.0/11.0 * (x-1) + 270.0/209.0) * (x-1) - 156.0/ 209.0) * (x-1);  // 1.0 <= x < 2.0 条件下的权重计算
           }
           return ((1.0/11.0 * (x-2) - 45.0/209.0) * (x-2) +  26.0/209.0) * (x-2);  // 其他情况下的权重计算
        }
    };


    //----------------------------------------------image_filter_gaussian
    // 高斯过滤器类，静态方法提供半径和权重的计算
    struct image_filter_gaussian
    {
        // 静态方法，返回高斯过滤器的半径，值为2.0
        static double radius() { return 2.0; }

        // 静态方法，根据输入的x值计算并返回对应的权重
        static double calc_weight(double x) 
        {
            // 根据高斯函数的定义计算并返回对应的权重
            return exp(-2.0 * x * x) * sqrt(2.0 / pi);  // 高斯函数权重的计算
        }
    };


    //------------------------------------------------image_filter_bessel
    // 贝塞尔过滤器类，静态方法提供半径和权重的计算
    struct image_filter_bessel
    {
        // 静态方法，返回贝塞尔过滤器的半径，值为3.2383
        static double radius() { return 3.2383; } 

        // 静态方法，根据输入的x值计算并返回对应的权重
        static double calc_weight(double x)
        {
            // 如果x为0，则返回pi / 4.0，否则计算贝塞尔函数并返回对应的权重
            return (x == 0.0) ? pi / 4.0 : besj(pi * x, 1) / (2.0 * x);  // 贝塞尔函数权重的计算
        }
    };


    //-------------------------------------------------image_filter_sinc
    // Sinc 过滤器类，提供半径和权重的计算
    class image_filter_sinc
    {
    public:
        // 构造函数，初始化半径m_radius，默认为输入r的值，如果r小于2.0，则设为2.0
        image_filter_sinc(double r) : m_radius(r < 2.0 ? 2.0 : r) {}

        // 实例方法，返回该过滤器的半径
        double radius() const { return m_radius; }

        // 实例方法，根据输入的x值计算并返回对应的权重
        double calc_weight(double x) const
        {
            // 如果x为0，则返回1.0，否则计算sinc函数并返回对应的权重
            if(x == 0.0) return 1.0;  // x为0时的权重计算
            x *= pi;  // 将x乘以π
            return sin(x) / x;  // sinc函数权重的计算
        }
    private:
        double m_radius;  // 私有成员变量，存储过滤器的半径
    };


    //-----------------------------------------------image_filter_lanczos
    // Lanczos 过滤器类，尚未完全定义
    class image_filter_lanczos
    {
    //--------------------------------------------image_filter_lanczos144
    // 定义一个名为 image_filter_lanczos144 的类，继承自 image_filter_lanczos 类
    class image_filter_lanczos144 : public image_filter_lanczos
    {
    public:
        // 构造函数，初始化基类 image_filter_lanczos 的半径为 6.0 或给定的 r 值（如果 r < 2.0）
        image_filter_lanczos144() : image_filter_lanczos(6.0){}
    };

    //--------------------------------------------image_filter_lanczos196
    // 定义一个名为 image_filter_lanczos196 的类，继承自 image_filter_lanczos 类
    class image_filter_lanczos196 : public image_filter_lanczos
    {
    public:
        // 构造函数，初始化基类 image_filter_lanczos 的半径为 7.0 或给定的 r 值（如果 r < 2.0）
        image_filter_lanczos196() : image_filter_lanczos(7.0){}
    };

    //--------------------------------------------image_filter_lanczos256
    // 定义一个名为 image_filter_lanczos256 的类，继承自 image_filter_lanczos 类
    class image_filter_lanczos256 : public image_filter_lanczos
    {
    public:
        // 构造函数，初始化基类 image_filter_lanczos 的半径为 8.0 或给定的 r 值（如果 r < 2.0）
        image_filter_lanczos256() : image_filter_lanczos(8.0){}
    };

    //--------------------------------------------image_filter_blackman
    // 定义一个名为 image_filter_blackman 的类
    class image_filter_blackman
    {
    public:
        // 构造函数，初始化 m_radius 为 r（如果 r < 2.0，则为 2.0）
        image_filter_blackman(double r) : m_radius(r < 2.0 ? 2.0 : r) {}

        // 返回 m_radius 的值
        double radius() const { return m_radius; }

        // 计算权重函数，根据给定的 x 值计算对应的权重
        double calc_weight(double x) const
        {
           // 如果 x 为 0.0，返回权重 1.0
           if(x == 0.0) return 1.0;
           
           // 如果 x 大于 m_radius，返回权重 0.0
           if(x > m_radius) return 0.0;
           
           // 将 x 转换为弧度值（乘以 π）
           x *= pi;
           
           // 计算 x 与 m_radius 的比值
           double xr = x / m_radius;
           
           // 返回计算得到的权重，使用 Blackman 窗函数计算
           return (sin(x) / x) * (0.42 + 0.5*cos(xr) + 0.08*cos(2*xr));
        }
    private:
        // 成员变量，表示滤波器的半径
        double m_radius;
    };

    //----------------------------------------------image_filter_lanczos
    // 定义一个名为 image_filter_lanczos 的类
    class image_filter_lanczos
    {
    public:
        // 构造函数，初始化 m_radius 为 r（如果 r < 2.0，则为 2.0）
        image_filter_lanczos(double r) : m_radius(r < 2.0 ? 2.0 : r) {}

        // 返回 m_radius 的值
        double radius() const { return m_radius; }

        // 计算权重函数，根据给定的 x 值计算对应的权重
        double calc_weight(double x) const
        {
           // 如果 x 为 0.0，返回权重 1.0
           if(x == 0.0) return 1.0;
           
           // 如果 x 大于 m_radius，返回权重 0.0
           if(x > m_radius) return 0.0;
           
           // 将 x 转换为弧度值（乘以 π）
           x *= pi;
           
           // 计算 x 与 m_radius 的比值
           double xr = x / m_radius;
           
           // 返回计算得到的权重，使用 Lanczos 窗函数计算
           return (sin(x) / x) * (sin(xr) / xr);
        }
    private:
        // 成员变量，表示滤波器的半径
        double m_radius;
    };

    //------------------------------------------------image_filter_sinc36
    // 定义一个名为 image_filter_sinc36 的类，继承自 image_filter_sinc 类
    class image_filter_sinc36 : public image_filter_sinc
    {
    public:
        // 构造函数，调用基类 image_filter_sinc 的构造函数，初始化半径为 3.0
        image_filter_sinc36() : image_filter_sinc(3.0){}
    };

    //------------------------------------------------image_filter_sinc64
    // 定义一个名为 image_filter_sinc64 的类，继承自 image_filter_sinc 类
    class image_filter_sinc64 : public image_filter_sinc
    {
    public:
        // 构造函数，调用基类 image_filter_sinc 的构造函数，初始化半径为 4.0
        image_filter_sinc64() : image_filter_sinc(4.0){}
    };

    //-----------------------------------------------image_filter_sinc100
    // 定义一个名为 image_filter_sinc100 的类，继承自 image_filter_sinc 类
    class image_filter_sinc100 : public image_filter_sinc
    {
    public:
        // 构造函数，调用基类 image_filter_sinc 的构造函数，初始化半径为 5.0
        image_filter_sinc100() : image_filter_sinc(5.0){}
    };

    //-----------------------------------------------image_filter_sinc144
    // 定义一个名为 image_filter_sinc144 的类，继承自 image_filter_sinc 类
    class image_filter_sinc144 : public image_filter_sinc
    {
    public:
        // 构造函数，调用基类 image_filter_sinc 的构造函数，初始化半径为 6.0
        image_filter_sinc144() : image_filter_sinc(6.0){}
    };

    //-----------------------------------------------image_filter_sinc196
    // 定义一个名为 image_filter_sinc196 的类，继承自 image_filter_sinc 类
    class image_filter_sinc196 : public image_filter_sinc
    {
    public:
        // 构造函数，调用基类 image_filter_sinc 的构造函数，初始化半径为 7.0
        image_filter_sinc196() : image_filter_sinc(7.0){}
    };

    //-----------------------------------------------image_filter_sinc256
    // 定义一个名为 image_filter_sinc256 的类，继承自 image_filter_sinc 类
    class image_filter_sinc256 : public image_filter_sinc
    {
    public:
        // 构造函数，调用基类 image_filter_sinc 的构造函数，初始化半径为 8.0
        image_filter_sinc256() : image_filter_sinc(8.0){}
    };
    // 创建名为 image_filter_lanczos144 的类，继承自 image_filter_lanczos 类，并初始化参数为 6.0
    class image_filter_lanczos144 : public image_filter_lanczos
    { public: image_filter_lanczos144() : image_filter_lanczos(6.0){} };
    
    //--------------------------------------------image_filter_lanczos196
    // 创建名为 image_filter_lanczos196 的类，继承自 image_filter_lanczos 类，并初始化参数为 7.0
    class image_filter_lanczos196 : public image_filter_lanczos
    { public: image_filter_lanczos196() : image_filter_lanczos(7.0){} };
    
    //--------------------------------------------image_filter_lanczos256
    // 创建名为 image_filter_lanczos256 的类，继承自 image_filter_lanczos 类，并初始化参数为 8.0
    class image_filter_lanczos256 : public image_filter_lanczos
    { public: image_filter_lanczos256() : image_filter_lanczos(8.0){} };
    
    //--------------------------------------------image_filter_blackman36
    // 创建名为 image_filter_blackman36 的类，继承自 image_filter_blackman 类，并初始化参数为 3.0
    class image_filter_blackman36 : public image_filter_blackman
    { public: image_filter_blackman36() : image_filter_blackman(3.0){} };
    
    //--------------------------------------------image_filter_blackman64
    // 创建名为 image_filter_blackman64 的类，继承自 image_filter_blackman 类，并初始化参数为 4.0
    class image_filter_blackman64 : public image_filter_blackman
    { public: image_filter_blackman64() : image_filter_blackman(4.0){} };
    
    //-------------------------------------------image_filter_blackman100
    // 创建名为 image_filter_blackman100 的类，继承自 image_filter_blackman 类，并初始化参数为 5.0
    class image_filter_blackman100 : public image_filter_blackman
    { public: image_filter_blackman100() : image_filter_blackman(5.0){} };
    
    //-------------------------------------------image_filter_blackman144
    // 创建名为 image_filter_blackman144 的类，继承自 image_filter_blackman 类，并初始化参数为 6.0
    class image_filter_blackman144 : public image_filter_blackman
    { public: image_filter_blackman144() : image_filter_blackman(6.0){} };
    
    //-------------------------------------------image_filter_blackman196
    // 创建名为 image_filter_blackman196 的类，继承自 image_filter_blackman 类，并初始化参数为 7.0
    class image_filter_blackman196 : public image_filter_blackman
    { public: image_filter_blackman196() : image_filter_blackman(7.0){} };
    
    //-------------------------------------------image_filter_blackman256
    // 创建名为 image_filter_blackman256 的类，继承自 image_filter_blackman 类，并初始化参数为 8.0
    class image_filter_blackman256 : public image_filter_blackman
    { public: image_filter_blackman256() : image_filter_blackman(8.0){} };
}



#endif



// 这两行代码是 C 或 C++ 中的预处理指令，用于条件编译
// } 表示代码块的结束
// #endif 表示结束一个条件编译块
```