# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_math.h`

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
// Bessel function (besj) was adapted for use in AGG library by Andy Wilk 
// Contact: castor.vulgaris@gmail.com
//----------------------------------------------------------------------------

#ifndef AGG_MATH_INCLUDED
#define AGG_MATH_INCLUDED

#include <math.h>
#include "agg_basics.h"

namespace agg
{

    //------------------------------------------------------vertex_dist_epsilon
    // Coinciding points maximal distance (Epsilon)
    const double vertex_dist_epsilon = 1e-14;

    //-----------------------------------------------------intersection_epsilon
    // Tolerance for intersection calculations
    const double intersection_epsilon = 1.0e-30;

    //------------------------------------------------------------cross_product
    // Calculate the cross product of vectors (x1,y1)-(x2,y2) and (x,y)-(x2,y2)
    AGG_INLINE double cross_product(double x1, double y1, 
                                    double x2, double y2, 
                                    double x,  double y)
    {
        return (x - x2) * (y2 - y1) - (y - y2) * (x2 - x1);
    }

    //--------------------------------------------------------point_in_triangle
    // Check if a point (x,y) lies inside the triangle defined by (x1,y1), (x2,y2), (x3,y3)
    AGG_INLINE bool point_in_triangle(double x1, double y1, 
                                      double x2, double y2, 
                                      double x3, double y3, 
                                      double x,  double y)
    {
        bool cp1 = cross_product(x1, y1, x2, y2, x, y) < 0.0;
        bool cp2 = cross_product(x2, y2, x3, y3, x, y) < 0.0;
        bool cp3 = cross_product(x3, y3, x1, y1, x, y) < 0.0;
        return cp1 == cp2 && cp2 == cp3 && cp3 == cp1;
    }

    //-----------------------------------------------------------calc_distance
    // Calculate the Euclidean distance between two points (x1,y1) and (x2,y2)
    AGG_INLINE double calc_distance(double x1, double y1, double x2, double y2)
    {
        double dx = x2-x1;
        double dy = y2-y1;
        return sqrt(dx * dx + dy * dy);
    }

    //--------------------------------------------------------calc_sq_distance
    // Calculate the squared Euclidean distance between two points (x1,y1) and (x2,y2)
    AGG_INLINE double calc_sq_distance(double x1, double y1, double x2, double y2)
    {
        double dx = x2-x1;
        double dy = y2-y1;
        return dx * dx + dy * dy;
    }

    //------------------------------------------------calc_line_point_distance
    AGG_INLINE double calc_line_point_distance(double x1, double y1, 
                                               double x2, double y2, 
                                               double x,  double y)
    {
        // 计算线段起点到终点的差值
        double dx = x2 - x1;
        double dy = y2 - y1;
        // 计算线段起点到终点的距离
        double d = sqrt(dx * dx + dy * dy);
        // 如果距离小于设定的顶点距离阈值，则返回起点到给定点的距离
        if (d < vertex_dist_epsilon)
        {
            return calc_distance(x1, y1, x, y);
        }
        // 否则，计算给定点到线段的垂直距离
        return ((x - x2) * dy - (y - y2) * dx) / d;
    }

    //-------------------------------------------------------calc_line_point_u
    AGG_INLINE double calc_segment_point_u(double x1, double y1, 
                                           double x2, double y2, 
                                           double x,  double y)
    {
        // 计算线段的方向向量
        double dx = x2 - x1;
        double dy = y2 - y1;

        // 如果线段为一个点，则返回参数u为0
        if (dx == 0 && dy == 0)
        {
            return 0;
        }

        // 计算给定点到线段起点的向量
        double pdx = x - x1;
        double pdy = y - y1;

        // 计算参数u，表示给定点在线段上的位置
        return (pdx * dx + pdy * dy) / (dx * dx + dy * dy);
    }

    //---------------------------------------------calc_line_point_sq_distance
    AGG_INLINE double calc_segment_point_sq_distance(double x1, double y1, 
                                                     double x2, double y2, 
                                                     double x,  double y,
                                                     double u)
    {
        // 如果参数u小于等于0，则返回给定点到线段起点的平方距离
        if (u <= 0)
        {
            return calc_sq_distance(x, y, x1, y1);
        }
        // 如果参数u大于等于1，则返回给定点到线段终点的平方距离
        else if (u >= 1)
        {
            return calc_sq_distance(x, y, x2, y2);
        }
        // 否则，返回给定点到线段上最近点的平方距离
        return calc_sq_distance(x, y, x1 + u * (x2 - x1), y1 + u * (y2 - y1));
    }

    //---------------------------------------------calc_line_point_sq_distance
    AGG_INLINE double calc_segment_point_sq_distance(double x1, double y1, 
                                                     double x2, double y2, 
                                                     double x,  double y)
    {
        // 返回给定点到线段上最近点的平方距离，利用参数u计算
        return 
            calc_segment_point_sq_distance(
                x1, y1, x2, y2, x, y,
                calc_segment_point_u(x1, y1, x2, y2, x, y));
    }

    //-------------------------------------------------------calc_intersection
    AGG_INLINE bool calc_intersection(double ax, double ay, double bx, double by,
                                      double cx, double cy, double dx, double dy,
                                      double* x, double* y)
    {
        // 计算两条线段的交点坐标
        double num = (ay - cy) * (dx - cx) - (ax - cx) * (dy - cy);
        double den = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
        // 如果分母接近于0，则认为没有有效交点
        if (fabs(den) < intersection_epsilon) return false;
        // 计算参数r，表示交点在第一条线段上的位置
        double r = num / den;
        // 计算交点的坐标
        *x = ax + r * (bx - ax);
        *y = ay + r * (by - ay);
        return true;
    }

    //-----------------------------------------------------intersection_exists
    // 检查两条线段是否相交，返回布尔值
    AGG_INLINE bool intersection_exists(double x1, double y1, double x2, double y2,
                                        double x3, double y3, double x4, double y4)
    {
        // 计算第一条线段的向量差
        double dx1 = x2 - x1;
        double dy1 = y2 - y1;
        // 计算第二条线段的向量差
        double dx2 = x4 - x3;
        double dy2 = y4 - y3;
        // 判断线段相交的条件，使用叉乘来判断是否同侧
        return ((x3 - x2) * dy1 - (y3 - y2) * dx1 < 0.0) != 
               ((x4 - x2) * dy1 - (y4 - y2) * dx1 < 0.0) &&
               ((x1 - x4) * dy2 - (y1 - y4) * dx2 < 0.0) !=
               ((x2 - x4) * dy2 - (y2 - y4) * dx2 < 0.0);
    }
    
    //--------------------------------------------------------calc_orthogonal
    // 计算给定线段的法线方向，用于线段外扩
    AGG_INLINE void calc_orthogonal(double thickness,
                                    double x1, double y1,
                                    double x2, double y2,
                                    double* x, double* y)
    {
        // 计算线段的方向向量
        double dx = x2 - x1;
        double dy = y2 - y1;
        // 计算线段长度
        double d = sqrt(dx*dx + dy*dy); 
        // 计算法线方向，并存储在指定的指针变量中
        *x =  thickness * dy / d;
        *y = -thickness * dx / d;
    }
    
    //--------------------------------------------------------dilate_triangle
    // 扩展三角形的顶点，生成一个扩展后的新三角形
    AGG_INLINE void dilate_triangle(double x1, double y1,
                                    double x2, double y2,
                                    double x3, double y3,
                                    double *x, double* y,
                                    double d)
    {
        // 初始化法线方向的变量
        double dx1=0.0;
        double dy1=0.0; 
        double dx2=0.0;
        double dy2=0.0; 
        double dx3=0.0;
        double dy3=0.0; 
        // 计算三角形的有向面积
        double loc = cross_product(x1, y1, x2, y2, x3, y3);
        // 如果有向面积大于指定的阈值
        if(fabs(loc) > intersection_epsilon)
        {
            // 根据三角形的面积方向，决定法线方向的符号
            if(cross_product(x1, y1, x2, y2, x3, y3) > 0.0) 
            {
                d = -d;
            }
            // 计算每条边的法线方向
            calc_orthogonal(d, x1, y1, x2, y2, &dx1, &dy1);
            calc_orthogonal(d, x2, y2, x3, y3, &dx2, &dy2);
            calc_orthogonal(d, x3, y3, x1, y1, &dx3, &dy3);
        }
        // 扩展三角形顶点，生成扩展后的顶点坐标
        *x++ = x1 + dx1;  *y++ = y1 + dy1;
        *x++ = x2 + dx1;  *y++ = y2 + dy1;
        *x++ = x2 + dx2;  *y++ = y2 + dy2;
        *x++ = x3 + dx2;  *y++ = y3 + dy2;
        *x++ = x3 + dx3;  *y++ = y3 + dy3;
        *x++ = x1 + dx3;  *y++ = y1 + dy3;
    }
    AGG_INLINE double calc_triangle_area(double x1, double y1,
                                         double x2, double y2,
                                         double x3, double y3)
    {
        // 计算三角形面积公式
        return (x1*y2 - x2*y1 + x2*y3 - x3*y2 + x3*y1 - x1*y3) * 0.5;
    }

    //-------------------------------------------------------calc_polygon_area
    template<class Storage> double calc_polygon_area(const Storage& st)
    {
        unsigned i;
        double sum = 0.0;
        double x  = st[0].x;
        double y  = st[0].y;
        double xs = x;
        double ys = y;

        // 计算多边形的面积
        for(i = 1; i < st.size(); i++)
        {
            const typename Storage::value_type& v = st[i];
            sum += x * v.y - y * v.x;
            x = v.x;
            y = v.y;
        }
        return (sum + x * ys - y * xs) * 0.5;
    }

    //------------------------------------------------------------------------
    // Tables for fast sqrt
    extern int16u g_sqrt_table[1024];
    extern int8   g_elder_bit_table[256];


    //---------------------------------------------------------------fast_sqrt
    // 快速整数平方根计算 - 非常快速：无循环、除法或乘法
    #if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4035) //Disable warning "no return value"
    #endif
    AGG_INLINE unsigned fast_sqrt(unsigned val)
    {
    #if defined(_M_IX86) && defined(_MSC_VER) && !defined(AGG_NO_ASM)
        // 对于Ix86系列处理器使用此汇编代码。
        // 主要命令是bsr - 确定值的最高有效位数。对于其他处理器（和可能的编译器），
        // 使用纯C的“#else”部分。
        __asm
        {
            mov ebx, val
            mov edx, 11
            bsr ecx, ebx
            sub ecx, 9
            jle less_than_9_bits
            shr ecx, 1
            adc ecx, 0
            sub edx, ecx
            shl ecx, 1
            shr ebx, cl
    less_than_9_bits:
            xor eax, eax
            mov  ax, g_sqrt_table[ebx*2]
            mov ecx, edx
            shr eax, cl
        }
    ```
    //else

        //This code is actually pure C and portable to most 
        //arcitectures including 64bit ones. 
        unsigned t = val;
        int bit=0;
        unsigned shift = 11;

        //The following piece of code is just an emulation of the
        //Ix86 assembler command "bsr" (see above). However on old
        //Intels (like Intel MMX 233MHz) this code is about twice 
        //faster (sic!) then just one "bsr". On PIII and PIV the
        //bsr is optimized quite well.
        // 将整数 t 的高位非零部分找到第一个置位的位号，使用预先计算好的表格 g_elder_bit_table
        bit = t >> 24;
        if(bit)
        {
            bit = g_elder_bit_table[bit] + 24;
        }
        else
        {
            bit = (t >> 16) & 0xFF;
            if(bit)
            {
                bit = g_elder_bit_table[bit] + 16;
            }
            else
            {
                bit = (t >> 8) & 0xFF;
                if(bit)
                {
                    bit = g_elder_bit_table[bit] + 8;
                }
                else
                {
                    bit = g_elder_bit_table[t];
                }
            }
        }

        //This code calculates the sqrt.
        // 计算输入整数的平方根，通过调整 shift 和 val 的值来实现
        bit -= 9;
        if(bit > 0)
        {
            bit = (bit >> 1) + (bit & 1);
            shift -= bit;
            val >>= (bit << 1);
        }
        // 返回经过预计算表格 g_sqrt_table 处理后的平方根值
        return g_sqrt_table[val] >> shift;
    #endif
    }
    #if defined(_MSC_VER)
    #pragma warning(pop)
    #endif




    //--------------------------------------------------------------------besj
    // Function BESJ calculates Bessel function of first kind of order n
    // Arguments:
    //     n - an integer (>=0), the order
    //     x - value at which the Bessel function is required
    //--------------------
    // C++ Mathematical Library
    // Convereted from equivalent FORTRAN library
    // Converetd by Gareth Walker for use by course 392 computational project
    // All functions tested and yield the same results as the corresponding
    // FORTRAN versions.
    //
    // If you have any problems using these functions please report them to
    // M.Muldoon@UMIST.ac.uk
    //
    // Documentation available on the web
    // http://www.ma.umist.ac.uk/mrm/Teaching/392/libs/392.html
    // Version 1.0   8/98
    // 29 October, 1999
    //--------------------
    // Adapted for use in AGG library by Andy Wilk (castor.vulgaris@gmail.com)
    //------------------------------------------------------------------------
    inline double besj(double x, int n)
    {
        // 如果 n 小于 0，则直接返回 0
        if(n < 0)
        {
            return 0;
        }
        // 定义精度为 1E-6
        double d = 1E-6;
        // 初始化 b 为 0
        double b = 0;
        // 如果 x 的绝对值小于等于 d
        if(fabs(x) <= d) 
        {
            // 如果 n 不等于 0，则返回 0；否则返回 1
            if(n != 0) return 0;
            return 1;
        }
        double b1 = 0; // b1 是上一次迭代的值
        // 设置起始的递归顺序
        int m1 = (int)fabs(x) + 6;
        // 如果 x 的绝对值大于 5
        if(fabs(x) > 5) 
        {
            // 根据公式重新计算 m1
            m1 = (int)(fabs(1.4 * x + 60 / x));
        }
        // 计算 m2
        int m2 = (int)(n + 2 + fabs(x) / 4);
        // 如果 m1 大于 m2，则将 m2 更新为 m1
        if (m1 > m2) 
        {
            m2 = m1;
        }
    
        // 从当前最大顺序开始应用递归
        for(;;) 
        {
            double c3 = 0;
            double c2 = 1E-30;
            double c4 = 0;
            int m8 = 1;
            // 如果 m2 是偶数，则 m8 设为 -1
            if (m2 / 2 * 2 == m2) 
            {
                m8 = -1;
            }
            // 计算最大索引 imax
            int imax = m2 - 2;
            // 开始迭代计算
            for (int i = 1; i <= imax; i++) 
            {
                double c6 = 2 * (m2 - i) * c2 / x - c3;
                c3 = c2;
                c2 = c6;
                // 如果 m2 - i - 1 等于 n，则将 c6 赋给 b
                if(m2 - i - 1 == n)
                {
                    b = c6;
                }
                // 切换符号 m8
                m8 = -1 * m8;
                // 如果 m8 大于 0，则更新 c4
                if (m8 > 0)
                {
                    c4 = c4 + 2 * c6;
                }
            }
            // 计算最后一个 c6
            double c6 = 2 * c2 / x - c3;
            // 如果 n 等于 0，则将 c6 赋给 b
            if(n == 0)
            {
                b = c6;
            }
            c4 += c6;
            // 计算 b 的最终值
            b /= c4;
            // 如果 b 和上一次迭代的值 b1 的差的绝对值小于 d，则返回 b
            if(fabs(b - b1) < d)
            {
                return b;
            }
            b1 = b;
            // 增加 m2 的值，继续迭代
            m2 += 3;
        }
    }
}


注释：


// 结束一个 C/C++ 的预处理器的条件编译指令，配合 #ifdef 或 #if 使用，表示条件编译的结束



#endif


注释：


// 结束一个 C/C++ 的条件编译指令块，用于关闭之前由 #ifdef 或 #ifndef 开始的条件编译
```