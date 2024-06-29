# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\ctrl\agg_gamma_spline.h`

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
// class gamma_spline
//
//----------------------------------------------------------------------------

#ifndef AGG_GAMMA_SPLINE_INCLUDED
#define AGG_GAMMA_SPLINE_INCLUDED

#include "agg_basics.h"
#include "agg_bspline.h"

namespace agg
{
    
    //------------------------------------------------------------------------
    // Class-helper for calculation gamma-correction arrays. A gamma-correction
    // array is an array of 256 unsigned chars that determine the actual values 
    // of Anti-Aliasing for each pixel coverage value from 0 to 255. If all the 
    // values in the array are equal to its index, i.e. 0,1,2,3,... there's
    // no gamma-correction. Class agg::polyfill allows you to use custom 
    // gamma-correction arrays. You can calculate it using any approach, and
    // class gamma_spline allows you to calculate almost any reasonable shape 
    // of the gamma-curve with using only 4 values - kx1, ky1, kx2, ky2.
    // 
    //                                      kx2
    //        +----------------------------------+
    //        |                 |        |    .  | 
    //        |                 |        | .     | ky2
    //        |                 |       .  ------|
    //        |                 |    .           |
    //        |                 | .              |
    //        |----------------.|----------------|
    //        |             .   |                |
    //        |          .      |                |
    //        |-------.         |                |
    //    ky1 |    .   |        |                |
    //        | .      |        |                |
    //        +----------------------------------+
    //            kx1
    // 
    // Each value can be in range [0...2]. Value 1.0 means one quarter of the
    // bounding rectangle. Function values() calculates the curve by these
    // 4 values. After calling it one can get the gamma-array with call gamma(). 
    // Class also supports the vertex source interface, i.e rewind() and
    // vertex(). It's made for convinience and used in class gamma_ctrl. 
    // Before calling rewind/vertex one must set the bounding box
    // box() using pixel coordinates. 
    //------------------------------------------------------------------------


注释：这部分是版权声明和类声明的头注释，介绍了Anti-Grain Geometry软件的版本和许可条款。
    // 定义一个名为 gamma_spline 的类
    class gamma_spline
    {
    public:
        // 默认构造函数
        gamma_spline();

        // 设置 gamma_spline 对象的两个控制点坐标
        void values(double kx1, double ky1, double kx2, double ky2);

        // 返回指向 m_gamma 数组的常指针，用于访问伽马曲线数据
        const unsigned char* gamma() const { return m_gamma; }

        // 根据输入的 x 值，返回伽马曲线上的 y 值
        double y(double x) const;

        // 获取 gamma_spline 对象当前设置的两个控制点坐标
        void values(double* kx1, double* ky1, double* kx2, double* ky2) const;

        // 设置 gamma_spline 对象的外接框范围
        void box(double x1, double y1, double x2, double y2);

        // 重置 gamma_spline 对象到指定的位置
        void rewind(unsigned);

        // 获取 gamma_spline 对象的当前顶点坐标，并返回索引
        unsigned vertex(double* x, double* y);

    private:
        // 存储伽马曲线数据的数组，长度为 256
        unsigned char m_gamma[256];

        // 存储控制点坐标的数组，用于计算伽马曲线
        double        m_x[4];
        double        m_y[4];

        // 存储 B 样条对象，用于计算伽马曲线
        bspline       m_spline;

        // 存储外接框的坐标范围
        double        m_x1;
        double        m_y1;
        double        m_x2;
        double        m_y2;

        // 当前位置的 x 坐标
        double        m_cur_x;
    };
}


注释：


// 结束一个预处理指令的代码块，通常用于结束条件编译指令



#endif


注释：


// 结束一个条件编译块，用于关闭之前的 #ifdef 或 #ifndef 指令块
```