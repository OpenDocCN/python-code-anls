# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_line_aa_basics.cpp`

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

#include <math.h>
#include "agg_line_aa_basics.h"

namespace agg
{
    //-------------------------------------------------------------------------
    // The number of the octant is determined as a 3-bit value as follows:
    // bit 0 = vertical flag
    // bit 1 = sx < 0
    // bit 2 = sy < 0
    //
    // [N] shows the number of the orthogonal quadrant
    // <M> shows the number of the diagonal quadrant
    //               <1>
    //   [1]          |          [0]
    //       . (3)011 | 001(1) .
    //         .      |      .
    //           .    |    . 
    //             .  |  . 
    //    (2)010     .|.     000(0)
    // <2> ----------.+.----------- <0>
    //    (6)110   .  |  .   100(4)
    //           .    |    .
    //         .      |      .
    //       .        |        .
    //         (7)111 | 101(5) 
    //   [2]          |          [3]
    //               <3> 
    //                                                        0,1,2,3,4,5,6,7 
    // 定义了八个方向的常量，用于表示直线在哪个八分之一区域
    const int8u line_parameters::s_orthogonal_quadrant[8] = { 0,0,1,1,3,3,2,2 };
    // 定义了八个方向的常量，用于表示直线在哪个八分之一区域的对角线
    const int8u line_parameters::s_diagonal_quadrant[8]   = { 0,1,2,1,0,3,2,3 };

    //-------------------------------------------------------------------------
    // 计算两条直线的角平分线
    void bisectrix(const line_parameters& l1, 
                   const line_parameters& l2, 
                   int* x, int* y)
    {
        // 计算线段 l2 的长度与线段 l1 的长度的比值
        double k = double(l2.len) / double(l1.len);
    
        // 计算变换后的新起点坐标 tx 和 ty
        double tx = l2.x2 - (l2.x1 - l1.x1) * k;
        double ty = l2.y2 - (l2.y1 - l1.y1) * k;
    
        // 检查是否需要旋转角度 180 度以确保双分线位于右侧
        if (double(l2.x2 - l2.x1) * double(l2.y1 - l1.y1) <
            double(l2.y2 - l2.y1) * double(l2.x1 - l1.x1) + 100.0)
        {
            // 对 tx 和 ty 进行反向旋转处理
            tx -= (tx - l2.x1) * 2.0;
            ty -= (ty - l2.y1) * 2.0;
        }
    
        // 检查双分线是否过短
        double dx = tx - l2.x1;
        double dy = ty - l2.y1;
        if ((int)sqrt(dx * dx + dy * dy) < line_subpixel_scale)
        {
            // 如果双分线长度过短，则返回中点坐标作为结果
            *x = (l2.x1 + l2.x1 + (l2.y1 - l1.y1) + (l2.y2 - l2.y1)) >> 1;
            *y = (l2.y1 + l2.y1 - (l2.x1 - l1.x1) - (l2.x2 - l2.x1)) >> 1;
            return;
        }
    
        // 将计算得到的坐标 tx 和 ty 取整后返回
        *x = iround(tx);
        *y = iround(ty);
    }
}


注释：

# 这行代码关闭了一个代码块。在很多编程语言中，花括号用于定义代码块，这里的 `}` 表示结束了一个块的范围。
```