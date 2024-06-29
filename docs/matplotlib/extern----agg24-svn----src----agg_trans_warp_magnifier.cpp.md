# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_trans_warp_magnifier.cpp`

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
#include "agg_trans_warp_magnifier.h"

namespace agg
{

    //------------------------------------------------------------------------
    // 对给定的坐标进行变换，实现扭曲放大效果
    void trans_warp_magnifier::transform(double* x, double* y) const
    {
        double dx = *x - m_xc;          // 计算x方向上的偏移量
        double dy = *y - m_yc;          // 计算y方向上的偏移量
        double r = sqrt(dx * dx + dy * dy); // 计算到中心点的距离

        // 如果距离小于指定半径，应用放大效果
        if(r < m_radius)
        {
            *x = m_xc + dx * m_magn;    // 应用放大变换到x坐标
            *y = m_yc + dy * m_magn;    // 应用放大变换到y坐标
            return;
        }

        // 如果距离大于等于指定半径，按比例缩放到新的距离
        double m = (r + m_radius * (m_magn - 1.0)) / r;
        *x = m_xc + dx * m;             // 缩放后的x坐标
        *y = m_yc + dy * m;             // 缩放后的y坐标
    }

    //------------------------------------------------------------------------
    // 反向变换，实现将扭曲放大效果逆转回原始坐标
    void trans_warp_magnifier::inverse_transform(double* x, double* y) const
    {
        // Andrew Skalkin的新版本
        //-----------------
        double dx = *x - m_xc;          // 计算x方向上的偏移量
        double dy = *y - m_yc;          // 计算y方向上的偏移量
        double r = sqrt(dx * dx + dy * dy); // 计算到中心点的距离

        // 如果距离小于指定半径乘以放大倍数，应用逆放大效果
        if(r < m_radius * m_magn) 
        {
            *x = m_xc + dx / m_magn;    // 应用逆放大变换到x坐标
            *y = m_yc + dy / m_magn;    // 应用逆放大变换到y坐标
        }
        else
        {
            // 计算新的距离
            double rnew = r - m_radius * (m_magn - 1.0);
            *x = m_xc + rnew * dx / r; // 缩放后的x坐标
            *y = m_yc + rnew * dy / r; // 缩放后的y坐标
        }

        // 旧版本（注释掉的代码）
        //-----------------
        //trans_warp_magnifier t(*this);
        //t.magnification(1.0 / m_magn);
        //t.radius(m_radius * m_magn);
        //t.transform(x, y);
    }

}
```