# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_trans_warp_magnifier.h`

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

#ifndef AGG_WARP_MAGNIFIER_INCLUDED
#define AGG_WARP_MAGNIFIER_INCLUDED

// 引入命名空间 agg
namespace agg
{

    //----------------------------------------------------trans_warp_magnifier
    //
    // See Inmplementation agg_trans_warp_magnifier.cpp
    // trans_warp_magnifier 类的声明，用于实现图像的扭曲放大
    class trans_warp_magnifier
    {
    public:
        // 默认构造函数，初始化变量
        trans_warp_magnifier() : m_xc(0.0), m_yc(0.0), m_magn(1.0), m_radius(1.0) {}
 
        // 设置中心点坐标
        void center(double x, double y) { m_xc = x; m_yc = y; }
        // 设置放大倍数
        void magnification(double m)    { m_magn = m;         }
        // 设置半径
        void radius(double r)           { m_radius = r;       }

        // 获取中心点 x 坐标
        double xc()            const { return m_xc; }
        // 获取中心点 y 坐标
        double yc()            const { return m_yc; }
        // 获取放大倍数
        double magnification() const { return m_magn;   }
        // 获取半径
        double radius()        const { return m_radius; }

        // 坐标变换函数声明：将输入的 x, y 坐标根据当前设置进行变换
        void transform(double* x, double* y) const;
        // 逆向坐标变换函数声明：将变换后的坐标还原为原始坐标
        void inverse_transform(double* x, double* y) const;

    private:
        // 成员变量：中心点 x 坐标
        double m_xc;
        // 成员变量：中心点 y 坐标
        double m_yc;
        // 成员变量：放大倍数
        double m_magn;
        // 成员变量：半径
        double m_radius;
    };


}

// 结束命名空间 agg
#endif
```