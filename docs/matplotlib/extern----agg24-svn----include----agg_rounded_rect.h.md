# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_rounded_rect.h`

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
// Rounded rectangle vertex generator
//
//----------------------------------------------------------------------------

#ifndef AGG_ROUNDED_RECT_INCLUDED
#define AGG_ROUNDED_RECT_INCLUDED

#include "agg_basics.h"
#include "agg_arc.h"

namespace agg
{
    //------------------------------------------------------------rounded_rect
    //
    // See Implemantation agg_rounded_rect.cpp
    //
    class rounded_rect
    {
    public:
        // 默认构造函数
        rounded_rect() {}
        
        // 构造函数，初始化圆角矩形的位置和圆角半径
        rounded_rect(double x1, double y1, double x2, double y2, double r);

        // 设置矩形的位置
        void rect(double x1, double y1, double x2, double y2);
        
        // 设置圆角的半径
        void radius(double r);
        
        // 分别设置底部和顶部圆角的半径
        void radius(double rx, double ry);
        
        // 设置四个不同角的圆角半径
        void radius(double rx_bottom, double ry_bottom, double rx_top, double ry_top);
        
        // 设置每个角的不同圆角半径
        void radius(double rx1, double ry1, double rx2, double ry2, 
                    double rx3, double ry3, double rx4, double ry4);
        
        // 标准化圆角的半径，确保半径不超出矩形范围
        void normalize_radius();

        // 设置圆弧的近似比例
        void approximation_scale(double s) { m_arc.approximation_scale(s); }
        
        // 获取圆弧的近似比例
        double approximation_scale() const { return m_arc.approximation_scale(); }

        // 重置生成器状态
        void rewind(unsigned);
        
        // 获取下一个顶点的坐标
        unsigned vertex(double* x, double* y);

    private:
        double m_x1;    // 矩形左上角的 x 坐标
        double m_y1;    // 矩形左上角的 y 坐标
        double m_x2;    // 矩形右下角的 x 坐标
        double m_y2;    // 矩形右下角的 y 坐标
        double m_rx1;   // 左下角圆角的 x 半径
        double m_ry1;   // 左下角圆角的 y 半径
        double m_rx2;   // 右下角圆角的 x 半径
        double m_ry2;   // 右下角圆角的 y 半径
        double m_rx3;   // 右上角圆角的 x 半径
        double m_ry3;   // 右上角圆角的 y 半径
        double m_rx4;   // 左上角圆角的 x 半径
        double m_ry4;   // 左上角圆角的 y 半径
        unsigned m_status;  // 生成器的当前状态
        arc m_arc;          // 圆弧对象，用于生成圆角
    };

}

#endif
```