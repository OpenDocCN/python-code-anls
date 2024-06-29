# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_arc.h`

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
// Arc vertex generator
//
//----------------------------------------------------------------------------

#ifndef AGG_ARC_INCLUDED
#define AGG_ARC_INCLUDED

#include <math.h>
#include "agg_basics.h"

namespace agg
{

    //=====================================================================arc
    //
    // See Implementation agg_arc.cpp 
    //
    class arc
    {
    public:
        // 构造函数，初始化默认值
        arc() : m_scale(1.0), m_initialized(false) {}

        // 参数化构造函数，设置圆弧的基本参数
        arc(double x,  double y, 
            double rx, double ry, 
            double a1, double a2, 
            bool ccw=true);

        // 初始化函数，设置圆弧的基本参数
        void init(double x,  double y, 
                  double rx, double ry, 
                  double a1, double a2, 
                  bool ccw=true);

        // 设置近似精度比例尺
        void approximation_scale(double s);

        // 获取近似精度比例尺
        double approximation_scale() const { return m_scale;  }

        // 重置路径状态为指定的起始状态
        void rewind(unsigned);

        // 生成下一个顶点，返回顶点坐标
        unsigned vertex(double* x, double* y);

    private:
        // 规范化角度，确保起始角和终止角符合旋转方向
        void normalize(double a1, double a2, bool ccw);

        double   m_x;           // 圆弧中心点 x 坐标
        double   m_y;           // 圆弧中心点 y 坐标
        double   m_rx;          // 圆弧 x 轴半径
        double   m_ry;          // 圆弧 y 轴半径
        double   m_angle;       // 旋转角度
        double   m_start;       // 起始角度
        double   m_end;         // 终止角度
        double   m_scale;       // 近似精度比例尺
        double   m_da;          // 角度步长
        bool     m_ccw;         // 顺时针方向标志
        bool     m_initialized; // 初始化标志
        unsigned m_path_cmd;    // 路径命令
    };


}


#endif
//----------------------------------------------------------------------------
```