# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_vpgen_segmentator.h`

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

#ifndef AGG_VPGEN_SEGMENTATOR_INCLUDED
#define AGG_VPGEN_SEGMENTATOR_INCLUDED

#include <math.h>
#include "agg_basics.h"

namespace agg
{

    //=======================================================vpgen_segmentator
    // 
    // See Implementation agg_vpgen_segmentator.cpp
    //
    // 声明 vpgen_segmentator 类，用于生成图形路径分割器
    class vpgen_segmentator
    {
    public:
        // 构造函数，默认近似比例为1.0
        vpgen_segmentator() : m_approximation_scale(1.0) {}

        // 设置近似比例
        void approximation_scale(double s) { m_approximation_scale = s;     }
        // 获取当前近似比例
        double approximation_scale() const { return m_approximation_scale;  }

        // 静态方法，自动闭合路径的状态
        static bool auto_close()   { return false; }
        // 静态方法，自动不闭合路径的状态
        static bool auto_unclose() { return false; }

        // 重置路径生成器状态
        void reset() { m_cmd = path_cmd_stop; }
        // 将当前点移动到指定坐标
        void move_to(double x, double y);
        // 添加直线段到指定坐标
        void line_to(double x, double y);
        // 获取当前顶点坐标，并更新内部状态
        unsigned vertex(double* x, double* y);

    private:
        // 成员变量
        double   m_approximation_scale; // 近似比例
        double   m_x1;                 // 起始点 x 坐标
        double   m_y1;                 // 起始点 y 坐标
        double   m_dx;                 // x 方向增量
        double   m_dy;                 // y 方向增量
        double   m_dl;                 // 线段长度
        double   m_ddl;                // 线段长度增量
        unsigned m_cmd;                // 当前路径命令
    };



}

#endif
```