# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_vpgen_segmentator.cpp`

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
#include "agg_vpgen_segmentator.h"

namespace agg
{

    // 移动到指定坐标
    void vpgen_segmentator::move_to(double x, double y)
    {
        // 设置起始点坐标
        m_x1 = x;
        m_y1 = y;
        // 重置增量为零，长度和长度递减值为初始值
        m_dx = 0.0;
        m_dy = 0.0;
        m_dl = 2.0;
        m_ddl = 2.0;
        // 设置命令为移动到路径命令
        m_cmd = path_cmd_move_to;
    }

    // 添加直线到指定坐标
    void vpgen_segmentator::line_to(double x, double y)
    {
        // 更新当前点坐标
        m_x1 += m_dx;
        m_y1 += m_dy;
        // 计算新的增量
        m_dx  = x - m_x1;
        m_dy  = y - m_y1;
        // 计算线段长度并乘以近似比例因子
        double len = sqrt(m_dx * m_dx + m_dy * m_dy) * m_approximation_scale;
        if(len < 1e-30) len = 1e-30;  // 避免长度过小
        // 计算长度递减值
        m_ddl = 1.0 / len;
        // 如果是移动到命令，长度设置为零，否则为计算得到的递减值
        m_dl  = (m_cmd == path_cmd_move_to) ? 0.0 : m_ddl;
        // 如果当前命令是停止，设置为直线到命令
        if(m_cmd == path_cmd_stop) m_cmd = path_cmd_line_to;
    }

    // 返回顶点坐标
    unsigned vpgen_segmentator::vertex(double* x, double* y)
    {
        // 如果当前命令是停止，返回停止命令
        if(m_cmd == path_cmd_stop) return path_cmd_stop;

        // 保存当前命令并设置为直线到命令
        unsigned cmd = m_cmd;
        m_cmd = path_cmd_line_to;

        // 如果长度递减值大于等于1.0减去递减递增值，则设置为停止命令并返回终点坐标
        if(m_dl >= 1.0 - m_ddl)
        {
            m_dl = 1.0;
            m_cmd = path_cmd_stop;
            *x = m_x1 + m_dx;
            *y = m_y1 + m_dy;
            return cmd;
        }

        // 计算当前顶点坐标并递增长度递减值
        *x = m_x1 + m_dx * m_dl;
        *y = m_y1 + m_dy * m_dl;
        m_dl += m_ddl;
        return cmd;
    }

}
```