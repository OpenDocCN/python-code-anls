# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_arrowhead.cpp`

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
// Simple arrowhead/arrowtail generator 
//
//----------------------------------------------------------------------------

#include "agg_arrowhead.h"

namespace agg
{

    //------------------------------------------------------------------------
    // 构造函数：初始化箭头头部和尾部参数及标志
    arrowhead::arrowhead() :
        m_head_d1(1.0),         // 头部参数1，默认为1.0
        m_head_d2(1.0),         // 头部参数2，默认为1.0
        m_head_d3(1.0),         // 头部参数3，默认为1.0
        m_head_d4(0.0),         // 头部参数4，默认为0.0
        m_tail_d1(1.0),         // 尾部参数1，默认为1.0
        m_tail_d2(1.0),         // 尾部参数2，默认为1.0
        m_tail_d3(1.0),         // 尾部参数3，默认为1.0
        m_tail_d4(0.0),         // 尾部参数4，默认为0.0
        m_head_flag(false),     // 头部标志，默认为false
        m_tail_flag(false),     // 尾部标志，默认为false
        m_curr_id(0),           // 当前路径 ID，默认为0
        m_curr_coord(0)         // 当前坐标索引，默认为0
    {
    }



    //------------------------------------------------------------------------
    // 函数：初始化箭头重置指定路径的迭代器
    void arrowhead::rewind(unsigned path_id)
    {
        // 将当前路径ID设置为给定的路径ID，并将当前坐标索引重置为0
        m_curr_id = path_id;
        m_curr_coord = 0;
        
        // 如果路径ID为0，则执行以下操作
        if(path_id == 0)
        {
            // 如果尾部标志未设置，则设置第一个命令为停止命令并返回
            if(!m_tail_flag)
            {
                m_cmd[0] = path_cmd_stop;
                return;
            }
            
            // 设置路径坐标为尾部点的坐标值
            m_coord[0]  =  m_tail_d1;             m_coord[1]  =  0.0;
            m_coord[2]  =  m_tail_d1 - m_tail_d4; m_coord[3]  =  m_tail_d3;
            m_coord[4]  = -m_tail_d2 - m_tail_d4; m_coord[5]  =  m_tail_d3;
            m_coord[6]  = -m_tail_d2;             m_coord[7]  =  0.0;
            m_coord[8]  = -m_tail_d2 - m_tail_d4; m_coord[9]  = -m_tail_d3;
            m_coord[10] =  m_tail_d1 - m_tail_d4; m_coord[11] = -m_tail_d3;
    
            // 设置路径命令，包括移动到、线段到和多边形结束命令
            m_cmd[0] = path_cmd_move_to;
            m_cmd[1] = path_cmd_line_to;
            m_cmd[2] = path_cmd_line_to;
            m_cmd[3] = path_cmd_line_to;
            m_cmd[4] = path_cmd_line_to;
            m_cmd[5] = path_cmd_line_to;
            m_cmd[7] = path_cmd_end_poly | path_flags_close | path_flags_ccw;
            m_cmd[6] = path_cmd_stop;
            
            // 返回
            return;
        }
        
        // 如果路径ID为1，则执行以下操作
        if(path_id == 1)
        {
            // 如果头部标志未设置，则设置第一个命令为停止命令并返回
            if(!m_head_flag)
            {
                m_cmd[0] = path_cmd_stop;
                return;
            }
            
            // 设置路径坐标为头部点的坐标值
            m_coord[0]  = -m_head_d1;            m_coord[1]  = 0.0;
            m_coord[2]  = m_head_d2 + m_head_d4; m_coord[3]  = -m_head_d3;
            m_coord[4]  = m_head_d2;             m_coord[5]  = 0.0;
            m_coord[6]  = m_head_d2 + m_head_d4; m_coord[7]  = m_head_d3;
    
            // 设置路径命令，包括移动到、线段到和多边形结束命令
            m_cmd[0] = path_cmd_move_to;
            m_cmd[1] = path_cmd_line_to;
            m_cmd[2] = path_cmd_line_to;
            m_cmd[3] = path_cmd_line_to;
            m_cmd[4] = path_cmd_end_poly | path_flags_close | path_flags_ccw;
            m_cmd[5] = path_cmd_stop;
            
            // 返回
            return;
        }
    }
    
    
    //------------------------------------------------------------------------
    // 返回当前坐标索引处的坐标值，并更新坐标索引，同时返回当前命令
    unsigned arrowhead::vertex(double* x, double* y)
    {
        // 如果当前路径ID小于2，则继续
        if(m_curr_id < 2)
        {
            // 计算当前坐标在坐标数组中的索引
            unsigned curr_idx = m_curr_coord * 2;
            
            // 设置输出参数x和y为当前坐标的值
            *x = m_coord[curr_idx];
            *y = m_coord[curr_idx + 1];
            
            // 返回当前坐标对应的命令，并递增坐标索引
            return m_cmd[m_curr_coord++];
        }
        
        // 如果当前路径ID不小于2，则返回停止命令
        return path_cmd_stop;
    }
}



# 闭合了一个代码块，这里的 '}' 应该是对应某个控制结构（如if、for、while等）的结束标志。
```