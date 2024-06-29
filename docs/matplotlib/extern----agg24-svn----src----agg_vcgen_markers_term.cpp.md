# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\agg_vcgen_markers_term.cpp`

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
// Terminal markers generator (arrowhead/arrowtail)
//
//----------------------------------------------------------------------------

#include "agg_vcgen_markers_term.h"

namespace agg
{

    //------------------------------------------------------------------------
    // 清空当前标记点集合
    void vcgen_markers_term::remove_all()
    {
        m_markers.remove_all();
    }


    //------------------------------------------------------------------------
    // 添加顶点坐标到标记点集合
    void vcgen_markers_term::add_vertex(double x, double y, unsigned cmd)
    {
        if(is_move_to(cmd))
        {
            if(m_markers.size() & 1)
            {
                // 初始状态，已添加第一个坐标点。
                // 如果再次调用 start_vertex() 两次或更多，只修改最后一个。
                m_markers.modify_last(coord_type(x, y));
            }
            else
            {
                // 添加新的坐标点
                m_markers.add(coord_type(x, y));
            }
        }
        else
        {
            if(is_vertex(cmd))
            {
                if(m_markers.size() & 1)
                {
                    // 初始状态，已添加第一个坐标点。
                    // 添加三个额外的点，0,1,1,0
                    m_markers.add(coord_type(x, y));
                    m_markers.add(m_markers[m_markers.size() - 1]);
                    m_markers.add(m_markers[m_markers.size() - 3]);
                }
                else
                {
                    if(m_markers.size())
                    {
                        // 替换最后两个点：0,1,1,0 -> 0,1,2,1
                        m_markers[m_markers.size() - 1] = m_markers[m_markers.size() - 2];
                        m_markers[m_markers.size() - 2] = coord_type(x, y);
                    }
                }
            }
        }
    }


    //------------------------------------------------------------------------
    // 设置路径的起始位置为指定路径ID的第一个点
    void vcgen_markers_term::rewind(unsigned path_id)
    {
        m_curr_id = path_id * 2;
        m_curr_idx = m_curr_id;
    }


    //------------------------------------------------------------------------
    // 获取当前标记点坐标，并将其存储到提供的指针位置
    unsigned vcgen_markers_term::vertex(double* x, double* y)
    {
        // 检查当前索引是否超出允许范围，若超出则返回停止命令
        if(m_curr_id > 2 || m_curr_idx >= m_markers.size()) 
        {
            return path_cmd_stop;
        }
        // 从标记数组中获取当前坐标点的引用
        const coord_type& c = m_markers[m_curr_idx];
        // 将当前点的 x 坐标赋值给指针 *x
        *x = c.x;
        // 将当前点的 y 坐标赋值给指针 *y
        *y = c.y;
        // 检查当前索引是否为奇数，若是，则设置索引增加 3 并返回直线命令
        if(m_curr_idx & 1)
        {
            m_curr_idx += 3;
            return path_cmd_line_to;
        }
        // 否则，增加索引并返回移动命令
        ++m_curr_idx;
        return path_cmd_move_to;
    }
}
```