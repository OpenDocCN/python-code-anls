# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_span_gouraud.h`

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

#ifndef AGG_SPAN_GOURAUD_INCLUDED
#define AGG_SPAN_GOURAUD_INCLUDED

#include "agg_basics.h"
#include "agg_math.h"

namespace agg
{

    //============================================================span_gouraud
    // span_gouraud 类模板，用于生成 Gouraud 渐变的图形段
    template<class ColorT> class span_gouraud
    {
    protected:
        //--------------------------------------------------------------------
        // arrange_vertices 函数，用于根据坐标排序顶点
        void arrange_vertices(coord_type* coord) const
        {
            // 将顶点坐标复制到传入的数组中
            coord[0] = m_coord[0];
            coord[1] = m_coord[1];
            coord[2] = m_coord[2];

            // 根据顶点的 y 坐标值排序顶点，确保顶点顺序正确
            if(m_coord[0].y > m_coord[2].y)
            {
                coord[0] = m_coord[2]; 
                coord[2] = m_coord[0];
            }

            coord_type tmp;
            if(coord[0].y > coord[1].y)
            {
                tmp      = coord[1];
                coord[1] = coord[0];
                coord[0] = tmp;
            }

            if(coord[1].y > coord[2].y)
            {
                tmp      = coord[2];
                coord[2] = coord[1];
                coord[1] = tmp;
            }
        }

    private:
        //--------------------------------------------------------------------
        coord_type m_coord[3];   // 存储三角形的顶点坐标
        double m_x[8];           // x 坐标数组，用于存储插值后的顶点坐标
        double m_y[8];           // y 坐标数组，用于存储插值后的顶点坐标
        unsigned m_cmd[8];       // 命令数组，用于指示绘制路径的操作
        unsigned m_vertex;       // 顶点计数器
    };

}

#endif
```