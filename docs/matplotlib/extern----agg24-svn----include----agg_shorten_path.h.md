# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_shorten_path.h`

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

#ifndef AGG_SHORTEN_PATH_INCLUDED
#define AGG_SHORTEN_PATH_INCLUDED

#include "agg_basics.h"
#include "agg_vertex_sequence.h"

namespace agg
{

    //===========================================================shorten_path
    // shorten_path 模板函数：用于在给定顶点序列中缩短路径长度
    // 参数：
    //   vs: 顶点序列，可以是任意顶点类型的容器
    //   s: 缩短的长度
    //   closed: 表示路径是否封闭的标志，默认为0表示非封闭
    template<class VertexSequence> 
    void shorten_path(VertexSequence& vs, double s, unsigned closed = 0)
    {
        // 定义顶点类型
        typedef typename VertexSequence::value_type vertex_type;

        // 如果需要缩短的长度大于0且顶点序列中的顶点数大于1
        if(s > 0.0 && vs.size() > 1)
        {
            double d;
            // 计算顶点数减2
            int n = int(vs.size() - 2);
            // 循环直到顶点数为0
            while(n)
            {
                // 获取当前顶点与前一顶点的距离
                d = vs[n].dist;
                // 如果当前距离大于要缩短的长度s，则退出循环
                if(d > s) break;
                // 否则移除最后一个顶点，更新缩短长度s，并减少顶点数n
                vs.remove_last();
                s -= d;
                --n;
            }
            // 如果顶点数小于2，则移除所有顶点
            if(vs.size() < 2)
            {
                vs.remove_all();
            }
            else
            {
                // 否则，更新最后一个顶点的坐标以及调整路径闭合状态
                n = vs.size() - 1;
                vertex_type& prev = vs[n-1];
                vertex_type& last = vs[n];
                // 根据比例d调整最后一个顶点的坐标
                d = (prev.dist - s) / prev.dist;
                double x = prev.x + (last.x - prev.x) * d;
                double y = prev.y + (last.y - prev.y) * d;
                last.x = x;
                last.y = y;
                // 如果前一个顶点不包含最后一个顶点，则移除最后一个顶点
                if(!prev(last)) vs.remove_last();
                // 根据closed参数决定是否闭合路径
                vs.close(closed != 0);
            }
        }
    }

}

#endif
```