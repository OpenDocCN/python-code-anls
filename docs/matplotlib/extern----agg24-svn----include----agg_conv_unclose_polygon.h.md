# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_unclose_polygon.h`

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

#ifndef AGG_CONV_UNCLOSE_POLYGON_INCLUDED
#define AGG_CONV_UNCLOSE_POLYGON_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //====================================================conv_unclose_polygon
    // 模板类定义：conv_unclose_polygon，接受一个模板参数 VertexSource
    template<class VertexSource> class conv_unclose_polygon
    {
    public:
        // 构造函数：初始化使用的顶点源
        explicit conv_unclose_polygon(VertexSource& vs) : m_source(&vs) {}
        
        // 方法：更改当前使用的顶点源
        void attach(VertexSource& source) { m_source = &source; }

        // 方法：重置顶点源中的路径迭代器到指定路径
        void rewind(unsigned path_id)
        {
            m_source->rewind(path_id);
        }

        // 方法：获取当前顶点坐标，并可能修改路径命令标志以表示未闭合路径
        unsigned vertex(double* x, double* y)
        {
            // 调用顶点源的 vertex 方法获取顶点坐标及路径命令
            unsigned cmd = m_source->vertex(x, y);
            // 如果当前路径命令表示路径结束并且路径标志包含关闭路径的标志，则清除该标志
            if(is_end_poly(cmd)) cmd &= ~path_flags_close;
            // 返回处理后的路径命令
            return cmd;
        }

    private:
        // 禁止拷贝构造和赋值操作
        conv_unclose_polygon(const conv_unclose_polygon<VertexSource>&);
        const conv_unclose_polygon<VertexSource>& 
            operator = (const conv_unclose_polygon<VertexSource>&);

        // 成员变量：指向当前使用的顶点源的指针
        VertexSource* m_source;
    };

}

#endif
```