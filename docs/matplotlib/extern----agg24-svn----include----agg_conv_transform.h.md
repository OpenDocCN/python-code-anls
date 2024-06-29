# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_transform.h`

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
// class conv_transform
//
//----------------------------------------------------------------------------
#ifndef AGG_CONV_TRANSFORM_INCLUDED
#define AGG_CONV_TRANSFORM_INCLUDED

#include "agg_basics.h"
#include "agg_trans_affine.h"

namespace agg
{

    //----------------------------------------------------------conv_transform
    // conv_transform 类模板，用于将顶点源按照指定仿射变换进行转换
    template<class VertexSource, class Transformer=trans_affine> class conv_transform
    {
    public:
        // 构造函数，初始化顶点源和变换器
        conv_transform(VertexSource& source, Transformer& tr) :
            m_source(&source), m_trans(&tr) {}
        
        // 重新绑定顶点源
        void attach(VertexSource& source) { m_source = &source; }

        // 回放路径指定的顶点
        void rewind(unsigned path_id) 
        { 
            m_source->rewind(path_id); 
        }

        // 获取转换后的顶点坐标
        unsigned vertex(double* x, double* y)
        {
            // 获取原始顶点，并检查其类型
            unsigned cmd = m_source->vertex(x, y);
            if(is_vertex(cmd))
            {
                // 对顶点坐标进行仿射变换
                m_trans->transform(x, y);
            }
            return cmd; // 返回顶点命令
        }

        // 设置新的变换器
        void transformer(Transformer& tr)
        {
            m_trans = &tr;
        }

    private:
        // 禁止复制构造函数和赋值操作符
        conv_transform(const conv_transform<VertexSource>&);
        const conv_transform<VertexSource>& 
            operator = (const conv_transform<VertexSource>&);

        VertexSource*      m_source; // 指向顶点源的指针
        Transformer* m_trans; // 指向变换器的指针
    };

}

#endif
//----------------------------------------------------------------------------
```