# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_concat.h`

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

#ifndef AGG_CONV_CONCAT_INCLUDED
#define AGG_CONV_CONCAT_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //=============================================================conv_concat
    // Concatenation of two paths. Usually used to combine lines or curves 
    // with markers such as arrowheads
    template<class VS1, class VS2> class conv_concat
    {
    public:
        // 构造函数，接受两个路径源对象作为参数，初始化成员变量
        conv_concat(VS1& source1, VS2& source2) :
            m_source1(&source1), m_source2(&source2), m_status(2) {}
        
        // 附加新的路径源对象到第一个路径源
        void attach1(VS1& source) { m_source1 = &source; }
        
        // 附加新的路径源对象到第二个路径源
        void attach2(VS2& source) { m_source2 = &source; }

        // 重置路径迭代器状态，设置开始路径ID，并且准备迭代
        void rewind(unsigned path_id)
        { 
            m_source1->rewind(path_id); // 重置第一个路径源迭代器
            m_source2->rewind(0);       // 重置第二个路径源迭代器，指定路径ID为0
            m_status = 0;               // 设置状态为0，表示准备读取第一个路径
        }

        // 获取路径的顶点坐标，返回路径命令
        unsigned vertex(double* x, double* y)
        {
            unsigned cmd;
            if(m_status == 0)
            {
                // 从第一个路径源获取顶点坐标和路径命令
                cmd = m_source1->vertex(x, y);
                if(!is_stop(cmd)) return cmd; // 如果顶点命令不是停止命令，直接返回
                m_status = 1; // 切换状态到1，表示准备读取第二个路径
            }
            if(m_status == 1)
            {
                // 从第二个路径源获取顶点坐标和路径命令
                cmd = m_source2->vertex(x, y);
                if(!is_stop(cmd)) return cmd; // 如果顶点命令不是停止命令，直接返回
                m_status = 2; // 切换状态到2，表示读取完成，无需再读取
            }
            return path_cmd_stop; // 返回停止命令，表示路径读取结束
        }

    private:
        conv_concat(const conv_concat<VS1, VS2>&); // 禁止复制构造函数
        const conv_concat<VS1, VS2>& 
            operator = (const conv_concat<VS1, VS2>&); // 禁止赋值运算符

        VS1* m_source1; // 第一个路径源的指针
        VS2* m_source2; // 第二个路径源的指针
        int  m_status;  // 当前迭代状态
    };
}

#endif // AGG_CONV_CONCAT_INCLUDED


这段代码定义了一个模板类 `conv_concat`，用于将两个路径源对象连接起来，依次迭代获取路径的顶点坐标和命令。
```