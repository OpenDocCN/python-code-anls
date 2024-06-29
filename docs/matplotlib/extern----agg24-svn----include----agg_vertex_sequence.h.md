# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_vertex_sequence.h`

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
//
// vertex_sequence container and vertex_dist struct
//
//----------------------------------------------------------------------------
#ifndef AGG_VERTEX_SEQUENCE_INCLUDED
#define AGG_VERTEX_SEQUENCE_INCLUDED

#include "agg_basics.h"
#include "agg_array.h"
#include "agg_math.h"

namespace agg
{

    //----------------------------------------------------------vertex_sequence
    // Modified agg::pod_bvector. The data is interpreted as a sequence 
    // of vertices. It means that the type T must expose:
    //
    // bool T::operator() (const T& val)
    // 
    // that is called every time new vertex is being added. The main purpose
    // of this operator is the possibility to calculate some values during 
    // adding and to return true if the vertex fits some criteria or false if
    // it doesn't. In the last case the new vertex is not added. 
    // 
    // The simple example is filtering coinciding vertices with calculation 
    // of the distance between the current and previous ones:
    //
    //    struct vertex_dist
    //    {
    //        double   x;
    //        double   y;
    //        double   dist;
    //
    //        vertex_dist() {}
    //        vertex_dist(double x_, double y_) :
    //            x(x_),
    //            y(y_),
    //            dist(0.0)
    //        {
    //        }
    //
    //        bool operator () (const vertex_dist& val)
    //        {
    //            return (dist = calc_distance(x, y, val.x, val.y)) > EPSILON;
    //        }
    //    };
    //
    // Function close() calls this operator and removes the last vertex if 
    // necessary.
    //------------------------------------------------------------------------
    template<class T, unsigned S=6> 
    class vertex_sequence : public pod_bvector<T, S>
    {
    public:
        typedef pod_bvector<T, S> base_type;

        // 添加一个新顶点到顶点序列中
        void add(const T& val);
        
        // 修改最后一个顶点的数值
        void modify_last(const T& val);
        
        // 根据标志决定是否关闭顶点序列，可能会移除最后一个顶点
        void close(bool remove_flag);
    };



    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    void vertex_sequence<T, S>::add(const T& val)
    {
        // 如果顶点序列中有多于一个顶点
        if(base_type::size() > 1)
        {
            // 检查最后两个顶点是否满足特定条件，如果不满足则移除倒数第二个顶点
            if(!(*this)[base_type::size() - 2]((*this)[base_type::size() - 1])) 
            {
                base_type::remove_last();
            }
        }
        // 将新的值添加到顶点序列中
        base_type::add(val);
    }


    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    void vertex_sequence<T, S>::modify_last(const T& val)
    {
        // 移除最后一个顶点
        base_type::remove_last();
        // 添加新的顶点值到顶点序列中
        add(val);
    }



    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    void vertex_sequence<T, S>::close(bool closed)
    {
        // 当顶点序列中的顶点数大于1时执行循环
        while(base_type::size() > 1)
        {
            // 检查倒数第二个顶点和最后一个顶点是否满足特定条件，如果满足则退出循环
            if((*this)[base_type::size() - 2]((*this)[base_type::size() - 1])) break;
            // 将最后一个顶点保存到临时变量t中
            T t = (*this)[base_type::size() - 1];
            // 移除最后一个顶点
            base_type::remove_last();
            // 修改最后一个顶点为临时变量t的值
            modify_last(t);
        }

        // 如果需要闭合多边形
        if(closed)
        {
            // 当顶点序列中的顶点数大于1时执行循环
            while(base_type::size() > 1)
            {
                // 检查最后一个顶点和第一个顶点是否满足特定条件，如果满足则退出循环
                if((*this)[base_type::size() - 1]((*this)[0])) break;
                // 移除最后一个顶点
                base_type::remove_last();
            }
        }
    }


    //-------------------------------------------------------------vertex_dist
    // 顶点(x, y)，同时保存到下一个顶点的距离。如果多边形闭合，则最后一个顶点到第一个顶点的距离为dist，
    // 如果是折线则为0.0。
    struct vertex_dist
    {
        double   x;    // x坐标
        double   y;    // y坐标
        double   dist; // 到下一个顶点的距离

        vertex_dist() {} // 默认构造函数

        // 带参数的构造函数，初始化x、y坐标和距离dist
        vertex_dist(double x_, double y_) :
            x(x_),
            y(y_),
            dist(0.0)
        {
        }

        // 重载函数调用操作符，计算当前顶点到另一个顶点的距离，并根据计算结果返回布尔值
        bool operator () (const vertex_dist& val)
        {
            // 计算当前顶点到val顶点的距离
            bool ret = (dist = calc_distance(x, y, val.x, val.y)) > vertex_dist_epsilon;
            // 如果距离小于等于vertex_dist_epsilon，则将dist设置为1.0 / vertex_dist_epsilon
            if(!ret) dist = 1.0 / vertex_dist_epsilon;
            return ret; // 返回计算结果
        }
    };



    //--------------------------------------------------------vertex_dist_cmd
    // 与上述结构相同，但增加了额外的“command”值
    struct vertex_dist_cmd : public vertex_dist
    {
        unsigned cmd; // 额外的命令值

        vertex_dist_cmd() {} // 默认构造函数

        // 带参数的构造函数，初始化x、y坐标、距离dist和命令值cmd
        vertex_dist_cmd(double x_, double y_, unsigned cmd_) :
            vertex_dist(x_, y_),
            cmd(cmd_)
        {
        }
    };
}
// 结束一个条件编译指令块，对应于 #ifdef 或 #ifndef，用于结束预处理器条件指令的作用域
#endif
// 结束条件编译指令，对应于 #ifdef 或 #ifndef，指示条件编译指令的结束
```