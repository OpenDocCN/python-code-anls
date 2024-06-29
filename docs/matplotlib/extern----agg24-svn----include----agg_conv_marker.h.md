# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_marker.h`

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

// conv_marker
//
//----------------------------------------------------------------------------

#ifndef AGG_CONV_MARKER_INCLUDED
#define AGG_CONV_MARKER_INCLUDED

#include "agg_basics.h"
#include "agg_trans_affine.h"

namespace agg
{
    //-------------------------------------------------------------conv_marker
    // conv_marker 类模板，用于处理标记（marker）的转换
    template<class MarkerLocator, class MarkerShapes>
    class conv_marker
    {
    public:
        // 构造函数，初始化 MarkerLocator 和 MarkerShapes
        conv_marker(MarkerLocator& ml, MarkerShapes& ms);

        // 返回变换矩阵的引用和常量引用
        trans_affine& transform() { return m_transform; }
        const trans_affine& transform() const { return m_transform; }

        // 函数重置处理状态和标记数量
        void rewind(unsigned path_id);
        
        // 返回下一个顶点的索引
        unsigned vertex(double* x, double* y);

    private:
        // 复制构造函数和赋值运算符声明为私有，禁止使用
        conv_marker(const conv_marker<MarkerLocator, MarkerShapes>&);
        const conv_marker<MarkerLocator, MarkerShapes>& 
            operator = (const conv_marker<MarkerLocator, MarkerShapes>&);

        // 内部状态枚举
        enum status_e 
        {
            initial,   // 初始状态
            markers,   // 处理标记状态
            polygon,   // 处理多边形状态
            stop       // 停止状态
        };

        // 成员变量
        MarkerLocator* m_marker_locator;   // 标记定位器对象指针
        MarkerShapes*  m_marker_shapes;    // 标记形状对象指针
        trans_affine   m_transform;        // 变换矩阵
        trans_affine   m_mtx;              // 变换矩阵副本
        status_e       m_status;           // 当前状态
        unsigned       m_marker;           // 当前标记索引
        unsigned       m_num_markers;      // 标记数量
    };


    //------------------------------------------------------------------------
    // 构造函数定义，初始化成员变量
    template<class MarkerLocator, class MarkerShapes> 
    conv_marker<MarkerLocator, MarkerShapes>::conv_marker(MarkerLocator& ml, MarkerShapes& ms) :
        m_marker_locator(&ml),
        m_marker_shapes(&ms),
        m_status(initial),
        m_marker(0),
        m_num_markers(1)
    {
    }


    //------------------------------------------------------------------------
    // 重置函数定义，将状态和标记索引重置为初始状态
    template<class MarkerLocator, class MarkerShapes> 
    void conv_marker<MarkerLocator, MarkerShapes>::rewind(unsigned)
    {
        m_status = initial;
        m_marker = 0;
        m_num_markers = 1;
    }


    //------------------------------------------------------------------------
    // 返回下一个顶点的索引函数定义
    template<class MarkerLocator, class MarkerShapes> 
    unsigned conv_marker<MarkerLocator, MarkerShapes>::vertex(double* x, double* y)


这段代码是 C++ 中的类模板定义，用于处理图形标记（markers）。注释解释了每个函数的作用和每个成员变量的用途，帮助理解代码的功能和结构。
    {
        unsigned cmd = path_cmd_move_to;  // 定义无符号整型变量 cmd，并初始化为 path_cmd_move_to，表示移动到路径命令
        double x1, y1, x2, y2;  // 定义四个双精度浮点数变量，用于存储路径操作的坐标点
    
        while(!is_stop(cmd))  // 当命令不是停止命令时执行循环
        {
            switch(m_status)  // 根据当前状态 m_status 进行不同的操作
            {
            case initial:  // 当前状态为 initial
                if(m_num_markers == 0)  // 如果标记数量为 0
                {
                   cmd = path_cmd_stop;  // 将命令设置为停止命令
                   break;  // 跳出 switch
                }
                m_marker_locator->rewind(m_marker);  // 重新定位到标记
                ++m_marker;  // 标记数加一
                m_num_markers = 0;  // 标记数量重置为 0
                m_status = markers;  // 状态转换为 markers
    
            case markers:  // 当前状态为 markers
                if(is_stop(m_marker_locator->vertex(&x1, &y1)))  // 如果定位器返回的顶点是停止命令
                {
                    m_status = initial;  // 状态转换为 initial
                    break;  // 跳出 switch
                }
                if(is_stop(m_marker_locator->vertex(&x2, &y2)))  // 如果定位器返回的顶点是停止命令
                {
                    m_status = initial;  // 状态转换为 initial
                    break;  // 跳出 switch
                }
                ++m_num_markers;  // 增加标记数量
                m_mtx = m_transform;  // 将变换矩阵设置为初始变换
                m_mtx *= trans_affine_rotation(atan2(y2 - y1, x2 - x1));  // 执行旋转变换
                m_mtx *= trans_affine_translation(x1, y1);  // 执行平移变换
                m_marker_shapes->rewind(m_marker - 1);  // 重新定位到前一个标记的形状
                m_status = polygon;  // 状态转换为 polygon
    
            case polygon:  // 当前状态为 polygon
                cmd = m_marker_shapes->vertex(x, y);  // 获取标记形状的顶点命令
                if(is_stop(cmd))  // 如果命令是停止命令
                {
                    cmd = path_cmd_move_to;  // 将命令设置为移动到路径命令
                    m_status = markers;  // 状态转换为 markers
                    break;  // 跳出 switch
                }
                m_mtx.transform(x, y);  // 使用变换矩阵对坐标进行变换
                return cmd;  // 返回当前命令
    
            case stop:  // 当前状态为 stop
                cmd = path_cmd_stop;  // 将命令设置为停止命令
                break;  // 跳出 switch
            }
        }
        return cmd;  // 返回最终的命令
    }
}


注释：


// 关闭一个条件编译的预处理器指令块。在这种情况下，可能是结束一个 #ifdef 或 #ifndef 块。



#endif


注释：


// 关闭一个条件编译的预处理器指令块，该指令与 #ifdef 或 #ifndef 配对使用，用于控制是否编译某些代码块。
```