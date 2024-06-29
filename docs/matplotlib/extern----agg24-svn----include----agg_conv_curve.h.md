# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_conv_curve.h`

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
// classes conv_curve
//
//----------------------------------------------------------------------------

#ifndef AGG_CONV_CURVE_INCLUDED
#define AGG_CONV_CURVE_INCLUDED

#include "agg_basics.h"
#include "agg_curves.h"

namespace agg
{

    //---------------------------------------------------------------conv_curve
    // Curve converter class. Any path storage can have Bezier curves defined 
    // by their control points. There're two types of curves supported: curve3 
    // and curve4. Curve3 is a conic Bezier curve with 2 endpoints and 1 control
    // point. Curve4 has 2 control points (4 points in total) and can be used
    // to interpolate more complicated curves. Curve4, unlike curve3 can be used 
    // to approximate arcs, both circular and elliptical. Curves are approximated 
    // with straight lines and one of the approaches is just to store the whole 
    // sequence of vertices that approximate our curve. It takes additional 
    // memory, and at the same time the consecutive vertices can be calculated 
    // on demand. 
    //
    // Initially, path storages are not suppose to keep all the vertices of the
    // curves (although, nothing prevents us from doing so). Instead, path_storage
    // keeps only vertices, needed to calculate a curve on demand. Those vertices
    // are marked with special commands. So, if the path_storage contains curves 
    // (which are not real curves yet), and we render this storage directly, 
    // all we will see is only 2 or 3 straight line segments (for curve3 and 
    // curve4 respectively). If we need to see real curves drawn we need to 
    // include this class into the conversion pipeline. 
    //
    // Class conv_curve recognizes commands path_cmd_curve3 and path_cmd_curve4 
    // and converts these vertices into a move_to/line_to sequence. 
    //-----------------------------------------------------------------------
    template<class VertexSource, 
             class Curve3=curve3, 
             class Curve4=curve4> class conv_curve
    {
    # 公有部分开始，定义了类型别名 curve3_type 和 curve4_type
    typedef Curve3 curve3_type;
    typedef Curve4 curve4_type;
    typedef conv_curve<VertexSource, Curve3, Curve4> self_type;

    # 显式构造函数，将传入的 VertexSource 对象赋给 m_source，并初始化 m_last_x 和 m_last_y
    explicit conv_curve(VertexSource& source) :
      m_source(&source), m_last_x(0.0), m_last_y(0.0) {}

    # 将新的 VertexSource 对象附加到 conv_curve 对象上
    void attach(VertexSource& source) { m_source = &source; }

    # 设置曲线逼近方法，同时应用于 m_curve3 和 m_curve4
    void approximation_method(curve_approximation_method_e v) 
    { 
        m_curve3.approximation_method(v);
        m_curve4.approximation_method(v);
    }

    # 返回当前曲线逼近方法，从 m_curve4 获取
    curve_approximation_method_e approximation_method() const 
    { 
        return m_curve4.approximation_method();
    }

    # 设置曲线逼近比例因子，同时应用于 m_curve3 和 m_curve4
    void approximation_scale(double s) 
    { 
        m_curve3.approximation_scale(s); 
        m_curve4.approximation_scale(s); 
    }

    # 返回当前曲线逼近比例因子，从 m_curve4 获取
    double approximation_scale() const 
    { 
        return m_curve4.approximation_scale();  
    }

    # 设置角度容差，同时应用于 m_curve3 和 m_curve4
    void angle_tolerance(double v) 
    { 
        m_curve3.angle_tolerance(v); 
        m_curve4.angle_tolerance(v); 
    }

    # 返回当前角度容差，从 m_curve4 获取
    double angle_tolerance() const 
    { 
        return m_curve4.angle_tolerance();  
    }

    # 设置 cusp 限制，同时应用于 m_curve3 和 m_curve4
    void cusp_limit(double v) 
    { 
        m_curve3.cusp_limit(v); 
        m_curve4.cusp_limit(v); 
    }

    # 返回当前 cusp 限制，从 m_curve4 获取
    double cusp_limit() const 
    { 
        return m_curve4.cusp_limit();  
    }

    # 重置对象状态：重置 m_source 对应路径的状态，重置 m_last_x 和 m_last_y，重置 m_curve3 和 m_curve4
    void rewind(unsigned path_id); 

    # 获取下一个顶点的坐标，并更新 m_last_x 和 m_last_y
    unsigned vertex(double* x, double* y);

private:
    # 私有拷贝构造函数和赋值运算符，禁止使用
    conv_curve(const self_type&);
    const self_type& operator = (const self_type&);

    # 成员变量定义：
    VertexSource* m_source;
    double        m_last_x;
    double        m_last_y;
    curve3_type   m_curve3;
    curve4_type   m_curve4;
};

//------------------------------------------------------------------------
# conv_curve 类模板的 rewind 方法的定义
template<class VertexSource, class Curve3, class Curve4>
void conv_curve<VertexSource, Curve3, Curve4>::rewind(unsigned path_id)
{
    # 调用 m_source 所指向的对象的 rewind 方法，重置路径状态
    m_source->rewind(path_id);
    # 重置 m_last_x 和 m_last_y
    m_last_x = 0.0;
    m_last_y = 0.0;
    # 重置 m_curve3 和 m_curve4 对象状态
    m_curve3.reset();
    m_curve4.reset();
}

//------------------------------------------------------------------------
# conv_curve 类模板的 vertex 方法的定义
template<class VertexSource, class Curve3, class Curve4>
unsigned conv_curve<VertexSource, Curve3, Curve4>::vertex(double* x, double* y)
    {
        // 检查曲线 m_curve3 在顶点 (x, y) 处是否停止
        if (!is_stop(m_curve3.vertex(x, y)))
        {
            // 更新最后的 x 和 y 值
            m_last_x = *x;
            m_last_y = *y;
            // 返回直线到达指令
            return path_cmd_line_to;
        }
    
        // 检查曲线 m_curve4 在顶点 (x, y) 处是否停止
        if (!is_stop(m_curve4.vertex(x, y)))
        {
            // 更新最后的 x 和 y 值
            m_last_x = *x;
            m_last_y = *y;
            // 返回直线到达指令
            return path_cmd_line_to;
        }
    
        // 声明变量以存储曲线控制点和终点的坐标
        double ct2_x;
        double ct2_y;
        double end_x;
        double end_y;
    
        // 从路径数据源中获取下一个顶点的指令类型
        unsigned cmd = m_source->vertex(x, y);
    
        // 根据指令类型进行相应处理
        switch (cmd)
        {
        case path_cmd_curve3:
            // 获取曲线的结束点坐标
            m_source->vertex(&end_x, &end_y);
    
            // 初始化三次贝塞尔曲线对象 m_curve3
            m_curve3.init(m_last_x, m_last_y,
                          *x, *y,
                          end_x, end_y);
    
            // 获取曲线的第一个顶点坐标 (即移动到的位置)
            m_curve3.vertex(x, y);    // 第一次调用返回 path_cmd_move_to
            m_curve3.vertex(x, y);    // 这是曲线的第一个顶点
            // 设置指令为直线到达
            cmd = path_cmd_line_to;
            break;
    
        case path_cmd_curve4:
            // 获取曲线的第一个控制点坐标
            m_source->vertex(&ct2_x, &ct2_y);
            // 获取曲线的结束点坐标
            m_source->vertex(&end_x, &end_y);
    
            // 初始化四次贝塞尔曲线对象 m_curve4
            m_curve4.init(m_last_x, m_last_y,
                          *x, *y,
                          ct2_x, ct2_y,
                          end_x, end_y);
    
            // 获取曲线的第一个顶点坐标 (即移动到的位置)
            m_curve4.vertex(x, y);    // 第一次调用返回 path_cmd_move_to
            m_curve4.vertex(x, y);    // 这是曲线的第一个顶点
            // 设置指令为直线到达
            cmd = path_cmd_line_to;
            break;
        }
    
        // 更新最后的 x 和 y 值
        m_last_x = *x;
        m_last_y = *y;
        // 返回当前处理的指令类型
        return cmd;
    }
}


这行代码表示一个闭合的大括号（`}`），通常用于结束代码块或函数定义。在本例中，它可能是用于结束某个函数或条件语句的定义。


#endif


这行代码通常用于条件编译，它会关闭一个由 `#ifdef` 或 `#ifndef` 打开的条件代码块。在这里，`#endif` 表示结束条件编译的部分，与之前的条件预处理指令对应。
```