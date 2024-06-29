# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_curves.h`

```
//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
// Copyright (C) 2005 Tony Juricic (tonygeek@yahoo.com)
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

#ifndef AGG_CURVES_INCLUDED
#define AGG_CURVES_INCLUDED

#include "agg_array.h"

namespace agg
{

    // See Implementation agg_curves.cpp

    //--------------------------------------------curve_approximation_method_e
    enum curve_approximation_method_e
    {
        curve_inc,  // Incremental curve approximation method
        curve_div   // Recursive division curve approximation method
    };
    
    //--------------------------------------------------------------curve3_inc
    class curve3_inc
    {
    public:
        curve3_inc() :
          m_num_steps(0), m_step(0), m_scale(1.0) { }

        curve3_inc(double x1, double y1, 
                   double x2, double y2, 
                   double x3, double y3) :
            m_num_steps(0), m_step(0), m_scale(1.0) 
        { 
            init(x1, y1, x2, y2, x3, y3);  // Initialize curve with given control points
        }

        // Reset internal state
        void reset() { m_num_steps = 0; m_step = -1; }

        // Initialize curve with new control points
        void init(double x1, double y1, 
                  double x2, double y2, 
                  double x3, double y3);

        // Set approximation method (only incremental supported here)
        void approximation_method(curve_approximation_method_e) {}
        
        // Get current approximation method
        curve_approximation_method_e approximation_method() const { return curve_inc; }

        // Set approximation scale factor
        void approximation_scale(double s);
        
        // Get current approximation scale factor
        double approximation_scale() const;

        // Set angle tolerance (not used in this implementation)
        void angle_tolerance(double) {}
        
        // Get angle tolerance (not used in this implementation)
        double angle_tolerance() const { return 0.0; }

        // Set cusp limit (not used in this implementation)
        void cusp_limit(double) {}
        
        // Get cusp limit (not used in this implementation)
        double cusp_limit() const { return 0.0; }

        // Rewind to the beginning of the path
        void rewind(unsigned path_id);
        
        // Get the next vertex coordinates
        unsigned vertex(double* x, double* y);

    private:
        int      m_num_steps;   // Number of steps for approximation
        int      m_step;        // Current step index
        double   m_scale;       // Scale factor for approximation
        double   m_start_x;     // Start x-coordinate of the curve
        double   m_start_y;     // Start y-coordinate of the curve
        double   m_end_x;       // End x-coordinate of the curve
        double   m_end_y;       // End y-coordinate of the curve
        double   m_fx;          // Current x-coordinate
        double   m_fy;          // Current y-coordinate
        double   m_dfx;         // First derivative x-component
        double   m_dfy;         // First derivative y-component
        double   m_ddfx;        // Second derivative x-component
        double   m_ddfy;        // Second derivative y-component
        double   m_saved_fx;    // Saved current x-coordinate
        double   m_saved_fy;    // Saved current y-coordinate
        double   m_saved_dfx;   // Saved first derivative x-component
        double   m_saved_dfy;   // Saved first derivative y-component
    };





    //-------------------------------------------------------------curve3_div
    class curve3_div
    {
    // 公共接口，表示一个三次曲线分割器
    public:
        // 默认构造函数，初始化各参数为默认值
        curve3_div() : 
            m_approximation_scale(1.0),       // 设置近似比例为1.0
            m_angle_tolerance(0.0),           // 设置角度容差为0.0
            m_count(0)                        // 计数器初始化为0
        {}

        // 参数化构造函数，初始化参数并调用 init 方法
        curve3_div(double x1, double y1, 
                   double x2, double y2, 
                   double x3, double y3) :
            m_approximation_scale(1.0),       // 设置近似比例为1.0
            m_angle_tolerance(0.0),           // 设置角度容差为0.0
            m_count(0)                        // 计数器初始化为0
        { 
            init(x1, y1, x2, y2, x3, y3);     // 调用初始化方法 init
        }

        // 清空所有点数据和计数器
        void reset() { m_points.remove_all(); m_count = 0; }

        // 初始化三次曲线的控制点
        void init(double x1, double y1, 
                  double x2, double y2, 
                  double x3, double y3);

        // 设置近似方法（在此处无操作）
        void approximation_method(curve_approximation_method_e) {}

        // 获取当前近似方法（总是返回 curve_div 表示三次曲线分割）
        curve_approximation_method_e approximation_method() const { return curve_div; }

        // 设置近似比例
        void approximation_scale(double s) { m_approximation_scale = s; }

        // 获取当前近似比例
        double approximation_scale() const { return m_approximation_scale;  }

        // 设置角度容差
        void angle_tolerance(double a) { m_angle_tolerance = a; }

        // 获取当前角度容差
        double angle_tolerance() const { return m_angle_tolerance;  }

        // 设置尖点限制（在此处无操作）
        void cusp_limit(double) {}

        // 获取尖点限制（总是返回0.0）
        double cusp_limit() const { return 0.0; }

        // 重置计数器为指定值
        void rewind(unsigned)
        {
            m_count = 0;
        }

        // 获取下一个顶点坐标，并返回路径命令类型
        unsigned vertex(double* x, double* y)
        {
            if(m_count >= m_points.size()) return path_cmd_stop;  // 如果计数超过点集大小，则停止
            const point_d& p = m_points[m_count++];               // 获取下一个点
            *x = p.x;                                             
            *y = p.y;
            return (m_count == 1) ? path_cmd_move_to : path_cmd_line_to;  // 如果是第一个点，则返回移动到命令，否则返回直线到命令
        }

    private:
        // 计算贝塞尔曲线
        void bezier(double x1, double y1, 
                    double x2, double y2, 
                    double x3, double y3);

        // 递归分割贝塞尔曲线
        void recursive_bezier(double x1, double y1, 
                              double x2, double y2, 
                              double x3, double y3,
                              unsigned level);

        // 近似比例
        double               m_approximation_scale;
        // 距离容差的平方
        double               m_distance_tolerance_square;
        // 角度容差
        double               m_angle_tolerance;
        // 点的数量
        unsigned             m_count;
        // 点的向量
        pod_bvector<point_d> m_points;
    };
    {
        // 曲线的控制点数组，包含8个元素
        double cp[8];
    
        // 默认构造函数，未做任何初始化动作
        curve4_points() {}
    
        // 带参构造函数，接受四对控制点坐标，并初始化控制点数组
        curve4_points(double x1, double y1,
                      double x2, double y2,
                      double x3, double y3,
                      double x4, double y4)
        {
            // 将传入的8个参数分别赋值给控制点数组的对应位置
            cp[0] = x1; cp[1] = y1; cp[2] = x2; cp[3] = y2;
            cp[4] = x3; cp[5] = y3; cp[6] = x4; cp[7] = y4;
        }
    
        // 初始化函数，用于设置控制点数组的值
        void init(double x1, double y1,
                  double x2, double y2,
                  double x3, double y3,
                  double x4, double y4)
        {
            // 将传入的8个参数分别赋值给控制点数组的对应位置
            cp[0] = x1; cp[1] = y1; cp[2] = x2; cp[3] = y2;
            cp[4] = x3; cp[5] = y3; cp[6] = x4; cp[7] = y4;
        }
    
        // 重载运算符[]，返回控制点数组中指定位置的值（只读）
        double  operator [] (unsigned i) const { return cp[i]; }
    
        // 重载运算符[]，返回控制点数组中指定位置的引用（可读写）
        double& operator [] (unsigned i)       { return cp[i]; }
    };
    
    //-------------------------------------------------------------curve4_inc
    class curve4_inc
    {
    public:
        // 默认构造函数，初始化为默认值
        curve4_inc() :
            m_num_steps(0), m_step(0), m_scale(1.0) { }
    
        // 带参构造函数，接受四对控制点坐标，并初始化为默认值
        curve4_inc(double x1, double y1, 
                   double x2, double y2, 
                   double x3, double y3,
                   double x4, double y4) :
            m_num_steps(0), m_step(0), m_scale(1.0) 
        { 
            // 调用初始化函数，设置控制点的值
            init(x1, y1, x2, y2, x3, y3, x4, y4);
        }
    
        // 带参构造函数，接受 curve4_points 对象作为参数，并初始化为默认值
        curve4_inc(const curve4_points& cp) :
            m_num_steps(0), m_step(0), m_scale(1.0) 
        { 
            // 调用初始化函数，从 curve4_points 对象中获取控制点的值并设置
            init(cp[0], cp[1], cp[2], cp[3], cp[4], cp[5], cp[6], cp[7]);
        }
    
        // 重置函数，将步数和当前步骤重置为初始状态
        void reset() { m_num_steps = 0; m_step = -1; }
    
        // 初始化函数声明，用于设置控制点的值
        void init(double x1, double y1, 
                  double x2, double y2, 
                  double x3, double y3,
                  double x4, double y4);
    
        // 初始化函数，接受 curve4_points 对象作为参数，并设置控制点的值
        void init(const curve4_points& cp)
        {
            init(cp[0], cp[1], cp[2], cp[3], cp[4], cp[5], cp[6], cp[7]);
        }
    
        // 设置近似方法，但是未实现具体内容
        void approximation_method(curve_approximation_method_e) {}
    
        // 返回近似方法的枚举类型，默认返回 curve_inc
        curve_approximation_method_e approximation_method() const { return curve_inc; }
    
        // 设置近似比例的函数声明
        void approximation_scale(double s);
    
        // 返回当前的近似比例值
        double approximation_scale() const;
    
        // 设置角度容差，但是未实现具体内容
        void angle_tolerance(double) {}
    
        // 返回角度容差的值，默认为0.0
        double angle_tolerance() const { return 0.0; }
    
        // 设置尖点限制，但是未实现具体内容
        void cusp_limit(double) {}
    
        // 返回尖点限制的值，默认为0.0
        double cusp_limit() const { return 0.0; }
    
        // 设置起点和终点，未实现具体内容
        void     rewind(unsigned path_id);
    
        // 计算下一个顶点坐标，未实现具体内容
        unsigned vertex(double* x, double* y);
    
    private:
        // 私有成员变量，用于存储曲线计算过程中的各种状态和中间值
        int      m_num_steps;
        int      m_step;
        double   m_scale;
        double   m_start_x; 
        double   m_start_y;
        double   m_end_x; 
        double   m_end_y;
        double   m_fx; 
        double   m_fy;
        double   m_dfx; 
        double   m_dfy;
        double   m_ddfx; 
        double   m_ddfy;
        double   m_dddfx; 
        double   m_dddfy;
        double   m_saved_fx; 
        double   m_saved_fy;
        double   m_saved_dfx; 
        double   m_saved_dfy;
        double   m_saved_ddfx; 
        double   m_saved_ddfy;
    };
    //-------------------------------------------------------catrom_to_bezier
    inline curve4_points catrom_to_bezier(double x1, double y1, 
                                          double x2, double y2, 
                                          double x3, double y3,
                                          double x4, double y4)
    {
        // Trans. matrix Catmull-Rom to Bezier
        //
        //  0       1       0       0
        //  -1/6    1       1/6     0
        //  0       1/6     1       -1/6
        //  0       0       1       0
        //
        // 将 Catmull-Rom 曲线转换为 Bezier 曲线的转换矩阵
        // 返回经转换后的四次贝塞尔曲线的控制点
        return curve4_points(
            x2,                                 // 控制点 P1(x2, y2)
            y2,
            (-x1 + 6*x2 + x3) / 6,              // 控制点 P2
            (-y1 + 6*y2 + y3) / 6,
            (x2 + 6*x3 - x4) / 6,               // 控制点 P3
            (y2 + 6*y3 - y4) / 6,
            x3,                                 // 控制点 P4(x3, y3)
            y3);
    }
    
    
    //-----------------------------------------------------------------------
    inline curve4_points
    catrom_to_bezier(const curve4_points& cp)
    {
        // 调用上面定义的函数，将参数控制点数组 cp 转换为四次贝塞尔曲线的控制点
        return catrom_to_bezier(cp[0], cp[1], cp[2], cp[3], 
                                cp[4], cp[5], cp[6], cp[7]);
    }
    
    
    
    //-----------------------------------------------------ubspline_to_bezier
    inline curve4_points ubspline_to_bezier(double x1, double y1, 
                                            double x2, double y2, 
                                            double x3, double y3,
                                            double x4, double y4)
    {
        // Trans. matrix Uniform BSpline to Bezier
        //
        //  1/6     4/6     1/6     0
        //  0       4/6     2/6     0
        //  0       2/6     4/6     0
        //  0       1/6     4/6     1/6
        //
        // 将均匀 B 样条曲线转换为 Bezier 曲线的转换矩阵
        // 返回经转换后的四次贝塞尔曲线的控制点
        return curve4_points(
            (x1 + 4*x2 + x3) / 6,               // 控制点 P1
            (y1 + 4*y2 + y3) / 6,
            (4*x2 + 2*x3) / 6,                  // 控制点 P2
            (4*y2 + 2*y3) / 6,
            (2*x2 + 4*x3) / 6,                  // 控制点 P3
            (2*y2 + 4*y3) / 6,
            (x2 + 4*x3 + x4) / 6,               // 控制点 P4
            (y2 + 4*y3 + y4) / 6);
    }
    
    
    //-----------------------------------------------------------------------
    inline curve4_points 
    ubspline_to_bezier(const curve4_points& cp)
    {
        // 调用上面定义的函数，将参数控制点数组 cp 转换为四次贝塞尔曲线的控制点
        return ubspline_to_bezier(cp[0], cp[1], cp[2], cp[3], 
                                  cp[4], cp[5], cp[6], cp[7]);
    }
    
    
    
    
    //------------------------------------------------------hermite_to_bezier
    inline curve4_points hermite_to_bezier(double x1, double y1, 
                                           double x2, double y2, 
                                           double x3, double y3,
                                           double x4, double y4)
    {
        // Hermite 到 Bezier 的转换矩阵
        //
        //  1       0       0       0
        //  1       0       1/3     0
        //  0       1       0       -1/3
        //  0       1       0       0
        //
        // 输入 Hermite 曲线的控制点，返回相应的 Bezier 曲线的控制点
        return curve4_points(
            x1,                                 // Bezier 曲线的第一个控制点 x 坐标
            y1,                                 // Bezier 曲线的第一个控制点 y 坐标
            (3*x1 + x3) / 3,                    // Bezier 曲线的第二个控制点 x 坐标
            (3*y1 + y3) / 3,                    // Bezier 曲线的第二个控制点 y 坐标
            (3*x2 - x4) / 3,                    // Bezier 曲线的第三个控制点 x 坐标
            (3*y2 - y4) / 3,                    // Bezier 曲线的第三个控制点 y 坐标
            x2,                                 // Bezier 曲线的第四个控制点 x 坐标
            y2                                  // Bezier 曲线的第四个控制点 y 坐标
        );
    }
    
    
    
    //-----------------------------------------------------------------------
    inline curve4_points 
    hermite_to_bezier(const curve4_points& cp)
    {
        // 调用重载的 hermite_to_bezier 函数，将输入的 curve4_points 对象转换为 Bezier 曲线的控制点
        return hermite_to_bezier(cp[0], cp[1], cp[2], cp[3], 
                                 cp[4], cp[5], cp[6], cp[7]);
    }
    
    
    //-------------------------------------------------------------curve4_div
    class curve4_div
    {
    // 公有成员函数，无参数构造函数，初始化曲线分割对象的参数
    public:
        curve4_div() : 
            m_approximation_scale(1.0),          // 设置近似比例为1.0
            m_angle_tolerance(0.0),              // 设置角度容限为0.0
            m_cusp_limit(0.0),                   // 设置尖端极限为0.0
            m_count(0)                           // 计数器初始化为0
        {}

        // 公有成员函数，接受四对点坐标参数的构造函数，初始化曲线分割对象的参数并调用初始化函数
        curve4_div(double x1, double y1, 
                   double x2, double y2, 
                   double x3, double y3,
                   double x4, double y4) :
            m_approximation_scale(1.0),          // 设置近似比例为1.0
            m_angle_tolerance(0.0),              // 设置角度容限为0.0
            m_cusp_limit(0.0),                   // 设置尖端极限为0.0
            m_count(0)                           // 计数器初始化为0
        { 
            init(x1, y1, x2, y2, x3, y3, x4, y4); // 调用初始化函数进行曲线的初始化
        }

        // 公有成员函数，接受curve4_points结构的构造函数，初始化曲线分割对象的参数并调用初始化函数
        curve4_div(const curve4_points& cp) :
            m_approximation_scale(1.0),          // 设置近似比例为1.0
            m_angle_tolerance(0.0),              // 设置角度容限为0.0
            m_cusp_limit(0.0),                   // 设置尖端极限为0.0
            m_count(0)                           // 计数器初始化为0
        { 
            init(cp[0], cp[1], cp[2], cp[3], cp[4], cp[5], cp[6], cp[7]); // 调用初始化函数进行曲线的初始化
        }

        // 重置曲线分割对象，清空点集合并将计数器归零
        void reset() { m_points.remove_all(); m_count = 0; }

        // 初始化曲线分割对象，接受八个参数对曲线进行设置
        void init(double x1, double y1, 
                  double x2, double y2, 
                  double x3, double y3,
                  double x4, double y4);

        // 初始化曲线分割对象，接受curve4_points结构进行设置
        void init(const curve4_points& cp)
        {
            init(cp[0], cp[1], cp[2], cp[3], cp[4], cp[5], cp[6], cp[7]); // 调用初始化函数进行曲线的初始化
        }

        // 设置近似方法，这里的方法是curve_div
        void approximation_method(curve_approximation_method_e) {}

        // 返回当前近似方法，始终返回curve_div
        curve_approximation_method_e approximation_method() const 
        { 
            return curve_div; 
        }

        // 设置近似比例
        void approximation_scale(double s) { m_approximation_scale = s; }
        // 返回当前近似比例
        double approximation_scale() const { return m_approximation_scale;  }

        // 设置角度容限
        void angle_tolerance(double a) { m_angle_tolerance = a; }
        // 返回当前角度容限
        double angle_tolerance() const { return m_angle_tolerance;  }

        // 设置尖端极限
        void cusp_limit(double v) 
        { 
            m_cusp_limit = (v == 0.0) ? 0.0 : pi - v; 
        }

        // 返回当前尖端极限
        double cusp_limit() const 
        { 
            return (m_cusp_limit == 0.0) ? 0.0 : pi - m_cusp_limit; 
        }

        // 重置计数器，将计数器归零
        void rewind(unsigned)
        {
            m_count = 0;
        }

        // 返回下一个点的坐标，并移动计数器
        unsigned vertex(double* x, double* y)
        {
            if(m_count >= m_points.size()) return path_cmd_stop; // 如果计数器超过点的数量，返回停止命令
            const point_d& p = m_points[m_count++]; // 获取当前计数器对应的点
            *x = p.x; // 将x坐标赋值给传入的指针
            *y = p.y; // 将y坐标赋值给传入的指针
            return (m_count == 1) ? path_cmd_move_to : path_cmd_line_to; // 返回移动到或者画线到命令
        }

    private:
        // 贝塞尔曲线绘制函数声明
        void bezier(double x1, double y1, 
                    double x2, double y2, 
                    double x3, double y3, 
                    double x4, double y4);

        // 递归贝塞尔曲线绘制函数声明
        void recursive_bezier(double x1, double y1, 
                              double x2, double y2, 
                              double x3, double y3, 
                              double x4, double y4,
                              unsigned level);

        // 近似比例
        double               m_approximation_scale;
        double               m_distance_tolerance_square; // 距离容限的平方
        double               m_angle_tolerance;          // 角度容限
        double               m_cusp_limit;               // 尖端极限
        unsigned             m_count;                    // 计数器
        pod_bvector<point_d> m_points;                    // 点集合
    };
    //-----------------------------------------------------------------curve3
    // curve3 类定义，用于表示三次曲线
    class curve3
    {
    public:
        // 默认构造函数，初始化为曲线分割方法
        curve3() : m_approximation_method(curve_div) {}
        
        // 带参数的构造函数，初始化为曲线分割方法，并调用 init 方法初始化曲线参数
        curve3(double x1, double y1,
               double x2, double y2,
               double x3, double y3) :
            m_approximation_method(curve_div)
        { 
            init(x1, y1, x2, y2, x3, y3);
        }
    
        // 重置曲线对象
        void reset() 
        { 
            m_curve_inc.reset();
            m_curve_div.reset();
        }
    
        // 初始化曲线参数，根据当前的曲线分割方法选择不同的初始化方式
        void init(double x1, double y1,
                  double x2, double y2,
                  double x3, double y3)
        {
            if(m_approximation_method == curve_inc) 
            {
                m_curve_inc.init(x1, y1, x2, y2, x3, y3);
            }
            else
            {
                m_curve_div.init(x1, y1, x2, y2, x3, y3);
            }
        }
    
        // 设置曲线的近似方法
        void approximation_method(curve_approximation_method_e v) 
        { 
            m_approximation_method = v; 
        }
    
        // 获取当前曲线的近似方法
        curve_approximation_method_e approximation_method() const 
        { 
            return m_approximation_method; 
        }
    
        // 设置曲线的近似比例尺度
        void approximation_scale(double s) 
        { 
            m_curve_inc.approximation_scale(s);
            m_curve_div.approximation_scale(s);
        }
    
        // 获取曲线的近似比例尺度
        double approximation_scale() const 
        { 
            return m_curve_inc.approximation_scale(); 
        }
    
        // 设置曲线的角度容差
        void angle_tolerance(double a) 
        { 
            m_curve_div.angle_tolerance(a); 
        }
    
        // 获取曲线的角度容差
        double angle_tolerance() const 
        { 
            return m_curve_div.angle_tolerance(); 
        }
    
        // 设置曲线的尖点限制
        void cusp_limit(double v) 
        { 
            m_curve_div.cusp_limit(v); 
        }
    
        // 获取曲线的尖点限制
        double cusp_limit() const 
        { 
            return m_curve_div.cusp_limit();  
        }
    
        // 根据当前的曲线分割方法选择对应曲线对象进行重置
        void rewind(unsigned path_id)
        {
            if(m_approximation_method == curve_inc) 
            {
                m_curve_inc.rewind(path_id);
            }
            else
            {
                m_curve_div.rewind(path_id);
            }
        }
    
        // 根据当前的曲线分割方法选择对应曲线对象获取顶点坐标
        unsigned vertex(double* x, double* y)
        {
            if(m_approximation_method == curve_inc) 
            {
                return m_curve_inc.vertex(x, y);
            }
            return m_curve_div.vertex(x, y);
        }
    
    private:
        curve3_inc m_curve_inc; // 曲线分割方法为增量方法的对象
        curve3_div m_curve_div; // 曲线分割方法为分割方法的对象
        curve_approximation_method_e m_approximation_method; // 当前曲线的近似方法
    };
    //-----------------------------------------------------------------curve4
    // curve4 类定义的开始
    class curve4
    {
    // 定义公共部分的 curve4 类
    public:
        // 默认构造函数，使用 curve_div 近似方法初始化
        curve4() : m_approximation_method(curve_div) {}

        // 带参数的构造函数，初始化并调用 init 方法
        curve4(double x1, double y1, 
               double x2, double y2, 
               double x3, double y3,
               double x4, double y4) : 
            m_approximation_method(curve_div)
        { 
            init(x1, y1, x2, y2, x3, y3, x4, y4);
        }

        // 复制构造函数，从 curve4_points 类型初始化并调用 init 方法
        curve4(const curve4_points& cp) :
            m_approximation_method(curve_div)
        { 
            init(cp[0], cp[1], cp[2], cp[3], cp[4], cp[5], cp[6], cp[7]);
        }

        // 重置函数，重置 m_curve_inc 和 m_curve_div
        void reset() 
        { 
            m_curve_inc.reset();
            m_curve_div.reset();
        }

        // 初始化函数，根据近似方法选择使用 m_curve_inc 或 m_curve_div 进行初始化
        void init(double x1, double y1, 
                  double x2, double y2, 
                  double x3, double y3,
                  double x4, double y4)
        {
            if(m_approximation_method == curve_inc) 
            {
                m_curve_inc.init(x1, y1, x2, y2, x3, y3, x4, y4);
            }
            else
            {
                m_curve_div.init(x1, y1, x2, y2, x3, y3, x4, y4);
            }
        }

        // 初始化函数，从 curve4_points 类型初始化
        void init(const curve4_points& cp)
        {
            init(cp[0], cp[1], cp[2], cp[3], cp[4], cp[5], cp[6], cp[7]);
        }

        // 设置近似方法
        void approximation_method(curve_approximation_method_e v) 
        { 
            m_approximation_method = v; 
        }

        // 获取当前的近似方法
        curve_approximation_method_e approximation_method() const 
        { 
            return m_approximation_method; 
        }

        // 设置近似比例
        void approximation_scale(double s) 
        { 
            m_curve_inc.approximation_scale(s);
            m_curve_div.approximation_scale(s);
        }

        // 获取近似比例
        double approximation_scale() const { return m_curve_inc.approximation_scale(); }

        // 设置角度容差
        void angle_tolerance(double v) 
        { 
            m_curve_div.angle_tolerance(v); 
        }

        // 获取角度容差
        double angle_tolerance() const 
        { 
            return m_curve_div.angle_tolerance();  
        }

        // 设置尖点限制
        void cusp_limit(double v) 
        { 
            m_curve_div.cusp_limit(v); 
        }

        // 获取尖点限制
        double cusp_limit() const 
        { 
            return m_curve_div.cusp_limit();  
        }

        // 倒带函数，根据近似方法选择使用 m_curve_inc 或 m_curve_div 的倒带函数
        void rewind(unsigned path_id)
        {
            if(m_approximation_method == curve_inc) 
            {
                m_curve_inc.rewind(path_id);
            }
            else
            {
                m_curve_div.rewind(path_id);
            }
        }

        // 获取顶点函数，根据近似方法选择使用 m_curve_inc 或 m_curve_div 的顶点函数
        unsigned vertex(double* x, double* y)
        {
            if(m_approximation_method == curve_inc) 
            {
                return m_curve_inc.vertex(x, y);
            }
            return m_curve_div.vertex(x, y);
        }

    private:
        curve4_inc m_curve_inc;  // 曲线增量近似对象
        curve4_div m_curve_div;  // 曲线分割近似对象
        curve_approximation_method_e m_approximation_method;  // 当前近似方法
    };
}


注释：

// 这是 C/C++ 中的预处理器指令，用于结束一个条件编译段落或者条件编译的结尾。
// 在此之前通常有与条件编译相关的 #if、#ifdef 或 #ifndef 等指令。
// 这里的 `}` 是与前面的 `#ifdef` 或 `#ifndef` 相匹配的结束符。



#endif


注释：

// 这是 C/C++ 中的预处理器指令，用于结束一个条件编译段落或者条件编译的结尾。
// 在条件编译中，#if、#ifdef 或 #ifndef 通常与 #else 或 #elif 一起使用来控制代码的编译过程。
// `#endif` 表示与之前的 `#if`、`#ifdef` 或 `#ifndef` 相匹配的结束符。
```