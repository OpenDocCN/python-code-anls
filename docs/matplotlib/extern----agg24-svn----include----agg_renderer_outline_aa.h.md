# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_renderer_outline_aa.h`

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

#ifndef AGG_RENDERER_OUTLINE_AA_INCLUDED
#define AGG_RENDERER_OUTLINE_AA_INCLUDED

#include "agg_array.h"                 // 包含数组操作的头文件
#include "agg_math.h"                  // 包含数学函数的头文件
#include "agg_line_aa_basics.h"        // 包含抗锯齿线段基础函数的头文件
#include "agg_dda_line.h"              // 包含DDA算法的线段绘制函数的头文件
#include "agg_ellipse_bresenham.h"     // 包含椭圆绘制的Bresenham算法的头文件
#include "agg_renderer_base.h"         // 包含渲染器基类的头文件
#include "agg_gamma_functions.h"       // 包含伽马函数的头文件
#include "agg_clip_liang_barsky.h"     // 包含梁-巴斯基裁剪算法的头文件

namespace agg
{

    //===================================================distance_interpolator0
    // 距离插值器类，用于计算点到直线距离的插值
    class distance_interpolator0
    {
    public:
        //---------------------------------------------------------------------
        // 构造函数，初始化插值器
        distance_interpolator0() {}
        
        // 根据给定的直线两端点和测试点，计算初始距离
        distance_interpolator0(int x1, int y1, int x2, int y2, int x, int y) :
            m_dx(line_mr(x2) - line_mr(x1)),                              // 计算线段x方向的整数坐标增量
            m_dy(line_mr(y2) - line_mr(y1)),                              // 计算线段y方向的整数坐标增量
            m_dist((line_mr(x + line_subpixel_scale/2) - line_mr(x2)) * m_dy - 
                   (line_mr(y + line_subpixel_scale/2) - line_mr(y2)) * m_dx)  // 计算初始距离
        {
            m_dx <<= line_mr_subpixel_shift;                              // 将m_dx左移subpixel位，获得更高的精度
            m_dy <<= line_mr_subpixel_shift;                              // 将m_dy左移subpixel位，获得更高的精度
        }

        //---------------------------------------------------------------------
        // 增加x方向上的距离
        void inc_x() { m_dist += m_dy; }
        
        // 获取当前距离
        int  dist() const { return m_dist; }

    private:
        //---------------------------------------------------------------------
        int m_dx;   // x方向的整数坐标增量
        int m_dy;   // y方向的整数坐标增量
        int m_dist; // 当前距离
    };

    //==================================================distance_interpolator00
    // 距离插值器类的第二个版本，具体实现细节未给出
    class distance_interpolator00
    {
    //===================================================distance_interpolator00
    // 构造函数，用于初始化距离插值器对象
    public:
        //---------------------------------------------------------------------
        // 默认构造函数，无参数
        distance_interpolator00() {}

        // 带参数的构造函数，用给定的坐标和数据初始化距离插值器对象
        // xc, yc: 中心坐标
        // x1, y1: 点1坐标
        // x2, y2: 点2坐标
        // x, y: 待插值的坐标
        distance_interpolator00(int xc, int yc, 
                                int x1, int y1, int x2, int y2, 
                                int x,  int y) :
            // 计算点1到中心点的差值，左移固定小数位数，用于精确计算
            m_dx1(line_mr(x1) - line_mr(xc)),
            m_dy1(line_mr(y1) - line_mr(yc)),
            // 计算点2到中心点的差值，左移固定小数位数，用于精确计算
            m_dx2(line_mr(x2) - line_mr(xc)),
            m_dy2(line_mr(y2) - line_mr(yc)),
            // 计算待插值点到点1和点2的距离误差，用于后续插值计算
            m_dist1((line_mr(x + line_subpixel_scale/2) - line_mr(x1)) * m_dy1 - 
                    (line_mr(y + line_subpixel_scale/2) - line_mr(y1)) * m_dx1),
            m_dist2((line_mr(x + line_subpixel_scale/2) - line_mr(x2)) * m_dy2 - 
                    (line_mr(y + line_subpixel_scale/2) - line_mr(y2)) * m_dx2)
        {
            // 将差值左移，以适应固定小数位数的运算精度要求
            m_dx1 <<= line_mr_subpixel_shift;
            m_dy1 <<= line_mr_subpixel_shift;
            m_dx2 <<= line_mr_subpixel_shift;
            m_dy2 <<= line_mr_subpixel_shift;
        }

        //---------------------------------------------------------------------
        // 增加 x 方向的插值误差
        void inc_x() { m_dist1 += m_dy1; m_dist2 += m_dy2; }

        // 获取第一个距离插值误差
        int  dist1() const { return m_dist1; }

        // 获取第二个距离插值误差
        int  dist2() const { return m_dist2; }

    private:
        //---------------------------------------------------------------------
        // 中心点到点1和点2在 x 和 y 方向上的差值
        int m_dx1;
        int m_dy1;
        int m_dx2;
        int m_dy2;
        // 待插值点到点1和点2的距离误差
        int m_dist1;
        int m_dist2;
    };
    //================================================line_interpolator_aa_base

    //===================================================distance_interpolator1
    // 描述：距离插值器，计算两点间距离的插值器
    class distance_interpolator1 {
    public:
        //---------------------------------------------------------------------
        // 默认构造函数，无参数
        distance_interpolator1() {}

        // 参数化构造函数，接收两点坐标及当前点坐标，并计算距离插值器
        distance_interpolator1(int x1, int y1, int x2, int y2, int x, int y) :
            m_dx(x2 - x1),                               // 计算x方向增量
            m_dy(y2 - y1),                               // 计算y方向增量
            m_dist(iround(double(x + line_subpixel_scale/2 - x2) * double(m_dy) - 
                          double(y + line_subpixel_scale/2 - y2) * double(m_dx)))  // 计算初始距离
        {
            m_dx <<= line_subpixel_shift;                 // 将x方向增量左移line_subpixel_shift位，相当于乘以2的line_subpixel_shift次方
            m_dy <<= line_subpixel_shift;                 // 将y方向增量左移line_subpixel_shift位，相当于乘以2的line_subpixel_shift次方
        }

        //---------------------------------------------------------------------
        // 向x方向增加一步距离
        void inc_x() { m_dist += m_dy; }

        // 向x方向减少一步距离
        void dec_x() { m_dist -= m_dy; }

        // 向y方向增加一步距离
        void inc_y() { m_dist -= m_dx; }

        // 向y方向减少一步距离
        void dec_y() { m_dist += m_dx; }

        //---------------------------------------------------------------------
        // 向x方向增加一步距离，并根据dy的符号调整距离
        void inc_x(int dy)
        {
            m_dist += m_dy; 
            if(dy > 0) m_dist -= m_dx; 
            if(dy < 0) m_dist += m_dx; 
        }

        //---------------------------------------------------------------------
        // 向x方向减少一步距离，并根据dy的符号调整距离
        void dec_x(int dy)
        {
            m_dist -= m_dy; 
            if(dy > 0) m_dist -= m_dx; 
            if(dy < 0) m_dist += m_dx; 
        }

        //---------------------------------------------------------------------
        // 向y方向增加一步距离，并根据dx的符号调整距离
        void inc_y(int dx)
        {
            m_dist -= m_dx; 
            if(dx > 0) m_dist += m_dy; 
            if(dx < 0) m_dist -= m_dy; 
        }

        //---------------------------------------------------------------------
        // 向y方向减少一步距离，并根据dx的符号调整距离
        void dec_y(int dx)
        {
            m_dist += m_dx; 
            if(dx > 0) m_dist += m_dy; 
            if(dx < 0) m_dist -= m_dy; 
        }

        //---------------------------------------------------------------------
        // 返回当前距离
        int dist() const { return m_dist; }

        // 返回x方向增量
        int dx()   const { return m_dx;   }

        // 返回y方向增量
        int dy()   const { return m_dy;   }

    private:
        //---------------------------------------------------------------------
        int m_dx;        // x方向增量
        int m_dy;        // y方向增量
        int m_dist;      // 当前距离
    };





    //===================================================distance_interpolator2
    // 描述：距离插值器2，用于特定的距离计算
    class distance_interpolator2
    {
    private:
        //---------------------------------------------------------------------
        int m_dx;          // x方向增量
        int m_dy;          // y方向增量
        int m_dx_start;    // x方向初始增量
        int m_dy_start;    // y方向初始增量

        int m_dist;        // 当前距离
        int m_dist_start;  // 初始距离
    };





    //===================================================distance_interpolator3
    // 描述：距离插值器3，用于更复杂的距离计算
    class distance_interpolator3
    {
    private:
        //---------------------------------------------------------------------
        int m_dx;          // x方向增量
        int m_dy;          // y方向增量
        int m_dx_start;    // x方向初始增量
        int m_dy_start;    // y方向初始增量
        int m_dx_end;      // x方向结束增量
        int m_dy_end;      // y方向结束增量

        int m_dist;        // 当前距离
        int m_dist_start;  // 初始距离
        int m_dist_end;    // 结束距离
    };
    // 定义一个模板类 line_interpolator_aa_base，用于处理抗锯齿线段插值
    template<class Renderer> class line_interpolator_aa_base
    {
    private:
        // 私有构造函数，禁止复制构造
        line_interpolator_aa_base(const line_interpolator_aa_base<Renderer>&);
        // 禁止赋值操作符重载
        const line_interpolator_aa_base<Renderer>& 
            operator = (const line_interpolator_aa_base<Renderer>&);
    
    protected:
        // 线段参数指针
        line_parameters* m_lp;
        // DDA2 算法的线段插值器
        dda2_line_interpolator m_li;
        // 渲染器类型的引用
        renderer_type&         m_ren;
        // 线段长度
        int m_len;
        // 当前 x 坐标
        int m_x;
        // 当前 y 坐标
        int m_y;
        // 上一个 x 坐标
        int m_old_x;
        // 上一个 y 坐标
        int m_old_y;
        // 计数器
        int m_count;
        // 线段宽度
        int m_width;
        // 最大扩展距离
        int m_max_extent;
        // 步长
        int m_step;
        // 距离数组，存储最大宽度一半的距离
        int m_dist[max_half_width + 1];
        // 覆盖类型数组，存储最大宽度一半乘以2再加4的覆盖信息
        cover_type m_covers[max_half_width * 2 + 4];
    };
    // 定义公共成员
    public:
        // 定义渲染器类型为 Renderer
        typedef Renderer renderer_type;
        // 定义颜色类型为 Renderer 的 color_type
        typedef typename Renderer::color_type color_type;
        // 定义基类类型为 line_interpolator_aa_base<Renderer>
        typedef line_interpolator_aa_base<Renderer> base_type;

        //---------------------------------------------------------------------
        // 构造函数，接受渲染器和线条参数对象作为参数
        line_interpolator_aa0(renderer_type& ren, line_parameters& lp) :
            // 调用基类的构造函数进行初始化
            line_interpolator_aa_base<Renderer>(ren, lp),
            // 初始化 m_di 对象，使用 lp 中的坐标信息
            m_di(lp.x1, lp.y1, lp.x2, lp.y2, 
                 lp.x1 & ~line_subpixel_mask, lp.y1 & ~line_subpixel_mask)
        {
            // 调整基类中的线段插值器
            base_type::m_li.adjust_forward();
        }

        //---------------------------------------------------------------------
        // 水平步进函数，返回布尔值表示是否还有下一步
        bool step_hor()
        {
            int dist; // 距离
            int dy;   // 垂直距离
            // 计算水平步进量并获取覆盖信息
            int s1 = base_type::step_hor_base(m_di);
            cover_type* p0 = base_type::m_covers + base_type::max_half_width + 2;
            cover_type* p1 = p0;

            // 在覆盖数组中存储当前步进量对应的覆盖值
            *p1++ = (cover_type)base_type::m_ren.cover(s1);

            dy = 1;
            // 循环计算并存储正向覆盖值，直到距离超过宽度
            while((dist = base_type::m_dist[dy] - s1) <= base_type::m_width)
            {
                *p1++ = (cover_type)base_type::m_ren.cover(dist);
                ++dy;
            }

            dy = 1;
            // 循环计算并存储反向覆盖值，直到距离超过宽度
            while((dist = base_type::m_dist[dy] + s1) <= base_type::m_width)
            {
                *--p0 = (cover_type)base_type::m_ren.cover(dist);
                ++dy;
            }
            // 在渲染器中进行竖直方向的实心填充混合操作
            base_type::m_ren.blend_solid_vspan(base_type::m_x, 
                                               base_type::m_y - dy + 1, 
                                               unsigned(p1 - p0), 
                                               p0);
            // 返回是否还有下一步的布尔值
            return ++base_type::m_step < base_type::m_count;
        }

        //---------------------------------------------------------------------
        // 垂直步进函数，返回布尔值表示是否还有下一步
        bool step_ver()
        {
            int dist; // 距离
            int dx;   // 水平距离
            // 计算垂直步进量并获取覆盖信息
            int s1 = base_type::step_ver_base(m_di);
            cover_type* p0 = base_type::m_covers + base_type::max_half_width + 2;
            cover_type* p1 = p0;

            // 在覆盖数组中存储当前步进量对应的覆盖值
            *p1++ = (cover_type)base_type::m_ren.cover(s1);

            dx = 1;
            // 循环计算并存储正向覆盖值，直到距离超过宽度
            while((dist = base_type::m_dist[dx] - s1) <= base_type::m_width)
            {
                *p1++ = (cover_type)base_type::m_ren.cover(dist);
                ++dx;
            }

            dx = 1;
            // 循环计算并存储反向覆盖值，直到距离超过宽度
            while((dist = base_type::m_dist[dx] + s1) <= base_type::m_width)
            {
                *--p0 = (cover_type)base_type::m_ren.cover(dist);
                ++dx;
            }
            // 在渲染器中进行水平方向的实心填充混合操作
            base_type::m_ren.blend_solid_hspan(base_type::m_x - dx + 1, 
                                               base_type::m_y,
                                               unsigned(p1 - p0), 
                                               p0);
            // 返回是否还有下一步的布尔值
            return ++base_type::m_step < base_type::m_count;
        }
    // 定义一个模板类 line_interpolator_aa0，继承自 line_interpolator_aa_base<Renderer>
    template<class Renderer> class line_interpolator_aa0 :
    public line_interpolator_aa_base<Renderer>
    {
    private:
        // 禁止拷贝构造函数的使用
        line_interpolator_aa0(const line_interpolator_aa0<Renderer>&);
        // 禁止赋值操作符的使用
        const line_interpolator_aa0<Renderer>& 
            operator = (const line_interpolator_aa0<Renderer>&);
    
        //---------------------------------------------------------------------
        // 使用 distance_interpolator1 类型的成员变量 m_di
        distance_interpolator1 m_di; 
    };
    
    
    
    
    
    
    
    // 定义一个模板类 line_interpolator_aa1，继承自 line_interpolator_aa_base<Renderer>
    template<class Renderer> class line_interpolator_aa1 :
    public line_interpolator_aa_base<Renderer>
    {
    private:
        // 禁止拷贝构造函数的使用
        line_interpolator_aa1(const line_interpolator_aa1<Renderer>&);
        // 禁止赋值操作符的使用
        const line_interpolator_aa1<Renderer>& 
            operator = (const line_interpolator_aa1<Renderer>&);
    
        //---------------------------------------------------------------------
        // 使用 distance_interpolator2 类型的成员变量 m_di
        distance_interpolator2 m_di; 
    };
    
    
    
    
    
    
    
    
    
    
    
    
    // 定义一个模板类 line_interpolator_aa2，继承自 line_interpolator_aa_base<Renderer>
    template<class Renderer> class line_interpolator_aa2 :
    public line_interpolator_aa_base<Renderer>
    {
    private:
        // 禁止拷贝构造函数的使用
        line_interpolator_aa2(const line_interpolator_aa2<Renderer>&);
        // 禁止赋值操作符的使用
        const line_interpolator_aa2<Renderer>& 
            operator = (const line_interpolator_aa2<Renderer>&);
    
        //---------------------------------------------------------------------
        // 使用 distance_interpolator2 类型的成员变量 m_di
        distance_interpolator2 m_di; 
    };
    
    
    
    
    
    
    
    
    
    
    // 定义一个模板类 line_interpolator_aa3，继承自 line_interpolator_aa_base<Renderer>
    template<class Renderer> class line_interpolator_aa3 :
    public line_interpolator_aa_base<Renderer>
    {
    private:
        // 禁止拷贝构造函数的使用
        line_interpolator_aa3(const line_interpolator_aa3<Renderer>&);
        // 禁止赋值操作符的使用
        const line_interpolator_aa3<Renderer>& 
            operator = (const line_interpolator_aa3<Renderer>&);
    
        //---------------------------------------------------------------------
        // 使用 distance_interpolator3 类型的成员变量 m_di
        distance_interpolator3 m_di; 
    };
    
    
    
    
    // 定义类 line_profile_aa
    // 详见实现文件 agg_line_profile_aa.cpp
    class line_profile_aa
    {
    // 公共部分开始

    //---------------------------------------------------------------------
    // value_type 类型定义为 int8u
    typedef int8u value_type;

    // subpixel_scale_e 枚举定义
    enum subpixel_scale_e
    {
        // subpixel_shift 设置为 line_subpixel_shift 的值
        subpixel_shift = line_subpixel_shift,
        // subpixel_scale 为 subpixel_shift 的 2 次方
        subpixel_scale = 1 << subpixel_shift,
        // subpixel_mask 为 subpixel_scale 减 1
        subpixel_mask  = subpixel_scale - 1
    };

    // aa_scale_e 枚举定义
    enum aa_scale_e
    {
        // aa_shift 设置为 8
        aa_shift = 8,
        // aa_scale 为 aa_shift 的 2 次方
        aa_scale = 1 << aa_shift,
        // aa_mask 为 aa_scale 减 1
        aa_mask  = aa_scale - 1
    };
    
    // 公共部分结束

    //---------------------------------------------------------------------
    // line_profile_aa 类定义开始
    class line_profile_aa()
    {
    public:
        // 构造函数，初始化各个成员变量
        line_profile_aa() : 
            m_subpixel_width(0),       // m_subpixel_width 初始化为 0
            m_min_width(1.0),          // m_min_width 初始化为 1.0
            m_smoother_width(1.0)      // m_smoother_width 初始化为 1.0
        {
            int i;
            // 初始化 m_gamma 数组，从 0 到 aa_scale - 1 的值依次为 0, 1, 2, ..., aa_scale - 1
            for(i = 0; i < aa_scale; i++) m_gamma[i] = (value_type)i;
        }

        //---------------------------------------------------------------------
        // 模板构造函数，初始化并调用 gamma 和 width 方法
        template<class GammaF> 
        line_profile_aa(double w, const GammaF& gamma_function) : 
            m_subpixel_width(0),       // m_subpixel_width 初始化为 0
            m_min_width(1.0),          // m_min_width 初始化为 1.0
            m_smoother_width(1.0)      // m_smoother_width 初始化为 1.0
        {
            gamma(gamma_function);      // 调用 gamma 函数，设置 m_gamma 数组
            width(w);                   // 调用 width 函数，设置 m_min_width 和 m_smoother_width
        }

        //---------------------------------------------------------------------
        // 设置最小宽度
        void min_width(double w) { m_min_width = w; }
        
        // 设置平滑宽度
        void smoother_width(double w) { m_smoother_width = w; }

        //---------------------------------------------------------------------
        // 模板函数，设置 gamma 值
        template<class GammaF> void gamma(const GammaF& gamma_function)
        { 
            int i;
            // 根据 gamma_function 计算并设置 m_gamma 数组的值
            for(i = 0; i < aa_scale; i++)
            {
                m_gamma[i] = value_type(
                    uround(gamma_function(double(i) / aa_mask) * aa_mask));
            }
        }

        // 设置宽度的抽象方法声明
        void width(double w);

        // 返回 profile 大小
        unsigned profile_size() const { return m_profile.size(); }
        
        // 返回 subpixel_width 值
        int subpixel_width() const { return m_subpixel_width; }

        //---------------------------------------------------------------------
        // 返回最小宽度
        double min_width() const { return m_min_width; }
        
        // 返回平滑宽度
        double smoother_width() const { return m_smoother_width; }

        //---------------------------------------------------------------------
        // 返回给定距离 dist 处的 profile 值
        value_type value(int dist) const
        {
            return m_profile[dist + subpixel_scale*2];
        }

    private:
        // 私有拷贝构造函数
        line_profile_aa(const line_profile_aa&);
        
        // 私有赋值运算符重载
        const line_profile_aa& operator = (const line_profile_aa&);

        // profile 方法声明
        value_type* profile(double w);
        
        // set 方法声明
        void set(double center_width, double smoother_width);

        //---------------------------------------------------------------------
        // pod_array 类型的 m_profile 对象
        pod_array<value_type> m_profile;
        
        // value_type 类型的 m_gamma 数组
        value_type            m_gamma[aa_scale];
        
        // 整型 m_subpixel_width 变量
        int                   m_subpixel_width;
        
        // 双精度浮点型 m_min_width 变量
        double                m_min_width;
        
        // 双精度浮点型 m_smoother_width 变量
        double                m_smoother_width;
    };
    // line_profile_aa 类定义结束

    //======================================================
    // renderer_outline_aa 模板类定义开始
    template<class BaseRenderer> class renderer_outline_aa
    {
    # 声明私有成员变量，指向基本渲染类型的指针
    private:
        base_ren_type*         m_ren;
    # 声明私有成员变量，指向线条剖面（antialiasing）的指针
        line_profile_aa* m_profile; 
    # 声明私有成员变量，表示颜色类型
        color_type             m_color;
    # 声明私有成员变量，表示矩形剪裁框的坐标
        rect_i                 m_clip_box;
    # 声明私有成员变量，表示是否启用剪裁功能
        bool                   m_clipping;
    };
}


注释：

// 这是 C/C++ 中的预处理器指令，表示结束一个条件编译段落
// 通常与 #ifdef 或 #if 配对使用，用于条件性地包含代码



#endif


注释：

// 这是 C/C++ 中的预处理器指令，表示结束一个条件编译段落
// 通常与 #ifdef 或 #if 配对使用，用于条件性地包含代码
```