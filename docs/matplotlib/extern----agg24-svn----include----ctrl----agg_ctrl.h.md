# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\ctrl\agg_ctrl.h`

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
// Function render_ctrl
//
//----------------------------------------------------------------------------

#ifndef AGG_CTRL_INCLUDED
#define AGG_CTRL_INCLUDED

#include "agg_trans_affine.h"
#include "agg_renderer_scanline.h"

namespace agg
{

    //--------------------------------------------------------------------ctrl
    // 控件基类
    class ctrl
    {
    public:
        //--------------------------------------------------------------------
        // 虚析构函数
        virtual ~ctrl() {}
        
        // 构造函数，初始化控件位置和翻转标志
        ctrl(double x1, double y1, double x2, double y2, bool flip_y) :
            m_x1(x1), m_y1(y1), m_x2(x2), m_y2(y2), 
            m_flip_y(flip_y),
            m_mtx(0)
        {
        }

        //--------------------------------------------------------------------
        // 纯虚函数，判断给定坐标是否在控件区域内
        virtual bool in_rect(double x, double y) const = 0;
        
        // 纯虚函数，处理鼠标按下事件
        virtual bool on_mouse_button_down(double x, double y) = 0;
        
        // 纯虚函数，处理鼠标释放事件
        virtual bool on_mouse_button_up(double x, double y) = 0;
        
        // 纯虚函数，处理鼠标移动事件
        virtual bool on_mouse_move(double x, double y, bool button_flag) = 0;
        
        // 纯虚函数，处理方向键事件
        virtual bool on_arrow_keys(bool left, bool right, bool down, bool up) = 0;

        //--------------------------------------------------------------------
        // 设置变换矩阵
        void transform(const trans_affine& mtx) { m_mtx = &mtx; }
        
        // 清除变换矩阵
        void no_transform() { m_mtx = 0; }

        //--------------------------------------------------------------------
        // 根据翻转标志和变换矩阵转换坐标
        void transform_xy(double* x, double* y) const
        {
            if(m_flip_y) *y = m_y1 + m_y2 - *y;
            if(m_mtx) m_mtx->transform(x, y);
        }

        //--------------------------------------------------------------------
        // 根据翻转标志和变换矩阵逆转换坐标
        void inverse_transform_xy(double* x, double* y) const
        {
            if(m_mtx) m_mtx->inverse_transform(x, y);
            if(m_flip_y) *y = m_y1 + m_y2 - *y;
        }

        //--------------------------------------------------------------------
        // 返回当前变换的比例因子
        double scale() const { return m_mtx ? m_mtx->scale() : 1.0; }

    private:
        // 禁止拷贝构造和赋值操作
        ctrl(const ctrl&);
        const ctrl& operator = (const ctrl&);

    protected:
        double m_x1;                // 控件左上角 x 坐标
        double m_y1;                // 控件左上角 y 坐标
        double m_x2;                // 控件右下角 x 坐标
        double m_y2;                // 控件右下角 y 坐标

    private:
        bool m_flip_y;              // 控件是否翻转 y 轴
        const trans_affine* m_mtx;  // 控件的变换矩阵
    };
    //--------------------------------------------------------------------
    template<class Rasterizer, class Scanline, class Renderer, class Ctrl> 
    void render_ctrl(Rasterizer& ras, Scanline& sl, Renderer& r, Ctrl& c)
    {
        // 初始化循环变量 i
        unsigned i;
        // 循环遍历控制器 c 中的路径数量
        for(i = 0; i < c.num_paths(); i++)
        {
            // 重置光栅化器
            ras.reset();
            // 向光栅化器中添加控制器 c 的第 i 条路径
            ras.add_path(c, i);
            // 使用抗锯齿实现的单色填充渲染器渲染扫描线
            render_scanlines_aa_solid(ras, sl, r, c.color(i));
        }
    }
    
    
    //--------------------------------------------------------------------
    template<class Rasterizer, class Scanline, class Renderer, class Ctrl> 
    void render_ctrl_rs(Rasterizer& ras, Scanline& sl, Renderer& r, Ctrl& c)
    {
        // 初始化循环变量 i
        unsigned i;
        // 循环遍历控制器 c 中的路径数量
        for(i = 0; i < c.num_paths(); i++)
        {
            // 重置光栅化器
            ras.reset();
            // 向光栅化器中添加控制器 c 的第 i 条路径
            ras.add_path(c, i);
            // 设置渲染器颜色为控制器 c 的第 i 条路径的颜色
            r.color(c.color(i));
            // 渲染扫描线
            render_scanlines(ras, sl, r);
        }
    }
}


这行代码表示一个C/C++代码块的结束，通常用于结束一个函数、条件语句或循环。


#endif


这行代码通常用于条件编译（conditional compilation），在预处理阶段检查是否定义了与`#ifdef`或`#ifndef`匹配的宏，并且如果有定义，则编译与之对应的代码。在这里，`#endif`表示条件编译块的结束。

这两行代码一起用于控制代码的编译范围，只有在特定条件下（通常是预定义的宏存在或不存在）才会编译其中的代码。
```