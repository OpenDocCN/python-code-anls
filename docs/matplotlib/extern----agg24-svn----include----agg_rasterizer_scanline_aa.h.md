# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_rasterizer_scanline_aa.h`

```py
#ifndef AGG_RASTERIZER_SCANLINE_AA_INCLUDED
#define AGG_RASTERIZER_SCANLINE_AA_INCLUDED


// 如果未定义 AGG_RASTERIZER_SCANLINE_AA_INCLUDED 宏，则定义它，防止重复包含


#include "agg_rasterizer_cells_aa.h"
#include "agg_rasterizer_sl_clip.h"
#include "agg_rasterizer_scanline_aa_nogamma.h"
#include "agg_gamma_functions.h"


// 包含其他头文件，这些文件可能是该头文件所依赖的其他组件或功能的实现


namespace agg
{
    //==================================================rasterizer_scanline_aa
    // Anti-Grain Geometry 中用于渲染带有高质量抗锯齿效果的填充多边形的光栅化器。
    // 默认情况下，类使用 24.8 格式的整数坐标，即 24 位用于整数部分，8 位用于小数部分 - 参见 poly_subpixel_shift。
    // 可以按以下方式使用该类：
    //
    // 1. filling_rule(filling_rule_e ft) - 可选。
    //
    // 2. gamma() - 可选。
    //
    // 3. reset()
    //
    // 4. move_to(x, y) / line_to(x, y) - 构造多边形。可以创建多个轮廓，但每个轮廓至少必须包含3个顶点，
    //    即 move_to(x1, y1); line_to(x2, y2); line_to(x3, y3); 是定义三角形的最低顶点数。
    //    算法不检查顶点数量或它们的坐标是否重合，但在最坏的情况下，不会绘制任何内容。
    //    在使用非零填充规则 (fill_non_zero) 时，顶点的顺序（顺时针或逆时针）很重要。
    //    在这种情况下，如果希望交叉的多边形没有“孔”，所有轮廓的顶点顺序必须相同。


    // 接下来的部分可能包括更多成员函数、变量和注释，用于详细说明该类的用法和内部实现。
    // You actually can use different vertices order. If the contours do not 
    // intersect each other the order is not important anyway. If they do, 
    // contours with the same vertex order will be rendered without "holes" 
    // while the intersecting contours with different orders will have "holes".
    //
    // filling_rule() and gamma() can be called anytime before "sweeping".
    //------------------------------------------------------------------------
    template<class Clip=rasterizer_sl_clip_int> class rasterizer_scanline_aa
    {
        enum status
        {
            status_initial,    // Initial state of the rasterizer
            status_move_to,    // Moving to a new point
            status_line_to,    // Drawing a line from the current point
            status_closed      // Polygon has been closed
        };

    private:
        //--------------------------------------------------------------------
        // Disable copying
        rasterizer_scanline_aa(const rasterizer_scanline_aa<Clip>&);
        const rasterizer_scanline_aa<Clip>& 
        operator = (const rasterizer_scanline_aa<Clip>&);

    private:
        rasterizer_cells_aa<cell_aa> m_outline;  // Outline storage for the rasterizer
        clip_type      m_clipper;                // Clipper type for clipping operations
        int            m_gamma[aa_scale];        // Gamma correction values
        filling_rule_e m_filling_rule;           // Filling rule for polygon filling
        bool           m_auto_close;             // Auto close flag
        coord_type     m_start_x;                // Starting x-coordinate of current contour
        coord_type     m_start_y;                // Starting y-coordinate of current contour
        unsigned       m_status;                 // Current status of the rasterizer
        int            m_scan_y;                 // Current scanline y-coordinate
    };

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_scanline_aa<Clip>::reset() 
    { 
        m_outline.reset();    // Reset the outline storage
        m_status = status_initial;   // Set status to initial
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_scanline_aa<Clip>::filling_rule(filling_rule_e filling_rule) 
    { 
        m_filling_rule = filling_rule;   // Set the filling rule for polygon filling
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_scanline_aa<Clip>::clip_box(double x1, double y1, 
                                                double x2, double y2)
    {
        reset();    // Reset rasterizer state
        // Set clip box using upscaled coordinates
        m_clipper.clip_box(conv_type::upscale(x1), conv_type::upscale(y1), 
                           conv_type::upscale(x2), conv_type::upscale(y2));
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_scanline_aa<Clip>::reset_clipping()
    {
        reset();    // Reset rasterizer state
        m_clipper.reset_clipping();    // Reset clipping region in clipper
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_scanline_aa<Clip>::close_polygon()
    {
        if(m_status == status_line_to)
        {
            // If currently drawing a line, close the polygon and update status
            m_clipper.line_to(m_outline, m_start_x, m_start_y);
            m_status = status_closed;
        }
    }
    // 将当前位置移动到指定的坐标 (x, y)
    void rasterizer_scanline_aa<Clip>::move_to(int x, int y)
    {
        // 如果轮廓已经排序，重置轮廓
        if(m_outline.sorted()) reset();
        // 如果自动闭合开启，闭合当前多边形
        if(m_auto_close) close_polygon();
        // 将起始点移动到指定的 (x, y) 坐标，并将其下采样后的值分别赋给 m_start_x 和 m_start_y
        m_clipper.move_to(m_start_x = conv_type::downscale(x), 
                          m_start_y = conv_type::downscale(y));
        // 设置状态为移动到状态
        m_status = status_move_to;
    }
    
    //------------------------------------------------------------------------
    // 添加直线到指定坐标 (x, y)
    template<class Clip> 
    void rasterizer_scanline_aa<Clip>::line_to(int x, int y)
    {
        // 使用剪切器将直线添加到轮廓中，将坐标 (x, y) 下采样后传入
        m_clipper.line_to(m_outline, 
                          conv_type::downscale(x), 
                          conv_type::downscale(y));
        // 设置状态为直线到状态
        m_status = status_line_to;
    }
    
    //------------------------------------------------------------------------
    // 使用双精度坐标移动到指定位置 (x, y)
    template<class Clip> 
    void rasterizer_scanline_aa<Clip>::move_to_d(double x, double y) 
    { 
        // 如果轮廓已经排序，重置轮廓
        if(m_outline.sorted()) reset();
        // 如果自动闭合开启，闭合当前多边形
        if(m_auto_close) close_polygon();
        // 将起始点移动到指定的 (x, y) 双精度坐标，并将其上采样后的值分别赋给 m_start_x 和 m_start_y
        m_clipper.move_to(m_start_x = conv_type::upscale(x), 
                          m_start_y = conv_type::upscale(y)); 
        // 设置状态为移动到状态
        m_status = status_move_to;
    }
    
    //------------------------------------------------------------------------
    // 使用双精度坐标添加直线到指定位置 (x, y)
    template<class Clip> 
    void rasterizer_scanline_aa<Clip>::line_to_d(double x, double y) 
    { 
        // 使用剪切器将直线添加到轮廓中，将坐标 (x, y) 上采样后传入
        m_clipper.line_to(m_outline, 
                          conv_type::upscale(x), 
                          conv_type::upscale(y)); 
        // 设置状态为直线到状态
        m_status = status_line_to;
    }
    
    //------------------------------------------------------------------------
    // 添加顶点到指定位置 (x, y)，cmd 表示命令类型
    template<class Clip> 
    void rasterizer_scanline_aa<Clip>::add_vertex(double x, double y, unsigned cmd)
    {
        // 如果是移动到命令
        if(is_move_to(cmd)) 
        {
            // 调用双精度坐标移动到函数
            move_to_d(x, y);
        }
        else 
        // 如果是顶点命令
        if(is_vertex(cmd))
        {
            // 调用双精度坐标添加直线函数
            line_to_d(x, y);
        }
        else
        // 如果是闭合命令
        if(is_close(cmd))
        {
            // 闭合当前多边形
            close_polygon();
        }
    }
    
    //------------------------------------------------------------------------
    // 添加直线段，使用整数坐标 (x1, y1) 到 (x2, y2)
    template<class Clip> 
    void rasterizer_scanline_aa<Clip>::edge(int x1, int y1, int x2, int y2)
    {
        // 如果轮廓已经排序，重置轮廓
        if(m_outline.sorted()) reset();
        // 将起始点移动到 (x1, y1) 整数坐标，并将其下采样后传入
        m_clipper.move_to(conv_type::downscale(x1), conv_type::downscale(y1));
        // 使用剪切器将直线添加到轮廓中，将 (x2, y2) 下采样后传入
        m_clipper.line_to(m_outline, 
                          conv_type::downscale(x2), 
                          conv_type::downscale(y2));
        // 设置状态为移动到状态
        m_status = status_move_to;
    }
    
    //------------------------------------------------------------------------
    // 添加直线段，使用双精度坐标 (x1, y1) 到 (x2, y2)
    template<class Clip> 
    void rasterizer_scanline_aa<Clip>::edge_d(double x1, double y1, 
                                              double x2, double y2)
    {
        // 如果轮廓已经排序，重置轮廓
        if(m_outline.sorted()) reset();
        // 将起始点移动到 (x1, y1) 双精度坐标，并将其上采样后传入
        m_clipper.move_to(conv_type::upscale(x1), conv_type::upscale(y1)); 
        // 使用剪切器将直线添加到轮廓中，将 (x2, y2) 上采样后传入
        m_clipper.line_to(m_outline, 
                          conv_type::upscale(x2), 
                          conv_type::upscale(y2)); 
        // 设置状态为移动到状态
        m_status = status_move_to;
    }
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_scanline_aa<Clip>::sort()
    {
        // 如果设置了自动闭合选项，关闭多边形
        if(m_auto_close) close_polygon();
        // 对多边形轮廓进行单元格排序
        m_outline.sort_cells();
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    AGG_INLINE bool rasterizer_scanline_aa<Clip>::rewind_scanlines()
    {
        // 如果设置了自动闭合选项，关闭多边形
        if(m_auto_close) close_polygon();
        // 对多边形轮廓进行单元格排序
        m_outline.sort_cells();
        // 如果多边形轮廓中没有单元格，则返回false
        if(m_outline.total_cells() == 0) 
        {
            return false;
        }
        // 设置扫描线的初始位置为轮廓的最小y坐标
        m_scan_y = m_outline.min_y();
        return true;
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    AGG_INLINE bool rasterizer_scanline_aa<Clip>::navigate_scanline(int y)
    {
        // 如果设置了自动闭合选项，关闭多边形
        if(m_auto_close) close_polygon();
        // 对多边形轮廓进行单元格排序
        m_outline.sort_cells();
        // 如果多边形轮廓中没有单元格，或者指定的y坐标超出了轮廓的y范围，则返回false
        if(m_outline.total_cells() == 0 || 
           y < m_outline.min_y() || 
           y > m_outline.max_y()) 
        {
            return false;
        }
        // 设置扫描线的位置为指定的y坐标
        m_scan_y = y;
        return true;
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    bool rasterizer_scanline_aa<Clip>::hit_test(int tx, int ty)
    {
        // 导航到指定的扫描线位置ty，如果失败则返回false
        if(!navigate_scanline(ty)) return false;
        // 创建扫描线命中测试对象
        scanline_hit_test sl(tx);
        // 进行扫描线的扫描
        sweep_scanline(sl);
        // 返回扫描结果是否命中
        return sl.hit();
    }
}


注释：


// 关闭一个 #if 或 #ifdef 所开始的条件编译块
#endif


这段代码看起来是 C/C++ 的预处理器条件编译代码块的结尾部分。在这里，```}```py 闭合了之前打开的条件编译块，```#endif```py 则用于结束这个条件编译块，使得在条件满足时编译器可以继续处理下面的代码。
```