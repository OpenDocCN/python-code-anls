# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_rasterizer_scanline_aa_nogamma.h`

```
#ifndef AGG_RASTERIZER_SCANLINE_AA_NOGAMMA_INCLUDED
#define AGG_RASTERIZER_SCANLINE_AA_NOGAMMA_INCLUDED

// 定义防止重复包含的预处理指令，用于条件编译防止头文件重复引入


#include "agg_rasterizer_cells_aa.h"
#include "agg_rasterizer_sl_clip.h"

// 引入所需的头文件，包括像素单元的抗锯齿和裁剪扫描线光栅化器的头文件


namespace agg
{

// 命名空间agg的开始


struct cell_aa
{
    int x;
    int y;
    int cover;
    int area;

    void initial()
    {
        x = 0x7FFFFFFF;
        y = 0x7FFFFFFF;
        cover = 0;
        area  = 0;
    }

    void style(const cell_aa&) {}

    int not_equal(int ex, int ey, const cell_aa&) const
    {
        return ex != x || ey != y;
    }
};

// 定义一个像素单元结构体cell_aa，表示像素的位置和覆盖信息，提供了初始化函数和比较函数


// Polygon rasterizer that is used to render filled polygons with 
// high-quality Anti-Aliasing. Internally, by default, the class uses 
// integer coordinates in format 24.8, i.e. 24 bits for integer part 
// and 8 bits for fractional - see poly_subpixel_shift. This class can be 
// used in the following  way:
//
// 1. filling_rule(filling_rule_e ft) - optional.
//
// 2. gamma() - optional.
//
// 3. reset()
//
// 4. move_to(x, y) / line_to(x, y) - make the polygon. One can create 
//    more than one contour, but each contour must consist of at least 3
//    vertices, i.e. move_to(x1, y1); line_to(x2, y2); line_to(x3, y3);
struct rasterizer_scanline_aa_nogamma
{

// 多边形光栅化器，用于渲染带有高质量抗锯齿效果的填充多边形。默认使用24.8格式的整数坐标，24位用于整数部分，8位用于小数部分。支持以下操作：
// 1. filling_rule(filling_rule_e ft) - 可选
// 2. gamma() - 可选
// 3. reset()
// 4. move_to(x, y) / line_to(x, y) - 创建多边形，每个轮廓至少包含3个顶点
    //  是定义三角形的顶点的绝对最小值。
    //  该算法不检查顶点的数量或它们坐标的巧合，但在最坏情况下，它可能不会绘制任何内容。
    //  顶点的顺序（顺时针或逆时针）对于使用非零填充规则（fill_non_zero）是重要的。
    //  在这种情况下，所有轮廓的顶点顺序必须相同，如果要使交叉的多边形没有“孔”。
    //  实际上，您可以使用不同的顶点顺序。如果轮廓不相交，则顺序无关紧要。如果相交，
    //  具有相同顶点顺序的轮廓将被渲染为没有“孔”，而具有不同顺序的相交轮廓将具有“孔”。
    //
    //  可以在“扫描”之前随时调用 filling_rule() 和 gamma()。
    //------------------------------------------------------------------------
    template<class Clip=rasterizer_sl_clip_int> class rasterizer_scanline_aa_nogamma
    {
        enum status
        {
            status_initial,  // 初始状态
            status_move_to,  // 移动到新位置状态
            status_line_to,  // 画线状态
            status_closed    // 轮廓封闭状态
        };

    private:
        //--------------------------------------------------------------------
        // 禁止复制
        rasterizer_scanline_aa_nogamma(const rasterizer_scanline_aa_nogamma<Clip>&);
        const rasterizer_scanline_aa_nogamma<Clip>& 
        operator = (const rasterizer_scanline_aa_nogamma<Clip>&);

    private:
        rasterizer_cells_aa<cell_aa> m_outline;  // 光栅化单元
        clip_type      m_clipper;        // 裁剪器类型
        filling_rule_e m_filling_rule;   // 填充规则
        bool           m_auto_close;     // 自动封闭标志
        coord_type     m_start_x;        // 起始 X 坐标
        coord_type     m_start_y;        // 起始 Y 坐标
        unsigned       m_status;         // 状态
        int            m_scan_y;         // 扫描线 Y 坐标
    };

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_scanline_aa_nogamma<Clip>::reset() 
    { 
        m_outline.reset();  // 重置光栅化单元
        m_status = status_initial;  // 将状态设置为初始状态
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_scanline_aa_nogamma<Clip>::filling_rule(filling_rule_e filling_rule) 
    { 
        m_filling_rule = filling_rule;  // 设置填充规则
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_scanline_aa_nogamma<Clip>::clip_box(double x1, double y1, 
                                                double x2, double y2)
    {
        reset();  // 重置状态和光栅化单元
        m_clipper.clip_box(conv_type::upscale(x1), conv_type::upscale(y1), 
                           conv_type::upscale(x2), conv_type::upscale(y2));  // 裁剪框
    }
    // 重置剪裁器和多边形轮廓状态
    void rasterizer_scanline_aa_nogamma<Clip>::reset_clipping()
    {
        // 调用reset函数重置当前对象状态
        reset();
        // 调用剪裁器对象的reset_clipping方法重置剪裁状态
        m_clipper.reset_clipping();
    }
    
    //------------------------------------------------------------------------
    // 关闭多边形，如果当前状态为线段状态，则添加起点到终点的线段到剪裁器
    template<class Clip> 
    void rasterizer_scanline_aa_nogamma<Clip>::close_polygon()
    {
        if(m_status == status_line_to)
        {
            // 将当前多边形的起点到终点的线段添加到剪裁器
            m_clipper.line_to(m_outline, m_start_x, m_start_y);
            // 设置当前状态为已闭合状态
            m_status = status_closed;
        }
    }
    
    //------------------------------------------------------------------------
    // 移动到指定坐标点，更新起点坐标并设置状态为移动状态
    template<class Clip> 
    void rasterizer_scanline_aa_nogamma<Clip>::move_to(int x, int y)
    {
        // 如果多边形轮廓已排序，则重置状态
        if(m_outline.sorted()) reset();
        // 如果需要自动闭合，则关闭当前多边形
        if(m_auto_close) close_polygon();
        // 将坐标x和y按比例缩小，并将起始点更新为这些坐标，并移动到此点
        m_clipper.move_to(m_start_x = conv_type::downscale(x), 
                          m_start_y = conv_type::downscale(y));
        // 设置当前状态为移动状态
        m_status = status_move_to;
    }
    
    //------------------------------------------------------------------------
    // 添加直线到指定坐标点，按比例缩小坐标后添加到剪裁器
    template<class Clip> 
    void rasterizer_scanline_aa_nogamma<Clip>::line_to(int x, int y)
    {
        // 将线段起点到指定坐标(x, y)的线段添加到剪裁器
        m_clipper.line_to(m_outline, 
                          conv_type::downscale(x), 
                          conv_type::downscale(y));
        // 设置当前状态为线段状态
        m_status = status_line_to;
    }
    
    //------------------------------------------------------------------------
    // 移动到指定双精度浮点坐标点，更新起点坐标并设置状态为移动状态
    template<class Clip> 
    void rasterizer_scanline_aa_nogamma<Clip>::move_to_d(double x, double y) 
    { 
        // 如果多边形轮廓已排序，则重置状态
        if(m_outline.sorted()) reset();
        // 如果需要自动闭合，则关闭当前多边形
        if(m_auto_close) close_polygon();
        // 将坐标x和y按比例放大，并将起始点更新为这些坐标，并移动到此点
        m_clipper.move_to(m_start_x = conv_type::upscale(x), 
                          m_start_y = conv_type::upscale(y)); 
        // 设置当前状态为移动状态
        m_status = status_move_to;
    }
    
    //------------------------------------------------------------------------
    // 添加直线到指定双精度浮点坐标点，按比例放大坐标后添加到剪裁器
    template<class Clip> 
    void rasterizer_scanline_aa_nogamma<Clip>::line_to_d(double x, double y) 
    { 
        // 将线段起点到指定双精度浮点坐标(x, y)的线段添加到剪裁器
        m_clipper.line_to(m_outline, 
                          conv_type::upscale(x), 
                          conv_type::upscale(y)); 
        // 设置当前状态为线段状态
        m_status = status_line_to;
    }
    
    //------------------------------------------------------------------------
    // 添加顶点到多边形，根据指令类型调用相应的处理方法：移动到、顶点、闭合多边形
    template<class Clip> 
    void rasterizer_scanline_aa_nogamma<Clip>::add_vertex(double x, double y, unsigned cmd)
    {
        // 如果指令是移动到类型，则调用move_to_d方法移动到指定坐标点
        if(is_move_to(cmd)) 
        {
            move_to_d(x, y);
        }
        else 
        // 如果指令是顶点类型，则调用line_to_d方法添加直线到指定坐标点
        if(is_vertex(cmd))
        {
            line_to_d(x, y);
        }
        else
        // 如果指令是闭合多边形类型，则调用close_polygon方法关闭当前多边形
        if(is_close(cmd))
        {
            close_polygon();
        }
    }
    
    //------------------------------------------------------------------------
    // 处理给定的边界点坐标，但没有提供具体实现
    template<class Clip> 
    void rasterizer_scanline_aa_nogamma<Clip>::edge(int x1, int y1, int x2, int y2)
    {
        // 略，此处通常用于处理边界点，具体实现未提供
    }
    //------------------------------------------------------------------------
    {
        // 如果轮廓已经排序，则重置状态
        if(m_outline.sorted()) reset();
        // 移动裁剪器到指定的起始点（下采样）
        m_clipper.move_to(conv_type::downscale(x1), conv_type::downscale(y1));
        // 从起始点画直线到指定的结束点（下采样）
        m_clipper.line_to(m_outline, 
                          conv_type::downscale(x2), 
                          conv_type::downscale(y2));
        // 设置当前状态为移动到状态
        m_status = status_move_to;
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_scanline_aa_nogamma<Clip>::edge_d(double x1, double y1, 
                                              double x2, double y2)
    {
        // 如果轮廓已经排序，则重置状态
        if(m_outline.sorted()) reset();
        // 移动裁剪器到指定的起始点（上采样）
        m_clipper.move_to(conv_type::upscale(x1), conv_type::upscale(y1)); 
        // 从起始点画直线到指定的结束点（上采样）
        m_clipper.line_to(m_outline, 
                          conv_type::upscale(x2), 
                          conv_type::upscale(y2)); 
        // 设置当前状态为移动到状态
        m_status = status_move_to;
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_scanline_aa_nogamma<Clip>::sort()
    {
        // 如果启用自动闭合多边形，先关闭多边形
        if(m_auto_close) close_polygon();
        // 对轮廓的单元格进行排序
        m_outline.sort_cells();
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    AGG_INLINE bool rasterizer_scanline_aa_nogamma<Clip>::rewind_scanlines()
    {
        // 如果启用自动闭合多边形，先关闭多边形
        if(m_auto_close) close_polygon();
        // 对轮廓的单元格进行排序
        m_outline.sort_cells();
        // 如果轮廓中没有单元格，则返回false
        if(m_outline.total_cells() == 0) 
        {
            return false;
        }
        // 设置扫描线的起始y坐标为轮廓的最小y坐标
        m_scan_y = m_outline.min_y();
        return true;
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    AGG_INLINE bool rasterizer_scanline_aa_nogamma<Clip>::navigate_scanline(int y)
    {
        // 如果启用自动闭合多边形，先关闭多边形
        if(m_auto_close) close_polygon();
        // 对轮廓的单元格进行排序
        m_outline.sort_cells();
        // 如果轮廓中没有单元格或者指定的y坐标超出了轮廓的范围，则返回false
        if(m_outline.total_cells() == 0 || 
           y < m_outline.min_y() || 
           y > m_outline.max_y()) 
        {
            return false;
        }
        // 设置当前扫描线的y坐标
        m_scan_y = y;
        return true;
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    bool rasterizer_scanline_aa_nogamma<Clip>::hit_test(int tx, int ty)
    {
        // 导航到指定的扫描线y坐标，如果失败则返回false
        if(!navigate_scanline(ty)) return false;
        // 创建扫描线命中测试对象
        scanline_hit_test sl(tx);
        // 执行扫描线的扫描
        sweep_scanline(sl);
        // 返回扫描线是否命中
        return sl.hit();
    }
}


注释：

// 结束一个函数定义的标准语法，表示函数定义的结束



#endif


注释：

// 预处理指令，用于条件编译，指示如果未定义相应的宏，则跳过后续代码段
// 在此处，#ifdef 检查前面是否有定义过相应的宏，#endif 标记条件编译的结束
```