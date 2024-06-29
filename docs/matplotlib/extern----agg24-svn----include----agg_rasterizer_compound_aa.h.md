# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_rasterizer_compound_aa.h`

```
//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.3
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//
// Permission to copy, use, modify, sell and distribute this software 
// is granted provided this copyright notice appears in all copies. 
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//----------------------------------------------------------------------------
//
// The author gratefully acknowleges the support of David Turner, 
// Robert Wilhelm, and Werner Lemberg - the authors of the FreeType 
// libray - in producing this work. See http://www.freetype.org for details.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------
//
// Adaptation for 32-bit screen coordinates has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------
// 如果尚未包含 AGG_RASTERIZER_COMPOUND_AA_INCLUDED 的定义，则包含该头文件
#ifndef AGG_RASTERIZER_COMPOUND_AA_INCLUDED
#define AGG_RASTERIZER_COMPOUND_AA_INCLUDED

// 包含 AGG 中的相关头文件，包括像素单元的抗锯齿渲染器和裁剪器
#include "agg_rasterizer_cells_aa.h"
#include "agg_rasterizer_sl_clip.h"

// 命名空间 agg 开始
namespace agg
{

    //-----------------------------------------------------------cell_style_aa
    // 一个像素单元结构体，没有定义构造函数，以避免在分配单元数组时产生额外开销
    struct cell_style_aa
    {
        int   x;          // 像素单元的 x 坐标
        int   y;          // 像素单元的 y 坐标
        int   cover;      // 覆盖像素的覆盖值
        int   area;       // 像素单元的区域面积
        int16 left, right; // 像素单元的左右边界

        // 初始化像素单元
        void initial()
        {
            x     = 0x7FFFFFFF; // x 坐标初始化为最大整数值
            y     = 0x7FFFFFFF; // y 坐标初始化为最大整数值
            cover = 0;          // 覆盖值初始化为 0
            area  = 0;          // 区域面积初始化为 0
            left  = -1;         // 左边界初始化为 -1
            right = -1;         // 右边界初始化为 -1
        }

        // 设置像素单元的样式
        void style(const cell_style_aa& c)
        {
            left  = c.left;   // 设置左边界
            right = c.right;  // 设置右边界
        }

        // 检查像素单元是否与给定参数不相等
        int not_equal(int ex, int ey, const cell_style_aa& c) const
        {
            // 使用按位或操作符检查所有属性是否都不相等
            return (ex - x) | (ey - y) | (left - c.left) | (right - c.right);
        }
    };


    //===========================================================layer_order_e
    // 图层顺序枚举
    enum layer_order_e
    {
        layer_unsorted, // 未排序的图层
        layer_direct,   // 直接顺序的图层
        layer_inverse   // 逆序的图层
    };


    //==================================================rasterizer_compound_aa
    // 复合抗锯齿渲染器模板类
    template<class Clip=rasterizer_sl_clip_int> class rasterizer_compound_aa
    {
        // 定义结构体 `style_info`，用于保存样式信息
        struct style_info 
        { 
            unsigned start_cell; // 起始单元格索引
            unsigned num_cells;  // 单元格数量
            int      last_x;     // 最后一个 x 坐标
        };
    
        // 定义结构体 `cell_info`，保存单元格信息
        struct cell_info
        {
            int x, area, cover;  // x 坐标，区域，覆盖
        };
    
    private:
        void add_style(int style_id); // 声明一个私有函数 `add_style`，用于添加样式
    
        //--------------------------------------------------------------------
        // 禁止复制
        rasterizer_compound_aa(const rasterizer_compound_aa<Clip>&);  // 复制构造函数声明
        const rasterizer_compound_aa<Clip>& 
        operator = (const rasterizer_compound_aa<Clip>&);  // 赋值运算符声明
    
    private:
        rasterizer_cells_aa<cell_style_aa> m_outline;    // 单元格 AA 光栅化器对象
        clip_type              m_clipper;                // 剪切类型对象
        filling_rule_e         m_filling_rule;           // 填充规则枚举
        layer_order_e          m_layer_order;            // 图层顺序枚举
        pod_vector<style_info> m_styles;                 // 活动样式集合
        pod_vector<unsigned>   m_ast;                    // 活动样式表（唯一值）
        pod_vector<int8u>      m_asm;                    // 活动样式掩码
        pod_vector<cell_info>  m_cells;                  // 单元格信息集合
        pod_vector<cover_type> m_cover_buf;              // 覆盖缓冲区
    
        int        m_min_style;     // 最小样式值
        int        m_max_style;     // 最大样式值
        coord_type m_start_x;       // 起始 x 坐标
        coord_type m_start_y;       // 起始 y 坐标
        int        m_scan_y;        // 扫描 y 坐标
        int        m_sl_start;      // SL 起始索引
        unsigned   m_sl_len;        // SL 长度
    };
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::reset() 
    { 
        m_outline.reset();        // 重置单元格 AA 光栅化器对象
        m_min_style =  0x7FFFFFFF; // 初始化最小样式为最大整数值
        m_max_style = -0x7FFFFFFF; // 初始化最大样式为负的最大整数值
        m_scan_y    =  0x7FFFFFFF; // 初始化扫描 y 坐标为最大整数值
        m_sl_start  =  0;          // 初始化 SL 起始索引为 0
        m_sl_len    = 0;           // 初始化 SL 长度为 0
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::filling_rule(filling_rule_e filling_rule) 
    { 
        m_filling_rule = filling_rule;  // 设置填充规则
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::layer_order(layer_order_e order)
    {
        m_layer_order = order;  // 设置图层顺序
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::clip_box(double x1, double y1, 
                                                double x2, double y2)
    {
        reset();  // 重置光栅化器状态
        m_clipper.clip_box(conv_type::upscale(x1), conv_type::upscale(y1), 
                           conv_type::upscale(x2), conv_type::upscale(y2));  // 剪切框选范围
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::reset_clipping()
    {
        reset();           // 重置光栅化器状态
        m_clipper.reset_clipping();  // 重置剪切器
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::styles(int left, int right)
    {
        // 创建一个名为 cell 的变量，类型为 cell_style_aa
        cell_style_aa cell;
        // 初始化 cell 变量
        cell.initial();
        // 将 left 转换为 int16 类型并赋值给 cell 的 left 成员变量
        cell.left = (int16)left;
        // 将 right 转换为 int16 类型并赋值给 cell 的 right 成员变量
        cell.right = (int16)right;
        // 将 cell 应用到 m_outline 对象上，设定样式
        m_outline.style(cell);
        // 如果 left 在非负且小于 m_min_style，则更新 m_min_style
        if(left >= 0 && left < m_min_style) m_min_style = left;
        // 如果 left 在非负且大于 m_max_style，则更新 m_max_style
        if(left >= 0 && left > m_max_style) m_max_style = left;
        // 如果 right 在非负且小于 m_min_style，则更新 m_min_style
        if(right >= 0 && right < m_min_style) m_min_style = right;
        // 如果 right 在非负且大于 m_max_style，则更新 m_max_style
        if(right >= 0 && right > m_max_style) m_max_style = right;
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::move_to(int x, int y)
    {
        // 如果 m_outline 已经排序过，则重置其状态
        if(m_outline.sorted()) reset();
        // 将 x 和 y 坐标经过缩放后，作为起始点传给 m_clipper 的 move_to 方法
        m_clipper.move_to(m_start_x = conv_type::downscale(x), 
                          m_start_y = conv_type::downscale(y));
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::line_to(int x, int y)
    {
        // 将 x 和 y 坐标经过缩放后，作为线段终点传给 m_clipper 的 line_to 方法
        m_clipper.line_to(m_outline, 
                          conv_type::downscale(x), 
                          conv_type::downscale(y));
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::move_to_d(double x, double y) 
    { 
        // 如果 m_outline 已经排序过，则重置其状态
        if(m_outline.sorted()) reset();
        // 将 x 和 y 坐标经过放大后，作为起始点传给 m_clipper 的 move_to 方法
        m_clipper.move_to(m_start_x = conv_type::upscale(x), 
                          m_start_y = conv_type::upscale(y)); 
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::line_to_d(double x, double y) 
    { 
        // 将 x 和 y 坐标经过放大后，作为线段终点传给 m_clipper 的 line_to 方法
        m_clipper.line_to(m_outline, 
                          conv_type::upscale(x), 
                          conv_type::upscale(y)); 
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::add_vertex(double x, double y, unsigned cmd)
    {
        // 如果命令是移动到点的命令
        if(is_move_to(cmd)) 
        {
            // 调用 move_to_d 方法，传入 x 和 y 作为坐标
            move_to_d(x, y);
        }
        else 
        // 如果命令是顶点命令
        if(is_vertex(cmd))
        {
            // 调用 line_to_d 方法，传入 x 和 y 作为坐标
            line_to_d(x, y);
        }
        else
        // 如果命令是闭合命令
        if(is_close(cmd))
        {
            // 将起始点坐标作为终点传给 m_clipper 的 line_to 方法，实现闭合路径
            m_clipper.line_to(m_outline, m_start_x, m_start_y);
        }
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::edge(int x1, int y1, int x2, int y2)
    {
        // 如果 m_outline 已经排序过，则重置其状态
        if(m_outline.sorted()) reset();
        // 将 x1 和 y1 坐标经过缩放后，作为起始点传给 m_clipper 的 move_to 方法
        m_clipper.move_to(conv_type::downscale(x1), conv_type::downscale(y1));
        // 将 x2 和 y2 坐标经过缩放后，作为线段终点传给 m_clipper 的 line_to 方法
        m_clipper.line_to(m_outline, 
                          conv_type::downscale(x2), 
                          conv_type::downscale(y2));
    }
    {
        // 如果 m_outline 已经排序，则重置其状态
        if(m_outline.sorted()) reset();
        // 将起始点移动到指定坐标 (x1, y1)，使用 upscale 方法进行坐标转换
        m_clipper.move_to(conv_type::upscale(x1), conv_type::upscale(y1)); 
        // 添加一条从当前位置到 (x2, y2) 的直线到 m_outline 中，使用 upscale 方法进行坐标转换
        m_clipper.line_to(m_outline, 
                          conv_type::upscale(x2), 
                          conv_type::upscale(y2)); 
    }



    //------------------------------------------------------------------------
    template<class Clip> 
    AGG_INLINE void rasterizer_compound_aa<Clip>::sort()
    {
        // 对 m_outline 中的单元格进行排序
        m_outline.sort_cells();
    }



    //------------------------------------------------------------------------
    template<class Clip> 
    AGG_INLINE bool rasterizer_compound_aa<Clip>::rewind_scanlines()
    {
        // 对 m_outline 中的单元格进行排序
        m_outline.sort_cells();
        // 如果 m_outline 中的单元格数为 0，则返回 false
        if(m_outline.total_cells() == 0) 
        {
            return false;
        }
        // 如果 m_max_style 小于 m_min_style，则返回 false
        if(m_max_style < m_min_style)
        {
            return false;
        }
        // 设置当前扫描线的起始位置为 m_outline 的最小 y 坐标
        m_scan_y = m_outline.min_y();
        // 分配样式数组的空间，大小为 m_max_style - m_min_style + 2，每个样式大小为 128
        m_styles.allocate(m_max_style - m_min_style + 2, 128);
        return true;
    }



    //------------------------------------------------------------------------
    template<class Clip> 
    AGG_INLINE void rasterizer_compound_aa<Clip>::add_style(int style_id)
    {
        // 如果 style_id 小于 0，则将其设置为 0
        if(style_id < 0) style_id  = 0;
        else             style_id -= m_min_style - 1;

        // 计算样式索引对应的字节位置和位掩码
        unsigned nbyte = style_id >> 3;
        unsigned mask = 1 << (style_id & 7);

        // 获取对应样式信息的指针
        style_info* style = &m_styles[style_id];
        // 如果对应位未设置，则添加样式信息
        if((m_asm[nbyte] & mask) == 0)
        {
            m_ast.add(style_id);
            m_asm[nbyte] |= mask;
            style->start_cell = 0;
            style->num_cells = 0;
            style->last_x = -0x7FFFFFFF;
        }
        // 增加样式的起始单元格计数
        ++style->start_cell;
    }



    //------------------------------------------------------------------------
    // 返回样式数量
    template<class Clip> 
    unsigned rasterizer_compound_aa<Clip>::sweep_styles()
    {
        // 返回当前已添加的样式数量
    }



    //------------------------------------------------------------------------
    // 根据样式索引返回样式 ID
    template<class Clip> 
    AGG_INLINE 
    unsigned rasterizer_compound_aa<Clip>::style(unsigned style_idx) const
    {
        // 返回样式索引对应的样式 ID
        return m_ast[style_idx + 1] + m_min_style - 1;
    }



    //------------------------------------------------------------------------ 
    template<class Clip> 
    AGG_INLINE bool rasterizer_compound_aa<Clip>::navigate_scanline(int y)
    {
        // 对 m_outline 中的单元格进行排序
        m_outline.sort_cells();
        // 如果 m_outline 中的单元格数为 0，则返回 false
        if(m_outline.total_cells() == 0) 
        {
            return false;
        }
        // 如果 m_max_style 小于 m_min_style，则返回 false
        if(m_max_style < m_min_style)
        {
            return false;
        }
        // 如果指定的 y 值不在 m_outline 的范围内，则返回 false
        if(y < m_outline.min_y() || y > m_outline.max_y()) 
        {
            return false;
        }
        // 设置当前扫描线的位置为指定的 y 坐标
        m_scan_y = y;
        // 分配样式数组的空间，大小为 m_max_style - m_min_style + 2，每个样式大小为 128
        m_styles.allocate(m_max_style - m_min_style + 2, 128);
        return true;
    }



    //------------------------------------------------------------------------ 
    template<class Clip> 
    bool rasterizer_compound_aa<Clip>::hit_test(int tx, int ty)
    {
        // 该函数未提供实现，需根据具体需求进行实现
    }
    {
        // 如果 navigate_scanline 函数返回 false，直接返回 false
        if(!navigate_scanline(ty)) 
        {
            return false;
        }
    
        // 调用 sweep_styles 函数获取样式数量
        unsigned num_styles = sweep_styles(); 
    
        // 如果样式数量小于等于 0，返回 false
        if(num_styles <= 0)
        {
            return false;
        }
    
        // 创建 scanline_hit_test 对象 sl，使用 tx 初始化
        scanline_hit_test sl(tx);
        
        // 调用 sweep_scanline 函数，对 scanline_hit_test 对象 sl 进行扫描
        sweep_scanline(sl, -1);
        
        // 返回 sl.hit() 的结果，表示扫描线是否命中
        return sl.hit();
    }
    
    //------------------------------------------------------------------------ 
    template<class Clip> 
    // 分配覆盖缓冲区，长度为 len，使用 m_cover_buf 分配
    cover_type* rasterizer_compound_aa<Clip>::allocate_cover_buffer(unsigned len)
    {
        // 调用 m_cover_buf 的 allocate 方法，分配长度为 len，每个元素大小为 256
        m_cover_buf.allocate(len, 256);
    
        // 返回 m_cover_buf 的第一个元素的指针
        return &m_cover_buf[0];
    }
}



#endif



// 这两行是 C/C++ 中的预处理指令，用于结束代码块和条件编译的部分
// } 是用来结束代码块的符号，在这里可能用于结束函数定义或其他代码结构
// #endif 是条件编译指令的结束标志，用于结束条件编译块
// 它们一起确保代码块或条件编译部分正确地关闭和结束
```