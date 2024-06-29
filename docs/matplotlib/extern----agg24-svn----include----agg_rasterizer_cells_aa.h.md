# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_rasterizer_cells_aa.h`

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

#ifndef AGG_RASTERIZER_CELLS_AA_INCLUDED
#define AGG_RASTERIZER_CELLS_AA_INCLUDED

#include <stdexcept>    // 引入标准异常类的头文件
#include <string.h>     // 引入处理字符串相关函数的头文件
#include <math.h>       // 引入数学函数的头文件
#include "agg_math.h"   // 引入 AGG 数学函数的头文件
#include "agg_array.h"  // 引入 AGG 数组处理的头文件

namespace agg
{

    //-----------------------------------------------------rasterizer_cells_aa
    // An internal class that implements the main rasterization algorithm.
    // Used in the rasterizer. Should not be used direcly.
    template<class Cell> class rasterizer_cells_aa
    {
        enum cell_block_scale_e
        {
            cell_block_shift = 12,             // 定义块大小的位移量
            cell_block_size  = 1 << cell_block_shift,  // 块大小的实际值
            cell_block_mask  = cell_block_size - 1,   // 块掩码，用于快速计算块内偏移
            cell_block_pool  = 256                      // 块池的大小
        };

        struct sorted_y
        {
            unsigned start;   // 排序后的起始位置
            unsigned num;     // 排序后的数量
        };
    // 公共部分开始
    public:
        // 定义类型别名 cell_type 和 self_type
        typedef Cell cell_type;
        typedef rasterizer_cells_aa<Cell> self_type;

        // 析构函数定义
        ~rasterizer_cells_aa();
        
        // 构造函数定义，可选参数为 cell_block_limit，默认为 1024
        rasterizer_cells_aa(unsigned cell_block_limit=1024);

        // 重置函数声明
        void reset();
        
        // 设置样式函数声明，参数为样式单元格 style_cell
        void style(const cell_type& style_cell);
        
        // 绘制直线函数声明，参数为起点 (x1, y1) 和终点 (x2, y2) 的坐标
        void line(int x1, int y1, int x2, int y2);

        // 返回最小 x 坐标值的函数声明
        int min_x() const { return m_min_x; }
        
        // 返回最小 y 坐标值的函数声明
        int min_y() const { return m_min_y; }
        
        // 返回最大 x 坐标值的函数声明
        int max_x() const { return m_max_x; }
        
        // 返回最大 y 坐标值的函数声明
        int max_y() const { return m_max_y; }

        // 排序单元格函数声明
        void sort_cells();

        // 返回总单元格数的函数定义
        unsigned total_cells() const
        {
            return m_num_cells;
        }

        // 返回指定扫描线 y 上单元格的数量的函数声明，参数为扫描线 y 坐标
        unsigned scanline_num_cells(unsigned y) const
        {
            return m_sorted_y[y - m_min_y].num;
        }

        // 返回指定扫描线 y 上单元格指针数组的函数声明，参数为扫描线 y 坐标
        const cell_type* const* scanline_cells(unsigned y) const
        {
            return m_sorted_cells.data() + m_sorted_y[y - m_min_y].start;
        }

        // 返回是否已排序的布尔函数声明
        bool sorted() const { return m_sorted; }

    // 私有部分开始
    private:
        // 复制构造函数声明，禁止使用
        rasterizer_cells_aa(const self_type&);
        
        // 赋值操作符重载声明，禁止使用
        const self_type& operator = (const self_type&);

        // 设置当前单元格函数声明，参数为当前单元格的 x 和 y 坐标
        void set_curr_cell(int x, int y);
        
        // 添加当前单元格函数声明
        void add_curr_cell();
        
        // 渲染水平线函数声明，参数为结束 y 坐标 ey，起点 (x1, y1) 和终点 (x2, y2) 的坐标
        void render_hline(int ey, int x1, int y1, int x2, int y2);
        
        // 分配块函数声明
        void allocate_block();

    // 私有变量声明
    private:
        // 块数、最大块数、当前块、单元格数、单元格块限制
        unsigned                m_num_blocks;
        unsigned                m_max_blocks;
        unsigned                m_curr_block;
        unsigned                m_num_cells;
        unsigned                m_cell_block_limit;
        
        // 单元格指针数组、当前单元格指针
        cell_type**             m_cells;
        cell_type*              m_curr_cell_ptr;
        
        // 排序单元格和排序扫描线的向量
        pod_vector<cell_type*>  m_sorted_cells;
        pod_vector<sorted_y>    m_sorted_y;
        
        // 当前单元格和样式单元格
        cell_type               m_curr_cell;
        cell_type               m_style_cell;
        
        // 最小和最大 x、y 坐标
        int                     m_min_x;
        int                     m_min_y;
        int                     m_max_x;
        int                     m_max_y;
        
        // 是否已排序的标志
        bool                    m_sorted;
    };
    // 构造函数：初始化 rasterizer_cells_aa 类的实例
    // 参数：cell_block_limit - 最大单元块数限制
    rasterizer_cells_aa<Cell>::rasterizer_cells_aa(unsigned cell_block_limit) :
        // 初始化成员变量
        m_num_blocks(0),                     // 当前块数
        m_max_blocks(0),                     // 最大块数
        m_curr_block(0),                     // 当前块索引
        m_num_cells(0),                      // 当前单元数
        m_cell_block_limit(cell_block_limit),// 单元块数限制
        m_cells(0),                          // 单元数组指针
        m_curr_cell_ptr(0),                  // 当前单元指针
        m_sorted_cells(),                    // 已排序的单元
        m_sorted_y(),                        // 已排序的 Y 坐标
        m_min_x(0x7FFFFFFF),                 // 最小 X 坐标
        m_min_y(0x7FFFFFFF),                 // 最小 Y 坐标
        m_max_x(-0x7FFFFFFF),                // 最大 X 坐标
        m_max_y(-0x7FFFFFFF),                // 最大 Y 坐标
        m_sorted(false)                      // 是否已排序标志位
    {
        // 初始化当前样式单元和当前单元
        m_style_cell.initial();              // 样式单元初始化
        m_curr_cell.initial();               // 当前单元初始化
    }

    //------------------------------------------------------------------------
    template<class Cell>
    // 重置 rasterizer_cells_aa 类的状态
    void rasterizer_cells_aa<Cell>::reset()
    {
        m_num_cells = 0;                      // 单元数归零
        m_curr_block = 0;                     // 当前块索引归零
        m_curr_cell.initial();                // 当前单元初始化
        m_style_cell.initial();               // 样式单元初始化
        m_sorted = false;                     // 未排序标志位设为假
        m_min_x =  0x7FFFFFFF;                // 最小 X 坐标设为最大值
        m_min_y =  0x7FFFFFFF;                // 最小 Y 坐标设为最大值
        m_max_x = -0x7FFFFFFF;                // 最大 X 坐标设为最小值
        m_max_y = -0x7FFFFFFF;                // 最大 Y 坐标设为最小值
    }

    //------------------------------------------------------------------------
    template<class Cell>
    // 向当前单元数组中添加当前单元
    AGG_INLINE void rasterizer_cells_aa<Cell>::add_curr_cell()
    {
        // 如果当前单元的面积或覆盖值不为零
        if(m_curr_cell.area | m_curr_cell.cover)
        {
            // 如果当前单元数与单元块掩码的按位与结果为零
            if((m_num_cells & cell_block_mask) == 0)
            {
                // 如果当前块数超过了单元块限制，抛出溢出错误
                if(m_num_blocks >= m_cell_block_limit) {
                    throw std::overflow_error("Exceeded cell block limit");
                }
                // 分配一个新的单元块
                allocate_block();
            }
            // 将当前单元添加到单元数组中，并更新当前单元数
            *m_curr_cell_ptr++ = m_curr_cell;
            ++m_num_cells;
        }
    }

    //------------------------------------------------------------------------
    template<class Cell>
    // 设置当前单元的位置
    AGG_INLINE void rasterizer_cells_aa<Cell>::set_curr_cell(int x, int y)
    {
        // 如果当前单元与给定的 x、y 和样式单元不相等
        if(m_curr_cell.not_equal(x, y, m_style_cell))
        {
            // 添加当前单元到单元数组中
            add_curr_cell();
            // 设置当前单元的样式为当前样式单元
            m_curr_cell.style(m_style_cell);
            // 更新当前单元的 x 和 y 坐标，覆盖和面积重置为零
            m_curr_cell.x     = x;
            m_curr_cell.y     = y;
            m_curr_cell.cover = 0;
            m_curr_cell.area  = 0;
        }
    }

    //------------------------------------------------------------------------
    template<class Cell>
    // 渲染水平线
    AGG_INLINE void rasterizer_cells_aa<Cell>::render_hline(int ey,
                                                            int x1, int y1,
                                                            int x2, int y2)
    {
        // 计算 x1 和 x2 的整数部分，用于确定单元格位置
        int ex1 = x1 >> poly_subpixel_shift;
        int ex2 = x2 >> poly_subpixel_shift;
        // 计算 x1 和 x2 的小数部分，用于后续计算
        int fx1 = x1 & poly_subpixel_mask;
        int fx2 = x2 & poly_subpixel_mask;
    
        int delta, p, first, dx;
        int incr, lift, mod, rem;
    
        // 如果 y1 等于 y2，则直接设置当前单元格并返回
        if(y1 == y2)
        {
            set_curr_cell(ex2, ey);
            return;
        }
    
        // 如果 ex1 等于 ex2，则两点位于同一单元格内，简化处理
        if(ex1 == ex2)
        {
            // 计算 y1 到 y2 的距离
            delta = y2 - y1;
            // 更新当前单元格的覆盖量和面积
            m_curr_cell.cover += delta;
            m_curr_cell.area  += (fx1 + fx2) * delta;
            return;
        }
    
        // 如果 ex1 不等于 ex2，则需要在相邻的单元格上渲染一系列连续的行
        // 计算跨越的像素数量
        p     = (poly_subpixel_scale - fx1) * (y2 - y1);
        first = poly_subpixel_scale;
        incr  = 1;
    
        dx = x2 - x1;
    
        // 如果 dx 小于 0，则调整参数以保证处理正确的方向
        if(dx < 0)
        {
            p     = fx1 * (y2 - y1);
            first = 0;
            incr  = -1;
            dx    = -dx;
        }
    
        // 计算每个 x 像素增量和余数
        delta = p / dx;
        mod   = p % dx;
    
        // 如果余数为负数，则调整 delta 值
        if(mod < 0)
        {
            delta--;
            mod += dx;
        }
    
        // 更新当前单元格的覆盖量和面积
        m_curr_cell.cover += delta;
        m_curr_cell.area  += (fx1 + first) * delta;
    
        // 更新 ex1 的值，并设置当前单元格
        ex1 += incr;
        set_curr_cell(ex1, ey);
        // 更新 y1 的值
        y1  += delta;
    
        // 如果 ex1 不等于 ex2，则继续循环处理直到相等
        if(ex1 != ex2)
        {
            // 计算新的 p、lift 和 rem 值
            p     = poly_subpixel_scale * (y2 - y1 + delta);
            lift  = p / dx;
            rem   = p % dx;
    
            // 如果 rem 为负数，则调整 lift 值
            if (rem < 0)
            {
                lift--;
                rem += dx;
            }
    
            mod -= dx;
    
            // 循环处理直到 ex1 等于 ex2
            while (ex1 != ex2)
            {
                // 更新 delta 和 mod 值
                delta = lift;
                mod  += rem;
                if(mod >= 0)
                {
                    mod -= dx;
                    delta++;
                }
    
                // 更新当前单元格的覆盖量和面积
                m_curr_cell.cover += delta;
                m_curr_cell.area  += poly_subpixel_scale * delta;
                y1  += delta;
                ex1 += incr;
                set_curr_cell(ex1, ey);
            }
        }
    
        // 更新最后一段 y1 到 y2 的距离的覆盖量和面积
        delta = y2 - y1;
        m_curr_cell.cover += delta;
        m_curr_cell.area  += (fx2 + poly_subpixel_scale - first) * delta;
    }
    
    //------------------------------------------------------------------------
    template<class Cell>
    // 设置样式单元格的方法
    AGG_INLINE void rasterizer_cells_aa<Cell>::style(const cell_type& style_cell)
    {
        m_style_cell.style(style_cell);
    }
    
    //------------------------------------------------------------------------
    template<class Cell>
    // 绘制线段的方法
    void rasterizer_cells_aa<Cell>::line(int x1, int y1, int x2, int y2)
    }
    
    //------------------------------------------------------------------------
    template<class Cell>
    // 分配内存块的方法
    void rasterizer_cells_aa<Cell>::allocate_block()
    {
        // 检查当前块索引是否超过了总块数，如果是，则执行以下逻辑
        if(m_curr_block >= m_num_blocks)
        {
            // 如果总块数超过了最大块数限制，需要重新分配内存
            if(m_num_blocks >= m_max_blocks)
            {
                // 分配新的内存块，大小为当前最大块数加上一个预设值
                cell_type** new_cells =
                    pod_allocator<cell_type*>::allocate(m_max_blocks +
                                                        cell_block_pool);
    
                // 如果原来有分配过内存，则将旧数据拷贝到新的内存区域
                if(m_cells)
                {
                    memcpy(new_cells, m_cells, m_max_blocks * sizeof(cell_type*));
                    // 释放原来的内存
                    pod_allocator<cell_type*>::deallocate(m_cells, m_max_blocks);
                }
                // 更新 m_cells 指向新的内存块
                m_cells = new_cells;
                // 增加最大块数限制
                m_max_blocks += cell_block_pool;
            }
    
            // 分配新的单元格并存储到 m_cells 数组中，增加总块数
            m_cells[m_num_blocks++] =
                pod_allocator<cell_type>::allocate(cell_block_size);
    
        }
        // 将当前块的指针指向下一个块
        m_curr_cell_ptr = m_cells[m_curr_block++];
    }
    
    
    
    //------------------------------------------------------------------------
    // 模板函数：交换两个元素的内容
    template <class T> static AGG_INLINE void swap_cells(T* a, T* b)
    {
        // 交换两个元素的值
        T temp = *a;
        *a = *b;
        *b = temp;
    }
    
    
    //------------------------------------------------------------------------
    // 常量枚举：快速排序的阈值
    enum
    {
        qsort_threshold = 9
    };
    
    
    //------------------------------------------------------------------------
    // 模板函数：对指定类型的指针数组进行快速排序
    template<class Cell>
    void qsort_cells(Cell** start, unsigned num)
    {
        // 声明一个指向指针的数组，用于存储 Cell 对象的地址
        Cell**  stack[80];
        // 指向指针的指针，指向当前栈顶
        Cell*** top;
        // 指向指针的指针，指向数组的末尾
        Cell**  limit;
        // 指向指针的指针，指向数组的起始位置
        Cell**  base;
    
        // 设置 limit 指针指向数组的末尾
        limit = start + num;
        // 设置 base 指针指向数组的起始位置
        base  = start;
        // 设置 top 指向 stack 数组的起始位置
        top   = stack;
    
        // 进入无限循环，直到 break 被触发
        for (;;)
        {
            // 计算当前子数组的长度
            int len = int(limit - base);
    
            // 声明三个指针，用于快速排序的操作
            Cell** i;
            Cell** j;
            Cell** pivot;
    
            // 如果子数组长度大于快速排序的阈值 qsort_threshold
            if(len > qsort_threshold)
            {
                // 使用 base + len/2 作为枢轴
                pivot = base + len / 2;
                // 将枢轴与基准位置的元素交换
                swap_cells(base, pivot);
    
                // 初始化 i 和 j 指针
                i = base + 1;
                j = limit - 1;
    
                // 确保 *i <= *base <= *j
                if((*j)->x < (*i)->x)
                {
                    swap_cells(i, j);
                }
    
                if((*base)->x < (*i)->x)
                {
                    swap_cells(base, i);
                }
    
                if((*j)->x < (*base)->x)
                {
                    swap_cells(base, j);
                }
    
                // 开始快速排序的核心循环
                for(;;)
                {
                    int x = (*base)->x;
                    // 移动 i 指针，直到找到大于等于枢轴值 x 的元素
                    do i++; while( (*i)->x < x );
                    // 移动 j 指针，直到找到小于等于枢轴值 x 的元素
                    do j--; while( x < (*j)->x );
    
                    // 如果 i > j，则说明已经完成一轮排序
                    if(i > j)
                    {
                        break;
                    }
    
                    // 交换 i 和 j 指针所指向的元素
                    swap_cells(i, j);
                }
    
                // 将基准元素移到正确的位置
                swap_cells(base, j);
    
                // 将较大的子数组压入栈中
                if(j - base > limit - i)
                {
                    top[0] = base;
                    top[1] = j;
                    base   = i;
                }
                else
                {
                    top[0] = i;
                    top[1] = limit;
                    limit  = j;
                }
                top += 2;
            }
            else
            {
                // 当子数组很小时，执行插入排序
                j = base;
                i = j + 1;
    
                for(; i < limit; j = i, i++)
                {
                    // 插入排序核心循环，将当前元素插入到已排序的序列中
                    for(; j[1]->x < (*j)->x; j--)
                    {
                        swap_cells(j + 1, j);
                        // 如果 j 到达基准位置，退出内层循环
                        if (j == base)
                        {
                            break;
                        }
                    }
                }
    
                // 弹出栈顶的子数组信息
                if(top > stack)
                {
                    top  -= 2;
                    base  = top[0];
                    limit = top[1];
                }
                else
                {
                    // 栈空，退出外层循环
                    break;
                }
            }
        }
    }
    
    //------------------------------------------------------------------------
    template<class Cell>
    void rasterizer_cells_aa<Cell>::sort_cells()
    {
        // 如果已经排序过，直接返回，不进行重复排序
        if(m_sorted) return; //Perform sort only the first time.
    
        // 添加当前单元格到排序中
        add_curr_cell();
        // 初始化当前单元格的属性
        m_curr_cell.x     = 0x7FFFFFFF;
        m_curr_cell.y     = 0x7FFFFFFF;
        m_curr_cell.cover = 0;
        m_curr_cell.area  = 0;
    
        // 如果没有单元格，直接返回
        if(m_num_cells == 0) return;
    }
// 分配排序后的单元格指针数组
m_sorted_cells.allocate(m_num_cells, 16);

// 分配并清零 Y 数组
m_sorted_y.allocate(m_max_y - m_min_y + 1, 16);
m_sorted_y.zero();

// 创建 Y 直方图（统计每个 Y 坐标上的单元格数量）
cell_type** block_ptr = m_cells;
cell_type*  cell_ptr;
unsigned nb = m_num_cells;
unsigned i;
while(nb)
{
    cell_ptr = *block_ptr++;
    i = (nb > cell_block_size) ? cell_block_size : nb;
    nb -= i;
    while(i--)
    {
        // 增加当前 Y 坐标的单元格计数
        m_sorted_y[cell_ptr->y - m_min_y].start++;
        ++cell_ptr;
    }
}

// 将 Y 直方图转换为起始索引数组
unsigned start = 0;
for(i = 0; i < m_sorted_y.size(); i++)
{
    unsigned v = m_sorted_y[i].start;
    m_sorted_y[i].start = start;
    start += v;
}

// 填充按 Y 排序的单元格指针数组
block_ptr = m_cells;
nb = m_num_cells;
while(nb)
{
    cell_ptr = *block_ptr++;
    i = (nb > cell_block_size) ? cell_block_size : nb;
    nb -= i;
    while(i--)
    {
        // 获取当前单元格的 Y 相关信息
        sorted_y& curr_y = m_sorted_y[cell_ptr->y - m_min_y];
        // 将单元格指针放入排序后的单元格数组中
        m_sorted_cells[curr_y.start + curr_y.num] = cell_ptr;
        // 增加当前 Y 坐标对应的单元格数量计数
        ++curr_y.num;
        ++cell_ptr;
    }
}

// 最后，对 X 数组进行排序
for(i = 0; i < m_sorted_y.size(); i++)
{
    const sorted_y& curr_y = m_sorted_y[i];
    if(curr_y.num)
    {
        // 对当前 Y 坐标范围内的单元格指针进行排序
        qsort_cells(m_sorted_cells.data() + curr_y.start, curr_y.num);
    }
}
m_sorted = true;
```