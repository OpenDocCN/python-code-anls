# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_renderer_scanline.h`

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

#ifndef AGG_RENDERER_SCANLINE_INCLUDED
#define AGG_RENDERER_SCANLINE_INCLUDED

#include "agg_basics.h"
#include "agg_renderer_base.h"

namespace agg
{

    //================================================render_scanline_aa_solid
    // 渲染抗锯齿单色扫描线
    template<class Scanline, class BaseRenderer, class ColorT> 
    void render_scanline_aa_solid(const Scanline& sl, 
                                  BaseRenderer& ren, 
                                  const ColorT& color)
    {
        // 获取扫描线的起始 Y 坐标
        int y = sl.y();
        // 获取扫描线中跨度的数量
        unsigned num_spans = sl.num_spans();
        // 获取扫描线的迭代器，用于遍历每个跨度
        typename Scanline::const_iterator span = sl.begin();

        // 循环处理每个跨度
        for(;;)
        {
            // 获取当前跨度的起始 X 坐标
            int x = span->x;
            // 如果跨度长度大于 0，则渲染水平跨度
            if(span->len > 0)
            {
                // 使用渲染器对象渲染一段水平的实心区域
                ren.blend_solid_hspan(x, y, (unsigned)span->len, 
                                      color, 
                                      span->covers);
            }
            else
            {
                // 如果跨度长度小于等于 0，则渲染水平线
                ren.blend_hline(x, y, (unsigned)(x - span->len - 1), 
                                color, 
                                *(span->covers));
            }
            // 减少剩余跨度计数
            if(--num_spans == 0) break;
            // 移动到下一个跨度
            ++span;
        }
    }

    //===============================================render_scanlines_aa_solid
    // 渲染抗锯齿单色扫描线集合
    template<class Rasterizer, class Scanline, 
             class BaseRenderer, class ColorT>
    void render_scanlines_aa_solid(Rasterizer& ras, Scanline& sl, 
                                   BaseRenderer& ren, const ColorT& color)
    {
        // 如果重置扫描线失败，则不执行以下代码
        if(ras.rewind_scanlines())
        {
            // 将"color"显式转换为BaseRenderer颜色类型。
            // 例如，它可能被调用为"rgba"类型，但需要"rgba8"类型。
            // 否则，在循环中会隐式转换多次。
            //----------------------
            typename BaseRenderer::color_type ren_color(color);

            // 重置扫描线对象，并设置其范围为最小到最大X坐标
            sl.reset(ras.min_x(), ras.max_x());
            while(ras.sweep_scanline(sl))
            {
                //render_scanline_aa_solid(sl, ren, ren_color);

                // 该段代码等同于上述调用（复制/粘贴）。
                // 这只是对老的编译器（如Microsoft Visual C++ v6.0）的一种手动优化。
                //-------------------------------
                int y = sl.y();
                unsigned num_spans = sl.num_spans();
                typename Scanline::const_iterator span = sl.begin();

                // 循环处理扫描线的每个 span
                for(;;)
                {
                    int x = span->x;
                    if(span->len > 0)
                    {
                        // 对水平跨度进行混合渲染
                        ren.blend_solid_hspan(x, y, (unsigned)span->len, 
                                              ren_color, 
                                              span->covers);
                    }
                    else
                    {
                        // 对水平线进行混合渲染
                        ren.blend_hline(x, y, (unsigned)(x - span->len - 1), 
                                        ren_color, 
                                        *(span->covers));
                    }
                    // 减少剩余 span 数量，如果为零则跳出循环
                    if(--num_spans == 0) break;
                    ++span;
                }
            }
        }
    }

    //==============================================renderer_scanline_aa_solid
    // 定义模板类 renderer_scanline_aa_solid
    template<class BaseRenderer> class renderer_scanline_aa_solid
    {
    public:
        // 类型定义
        typedef BaseRenderer base_ren_type;
        typedef typename base_ren_type::color_type color_type;

        //--------------------------------------------------------------------
        // 构造函数，默认初始化 m_ren 为 nullptr
        renderer_scanline_aa_solid() : m_ren(0) {}
        // 显式构造函数，将 m_ren 初始化为给定的 base_ren_type 引用
        explicit renderer_scanline_aa_solid(base_ren_type& ren) : m_ren(&ren) {}
        // 附加函数，将 m_ren 指向给定的 base_ren_type 引用
        void attach(base_ren_type& ren)
        {
            m_ren = &ren;
        }
        
        //--------------------------------------------------------------------
        // 设置颜色函数，将 m_color 设置为给定的颜色 c
        void color(const color_type& c) { m_color = c; }
        // 获取颜色函数，返回当前存储的颜色 m_color
        const color_type& color() const { return m_color; }

        //--------------------------------------------------------------------
        // 准备函数，空实现
        void prepare() {}

        //--------------------------------------------------------------------
        // 渲染函数模板，调用 render_scanline_aa_solid 函数进行渲染
        template<class Scanline> void render(const Scanline& sl)
        {
            render_scanline_aa_solid(sl, *m_ren, m_color);
        }
        
    private:
        // 成员变量，指向基础渲染器的指针和当前颜色
        base_ren_type* m_ren;
        color_type m_color;
    };
    //======================================================render_scanline_aa
    template<class Scanline, class BaseRenderer, 
             class SpanAllocator, class SpanGenerator> 
    void render_scanline_aa(const Scanline& sl, BaseRenderer& ren, 
                            SpanAllocator& alloc, SpanGenerator& span_gen)
    {
        int y = sl.y();  // 获取扫描线的纵坐标
    
        unsigned num_spans = sl.num_spans();  // 获取扫描线中的跨度数量
        typename Scanline::const_iterator span = sl.begin();  // 获取扫描线的起始迭代器
        for(;;)
        {
            int x = span->x;  // 获取跨度的横坐标起始点
            int len = span->len;  // 获取跨度的长度
            const typename Scanline::cover_type* covers = span->covers;  // 获取跨度的覆盖信息数组
    
            if(len < 0) len = -len;  // 如果长度为负数，则取其绝对值
            typename BaseRenderer::color_type* colors = alloc.allocate(len);  // 分配颜色数组的内存空间
            span_gen.generate(colors, x, y, len);  // 生成指定长度的颜色数据
            ren.blend_color_hspan(x, y, len, colors, 
                                  (span->len < 0) ? 0 : covers, *covers);  // 将生成的颜色数据与覆盖信息进行混合渲染
    
            if(--num_spans == 0) break;  // 如果扫描线中的跨度数量减为零，则退出循环
            ++span;  // 否则，移动到下一个跨度
        }
    }
    
    //=====================================================render_scanlines_aa
    template<class Rasterizer, class Scanline, class BaseRenderer, 
             class SpanAllocator, class SpanGenerator>
    void render_scanlines_aa(Rasterizer& ras, Scanline& sl, BaseRenderer& ren, 
                             SpanAllocator& alloc, SpanGenerator& span_gen)
    {
        if(ras.rewind_scanlines())  // 如果光栅化器可以倒回扫描线
        {
            sl.reset(ras.min_x(), ras.max_x());  // 重置扫描线的范围
            span_gen.prepare();  // 准备生成跨度信息
            while(ras.sweep_scanline(sl))  // 当光栅化器扫描线逐行扫描时
            {
                render_scanline_aa(sl, ren, alloc, span_gen);  // 渲染每一条扫描线
            }
        }
    }
    
    //====================================================renderer_scanline_aa
    template<class BaseRenderer, class SpanAllocator, class SpanGenerator> 
    class renderer_scanline_aa
    {
    public:
        typedef BaseRenderer  base_ren_type;
        typedef SpanAllocator alloc_type;
        typedef SpanGenerator span_gen_type;
    
        //--------------------------------------------------------------------
        renderer_scanline_aa() : m_ren(0), m_alloc(0), m_span_gen(0) {}
        renderer_scanline_aa(base_ren_type& ren, 
                             alloc_type& alloc, 
                             span_gen_type& span_gen) :
            m_ren(&ren),
            m_alloc(&alloc),
            m_span_gen(&span_gen)
        {}
        void attach(base_ren_type& ren, 
                    alloc_type& alloc, 
                    span_gen_type& span_gen)
        {
            m_ren = &ren;
            m_alloc = &alloc;
            m_span_gen = &span_gen;
        }
        
        //--------------------------------------------------------------------
        void prepare() { m_span_gen->prepare(); }  // 准备生成跨度信息
    
        //--------------------------------------------------------------------
        template<class Scanline> void render(const Scanline& sl)
        {
            render_scanline_aa(sl, *m_ren, *m_alloc, *m_span_gen);  // 渲染给定的扫描线
        }
    // 声明私有成员变量
    private:
        base_ren_type* m_ren;       // 指向基础渲染器对象的指针
        alloc_type*    m_alloc;     // 指向分配器对象的指针
        span_gen_type* m_span_gen;  // 指向扫描线生成器对象的指针
    };

    //===============================================render_scanline_bin_solid
    // 渲染二进制实心扫描线
    template<class Scanline, class BaseRenderer, class ColorT> 
    void render_scanline_bin_solid(const Scanline& sl, 
                                   BaseRenderer& ren, 
                                   const ColorT& color)
    {
        unsigned num_spans = sl.num_spans();  // 获取扫描线中的跨度数量
        typename Scanline::const_iterator span = sl.begin();  // 获取扫描线的迭代器
        for(;;)
        {
            // 在基础渲染器上绘制水平线段，根据跨度的长度进行调整
            ren.blend_hline(span->x, 
                            sl.y(), 
                            span->x - 1 + ((span->len < 0) ? 
                                              -span->len : 
                                               span->len), 
                               color, 
                               cover_full);
            if(--num_spans == 0) break;  // 若跨度数量减至零则结束循环
            ++span;  // 否则移动到下一个跨度
        }
    }

    //==============================================render_scanlines_bin_solid
    // 渲染一组二进制实心扫描线
    template<class Rasterizer, class Scanline, 
             class BaseRenderer, class ColorT>
    void render_scanlines_bin_solid(Rasterizer& ras, Scanline& sl, 
                                    BaseRenderer& ren, const ColorT& color)
    {
        if(ras.rewind_scanlines())  // 若扫描线回溯成功
        {
            // 显式将 "color" 转换为 BaseRenderer 的颜色类型
            typename BaseRenderer::color_type ren_color(color);

            sl.reset(ras.min_x(), ras.max_x());  // 重置扫描线范围
            while(ras.sweep_scanline(sl))  // 循环处理扫描线
            {
                // 使用渲染二进制实心扫描线函数来渲染当前扫描线
                //----------------------
                unsigned num_spans = sl.num_spans();  // 获取当前扫描线中的跨度数量
                typename Scanline::const_iterator span = sl.begin();  // 获取扫描线的迭代器
                for(;;)
                {
                    // 在基础渲染器上绘制水平线段，根据跨度的长度进行调整
                    ren.blend_hline(span->x, 
                                    sl.y(), 
                                    span->x - 1 + ((span->len < 0) ? 
                                                      -span->len : 
                                                       span->len), 
                                       ren_color, 
                                       cover_full);
                    if(--num_spans == 0) break;  // 若跨度数量减至零则结束循环
                    ++span;  // 否则移动到下一个跨度
                }
            }
        }
    }

    //=============================================renderer_scanline_bin_solid
    // 用于渲染二进制实心扫描线的渲染器模板类
    template<class BaseRenderer> class renderer_scanline_bin_solid
    {
    //======================================================renderer_scanline_bin_solid
    // renderer_scanline_bin_solid 类的定义
    template<class BaseRenderer>
    class renderer_scanline_bin_solid
    {
    public:
        // 定义类型别名 base_ren_type 为 BaseRenderer 类型，color_type 为 BaseRenderer::color_type 类型
        typedef BaseRenderer base_ren_type;
        typedef typename base_ren_type::color_type color_type;

        //--------------------------------------------------------------------
        // 默认构造函数，初始化 m_ren 指针为空
        renderer_scanline_bin_solid() : m_ren(0) {}
        // 显式构造函数，初始化 m_ren 指针为给定的 ren
        explicit renderer_scanline_bin_solid(base_ren_type& ren) : m_ren(&ren) {}
        // 将 renderer_scanline_bin_solid 对象与指定的渲染器 ren 关联
        void attach(base_ren_type& ren)
        {
            m_ren = &ren;
        }
        
        //--------------------------------------------------------------------
        // 设置当前绘制颜色为 c
        void color(const color_type& c) { m_color = c; }
        // 返回当前设置的绘制颜色
        const color_type& color() const { return m_color; }

        //--------------------------------------------------------------------
        // 准备渲染，空实现
        void prepare() {}

        //--------------------------------------------------------------------
        // 渲染给定的扫描线 sl
        template<class Scanline> void render(const Scanline& sl)
        {
            // 调用 render_scanline_bin_solid 函数，使用 m_ren 指定的渲染器和 m_color 进行实际的渲染
            render_scanline_bin_solid(sl, *m_ren, m_color);
        }
        
    private:
        // 指向基础渲染器的指针
        base_ren_type* m_ren;
        // 当前设置的颜色
        color_type m_color;
    };








    //======================================================render_scanline_bin
    // 渲染二进制化的扫描线，使用给定的渲染器 ren、分配器 alloc 和生成器 span_gen
    template<class Scanline, class BaseRenderer, 
             class SpanAllocator, class SpanGenerator> 
    void render_scanline_bin(const Scanline& sl, BaseRenderer& ren, 
                             SpanAllocator& alloc, SpanGenerator& span_gen)
    {
        // 获取当前扫描线的 y 坐标
        int y = sl.y();

        // 获取扫描线中的段数
        unsigned num_spans = sl.num_spans();
        // 获取扫描线的迭代器
        typename Scanline::const_iterator span = sl.begin();
        for(;;)
        {
            // 获取当前段的起始 x 坐标和长度
            int x = span->x;
            int len = span->len;
            // 如果长度为负数，取其绝对值
            if(len < 0) len = -len;
            // 分配颜色数组，长度为当前段的长度
            typename BaseRenderer::color_type* colors = alloc.allocate(len);
            // 使用 span_gen 生成颜色数据，填充到 colors 中
            span_gen.generate(colors, x, y, len);
            // 在渲染器 ren 中使用颜色数组 colors 渲染水平段
            ren.blend_color_hspan(x, y, len, colors, 0, cover_full); 
            // 减少剩余段数计数
            if(--num_spans == 0) break;
            // 移动到下一个段
            ++span;
        }
    }

    //=====================================================render_scanlines_bin
    // 渲染二进制化的扫描线集合，使用给定的光栅化器 ras、扫描线 sl、渲染器 ren、
    // 分配器 alloc 和生成器 span_gen
    template<class Rasterizer, class Scanline, class BaseRenderer, 
             class SpanAllocator, class SpanGenerator>
    void render_scanlines_bin(Rasterizer& ras, Scanline& sl, BaseRenderer& ren, 
                              SpanAllocator& alloc, SpanGenerator& span_gen)
    {
        // 如果光栅化器 ras 可以倒回扫描线
        if(ras.rewind_scanlines())
        {
            // 重置扫描线 sl，设定其 x 范围为 ras 的最小到最大 x 值
            sl.reset(ras.min_x(), ras.max_x());
            // 准备生成器 span_gen
            span_gen.prepare();
            // 当光栅化器 ras 生成扫描线 sl 时
            while(ras.sweep_scanline(sl))
            {
                // 调用 render_scanline_bin 函数，渲染当前扫描线 sl
                render_scanline_bin(sl, ren, alloc, span_gen);
            }
        }
    }

    //====================================================renderer_scanline_bin
    // renderer_scanline_bin 类的定义，使用给定的基础渲染器、分配器和生成器
    template<class BaseRenderer, class SpanAllocator, class SpanGenerator> 
    class renderer_scanline_bin
    {
    // 公共部分定义别名：基础渲染器、跨距分配器、扫描线生成器
    public:
        typedef BaseRenderer  base_ren_type;
        typedef SpanAllocator alloc_type;
        typedef SpanGenerator span_gen_type;

        //--------------------------------------------------------------------
        // 默认构造函数初始化成员指针为 null
        renderer_scanline_bin() : m_ren(0), m_alloc(0), m_span_gen(0) {}

        // 带参数的构造函数，初始化成员指针为传入的对象引用
        renderer_scanline_bin(base_ren_type& ren, 
                              alloc_type& alloc, 
                              span_gen_type& span_gen) :
            m_ren(&ren),
            m_alloc(&alloc),
            m_span_gen(&span_gen)
        {}

        // 重新设置成员指针为传入的对象引用
        void attach(base_ren_type& ren, 
                    alloc_type& alloc, 
                    span_gen_type& span_gen)
        {
            m_ren = &ren;
            m_alloc = &alloc;
            m_span_gen = &span_gen;
        }
        
        //--------------------------------------------------------------------
        // 准备函数调用 span_gen_type 类型对象的 prepare() 函数
        void prepare() { m_span_gen->prepare(); }

        //--------------------------------------------------------------------
        // 模板函数，渲染给定的扫描线 sl
        template<class Scanline> void render(const Scanline& sl)
        {
            // 调用 render_scanline_bin 函数，将 m_ren、m_alloc、m_span_gen 作为参数
            render_scanline_bin(sl, *m_ren, *m_alloc, *m_span_gen);
        }

    private:
        // 成员变量：基础渲染器、跨距分配器、扫描线生成器的指针
        base_ren_type* m_ren;
        alloc_type*    m_alloc;
        span_gen_type* m_span_gen;
    };










    //========================================================render_scanlines
    // 渲染扫描线的函数模板
    template<class Rasterizer, class Scanline, class Renderer>
    void render_scanlines(Rasterizer& ras, Scanline& sl, Renderer& ren)
    {
        // 如果 ras 可以重置扫描线
        if(ras.rewind_scanlines())
        {
            // 重置扫描线 sl 的范围
            sl.reset(ras.min_x(), ras.max_x());
            // 准备渲染器 ren
            ren.prepare();
            // 循环扫描线 ras 的扫描
            while(ras.sweep_scanline(sl))
            {
                // 渲染当前扫描线 sl
                ren.render(sl);
            }
        }
    }

    //========================================================render_all_paths
    // 渲染所有路径的函数模板
    template<class Rasterizer, class Scanline, class Renderer, 
             class VertexSource, class ColorStorage, class PathId>
    void render_all_paths(Rasterizer& ras, 
                          Scanline& sl,
                          Renderer& r, 
                          VertexSource& vs, 
                          const ColorStorage& as, 
                          const PathId& path_id,
                          unsigned num_paths)
    {
        // 遍历所有路径
        for(unsigned i = 0; i < num_paths; i++)
        {
            // 重置 ras 的状态
            ras.reset();
            // 添加路径到 ras
            ras.add_path(vs, path_id[i]);
            // 设置渲染器 r 的颜色
            r.color(as[i]);
            // 渲染路径的扫描线
            render_scanlines(ras, sl, r);
        }
    }






    //=============================================render_scanlines_compound
    // 复合渲染扫描线的函数模板
    template<class Rasterizer, 
             class ScanlineAA, 
             class ScanlineBin, 
             class BaseRenderer, 
             class SpanAllocator,
             class StyleHandler>
    // 定义一个函数 render_scanlines_compound，用于渲染复合图形的扫描线
    void render_scanlines_compound(Rasterizer& ras, 
                                   ScanlineAA& sl_aa,
                                   ScanlineBin& sl_bin,
                                   BaseRenderer& ren,
                                   SpanAllocator& alloc,
                                   StyleHandler& sh)
    {
        // 这里是函数的定义，接收了六个参数：
        // ras: 光栅化器对象的引用，用于处理几何图形的光栅化
        // sl_aa: 抗锯齿扫描线对象的引用，用于处理光栅化后的抗锯齿扫描线
        // sl_bin: 二进制扫描线对象的引用，可能用于某些特定的光栅化方法
        // ren: 基本渲染器对象的引用，负责将扫描线转换为最终的像素值
        // alloc: 跨度分配器对象的引用，用于管理和分配渲染过程中的内存空间
        // sh: 样式处理器对象的引用，用于处理渲染过程中的样式和属性
    
        // 函数体未提供，这里只有函数声明，实际功能需要查看函数的实现部分
    }
    
    //=======================================render_scanlines_compound_layered
    // 定义一个模板函数 render_scanlines_compound_layered，用于分层渲染复合图形的扫描线
    template<class Rasterizer, 
             class ScanlineAA, 
             class BaseRenderer, 
             class SpanAllocator,
             class StyleHandler>
    void render_scanlines_compound_layered(Rasterizer& ras, 
                                           ScanlineAA& sl_aa,
                                           BaseRenderer& ren,
                                           SpanAllocator& alloc,
                                           StyleHandler& sh)
    {
        // 这里是函数的定义，接收了五个模板参数和五个参数：
        // Rasterizer: 光栅化器类型，用于处理几何图形的光栅化
        // ScanlineAA: 抗锯齿扫描线类型，处理光栅化后的抗锯齿扫描线
        // BaseRenderer: 基本渲染器类型，将扫描线转换为最终的像素值
        // SpanAllocator: 跨度分配器类型，管理和分配渲染过程中的内存空间
        // StyleHandler: 样式处理器类型，处理渲染过程中的样式和属性
        // ras: 光栅化器对象的引用，用于具体的光栅化操作
        // sl_aa: 抗锯齿扫描线对象的引用，处理光栅化后的抗锯齿扫描线
        // ren: 基本渲染器对象的引用，转换扫描线为像素值
        // alloc: 跨度分配器对象的引用，管理和分配渲染过程中的内存空间
        // sh: 样式处理器对象的引用，处理渲染过程中的样式和属性
    
        // 函数体未提供，这里只有函数声明，实际功能需要查看函数的实现部分
    }
}


这是一个代码片段的结尾标记，通常用于表示某个条件编译指令或者预处理器指令的结束。在这个例子中，`} }` 可能用于结束一个特定的条件编译段落或者预处理器的定义。


#endif


`#endif` 是预处理器指令，用于结束一个条件编译块，它通常与 `#ifdef` 或者 `#ifndef` 配对使用，用来控制在特定条件下是否编译一段代码。在这里，`#endif` 表示结束了之前通过 `#ifdef` 或 `#ifndef` 开始的条件编译区段。
```