# `D:\src\scipysrc\matplotlib\src\_backend_agg.cpp`

```py
// 设置 C++ 编译模式为 c++，缩进为 4 个空格

// 导入 Python.h 头文件
#include <Python.h>
// 导入 _backend_agg.h 头文件
#include "_backend_agg.h"

// RendererAgg 类的构造函数，初始化各个成员变量和缓冲区
RendererAgg::RendererAgg(unsigned int width, unsigned int height, double dpi)
    : width(width),                                // 设置 RendererAgg 的宽度
      height(height),                              // 设置 RendererAgg 的高度
      dpi(dpi),                                    // 设置 RendererAgg 的 DPI
      NUMBYTES((size_t)width * (size_t)height * 4), // 计算渲染缓冲区字节数
      pixBuffer(NULL),                             // 初始化像素缓冲区指针为空
      renderingBuffer(),                           // 初始化渲染缓冲区
      alphaBuffer(NULL),                           // 初始化 alpha 缓冲区指针为空
      alphaMaskRenderingBuffer(),                  // 初始化 alpha 蒙版渲染缓冲区
      alphaMask(alphaMaskRenderingBuffer),         // 设置 alpha 蒙版
      pixfmtAlphaMask(alphaMaskRenderingBuffer),   // 设置 alpha 蒙版像素格式
      rendererBaseAlphaMask(),                     // 初始化 alpha 蒙版渲染器基类
      rendererAlphaMask(),                         // 初始化 alpha 蒙版渲染器
      scanlineAlphaMask(),                         // 初始化 alpha 蒙版扫描线
      slineP8(),                                   // 初始化 P8 扫描线
      slineBin(),                                  // 初始化二进制扫描线
      pixFmt(),                                    // 初始化像素格式
      rendererBase(),                              // 初始化渲染器基类
      rendererAA(),                                // 初始化抗锯齿渲染器
      rendererBin(),                               // 初始化二进制渲染器
      theRasterizer(32768),                        // 设置光栅化器容量
      lastclippath(NULL),                          // 初始化最后剪辑路径为空
      _fill_color(agg::rgba(1, 1, 1, 0))            // 设置填充颜色
{
    unsigned stride(width * 4);                    // 计算像素行字节数

    // 分配像素缓冲区内存
    pixBuffer = new agg::int8u[NUMBYTES];
    // 将像素缓冲区与渲染缓冲区关联
    renderingBuffer.attach(pixBuffer, width, height, stride);
    // 将像素格式与渲染缓冲区关联
    pixFmt.attach(renderingBuffer);
    // 将基础渲染器与像素格式关联，并用指定颜色清除
    rendererBase.attach(pixFmt);
    rendererBase.clear(_fill_color);
    // 将抗锯齿渲染器与基础渲染器关联
    rendererAA.attach(rendererBase);
    // 将二进制渲染器与基础渲染器关联
    rendererBin.attach(rendererBase);
    // 设置 hatch_size 为 DPI 的整数部分
    hatch_size = int(dpi);
    // 分配 hatchBuffer 内存
    hatchBuffer = new agg::int8u[hatch_size * hatch_size * 4];
    // 将 hatchBuffer 与渲染缓冲区关联
    hatchRenderingBuffer.attach(hatchBuffer, hatch_size, hatch_size, hatch_size * 4);
}

// RendererAgg 类的析构函数，释放动态分配的内存
RendererAgg::~RendererAgg()
{
    delete[] hatchBuffer;   // 释放 hatchBuffer 内存
    delete[] alphaBuffer;   // 释放 alpha 缓冲区内存
    delete[] pixBuffer;     // 释放像素缓冲区内存
}

// 创建 alpha 缓冲区
void RendererAgg::create_alpha_buffers()
{
    // 如果 alpha 缓冲区尚未创建
    if (!alphaBuffer) {
        // 分配 alpha 缓冲区内存
        alphaBuffer = new agg::int8u[width * height];
        // 将 alpha 缓冲区与渲染缓冲区关联
        alphaMaskRenderingBuffer.attach(alphaBuffer, width, height, width);
        // 将 alpha 蒙版像素格式与基础渲染器关联
        rendererBaseAlphaMask.attach(pixfmtAlphaMask);
        // 将 alpha 蒙版渲染器与基础 alpha 蒙版关联
        rendererAlphaMask.attach(rendererBaseAlphaMask);
    }
}

// 从边界框复制数据到新的缓冲区
BufferRegion *RendererAgg::copy_from_bbox(agg::rect_d in_rect)
{
    // 将浮点边界框转换为整数边界框
    agg::rect_i rect((int)in_rect.x1, height - (int)in_rect.y2,
                     (int)in_rect.x2, height - (int)in_rect.y1);

    BufferRegion *reg = NULL;
    // 创建一个新的缓冲区区域对象
    reg = new BufferRegion(rect);

    // 将区域数据与渲染缓冲区关联
    agg::rendering_buffer rbuf;
    rbuf.attach(reg->get_data(), reg->get_width(), reg->get_height(), reg->get_stride());

    // 使用像素格式和渲染基类复制数据到新缓冲区
    pixfmt pf(rbuf);
    renderer_base rb(pf);
    rb.copy_from(renderingBuffer, &rect, -rect.x1, -rect.y1);

    return reg; // 返回新创建的区域对象
}

// 恢复保存区域的部分数据，并带有偏移量
void RendererAgg::restore_region(BufferRegion &region)
{
    // 如果区域数据为空，抛出运行时错误
    if (region.get_data() == NULL) {
        throw std::runtime_error("Cannot restore_region from NULL data");
    }

    // 将区域数据与渲染缓冲区关联
    agg::rendering_buffer rbuf;
    rbuf.attach(region.get_data(), region.get_width(), region.get_height(), region.get_stride());

    // 使用基础渲染器从区域数据恢复到渲染缓冲区
    rendererBase.copy_from(rbuf, 0, region.get_rect().x1, region.get_rect().y1);
}

// 恢复保存区域的部分数据，并带有偏移量
void
RendererAgg::restore_region(BufferRegion &region, int xx1, int yy1, int xx2, int yy2, int x, int y )
{
    // 如果区域数据为空，抛出运行时错误
    if (region.get_data() == NULL) {
        throw std::runtime_error("Cannot restore_region from NULL data");
    }

    // 获取区域的整数边界框
    agg::rect_i &rrect = region.get_rect();

    // 省略部分代码注释以保持规定格式
}
    # 创建一个矩形对象 `rect`，其位置和大小是根据给定的坐标计算得到的
    agg::rect_i rect(xx1 - rrect.x1, (yy1 - rrect.y1), xx2 - rrect.x1, (yy2 - rrect.y1));
    
    # 创建一个渲染缓冲区对象 `rbuf`
    # 将 `region` 对象的数据、宽度、高度和步幅附加到渲染缓冲区 `rbuf` 上
    agg::rendering_buffer rbuf;
    rbuf.attach(region.get_data(), region.get_width(), region.get_height(), region.get_stride());
    
    # 使用 `rendererBase` 对象的 `copy_from` 方法
    # 从 `rbuf` 渲染缓冲区中指定的矩形 `rect` 复制数据到指定的坐标 `(x, y)`
    rendererBase.copy_from(rbuf, &rect, x, y);
}

bool RendererAgg::render_clippath(mpl::PathIterator &clippath,
                                  const agg::trans_affine &clippath_trans,
                                  e_snap_mode snap_mode)
{
    // 定义转换后的路径类型
    typedef agg::conv_transform<mpl::PathIterator> transformed_path_t;
    // 定义处理 NaN（不是数字）的路径类型
    typedef PathNanRemover<transformed_path_t> nan_removed_t;
    /* 与普通路径不同，剪切路径不能被剪切到Figure的边界框，
     * 因为它需要保持为完整的封闭路径，所以没有PathClipper<nan_removed_t>步骤。 */
    typedef PathSnapper<nan_removed_t> snapped_t;
    typedef PathSimplifier<snapped_t> simplify_t;
    typedef agg::conv_curve<simplify_t> curve_t;

    // 检查是否存在剪切路径
    bool has_clippath = (clippath.total_vertices() != 0);

    // 如果存在剪切路径且路径 ID 或变换矩阵与上次不同
    if (has_clippath &&
        (clippath.get_id() != lastclippath || clippath_trans != lastclippath_transform)) {
        // 创建 alpha 缓冲区
        create_alpha_buffers();
        // 创建变换矩阵并进行平移和缩放操作
        agg::trans_affine trans(clippath_trans);
        trans *= agg::trans_affine_scaling(1.0, -1.0);
        trans *= agg::trans_affine_translation(0.0, (double)height);

        // 清空渲染器的 alpha 掩码
        rendererBaseAlphaMask.clear(agg::gray8(0, 0));
        // 转换剪切路径并移除 NaN
        transformed_path_t transformed_clippath(clippath, trans);
        nan_removed_t nan_removed_clippath(transformed_clippath, true, clippath.has_codes());
        // 对路径进行捕捉操作
        snapped_t snapped_clippath(nan_removed_clippath, snap_mode, clippath.total_vertices(), 0.0);
        // 简化路径
        simplify_t simplified_clippath(snapped_clippath,
                                       clippath.should_simplify() && !clippath.has_codes(),
                                       clippath.simplify_threshold());
        // 创建曲线路径
        curve_t curved_clippath(simplified_clippath);
        // 将路径添加到光栅化器中
        theRasterizer.add_path(curved_clippath);
        // 设置渲染器 alpha 掩码的颜色
        rendererAlphaMask.color(agg::gray8(255, 255));
        // 渲染扫描线到 alpha 掩码上
        agg::render_scanlines(theRasterizer, scanlineAlphaMask, rendererAlphaMask);
        // 更新最后使用的剪切路径的 ID 和变换矩阵
        lastclippath = clippath.get_id();
        lastclippath_transform = clippath_trans;
    }

    // 返回是否存在剪切路径的标志
    return has_clippath;
}

void RendererAgg::clear()
{
    //"clear the rendered buffer";
    // 清空渲染缓冲区并填充指定的填充颜色
    rendererBase.clear(_fill_color);
}
```