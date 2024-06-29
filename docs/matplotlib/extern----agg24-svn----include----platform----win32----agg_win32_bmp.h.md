# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\platform\win32\agg_win32_bmp.h`

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
//
// class pixel_map
//
//----------------------------------------------------------------------------
#ifndef AGG_WIN32_BMP_INCLUDED
#define AGG_WIN32_BMP_INCLUDED

#include <windows.h>  // 包含 Windows 头文件，提供 Windows 平台的 API 支持
#include <stdio.h>    // 包含标准输入输出头文件

namespace agg
{
    // 像素地图的原始数据类型枚举
    enum org_e
    {
        org_mono8   = 8,    // 单色 8 位
        org_color16 = 16,   // 彩色 16 位
        org_color24 = 24,   // 彩色 24 位
        org_color32 = 32,   // 彩色 32 位
        org_color48 = 48,   // 彩色 48 位
        org_color64 = 64    // 彩色 64 位
    };

    // 像素地图类定义
    class pixel_map
    {
    public:
        ~pixel_map();        // 析构函数声明
        pixel_map();         // 默认构造函数声明
    // 定义一个公共接口（public）用于像素图像操作

    // 销毁像素图像对象，释放资源
    void destroy();

    // 创建像素图像，指定宽度、高度、原点和清除值，默认为256
    void create(unsigned width, 
                unsigned height, 
                org_e org,
                unsigned clear_val=256);

    // 在指定设备上创建DIB段，返回HBITMAP句柄
    HBITMAP create_dib_section(HDC h_dc,
                               unsigned width, 
                               unsigned height, 
                               org_e org,
                               unsigned clear_val=256);

    // 清除像素图像数据，默认使用清除值256
    void clear(unsigned clear_val=256);

    // 将像素图像对象关联到指定的BITMAPINFO结构
    void attach_to_bmp(BITMAPINFO* bmp);

    // 返回当前像素图像的BITMAPINFO结构指针
    BITMAPINFO* bitmap_info() { return m_bmp; }

    // 从文件流中加载BMP文件数据到像素图像对象，返回操作成功状态
    bool load_from_bmp(FILE* fd);

    // 将像素图像对象保存为BMP格式到文件流中，返回操作成功状态
    bool save_as_bmp(FILE* fd) const;

    // 从指定文件名加载BMP文件数据到像素图像对象，返回操作成功状态
    bool load_from_bmp(const char* filename);

    // 将像素图像对象保存为BMP格式到指定文件名，返回操作成功状态
    bool save_as_bmp(const char* filename) const;

    // 在指定设备上绘制像素图像，可指定设备区域和图像区域
    void draw(HDC h_dc, 
              const RECT* device_rect=0, 
              const RECT* bmp_rect=0) const;

    // 在指定设备上绘制像素图像，指定位置和缩放比例
    void draw(HDC h_dc, int x, int y, double scale=1.0) const;

    // 在指定设备上混合绘制像素图像，可指定设备区域和图像区域
    void blend(HDC h_dc, 
               const RECT* device_rect=0, 
               const RECT* bmp_rect=0) const;

    // 在指定设备上混合绘制像素图像，指定位置和缩放比例
    void blend(HDC h_dc, int x, int y, double scale=1.0) const;

    // 返回像素图像数据缓冲区的指针
    unsigned char* buf();

    // 返回像素图像的宽度
    unsigned width() const;

    // 返回像素图像的高度
    unsigned height() const;

    // 返回像素图像数据行的跨度
    int stride() const;

    // 返回每像素的位数（bpp）
    unsigned bpp() const { return m_bpp; }

    // 计算整个BITMAPINFO结构的大小
    static unsigned calc_full_size(BITMAPINFO *bmp);

    // 计算BITMAPINFO结构头部的大小
    static unsigned calc_header_size(BITMAPINFO *bmp);

    // 计算调色板大小，给定颜色数和每像素位数
    static unsigned calc_palette_size(unsigned clr_used, 
                                      unsigned bits_per_pixel);

    // 计算BITMAPINFO结构调色板的大小
    static unsigned calc_palette_size(BITMAPINFO *bmp);

    // 计算图像数据的指针位置
    static unsigned char* calc_img_ptr(BITMAPINFO *bmp);

    // 创建指定参数的BITMAPINFO结构
    static BITMAPINFO* create_bitmap_info(unsigned width, 
                                          unsigned height, 
                                          unsigned bits_per_pixel);

    // 创建灰度调色板的BITMAPINFO结构
    static void create_gray_scale_palette(BITMAPINFO *bmp);

    // 计算给定宽度和每像素位数的图像数据行长度
    static unsigned calc_row_len(unsigned width, unsigned bits_per_pixel);

private:
    // 禁止复制和赋值操作
    pixel_map(const pixel_map&);
    const pixel_map& operator = (const pixel_map&);

    // 从BITMAPINFO结构创建像素图像
    void create_from_bmp(BITMAPINFO *bmp);

    // 在指定设备上创建DIB段，返回HBITMAP句柄，使用给定参数
    HBITMAP create_dib_section_from_args(HDC h_dc,
                                         unsigned width,
                                         unsigned height,
                                         unsigned bits_per_pixel);

private:
    BITMAPINFO*    m_bmp;           // 指向位图信息结构的指针
    unsigned char* m_buf;           // 指向像素数据缓冲区的指针
    unsigned       m_bpp;           // 每像素位数（bits per pixel）
    bool           m_is_internal;   // 标志位，指示是否是内部创建的像素图像
    unsigned       m_img_size;      // 图像数据大小
    unsigned       m_full_size;     // 完整BITMAPINFO结构大小
};
}


这行代码表示一个C或C++的预处理器指令，用于结束一个条件编译的代码块。


#endif


这行代码是预处理器指令的一部分，用于结束一个条件编译的代码块，与 `#ifdef` 或 `#if` 配对使用，用于控制编译器在特定条件下是否包含某段代码。

这两行代码通常用于确保在特定条件下才编译某段代码，可以根据预定义的宏或条件判断是否包含或排除一部分代码。
```