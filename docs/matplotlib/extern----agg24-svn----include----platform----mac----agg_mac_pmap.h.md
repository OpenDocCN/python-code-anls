# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\platform\mac\agg_mac_pmap.h`

```py
//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4 
// Copyright (C) 2002-2005 Maxim Shemanarev (McSeem)
// Copyright (C) 2002 Hansruedi Baer (MacOS support)
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
//            baer@karto.baug.eth.ch
//----------------------------------------------------------------------------
//
// class pixel_map
//
//----------------------------------------------------------------------------
#ifndef AGG_MAC_PMAP_INCLUDED
#define AGG_MAC_PMAP_INCLUDED

// 包含标准输入输出头文件
#include <stdio.h>
// 包含 Carbon 库
#include <Carbon.h>

// 使用 agg 命名空间
namespace agg
{
    // 定义像素映射的不同颜色组织方式
    enum org_e
    {
        org_mono8   = 8,
        org_color16 = 16,
        org_color24 = 24,
        org_color32 = 32
    };

    // 像素映射类声明
    class pixel_map
    {
    public:
        // 析构函数声明
        ~pixel_map();
        // 默认构造函数声明
        pixel_map();

    public:
        // 销毁像素映射对象
        void        destroy();
        // 创建像素映射对象
        void        create(unsigned width, 
                           unsigned height, 
                           org_e    org,
                           unsigned clear_val=255);

        // 清空像素映射对象
        void        clear(unsigned clear_val=255);
        // 从 QuickTime 文件加载像素映射数据
        bool        load_from_qt(const char* filename);
        // 将像素映射对象保存为 QuickTime 文件
        bool        save_as_qt(const char* filename) const;

        // 将像素映射对象绘制到指定窗口
        void        draw(WindowRef window, 
                         const Rect* device_rect=0, 
                         const Rect* bmp_rect=0) const;
        // 将像素映射对象按指定比例绘制到指定位置
        void        draw(WindowRef window, int x, int y, double scale=1.0) const;
        // 将像素映射对象混合到指定窗口
        void        blend(WindowRef window, 
                          const Rect* device_rect=0, 
                          const Rect* bmp_rect=0) const;
        // 将像素映射对象按指定比例混合到指定位置
        void        blend(WindowRef window, int x, int y, double scale=1.0) const;

        // 获取像素映射数据缓冲区指针
        unsigned char* buf();
        // 获取像素映射宽度
        unsigned       width() const;
        // 获取像素映射高度
        unsigned       height() const;
        // 获取像素映射每行字节数
        int            row_bytes() const;
        // 获取像素映射的位深度
        unsigned       bpp() const { return m_bpp; }

        // 辅助静态函数，计算给定位深度下像素映射每行字节数
        static unsigned calc_row_len(unsigned width, unsigned bits_per_pixel);
    private:
        // 复制构造函数声明（禁用）
        pixel_map(const pixel_map&);
        // 赋值操作符声明（禁用）
        const pixel_map& operator = (const pixel_map&);

    private:
        // 像素映射的 GWorldPtr
        GWorldPtr      m_pmap;
        // 像素映射数据缓冲区指针
        unsigned char* m_buf;
        // 像素映射的位深度
        unsigned       m_bpp;
        // 像素映射的图像数据大小
        unsigned       m_img_size;
    };

}

#endif
```