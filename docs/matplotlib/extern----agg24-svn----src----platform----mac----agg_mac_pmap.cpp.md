# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\platform\mac\agg_mac_pmap.cpp`

```
//----------------------------------------------------------------------------
//
//----------------------------------------------------------------------------
// Contact: mcseemagg@yahoo.com
//          baer@karto.baug.ethz.ch
//----------------------------------------------------------------------------
//
// class pixel_map
//
//----------------------------------------------------------------------------

#include <string.h> // 包含标准库函数 string.h，用于内存操作
#include <Carbon.h> // 包含 Carbon 库，为 MacOS 经典环境提供支持
#include <QuickTimeComponents.h> // 包含 QuickTime 组件头文件
#include <ImageCompression.h> // 包含图像压缩相关头文件
#include "platform/mac/agg_mac_pmap.h" // 包含 Mac 平台相关的像素映射头文件
#include "agg_basics.h" // 包含 AGG 库的基础功能头文件

namespace agg
{

    //------------------------------------------------------------------------
    // 析构函数，销毁对象时调用
    pixel_map::~pixel_map()
    {
        destroy();
    }


    //------------------------------------------------------------------------
    // 默认构造函数，初始化成员变量
    pixel_map::pixel_map() :
        m_pmap(0),
        m_buf(0),
        m_bpp(0),
        m_img_size(0)
    {
    }


    //------------------------------------------------------------------------
    // 销毁像素映射对象
    void pixel_map::destroy()
    {
        delete[] m_buf; // 删除像素数据缓冲区
        m_buf = NULL;
        if (m_pmap != nil) // 如果 GWorld 对象不为空
        {
            DisposeGWorld(m_pmap); // 销毁 GWorld 对象
            m_pmap = nil;
        }
    }


    //------------------------------------------------------------------------
    // 创建像素映射对象
    void pixel_map::create(unsigned width, 
                           unsigned height, 
                           org_e    org,
                           unsigned clear_val)
    {
        destroy(); // 先销毁已有的对象
        if(width == 0)  width = 1; // 宽度为零时设为最小值1
        if(height == 0) height = 1; // 高度为零时设为最小值1
        m_bpp = org; // 设置像素映射对象的原点类型
        
        Rect    r; // 定义矩形对象 r
        int     row_bytes = calc_row_len (width, m_bpp); // 计算每行字节数
        MacSetRect(&r, 0, 0, width, height); // 设置矩形 r 的位置和大小
        m_buf = new unsigned char[m_img_size = row_bytes * height]; // 分配像素数据缓冲区
        QTNewGWorldFromPtr (&m_pmap, m_bpp, &r, nil, nil, 0, m_buf, row_bytes); // 使用 QuickTime API 创建 GWorld 对象

        // create_gray_scale_palette(m_pmap);  暂时不处理灰度调色板

        if(clear_val <= 255)
        {
            memset(m_buf, clear_val, m_img_size); // 使用指定值填充像素数据缓冲区
        }
    }


    //------------------------------------------------------------------------
    // 清除像素映射对象的像素数据
    void pixel_map::clear(unsigned clear_val)
    {
        if(m_buf) memset(m_buf, clear_val, m_img_size); // 如果像素数据缓冲区存在，则使用指定值清除
    }


    //static
    // 该函数从 Win32 平台支持中复制过来，似乎也适用于 MacOS，但还没有充分测试过。
    //------------------------------------------------------------------------
    // 计算每行像素数据所需的字节数
    unsigned pixel_map::calc_row_len(unsigned width, unsigned bits_per_pixel)
    {
        // 根据像素宽度和像素深度计算所需的字节数
        unsigned n = width;
        unsigned k;
    
        // 根据每像素位数进行不同情况的处理
        switch(bits_per_pixel)
        {
            // 每像素1位
            case  1: k = n;
                     n = n >> 3;          // 每字节8位，右移3位相当于除以8
                     if(k & 7) n++;       // 如果像素宽度不是8的倍数，则需要额外的字节
                     break;
    
            // 每像素4位
            case  4: k = n;
                     n = n >> 1;          // 每字节2位，右移1位相当于除以2
                     if(k & 3) n++;       // 如果像素宽度不是4的倍数，则需要额外的字节
                     break;
    
            // 每像素8位
            case  8:
                     break;
    
            // 每像素16位
            case 16: n = n << 1;          // 每像素占2字节，左移1位相当于乘以2
                     break;
    
            // 每像素24位
            case 24: n = (n << 1) + n;    // 每像素占3字节，左移1位相当于乘以2，再加上原值
                     break;
    
            // 每像素32位
            case 32: n = n << 2;          // 每像素占4字节，左移2位相当于乘以4
                     break;
    
            // 默认情况（未知像素深度）
            default: n = 0;
                     break;
        }
    
        // 返回按4字节对齐后的字节数
        return ((n + 3) >> 2) << 2;
    }
    
    
    
    
    //------------------------------------------------------------------------
    void pixel_map::draw(WindowRef window, const Rect *device_rect, const Rect *pmap_rect) const
    {
        if(m_pmap == nil || m_buf == NULL) return;
    
        // 获取 PixMap 句柄和窗口图形上下文
        PixMapHandle pm = GetGWorldPixMap(m_pmap);
        CGrafPtr port = GetWindowPort(window);
        Rect dest_rect;
    
        // 设置目标绘制矩形
        MacSetRect(&dest_rect, 0, 0, this->width(), this->height());
        ImageDescriptionHandle image_description;
    
        // 为 PixMap 创建图像描述
        MakeImageDescriptionForPixMap(pm, &image_description);
        
        // 如果成功创建了图像描述
        if (image_description != nil)
        {
            // 解压缩并绘制图像到指定的设备矩形
            DecompressImage(GetPixBaseAddr(pm), image_description, GetPortPixMap(port), nil, &dest_rect, ditherCopy, nil);
            DisposeHandle((Handle)image_description);
        }
    }
    
    
    
    //------------------------------------------------------------------------
    void pixel_map::draw(WindowRef window, int x, int y, double scale) const
    {
        if(m_pmap == nil || m_buf == NULL) return;
    
        // 计算缩放后的宽度和高度
        unsigned width  = (unsigned)(this->width() * scale);
        unsigned height = (unsigned)(this->height() * scale);
        Rect rect;
    
        // 设置绘制矩形
        SetRect(&rect, x, y, x + width, y + height);
    
        // 调用前一个重载函数进行绘制
        draw(window, &rect);
    }
    
    
    
    //------------------------------------------------------------------------
    void pixel_map::blend(WindowRef window, const Rect *device_rect, const Rect *bmp_rect) const
    {
        // 目前仅映射到绘制方法
        draw(window, device_rect, bmp_rect);
    }
    
    
    
    //------------------------------------------------------------------------
    void pixel_map::blend(WindowRef window, int x, int y, double scale) const
    {
        // 目前仅映射到绘制方法
        draw(window, x, y, scale);
    }
    
    
    // 我让 Quicktime 处理图像导入，因为它支持大多数流行的图像格式，
    // 如：*.psd, *.bmp, *.tif, *.png, *.jpg, *.gif, *.pct, *.pcx
    //------------------------------------------------------------------------
    bool pixel_map::load_from_qt(const char *filename)
    {
        FSSpec                        fss;
        OSErr                        err;
        
        // 获取应用程序目录的文件规范
        err = HGetVol(nil, &fss.vRefNum, &fss.parID);
        // 如果没有错误发生，则继续
        if (err == noErr)
        {
            // 将C字符串复制到Pascal字符串
            CopyCStringToPascal(filename, fss.name);
            // 图形导入组件对象
            GraphicsImportComponent        gi;
            // 获取指定文件的图形导入器
            err = GetGraphicsImporterForFile (&fss, &gi);
            // 如果没有错误发生，则继续
            if (err == noErr)
            {
                // 图像描述句柄
                ImageDescriptionHandle    desc;
                // 获取图形导入器的图像描述
                GraphicsImportGetImageDescription(gi, &desc);
// 对所有图像简化为32位深度进行处理。
// 创建一个空的像素图
short depth = 32;
create((**desc).width, (**desc).height, (org_e)depth, 0xff);
DisposeHandle((Handle)desc);
// 使用 Quicktime 绘制到像素图
GraphicsImportSetGWorld(gi, m_pmap, nil);
GraphicsImportDraw(gi);

// 这是一个技巧。图形导入器会将没有 alpha 通道的导入图像的像素图的 alpha 通道设置为 0x00，
// 这会导致 agg 绘制一个看不见的图像。
// 将 alpha 通道设置为 0xff
unsigned char *buf = m_buf;
for (unsigned int size = 0; size < m_img_size; size += 4)
{
    *buf = 0xff;
    buf += 4;
}
}


//------------------------------------------------------------------------
bool pixel_map::save_as_qt(const char *filename) const
{
    FSSpec fss;
    OSErr err;
    
    // 获取应用程序目录的文件规范
    err = HGetVol(nil, &fss.vRefNum, &fss.parID);
    if (err == noErr)
    {
        GraphicsExportComponent ge;
        CopyCStringToPascal(filename, fss.name);
        // 我决定使用 PNG 作为输出图像文件类型。
        // 还有很多其他可用的格式。
        // 我是否应该检查文件后缀来选择图像文件格式？
        err = OpenADefaultComponent(GraphicsExporterComponentType, kQTFileTypePNG, &ge);
        if (err == noErr)
        {
            err = GraphicsExportSetInputGWorld(ge, m_pmap);
            if (err == noErr)
            {
                err = GraphicsExportSetOutputFile(ge, &fss);
                if (err == noErr)
                {
                    GraphicsExportDoExport(ge, nil);
                }
            }
            CloseComponent(ge);
        }
    }
    
    return err == noErr;
}

//------------------------------------------------------------------------
unsigned char* pixel_map::buf()
{
    return m_buf;
}

//------------------------------------------------------------------------
unsigned pixel_map::width() const
{
    if(m_pmap == nil) return 0;
    PixMapHandle pm = GetGWorldPixMap(m_pmap);
    Rect bounds;
    GetPixBounds(pm, &bounds);
    return bounds.right - bounds.left;
}

//------------------------------------------------------------------------
unsigned pixel_map::height() const
    {
        // 如果 m_pmap 指针为空，则返回 0
        if(m_pmap == nil) return 0;
        // 通过 m_pmap 获取 PixMapHandle 对象
        PixMapHandle pm = GetGWorldPixMap(m_pmap);
        // 定义一个 Rect 结构体变量 bounds
        Rect bounds;
        // 获取 pm 对应的像素边界信息，存储在 bounds 变量中
        GetPixBounds(pm, &bounds);
        // 返回图像的垂直尺寸，即底部坐标减去顶部坐标
        return bounds.bottom - bounds.top;
    }
    
    
    
    //------------------------------------------------------------------------
    int pixel_map::row_bytes() const
    {
        // 如果 m_pmap 指针为空，则返回 0
        if(m_pmap == nil) return 0;
        // 通过 m_pmap 获取 PixMapHandle 对象
        PixMapHandle pm = GetGWorldPixMap(m_pmap);
        // 调用 calc_row_len 函数计算一行像素数据的长度，并返回结果
        return calc_row_len(width(), GetPixDepth(pm));
    }
    
    
    这些注释详细解释了每行代码的作用，包括变量的声明和初始化，函数调用，以及条件判断，以帮助读者理解代码的执行过程和目的。
}
```