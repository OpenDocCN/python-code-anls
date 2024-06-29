# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\platform\win32\agg_win32_bmp.cpp`

```
//----------------------------------------------------------------------------
//
//----------------------------------------------------------------------------
// Contact: mcseemagg@yahoo.com
//----------------------------------------------------------------------------
//
// class pixel_map
//
//----------------------------------------------------------------------------

#include "platform/win32/agg_win32_bmp.h"
#include "agg_basics.h"

namespace agg
{

    //------------------------------------------------------------------------
    // 析构函数，用于释放资源
    pixel_map::~pixel_map()
    {
        destroy();
    }


    //------------------------------------------------------------------------
    // 默认构造函数，初始化成员变量
    pixel_map::pixel_map() :
        m_bmp(0),
        m_buf(0),
        m_bpp(0),
        m_is_internal(false),
        m_img_size(0),
        m_full_size(0)
    {
    }


    //------------------------------------------------------------------------
    // 销毁像素图，释放内存
    void pixel_map::destroy()
    {
        if(m_bmp && m_is_internal) delete [] (unsigned char*)m_bmp;
        m_bmp  = 0;
        m_is_internal = false;
        m_buf = 0;
    }


    //------------------------------------------------------------------------
    // 创建指定尺寸的像素图
    void pixel_map::create(unsigned width, 
                           unsigned height, 
                           org_e    org,
                           unsigned clear_val)
    {
        destroy();
        if(width == 0)  width = 1;
        if(height == 0) height = 1;
        m_bpp = org;
        create_from_bmp(create_bitmap_info(width, height, m_bpp));
        create_gray_scale_palette(m_bmp);
        m_is_internal = true;
        if(clear_val <= 255)
        {
            memset(m_buf, clear_val, m_img_size);
        }
    }


    //------------------------------------------------------------------------
    // 从设备上下文创建 DIB 段
    HBITMAP pixel_map::create_dib_section(HDC h_dc,
                                          unsigned width, 
                                          unsigned height, 
                                          org_e    org,
                                          unsigned clear_val)
    {
        destroy();
        if(width == 0)  width = 1;
        if(height == 0) height = 1;
        m_bpp = org;
        HBITMAP h_bitmap = create_dib_section_from_args(h_dc, width, height, m_bpp);
        create_gray_scale_palette(m_bmp);
        m_is_internal = true;
        if(clear_val <= 255)
        {
            memset(m_buf, clear_val, m_img_size);
        }
        return h_bitmap;
    }


    //------------------------------------------------------------------------
    // 清空像素图中的数据
    void pixel_map::clear(unsigned clear_val)
    {
        if(m_buf) memset(m_buf, clear_val, m_img_size);
    }


    //------------------------------------------------------------------------
    // 将像素图附加到指定的位图信息结构
    void pixel_map::attach_to_bmp(BITMAPINFO *bmp)
    {
        if(bmp)
        {
            destroy();
            create_from_bmp(bmp);
            m_is_internal = false;
        }
    }
    //------------------------------------------------------------------------
    // 计算完整的位图信息结构体大小
    unsigned pixel_map::calc_full_size(BITMAPINFO *bmp)
    {
        // 如果传入的位图信息指针为空，返回大小为0
        if(bmp == 0) return 0;
    
        // 计算并返回包括位图信息头、调色板和图像数据大小的总大小
        return sizeof(BITMAPINFOHEADER) +
               sizeof(RGBQUAD) * calc_palette_size(bmp) +
               bmp->bmiHeader.biSizeImage;
    }
    
    //static
    //------------------------------------------------------------------------
    // 计算仅包含位图信息头和调色板大小的大小
    unsigned pixel_map::calc_header_size(BITMAPINFO *bmp)
    {
        // 如果传入的位图信息指针为空，返回大小为0
        if(bmp == 0) return 0;
    
        // 返回包括位图信息头和调色板大小的总大小
        return sizeof(BITMAPINFOHEADER) + sizeof(RGBQUAD) * calc_palette_size(bmp);
    }
    
    
    //static
    //------------------------------------------------------------------------
    // 计算调色板的大小
    unsigned  pixel_map::calc_palette_size(unsigned  clr_used, unsigned bits_per_pixel)
    {
        int palette_size = 0;
    
        // 如果像素位数小于等于8位
        if(bits_per_pixel <= 8)
        {
            // 使用的颜色数等于传入的clr_used
            palette_size = clr_used;
            // 如果clr_used为0，则调色板大小为2的bits_per_pixel次方
            if(palette_size == 0)
            {
                palette_size = 1 << bits_per_pixel;
            }
        }
        // 返回计算得到的调色板大小
        return palette_size;
    }
    
    //static
    //------------------------------------------------------------------------
    // 根据位图信息计算调色板的大小
    unsigned pixel_map::calc_palette_size(BITMAPINFO *bmp)
    {
        // 如果传入的位图信息指针为空，返回大小为0
        if(bmp == 0) return 0;
    
        // 调用前一个静态方法计算调色板的大小
        return calc_palette_size(bmp->bmiHeader.biClrUsed, bmp->bmiHeader.biBitCount);
    }
    
    
    //static
    //------------------------------------------------------------------------
    // 计算图像数据的起始指针
    unsigned char * pixel_map::calc_img_ptr(BITMAPINFO *bmp)
    {
        // 如果传入的位图信息指针为空，返回空指针
        if(bmp == 0) return 0;
    
        // 返回位图信息地址加上位图头和调色板大小后的偏移量，即图像数据的起始指针
        return ((unsigned char*)bmp) + calc_header_size(bmp);
    }
    
    //static
    //------------------------------------------------------------------------
    // 创建位图信息结构体并初始化
    BITMAPINFO* pixel_map::create_bitmap_info(unsigned width, 
                                              unsigned height, 
                                              unsigned bits_per_pixel)
    {
        // 计算每行像素的字节数
        unsigned line_len = calc_row_len(width, bits_per_pixel);
        // 计算图像数据的总字节数
        unsigned img_size = line_len * height;
        // 计算调色板的字节数
        unsigned rgb_size = calc_palette_size(0, bits_per_pixel) * sizeof(RGBQUAD);
        // 计算完整的位图信息结构体大小
        unsigned full_size = sizeof(BITMAPINFOHEADER) + rgb_size + img_size;
    
        // 分配内存空间来存放位图信息结构体
        BITMAPINFO *bmp = (BITMAPINFO *) new unsigned char[full_size];
    
        // 初始化位图信息结构体的各个字段
        bmp->bmiHeader.biSize   = sizeof(BITMAPINFOHEADER);
        bmp->bmiHeader.biWidth  = width;
        bmp->bmiHeader.biHeight = height;
        bmp->bmiHeader.biPlanes = 1;
        bmp->bmiHeader.biBitCount = (unsigned short)bits_per_pixel;
        bmp->bmiHeader.biCompression = 0;
        bmp->bmiHeader.biSizeImage = img_size;
        bmp->bmiHeader.biXPelsPerMeter = 0;
        bmp->bmiHeader.biYPelsPerMeter = 0;
        bmp->bmiHeader.biClrUsed = 0;
        bmp->bmiHeader.biClrImportant = 0;
    
        // 返回创建的位图信息结构体指针
        return bmp;
    }
    
    //static
    //------------------------------------------------------------------------
    // 创建灰度调色板
    void pixel_map::create_gray_scale_palette(BITMAPINFO *bmp)
    {
        // 如果传入的 bmp 指针为空，则直接返回，不进行任何操作
        if(bmp == 0) return;
    
        // 计算调色板的大小
        unsigned rgb_size = calc_palette_size(bmp);
        // 获取指向调色板数据的指针
        RGBQUAD *rgb = (RGBQUAD*)(((unsigned char*)bmp) + sizeof(BITMAPINFOHEADER));
        // 用于存储亮度值的变量
        unsigned brightness;
        unsigned i;
    
        // 遍历调色板中的每个颜色项
        for(i = 0; i < rgb_size; i++)
        {
            // 计算当前颜色的亮度值
            brightness = (255 * i) / (rgb_size - 1);
            // 设置当前颜色的蓝色通道、绿色通道和红色通道为相同的亮度值
            rgb->rgbBlue =
            rgb->rgbGreen =  
            rgb->rgbRed = (unsigned char)brightness; 
            // 设置当前颜色的 Alpha 通道为 0（假定为不透明）
            rgb->rgbReserved = 0;
            // 移动到下一个颜色项
            rgb++;
        }
    }
    
    
    
    //static
    //------------------------------------------------------------------------
    // 计算每行像素数据在内存中的长度
    unsigned pixel_map::calc_row_len(unsigned width, unsigned bits_per_pixel)
    {
        // 将宽度值保存在 n 中
        unsigned n = width;
        unsigned k;
    
        // 根据每像素位数不同采取不同的计算方法
        switch(bits_per_pixel)
        {
            // 每像素1位，按位数计算行长度
            case  1: k = n;
                     n = n >> 3;
                     if(k & 7) n++; 
                     break;
    
            // 每像素4位，按位数计算行长度
            case  4: k = n;
                     n = n >> 1;
                     if(k & 3) n++; 
                     break;
    
            // 每像素8位，长度不变
            case  8:
                     break;
    
            // 每像素16位，长度乘以2
            case 16: n *= 2;
                     break;
    
            // 每像素24位，长度乘以3
            case 24: n *= 3; 
                     break;
    
            // 每像素32位，长度乘以4
            case 32: n *= 4;
                     break;
    
            // 每像素48位，长度乘以6
            case 48: n *= 6; 
                     break;
    
            // 每像素64位，长度乘以8
            case 64: n *= 8; 
                     break;
    
            // 每像素96位，长度乘以12
            case 96: n *= 12; 
                     break;
    
            // 每像素128位，长度乘以16
            case 128: n *= 16; 
                      break;
    
            // 默认情况下，像素位数不在已知范围内，行长度设为0
            default: n = 0;
                     break;
        }
        // 返回四字节对齐后的行长度
        return ((n + 3) >> 2) << 2;
    }
    
    
    
    //------------------------------------------------------------------------
    // 将位图绘制到指定的设备上下文中的指定矩形区域内
    void pixel_map::draw(HDC h_dc, const RECT *device_rect, const RECT *bmp_rect) const
    {
        // 检查 m_bmp 和 m_buf 是否为空，如果是则直接返回
        if(m_bmp == 0 || m_buf == 0) return;
    
        // 初始化 BMP 图像的起始位置和大小
        unsigned bmp_x = 0;
        unsigned bmp_y = 0;
        unsigned bmp_width  = m_bmp->bmiHeader.biWidth;
        unsigned bmp_height = m_bmp->bmiHeader.biHeight;
    
        // 初始化设备上下文的起始位置和大小，初始值与 BMP 图像相同
        unsigned dvc_x = 0;
        unsigned dvc_y = 0; 
        unsigned dvc_width  = m_bmp->bmiHeader.biWidth;
        unsigned dvc_height = m_bmp->bmiHeader.biHeight;
        
        // 如果有指定 BMP 的矩形区域 bmp_rect，则更新 BMP 图像的位置和大小
        if(bmp_rect) 
        {
            bmp_x      = bmp_rect->left;
            bmp_y      = bmp_rect->top;
            bmp_width  = bmp_rect->right  - bmp_rect->left;
            bmp_height = bmp_rect->bottom - bmp_rect->top;
        } 
    
        // 将设备上下文的位置和大小初始化为 BMP 图像的位置和大小
        dvc_x      = bmp_x;
        dvc_y      = bmp_y;
        dvc_width  = bmp_width;
        dvc_height = bmp_height;
    
        // 如果有指定设备的矩形区域 device_rect，则更新设备上下文的位置和大小
        if(device_rect) 
        {
            dvc_x      = device_rect->left;
            dvc_y      = device_rect->top;
            dvc_width  = device_rect->right  - device_rect->left;
            dvc_height = device_rect->bottom - device_rect->top;
        }
    
        // 如果设备上下文的宽度或高度与 BMP 图像的宽度或高度不相等，则进行拉伸绘制
        if(dvc_width != bmp_width || dvc_height != bmp_height)
        {
            // 设置拉伸绘制模式为 COLORONCOLOR
            ::SetStretchBltMode(h_dc, COLORONCOLOR);
            // 执行拉伸绘制操作
            ::StretchDIBits(
                h_dc,            // 设备上下文句柄
                dvc_x,           // 源矩形左上角的 x 坐标
                dvc_y,           // 源矩形左上角的 y 坐标
                dvc_width,       // 源矩形的宽度
                dvc_height,      // 源矩形的高度
                bmp_x,           // 目标矩形左上角的 x 坐标
                bmp_y,           // 目标矩形左上角的 y 坐标
                bmp_width,       // 目标矩形的宽度
                bmp_height,      // 目标矩形的高度
                m_buf,           // 位图数据的地址
                m_bmp,           // 位图信息的地址
                DIB_RGB_COLORS,  // 使用的颜色模式
                SRCCOPY          // 光栅操作代码
            );
        }
        else
        {
            // 执行普通绘制操作
            ::SetDIBitsToDevice(
                h_dc,            // 设备上下文句柄
                dvc_x,           // 目标矩形左上角的 x 坐标
                dvc_y,           // 目标矩形左上角的 y 坐标
                dvc_width,       // 源矩形的宽度
                dvc_height,      // 源矩形的高度
                bmp_x,           // 源矩形左下角的 x 坐标
                bmp_y,           // 源矩形左下角的 y 坐标
                0,               // 数组中第一个扫描行
                bmp_height,      // 扫描行数
                m_buf,           // 包含 DIB 位的数组地址
                m_bmp,           // 位图信息的地址
                DIB_RGB_COLORS   // RGB 或调色板索引
            );
        }
    }
    //------------------------------------------------------------------------
    void pixel_map::draw(HDC h_dc, int x, int y, double scale) const
    {
        // 如果位图或缓冲区为空，则直接返回，不进行绘制操作
        if(m_bmp == 0 || m_buf == 0) return;
    
        // 计算经过缩放后的位图宽度和高度
        unsigned width  = unsigned(m_bmp->bmiHeader.biWidth * scale);
        unsigned height = unsigned(m_bmp->bmiHeader.biHeight * scale);
    
        // 创建矩形区域并设定其位置和大小
        RECT rect;
        rect.left   = x;
        rect.top    = y;
        rect.right  = x + width;
        rect.bottom = y + height;
    
        // 调用另一个重载的 draw 函数，传递绘制设备上下文和矩形区域进行绘制
        draw(h_dc, &rect);
    }
    
    
    
    
    //------------------------------------------------------------------------
    void pixel_map::blend(HDC h_dc, const RECT *device_rect, const RECT *bmp_rect) const
    {
#if !defined(AGG_BMP_ALPHA_BLEND)
        // 如果未定义 AGG_BMP_ALPHA_BLEND 宏，则直接调用 draw 函数绘制，然后返回
        draw(h_dc, device_rect, bmp_rect);
        return;
#else
        // 如果定义了 AGG_BMP_ALPHA_BLEND 宏
        if(m_bpp != 32)
        {
            // 如果位深度不为32位，则调用 draw 函数绘制，然后返回
            draw(h_dc, device_rect, bmp_rect);
            return;
        }

        // 如果位深度为32位且 m_bmp 或 m_buf 为0，则直接返回
        if(m_bmp == 0 || m_buf == 0) return;

        // 初始化变量用于存储图像和设备矩形的位置和大小
        unsigned bmp_x = 0;
        unsigned bmp_y = 0;
        unsigned bmp_width  = m_bmp->bmiHeader.biWidth;
        unsigned bmp_height = m_bmp->bmiHeader.biHeight;
        unsigned dvc_x = 0;
        unsigned dvc_y = 0; 
        unsigned dvc_width  = m_bmp->bmiHeader.biWidth;
        unsigned dvc_height = m_bmp->bmiHeader.biHeight;
        
        // 如果有指定 bmp_rect，则更新 bmp_x, bmp_y, bmp_width, bmp_height
        if(bmp_rect) 
        {
            bmp_x      = bmp_rect->left;
            bmp_y      = bmp_rect->top;
            bmp_width  = bmp_rect->right  - bmp_rect->left;
            bmp_height = bmp_rect->bottom - bmp_rect->top;
        } 

        // 将设备矩形的位置和大小设置为与 bmp_rect 相同
        dvc_x      = bmp_x;
        dvc_y      = bmp_y;
        dvc_width  = bmp_width;
        dvc_height = bmp_height;

        // 如果有指定 device_rect，则更新 dvc_x, dvc_y, dvc_width, dvc_height
        if(device_rect) 
        {
            dvc_x      = device_rect->left;
            dvc_y      = device_rect->top;
            dvc_width  = device_rect->right  - device_rect->left;
            dvc_height = device_rect->bottom - device_rect->top;
        }

        // 创建兼容于 h_dc 的内存设备上下文 mem_dc
        HDC mem_dc = ::CreateCompatibleDC(h_dc);
        // 初始化 buf 为0，创建一个与 m_bmp 兼容的 DIB section，将 m_buf 的数据复制到 buf 中
        void* buf = 0;
        HBITMAP bmp = ::CreateDIBSection(
            mem_dc, 
            m_bmp,  
            DIB_RGB_COLORS,
            &buf,
            0,
            0
        );
        memcpy(buf, m_buf, m_bmp->bmiHeader.biSizeImage);

        // 选择创建的位图对象 bmp 到内存设备上下文 mem_dc，并保存之前的位图对象到 temp
        HBITMAP temp = (HBITMAP)::SelectObject(mem_dc, bmp);

        // 初始化混合函数所需的 BLENDFUNCTION 结构体 blend
        BLENDFUNCTION blend;
        blend.BlendOp = AC_SRC_OVER;
        blend.BlendFlags = 0;

#if defined(AC_SRC_ALPHA)
        // 如果定义了 AC_SRC_ALPHA 宏，则设置 blend 的 AlphaFormat 为 AC_SRC_ALPHA
        blend.AlphaFormat = AC_SRC_ALPHA;
//#elif defined(AC_SRC_NO_PREMULT_ALPHA)
//        blend.AlphaFormat = AC_SRC_NO_PREMULT_ALPHA;
#else 
// 如果未定义合适的 AlphaFormat 常量，输出错误信息
#error "No appropriate constant for alpha format. Check version of wingdi.h, There must be AC_SRC_ALPHA or AC_SRC_NO_PREMULT_ALPHA"
#endif

        // 设置混合函数的透明度为 255（不透明）
        blend.SourceConstantAlpha = 255;
        // 执行 AlphaBlend 操作，将 mem_dc 中的图像混合到 h_dc 中指定的位置和大小
        ::AlphaBlend(
          h_dc,      
          dvc_x,      
          dvc_y,      
          dvc_width,  
          dvc_height, 
          mem_dc,
          bmp_x,
          bmp_y,     
          bmp_width, 
          bmp_height,
          blend
        );

        // 恢复 mem_dc 到之前保存的临时位图对象 temp
        ::SelectObject(mem_dc, temp);
        // 删除创建的位图对象 bmp 和内存设备上下文 mem_dc
        ::DeleteObject(bmp);
        ::DeleteObject(mem_dc);
#endif //defined(AGG_BMP_ALPHA_BLEND)
    }


    //------------------------------------------------------------------------
    void pixel_map::blend(HDC h_dc, int x, int y, double scale) const
    {
        // 如果 m_bmp 或 m_buf 为0，则直接返回
        if(m_bmp == 0 || m_buf == 0) return;
        // 计算经过缩放后的宽度和高度
        unsigned width  = unsigned(m_bmp->bmiHeader.biWidth * scale);
        unsigned height = unsigned(m_bmp->bmiHeader.biHeight * scale);
        RECT rect;
        // 设置矩形的位置和大小
        rect.left   = x;
        rect.top    = y;
        rect.right  = x + width;
        rect.bottom = y + height;
        // 调用 blend 函数进行混合操作
        blend(h_dc, &rect);
    }
    //------------------------------------------------------------------------
    bool pixel_map::load_from_bmp(FILE *fd)
    {
        // 读取位图文件头
        BITMAPFILEHEADER bmf;
        fread(&bmf, sizeof(bmf), 1, fd);
        // 检查文件类型是否为BM（0x4D42是BM的ASCII码）
        if(bmf.bfType != 0x4D42) goto bmperr;
    
        // 计算位图数据的大小
        unsigned bmp_size = bmf.bfSize - sizeof(BITMAPFILEHEADER);
    
        // 分配内存以存储位图信息头
        BITMAPINFO *bmi = (BITMAPINFO*) new unsigned char [bmp_size];
        // 从文件中读取位图信息头
        if(fread(bmi, 1, bmp_size, fd) != bmp_size) goto bmperr;
    
        // 销毁当前对象的数据
        destroy();
        // 设置像素位深度
        m_bpp = bmi->bmiHeader.biBitCount;
        // 从位图信息头创建像素图像
        create_from_bmp(bmi);
        // 设置内部标志为1
        m_is_internal = 1;
        return true;
    
    bmperr:
        // 如果出现错误，释放分配的内存并返回false
        if(bmi) delete [] (unsigned char*) bmi;
        return false;
    }
    
    
    //------------------------------------------------------------------------
    bool pixel_map::load_from_bmp(const char *filename)
    {
        // 打开指定文件
        FILE *fd = fopen(filename, "rb");
        bool ret = false;
        if(fd)
        {
            // 调用load_from_bmp(FILE*)加载位图数据
            ret = load_from_bmp(fd);
            // 关闭文件
            fclose(fd);
        }
        return ret;
    }
    
    
    //------------------------------------------------------------------------
    bool pixel_map::save_as_bmp(FILE *fd) const
    {
        // 如果像素图为空，则返回false
        if(m_bmp == 0) return 0;
    
        // 设置位图文件头
        BITMAPFILEHEADER bmf;
        bmf.bfType      = 0x4D42;  // 设置文件类型为BM
        bmf.bfOffBits   = calc_header_size(m_bmp) + sizeof(bmf);  // 设置文件偏移量
        bmf.bfSize      = bmf.bfOffBits + m_img_size;  // 设置文件大小
        bmf.bfReserved1 = 0;  // 保留字段1
        bmf.bfReserved2 = 0;  // 保留字段2
    
        // 写入位图文件头
        fwrite(&bmf, sizeof(bmf), 1, fd);
        // 写入像素数据
        fwrite(m_bmp, m_full_size, 1, fd);
        return true;
    }
    
    
    //------------------------------------------------------------------------
    bool pixel_map::save_as_bmp(const char *filename) const
    {
        // 打开指定文件
        FILE *fd = fopen(filename, "wb");
        bool ret = false;
        if(fd)
        {
            // 调用save_as_bmp(FILE*)保存位图数据
            ret = save_as_bmp(fd);
            // 关闭文件
            fclose(fd);
        }
        return ret;
    }
    
    
    //------------------------------------------------------------------------
    unsigned char* pixel_map::buf()
    {
        // 返回像素图缓冲区指针
        return m_buf;
    }
    
    
    //------------------------------------------------------------------------
    unsigned pixel_map::width() const
    {
        // 返回像素图像宽度
        return m_bmp->bmiHeader.biWidth;
    }
    
    
    //------------------------------------------------------------------------
    unsigned pixel_map::height() const
    {
        // 返回像素图像高度
        return m_bmp->bmiHeader.biHeight;
    }
    
    
    //------------------------------------------------------------------------
    int pixel_map::stride() const
    {
        // 返回行字节数
        return calc_row_len(m_bmp->bmiHeader.biWidth, 
                            m_bmp->bmiHeader.biBitCount);
    }
    {
        // 如果传入的 bmp 指针非空
        if(bmp)
        {
            // 计算图像每行的字节数，并乘以图像高度，得到图像总字节数
            m_img_size  = calc_row_len(bmp->bmiHeader.biWidth, 
                                       bmp->bmiHeader.biBitCount) * 
                          bmp->bmiHeader.biHeight;
    
            // 计算图像的完整大小
            m_full_size = calc_full_size(bmp);
            
            // 将传入的 bmp 指针赋值给成员变量 m_bmp
            m_bmp       = bmp;
            
            // 计算图像数据的指针并赋值给成员变量 m_buf
            m_buf       = calc_img_ptr(bmp);
        }
    }
    
    
    //private
    //------------------------------------------------------------------------
    // 根据指定参数在设备上下文中创建一个 DIB（设备无关位图）段
    HBITMAP pixel_map::create_dib_section_from_args(HDC h_dc,
                                                    unsigned width, 
                                                    unsigned height, 
                                                    unsigned bits_per_pixel)
    {
        // 计算图像每行的字节数
        unsigned line_len  = calc_row_len(width, bits_per_pixel);
        
        // 计算图像的总字节数
        unsigned img_size  = line_len * height;
        
        // 计算调色板的大小
        unsigned rgb_size  = calc_palette_size(0, bits_per_pixel) * sizeof(RGBQUAD);
        
        // 计算完整的 BITMAPINFO 结构的大小
        unsigned full_size = sizeof(BITMAPINFOHEADER) + rgb_size;
        
        // 分配内存以存储 BITMAPINFO 结构
        BITMAPINFO *bmp = (BITMAPINFO *) new unsigned char[full_size];
        
        // 设置 BITMAPINFOHEADER 的各个字段
        bmp->bmiHeader.biSize   = sizeof(BITMAPINFOHEADER);
        bmp->bmiHeader.biWidth  = width;
        bmp->bmiHeader.biHeight = height;
        bmp->bmiHeader.biPlanes = 1;
        bmp->bmiHeader.biBitCount = (unsigned short)bits_per_pixel;
        bmp->bmiHeader.biCompression = 0;
        bmp->bmiHeader.biSizeImage = img_size;
        bmp->bmiHeader.biXPelsPerMeter = 0;
        bmp->bmiHeader.biYPelsPerMeter = 0;
        bmp->bmiHeader.biClrUsed = 0;
        bmp->bmiHeader.biClrImportant = 0;
        
        void*   img_ptr  = 0;
        
        // 在指定设备上下文中创建 DIB 段，并返回 DIB 句柄
        HBITMAP h_bitmap = ::CreateDIBSection(h_dc, bmp, DIB_RGB_COLORS, &img_ptr, NULL, 0);
        
        // 如果成功创建 DIB 段，则进行以下操作
        if(img_ptr)
        {
            // 计算图像每行的字节数，并乘以图像高度，得到图像总字节数
            m_img_size  = calc_row_len(width, bits_per_pixel) * height;
            
            // 将图像完整大小置为 0（此处的 m_full_size 意图不明确）
            m_full_size = 0;
            
            // 将 BITMAPINFO 结构指针赋值给成员变量 m_bmp
            m_bmp       = bmp;
            
            // 将图像数据指针赋值给成员变量 m_buf
            m_buf       = (unsigned char *) img_ptr;
        }
        
        // 返回创建的 DIB 句柄
        return h_bitmap;
    }
}
```