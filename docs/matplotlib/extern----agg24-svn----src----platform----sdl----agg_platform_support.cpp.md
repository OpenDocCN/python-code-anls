# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\platform\sdl\agg_platform_support.cpp`

```
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
// class platform_support. SDL version.
//
//----------------------------------------------------------------------------

#include <string.h>   // 引入标准库中的字符串操作函数定义
#include "platform/agg_platform_support.h"  // 引入平台相关的头文件
#include "SDL.h"      // 引入SDL库的主头文件
#include "SDL_byteorder.h"  // 引入SDL字节序头文件

namespace agg
{

    //------------------------------------------------------------------------
    class platform_specific
    {
    public:
        // 构造函数，初始化平台特定的对象
        platform_specific(pix_format_e format, bool flip_y);
        // 析构函数，释放资源
        ~platform_specific();

        pix_format_e  m_format;      // 像素格式
        pix_format_e  m_sys_format;  // 系统像素格式
        bool          m_flip_y;      // 是否翻转Y轴
        unsigned      m_bpp;         // 每像素位数
        unsigned      m_sys_bpp;     // 系统每像素位数
        unsigned      m_rmask;       // 红色掩码
        unsigned      m_gmask;       // 绿色掩码
        unsigned      m_bmask;       // 蓝色掩码
        unsigned      m_amask;       // Alpha掩码
        bool          m_update_flag; // 更新标志
        bool          m_resize_flag; // 调整大小标志
        bool          m_initialized; // 是否已初始化
        SDL_Surface*  m_surf_screen; // 屏幕表面
        SDL_Surface*  m_surf_window; // 窗口表面
        SDL_Surface*  m_surf_img[platform_support::max_images];  // 图像表面数组
        int           m_cur_x;       // 当前X坐标
        int           m_cur_y;       // 当前Y坐标
        int           m_sw_start;    // 软件起始位置
    };



    //------------------------------------------------------------------------
    // platform_specific类的构造函数定义
    platform_specific::platform_specific(pix_format_e format, bool flip_y) :
        m_format(format),
        m_sys_format(pix_format_undefined),
        m_flip_y(flip_y),
        m_bpp(0),
        m_sys_bpp(0),
        m_update_flag(true), 
        m_resize_flag(true),
        m_initialized(false),
        m_surf_screen(0),
        m_surf_window(0),
        m_cur_x(0),
        m_cur_y(0),
        m_sw_start(0)
    {
        // 将图像表面数组清零
        memset(m_surf_img, 0, sizeof(m_surf_img));

        // 根据像素格式选择相应的设置
        switch(m_format)
        {
            // 灰度8位像素格式
            case pix_format_gray8:
                m_bpp = 8;
                break;

            // RGB565像素格式
            case pix_format_rgb565:
                m_rmask = 0xF800;
                m_gmask = 0x7E0;
                m_bmask = 0x1F;
                m_amask = 0;
                m_bpp = 16;
                break;

            // RGB555像素格式
            case pix_format_rgb555:
                m_rmask = 0x7C00;
                m_gmask = 0x3E0;
                m_bmask = 0x1F;
                m_amask = 0;
                m_bpp = 16;
                break;
        }
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    // 如果系统是小端序（字节序），则执行以下代码块
    switch (pix_format)
    {
        case pix_format_rgb24:
            // RGB24格式，设置各颜色通道的掩码和位深度
            m_rmask = 0xFF;
            m_gmask = 0xFF00;
            m_bmask = 0xFF0000;
            m_amask = 0;
            m_bpp = 24;
            break;

        case pix_format_bgr24:
            // BGR24格式，设置各颜色通道的掩码和位深度
            m_rmask = 0xFF0000;
            m_gmask = 0xFF00;
            m_bmask = 0xFF;
            m_amask = 0;
            m_bpp = 24;
            break;

        case pix_format_bgra32:
            // BGRA32格式，设置各颜色通道的掩码和位深度
            m_rmask = 0xFF0000;
            m_gmask = 0xFF00;
            m_bmask = 0xFF;
            m_amask = 0xFF000000;
            m_bpp = 32;
            break;

        case pix_format_abgr32:
            // ABGR32格式，设置各颜色通道的掩码和位深度
            m_rmask = 0xFF000000;
            m_gmask = 0xFF0000;
            m_bmask = 0xFF00;
            m_amask = 0xFF;
            m_bpp = 32;
            break;

        case pix_format_argb32:
            // ARGB32格式，设置各颜色通道的掩码和位深度
            m_rmask = 0xFF00;
            m_gmask = 0xFF0000;
            m_bmask = 0xFF000000;
            m_amask = 0xFF;
            m_bpp = 32;
            break;

        case pix_format_rgba32:
            // RGBA32格式，设置各颜色通道的掩码和位深度
            m_rmask = 0xFF;
            m_gmask = 0xFF00;
            m_bmask = 0xFF0000;
            m_amask = 0xFF000000;
            m_bpp = 32;
            break;
#else //SDL_BIG_ENDIAN (PPC)
    // 如果系统是大端序（例如PowerPC架构），则执行以下代码块
    switch (pix_format)
    {
        case pix_format_rgb24:
            // RGB24格式，设置各颜色通道的掩码和位深度
            m_rmask = 0xFF0000;
            m_gmask = 0xFF00;
            m_bmask = 0xFF;
            m_amask = 0;
            m_bpp = 24;
            break;

        case pix_format_bgr24:
            // BGR24格式，设置各颜色通道的掩码和位深度
            m_rmask = 0xFF;
            m_gmask = 0xFF00;
            m_bmask = 0xFF0000;
            m_amask = 0;
            m_bpp = 24;
            break;

        case pix_format_bgra32:
            // BGRA32格式，设置各颜色通道的掩码和位深度
            m_rmask = 0xFF00;
            m_gmask = 0xFF0000;
            m_bmask = 0xFF000000;
            m_amask = 0xFF;
            m_bpp = 32;
            break;

        case pix_format_abgr32:
            // ABGR32格式，设置各颜色通道的掩码和位深度
            m_rmask = 0xFF;
            m_gmask = 0xFF00;
            m_bmask = 0xFF0000;
            m_amask = 0xFF000000;
            m_bpp = 32;
            break;

        case pix_format_argb32:
            // ARGB32格式，设置各颜色通道的掩码和位深度
            m_rmask = 0xFF0000;
            m_gmask = 0xFF00;
            m_bmask = 0xFF;
            m_amask = 0xFF000000;
            m_bpp = 32;
            break;

        case pix_format_rgba32:
            // RGBA32格式，设置各颜色通道的掩码和位深度
            m_rmask = 0xFF000000;
            m_gmask = 0xFF0000;
            m_bmask = 0xFF00;
            m_amask = 0xFF;
            m_bpp = 32;
            break;
#endif
    }
}

//------------------------------------------------------------------------
// 平台特定析构函数，用于释放SDL表面资源
platform_specific::~platform_specific()
{
    int i;
    // 释放图像资源数组中的所有SDL表面
    for(i = platform_support::max_images - 1; i >= 0; --i)
    {
        if(m_surf_img[i]) SDL_FreeSurface(m_surf_img[i]);
    }
    // 释放窗口SDL表面
    if(m_surf_window) SDL_FreeSurface(m_surf_window);
    // 释放屏幕SDL表面
    if(m_surf_screen) SDL_FreeSurface(m_surf_screen);
}
//------------------------------------------------------------------------
    // 构造函数：根据指定的像素格式和是否垂直翻转初始化平台支持对象
    platform_support::platform_support(pix_format_e format, bool flip_y) :
        // 使用给定的像素格式和翻转设置创建特定平台的对象
        m_specific(new platform_specific(format, flip_y)),
        // 记录像素格式
        m_format(format),
        // 记录每像素位数
        m_bpp(m_specific->m_bpp),
        // 窗口标志位初始化
        m_window_flags(0),
        // 等待模式设为真
        m_wait_mode(true),
        // 记录是否垂直翻转
        m_flip_y(flip_y)
    {
        // 初始化SDL视频子系统
        SDL_Init(SDL_INIT_VIDEO);
        // 设置窗口标题
        strcpy(m_caption, "Anti-Grain Geometry Application");
    }


```    
    //------------------------------------------------------------------------
    // 析构函数：释放特定平台对象的内存
    platform_support::~platform_support()
    {
        delete m_specific;
    }



    //------------------------------------------------------------------------
    // 设置窗口标题
    void platform_support::caption(const char* cap)
    {
        // 复制新的标题到成员变量中
        strcpy(m_caption, cap);
        // 如果特定平台已经初始化，则设置SDL窗口标题
        if(m_specific->m_initialized)
        {
            SDL_WM_SetCaption(cap, 0);
        }
    }


```    
    //------------------------------------------------------------------------
    // 初始化平台支持
    bool platform_support::init(unsigned width, unsigned height, unsigned flags)
    {
        // 将传入的窗口标志设置为成员变量
        m_window_flags = flags;
        // 默认窗口标志为 SDL_SWSURFACE
        unsigned wflags = SDL_SWSURFACE;
    
        // 如果窗口标志包含 window_hw_buffer，则使用硬件表面标志 SDL_HWSURFACE
        if (m_window_flags & window_hw_buffer)
        {
            wflags = SDL_HWSURFACE;
        }
    
        // 如果窗口标志包含 window_resize，则添加可调整大小标志 SDL_RESIZABLE
        if (m_window_flags & window_resize)
        {
            wflags |= SDL_RESIZABLE;
        }
    
        // 如果之前已经存在屏幕表面，则释放其内存
        if (m_specific->m_surf_screen) SDL_FreeSurface(m_specific->m_surf_screen);
    
        // 设置视频模式，返回的表面存储在 m_specific->m_surf_screen
        m_specific->m_surf_screen = SDL_SetVideoMode(width, height, m_bpp, wflags);
    
        // 如果设置视频模式失败，则打印错误信息并返回 false
        if (m_specific->m_surf_screen == 0) 
        {
            fprintf(stderr, 
                    "Unable to set %dx%d %d bpp video: %s\n", 
                    width, 
                    height, 
                    m_bpp, 
                    ::SDL_GetError());
            return false;
        }
    
        // 设置窗口标题
        SDL_WM_SetCaption(m_caption, 0);
    
        // 如果之前已经存在窗口表面，则释放其内存
        if (m_specific->m_surf_window) SDL_FreeSurface(m_specific->m_surf_window);
    
        // 创建与屏幕表面同样大小和位深的 RGB 表面，存储在 m_specific->m_surf_window
        m_specific->m_surf_window = 
            SDL_CreateRGBSurface(SDL_HWSURFACE, 
                                 m_specific->m_surf_screen->w, 
                                 m_specific->m_surf_screen->h,
                                 m_specific->m_surf_screen->format->BitsPerPixel,
                                 m_specific->m_rmask, 
                                 m_specific->m_gmask, 
                                 m_specific->m_bmask, 
                                 m_specific->m_amask);
    
        // 如果创建窗口表面失败，则打印错误信息并返回 false
        if (m_specific->m_surf_window == 0) 
        {
            fprintf(stderr, 
                    "Unable to create image buffer %dx%d %d bpp: %s\n", 
                    width, 
                    height, 
                    m_bpp, 
                    SDL_GetError());
            return false;
        }
    
        // 将窗口表面的像素数据附加到图形缓冲区对象中
        m_rbuf_window.attach((unsigned char*)m_specific->m_surf_window->pixels, 
                             m_specific->m_surf_window->w, 
                             m_specific->m_surf_window->h, 
                             m_flip_y ? -m_specific->m_surf_window->pitch : 
                                         m_specific->m_surf_window->pitch);
    
        // 如果平台支持未初始化，则设置初始宽度和高度，并调用初始化函数
        if (!m_specific->m_initialized)
        {
            m_initial_width = width;
            m_initial_height = height;
            on_init();
            m_specific->m_initialized = true;
        }
    
        // 调整窗口大小，通知相关函数
        on_resize(m_rbuf_window.width(), m_rbuf_window.height());
    
        // 设置更新标志为 true，表示窗口已更新
        m_specific->m_update_flag = true;
    
        // 返回 true 表示成功设置和更新窗口
        return true;
    }
    
    
    
    //------------------------------------------------------------------------
    void platform_support::update_window()
    {
        // 将窗口表面 m_specific->m_surf_window 的内容复制到屏幕表面 m_specific->m_surf_screen
        SDL_BlitSurface(m_specific->m_surf_window, 0, m_specific->m_surf_screen, 0);
        
        // 更新整个屏幕表面，使之与最新的数据匹配
        SDL_UpdateRect(m_specific->m_surf_screen, 0, 0, 0, 0);
    }
// 如果鼠标按下事件在 m_ctrls 上
if(m_ctrls.on_mouse_button_down(m_specific->m_cur_x,
                                m_specific->m_cur_y))
{
    // 设置当前控件位置为 m_specific 的当前坐标
    m_ctrls.set_cur(m_specific->m_cur_x, 
                    m_specific->m_cur_y);
    // 触发控件改变事件
    on_ctrl_change();
    // 强制重新绘制
    force_redraw();
}
else
{
    // 如果鼠标不在 m_ctrls 区域内
    if(m_ctrls.in_rect(m_specific->m_cur_x, 
                       m_specific->m_cur_y))
    {
        // 设置当前控件位置为 m_specific 的当前坐标
        if(m_ctrls.set_cur(m_specific->m_cur_x, 
                           m_specific->m_cur_y))
        {
            // 触发控件改变事件
            on_ctrl_change();
            // 强制重新绘制
            force_redraw();
        }
    }
    else
    {
        // 鼠标按键按下事件在当前 m_specific 坐标位置
        on_mouse_button_down(m_specific->m_cur_x, 
                             m_specific->m_cur_y, 
                             flags);
    }
}
    // 加载图片到指定索引的表面
    bool platform_support::load_img(unsigned idx, const char* file)
    {
        // 检查索引是否有效
        if(idx < max_images)
        {
            // 如果索引处的表面已经存在，释放它
            if(m_specific->m_surf_img[idx]) SDL_FreeSurface(m_specific->m_surf_img[idx]);

            // 复制文件名到临时变量 fn，并检查文件名是否以 ".bmp" 结尾，若不是则添加 ".bmp"
            char fn[1024];
            strcpy(fn, file);
            int len = strlen(fn);
            if(len < 4 || strcmp(fn + len - 4, ".bmp") != 0)
            {
                strcat(fn, ".bmp");
            }

            // 加载 BMP 图像到临时表面 tmp_surf
            SDL_Surface* tmp_surf = SDL_LoadBMP(fn);
            // 如果加载失败，输出错误信息并返回失败状态
            if (tmp_surf == 0) 
            {
                fprintf(stderr, "Couldn't load %s: %s\n", fn, SDL_GetError());
                return false;
            }

            // 设置目标表面的像素格式 format
            SDL_PixelFormat format;
            format.palette = 0;
            format.BitsPerPixel = m_bpp;
            format.BytesPerPixel = m_bpp >> 8;
            format.Rmask = m_specific->m_rmask;
            format.Gmask = m_specific->m_gmask;
            format.Bmask = m_specific->m_bmask;
            format.Amask = m_specific->m_amask;
            format.Rshift = 0;
            format.Gshift = 0;
            format.Bshift = 0;
            format.Ashift = 0;
            format.Rloss = 0;
            format.Gloss = 0;
            format.Bloss = 0;
            format.Aloss = 0;
            format.colorkey = 0;
            format.alpha = 0;

            // 将临时表面转换为指定格式的表面并赋值给 m_specific->m_surf_img[idx]
            m_specific->m_surf_img[idx] = 
                SDL_ConvertSurface(tmp_surf, 
                                   &format, 
                                   SDL_SWSURFACE);

            // 释放临时表面 tmp_surf
            SDL_FreeSurface(tmp_surf);
            
            // 如果转换后的表面为空，返回加载失败
            if(m_specific->m_surf_img[idx] == 0) return false;

            // 将加载后的表面数据附加到 m_rbuf_img[idx]，处理可能的 Y 轴翻转
            m_rbuf_img[idx].attach((unsigned char*)m_specific->m_surf_img[idx]->pixels, 
                                   m_specific->m_surf_img[idx]->w, 
                                   m_specific->m_surf_img[idx]->h, 
                                   m_flip_y ? -m_specific->m_surf_img[idx]->pitch : 
                                               m_specific->m_surf_img[idx]->pitch);
            return true; // 返回加载成功
        }
        return false; // 索引无效，返回加载失败
    }




    //------------------------------------------------------------------------
    // 将指定索引处的图像保存为 BMP 文件
    bool platform_support::save_img(unsigned idx, const char* file)
    {
        // 检查索引是否有效且对应的表面存在
        if(idx < max_images && m_specific->m_surf_img[idx])
        {
            // 复制文件名到临时变量 fn，并检查文件名是否以 ".bmp" 结尾，若不是则添加 ".bmp"
            char fn[1024];
            strcpy(fn, file);
            int len = strlen(fn);
            if(len < 4 || strcmp(fn + len - 4, ".bmp") != 0)
            {
                strcat(fn, ".bmp");
            }
            // 调用 SDL_SaveBMP 将指定表面保存为 BMP 文件，并返回保存成功与否的状态
            return SDL_SaveBMP(m_specific->m_surf_img[idx], fn) == 0;
        }
        return false; // 索引无效或对应表面不存在，返回保存失败
    }



    //------------------------------------------------------------------------
    // 创建指定索引处指定大小的空白图像（未实现详细说明）
    bool platform_support::create_img(unsigned idx, unsigned width, unsigned height)
    {
        // 如果索引小于最大图片数量
        if(idx < max_images)
        {
            // 如果指定索引已经有表面图像，释放它
            if(m_specific->m_surf_img[idx]) SDL_FreeSurface(m_specific->m_surf_img[idx]);
    
            // 创建一个新的 RGB 表面图像
            m_specific->m_surf_img[idx] = 
                SDL_CreateRGBSurface(SDL_SWSURFACE, 
                                     width, 
                                     height,
                                     m_specific->m_surf_screen->format->BitsPerPixel,
                                     m_specific->m_rmask, 
                                     m_specific->m_gmask, 
                                     m_specific->m_bmask, 
                                     m_specific->m_amask);
    
            // 如果创建表面图像失败，打印错误信息并返回 false
            if(m_specific->m_surf_img[idx] == 0) 
            {
                fprintf(stderr, "Couldn't create image: %s\n", SDL_GetError());
                return false;
            }
    
            // 将图像数据与图像缓冲区绑定，处理垂直翻转时调整像素行偏移
            m_rbuf_img[idx].attach((unsigned char*)m_specific->m_surf_img[idx]->pixels, 
                                   m_specific->m_surf_img[idx]->w, 
                                   m_specific->m_surf_img[idx]->h, 
                                   m_flip_y ? -m_specific->m_surf_img[idx]->pitch : 
                                               m_specific->m_surf_img[idx]->pitch);
    
            // 创建成功，返回 true
            return true;
        }
    
        // 如果索引超出范围，返回 false
        return false;
    }
    
    //------------------------------------------------------------------------
    // 启动计时器，记录开始时间
    void platform_support::start_timer()
    {
        m_specific->m_sw_start = SDL_GetTicks();
    }
    
    //------------------------------------------------------------------------
    // 获取经过的时间，以秒为单位
    double platform_support::elapsed_time() const
    {
        int stop = SDL_GetTicks();
        return double(stop - m_specific->m_sw_start);
    }
    
    //------------------------------------------------------------------------
    // 输出消息到标准错误流
    void platform_support::message(const char* msg)
    {
        fprintf(stderr, "%s\n", msg);
    }
    
    //------------------------------------------------------------------------
    // 强制重绘标志设置为 true
    void platform_support::force_redraw()
    {
        m_specific->m_update_flag = true;
    }
    
    //------------------------------------------------------------------------
    // 下面的函数是空函数，用于占位，没有实际功能
    void platform_support::on_init() {}
    void platform_support::on_resize(int sx, int sy) {}
    void platform_support::on_idle() {}
    void platform_support::on_mouse_move(int x, int y, unsigned flags) {}
    void platform_support::on_mouse_button_down(int x, int y, unsigned flags) {}
    void platform_support::on_mouse_button_up(int x, int y, unsigned flags) {}
    void platform_support::on_key(int x, int y, unsigned key, unsigned flags) {}
    void platform_support::on_ctrl_change() {}
    void platform_support::on_draw() {}
    void platform_support::on_post_draw(void* raw_handler) {}
}



// 结束 main 函数体的大括号，标志着 main 函数的结束

int agg_main(int argc, char* argv[]);

int main(int argc, char* argv[])
{
    // 调用 agg_main 函数，并将其返回值作为 main 函数的返回值
    return agg_main(argc, argv);
}


这段代码是一个简单的 C/C++ 程序，其中 `main` 函数调用了另一个函数 `agg_main`，并将其返回值作为程序的返回值。
```