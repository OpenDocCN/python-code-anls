# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\platform\X11\agg_platform_support.cpp`

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
// class platform_support. X11 version.
//
//----------------------------------------------------------------------------

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include "agg_basics.h"
#include "agg_pixfmt_gray.h"
#include "agg_pixfmt_rgb.h"
#include "agg_pixfmt_rgba.h"
#include "util/agg_color_conv_rgb8.h"
#include "platform/agg_platform_support.h"

// 命名空间 agg 中的定义

namespace agg
{
    //------------------------------------------------------------------------
    // 平台特定功能类的实现
    class platform_specific
    {
    public:
        // 构造函数，初始化平台特定功能对象
        platform_specific(pix_format_e format, bool flip_y);
        
        // 析构函数，清理平台特定功能对象
        ~platform_specific();
        
        // 设置窗口标题
        void caption(const char* capt);
        
        // 在窗口上放置图像
        void put_image(const rendering_buffer* src);
       
        pix_format_e         m_format;              // 像素格式
        pix_format_e         m_sys_format;          // 系统像素格式
        int                  m_byte_order;          // 字节顺序
        bool                 m_flip_y;              // Y轴翻转标志
        unsigned             m_bpp;                 // 每像素位数
        unsigned             m_sys_bpp;             // 系统每像素位数
        Display*             m_display;             // 显示器连接
        int                  m_screen;              // 屏幕号
        int                  m_depth;               // 色深
        Visual*              m_visual;              // 可视化
        Window               m_window;              // 窗口
        GC                   m_gc;                  // 绘图上下文
        XImage*              m_ximg_window;         // X窗口图像
        XSetWindowAttributes m_window_attributes;   // 窗口属性
        Atom                 m_close_atom;          // 关闭原子
        unsigned char*       m_buf_window;          // 窗口缓冲区
        unsigned char*       m_buf_img[platform_support::max_images];  // 图像缓冲区数组
        unsigned             m_keymap[256];         // 键码映射表
       
        bool m_update_flag;    // 更新标志
        bool m_resize_flag;    // 调整大小标志
        bool m_initialized;    // 是否初始化
        //bool m_wait_mode;    // 等待模式（已注释掉的变量）
        clock_t m_sw_start;    // 软件启动时钟
    };



    //------------------------------------------------------------------------
    `
        // 平台特定类构造函数，初始化成员变量
        platform_specific::platform_specific(pix_format_e format, bool flip_y) :
            m_format(format),                             // 初始化像素格式
            m_sys_format(pix_format_undefined),           // 初始化系统格式为未定义
            m_byte_order(LSBFirst),                        // 初始化字节序，设置为小端字节序
            m_flip_y(flip_y),                              // 初始化是否翻转Y轴
            m_bpp(0),                                       // 初始化每像素位数为0
            m_sys_bpp(0),                                    // 初始化系统每像素位数为0
            m_display(0),                                    // 初始化显示连接为0
            m_screen(0),                                      // 初始化屏幕编号为0
            m_depth(0),                                        // 初始化位深度为0
            m_visual(0),                                       // 初始化可视化样式为0
            m_window(0),                                       // 初始化窗口为0
            m_gc(0),                                              // 初始化GC为0
            m_ximg_window(0),                                 // 初始化X图像窗口为0
            m_close_atom(0),                                   // 初始化关闭原子为0
            m_buf_window(0),                                   // 初始化缓冲窗口为0
            m_update_flag(true),                              // 初始化更新标志为true
            m_resize_flag(true),                              // 初始化调整大小标志为true
            m_initialized(false)                              // 初始化为未初始化
            //m_wait_mode(true)  // 注释掉的等待模式
        }
    
        // 平台特定类析构函数
        platform_specific::~platform_specific()
        {
        }
    
        // 设置窗口标题和图标名称
        void platform_specific::caption(const char* capt)
        {
            XTextProperty tp;
            tp.value = (unsigned char *)capt;                   // 设置标题字符串的指针
            tp.encoding = XA_WM_NAME;                           // 设置编码为WM_NAME
            tp.format = 8;                                       // 设置格式为8位
            tp.nitems = strlen(capt);                           // 设置字符串长度
            XSetWMName(m_display, m_window, &tp);              // 设置窗口的WM_NAME属性
            XStoreName(m_display, m_window, capt);             // 存储窗口标题
            XSetIconName(m_display, m_window, capt);           // 设置窗口图标名称
            XSetWMIconName(m_display, m_window, &tp);          // 设置窗口的图标WM_NAME属性
        }
    
        // 设置图像函数（具体实现待定）
        void platform_specific::put_image(const rendering_buffer* src)
        {
        }
    
        // 平台支持类构造函数，初始化成员变量
        platform_support::platform_support(pix_format_e format, bool flip_y) :
            m_specific(new platform_specific(format, flip_y)), // 创建平台特定对象
            m_format(format),                                  // 初始化像素格式
            m_bpp(m_specific->m_bpp),                           // 初始化每像素位数
            m_window_flags(0),                                   // 初始化窗口标志为0
            m_wait_mode(true),                                  // 初始化等待模式为true
            m_flip_y(flip_y),                                     // 初始化翻转Y轴标志
            m_initial_width(10),                                 // 初始化初始宽度为10
            m_initial_height(10)                                 // 初始化初始高度为10
        {
            strcpy(m_caption, "AGG Application");              // 设置窗口标题为"AGG Application"
        }
    
        // 平台支持类析构函数，删除平台特定对象
        platform_support::~platform_support()
        {
            delete m_specific;
        }
    
        // 设置窗口标题
        void platform_support::caption(const char* cap)
        {
            strcpy(m_caption, cap);                            // 复制标题字符串
            if(m_specific->m_initialized)                     // 如果已初始化
            {
                m_specific->caption(cap);                     // 调用平台特定类的caption方法设置标题
            }
        }
    
        // 定义事件掩码枚举
        enum xevent_mask_e
        { 
            xevent_mask =
                PointerMotionMask|                          // 鼠标移动事件
                ButtonPressMask|                              // 鼠标按下事件
                ButtonReleaseMask|                           // 鼠标释放事件
                ExposureMask|                                   // 窗口曝光事件
                KeyPressMask|                                   // 键盘按键按下事件
                StructureNotifyMask                            // 窗口结构变更事件
        };
    
        // 初始化平台支持，设置窗口宽度、高度和标志
        bool platform_support::init(unsigned width, unsigned height, unsigned flags)
        {
        }
    {
        // 将窗口标志设置为给定的标志
        m_window_flags = flags;
        
        // 使用 XOpenDisplay 函数打开 X 显示，如果失败则输出错误信息并返回 false
        m_specific->m_display = XOpenDisplay(NULL);
        if (m_specific->m_display == 0) 
        {
            fprintf(stderr, "Unable to open DISPLAY!\n");
            return false;
        }
        
        // 获取默认屏幕、深度和可视化信息
        m_specific->m_screen = XDefaultScreen(m_specific->m_display);
        m_specific->m_depth  = XDefaultDepth(m_specific->m_display, 
                                             m_specific->m_screen);
        m_specific->m_visual = XDefaultVisual(m_specific->m_display, 
                                              m_specific->m_screen);
        
        // 提取可视化信息中的颜色掩码
        unsigned long r_mask = m_specific->m_visual->red_mask;
        unsigned long g_mask = m_specific->m_visual->green_mask;
        unsigned long b_mask = m_specific->m_visual->blue_mask;
    }
// 打印当前窗口的深度及颜色掩码信息，用于调试目的，但已被注释掉
//printf("depth=%d, red=%08x, green=%08x, blue=%08x\n",
//       m_specific->m_depth,
//       m_specific->m_visual->red_mask,
//       m_specific->m_visual->green_mask,
//       m_specific->m_visual->blue_mask);


//         // 还未完成！
//         // 尝试寻找一个适合的 Visual 如果默认的不符合条件。
//         if(m_specific->m_depth < 15 ||
//            r_mask == 0 || g_mask == 0 || b_mask == 0)
//         {
//             
//             // 当默认的 Visual 不符合最低要求时，尝试寻找一个适合的 Visual
//             static int depth[] = { 32, 24, 16, 15 };
//             int i;
//             for(int i = 0; i < 4; i++)
//             {
//                 XVisualInfo vi;
//                 if(XMatchVisualInfo(m_specific->m_display, 
//                                     m_specific->m_screen, 
//                                     depth[i], 
//                                     TrueColor, 
//                                     &vi)) 
//                 {
// //                     printf("TrueColor  depth=%d, red=%08x, green=%08x, blue=%08x, bits=%d\n",
// //                         vi.depth,
// //                         vi.visual->red_mask,
// //                         vi.visual->green_mask,
// //                         vi.visual->blue_mask,
// //                         vi.bits_per_rgb);
//                     m_specific->m_depth  = vi.depth;
//                     m_specific->m_visual = vi.visual;
//                     r_mask = m_specific->m_visual->red_mask;
//                     g_mask = m_specific->m_visual->green_mask;
//                     b_mask = m_specific->m_visual->blue_mask;
//                     break;
//                 }
//                 if(XMatchVisualInfo(m_specific->m_display, 
//                                     m_specific->m_screen, 
//                                     depth[i], 
//                                     DirectColor, 
//                                     &vi)) 
//                 {
// //                     printf("DirectColor depth=%d, red=%08x, green=%08x, blue=%08x, bits=%d\n",
// //                         vi.depth,
// //                         vi.visual->red_mask,
// //                         vi.visual->green_mask,
// //                         vi.visual->blue_mask,
// //                         vi.bits_per_rgb);
//                     m_specific->m_depth  = vi.depth;
//                     m_specific->m_visual = vi.visual;
//                     r_mask = m_specific->m_visual->red_mask;
//                     g_mask = m_specific->m_visual->green_mask;
//                     b_mask = m_specific->m_visual->blue_mask;
//                     break;
//                 }
//             }
//         }
    {
        // 将图像数据传递给特定平台的图像显示接口
        m_specific->put_image(&m_rbuf_window);
        
        // 当 m_wait_mode 为 true 时，绘制图像时可以丢弃所有事件，
        // 在此期间不累积鼠标移动事件。
        // 当 m_wait_mode 为 false 时，即存在空闲绘图时，不能错过任何事件。
        XSync(m_specific->m_display, m_wait_mode);
    }
    
    
    //------------------------------------------------------------------------
    int platform_support::run()
    {
        // 该函数可能是程序的主循环或执行入口点
    }
    
    
    
    //------------------------------------------------------------------------
    const char* platform_support::img_ext() const { return ".ppm"; }
    
    //------------------------------------------------------------------------
    const char* platform_support::full_file_name(const char* file_name)
    {
        // 返回给定文件名，这里没有对文件名进行任何更改或处理
        return file_name;
    }
    
    //------------------------------------------------------------------------
    bool platform_support::load_img(unsigned idx, const char* file)
    {
        // 加载指定索引的图像文件，这里可能涉及文件IO和图像数据处理
    }
    
    
    
    
    //------------------------------------------------------------------------
    bool platform_support::save_img(unsigned idx, const char* file)
    {
        // 保存指定索引的图像文件，具体实现可能涉及文件IO和图像数据处理
    }
    
    
    
    //------------------------------------------------------------------------
    bool platform_support::create_img(unsigned idx, unsigned width, unsigned height)
    {
        if(idx < max_images)
        {
            // 如果指定索引小于最大图像数，创建新的图像数据缓冲区
            if(width  == 0) width  = rbuf_window().width();
            if(height == 0) height = rbuf_window().height();
            delete [] m_specific->m_buf_img[idx];
            m_specific->m_buf_img[idx] = 
                new unsigned char[width * height * (m_bpp / 8)];
    
            // 将图像数据与指定属性附加到图像缓冲区对象
            m_rbuf_img[idx].attach(m_specific->m_buf_img[idx],
                                   width,
                                   height,
                                   m_flip_y ? 
                                       -width * (m_bpp / 8) : 
                                        width * (m_bpp / 8));
            return true;
        }
        // 如果索引超出最大图像数，返回 false
        return false;
    }
    
    
    //------------------------------------------------------------------------
    void platform_support::force_redraw()
    {
        // 设置更新标志以强制重新绘制图像
        m_specific->m_update_flag = true;
    }
    
    
    //------------------------------------------------------------------------
    void platform_support::message(const char* msg)
    {
        // 打印消息到标准错误输出
        fprintf(stderr, "%s\n", msg);
    }
    
    //------------------------------------------------------------------------
    void platform_support::start_timer()
    {
        // 记录当前时间，通常用于计时或性能测量
        m_specific->m_sw_start = clock();
    }
    
    //------------------------------------------------------------------------
    double platform_support::elapsed_time() const
    {
        // 计算自开始计时以来的经过时间（毫秒）
        clock_t stop = clock();
        return double(stop - m_specific->m_sw_start) * 1000.0 / CLOCKS_PER_SEC;
    }
    
    
    //------------------------------------------------------------------------
    void platform_support::on_init() {}
    # 定义空函数 platform_support::on_resize，接收 sx 和 sy 两个参数
    void platform_support::on_resize(int sx, int sy) {}
    
    # 定义空函数 platform_support::on_idle，无参数
    void platform_support::on_idle() {}
    
    # 定义空函数 platform_support::on_mouse_move，接收 x, y 和 flags 三个参数
    void platform_support::on_mouse_move(int x, int y, unsigned flags) {}
    
    # 定义空函数 platform_support::on_mouse_button_down，接收 x, y 和 flags 三个参数
    void platform_support::on_mouse_button_down(int x, int y, unsigned flags) {}
    
    # 定义空函数 platform_support::on_mouse_button_up，接收 x, y 和 flags 三个参数
    void platform_support::on_mouse_button_up(int x, int y, unsigned flags) {}
    
    # 定义空函数 platform_support::on_key，接收 x, y, key 和 flags 四个参数
    void platform_support::on_key(int x, int y, unsigned key, unsigned flags) {}
    
    # 定义空函数 platform_support::on_ctrl_change，无参数
    void platform_support::on_ctrl_change() {}
    
    # 定义空函数 platform_support::on_draw，无参数
    void platform_support::on_draw() {}
    
    # 定义空函数 platform_support::on_post_draw，接收一个 void* 类型的 raw_handler 参数
    void platform_support::on_post_draw(void* raw_handler) {}
}

# 定义一个返回整数的函数agg_main，接受两个参数：参数个数和参数列表
int agg_main(int argc, char* argv[]);

# 定义主函数main，接受两个参数：参数个数和参数列表
int main(int argc, char* argv[])
{
    # 调用agg_main函数并返回其结果
    return agg_main(argc, argv);
}
```