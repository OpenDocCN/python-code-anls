# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\platform\mac\agg_platform_support.cpp`

```py
//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4 
// Copyright (C) 2002-2005 Maxim Shemanarev (McSeem)
// Copyright (C) 2003 Hansruedi Baer (MacOS support)
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
// class platform_support
//
//----------------------------------------------------------------------------
//
// Note:
// I tried to retain the original structure for the Win32 platform as far
// as possible. Currently, not all features are implemented but the examples
// should work properly.
// HB
//----------------------------------------------------------------------------

#include <Carbon.h>
#if defined(__MWERKS__)
#include "console.h"
#endif
#include <string.h>
#include <unistd.h>
#include "platform/agg_platform_support.h"
#include "platform/mac/agg_mac_pmap.h"
#include "util/agg_color_conv_rgb8.h"


namespace agg
{
    
pascal OSStatus DoWindowClose (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoWindowDrawContent (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoAppQuit (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoMouseDown (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoMouseUp (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoMouseDragged (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoKeyDown (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoKeyUp (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal void DoPeriodicTask (EventLoopTimerRef theTimer, void* userData);


    //------------------------------------------------------------------------
    // 平台特定的类，用于支持不同操作系统的特定功能
    class platform_specific
    {
    // 构造函数，初始化 platform_specific 对象的各个成员变量
    platform_specific(pix_format_e format, bool flip_y);

    //------------------------------------------------------------------------
    void create_pmap(unsigned width, unsigned height, 
                     rendering_buffer* wnd)
    {
        // 使用指定的宽度、高度和位深度创建像素映射
        m_pmap_window.create(width, height, org_e(m_bpp));
        // 将渲染缓冲区与像素映射绑定，设置映射的宽度、高度和行字节数
        wnd->attach(m_pmap_window.buf(), 
                    m_pmap_window.width(),
                    m_pmap_window.height(),
                    m_flip_y ?
                    -m_pmap_window.row_bytes() :
                    m_pmap_window.row_bytes());
    }

    //------------------------------------------------------------------------
    void display_pmap(WindowRef window, const rendering_buffer* src)
    {
        // 检查当前系统格式是否与对象的格式相匹配
        if(m_sys_format == m_format)
        {
            // 如果相匹配，使用当前窗口对象绘制像素地图
            m_pmap_window.draw(window);
        }
        else
        {
            // 否则，创建一个临时的像素地图对象，匹配系统的位深度
            pixel_map pmap_tmp;
            pmap_tmp.create(m_pmap_window.width(), 
                            m_pmap_window.height(),
                            org_e(m_sys_bpp));

            // 创建渲染缓冲区对象，将其附加到临时像素地图对象上
            rendering_buffer rbuf_tmp;
            rbuf_tmp.attach(pmap_tmp.buf(),
                            pmap_tmp.width(),
                            pmap_tmp.height(),
                            m_flip_y ?
                             -pmap_tmp.row_bytes() :
                              pmap_tmp.row_bytes());

            // 根据当前对象的格式进行转换
            switch(m_format)
            {
            case pix_format_gray8:
                // 对于灰度图像，直接返回，不做处理
                return;

            case pix_format_rgb565:
                // 将输入缓冲区中的 RGB565 格式转换为 RGB555 格式
                color_conv(&rbuf_tmp, src, color_conv_rgb565_to_rgb555());
                break;

            case pix_format_bgr24:
                // 将输入缓冲区中的 BGR24 格式转换为 RGB24 格式
                color_conv(&rbuf_tmp, src, color_conv_bgr24_to_rgb24());
                break;

            case pix_format_abgr32:
                // 将输入缓冲区中的 ABGR32 格式转换为 ARGB32 格式
                color_conv(&rbuf_tmp, src, color_conv_abgr32_to_argb32());
                break;

            case pix_format_bgra32:
                // 将输入缓冲区中的 BGRA32 格式转换为 ARGB32 格式
                color_conv(&rbuf_tmp, src, color_conv_bgra32_to_argb32());
                break;

            case pix_format_rgba32:
                // 将输入缓冲区中的 RGBA32 格式转换为 ARGB32 格式
                color_conv(&rbuf_tmp, src, color_conv_rgba32_to_argb32());
                break;
            }

            // 使用临时像素地图对象绘制窗口
            pmap_tmp.draw(window);
        }
    }


    //------------------------------------------------------------------------
    bool platform_specific::save_pmap(const char* fn, unsigned idx, 
                                      const rendering_buffer* src)
    {
        // 如果系统格式与当前格式匹配，则保存图像为 Qt 图像并返回结果
        if(m_sys_format == m_format)
        {
            return m_pmap_img[idx].save_as_qt(fn);
        }
        else
        {
            // 否则，创建临时像素映射对象
            pixel_map pmap_tmp;
            // 使用系统位深度创建临时像素映射
            pmap_tmp.create(m_pmap_img[idx].width(), 
                            m_pmap_img[idx].height(),
                            org_e(m_sys_bpp));
    
            // 创建渲染缓冲对象并附加到临时像素映射
            rendering_buffer rbuf_tmp;
            rbuf_tmp.attach(pmap_tmp.buf(),
                            pmap_tmp.width(),
                            pmap_tmp.height(),
                            m_flip_y ?
                             -pmap_tmp.row_bytes() :
                              pmap_tmp.row_bytes());
    
            // 根据当前格式进行颜色转换
            switch(m_format)
            {
            case pix_format_gray8:
                // 灰度图像不支持保存为 Qt 图像，返回 false
                return false;
    
            case pix_format_rgb565:
                // RGB565 格式转换为 RGB555 格式
                color_conv(&rbuf_tmp, src, color_conv_rgb565_to_rgb555());
                break;
    
            case pix_format_rgb24:
                // RGB24 格式转换为 BGR24 格式
                color_conv(&rbuf_tmp, src, color_conv_rgb24_to_bgr24());
                break;
    
            case pix_format_abgr32:
                // ABGR32 格式转换为 BGRA32 格式
                color_conv(&rbuf_tmp, src, color_conv_abgr32_to_bgra32());
                break;
    
            case pix_format_argb32:
                // ARGB32 格式转换为 BGRA32 格式
                color_conv(&rbuf_tmp, src, color_conv_argb32_to_bgra32());
                break;
    
            case pix_format_rgba32:
                // RGBA32 格式转换为 BGRA32 格式
                color_conv(&rbuf_tmp, src, color_conv_rgba32_to_bgra32());
                break;
            }
    
            // 将临时像素映射保存为 Qt 图像并返回结果
            return pmap_tmp.save_as_qt(fn);
        }
        // 默认情况下返回 true
        return true;
    }
    
    
    
    //------------------------------------------------------------------------
    bool platform_specific::load_pmap(const char* fn, unsigned idx, 
                                      rendering_buffer* dst)
    {
        // 省略的方法体，无需注释内容
    }
    
    
    
    
    //------------------------------------------------------------------------
    unsigned platform_specific::translate(unsigned keycode)
    {
        // 返回键码对应的映射值
        return m_last_translated_key = (keycode > 255) ? 0 : m_keymap[keycode];
    }
    
    
    
    //------------------------------------------------------------------------
    platform_support::platform_support(pix_format_e format, bool flip_y) :
        m_specific(new platform_specific(format, flip_y)),
        m_format(format),
        m_bpp(m_specific->m_bpp),
        m_window_flags(0),
        m_wait_mode(true),
        m_flip_y(flip_y),
        m_initial_width(10),
        m_initial_height(10)
    {
        // 初始化窗口标题
        strcpy(m_caption, "Anti-Grain Geometry Application");
    }
    
    
    
    //------------------------------------------------------------------------
    platform_support::~platform_support()
    {
        // 释放平台特定对象的内存
        delete m_specific;
    }
    
    
    
    //------------------------------------------------------------------------
    void platform_support::caption(const char* cap)
    {
        // 设置应用程序标题
        strcpy(m_caption, cap);
        // 如果窗口存在，则设置窗口标题
        if(m_specific->m_window)
        {
            SetWindowTitleWithCFString(m_specific->m_window, CFStringCreateWithCStringNoCopy(nil, cap, kCFStringEncodingASCII, nil));
        }
    }
    //------------------------------------------------------------------------
    static unsigned get_key_flags(UInt32 wflags)
    {
        unsigned flags = 0;
        
        // 检查 wflags 是否包含 shiftKey 标志位，若包含则设置键盘 shift 标志位
        if (wflags & shiftKey)
            flags |= kbd_shift;
        
        // 检查 wflags 是否包含 controlKey 标志位，若包含则设置键盘 ctrl 标志位
        if (wflags & controlKey)
            flags |= kbd_ctrl;
    
        return flags;
    }
    
    
    //------------------------------------------------------------------------
    void platform_support::message(const char* msg)
    {
        SInt16 item;
        Str255 p_msg;
        
        // 将 C 字符串 msg 复制为 Pascal 字符串 p_msg
        ::CopyCStringToPascal(msg, p_msg);
        
        // 显示标准警告框，标题为 "\013AGG Message"，消息为 p_msg，按钮选项存入 item
        ::StandardAlert(kAlertPlainAlert, (const unsigned char*) "\013AGG Message", p_msg, NULL, &item);
        //::StandardAlert (kAlertPlainAlert, (const unsigned char*) "\pAGG Message", p_msg, NULL, &item);
    }
    
    
    //------------------------------------------------------------------------
    void platform_support::start_timer()
    {
        // 获取当前时间并存入 m_specific 的 m_sw_start
        ::Microseconds(&(m_specific->m_sw_start));
    }
    
    
    //------------------------------------------------------------------------
    double platform_support::elapsed_time() const
    {
        UnsignedWide stop;
        
        // 获取当前时间并存入 stop
        ::Microseconds(&stop);
        
        // 计算已经过的时间（微秒）并返回
        return double(stop.lo - m_specific->m_sw_start.lo) * 1e6 / double(m_specific->m_sw_freq.lo);
    }
    
    
    //------------------------------------------------------------------------
    bool platform_support::init(unsigned width, unsigned height, unsigned flags)
    {
        // 初始化操作，这里未提供具体代码
    }
    
    
    //------------------------------------------------------------------------
    int platform_support::run()
    {
        // 启动应用程序事件循环
        RunApplicationEventLoop();
        return true;
    }
    
    
    //------------------------------------------------------------------------
    const char* platform_support::img_ext() const
    {
        // 返回图片文件的扩展名 ".bmp"
        return ".bmp";
    }
    
    
    //------------------------------------------------------------------------
    const char* platform_support::full_file_name(const char* file_name)
    {
        // 返回原始文件名 file_name
        return file_name;
    }
    
    
    //------------------------------------------------------------------------
    bool platform_support::load_img(unsigned idx, const char* file)
    {
        if (idx < max_images)
        {
            char fn[1024];
            
            // 复制文件名 file 到缓冲区 fn
            strcpy(fn, file);
            
            // 计算文件名长度
            int len = strlen(fn);
#if defined(__MWERKS__)
            // 如果编译器是 Metrowerks CodeWarrior
            // 检查文件名是否以 ".BMP" 结尾，不区分大小写
            if(len < 4 || stricmp(fn + len - 4, ".BMP") != 0)
#else
            // 对于其他编译器
            // 检查文件名是否以 ".BMP" 结尾，前四个字符不区分大小写
            if(len < 4 || strncasecmp(fn + len - 4, ".BMP", 4) != 0)
#endif
            {
                // 如果文件名不以 ".BMP" 结尾，则添加 ".bmp" 后缀
                strcat(fn, ".bmp");
            }
            // 调用特定平台的加载图像函数，加载处理后的文件名和索引对应的图像数据到缓冲区
            return m_specific->load_pmap(fn, idx, &m_rbuf_img[idx]);
        }
        // 如果索引超出最大图像数，返回 true 表示加载失败
        return true;
    }



    //------------------------------------------------------------------------
    bool platform_support::save_img(unsigned idx, const char* file)
    {
        if(idx < max_images)
        {
            // 复制文件名到本地缓冲区 fn
            char fn[1024];
            strcpy(fn, file);
            // 获取文件名长度
            int len = strlen(fn);
#if defined(__MWERKS__)
            // 如果编译器是 Metrowerks CodeWarrior
            // 检查文件名是否以 ".BMP" 结尾，不区分大小写
            if(len < 4 || stricmp(fn + len - 4, ".BMP") != 0)
#else
            // 对于其他编译器
            // 检查文件名是否以 ".BMP" 结尾，前四个字符不区分大小写
            if(len < 4 || strncasecmp(fn + len - 4, ".BMP", 4) != 0)
#endif
            {
                // 如果文件名不以 ".BMP" 结尾，则添加 ".bmp" 后缀
                strcat(fn, ".bmp");
            }
            // 调用特定平台的保存图像函数，保存索引对应的图像数据到处理后的文件名
            return m_specific->save_pmap(fn, idx, &m_rbuf_img[idx]);
        }
        // 如果索引超出最大图像数，返回 true 表示保存失败
        return true;
    }



    //------------------------------------------------------------------------
    bool platform_support::create_img(unsigned idx, unsigned width, unsigned height)
    {
        if(idx < max_images)
        {
            // 如果宽度或高度为零，则使用默认窗口宽度和高度
            if(width  == 0) width  = m_specific->m_pmap_window.width();
            if(height == 0) height = m_specific->m_pmap_window.height();
            // 创建图像数据并关联到图像缓冲区
            m_specific->m_pmap_img[idx].create(width, height, org_e(m_specific->m_bpp));
            m_rbuf_img[idx].attach(m_specific->m_pmap_img[idx].buf(), 
                                   m_specific->m_pmap_img[idx].width(),
                                   m_specific->m_pmap_img[idx].height(),
                                   // 根据翻转标志设置行字节数，影响图像的垂直翻转
                                   m_flip_y ?
                                   -m_specific->m_pmap_img[idx].row_bytes() :
                                    m_specific->m_pmap_img[idx].row_bytes());
            return true;
        }
        // 如果索引超出最大图像数，返回 false 表示创建失败
        return false;
    }


    //------------------------------------------------------------------------
    void platform_support::force_redraw()
    {
        Rect    bounds;
        
        // 设置重绘标志并调用绘制函数
        m_specific->m_redraw_flag = true;
        // on_ctrl_change ();
        on_draw();

        // 设置窗口边界并通知窗口需要重新绘制
        SetRect(&bounds, 0, 0, m_rbuf_window.width(), m_rbuf_window.height());
        InvalWindowRect(m_specific->m_window, &bounds);
    }



    //------------------------------------------------------------------------
    void platform_support::update_window()
    {
        // 显示图像缓冲区内容到窗口
        m_specific->display_pmap(m_specific->m_window, &m_rbuf_window);
    }


    //------------------------------------------------------------------------
    // 以下是空函数的定义，用于特定事件的处理
    void platform_support::on_init() {}
    void platform_support::on_resize(int sx, int sy) {}
    void platform_support::on_idle() {}
    void platform_support::on_mouse_move(int x, int y, unsigned flags) {}
    void platform_support::on_mouse_button_down(int x, int y, unsigned flags) {}
    void platform_support::on_mouse_button_up(int x, int y, unsigned flags) {}
    // 定义 platform_support 类的 on_key 方法，处理按键事件
    void platform_support::on_key(int x, int y, unsigned key, unsigned flags) {}
    
    // 定义 platform_support 类的 on_ctrl_change 方法，处理控制变化事件
    void platform_support::on_ctrl_change() {}
    
    // 定义 platform_support 类的 on_draw 方法，处理绘制事件
    void platform_support::on_draw() {}
    
    // 定义 platform_support 类的 on_post_draw 方法，处理绘制后事件，接受一个原始处理程序的指针参数
    void platform_support::on_post_draw(void* raw_handler) {}
//------------------------------------------------------------------------
// 当窗口关闭事件发生时的处理函数
pascal OSStatus DoWindowClose (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
    // 不使用用户数据，但需要保留此参数以符合函数签名
    userData;
    
    // 终止应用程序的事件循环，关闭应用程序
    QuitApplicationEventLoop ();

    // 调用下一个事件处理器处理事件
    return CallNextEventHandler (nextHandler, theEvent);
}


//------------------------------------------------------------------------
// 当应用程序退出事件发生时的处理函数
pascal OSStatus DoAppQuit (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
    // 不使用用户数据，但需要保留此参数以符合函数签名
    userData;
    
    // 直接返回到下一个事件处理器，不执行其他操作
    return CallNextEventHandler (nextHandler, theEvent);
}


//------------------------------------------------------------------------
// 当鼠标按下事件发生时的处理函数
pascal OSStatus DoMouseDown (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
    Point wheresMyMouse;  // 存储鼠标位置的变量
    UInt32 modifier;      // 存储键盘修饰键状态的变量
    
    // 获取鼠标位置参数
    GetEventParameter (theEvent, kEventParamMouseLocation, typeQDPoint, NULL, sizeof(Point), NULL, &wheresMyMouse);
    // 将全局坐标转换为本地坐标
    GlobalToLocal (&wheresMyMouse);
    // 获取键盘修饰键参数
    GetEventParameter (theEvent, kEventParamKeyModifiers, typeUInt32, NULL, sizeof(UInt32), NULL, &modifier);

    // 将用户数据转换为平台支持对象的指针
    platform_support * app = reinterpret_cast<platform_support*>(userData);

    // 更新当前鼠标位置的 x 坐标
    app->m_specific->m_cur_x = wheresMyMouse.h;
    // 根据平台支持对象的 Y 翻转状态更新当前鼠标位置的 y 坐标
    if(app->flip_y())
    {
        app->m_specific->m_cur_y = app->rbuf_window().height() - wheresMyMouse.v;
    }
    else
    {
        app->m_specific->m_cur_y = wheresMyMouse.v;
    }
    // 更新输入标志，包括鼠标左键状态和键盘修饰键状态
    app->m_specific->m_input_flags = mouse_left | get_key_flags(modifier);
    
    // 设置当前控制对象到当前鼠标位置
    app->m_ctrls.set_cur(app->m_specific->m_cur_x, 
                         app->m_specific->m_cur_y);
    
    // 如果鼠标按钮按下事件由控制对象处理，则执行控制对象改变和强制重绘操作
    if(app->m_ctrls.on_mouse_button_down(app->m_specific->m_cur_x, 
                                         app->m_specific->m_cur_y))
    {
        app->on_ctrl_change();
        app->force_redraw();
    }
    else
    {
        // 如果鼠标不在控制对象内部
        if(app->m_ctrls.in_rect(app->m_specific->m_cur_x, 
                                app->m_specific->m_cur_y))
        {
            // 尝试将当前控制对象设置为鼠标位置，并执行控制对象改变和强制重绘操作
            if(app->m_ctrls.set_cur(app->m_specific->m_cur_x, 
                                    app->m_specific->m_cur_y))
            {
                app->on_ctrl_change();
                app->force_redraw();
            }
        }
        else
        {
            // 如果鼠标不在控制对象内部，则执行鼠标按钮按下事件处理
            app->on_mouse_button_down(app->m_specific->m_cur_x, 
                                      app->m_specific->m_cur_y, 
                                      app->m_specific->m_input_flags);
        }
    }

    // 调用下一个事件处理器处理事件
    return CallNextEventHandler (nextHandler, theEvent);
}


//------------------------------------------------------------------------
// 当鼠标释放事件发生时的处理函数
pascal OSStatus DoMouseUp (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
    Point wheresMyMouse;  // 存储鼠标位置的变量
    UInt32 modifier;      // 存储键盘修饰键状态的变量
    
    // 获取鼠标位置参数
    GetEventParameter (theEvent, kEventParamMouseLocation, typeQDPoint, NULL, sizeof(Point), NULL, &wheresMyMouse);
    // 将全局坐标转换为本地坐标
    GlobalToLocal (&wheresMyMouse);
    // 获取键盘修饰键参数
    GetEventParameter (theEvent, kEventParamKeyModifiers, typeUInt32, NULL, sizeof(UInt32), NULL, &modifier);

    // 将用户数据转换为平台支持对象的指针
    platform_support * app = reinterpret_cast<platform_support*>(userData);
    // 设置当前应用程序特定对象的当前 x 坐标为鼠标位置的水平值
    app->m_specific->m_cur_x = wheresMyMouse.h;
    
    // 如果需要翻转 y 坐标，则设置当前应用程序特定对象的当前 y 坐标为窗口高度减去鼠标位置的垂直值
    if(app->flip_y())
    {
        app->m_specific->m_cur_y = app->rbuf_window().height() - wheresMyMouse.v;
    }
    else
    {
        // 否则，直接设置当前应用程序特定对象的当前 y 坐标为鼠标位置的垂直值
        app->m_specific->m_cur_y = wheresMyMouse.v;
    }
    
    // 设置当前应用程序特定对象的输入标志为鼠标左键状态或与修饰键相关的标志
    app->m_specific->m_input_flags = mouse_left | get_key_flags(modifier);

    // 如果在鼠标按钮松开事件中处理了控件的动作
    if(app->m_ctrls.on_mouse_button_up(app->m_specific->m_cur_x, 
                                       app->m_specific->m_cur_y))
    {
        // 调用应用程序的控件变化处理函数
        app->on_ctrl_change();
        // 强制应用程序重绘
        app->force_redraw();
    }
    
    // 在鼠标按钮松开事件中调用应用程序的处理函数，传入当前 x 坐标、y 坐标和输入标志
    app->on_mouse_button_up(app->m_specific->m_cur_x, 
                            app->m_specific->m_cur_y, 
                            app->m_specific->m_input_flags);

    // 调用下一个事件处理器处理事件
    return CallNextEventHandler(nextHandler, theEvent);
//------------------------------------------------------------------------
// 处理鼠标拖动事件的回调函数
pascal OSStatus DoMouseDragged (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
    // 存储鼠标位置和按键状态的变量
    Point wheresMyMouse;
    UInt32 modifier;
    
    // 从事件中获取鼠标位置信息
    GetEventParameter (theEvent, kEventParamMouseLocation, typeQDPoint, NULL, sizeof(Point), NULL, &wheresMyMouse);
    // 将全局坐标转换为本地窗口坐标
    GlobalToLocal (&wheresMyMouse);
    // 从事件中获取键盘修饰键状态信息
    GetEventParameter (theEvent, kEventParamKeyModifiers, typeUInt32, NULL, sizeof(UInt32), NULL, &modifier);

    // 将用户数据转换为特定平台支持对象
    platform_support * app = reinterpret_cast<platform_support*>(userData);

    // 更新当前鼠标位置的水平坐标
    app->m_specific->m_cur_x = wheresMyMouse.h;
    // 如果需要翻转Y轴坐标，则更新当前鼠标位置的垂直坐标
    if(app->flip_y())
    {
        app->m_specific->m_cur_y = app->rbuf_window().height() - wheresMyMouse.v;
    }
    else
    {
        app->m_specific->m_cur_y = wheresMyMouse.v;
    }
    // 设置当前输入标志，包括鼠标左键状态和键盘修饰键状态
    app->m_specific->m_input_flags = mouse_left | get_key_flags(modifier);

    // 如果鼠标移动事件被控件处理，则触发控件变化和强制重绘
    if(app->m_ctrls.on_mouse_move(
        app->m_specific->m_cur_x, 
        app->m_specific->m_cur_y,
        (app->m_specific->m_input_flags & mouse_left) != 0))
    {
        app->on_ctrl_change(); // 触发控件变化事件处理
        app->force_redraw();   // 强制重绘窗口
    }
    else
    {
        // 否则，触发一般的鼠标移动事件处理
        app->on_mouse_move(app->m_specific->m_cur_x, 
                           app->m_specific->m_cur_y, 
                           app->m_specific->m_input_flags);
    }

    // 调用下一个事件处理器处理后续事件
    return CallNextEventHandler (nextHandler, theEvent);
}
    {
        // 初始化方向键状态为 false
        bool left  = false;
        bool up    = false;
        bool right = false;
        bool down  = false;
    
        // 根据最近翻译的键来判断当前按下的是哪个方向键
        switch(app->m_specific->m_last_translated_key)
        {
        // 如果是左箭头键
        case key_left:
            left = true;
            break;
    
        // 如果是上箭头键
        case key_up:
            up = true;
            break;
    
        // 如果是右箭头键
        case key_right:
            right = true;
            break;
    
        // 如果是下箭头键
        case key_down:
            down = true;
            break;
    
        // 如果是 F2 键（在 Mac 上，处理截图由系统完成）
        case key_f2:                        
            // 复制窗口到图像
            app->copy_window_to_img(agg::platform_support::max_images - 1);
            // 保存图像
            app->save_img(agg::platform_support::max_images - 1, "screenshot");
            break;
        }
    
        // 将方向键状态传递给控制器处理，若处理成功
        if(app->m_ctrls.on_arrow_keys(left, right, down, up))
        {
            // 触发控制器变化事件
            app->on_ctrl_change();
            // 强制重绘应用程序
            app->force_redraw();
        }
        else
        {
            // 如果控制器处理失败，处理键盘事件
            app->on_key(app->m_specific->m_cur_x,
                        app->m_specific->m_cur_y,
                        app->m_specific->m_last_translated_key,
                        app->m_specific->m_input_flags);
        }
    }
    
    // 返回给定事件处理器的下一个处理程序
    return CallNextEventHandler(nextHandler, theEvent);
//------------------------------------------------------------------------
pascal OSStatus DoKeyUp (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
    // 声明变量存储按键码和修饰键信息
    char key_code;
    UInt32 modifier;
    
    // 从事件参数中获取按键字符代码
    GetEventParameter (theEvent, kEventParamKeyMacCharCodes, typeChar, NULL, sizeof(char), NULL, &key_code);
    // 从事件参数中获取按键的修饰键信息
    GetEventParameter (theEvent, kEventParamKeyModifiers, typeUInt32, NULL, sizeof(UInt32), NULL, &modifier);

    // 将用户数据转换为特定平台支持类的指针
    platform_support * app = reinterpret_cast<platform_support*>(userData);

    // 清除上一个翻译的按键信息
    app->m_specific->m_last_translated_key = 0;
    // 根据修饰键类型进行不同处理
    switch(modifier) 
    {
        // 如果是控制键
        case controlKey:
            // 清除控制键标志位
            app->m_specific->m_input_flags &= ~kbd_ctrl;
            break;

        // 如果是Shift键
        case shiftKey:
            // 清除Shift键标志位
            app->m_specific->m_input_flags &= ~kbd_shift;
            break;
    }
    
    // 调用下一个事件处理程序
    return CallNextEventHandler (nextHandler, theEvent);
}


//------------------------------------------------------------------------
pascal OSStatus DoWindowDrawContent (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
    // 将用户数据转换为特定平台支持类的指针
    platform_support * app = reinterpret_cast<platform_support*>(userData);

    // 检查应用程序是否有效
    if(app)
    {
        // 如果需要重新绘制
        if(app->m_specific->m_redraw_flag)
        {
            // 调用绘制函数
            app->on_draw();
            // 清除重新绘制标志
            app->m_specific->m_redraw_flag = false;
        }
        // 显示位图
        app->m_specific->display_pmap(app->m_specific->m_window, &app->rbuf_window());
    }

    // 调用下一个事件处理程序
    return CallNextEventHandler (nextHandler, theEvent);
}


//------------------------------------------------------------------------
pascal void DoPeriodicTask (EventLoopTimerRef theTimer, void* userData)
{
    // 将用户数据转换为特定平台支持类的指针
    platform_support * app = reinterpret_cast<platform_support*>(userData);
    
    // 如果不是等待模式
    if(!app->wait_mode())
        // 执行空闲任务
        app->on_idle();
}


//------------------------------------------------------------------------
int agg_main(int argc, char* argv[]);


//----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
#if defined(__MWERKS__)
    // 使用ccommand函数获取命令行参数
    // argc = ccommand (&argv);
#endif
    
    // 检查是否通过双击方式在OSX下启动
    // 如果第一个参数是"-psn"，则认为是由Finder启动
    // 将argc设置为1，避免与标准的参数解析冲突
    if ( argc >= 2 && strncmp (argv[1], "-psn", 4) == 0 ) {
        argc = 1;
    } 

launch:
    // 调用主函数进行程序执行
    return agg_main(argc, argv);
}
```