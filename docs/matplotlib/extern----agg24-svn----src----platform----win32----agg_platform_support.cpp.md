# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\platform\win32\agg_platform_support.cpp`

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
// class platform_support
//
//----------------------------------------------------------------------------

#include <windows.h>
#include <string.h>
#include "platform/agg_platform_support.h"
#include "platform/win32/agg_win32_bmp.h"
#include "util/agg_color_conv.h"
#include "util/agg_color_conv_rgb8.h"
#include "util/agg_color_conv_rgb16.h"
#include "agg_pixfmt_gray.h"
#include "agg_pixfmt_rgb.h"
#include "agg_pixfmt_rgba.h"


namespace agg
{
    
    //------------------------------------------------------------------------
    // 全局变量，保存 Windows 实例句柄和窗口显示命令
    HINSTANCE g_windows_instance = 0;
    int       g_windows_cmd_show = 0;


    //------------------------------------------------------------------------
    // 平台特定类，负责处理与平台相关的图像操作
    class platform_specific
    {
    public:
        // 构造函数，初始化像素格式和是否翻转 Y 轴
        platform_specific(pix_format_e format, bool flip_y);

        // 创建像素图，设置窗口的渲染缓冲区
        void create_pmap(unsigned width, unsigned height, 
                         rendering_buffer* wnd);

        // 显示像素图到设备上下文 DC
        void display_pmap(HDC dc, const rendering_buffer* src);

        // 加载像素图文件到指定缓冲区
        bool load_pmap(const char* fn, unsigned idx, 
                       rendering_buffer* dst);

        // 将像素图保存到文件
        bool save_pmap(const char* fn, unsigned idx, 
                       const rendering_buffer* src);

        // 翻译键盘按键为标准键盘码
        unsigned translate(unsigned keycode);

        // 成员变量
        pix_format_e  m_format;           // AGG 像素格式
        pix_format_e  m_sys_format;       // 系统像素格式
        bool          m_flip_y;           // 是否翻转 Y 轴
        unsigned      m_bpp;              // 每像素位数
        unsigned      m_sys_bpp;          // 系统每像素位数
        HWND          m_hwnd;             // 窗口句柄
        pixel_map     m_pmap_window;      // 窗口像素图
        pixel_map     m_pmap_img[platform_support::max_images]; // 图像数组
        unsigned      m_keymap[256];      // 键盘映射
        unsigned      m_last_translated_key; // 最后翻译的键
        int           m_cur_x;            // 当前 X 坐标
        int           m_cur_y;            // 当前 Y 坐标
        unsigned      m_input_flags;      // 输入标志
        bool          m_redraw_flag;      // 重绘标志
        HDC           m_current_dc;       // 当前设备上下文
        LARGE_INTEGER m_sw_freq;          // 计时器频率
        LARGE_INTEGER m_sw_start;         // 计时器开始时间
    };


    //------------------------------------------------------------------------
    // 构造函数 platform_specific 的初始化列表，设置各个成员变量的初始值
    platform_specific::platform_specific(pix_format_e format, bool flip_y) :
        m_format(format),
        m_sys_format(pix_format_undefined),
        m_flip_y(flip_y),
        m_bpp(0),
        m_sys_bpp(0),
        m_hwnd(0),
        m_last_translated_key(0),
        m_cur_x(0),
        m_cur_y(0),
        m_input_flags(0),
        m_redraw_flag(true),
        m_current_dc(0)
    {
    }


    //------------------------------------------------------------------------
    // 创建像素映射函数，初始化 m_pmap_window
    void platform_specific::create_pmap(unsigned width, 
                                        unsigned height,
                                        rendering_buffer* wnd)
    {
        // 使用 m_bpp 创建像素映射窗口
        m_pmap_window.create(width, height, org_e(m_bpp));
        // 将渲染缓冲区 wnd 与 m_pmap_window 关联
        wnd->attach(m_pmap_window.buf(), 
                    m_pmap_window.width(),
                    m_pmap_window.height(),
                    // 根据 m_flip_y 决定是否翻转 y 方向的像素映射窗口
                    m_flip_y ?
                    m_pmap_window.stride() :
                    -m_pmap_window.stride());
    }


    //------------------------------------------------------------------------
    // 静态函数，用于转换像素映射
    static void convert_pmap(rendering_buffer* dst, 
                             const rendering_buffer* src, 
                             pix_format_e format)
    {
    }


    //------------------------------------------------------------------------
    // 显示像素映射函数，根据当前系统格式显示像素映射
    void platform_specific::display_pmap(HDC dc, const rendering_buffer* src)
    {
        // 如果当前系统格式与设定格式一致，则直接绘制 m_pmap_window 到 dc
        if(m_sys_format == m_format)
        {
            m_pmap_window.draw(dc);
        }
        else
        {
            // 否则，创建临时像素映射 pmap_tmp
            pixel_map pmap_tmp;
            pmap_tmp.create(m_pmap_window.width(), 
                            m_pmap_window.height(),
                            org_e(m_sys_bpp));

            // 创建临时渲染缓冲区 rbuf_tmp，并根据 m_flip_y 设置其方向
            rendering_buffer rbuf_tmp;
            rbuf_tmp.attach(pmap_tmp.buf(),
                            pmap_tmp.width(),
                            pmap_tmp.height(),
                            m_flip_y ?
                            pmap_tmp.stride() :
                            -pmap_tmp.stride());

            // 转换源像素映射 src 到目标像素映射 rbuf_tmp，使用设定格式 m_format
            convert_pmap(&rbuf_tmp, src, m_format);
            // 绘制 pmap_tmp 到 dc
            pmap_tmp.draw(dc);
        }
    }


    //------------------------------------------------------------------------
    // 保存像素映射函数，根据当前系统格式保存像素映射
    bool platform_specific::save_pmap(const char* fn, unsigned idx, 
                                      const rendering_buffer* src)
    {
        // 如果当前系统格式与设定格式一致，则将 m_pmap_img[idx] 保存为 BMP 文件
        if(m_sys_format == m_format)
        {
            return m_pmap_img[idx].save_as_bmp(fn);
        }

        // 否则，创建临时像素映射 pmap_tmp
        pixel_map pmap_tmp;
        pmap_tmp.create(m_pmap_img[idx].width(), 
                          m_pmap_img[idx].height(),
                          org_e(m_sys_bpp));

        // 创建临时渲染缓冲区 rbuf_tmp，并根据 m_flip_y 设置其方向
        rendering_buffer rbuf_tmp;
        rbuf_tmp.attach(pmap_tmp.buf(),
                          pmap_tmp.width(),
                          pmap_tmp.height(),
                          m_flip_y ?
                          pmap_tmp.stride() :
                          -pmap_tmp.stride());

        // 转换源像素映射 src 到目标像素映射 rbuf_tmp，使用设定格式 m_format
        convert_pmap(&rbuf_tmp, src, m_format);
        // 将 pmap_tmp 保存为 BMP 文件，并返回保存结果
        return pmap_tmp.save_as_bmp(fn);
    }
    //------------------------------------------------------------------------
    bool platform_specific::load_pmap(const char* fn, unsigned idx, 
                                      rendering_buffer* dst)
    {
    }
    
    //------------------------------------------------------------------------
    unsigned platform_specific::translate(unsigned keycode)
    {
        // 返回键码对应的翻译后的键码，若大于255则返回0
        return m_last_translated_key = (keycode > 255) ? 0 : m_keymap[keycode];
    }
    
    //------------------------------------------------------------------------
    platform_support::platform_support(pix_format_e format, bool flip_y) :
        // 初始化特定平台相关的对象
        m_specific(new platform_specific(format, flip_y)),
        m_format(format),
        m_bpp(m_specific->m_bpp),
        m_window_flags(0),
        m_wait_mode(true),
        m_flip_y(flip_y),
        m_initial_width(10),
        m_initial_height(10)
    {
        // 设置窗口标题
        strcpy(m_caption, "Anti-Grain Geometry Application");
    }
    
    //------------------------------------------------------------------------
    platform_support::~platform_support()
    {
        // 删除特定平台对象的内存
        delete m_specific;
    }
    
    //------------------------------------------------------------------------
    void platform_support::caption(const char* cap)
    {
        // 设置窗口标题为给定的字符串
        strcpy(m_caption, cap);
        // 如果窗口句柄存在，设置窗口标题
        if(m_specific->m_hwnd)
        {
            SetWindowText(m_specific->m_hwnd, m_caption);
        }
    }
    
    //------------------------------------------------------------------------
    void platform_support::start_timer()
    {
        // 开始计时器，记录起始时间
        ::QueryPerformanceCounter(&(m_specific->m_sw_start));
    }
    
    //------------------------------------------------------------------------
    double platform_support::elapsed_time() const
    {
        LARGE_INTEGER stop;
        // 获取当前时间
        ::QueryPerformanceCounter(&stop);
        // 计算经过的时间（毫秒）
        return double(stop.QuadPart - 
                      m_specific->m_sw_start.QuadPart) * 1000.0 / 
                      double(m_specific->m_sw_freq.QuadPart);
    }
    
    //------------------------------------------------------------------------
    static unsigned get_key_flags(int wflags)
    {
        unsigned flags = 0;
        // 根据窗口标志位设置键盘鼠标状态标志
        if(wflags & MK_LBUTTON) flags |= mouse_left;
        if(wflags & MK_RBUTTON) flags |= mouse_right;
        if(wflags & MK_SHIFT)   flags |= kbd_shift;
        if(wflags & MK_CONTROL) flags |= kbd_ctrl;
        return flags;
    }
    
    void* platform_support::raw_display_handler()
    {
        // 返回当前平台特定的显示处理器
        return m_specific->m_current_dc;
    }
    
    //------------------------------------------------------------------------
    LRESULT CALLBACK window_proc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
    {
        // 窗口消息处理回调函数
    }
/*
            // 如果应用程序不处于等待模式
            if(!app->wait_mode())
            {
                // 在空闲状态下执行操作
                app->on_idle();
            }
*/
            // 跳出当前的 switch case
            break;

        //--------------------------------------------------------------------
        case WM_LBUTTONUP:
            // 释放鼠标捕获
            ::ReleaseCapture();
            // 设置当前鼠标位置的 x 坐标
            app->m_specific->m_cur_x = int16(LOWORD(lParam));
            // 如果需要翻转 y 坐标
            if(app->flip_y())
            {
                // 设置当前鼠标位置的 y 坐标
                app->m_specific->m_cur_y = app->rbuf_window().height() - int16(HIWORD(lParam));
            }
            else
            {
                // 设置当前鼠标位置的 y 坐标
                app->m_specific->m_cur_y = int16(HIWORD(lParam));
            }
            // 设置鼠标左键按下标志和键盘按键标志
            app->m_specific->m_input_flags = mouse_left | get_key_flags(wParam);

            // 如果鼠标左键抬起事件被控件处理
            if(app->m_ctrls.on_mouse_button_up(app->m_specific->m_cur_x, 
                                               app->m_specific->m_cur_y))
            {
                // 触发控件改变事件
                app->on_ctrl_change();
                // 强制重绘
                app->force_redraw();
            }
            // 处理鼠标左键抬起事件
            app->on_mouse_button_up(app->m_specific->m_cur_x, 
                                    app->m_specific->m_cur_y, 
                                    app->m_specific->m_input_flags);
/*
            // 如果应用程序不处于等待模式
            if(!app->wait_mode())
            {
                // 在空闲状态下执行操作
                app->on_idle();
            }
*/
            // 跳出当前的 switch case
            break;


        //--------------------------------------------------------------------
        case WM_RBUTTONDOWN:
            // 设置鼠标捕获
            ::SetCapture(app->m_specific->m_hwnd);
            // 设置当前鼠标位置的 x 坐标
            app->m_specific->m_cur_x = int16(LOWORD(lParam));
            // 如果需要翻转 y 坐标
            if(app->flip_y())
            {
                // 设置当前鼠标位置的 y 坐标
                app->m_specific->m_cur_y = app->rbuf_window().height() - int16(HIWORD(lParam));
            }
            else
            {
                // 设置当前鼠标位置的 y 坐标
                app->m_specific->m_cur_y = int16(HIWORD(lParam));
            }
            // 设置鼠标右键按下标志和键盘按键标志
            app->m_specific->m_input_flags = mouse_right | get_key_flags(wParam);
            // 处理鼠标右键按下事件
            app->on_mouse_button_down(app->m_specific->m_cur_x, 
                                      app->m_specific->m_cur_y, 
                                      app->m_specific->m_input_flags);
/*
            // 如果应用程序不处于等待模式
            if(!app->wait_mode())
            {
                // 在空闲状态下执行操作
                app->on_idle();
            }
*/
            // 跳出当前的 switch case
            break;

        //--------------------------------------------------------------------
        case WM_RBUTTONUP:
            // 释放鼠标捕获
            ::ReleaseCapture();
            // 设置当前鼠标位置的 x 坐标
            app->m_specific->m_cur_x = int16(LOWORD(lParam));
            // 如果需要翻转 y 坐标
            if(app->flip_y())
            {
                // 设置当前鼠标位置的 y 坐标
                app->m_specific->m_cur_y = app->rbuf_window().height() - int16(HIWORD(lParam));
            }
            else
            {
                // 设置当前鼠标位置的 y 坐标
                app->m_specific->m_cur_y = int16(HIWORD(lParam));
            }
            // 设置鼠标右键抬起标志和键盘按键标志
            app->m_specific->m_input_flags = mouse_right | get_key_flags(wParam);
            // 处理鼠标右键抬起事件
            app->on_mouse_button_up(app->m_specific->m_cur_x, 
                                    app->m_specific->m_cur_y, 
                                    app->m_specific->m_input_flags);
/*
            // 如果应用程序不处于等待模式
            if(!app->wait_mode())
            {
                // 在空闲状态下执行操作
                app->on_idle();
            }
*/
            break;

// 结束当前的 switch 语句块，跳出循环。

        //--------------------------------------------------------------------
        case WM_MOUSEMOVE:
            // 更新应用程序中鼠标当前的 x 坐标
            app->m_specific->m_cur_x = int16(LOWORD(lParam));
            // 如果需要翻转 y 坐标，则更新应用程序中鼠标当前的 y 坐标
            if(app->flip_y())
            {
                app->m_specific->m_cur_y = app->rbuf_window().height() - int16(HIWORD(lParam));
            }
            else
            {
                app->m_specific->m_cur_y = int16(HIWORD(lParam));
            }
            // 获取当前的鼠标键盘状态，并存储到应用程序的输入标志中
            app->m_specific->m_input_flags = get_key_flags(wParam);


            // 如果鼠标移动事件被控件处理
            if(app->m_ctrls.on_mouse_move(
                app->m_specific->m_cur_x, 
                app->m_specific->m_cur_y,
                (app->m_specific->m_input_flags & mouse_left) != 0))
            {
                // 触发控件改变事件
                app->on_ctrl_change();
                // 强制重绘应用程序
                app->force_redraw();
            }
            else
            {
                // 如果鼠标不在任何控件上移动
                if(!app->m_ctrls.in_rect(app->m_specific->m_cur_x, 
                                         app->m_specific->m_cur_y))
                {
                    // 触发鼠标移动事件
                    app->on_mouse_move(app->m_specific->m_cur_x, 
                                       app->m_specific->m_cur_y, 
                                       app->m_specific->m_input_flags);
                }
            }
/*
            // 如果应用程序不处于等待模式
            if(!app->wait_mode())
            {
                // 触发空闲事件
                app->on_idle();
            }
        */
            // 如果遇到 break 语句，则退出当前循环或 switch 语句
            break;

        //--------------------------------------------------------------------
        // 处理系统按键和普通按键消息
        case WM_SYSKEYDOWN:
        case WM_KEYDOWN:
            // 清空最后翻译的键值
            app->m_specific->m_last_translated_key = 0;
            // 根据按键码 wParam 执行不同的操作
            switch(wParam) 
            {
                // 按下的是 Ctrl 键
                case VK_CONTROL:
                    // 设置输入标志为键盘 Ctrl 按下
                    app->m_specific->m_input_flags |= kbd_ctrl;
                    break;

                // 按下的是 Shift 键
                case VK_SHIFT:
                    // 设置输入标志为键盘 Shift 按下
                    app->m_specific->m_input_flags |= kbd_shift;
                    break;

                // 其他按键
                default:
                    // 调用 translate 方法进行键值翻译
                    app->m_specific->translate(wParam);
                    break;
            }
        
            // 如果有成功翻译的按键
            if(app->m_specific->m_last_translated_key)
            {
                bool left  = false;
                bool up    = false;
                bool right = false;
                bool down  = false;

                // 根据翻译后的键值进行不同的操作
                switch(app->m_specific->m_last_translated_key)
                {
                // 按键是向左箭头
                case key_left:
                    left = true;
                    break;

                // 按键是向上箭头
                case key_up:
                    up = true;
                    break;

                // 按键是向右箭头
                case key_right:
                    right = true;
                    break;

                // 按键是向下箭头
                case key_down:
                    down = true;
                    break;

                // 按键是 F2 键
                case key_f2:                        
                    // 复制窗口内容到图像，然后保存截图
                    app->copy_window_to_img(agg::platform_support::max_images - 1);
                    app->save_img(agg::platform_support::max_images - 1, "screenshot");
                    break;
                }

                // 如果窗口标志要求处理所有键盘按键
                if(app->window_flags() & window_process_all_keys)
                {
                    // 调用 on_key 处理键盘按键事件
                    app->on_key(app->m_specific->m_cur_x,
                                app->m_specific->m_cur_y,
                                app->m_specific->m_last_translated_key,
                                app->m_specific->m_input_flags);
                }
                else
                {
                    // 否则，检查箭头键是否被控件处理
                    if(app->m_ctrls.on_arrow_keys(left, right, down, up))
                    {
                        // 控件处理了箭头键，触发控件改变事件并强制重新绘制
                        app->on_ctrl_change();
                        app->force_redraw();
                    }
                    else
                    {
                        // 控件未处理箭头键，继续调用 on_key 处理事件
                        app->on_key(app->m_specific->m_cur_x,
                                    app->m_specific->m_cur_y,
                                    app->m_specific->m_last_translated_key,
                                    app->m_specific->m_input_flags);
                    }
                }
            }
        /*
            // 如果不是等待模式，则调用 on_idle 方法处理空闲事件
            if(!app->wait_mode())
            {
                app->on_idle();
            }
        */
            break;

        //--------------------------------------------------------------------
        // 处理按键抬起事件
        case WM_SYSKEYUP:
        case WM_KEYUP:
            // 清除最近翻译的按键
            app->m_specific->m_last_translated_key = 0;
            switch(wParam) 
            {
                case VK_CONTROL:
                    // 清除控制键标志位
                    app->m_specific->m_input_flags &= ~kbd_ctrl;
                    break;

                case VK_SHIFT:
                    // 清除Shift键标志位
                    app->m_specific->m_input_flags &= ~kbd_shift;
                    break;
            }
            break;

        //--------------------------------------------------------------------
        // 处理字符输入事件
        case WM_CHAR:
        case WM_SYSCHAR:
            if(app->m_specific->m_last_translated_key == 0)
            {
                // 调用应用程序的按键处理函数
                app->on_key(app->m_specific->m_cur_x,
                            app->m_specific->m_cur_y,
                            wParam,
                            app->m_specific->m_input_flags);
            }
            break;
        
        //--------------------------------------------------------------------
        // 处理窗口重绘事件
        case WM_PAINT:
            // 开始绘制
            paintDC = ::BeginPaint(hWnd, &ps);
            app->m_specific->m_current_dc = paintDC;
            if(app->m_specific->m_redraw_flag)
            {
                // 调用应用程序的绘制函数
                app->on_draw();
                app->m_specific->m_redraw_flag = false;
            }
            // 显示位图
            app->m_specific->display_pmap(paintDC, &app->rbuf_window());
            // 完成绘制后处理
            app->on_post_draw(paintDC);
            app->m_specific->m_current_dc = 0;
            // 结束绘制
            ::EndPaint(hWnd, &ps);
            break;
        
        //--------------------------------------------------------------------
        // 处理命令事件
        case WM_COMMAND:
            break;
        
        //--------------------------------------------------------------------
        // 处理窗口销毁事件
        case WM_DESTROY:
            // 发送退出消息
            ::PostQuitMessage(0);
            break;
        
        //--------------------------------------------------------------------
        // 处理默认情况
        default:
            // 调用默认窗口处理过程
            ret = ::DefWindowProc(hWnd, msg, wParam, lParam);
            break;
        }
        app->m_specific->m_current_dc = 0;
        // 释放设备上下文
        ::ReleaseDC(app->m_specific->m_hwnd, dc);
        // 返回处理结果
        return ret;
    }


    //------------------------------------------------------------------------
    // 显示消息框
    void platform_support::message(const char* msg)
    {
        ::MessageBox(m_specific->m_hwnd, msg, "AGG Message", MB_OK);
    }


    //------------------------------------------------------------------------
    // 初始化平台支持
    bool platform_support::init(unsigned width, unsigned height, unsigned flags)
    {
        // 检查特定对象的系统格式是否为未定义，如果是，则返回 false
        if(m_specific->m_sys_format == pix_format_undefined)
        {
            return false;
        }
    
        // 将窗口标志设置为给定的标志
        m_window_flags = flags;
    
        // 设置窗口类的标志，包括 CS_OWNDC、CS_VREDRAW 和 CS_HREDRAW
        int wflags = CS_OWNDC | CS_VREDRAW | CS_HREDRAW;
    
        // 定义一个窗口类 WNDCLASS 对象
        WNDCLASS wc;
    
        // 设置窗口类的名称
        wc.lpszClassName = "AGGAppClass";
    
        // 设置窗口过程函数
        wc.lpfnWndProc = window_proc;
    
        // 设置窗口类的样式
        wc.style = wflags;
    
        // 设置窗口实例句柄
        wc.hInstance = g_windows_instance;
    
        // 设置窗口类的图标
        wc.hIcon = LoadIcon(0, IDI_APPLICATION);
    
        // 设置窗口类的光标
        wc.hCursor = LoadCursor(0, IDC_ARROW);
    
        // 设置窗口类的背景颜色
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
    
        // 设置窗口类的菜单名称
        wc.lpszMenuName = "AGGAppMenu";
    
        // 设置窗口类的额外类空间
        wc.cbClsExtra = 0;
    
        // 设置窗口类的额外窗口空间
        wc.cbWndExtra = 0;
    
        // 注册窗口类
        ::RegisterClass(&wc);
    
        // 设置窗口标志为 WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX
        wflags = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    
        // 如果窗口标志包含 window_resize 标志，则添加 WS_THICKFRAME 和 WS_MAXIMIZEBOX 标志
        if(m_window_flags & window_resize)
        {
            wflags |= WS_THICKFRAME | WS_MAXIMIZEBOX;
        }
    
        // 创建窗口并获取其句柄，设置窗口位置和大小
        m_specific->m_hwnd = ::CreateWindow("AGGAppClass",
                                            m_caption,
                                            wflags,
                                            100,
                                            100,
                                            width,
                                            height,
                                            0,
                                            0,
                                            g_windows_instance,
                                            0);
    
        // 如果窗口句柄为 0，则创建窗口失败，返回 false
        if(m_specific->m_hwnd == 0)
        {
            return false;
        }
    
        // 获取客户区域的矩形尺寸
        RECT rct;
        ::GetClientRect(m_specific->m_hwnd, &rct);
    
        // 调整窗口的位置和大小，确保客户区域达到指定的宽度和高度
        ::MoveWindow(m_specific->m_hwnd,   // 窗口句柄
                     100,                  // 水平位置
                     100,                  // 垂直位置
                     width + (width - (rct.right - rct.left)),
                     height + (height - (rct.bottom - rct.top)),
                     FALSE);
    
        // 将窗口实例与当前对象关联
        ::SetWindowLongPtr(m_specific->m_hwnd, GWLP_USERDATA, (LONG)this);
    
        // 使用指定的宽度和高度创建像素映射，将结果存储在 m_rbuf_window 中
        m_specific->create_pmap(width, height, &m_rbuf_window);
    
        // 记录初始宽度和高度
        m_initial_width = width;
        m_initial_height = height;
    
        // 在初始化后执行特定操作
        on_init();
    
        // 设置重绘标志为 true
        m_specific->m_redraw_flag = true;
    
        // 显示窗口
        ::ShowWindow(m_specific->m_hwnd, g_windows_cmd_show);
    
        // 返回 true 表示窗口创建成功
        return true;
    }
    
    
    
    //------------------------------------------------------------------------
    int platform_support::run()
    {
        // 定义 Windows 消息结构体
        MSG msg;
    
        // 无限循环处理消息
        for(;;)
        {
            // 如果处于等待模式
            if(m_wait_mode)
            {
                // 获取消息队列中的消息
                if(!::GetMessage(&msg, 0, 0, 0))
                {
                    break;  // 如果获取失败则退出循环
                }
                // 转换消息
                ::TranslateMessage(&msg);
                // 分发消息
                ::DispatchMessage(&msg);
            }
            else
            {
                // 如果有消息存在则处理
                if(::PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
                {
                    // 转换消息
                    ::TranslateMessage(&msg);
                    // 如果是退出消息则退出循环
                    if(msg.message == WM_QUIT)
                    {
                        break;
                    }
                    // 分发消息
                    ::DispatchMessage(&msg);
                }
                else
                {
                    // 没有消息时执行空闲处理
                    on_idle();
                }
            }
        }
        // 返回消息参数的整型值
        return (int)msg.wParam;
    }
    
    
    //------------------------------------------------------------------------
    // 返回图像文件的默认扩展名为 ".bmp"
    const char* platform_support::img_ext() const { return ".bmp"; }
    
    
    //------------------------------------------------------------------------
    // 返回文件的全文件名
    const char* platform_support::full_file_name(const char* file_name)
    {
        return file_name;
    }
    
    //------------------------------------------------------------------------
    // 加载指定索引的图像文件
    bool platform_support::load_img(unsigned idx, const char* file)
    {
        // 如果索引小于最大图像数
        if(idx < max_images)
        {
            char fn[1024];
            strcpy(fn, file);
            int len = strlen(fn);
            // 如果文件名长度小于4或者不是以 ".BMP" 结尾，则追加 ".bmp" 后缀
            if(len < 4 || _stricmp(fn + len - 4, ".BMP") != 0)
            {
                strcat(fn, ".bmp");
            }
            // 调用特定平台实现的加载图像方法
            return m_specific->load_pmap(fn, idx, &m_rbuf_img[idx]);
        }
        // 如果索引超出范围则返回 true
        return true;
    }
    
    
    
    //------------------------------------------------------------------------
    // 保存指定索引的图像文件
    bool platform_support::save_img(unsigned idx, const char* file)
    {
        // 如果索引小于最大图像数
        if(idx < max_images)
        {
            char fn[1024];
            strcpy(fn, file);
            int len = strlen(fn);
            // 如果文件名长度小于4或者不是以 ".BMP" 结尾，则追加 ".bmp" 后缀
            if(len < 4 || _stricmp(fn + len - 4, ".BMP") != 0)
            {
                strcat(fn, ".bmp");
            }
            // 调用特定平台实现的保存图像方法
            return m_specific->save_pmap(fn, idx, &m_rbuf_img[idx]);
        }
        // 如果索引超出范围则返回 true
        return true;
    }
    
    
    
    //------------------------------------------------------------------------
    // 创建指定索引的图像
    bool platform_support::create_img(unsigned idx, unsigned width, unsigned height)
    {
        // 检查索引是否小于最大图像数
        if (idx < max_images)
        {
            // 如果宽度为0，则使用指定对象的窗口宽度
            if (width == 0) width = m_specific->m_pmap_window.width();
            // 如果高度为0，则使用指定对象的窗口高度
            if (height == 0) height = m_specific->m_pmap_window.height();
            
            // 创建指定索引位置的图像对象，并设置其尺寸和像素格式
            m_specific->m_pmap_img[idx].create(width, height, org_e(m_specific->m_bpp));
            
            // 将渲染缓冲区与创建的图像对象关联，根据m_flip_y参数选择是否垂直翻转
            m_rbuf_img[idx].attach(m_specific->m_pmap_img[idx].buf(), 
                                   m_specific->m_pmap_img[idx].width(),
                                   m_specific->m_pmap_img[idx].height(),
                                   m_flip_y ?
                                    m_specific->m_pmap_img[idx].stride() :
                                   -m_specific->m_pmap_img[idx].stride());
            
            // 操作成功，返回true
            return true;
        }
        // 索引超出最大图像数，返回false
        return false;
    }
    
    
    //------------------------------------------------------------------------
    // 强制重新绘制窗口
    void platform_support::force_redraw()
    {
        // 设置重绘标志为true
        m_specific->m_redraw_flag = true;
        // 强制使窗口失效，需要系统重新绘制
        ::InvalidateRect(m_specific->m_hwnd, 0, FALSE);
    }
    
    
    //------------------------------------------------------------------------
    // 更新窗口显示
    void platform_support::update_window()
    {
        // 获取窗口的设备上下文
        HDC dc = ::GetDC(m_specific->m_hwnd);
        // 在指定设备上下文中显示pmap对象的内容
        m_specific->display_pmap(dc, &m_rbuf_window);
        // 释放设备上下文
        ::ReleaseDC(m_specific->m_hwnd, dc);
    }
    
    
    //------------------------------------------------------------------------
    // 下面的函数只是空操作，没有具体实现
    // 初始化操作
    void platform_support::on_init() {}
    
    // 窗口大小改变事件
    void platform_support::on_resize(int sx, int sy) {}
    
    // 空闲时的操作
    void platform_support::on_idle() {}
    
    // 鼠标移动事件
    void platform_support::on_mouse_move(int x, int y, unsigned flags) {}
    
    // 鼠标按键按下事件
    void platform_support::on_mouse_button_down(int x, int y, unsigned flags) {}
    
    // 鼠标按键释放事件
    void platform_support::on_mouse_button_up(int x, int y, unsigned flags) {}
    
    // 键盘按键事件
    void platform_support::on_key(int x, int y, unsigned key, unsigned flags) {}
    
    // 控制状态改变事件
    void platform_support::on_ctrl_change() {}
    
    // 绘制事件
    void platform_support::on_draw() {}
    
    // 绘制后处理事件
    void platform_support::on_post_draw(void* raw_handler) {}
    }
}


namespace agg
{
    // 这是一个用于解析命令行的类，因为 Windows 没有提供获取命令行参数的方法，比如 argc, argv。
    // 当然，有 CommandLineToArgv()，但它首先返回我不需要处理的 Unicode 格式，最重要的是它不兼容 Win98。
    //-----------------------------------------------------------------------
    class tokenizer
    {
    public:
        enum sep_flag
        {
            single,     // 单个分隔符模式
            multiple,   // 多个分隔符模式
            whole_str   // 整个字符串模式
        };

        struct token
        {
            const char* ptr;    // 指向 token 的指针
            unsigned    len;    // token 的长度
        };

    public:
        // 构造函数，初始化 tokenizer 对象
        tokenizer(const char* sep,
                  const char* trim=0,
                  const char* quote="\"",
                  char mask_chr='\\',
                  sep_flag sf=multiple);

        // 设置待解析的字符串
        void  set_str(const char* str);

        // 返回下一个 token
        token next_token();

    private:
        // 检查字符串中是否包含特定字符
        int  check_chr(const char *str, char chr);

    private:
        const char* m_src_string;    // 待解析的字符串
        int         m_start;         // 下一个 token 的起始位置
        const char* m_sep;           // 分隔符
        const char* m_trim;          // 要去除的字符
        const char* m_quote;         // 引号字符
        char        m_mask_chr;      // 转义字符
        unsigned    m_sep_len;       // 分隔符长度
        sep_flag    m_sep_flag;      // 分隔符模式
    };


    //-----------------------------------------------------------------------
    // 设置待解析的字符串
    inline void tokenizer::set_str(const char* str) 
    { 
        m_src_string = str; 
        m_start = 0;
    }


    //-----------------------------------------------------------------------
    // 检查字符串中是否包含特定字符
    inline int tokenizer::check_chr(const char *str, char chr)
    {
        return int(strchr(str, chr));
    }


    //-----------------------------------------------------------------------
    // 构造函数，初始化 tokenizer 对象
    tokenizer::tokenizer(const char* sep,
                         const char* trim,
                         const char* quote,
                         char mask_chr,
                         sep_flag sf) :
        m_src_string(0),
        m_start(0),
        m_sep(sep),
        m_trim(trim),
        m_quote(quote),
        m_mask_chr(mask_chr),
        m_sep_len(sep ? strlen(sep) : 0),
        m_sep_flag(sep ? sf : single)
    {
    }


    //-----------------------------------------------------------------------
    // 返回下一个 token
    tokenizer::token tokenizer::next_token()
    }
}



//----------------------------------------------------------------------------
// 声明 agg_main 函数
int agg_main(int argc, char* argv[]);


//----------------------------------------------------------------------------
// WinMain 函数的声明
int PASCAL WinMain(HINSTANCE hInstance,
                   HINSTANCE hPrevInstance,
                   LPSTR lpszCmdLine,
                   int nCmdShow)
{
    agg::g_windows_instance = hInstance;    // 设置全局变量 g_windows_instance
    agg::g_windows_cmd_show = nCmdShow;     // 设置全局变量 g_windows_cmd_show

    char* argv_str = new char [strlen(lpszCmdLine) + 3];    // 动态分配空间以存储 lpszCmdLine 的副本
    char* argv_ptr = argv_str;    // 设置 argv_ptr 指向 argv_str 的起始位置

    char* argv[64];    // 创建一个 char* 数组，用于存储命令行参数
    memset(argv, 0, sizeof(argv));    // 将 argv 数组清空
    // 使用给定的分隔符和引号设置命令行参数解析器
    agg::tokenizer cmd_line(" ", "\"' ", "\"'", '\\', agg::tokenizer::multiple);
    // 将传入的命令行参数字符串设置给解析器
    cmd_line.set_str(lpszCmdLine);

    // 初始化命令行参数计数器
    int argc = 0;
    // 将第一个参数存入参数数组，并准备填充下一个参数
    argv[argc++] = argv_ptr;
    *argv_ptr++ = 0;

    // 最多处理 64 个参数
    while(argc < 64)
    {
        // 获取下一个参数的标记
        agg::tokenizer::token tok = cmd_line.next_token();
        // 如果标记为空指针，则停止解析
        if(tok.ptr == 0) break;
        // 如果标记长度不为零，则将标记拷贝到参数数组中
        if(tok.len)
        {
            memcpy(argv_ptr, tok.ptr, tok.len);
            argv[argc++] = argv_ptr;
            argv_ptr += tok.len;
            *argv_ptr++ = 0;
        }
    }

    // 调用主函数 agg_main，并传递参数数量和参数数组
    int ret = agg_main(argc, argv);
    // 释放动态分配的参数字符串数组
    delete [] argv_str;

    // 返回主函数的返回值
    return ret;
}
```