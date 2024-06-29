# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\platform\agg_platform_support.h`

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
// It's not a part of the AGG library, it's just a helper class to create 
// interactive demo examples. Since the examples should not be too complex
// this class is provided to support some very basic interactive graphical
// functionality, such as putting the rendered image to the window, simple 
// keyboard and mouse input, window resizing, setting the window title,
// and catching the "idle" events.
// 
// The idea is to have a single header file that does not depend on any 
// platform (I hate these endless #ifdef/#elif/#elif.../#endif) and a number
// of different implementations depending on the concrete platform. 
// The most popular platforms are:
//
// Windows-32 API
// X-Window API
// SDL library (see http://www.libsdl.org/)
// MacOS C/C++ API
// 
// This file does not include any system dependent .h files such as
// windows.h or X11.h, so, your demo applications do not depend on the
// platform. The only file that can #include system dependend headers
// is the implementation file agg_platform_support.cpp. Different
// implementations are placed in different directories, such as
// ~/agg/src/platform/win32
// ~/agg/src/platform/sdl
// ~/agg/src/platform/X11
// and so on.
//
// All the system dependent stuff sits in the platform_specific 
// class which is forward-declared here but not defined. 
// The platform_support class has just a pointer to it and it's 
// the responsibility of the implementation to create/delete it.
// This class being defined in the implementation file can have 
// any platform dependent stuff such as HWND, X11 Window and so on.
//
//----------------------------------------------------------------------------


#ifndef AGG_PLATFORM_SUPPORT_INCLUDED
#define AGG_PLATFORM_SUPPORT_INCLUDED


#include "agg_basics.h"
#include "agg_rendering_buffer.h"
#include "agg_trans_viewport.h"
#include "ctrl/agg_ctrl.h"

namespace agg
{

    //----------------------------------------------------------window_flag_e
    // These are flags used in method init(). Not all of them are
    // applicable on different platforms, for example the win32_api
    // cannot use a hardware buffer (window_hw_buffer).
    // The implementation should simply ignore unsupported flags.
    enum window_flag_e
    {
        window_resize            = 1,  // 窗口可调整大小的标志位，数值为1
        window_hw_buffer         = 2,  // 窗口使用硬件缓冲的标志位，数值为2
        window_keep_aspect_ratio = 4,  // 窗口保持宽高比的标志位，数值为4
        window_process_all_keys  = 8   // 窗口处理所有按键的标志位，数值为8
    };
    
    //-----------------------------------------------------------pix_format_e
    // 渲染缓冲区的可能格式。最初认为根据系统的本机像素格式创建缓冲区和渲染函数是合理的，因为这样做不需要进行像素格式转换。
    // 但最终我得出结论，根据需要在请求时转换像素格式是个好主意。首先，像是 X11，那里有很多不同的格式和视觉效果，希望能够以 RGB-24 渲染并自动显示，不需要额外的努力。
    // 第二个原因是希望能够在同一台计算机和同一个系统上调试支持不同像素格式和色彩空间的渲染器。
    //
    // 这些功能不包括在基本的 AGG 功能中，因为支持的像素格式（和/或色彩空间）数量可能很大，如果需要添加新的格式，只需添加新的渲染文件，而无需修改任何现有文件（这是封装和隔离的一般原则）。
    //
    // 使用特定的像素格式并不一定意味着需要软件转换。例如，win32 API 可以原生显示 gray8、15 位 RGB、24 位 BGR 和 32 位 BGRA 格式。
    // 这个列表可以（并且将会！）在未来扩展。
    enum pix_format_e
    {
        pix_format_undefined = 0,  // 默认值，无需转换
        pix_format_bw,             // 单色，每像素1位黑白
        pix_format_gray8,          // 简单的256级灰度
        pix_format_sgray8,         // 简单的256级灰度（sRGB）
        pix_format_gray16,         // 简单的65535级灰度
        pix_format_gray32,         // 灰度，每像素32位浮点数
        pix_format_rgb555,         // 15位RGB色彩，字节顺序有关
        pix_format_rgb565,         // 16位RGB色彩，字节顺序有关
        pix_format_rgbAAA,         // 30位RGB色彩，字节顺序有关
        pix_format_rgbBBA,         // 32位RGB色彩，字节顺序有关
        pix_format_bgrAAA,         // 30位BGR色彩，字节顺序有关
        pix_format_bgrABB,         // 32位BGR色彩，字节顺序有关
        pix_format_rgb24,          // R-G-B，每颜色分量1字节
        pix_format_srgb24,         // R-G-B，每颜色分量1字节（sRGB）
        pix_format_bgr24,          // B-G-R，每颜色分量1字节
        pix_format_sbgr24,         // B-G-R，本地Win32 BMP格式（sRGB）
        pix_format_rgba32,         // R-G-B-A，每颜色分量1字节
        pix_format_srgba32,        // R-G-B-A，每颜色分量1字节（sRGB）
        pix_format_argb32,         // A-R-G-B，本地MAC格式
        pix_format_sargb32,        // A-R-G-B，本地MAC格式（sRGB）
        pix_format_abgr32,         // A-B-G-R，每颜色分量1字节
        pix_format_sabgr32,        // A-B-G-R，每颜色分量1字节（sRGB）
        pix_format_bgra32,         // B-G-R-A，本地Win32 BMP格式
        pix_format_sbgra32,        // B-G-R-A，本地Win32 BMP格式（sRGB）
        pix_format_rgb48,          // R-G-B，每颜色分量16位
        pix_format_bgr48,          // B-G-R，本地Win32 BMP格式
        pix_format_rgb96,          // R-G-B，每颜色分量32位浮点数
        pix_format_bgr96,          // B-G-R，每颜色分量32位浮点数
        pix_format_rgba64,         // R-G-B-A，每颜色分量16位字节
        pix_format_argb64,         // A-R-G-B，本地MAC格式
        pix_format_abgr64,         // A-B-G-R，每颜色分量1字节
        pix_format_bgra64,         // B-G-R-A，本地Win32 BMP格式
        pix_format_rgba128,        // R-G-B-A，每颜色分量32位浮点数
        pix_format_argb128,        // A-R-G-B，每颜色分量32位浮点数
        pix_format_abgr128,        // A-B-G-R，每颜色分量32位浮点数
        pix_format_bgra128,        // B-G-R-A，每颜色分量32位浮点数
    
        end_of_pix_formats         // 像素格式枚举结束标志
    };
    
    //-------------------------------------------------------------input_flag_e
    // 鼠标和键盘标志。在不同平台上可能有所不同
    // 定义输入标志的枚举，用于表示鼠标和键盘按键状态
    // 鼠标左键对应数值 1，鼠标右键对应数值 2
    // kbd_shift 表示键盘上的 Shift 键，对应数值 4
    // kbd_ctrl 表示键盘上的 Ctrl 键，对应数值 8
    enum input_flag_e
    {
        mouse_left  = 1,
        mouse_right = 2,
        kbd_shift   = 4,
        kbd_ctrl    = 8
    };

    //--------------------------------------------------------------key_code_e
    // 键盘按键代码枚举，用于表示键盘上不可打印的按键
    // 只定义了最基本的按键，可以在不同平台上得到支持
    // 任何依赖于特定平台的键盘代码应当转换为这些代码
    // 所有可打印的 ASCII 字符可以正常使用：如 ' ', 'A', '0', '+'
    // 由于该类用于创建非常简单的演示应用程序，不需要过多的功能
    // 数字键码实际上来自于 SDL 库，因此，SDL 支持的实现不需要进行映射
    enum key_code_e
    {
        // ASCII字符集。在所有地方都应该支持。
        key_backspace      = 8,    // 退格键
        key_tab            = 9,    // 制表符键
        key_clear          = 12,   // 清除键
        key_return         = 13,   // 回车键
        key_pause          = 19,   // 暂停键
        key_escape         = 27,   // ESC键
    
        // 小键盘
        key_delete         = 127,  // 删除键
        key_kp0            = 256,  // 小键盘 0
        key_kp1            = 257,  // 小键盘 1
        key_kp2            = 258,  // 小键盘 2
        key_kp3            = 259,  // 小键盘 3
        key_kp4            = 260,  // 小键盘 4
        key_kp5            = 261,  // 小键盘 5
        key_kp6            = 262,  // 小键盘 6
        key_kp7            = 263,  // 小键盘 7
        key_kp8            = 264,  // 小键盘 8
        key_kp9            = 265,  // 小键盘 9
        key_kp_period      = 266,  // 小键盘小数点
        key_kp_divide      = 267,  // 小键盘除号
        key_kp_multiply    = 268,  // 小键盘乘号
        key_kp_minus       = 269,  // 小键盘减号
        key_kp_plus        = 270,  // 小键盘加号
        key_kp_enter       = 271,  // 小键盘回车
        key_kp_equals      = 272,  // 小键盘等号
    
        // 方向键等
        key_up             = 273,  // 向上箭头键
        key_down           = 274,  // 向下箭头键
        key_right          = 275,  // 向右箭头键
        key_left           = 276,  // 向左箭头键
        key_insert         = 277,  // 插入键
        key_home           = 278,  // 起始键
        key_end            = 279,  // 结束键
        key_page_up        = 280,  // 上翻页键
        key_page_down      = 281,  // 下翻页键
    
        // 功能键。如果希望应用程序可移植，请避免使用 f11...f15。
        key_f1             = 282,
        key_f2             = 283,
        key_f3             = 284,
        key_f4             = 285,
        key_f5             = 286,
        key_f6             = 287,
        key_f7             = 288,
        key_f8             = 289,
        key_f9             = 290,
        key_f10            = 291,
        key_f11            = 292,
        key_f12            = 293,
        key_f13            = 294,
        key_f14            = 295,
        key_f15            = 296,
    
        // 只有在 win32_api 和 win32_sdl 实现中才能保证使用这些键
        key_numlock        = 300,  // 数字锁定键
        key_capslock       = 301,  // 大小写锁定键
        key_scrollock      = 302,  // 滚动锁定键
    
        // 结束键码定义部分
        end_of_key_codes
    };
    
    //------------------------------------------------------------------------
    // 平台相关类的预声明。因为我们在这里一无所知，所以唯一能做的就是将这个类的指针作为数据成员。
    // 它应该在 platform_support 类的构造函数/析构函数中显式创建和销毁。
    // 虽然 platform_specific 的指针是公共的，但应用程序不能访问其成员或方法，因为它对它们一无所知，
    // 这是完美的封装 :-)
    class platform_specific;
    
    
    //----------------------------------------------------------ctrl_container
    // 一个辅助类，包含指向多个控件的指针。
    // 定义一个名为 ctrl_container 的类，用于简化控件事件处理。
    // 当适当的事件发生时，实现应该调用此类的相应方法。
    class ctrl_container
    {
        // 定义一个枚举常量 max_ctrl_e，其值为 64，表示最大控件数目。
        enum max_ctrl_e { max_ctrl = 64 };
    //--------------------------------------------------------------------------
    // ctrl_container 类的公共部分

    //--------------------------------------------------------------------------
    // ctrl_container 类的默认构造函数，初始化控件数量和当前控件索引
    ctrl_container() : m_num_ctrl(0), m_cur_ctrl(-1) {}

    //--------------------------------------------------------------------------
    // 向控件容器中添加控件
    void add(ctrl& c)
    {
        // 如果控件数量小于最大允许数量，则将控件添加到容器中
        if(m_num_ctrl < max_ctrl)
        {
            m_ctrl[m_num_ctrl++] = &c;
        }
    }

    //--------------------------------------------------------------------------
    // 检查坐标 (x, y) 是否在任何一个控件的范围内
    bool in_rect(double x, double y)
    {
        unsigned i;
        for(i = 0; i < m_num_ctrl; i++)
        {
            // 如果找到坐标在控件范围内的控件，返回 true
            if(m_ctrl[i]->in_rect(x, y)) return true;
        }
        return false; // 如果所有控件都不包含此坐标，则返回 false
    }

    //--------------------------------------------------------------------------
    // 处理鼠标按下事件，查看是否有控件处理了该事件
    bool on_mouse_button_down(double x, double y)
    {
        unsigned i;
        for(i = 0; i < m_num_ctrl; i++)
        {
            // 如果某个控件处理了鼠标按下事件，则返回 true
            if(m_ctrl[i]->on_mouse_button_down(x, y)) return true;
        }
        return false; // 如果没有控件处理该事件，则返回 false
    }

    //--------------------------------------------------------------------------
    // 处理鼠标释放事件，查看是否有控件处理了该事件
    bool on_mouse_button_up(double x, double y)
    {
        unsigned i;
        bool flag = false;
        for(i = 0; i < m_num_ctrl; i++)
        {
            // 如果某个控件处理了鼠标释放事件，则设置 flag 为 true
            if(m_ctrl[i]->on_mouse_button_up(x, y)) flag = true;
        }
        return flag; // 返回 flag，表示是否有控件处理了该事件
    }

    //--------------------------------------------------------------------------
    // 处理鼠标移动事件，查看是否有控件处理了该事件
    bool on_mouse_move(double x, double y, bool button_flag)
    {
        unsigned i;
        for(i = 0; i < m_num_ctrl; i++)
        {
            // 如果某个控件处理了鼠标移动事件，则返回 true
            if(m_ctrl[i]->on_mouse_move(x, y, button_flag)) return true;
        }
        return false; // 如果没有控件处理该事件，则返回 false
    }

    //--------------------------------------------------------------------------
    // 处理方向键事件，查看当前焦点控件是否处理了该事件
    bool on_arrow_keys(bool left, bool right, bool down, bool up)
    {
        if(m_cur_ctrl >= 0)
        {
            // 如果有当前焦点控件，则交由该控件处理方向键事件
            return m_ctrl[m_cur_ctrl]->on_arrow_keys(left, right, down, up);
        }
        return false; // 如果没有当前焦点控件或者控件未处理该事件，则返回 false
    }

    //--------------------------------------------------------------------------
    // 设置当前焦点控件，根据坐标 (x, y) 确定焦点控件
    bool set_cur(double x, double y)
    {
        unsigned i;
        for(i = 0; i < m_num_ctrl; i++)
        {
            // 如果找到坐标在控件范围内的控件
            if(m_ctrl[i]->in_rect(x, y)) 
            {
                // 如果该控件与当前焦点控件不同，则更新当前焦点控件
                if(m_cur_ctrl != int(i))
                {
                    m_cur_ctrl = i;
                    return true; // 返回 true 表示更新了当前焦点控件
                }
                return false; // 如果焦点控件未变化，则返回 false
            }
        }
        // 如果没有找到控件包含此坐标，则将当前焦点控件设置为 -1
        if(m_cur_ctrl != -1)
        {
            m_cur_ctrl = -1;
            return true; // 返回 true 表示当前焦点控件更新为无
        }
        return false; // 如果当前焦点控件本来就是无，则返回 false
    }

private:
    ctrl*         m_ctrl[max_ctrl]; // 控件指针数组，存储指向控件对象的指针
    unsigned      m_num_ctrl;       // 当前存储的控件数量
    int           m_cur_ctrl;       // 当前焦点控件的索引，-1 表示无焦点控件
};
    //---------------------------------------------------------platform_support
    // This class is a base one to the application classes. It can be used 
    // as follows:
    //
    //  class the_application : public agg::platform_support
    //  {
    //  public:
    //      the_application(unsigned bpp, bool flip_y) :
    //          platform_support(bpp, flip_y) 
    //      . . .
    //
    //      //override stuff . . .
    //      virtual void on_init()
    //      {
    //         . . . // Initialization code specific to the application
    //      }
    //
    //      virtual void on_draw()
    //      {
    //          . . . // Drawing code specific to the application
    //      }
    //
    //      virtual void on_resize(int sx, int sy)
    //      {
    //          . . . // Resize handling code specific to the application
    //      }
    //      // . . . and so on, see virtual functions
    //
    //
    //      //any your own stuff . . .
    //  };
    //
    //
    //  int agg_main(int argc, char* argv[])
    //  {
    //      the_application app(pix_format_rgb24, true);
    //      app.caption("AGG Example. Lion");
    //
    //      if(app.init(500, 400, agg::window_resize))
    //      {
    //          return app.run();
    //      }
    //      return 1;
    //  }
    //
    // The reason to have agg_main() instead of just main() is that SDL
    // for Windows requires including SDL.h if you define main(). Since
    // the demo applications cannot rely on any platform/library specific
    // stuff it's impossible to include SDL.h into the application files.
    // The demo applications are simple and their use is restricted, so, 
    // this approach is quite reasonable.
    // 
    // This class encapsulates platform-specific support functionalities
    // for applications, including graphical rendering and user interface controls.
    class platform_support
    {
    public:
        platform_specific* m_specific; // Pointer to platform-specific data
        ctrl_container m_ctrls; // Container for UI controls
    
        // Sorry, I'm too tired to describe the private 
        // data members. See the implementations for different
        // platforms for details.
    private:
        platform_support(const platform_support&); // Private copy constructor
        const platform_support& operator = (const platform_support&); // Private assignment operator
    
        pix_format_e     m_format; // Pixel format
        unsigned         m_bpp; // Bits per pixel
        rendering_buffer m_rbuf_window; // Rendering buffer for the main window
        rendering_buffer m_rbuf_img[max_images]; // Array of rendering buffers for images
        unsigned         m_window_flags; // Flags for window behavior
        bool             m_wait_mode; // Wait mode flag
        bool             m_flip_y; // Flag for flipping Y axis
        char             m_caption[256]; // Window caption
        int              m_initial_width; // Initial width of the window
        int              m_initial_height; // Initial height of the window
        trans_affine     m_resize_mtx; // Transformation matrix for resizing
    };
}

这行代码结束了一个代码块或函数的定义。


#endif

这行代码通常用于条件编译，在预处理阶段判断条件是否成立，如果成立则包含或跳过后续代码。

这两行代码看起来是一种条件编译的结构，根据预处理器中的条件来确定是否包含在最终的编译结果中。
```