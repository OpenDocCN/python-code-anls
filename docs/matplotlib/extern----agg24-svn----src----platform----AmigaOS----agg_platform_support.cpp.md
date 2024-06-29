# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\platform\AmigaOS\agg_platform_support.cpp`

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

#include "platform/agg_platform_support.h"
#include "util/agg_color_conv_rgb8.h"

#include <sys/time.h>
#include <cstring>

#include <classes/requester.h>
#include <classes/window.h>
#include <datatypes/pictureclass.h>
#include <proto/exec.h>
#include <proto/datatypes.h>
#include <proto/dos.h>
#include <proto/graphics.h>
#include <proto/intuition.h>
#include <proto/keymap.h>
#include <proto/Picasso96API.h>
#include <proto/utility.h>

// 初始化库基础变量为 null
Library* DataTypesBase = 0;
Library* GraphicsBase = 0;
Library* IntuitionBase = 0;
Library* KeymapBase = 0;
Library* P96Base = 0;

// 初始化接口变量为 null
DataTypesIFace* IDataTypes = 0;
GraphicsIFace* IGraphics = 0;
IntuitionIFace* IIntuition = 0;
KeymapIFace* IKeymap = 0;
P96IFace* IP96 = 0;

// 初始化类变量为 null
Class* RequesterClass = 0;
Class* WindowClass = 0;

// 命名空间 agg 中的函数声明
namespace agg
{
    void handle_idcmp(Hook* hook, APTR win, IntuiMessage* msg);

    //------------------------------------------------------------------------
    // 平台特定实现类
    class platform_specific
    {
    public:
        // 构造函数，接收平台支持对象、像素格式和是否翻转 Y 轴作为参数
        platform_specific(platform_support& support, pix_format_e format,
            bool flip_y);
        // 析构函数
        ~platform_specific();
        // 处理输入事件的方法
        bool handle_input();
        // 加载图像文件到渲染缓冲区的方法
        bool load_img(const char* file, unsigned idx, rendering_buffer* rbuf);
        // 创建指定大小的图像到渲染缓冲区的方法
        bool create_img(unsigned idx, rendering_buffer* rbuf, unsigned width,
            unsigned height);
        // 创建位图的方法
        bool make_bitmap();
    public:
        platform_support& m_support; // 平台支持对象的引用
        RGBFTYPE m_ftype;            // RGBF 类型
        pix_format_e m_format;       // 像素格式枚举
        unsigned m_bpp;              // 每像素位数
        BitMap* m_bitmap;            // 位图指针
        bool m_flip_y;               // 是否翻转 Y 轴
        uint16 m_width;              // 图像宽度
        uint16 m_height;             // 图像高度
        APTR m_window_obj;           // 窗口对象指针
        Window* m_window;            // 窗口指针
        Hook* m_idcmp_hook;          // IDCMP 钩子指针
        unsigned m_input_flags;      // 输入标志
        bool m_dragging;             // 是否拖动中
        double m_start_time;         // 启动时间
        uint16 m_last_key;           // 上一个按键
        BitMap* m_img_bitmaps[platform_support::max_images]; // 图像位图数组
    };

    //------------------------------------------------------------------------
    // 构造函数，初始化 platform_specific 对象
    platform_specific::platform_specific(platform_support& support,
        pix_format_e format, bool flip_y) :
        // 初始化成员变量
        m_support(support),                       // 使用传入的 support 对象初始化 m_support 成员变量
        m_ftype(RGBFB_NONE),                      // 初始化 m_ftype 为 RGBFB_NONE
        m_format(format),                         // 使用传入的 format 参数初始化 m_format 成员变量
        m_bpp(0),                                 // 初始化 m_bpp 为 0
        m_bitmap(0),                              // 初始化 m_bitmap 为 0
        m_flip_y(flip_y),                         // 使用传入的 flip_y 参数初始化 m_flip_y 成员变量
        m_width(0),                               // 初始化 m_width 为 0
        m_height(0),                              // 初始化 m_height 为 0
        m_window_obj(0),                          // 初始化 m_window_obj 为 0
        m_window(0),                              // 初始化 m_window 为 0
        m_idcmp_hook(0),                          // 初始化 m_idcmp_hook 为 0
        m_input_flags(0),                         // 初始化 m_input_flags 为 0
        m_dragging(false),                        // 初始化 m_dragging 为 false
        m_start_time(0.0),                        // 初始化 m_start_time 为 0.0
        m_last_key(0)                             // 初始化 m_last_key 为 0
    {
        // 根据像素格式 format 设置对应的颜色类型和位深度
        switch ( format )
        {
        case pix_format_gray8:
            // 灰度图像格式不支持
            break;
        case pix_format_rgb555:
            m_ftype = RGBFB_R5G5B5;                // 设置 m_ftype 为 RGBFB_R5G5B5
            m_bpp = 15;                            // 设置 m_bpp 为 15
            break;
        case pix_format_rgb565:
            m_ftype = RGBFB_R5G6B5;                // 设置 m_ftype 为 RGBFB_R5G6B5
            m_bpp = 16;                            // 设置 m_bpp 为 16
            break;
        case pix_format_rgb24:
            m_ftype = RGBFB_R8G8B8;                // 设置 m_ftype 为 RGBFB_R8G8B8
            m_bpp = 24;                            // 设置 m_bpp 为 24
            break;
        case pix_format_bgr24:
            m_ftype = RGBFB_B8G8R8;                // 设置 m_ftype 为 RGBFB_B8G8R8
            m_bpp = 24;                            // 设置 m_bpp 为 24
            break;
        case pix_format_bgra32:
            m_ftype = RGBFB_B8G8R8A8;              // 设置 m_ftype 为 RGBFB_B8G8R8A8
            m_bpp = 32;                            // 设置 m_bpp 为 32
            break;
        case pix_format_abgr32:
            m_ftype = RGBFB_A8B8G8R8;              // 设置 m_ftype 为 RGBFB_A8B8G8R8
            m_bpp = 32;                            // 设置 m_bpp 为 32
            break;
        case pix_format_argb32:
            m_ftype = RGBFB_A8R8G8B8;              // 设置 m_ftype 为 RGBFB_A8R8G8B8
            m_bpp = 32;                            // 设置 m_bpp 为 32
            break;
        case pix_format_rgba32:
            m_ftype = RGBFB_R8G8B8A8;              // 设置 m_ftype 为 RGBFB_R8G8B8A8
            m_bpp = 32;                            // 设置 m_bpp 为 32
            break;
        }
    
        // 初始化 m_img_bitmaps 数组，将所有元素设置为 0
        for ( unsigned i = 0; i < platform_support::max_images; ++i )
        {
            m_img_bitmaps[i] = 0;
        }
    }
    
    //------------------------------------------------------------------------
    // 析构函数，释放 platform_specific 对象占用的资源
    platform_specific::~platform_specific()
    {
        IIntuition->DisposeObject(m_window_obj);   // 释放窗口对象占用的资源
    
        IP96->p96FreeBitMap(m_bitmap);             // 释放位图对象占用的资源
    
        // 释放所有图像位图对象占用的资源
        for ( unsigned i = 0; i < platform_support::max_images; ++i )
        {
            IP96->p96FreeBitMap(m_img_bitmaps[i]);
        }
    
        // 如果存在 idcmp 钩子对象，释放该对象占用的系统资源
        if ( m_idcmp_hook != 0 )
        {
            IExec->FreeSysObject(ASOT_HOOK, m_idcmp_hook);
        }
    }
    {
        // 初始化变量，code 是一个 16 位整数，用于存储返回的消息代码，result 是一个 32 位无符号整数，用于存储方法调用的返回结果
        int16 code = 0;
        uint32 result = 0;
        // 将 m_window_obj 强制转换为 Object* 类型的指针
        Object* obj = reinterpret_cast<Object*>(m_window_obj);
    
        // 使用 Intuition 接口调用 IDoMethod 方法，处理窗口对象的输入消息，直到接收到 WMHI_LASTMSG 消息为止
        while ( (result = IIntuition->IDoMethod(obj, WM_HANDLEINPUT, &code)) != WMHI_LASTMSG )
        {
            // 根据返回的消息类型进行处理
            switch ( result & WMHI_CLASSMASK )
            {
            // 如果是关闭窗口的消息
            case WMHI_CLOSEWINDOW:
                // 返回 true 表示窗口应该关闭
                return true;
                break;
            // 如果是定时器触发的消息
            case WMHI_INTUITICK:
                // 如果不处于等待模式，则调用 m_support 的 on_idle 方法
                if ( !m_support.wait_mode() )
                {
                    m_support.on_idle();
                }
                break;
            // 如果是窗口大小改变的消息
            case WMHI_NEWSIZE:
                // 调用 make_bitmap 方法重新生成位图
                if ( make_bitmap() )
                {
                    // 根据新的宽度和高度调整仿射变换和重绘
                    m_support.trans_affine_resizing(m_width, m_height);
                    m_support.on_resize(m_width, m_height);
                    m_support.force_redraw();
                }
                break;
            }
        }
    
        // 返回 false 表示窗口不需要关闭
        return false;
    }        
    
    //------------------------------------------------------------------------
    bool platform_specific::load_img(const char* file, unsigned idx,
        rendering_buffer* rbuf)
    {
        // 实现在具体平台上加载图片的功能，根据文件名 file 和索引 idx，将图像数据加载到 rbuf 所指定的渲染缓冲区中
    }
    
    //------------------------------------------------------------------------
    bool platform_specific::create_img(unsigned idx, rendering_buffer* rbuf,
        unsigned width, unsigned height)
    {
        // 如果已经存在指定索引 idx 的位图，则释放它
        if ( m_img_bitmaps[idx] != 0 )
        {
            IP96->p96FreeBitMap(m_img_bitmaps[idx]);
            m_img_bitmaps[idx] = 0;
        }
    
        // 使用 P96 接口分配一个新的位图，根据指定的宽度、高度、位深度等参数
        m_img_bitmaps[idx] = IP96->p96AllocBitMap(width, height,
            m_bpp, BMF_USERPRIVATE, m_bitmap, m_ftype);
        // 如果成功分配位图
        if ( m_img_bitmaps[idx] != 0 )
        {
            // 获取位图的内存地址和字节行数
            int8u* buf = reinterpret_cast<int8u*>(
                IP96->p96GetBitMapAttr(m_img_bitmaps[idx], P96BMA_MEMORY));
            int bpr = IP96->p96GetBitMapAttr(m_img_bitmaps[idx], P96BMA_BYTESPERROW);
            // 根据是否需要翻转 Y 轴计算跨距
            int stride = (m_flip_y) ? -bpr : bpr;
    
            // 将位图的数据附加到渲染缓冲区 rbuf 中
            rbuf->attach(buf, width, height, stride);
    
            return true;
        }
    
        // 分配位图失败，返回 false
        return false;
    }
    
    //------------------------------------------------------------------------
    bool platform_specific::make_bitmap()
    {
        // 实现在具体平台上生成位图的功能，用于处理窗口大小改变后的重绘等操作
    }
    {
        // 定义并初始化窗口的宽度和高度为零
        uint32 width = 0;
        uint32 height = 0;
        
        // 获取窗口的内部宽度和高度属性
        IIntuition->GetWindowAttrs(m_window,
            WA_InnerWidth, &width,
            WA_InnerHeight, &height,
            TAG_END);
        
        // 使用图形引擎的接口分配一个位图，尺寸为窗口的宽度和高度，像素位数为 m_bpp
        // BMF_USERPRIVATE 表示用户私有位图，BMF_CLEAR 表示清空位图内容
        BitMap* bm = IP96->p96AllocBitMap(width, height, m_bpp,
            BMF_USERPRIVATE|BMF_CLEAR, 0, m_ftype);
        
        // 如果位图分配失败，则返回 false
        if ( bm == 0 )
        {
            return false;
        }
        
        // 获取位图的内存地址，并将其解释为 int8u* 类型的缓冲区
        int8u* buf = reinterpret_cast<int8u*>(
            IP96->p96GetBitMapAttr(bm, P96BMA_MEMORY));
        
        // 获取位图每行字节数（Bytes Per Row）
        int bpr = IP96->p96GetBitMapAttr(bm, P96BMA_BYTESPERROW);
        
        // 根据 m_flip_y 的值确定位图的扫描行步长
        // 如果 m_flip_y 为真，则步长为负值（向上扫描）
        int stride = (m_flip_y) ? -bpr : bpr;
        
        // 将图形引擎的渲染缓冲区与新分配的位图关联
        m_support.rbuf_window().attach(buf, width, height, stride);
        
        // 如果之前有位图已分配，则释放它
        if ( m_bitmap != 0 )
        {
            IP96->p96FreeBitMap(m_bitmap);
            m_bitmap = 0;
        }
        
        // 更新当前位图、宽度和高度的成员变量
        m_bitmap = bm;
        m_width = width;
        m_height = height;
        
        // 返回操作成功
        return true;
    }
    
    //------------------------------------------------------------------------
    platform_support::platform_support(pix_format_e format, bool flip_y) :
        // 初始化平台相关的特定对象和成员变量
        m_specific(new platform_specific(*this, format, flip_y)),
        m_format(format),
        m_bpp(m_specific->m_bpp),
        m_window_flags(0),
        m_wait_mode(true),
        m_flip_y(flip_y),
        m_initial_width(10),
        m_initial_height(10)
    {
        // 设置默认窗口标题
        std::strncpy(m_caption, "Anti-Grain Geometry", 256);
    }
    
    //------------------------------------------------------------------------
    platform_support::~platform_support()
    {
        // 析构函数，释放平台特定对象的内存
        delete m_specific;
    }
    
    //------------------------------------------------------------------------
    void platform_support::caption(const char* cap)
    {
        // 设置窗口标题为给定的字符串 cap
        std::strncpy(m_caption, cap, 256);
        
        // 如果窗口已创建，则更新窗口的标题
        if ( m_specific->m_window != 0 )
        {
            const char* ignore = reinterpret_cast<const char*>(-1);
            IIntuition->SetWindowAttr(m_specific->m_window,
                WA_Title, m_caption, sizeof(char*));
        }
    }
    
    //------------------------------------------------------------------------
    void platform_support::start_timer()
    {
        // 获取当前时间，作为计时器的起始时间
        timeval tv;
        gettimeofday(&tv, 0);
        m_specific->m_start_time = tv.tv_secs + tv.tv_micro/1e6;
    }
    
    //------------------------------------------------------------------------
    double platform_support::elapsed_time() const
    {
        // 获取当前时间，计算自起始时间以来的经过时间（毫秒）
        timeval tv;
        gettimeofday(&tv, 0);
        double end_time = tv.tv_secs + tv.tv_micro/1e6;
    
        double elapsed_seconds = end_time - m_specific->m_start_time;
        double elapsed_millis = elapsed_seconds * 1e3;
    
        return elapsed_millis;
    }
    
    //------------------------------------------------------------------------
    void* platform_support::raw_display_handler()
    {
        // 返回空指针，表示不可用
        return 0;    // Not available.
    }
    
    //------------------------------------------------------------------------
    void platform_support::message(const char* msg)
    {
        // 创建一个新的请求对象，使用Intuition库中的RequesterClass
        APTR req = IIntuition->NewObject(RequesterClass, 0,
            REQ_TitleText, "Anti-Grain Geometry",
            REQ_Image, REQIMAGE_INFO,
            REQ_BodyText, msg,
            REQ_GadgetText, "_Ok",
            TAG_END);
        
        // 如果请求对象创建失败，打印消息并返回
        if ( req == 0 )
        {
            IDOS->Printf("Message: %s\n", msg);
            return;
        }
    
        // 准备一个OpenRequest消息对象
        orRequest reqmsg;
        reqmsg.MethodID = RM_OPENREQ;
        reqmsg.or_Attrs = 0;
        reqmsg.or_Window = m_specific->m_window;
        reqmsg.or_Screen = 0;
        
        // 发送OpenRequest消息给请求对象处理
        IIntuition->IDoMethodA(reinterpret_cast<Object*>(req),
            reinterpret_cast<Msg>(&reqmsg));
        
        // 处置（释放）请求对象
        IIntuition->DisposeObject(req);
    }
    
    //------------------------------------------------------------------------
    // 初始化平台支持，设置宽度、高度和标志
    bool platform_support::init(unsigned width, unsigned height,
        unsigned flags)
    {
        // 检查特定模式是否被支持，若不支持则打印消息并返回false
        if( m_specific->m_ftype == RGBFB_NONE )
        {
            message("Unsupported mode requested.");
            return false;
        }
    
        // 将传入的窗口标志存储到成员变量中
        m_window_flags = flags;
    
        // 分配并设置IDCMP挂钩，关联到当前对象，若分配失败则返回false
        m_specific->m_idcmp_hook = reinterpret_cast<Hook*>(
            IExec->AllocSysObjectTags(ASOT_HOOK,
                ASOHOOK_Entry, handle_idcmp,
                ASOHOOK_Data, this,
                TAG_END));
        if ( m_specific->m_idcmp_hook == 0 )
        {
            return false;
        }
    
        // 创建新窗口对象，设置其属性和行为，并关联IDCMP挂钩
        m_specific->m_window_obj = IIntuition->NewObject(WindowClass, 0,
                WA_Title, m_caption,
                WA_AutoAdjustDClip, TRUE,
                WA_InnerWidth, width,
                WA_InnerHeight, height,
                WA_Activate, TRUE,
                WA_SmartRefresh, TRUE,
                WA_NoCareRefresh, TRUE,
                WA_CloseGadget, TRUE,
                WA_DepthGadget, TRUE,
                WA_SizeGadget, (flags & agg::window_resize) ? TRUE : FALSE,
                WA_DragBar, TRUE,
                WA_AutoAdjust, TRUE,
                WA_ReportMouse, TRUE,
                WA_RMBTrap, TRUE,
                WA_MouseQueue, 1,
                WA_IDCMP,
                    IDCMP_NEWSIZE |
                    IDCMP_MOUSEBUTTONS |
                    IDCMP_MOUSEMOVE |
                    IDCMP_RAWKEY |
                    IDCMP_INTUITICKS,
                WINDOW_IDCMPHook, m_specific->m_idcmp_hook,
                WINDOW_IDCMPHookBits,
                    IDCMP_MOUSEBUTTONS |
                    IDCMP_MOUSEMOVE |
                    IDCMP_RAWKEY,
                TAG_END);
        if ( m_specific->m_window_obj == 0 )
        {
            return false;
        }
    
        // 从窗口对象中获取底层的Window指针
        Object* obj = reinterpret_cast<Object*>(m_specific->m_window_obj);
        // 使用IDoMethod打开窗口，获取窗口的指针，若失败则返回false
        m_specific->m_window =
            reinterpret_cast<Window*>(IIntuition->IDoMethod(obj, WM_OPEN));
        if ( m_specific->m_window == 0 )
        {
            return false;
        }
    
        // 获取当前窗口的色彩格式，若不支持特定格式则打印消息并返回false
        RGBFTYPE ftype = static_cast<RGBFTYPE>(IP96->p96GetBitMapAttr(
            m_specific->m_window->RPort->BitMap, P96BMA_RGBFORMAT));
    
        switch ( ftype )
        {
        case RGBFB_A8R8G8B8:
        case RGBFB_B8G8R8A8:
        case RGBFB_R5G6B5PC:
            break;
        default:
            message("Unsupported screen mode.\n");
            return false;
        }
    
        // 创建位图对象，若失败则返回false
        if ( !m_specific->make_bitmap() )
        {
            return false;
        }
    
        // 存储初始化时的窗口尺寸
        m_initial_width = width;
        m_initial_height = height;
    
        // 调用初始化回调函数
        on_init();
        // 调整窗口大小，并执行相应操作
        on_resize(width, height);
        // 强制重绘窗口内容
        force_redraw();
    
        // 返回操作成功
        return true;
    }
    {
        // 初始化一个用于窗口信号掩码的变量
        uint32 window_mask = 0;
        // 获取窗口对象的信号掩码属性，并将其存储在 window_mask 中
        IIntuition->GetAttr(WINDOW_SigMask, m_specific->m_window_obj,
            &window_mask);
        // 创建一个等待信号的掩码，包括窗口信号和中断信号（Ctrl+C）
        uint32 wait_mask = window_mask | SIGBREAKF_CTRL_C;
    
        // 循环等待信号，直到完成标志变为 true
        bool done = false;
        while ( !done )
        {
            // 等待指定信号的发生
            uint32 sig_mask = IExec->Wait(wait_mask);
            // 如果收到中断信号（Ctrl+C），设置完成标志为 true
            if ( sig_mask & SIGBREAKF_CTRL_C )
            {
                done = true;
            }
            // 否则调用特定平台的处理输入方法，并根据返回值设置完成标志
            else
            {
                done = m_specific->handle_input();
            }
        }
    
        // 返回成功标志
        return 0;
    }
    
    //------------------------------------------------------------------------
    // 返回 ".bmp" 作为平台支持的图像文件扩展名
    const char* platform_support::img_ext() const
    {
        return ".bmp";
    }
    
    //------------------------------------------------------------------------
    // 返回传入的文件名，作为平台支持的完整文件名
    const char* platform_support::full_file_name(const char* file_name)
    {
        return file_name;
    }
    
    //------------------------------------------------------------------------
    // 加载图像文件到指定索引处，如果文件名不以 ".bmp" 结尾，则添加该后缀
    bool platform_support::load_img(unsigned idx, const char* file)
    {
        if ( idx < max_images )
        {
            // 复制文件名到本地缓冲区 fn，并确保以 ".bmp" 结尾
            static char fn[1024];
            std::strncpy(fn, file, 1024);
            int len = std::strlen(fn);
            if ( len < 4 || std::strcmp(fn + len - 4, ".bmp") != 0 )
            {
                std::strncat(fn, ".bmp", 1024);
            }
    
            // 调用特定平台的加载图像方法，加载文件 fn 到索引 idx 处的图像缓冲区
            return m_specific->load_img(fn, idx, &m_rbuf_img[idx]);
        }
    
        // 如果索引超出最大图像数，返回加载失败
        return false;
    }
    
    //------------------------------------------------------------------------
    // 不支持保存图像的操作，打印一条消息并返回失败
    bool platform_support::save_img(unsigned idx, const char* file)
    {
        message("Not supported");
        return false;
    }
    
    //------------------------------------------------------------------------
    // 创建指定索引处的图像，如果宽度或高度为零，则使用平台特定的默认值
    bool platform_support::create_img(unsigned idx, unsigned width, unsigned height)
    {
        if ( idx < max_images )
        {
            // 如果宽度为零，使用平台特定的默认宽度
            if ( width == 0 )
            {
                width = m_specific->m_width;
            }
            // 如果高度为零，使用平台特定的默认高度
            if ( height == 0 )
            {
                height = m_specific->m_height;
            }
    
            // 调用特定平台的创建图像方法，创建索引 idx 处的图像，并指定宽度和高度
            return m_specific->create_img(idx, &m_rbuf_img[idx], width, height);
        }
    
        // 如果索引超出最大图像数，返回创建失败
        return false;
    }
    
    //------------------------------------------------------------------------
    // 强制重新绘制窗口，调用 on_draw() 和 update_window() 方法
    void platform_support::force_redraw()
    {
        // 调用绘制方法
        on_draw();
        // 更新窗口显示
        update_window();
    }
    
    //------------------------------------------------------------------------
    // 更新窗口显示，自动进行颜色转换
    void platform_support::update_window()
    {
        // 注意：此函数会自动进行颜色转换
        // 使用 BltBitMapRastPort 函数将位图渲染到窗口的 RastPort 中
        IGraphics->BltBitMapRastPort(m_specific->m_bitmap, 0, 0,
            m_specific->m_window->RPort, m_specific->m_window->BorderLeft,
            m_specific->m_window->BorderTop, m_specific->m_width,
            m_specific->m_height, ABC|ABNC);
    }
    
    //------------------------------------------------------------------------
    // 初始化操作，空函数，无需执行任何操作
    void platform_support::on_init() {}
    // 定义 platform_support 类的 on_resize 方法，处理窗口大小改变事件
    void platform_support::on_resize(int sx, int sy) {}

    // 定义 platform_support 类的 on_idle 方法，处理空闲状态下的事件
    void platform_support::on_idle() {}

    // 定义 platform_support 类的 on_mouse_move 方法，处理鼠标移动事件
    void platform_support::on_mouse_move(int x, int y, unsigned flags) {}

    // 定义 platform_support 类的 on_mouse_button_down 方法，处理鼠标按下事件
    void platform_support::on_mouse_button_down(int x, int y, unsigned flags) {}

    // 定义 platform_support 类的 on_mouse_button_up 方法，处理鼠标释放事件
    void platform_support::on_mouse_button_up(int x, int y, unsigned flags) {}

    // 定义 platform_support 类的 on_key 方法，处理键盘按键事件
    void platform_support::on_key(int x, int y, unsigned key, unsigned flags) {}

    // 定义 platform_support 类的 on_ctrl_change 方法，处理控制键状态改变事件
    void platform_support::on_ctrl_change() {}

    // 定义 platform_support 类的 on_draw 方法，处理绘制事件
    void platform_support::on_draw() {}

    // 定义 platform_support 类的 on_post_draw 方法，处理绘制后事件
    void platform_support::on_post_draw(void* raw_handler) {}

    //------------------------------------------------------------------------
    // 处理 Intuition IDCMP 消息的函数，接受一个 Hook 对象、一个 APTR 对象和一个 IntuiMessage 指针作为参数
    void handle_idcmp(Hook* hook, APTR obj, IntuiMessage* msg)
    {
        // 实现内容未提供，略
    }
//----------------------------------------------------------------------------
int agg_main(int argc, char* argv[]);
bool open_libs();
void close_libs();

//----------------------------------------------------------------------------
bool open_libs()
{
    // 打开 datatypes.library 库
    DataTypesBase = IExec->OpenLibrary("datatypes.library", 51);
    // 打开 graphics.library 库
    GraphicsBase = IExec->OpenLibrary("graphics.library", 51);
    // 打开 intuition.library 库
    IntuitionBase = IExec->OpenLibrary("intuition.library", 51);
    // 打开 keymap.library 库
    KeymapBase = IExec->OpenLibrary("keymap.library", 51);
    // 打开 Picasso96API.library 库
    P96Base = IExec->OpenLibrary("Picasso96API.library", 2);

    // 获取 datatypes.library 的接口
    IDataTypes = reinterpret_cast<DataTypesIFace*>(
        IExec->GetInterface(DataTypesBase, "main", 1, 0));
    // 获取 graphics.library 的接口
    IGraphics = reinterpret_cast<GraphicsIFace*>(
        IExec->GetInterface(GraphicsBase, "main", 1, 0));
    // 获取 intuition.library 的接口
    IIntuition = reinterpret_cast<IntuitionIFace*>(
        IExec->GetInterface(IntuitionBase, "main", 1, 0));
    // 获取 keymap.library 的接口
    IKeymap = reinterpret_cast<KeymapIFace*>(
        IExec->GetInterface(KeymapBase, "main", 1, 0));
    // 获取 Picasso96API.library 的接口
    IP96 = reinterpret_cast<P96IFace*>(
        IExec->GetInterface(P96Base, "main", 1, 0));

    // 检查是否所有的接口都成功获取
    if ( IDataTypes == 0 ||
         IGraphics == 0 ||
         IIntuition == 0 ||
         IKeymap == 0 ||
         IP96 == 0 )
    {
        // 如果有接口获取失败，关闭所有库并返回失败
        close_libs();
        return false;
    }
    else
    {
        // 接口获取成功，返回成功
        return true;
    }
}

//----------------------------------------------------------------------------
void close_libs()
{
    // 释放 Picasso96API.library 的接口
    IExec->DropInterface(reinterpret_cast<Interface*>(IP96));
    // 释放 keymap.library 的接口
    IExec->DropInterface(reinterpret_cast<Interface*>(IKeymap));
    // 释放 intuition.library 的接口
    IExec->DropInterface(reinterpret_cast<Interface*>(IIntuition));
    // 释放 graphics.library 的接口
    IExec->DropInterface(reinterpret_cast<Interface*>(IGraphics));
    // 释放 datatypes.library 的接口
    IExec->DropInterface(reinterpret_cast<Interface*>(IDataTypes));

    // 关闭 Picasso96API.library 库
    IExec->CloseLibrary(P96Base);
    // 关闭 keymap.library 库
    IExec->CloseLibrary(KeymapBase);
    // 关闭 intuition.library 库
    IExec->CloseLibrary(IntuitionBase);
    // 关闭 graphics.library 库
    IExec->CloseLibrary(GraphicsBase);
    // 关闭 datatypes.library 库
    IExec->CloseLibrary(DataTypesBase);
}

//----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // 尝试打开所有必要的库
    if ( !open_libs() )  {
        // 如果打开库失败，打印错误信息并返回错误码
        IDOS->Printf("Can't open libraries.\n");
        return -1;
    }

    // 打开 requester.class 和 window.class 类
    ClassLibrary* requester =
        IIntuition->OpenClass("requester.class", 51, &RequesterClass);
    ClassLibrary* window =
        IIntuition->OpenClass("window.class", 51, &WindowClass);
    // 检查类是否成功打开
    if ( requester == 0 || window == 0 )
    {
        // 如果类打开失败，打印错误信息，关闭类和库，并返回错误码
        IDOS->Printf("Can't open classes.\n");
        IIntuition->CloseClass(requester);
        IIntuition->CloseClass(window);
        close_libs();
        return -1;
    }

    // 调用主功能函数 agg_main，并获取返回码
    int rc = agg_main(argc, argv);

    // 关闭 window.class 和 requester.class 类
    IIntuition->CloseClass(window);
    IIntuition->CloseClass(requester);
    // 关闭所有库
    close_libs();

    // 返回主功能函数的返回码
    return rc;
}
```