# `D:\src\scipysrc\matplotlib\extern\agg24-svn\src\platform\BeOS\agg_platform_support.cpp`

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
// Contact: superstippi@gmx.de
//----------------------------------------------------------------------------
//
// class platform_support
//
//----------------------------------------------------------------------------

#include <new>
#include <stdio.h>

#include <Alert.h>
#include <Application.h>
#include <Bitmap.h>
#include <Message.h>
#include <MessageRunner.h>
#include <Messenger.h>
#include <Path.h>
#include <Roster.h>
#include <TranslationUtils.h>
#include <View.h>
#include <Window.h>

#include <string.h>
#include "platform/agg_platform_support.h"
#include "util/agg_color_conv_rgb8.h"

using std::nothrow;


static void
attach_buffer_to_BBitmap(agg::rendering_buffer& buffer, BBitmap* bitmap, bool flipY)
{
    // 获取 BBitmap 对象的像素数据指针
    uint8* bits = (uint8*)bitmap->Bits();
    // 获取 BBitmap 对象的宽度和高度
    uint32 width = bitmap->Bounds().IntegerWidth() + 1;
    uint32 height = bitmap->Bounds().IntegerHeight() + 1;
    // 获取 BBitmap 对象的每行字节数
    int32 bpr = bitmap->BytesPerRow();
    // 如果 flipY 参数为 true，则需要翻转图像，调整行字节数
    if (flipY) {
        // XXX: 为什么我不需要执行这个？！？
        //        bits += bpr * (height - 1);
        // 翻转行字节数
        bpr = -bpr;
    }
    // 将渲染缓冲区与 BBitmap 对象的像素数据关联起来
    buffer.attach(bits, width, height, bpr);
}


static color_space
pix_format_to_color_space(agg::pix_format_e format)
{
    // 初始化位图颜色空间为默认值 B_NO_COLOR_SPACE
    color_space bitmapFormat = B_NO_COLOR_SPACE;
    // 根据 AGG 的像素格式转换为 BeOS/Haiku 的颜色空间
    switch (format) {
        case agg::pix_format_rgb555:
            bitmapFormat = B_RGB15;
            break;

        case agg::pix_format_rgb565:
            bitmapFormat = B_RGB16;
            break;

        case agg::pix_format_rgb24:
        case agg::pix_format_bgr24:
            bitmapFormat = B_RGB24;
            break;

        case agg::pix_format_rgba32:
        case agg::pix_format_argb32:
        case agg::pix_format_abgr32:
        case agg::pix_format_bgra32:
            bitmapFormat = B_RGBA32;
            break;
    }
    // 返回转换后的颜色空间
    return bitmapFormat;
}


// #pragma mark -


class AGGView : public BView {
 public:
    // 构造函数，初始化 AGGView
    AGGView(BRect frame, agg::platform_support* agg,
            agg::pix_format_e format, bool flipY);
    // 虚析构函数，清理 AGGView 对象
    virtual ~AGGView();

    // 覆盖 AttachedToWindow 函数
    virtual void AttachedToWindow();
    // 覆盖 DetachedFromWindow 函数
    virtual void DetachedFromWindow();

    // 覆盖 MessageReceived 函数，处理消息
    virtual void MessageReceived(BMessage* message);
    // 覆盖 Draw 函数，绘制视图内容
    virtual void Draw(BRect updateRect);
    // 覆盖 FrameResized 函数，处理窗口大小改变事件
    virtual void FrameResized(float width, float height);

    // 覆盖 KeyDown 函数，处理按键事件
    virtual void KeyDown(const char* bytes, int32 numBytes);

    // 覆盖 MouseDown 函数，处理鼠标点击事件
    virtual void MouseDown(BPoint where);
    // 处理鼠标移动事件的虚拟函数，提供鼠标位置、状态信息
    virtual void            MouseMoved(BPoint where, uint32 transit,
                               const BMessage* dragMesage);

    // 处理鼠标抬起事件的虚拟函数，提供鼠标位置
    virtual void            MouseUp(BPoint where);

            // 返回当前使用的位图指针
            BBitmap*        Bitmap() const;

            // 返回最后按下的键的 ASCII 码
            uint8           LastKeyDown() const;

            // 返回当前鼠标按钮的状态
            uint32          MouseButtons();

            // 更新视图
            void            Update();

            // 强制重绘视图
            void            ForceRedraw();

            // 返回当前按下的键的标志位
            unsigned        GetKeyFlags();

 private:
    // 当前使用的位图对象指针
    BBitmap*                fBitmap;

    // AGG 像素格式
    agg::pix_format_e       fFormat;

    // 是否翻转 Y 轴
    bool                    fFlipY;

    // AGG 平台支持对象指针
    agg::platform_support*  fAGG;

    // 当前鼠标按钮的状态
    uint32                  fMouseButtons;

    // 鼠标的 X 坐标
    int32                   fMouseX;

    // 鼠标的 Y 坐标
    int32                   fMouseY;

    // 最后按下的键的 ASCII 码
    uint8                   fLastKeyDown;

    // 是否需要强制重绘视图
    bool                    fRedraw;

    // 消息运行器对象指针，用于周期性任务
    BMessageRunner*         fPulse;

    // 上次脉冲的时间戳
    bigtime_t               fLastPulse;

    // 是否启用脉冲事件
    bool                    fEnableTicks;
};

AGGView::AGGView(BRect frame,
                 agg::platform_support* agg,
                 agg::pix_format_e format,
                 bool flipY)
    : BView(frame, "AGG View", B_FOLLOW_ALL,
            B_FRAME_EVENTS | B_WILL_DRAW),
      fFormat(format),
      fFlipY(flipY),
      fAGG(agg),  // 初始化 AGG 平台支持对象

      fMouseButtons(0),  // 鼠标按键状态初始化为0
      fMouseX(-1),  // 鼠标X坐标初始化为-1
      fMouseY(-1),  // 鼠标Y坐标初始化为-1
      
      fLastKeyDown(0),  // 上一次按下的键盘按键初始化为0

      fRedraw(true),  // 设置重新绘制标志为真

      fPulse(NULL),  // 初始化为NULL，用于定时触发的消息
      fLastPulse(0),  // 上次脉冲时间初始化为0
      fEnableTicks(true)  // 启用定时器消息标志初始化为真
{
    SetViewColor(B_TRANSPARENT_32_BIT);  // 设置视图背景色为32位透明
    
    frame.OffsetTo(0.0, 0.0);  // 将视图框架偏移到原点
    fBitmap = new BBitmap(frame, 0, pix_format_to_color_space(fFormat));  // 创建与视图大小相匹配的位图
    if (fBitmap->IsValid()) {  // 如果位图有效
        attach_buffer_to_BBitmap(fAGG->rbuf_window(), fBitmap, fFlipY);  // 将AGG渲染缓冲区附加到位图上
    } else {
        delete fBitmap;  // 删除无效位图
        fBitmap = NULL;  // 置位图指针为空
    }
}


AGGView::~AGGView()
{
    delete fBitmap;  // 删除位图对象
    delete fPulse;  // 删除定时器消息对象
}


void
AGGView::AttachedToWindow()
{
    BMessage message('tick');  // 创建类型为 'tick' 的消息对象
    BMessenger target(this, Looper());  // 创建目标为当前视图的消息传递器
    delete fPulse;  // 删除现有的定时器消息对象
//    BScreen screen;
//    TODO: calc screen retrace
    fPulse = new BMessageRunner(target, &message, 40000);  // 创建新的定时器消息对象，每40毫秒发送一次消息

    // 确保调整窗口大小后调用此方法
    fAGG->on_resize(Bounds().IntegerWidth() + 1,
                    Bounds().IntegerHeight() + 1);
    MakeFocus();  // 设置视图为焦点
}


void
AGGView::DetachedFromWindow()
{
    delete fPulse;  // 删除定时器消息对象
    fPulse = NULL;  // 置定时器消息指针为空
}


void
AGGView::MessageReceived(BMessage* message)
{
    bigtime_t now = system_time();  // 获取当前系统时间
    switch (message->what) {
        case 'tick':
            // 丢弃积累的消息
            if (/*now - fLastPulse > 30000*/fEnableTicks) {  // 如果允许发送定时器消息
                fLastPulse = now;  // 更新上次发送消息的时间
                if (!fAGG->wait_mode())  // 如果AGG不在等待模式下
                    fAGG->on_idle();  // 执行空闲处理
                Window()->PostMessage('entk', this);  // 发送进入闲置消息
                fEnableTicks = false;  // 禁用定时器消息
            } else {
//                printf("dropping tick message (%lld)\n", now - fLastPulse);
            }
            break;
        case 'entk':
            fEnableTicks = true;  // 启用定时器消息
            if (now - fLastPulse > 30000) {  // 如果距离上次发送消息超过30秒
                fLastPulse = now;  // 更新上次发送消息的时间
                if (!fAGG->wait_mode())  // 如果AGG不在等待模式下
                    fAGG->on_idle();  // 执行空闲处理
            }
            break;
        default:
            BView::MessageReceived(message);  // 其他消息交给父类处理
            break;
    }
}


void
AGGView::Draw(BRect updateRect)
{
    # 检查是否存在有效的位图对象
    if (fBitmap) {
        # 如果需要重新绘制，调用AGG库的绘制方法，并标记不需要再次重绘
        if (fRedraw) {
            fAGG->on_draw();
            fRedraw = false;
        }
        # 根据位图的格式进行处理
        if (fFormat == agg::pix_format_bgra32) {
            # 如果位图格式已经是BGRA32，直接绘制该位图
            DrawBitmap(fBitmap, updateRect, updateRect);
        } else {
            # 创建一个新的BBitmap对象，用于转换格式
            BBitmap* bitmap = new BBitmap(fBitmap->Bounds(), 0, B_RGBA32);

            # 将源位图的渲染缓冲区绑定到AGG的渲染缓冲区
            agg::rendering_buffer rbufSrc;
            attach_buffer_to_BBitmap(rbufSrc, fBitmap, false);

            # 将目标位图的渲染缓冲区绑定到AGG的渲染缓冲区
            agg::rendering_buffer rbufDst;
            attach_buffer_to_BBitmap(rbufDst, bitmap, false);

            # 根据位图的格式选择合适的颜色转换器
            switch(fFormat) {
                case agg::pix_format_rgb555:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_rgb555_to_bgra32());
                    break;
                case agg::pix_format_rgb565:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_rgb565_to_bgra32());
                    break;
                case agg::pix_format_rgb24:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_rgb24_to_bgra32());
                    break;
                case agg::pix_format_bgr24:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_bgr24_to_bgra32());
                    break;
                case agg::pix_format_rgba32:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_rgba32_to_bgra32());
                    break;
                case agg::pix_format_argb32:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_argb32_to_bgra32());
                    break;
                case agg::pix_format_abgr32:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_abgr32_to_bgra32());
                    break;
                case agg::pix_format_bgra32:
                    # 若目标格式也为BGRA32，直接复制数据
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_bgra32_to_bgra32());
                    break;
            }
            # 绘制转换后的位图
            DrawBitmap(bitmap, updateRect, updateRect);
            # 释放内存，删除临时创建的位图对象
            delete bitmap;
        }
    } else {
        # 如果位图对象不存在，直接用填充矩形填充更新区域
        FillRect(updateRect);
    }
void
AGGView::FrameResized(float width, float height)
{
    // 创建一个 BRect 对象，表示窗口大小的矩形区域
    BRect r(0.0, 0.0, width, height);
    
    // 创建一个 BBitmap 对象，用于显示图像，初始化为指定大小和像素格式
    BBitmap* bitmap = new BBitmap(r, 0, pix_format_to_color_space(fFormat));
    
    // 检查 BBitmap 对象是否有效
    if (bitmap->IsValid()) {
        // 删除之前的 fBitmap 对象，并将新的 bitmap 赋值给 fBitmap
        delete fBitmap;
        fBitmap = bitmap;
        
        // 将 AGG 渲染缓冲区与 BBitmap 对象关联
        attach_buffer_to_BBitmap(fAGG->rbuf_window(), fBitmap, fFlipY);
        
        // 调整 AGG 渲染对象的仿射变换以适应新的窗口大小
        fAGG->trans_affine_resizing((int)width + 1, (int)height + 1);

        // 通知 AGG 对象窗口尺寸变化的事件处理
        fAGG->on_resize((int)width + 1, (int)height + 1);
        
        // 设置重绘标志
        fRedraw = true;
        
        // 使视图失效，需要重新绘制
        Invalidate();
    } else {
        delete bitmap; // 如果 BBitmap 无效，释放内存
    }
}


void
AGGView::KeyDown(const char* bytes, int32 numBytes)
{
    // 检查键盘事件是否有效
    if (bytes && numBytes > 0) {
        // 记录最后一个按下的键
        fLastKeyDown = bytes[0];

        // 初始化方向键状态
        bool left  = false;
        bool up    = false;
        bool right = false;
        bool down  = false;

        // 根据按键设置方向状态
        switch (fLastKeyDown) {
            case B_LEFT_ARROW:
                left = true;
                break;
            case B_UP_ARROW:
                up = true;
                break;
            case B_RIGHT_ARROW:
                right = true;
                break;
            case B_DOWN_ARROW:
                down = true;
                break;
        }

        // 处理方向键事件并通知 AGG 对象
        if (fAGG->m_ctrls.on_arrow_keys(left, right, down, up)) {
            fAGG->on_ctrl_change();
            fAGG->force_redraw();
        } else {
            fAGG->on_key(fMouseX, fMouseY, fLastKeyDown, GetKeyFlags());
        }
    }
}


void
AGGView::MouseDown(BPoint where)
{
    // 获取当前消息
    BMessage* currentMessage = Window()->CurrentMessage();
    
    // 确定鼠标按钮状态
    if (currentMessage) {
        if (currentMessage->FindInt32("buttons", (int32*)&fMouseButtons) < B_OK)
            fMouseButtons = B_PRIMARY_MOUSE_BUTTON;
    } else {
        fMouseButtons = B_PRIMARY_MOUSE_BUTTON;
    }

    // 记录鼠标位置
    fMouseX = (int)where.x;
    fMouseY = fFlipY ? (int)(Bounds().Height() - where.y) : (int)where.y;

    // 向 AGG 对象传递鼠标事件
    if (fMouseButtons == B_PRIMARY_MOUSE_BUTTON) {
        // 如果是左键，检查是否需要处理控件
        fAGG->m_ctrls.set_cur(fMouseX, fMouseY);
        
        // 处理鼠标左键按下事件
        if (fAGG->m_ctrls.on_mouse_button_down(fMouseX, fMouseY)) {
            fAGG->on_ctrl_change();
            fAGG->force_redraw();
        } else {
            // 如果不在控件区域内，则直接传递鼠标事件给 AGG
            if (fAGG->m_ctrls.in_rect(fMouseX, fMouseY)) {
                if (fAGG->m_ctrls.set_cur(fMouseX, fMouseY)) {
                    fAGG->on_ctrl_change();
                    fAGG->force_redraw();
                }
            } else {
                fAGG->on_mouse_button_down(fMouseX, fMouseY, GetKeyFlags());
            }
        }
    }
}
    } else if (fMouseButtons & B_SECONDARY_MOUSE_BUTTON) {
        // 如果鼠标右键被按下，则执行简单的操作
        fAGG->on_mouse_button_down(fMouseX, fMouseY, GetKeyFlags());
    }
    // 设置鼠标事件掩码为 B_POINTER_EVENTS，并锁定窗口焦点
    SetMouseEventMask(B_POINTER_EVENTS, B_LOCK_WINDOW_FOCUS);
void
AGGView::MouseMoved(BPoint where, uint32 transit, const BMessage* dragMessage)
{
    // workarround missed mouse up events
    // (if we react too slowly, app_server might have dropped events)
    // 获取当前窗口的当前消息
    BMessage* currentMessage = Window()->CurrentMessage();
    int32 buttons = 0;
    // 从当前消息中获取按钮状态，如果失败则默认为0
    if (currentMessage->FindInt32("buttons", &buttons) < B_OK) {
        buttons = 0;
    }
    // 如果没有按钮被按下，则调用 MouseUp 方法
    if (!buttons)
        MouseUp(where);

    // 将鼠标位置转换成视图坐标系下的坐标
    fMouseX = (int)where.x;
    fMouseY = fFlipY ? (int)(Bounds().Height() - where.y) : (int)where.y;

    // 将事件传递给 AGG 处理
    if (fAGG->m_ctrls.on_mouse_move(fMouseX, fMouseY,
                                    (GetKeyFlags() & agg::mouse_left) != 0)) {
        fAGG->on_ctrl_change();
        fAGG->force_redraw();
    } else {
        // 如果不在 AGG 控件的区域内，则调用 AGG 的鼠标移动事件处理方法
        if (!fAGG->m_ctrls.in_rect(fMouseX, fMouseY)) {
            fAGG->on_mouse_move(fMouseX, fMouseY, GetKeyFlags());
        }
    }
}


void
AGGView::MouseUp(BPoint where)
{
    // 将鼠标位置转换成视图坐标系下的坐标
    fMouseX = (int)where.x;
    fMouseY = fFlipY ? (int)(Bounds().Height() - where.y) : (int)where.y;

    // 将事件传递给 AGG 处理
    if (fMouseButtons == B_PRIMARY_MOUSE_BUTTON) {
        fMouseButtons = 0;

        // 如果是主鼠标按钮释放，则调用 AGG 的鼠标按钮释放事件处理方法
        if (fAGG->m_ctrls.on_mouse_button_up(fMouseX, fMouseY)) {
            fAGG->on_ctrl_change();
            fAGG->force_redraw();
        }
        fAGG->on_mouse_button_up(fMouseX, fMouseY, GetKeyFlags());
    } else if (fMouseButtons == B_SECONDARY_MOUSE_BUTTON) {
        fMouseButtons = 0;

        // 如果是次鼠标按钮释放，则调用 AGG 的鼠标按钮释放事件处理方法
        fAGG->on_mouse_button_up(fMouseX, fMouseY, GetKeyFlags());
    }
}


BBitmap*
AGGView::Bitmap() const
{
    // 返回当前视图的位图对象指针
    return fBitmap;
}


uint8
AGGView::LastKeyDown() const
{
    // 返回最后一个按下的键的键码
    return fLastKeyDown;
}


uint32
AGGView::MouseButtons()
{
    uint32 buttons = 0;
    // 如果能够锁定 looper 则获取当前鼠标按钮状态
    if (LockLooper()) {
        buttons = fMouseButtons;
        UnlockLooper();
    }
    return buttons;
}


void
AGGView::Update()
{
    // 触发显示更新
    if (LockLooper()) {
        Invalidate();
        UnlockLooper();
    }
}


void
AGGView::ForceRedraw()
{
    // 强制重绘（设置 fRedraw 为 true），并触发显示更新
    if (LockLooper()) {
        fRedraw = true;
        Invalidate();
        UnlockLooper();
    }
}


unsigned
AGGView::GetKeyFlags()
{
    uint32 buttons = fMouseButtons;
    uint32 mods = modifiers();
    unsigned flags = 0;
    // 根据当前按键状态和修饰键状态生成 AGG 框架所需的标志位
    if (buttons & B_PRIMARY_MOUSE_BUTTON)   flags |= agg::mouse_left;
    if (buttons & B_SECONDARY_MOUSE_BUTTON) flags |= agg::mouse_right;
    if (mods & B_SHIFT_KEY)                 flags |= agg::kbd_shift;
    if (mods & B_COMMAND_KEY)               flags |= agg::kbd_ctrl;
    return flags;
}

// #pragma mark -


class AGGWindow : public BWindow {
public:
    // AGGWindow 类的构造函数，创建一个带有指定位置和标题的窗口对象
    AGGWindow()
    : BWindow(BRect(-50.0, -50.0, -10.0, -10.0),
              "AGG Application", B_TITLED_WINDOW, B_ASYNCHRONOUS_CONTROLS)
    {
    }
    # 处理退出请求，发送退出消息并返回 true 表示已请求退出
    virtual bool QuitRequested()
    {
        be_app->PostMessage(B_QUIT_REQUESTED);
        return true;
    }

    # 初始化函数，设置窗口位置和大小，创建 AGGView 对象并添加到窗口中
    bool Init(BRect frame, agg::platform_support* agg, agg::pix_format_e format,
              bool flipY, uint32 flags)
    {
        # 移动窗口到左上角坐标，设置窗口大小
        MoveTo(frame.LeftTop());
        ResizeTo(frame.Width(), frame.Height());

        # 设置窗口标志位
        SetFlags(flags);

        # 将窗口位置偏移至原点
        frame.OffsetTo(0.0, 0.0);
        # 创建 AGGView 对象，并将其添加为子视图
        fView = new AGGView(frame, agg, format, flipY);
        AddChild(fView);

        # 返回是否成功创建了 AGGView 对象（即其 Bitmap 是否为非空）
        return fView->Bitmap() != NULL;
    }

    # 返回窗口的 AGGView 对象指针
    AGGView* View() const
    {
        return fView;
    }
// AGGApplication 类定义，继承自 BApplication
class AGGApplication : public BApplication {
 public:
                    // 构造函数，初始化 BApplication，并创建 AGGWindow 对象
                    AGGApplication()
                    : BApplication("application/x-vnd.AGG-AGG")
                    {
                        fWindow = new AGGWindow();
                    }

    // 准备就绪函数，当应用程序准备运行时被调用
    virtual void    ReadyToRun()
                    {
                        // 如果 fWindow 不为空，则显示窗口
                        if (fWindow) {
                            fWindow->Show();
                        }
                    }

    // 初始化函数，用于初始化应用程序和窗口
    virtual bool    Init(agg::platform_support* agg, int width, int height,
                         agg::pix_format_e format, bool flipY, uint32 flags)
                    {
                        // 创建窗口的矩形范围
                        BRect r(50.0, 50.0,
                                50.0 + width - 1.0,
                                50.0 + height - 1.0);
                        // 设置窗口标志
                        uint32 windowFlags = B_ASYNCHRONOUS_CONTROLS;
                        if (!(flags & agg::window_resize))
                            windowFlags |= B_NOT_RESIZABLE;

                        // 调用 AGGWindow 对象的初始化函数，并返回结果
                        return fWindow->Init(r, agg, format, flipY, windowFlags);;
                    }


        // 获取窗口对象的方法
        AGGWindow*  Window() const
                    {
                        return fWindow;
                    }

 private:
    AGGWindow*      fWindow;  // AGGWindow 对象指针
};
class platform_specific {
 public:
                    // 构造函数，初始化 platform_specific 对象
                    platform_specific(agg::platform_support* agg,
                                      agg::pix_format_e format, bool flip_y)
                        : fAGG(agg),                            // 初始化 fAGG 成员变量
                          fApp(NULL),                           // 初始化 fApp 成员变量为空指针
                          fFormat(format),                      // 初始化 fFormat 成员变量
                          fFlipY(flip_y),                       // 初始化 fFlipY 成员变量
                          fTimerStart(system_time())            // 初始化 fTimerStart 成员变量为当前系统时间
                    {
                        memset(fImages, 0, sizeof(fImages));    // 将 fImages 数组的所有元素置为 0
                        fApp = new AGGApplication();            // 创建一个 AGGApplication 的实例并赋给 fApp
                        fAppPath[0] = 0;                        // 初始化 fAppPath 数组第一个元素为 0
                        // 获取应用程序信息
                        app_info info;
                        status_t ret = fApp->GetAppInfo(&info); // 调用 fApp 的 GetAppInfo 方法获取应用信息
                        if (ret >= B_OK) {
                            BPath path(&info.ref);              // 根据应用信息创建 BPath 对象
                            ret = path.InitCheck();             // 检查路径初始化状态
                            if (ret >= B_OK) {
                                ret = path.GetParent(&path);    // 获取父文件夹路径
                                if (ret >= B_OK) {
                                    sprintf(fAppPath, "%s", path.Path());  // 将路径字符串格式化并存储到 fAppPath 中
                                } else {
                                    fprintf(stderr, "getting app parent folder failed: %s\n", strerror(ret));
                                    // 打印错误信息，指示获取应用程序父文件夹路径失败
                                }
                            } else {
                                fprintf(stderr, "making app path failed: %s\n", strerror(ret));
                                // 打印错误信息，指示创建应用程序路径失败
                            }
                        } else {
                            fprintf(stderr, "GetAppInfo() failed: %s\n", strerror(ret));
                            // 打印错误信息，指示获取应用程序信息失败
                        }
                    }
                    
                    // 析构函数，释放 platform_specific 对象资源
                    ~platform_specific()
                    {
                        for (int32 i = 0; i < agg::platform_support::max_images; i++)
                            delete fImages[i];                  // 释放 fImages 数组中的每个元素（图像对象）
                        delete fApp;                            // 释放 fApp 对象
                    }

    // 初始化函数，调用 fApp 的 Init 方法进行初始化
    bool            Init(int width, int height, unsigned flags)
                    {
                        return fApp->Init(fAGG, width, height, fFormat, fFlipY, flags);
                    }

    // 运行函数，调用 fApp 的 Run 方法开始运行应用程序
    int             Run()
                    {
                        status_t ret = B_NO_INIT;
                        if (fApp) {
                            fApp->Run();                       // 调用 fApp 的 Run 方法
                            ret = B_OK;                        // 设置返回状态为 B_OK
                        }
                        return ret;                            // 返回运行状态
                    }

    // 设置窗口标题，如果 fApp 存在且窗口可锁定，则设置标题
    void            SetTitle(const char* title)
                    {
                        if (fApp && fApp->Window() && fApp->Window()->Lock()) {
                            fApp->Window()->SetTitle(title);   // 设置窗口标题
                            fApp->Window()->Unlock();          // 解锁窗口
                        }
                    }

    // 记录当前时间作为计时器起始时间
    void            StartTimer()
                    {
                        fTimerStart = system_time();          // 记录当前系统时间到 fTimerStart
                    }

    // 返回自计时器启动以来经过的时间（单位为秒）
    double          ElapsedTime() const
                    {
                        return (system_time() - fTimerStart) / 1000.0;  // 计算并返回经过的时间
                    }
};
    // 强制重绘当前应用程序窗口视图
    void ForceRedraw()
    {
        fApp->Window()->View()->ForceRedraw();
    }

    // 更新当前应用程序窗口
    void UpdateWindow()
    {
        fApp->Window()->View()->Update();
    }

    // AGG 应用程序的平台支持对象
    agg::platform_support* fAGG;

    // 指向当前应用程序的 AGGApplication 对象
    AGGApplication* fApp;

    // 图像像素格式
    agg::pix_format_e fFormat;

    // 是否翻转 Y 轴
    bool fFlipY;

    // 计时器起始时间
    bigtime_t fTimerStart;

    // 存储多个图像的 BBitmap 数组
    BBitmap* fImages[agg::platform_support::max_images];

    // 应用程序路径字符串数组
    char fAppPath[B_PATH_NAME_LENGTH];

    // 文件路径字符串数组
    char fFilePath[B_PATH_NAME_LENGTH];
// platform_support 类的构造函数，接收图像格式和是否翻转 Y 轴作为参数
platform_support::platform_support(pix_format_e format, bool flip_y) :
    // 创建 platform_specific 对象，并传入当前 platform_support 对象的指针、图像格式和是否翻转 Y 轴的参数
    m_specific(new platform_specific(this, format, flip_y)),
    // 设置对象的图像格式
    m_format(format),
    // 设置每像素位数为 32 位（目前固定，不根据 platform_specific 内部 bpp 设置）
    m_bpp(32/*m_specific->m_bpp*/),
    // 窗口标志初始化为 0
    m_window_flags(0),
    // 等待模式设为 true
    m_wait_mode(true),
    // 设置是否翻转 Y 轴
    m_flip_y(flip_y),
    // 初始化窗口初始宽度为 10
    m_initial_width(10),
    // 初始化窗口初始高度为 10
    m_initial_height(10)
{
    // 将默认标题设置为 "Anti-Grain Geometry Application"
    strcpy(m_caption, "Anti-Grain Geometry Application");
}

//------------------------------------------------------------------------
// platform_support 类的析构函数
platform_support::~platform_support()
{
    // 删除 platform_specific 对象的内存
    delete m_specific;
}

//------------------------------------------------------------------------
// 设置窗口标题
void platform_support::caption(const char* cap)
{
    // 将传入的标题字符串复制到对象的标题成员变量 m_caption 中
    strcpy(m_caption, cap);
    // 调用 platform_specific 对象的 SetTitle 方法，设置窗口标题
    m_specific->SetTitle(cap);
}

//------------------------------------------------------------------------
// 启动定时器
void platform_support::start_timer()
{
    // 调用 platform_specific 对象的 StartTimer 方法，启动定时器
    m_specific->StartTimer();
}

//------------------------------------------------------------------------
// 获取经过的时间（秒）
double platform_support::elapsed_time() const
{
    // 调用 platform_specific 对象的 ElapsedTime 方法，返回经过的时间
    return m_specific->ElapsedTime();
}

//------------------------------------------------------------------------
// 获取原始显示处理程序的指针
void* platform_support::raw_display_handler()
{
    // TODO: 如果将来支持 BDirectWindow，这里会返回帧缓冲指针，偏移为窗口左上角
    return NULL;
}

//------------------------------------------------------------------------
// 显示消息框
void platform_support::message(const char* msg)
{
    // 创建 BAlert 对象，显示消息框，消息内容为 msg，只有一个 "Ok" 按钮
    BAlert* alert = new BAlert("AGG Message", msg, "Ok");
    alert->Go(/*NULL*/);
}

//------------------------------------------------------------------------
// 初始化平台支持，设置窗口的初始宽度、高度和标志
bool platform_support::init(unsigned width, unsigned height, unsigned flags)
{
    // 设置窗口的初始宽度和高度
    m_initial_width = width;
    m_initial_height = height;
    // 设置窗口的标志
    m_window_flags = flags;

    // 调用 platform_specific 对象的 Init 方法，初始化窗口
    if (m_specific->Init(width, height, flags)) {
        // 调用 on_init 方法，执行额外的初始化操作
        on_init();
        return true;
    }

    return false;
}

//------------------------------------------------------------------------
// 运行平台支持，委托给 platform_specific 对象的 Run 方法
int platform_support::run()
{
    return m_specific->Run();
}

//------------------------------------------------------------------------
// 返回图像文件扩展名 ".ppm"
const char* platform_support::img_ext() const { return ".ppm"; }

//------------------------------------------------------------------------
// 返回完整的文件名路径，拼接应用程序路径和文件名
const char* platform_support::full_file_name(const char* file_name)
{
    sprintf(m_specific->fFilePath, "%s/%s", m_specific->fAppPath, file_name);
    return m_specific->fFilePath;
}

//------------------------------------------------------------------------
// 加载图像，未实现具体功能，可能用于将图像加载到平台特定的对象中
bool platform_support::load_img(unsigned idx, const char* file)
    {
        // 检查索引是否小于最大图像数
        if (idx < max_images)
        {
            // 创建保存文件路径的字符数组，并格式化路径
            char path[B_PATH_NAME_LENGTH];
            sprintf(path, "%s/%s%s", m_specific->fAppPath, file, img_ext());
            
            // 使用路径加载位图
            BBitmap* transBitmap = BTranslationUtils::GetBitmap(path);
            
            // 检查位图加载是否成功且有效
            if (transBitmap && transBitmap->IsValid()) {
                // 检查位图的颜色空间是否不是 B_RGB32 或 B_RGBA32
                if(transBitmap->ColorSpace() != B_RGB32 && transBitmap->ColorSpace() != B_RGBA32) {
                    // 输出警告信息，删除位图对象并返回失败
                    delete transBitmap;
                    return false;
                }
    
                // 设置目标位图的颜色格式，默认为 B_RGB24
                color_space format = B_RGB24;
    
                // 根据目标格式设置颜色空间
                switch (m_format) {
                    case pix_format_gray8:
                        format = B_GRAY8;
                        break;
                    case pix_format_rgb555:
                        format = B_RGB15;
                        break;
                    case pix_format_rgb565:
                        format = B_RGB16;
                        break;
                    case pix_format_rgb24:
                        format = B_RGB24_BIG;
                        break;
                    case pix_format_bgr24:
                        format = B_RGB24;
                        break;
                    case pix_format_abgr32:
                    case pix_format_argb32:
                    case pix_format_bgra32:
                        format = B_RGB32;
                        break;
                    case pix_format_rgba32:
                        format = B_RGB32_BIG;
                        break;
                }
    
                // 创建新的目标位图对象
                BBitmap* bitmap = new (nothrow) BBitmap(transBitmap->Bounds(), 0, format);
                
                // 检查位图对象是否创建成功并且有效
                if (!bitmap || !bitmap->IsValid()) {
                    // 输出错误信息，删除位图对象并返回失败
                    fprintf(stderr, "failed to allocate temporary bitmap!\n");
                    delete transBitmap;
                    delete bitmap;
                    return false;
                }
    
                // 删除原来的图像对象
                delete m_specific->fImages[idx];
    
                // 创建渲染缓冲区对象并附加到原始位图
                rendering_buffer rbuf_tmp;
                attach_buffer_to_BBitmap(rbuf_tmp, transBitmap, m_flip_y);
    
                // 将新的目标位图设置为当前图像对象
                m_specific->fImages[idx] = bitmap;
    
                // 将渲染缓冲区附加到目标位图
                attach_buffer_to_BBitmap(m_rbuf_img[idx], bitmap, m_flip_y);
    
                // 获取当前图像的渲染缓冲区指针
                rendering_buffer* dst = &m_rbuf_img[idx];
    
                // 根据当前格式执行不同的操作
                switch(m_format)
                {
                // 对于灰度图像格式，返回失败
                case pix_format_gray8:
                    return false;
// 根据不同的像素格式进行颜色转换，并将结果存储到目标图像中
// case pix_format_gray8: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_gray8()); break;
// case pix_format_rgb555: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgb555()); break;
// case pix_format_rgb565: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgb565()); break;
// case pix_format_rgb24: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgb24()); break;
// case pix_format_bgr24: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_bgr24()); break;
// case pix_format_abgr32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_abgr32()); break;
// case pix_format_argb32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_argb32()); break;
// case pix_format_bgra32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_bgra32()); break;
// case pix_format_rgba32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgba32()); break;
// 删除 transBitmap 对象
// 返回 true
bool platform_support::load_img(unsigned idx, const char* file)
{
    // 如果文件名不为空
    if (file)
    {
        // 创建一个 BBitmap 对象
        BBitmap* transBitmap = BTranslationUtils::GetBitmap(file);
        // 如果成功加载了位图
        if (transBitmap && transBitmap->IsValid())
        {
            // 删除之前的图像
            delete m_specific->fImages[idx];
            // 将新的位图存储到指定索引位置
            m_specific->fImages[idx] = transBitmap;
            // 将图像数据附加到 BBitmap 对象
            attach_buffer_to_BBitmap(m_rbuf_img[idx], transBitmap, m_flip_y);
            // 返回加载成功
            return true;
        }
        else
        {
            // 打印加载失败的错误信息
            fprintf(stderr, "failed to load bitmap: '%s'\n", full_file_name(file));
        }
    }
    // 返回加载失败
    return false;
}

// 保存图像到文件
// TODO: 使用 BTranslatorRoster 和相关类实现
// 返回 false
bool platform_support::save_img(unsigned idx, const char* file)
{
    return false;
}

// 创建指定大小的图像
// 如果索引小于最大图像数量
// 如果宽度为0，则设置为窗口视图的宽度
// 如果高度为0，则设置为窗口视图的高度
// 创建一个 BBitmap 对象
// 如果成功创建并有效
// 删除之前的图像
// 将新的位图存储到指定索引位置
// 将图像数据附加到 BBitmap 对象
// 返回 true
// 否则，删除位图对象，返回 false
bool platform_support::create_img(unsigned idx, unsigned width, unsigned height)
{
    if (idx < max_images)
    {
        if (width == 0) width = m_specific->fApp->Window()->View()->Bitmap()->Bounds().IntegerWidth() + 1;
        if (height == 0) height = m_specific->fApp->Window()->View()->Bitmap()->Bounds().IntegerHeight() + 1;
        BBitmap* bitmap = new BBitmap(BRect(0.0, 0.0, width - 1, height - 1), 0, B_RGBA32);
        if (bitmap && bitmap->IsValid())
        {
            delete m_specific->fImages[idx];
            m_specific->fImages[idx] = bitmap;
            attach_buffer_to_BBitmap(m_rbuf_img[idx], bitmap, m_flip_y);
            return true;
        }
        else
        {
            delete bitmap;
        }
    }
    return false;
}

// 强制重绘
void platform_support::force_redraw()
{
    m_specific->ForceRedraw();
}
    //------------------------------------------------------------------------
    // 更新窗口的方法，在特定平台的实现中调用 UpdateWindow 函数来执行更新操作
    void platform_support::update_window()
    {
        // 调用特定平台接口的 UpdateWindow 方法
        m_specific->UpdateWindow();
    }
    
    
    //------------------------------------------------------------------------
    // 下列函数都是空函数，用于处理特定平台支持的各种事件，但在当前代码中未实现具体逻辑
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
```