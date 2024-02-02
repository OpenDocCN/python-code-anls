# `ZeroNet\plugins\Trayicon\lib\notificationicon.py`

```py
# 导入 ctypes 库，用于操作 Windows 窗口任务栏通知图标
# 导入 os、uuid、time、gevent、threading 库
import ctypes
import ctypes.wintypes
import os
import uuid
import time
import gevent
import threading
# 导入 queue 库，用于处理队列
try:
    from queue import Empty as queue_Empty  # Python 3
except ImportError:
    from Queue import Empty as queue_Empty  # Python 2

# 定义公开的类 NotificationIcon
__all__ = ['NotificationIcon']

# 创建弹出菜单

# 调用 Windows user32.dll 中的 CreatePopupMenu 函数，创建一个弹出菜单
CreatePopupMenu = ctypes.windll.user32.CreatePopupMenu
CreatePopupMenu.restype = ctypes.wintypes.HMENU
CreatePopupMenu.argtypes = []

# 定义弹出菜单的各种属性
MF_BYCOMMAND    = 0x0
MF_BYPOSITION   = 0x400
MF_BITMAP       = 0x4
MF_CHECKED      = 0x8
MF_DISABLED     = 0x2
MF_ENABLED      = 0x0
MF_GRAYED       = 0x1
MF_MENUBARBREAK = 0x20
MF_MENUBREAK    = 0x40
MF_OWNERDRAW    = 0x100
MF_POPUP        = 0x10
MF_SEPARATOR    = 0x800
MF_STRING       = 0x0
MF_UNCHECKED    = 0x0

# 调用 Windows user32.dll 中的 InsertMenuW 函数，向弹出菜单中插入菜单项
InsertMenu = ctypes.windll.user32.InsertMenuW
InsertMenu.restype = ctypes.wintypes.BOOL
InsertMenu.argtypes = [ctypes.wintypes.HMENU, ctypes.wintypes.UINT, ctypes.wintypes.UINT, ctypes.wintypes.UINT, ctypes.wintypes.LPCWSTR]

# 调用 Windows user32.dll 中的 AppendMenuW 函数，向弹出菜单中追加菜单项
AppendMenu = ctypes.windll.user32.AppendMenuW
AppendMenu.restype = ctypes.wintypes.BOOL
AppendMenu.argtypes = [ctypes.wintypes.HMENU, ctypes.wintypes.UINT, ctypes.wintypes.UINT, ctypes.wintypes.LPCWSTR]

# 调用 Windows user32.dll 中的 SetMenuDefaultItem 函数，设置弹出菜单的默认菜单项
SetMenuDefaultItem = ctypes.windll.user32.SetMenuDefaultItem
SetMenuDefaultItem.restype = ctypes.wintypes.BOOL
SetMenuDefaultItem.argtypes = [ctypes.wintypes.HMENU, ctypes.wintypes.UINT, ctypes.wintypes.UINT]

# 定义 POINT 结构体，用于表示坐标
class POINT(ctypes.Structure):
    _fields_ = [ ('x', ctypes.wintypes.LONG),
                 ('y', ctypes.wintypes.LONG)]

# 调用 Windows user32.dll 中的 GetCursorPos 函数，获取鼠标当前位置
GetCursorPos = ctypes.windll.user32.GetCursorPos
GetCursorPos.argtypes = [ctypes.POINTER(POINT)]

# 调用 Windows user32.dll 中的 SetForegroundWindow 函数，将指定窗口设置为前台窗口
SetForegroundWindow = ctypes.windll.user32.SetForegroundWindow
SetForegroundWindow.argtypes = [ctypes.wintypes.HWND]

# 定义弹出菜单的对齐方式
TPM_LEFTALIGN       = 0x0
TPM_CENTERALIGN     = 0x4
TPM_RIGHTALIGN      = 0x8
TPM_TOPALIGN        = 0x0
TPM_VCENTERALIGN    = 0x10
# 定义常量 TPM_BOTTOMALIGN，数值为 0x20
TPM_BOTTOMALIGN     = 0x20

# 定义常量 TPM_NONOTIFY，数值为 0x80
TPM_NONOTIFY        = 0x80
# 定义常量 TPM_RETURNCMD，数值为 0x100
TPM_RETURNCMD       = 0x100

# 定义常量 TPM_LEFTBUTTON，数值为 0x0
TPM_LEFTBUTTON      = 0x0
# 定义常量 TPM_RIGHTBUTTON，数值为 0x2
TPM_RIGHTBUTTON     = 0x2

# 定义常量 TPM_HORNEGANIMATION，数值为 0x800
TPM_HORNEGANIMATION = 0x800
# 定义常量 TPM_HORPOSANIMATION，数值为 0x400
TPM_HORPOSANIMATION = 0x400
# 定义常量 TPM_NOANIMATION，数值为 0x4000
TPM_NOANIMATION     = 0x4000
# 定义常量 TPM_VERNEGANIMATION，数值为 0x2000
TPM_VERNEGANIMATION = 0x2000
# 定义常量 TPM_VERPOSANIMATION，数值为 0x1000
TPM_VERPOSANIMATION = 0x1000

# 调用 ctypes.windll.user32.TrackPopupMenu 函数，并设置返回类型和参数类型
TrackPopupMenu = ctypes.windll.user32.TrackPopupMenu
TrackPopupMenu.restype = ctypes.wintypes.BOOL
TrackPopupMenu.argtypes = [ctypes.wintypes.HMENU, ctypes.wintypes.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.wintypes.HWND, ctypes.c_void_p]

# 调用 ctypes.windll.user32.PostMessageW 函数，并设置返回类型和参数类型
PostMessage = ctypes.windll.user32.PostMessageW
PostMessage.restype = ctypes.wintypes.BOOL
PostMessage.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.UINT, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM]

# 调用 ctypes.windll.user32.DestroyMenu 函数，并设置返回类型和参数类型
DestroyMenu = ctypes.windll.user32.DestroyMenu
DestroyMenu.restype = ctypes.wintypes.BOOL
DestroyMenu.argtypes = [ctypes.wintypes.HMENU]

# 定义 GUID 类型为长度为 16 的 ctypes.c_ubyte 数组
GUID = ctypes.c_ubyte * 16

# 定义 TimeoutVersionUnion 类型为 ctypes.Union，包含字段 uTimeout 和 uVersion
class TimeoutVersionUnion(ctypes.Union):
    _fields_ = [('uTimeout', ctypes.wintypes.UINT),
                ('uVersion', ctypes.wintypes.UINT),]

# 定义常量 NIS_HIDDEN，数值为 0x1
NIS_HIDDEN     = 0x1
# 定义常量 NIS_SHAREDICON，数值为 0x2
NIS_SHAREDICON = 0x2

# 定义 NOTIFYICONDATA 结构体，包含多个字段
class NOTIFYICONDATA(ctypes.Structure):
    # 初始化函数，设置 cbSize 字段的值为结构体大小
    def __init__(self, *args, **kwargs):
        super(NOTIFYICONDATA, self).__init__(*args, **kwargs)
        self.cbSize = ctypes.sizeof(self)
    _fields_ = [
        ('cbSize', ctypes.wintypes.DWORD),
        ('hWnd', ctypes.wintypes.HWND),
        ('uID', ctypes.wintypes.UINT),
        ('uFlags', ctypes.wintypes.UINT),
        ('uCallbackMessage', ctypes.wintypes.UINT),
        ('hIcon', ctypes.wintypes.HICON),
        ('szTip', ctypes.wintypes.WCHAR * 64),
        ('dwState', ctypes.wintypes.DWORD),
        ('dwStateMask', ctypes.wintypes.DWORD),
        ('szInfo', ctypes.wintypes.WCHAR * 256),
        ('union', TimeoutVersionUnion),
        ('szInfoTitle', ctypes.wintypes.WCHAR * 64),
        ('dwInfoFlags', ctypes.wintypes.DWORD),
        ('guidItem', GUID),
        ('hBalloonIcon', ctypes.wintypes.HICON),
    ]
# 定义通知图标的操作类型
NIM_ADD = 0
NIM_MODIFY = 1
NIM_DELETE = 2
NIM_SETFOCUS = 3
NIM_SETVERSION = 4

# 定义通知图标的信息类型
NIF_MESSAGE = 1
NIF_ICON = 2
NIF_TIP = 4
NIF_STATE = 8
NIF_INFO = 16
NIF_GUID = 32
NIF_REALTIME = 64
NIF_SHOWTIP = 128

# 定义通知图标的信息提示类型
NIIF_NONE = 0
NIIF_INFO = 1
NIIF_WARNING = 2
NIIF_ERROR = 3
NIIF_USER = 4

# 定义通知图标的版本
NOTIFYICON_VERSION = 3
NOTIFYICON_VERSION_4 = 4

# 调用 Windows API 中的 Shell_NotifyIcon 函数
Shell_NotifyIcon = ctypes.windll.shell32.Shell_NotifyIconW
Shell_NotifyIcon.restype = ctypes.wintypes.BOOL
Shell_NotifyIcon.argtypes = [ctypes.wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATA)]

# 加载图标/图片
IMAGE_BITMAP = 0
IMAGE_ICON = 1
IMAGE_CURSOR = 2

# 加载图像的选项
LR_CREATEDIBSECTION = 0x00002000
LR_DEFAULTCOLOR     = 0x00000000
LR_DEFAULTSIZE      = 0x00000040
LR_LOADFROMFILE     = 0x00000010
LR_LOADMAP3DCOLORS  = 0x00001000
LR_LOADTRANSPARENT  = 0x00000020
LR_MONOCHROME       = 0x00000001
LR_SHARED           = 0x00008000
LR_VGACOLOR         = 0x00000080

# 预定义图标
OIC_SAMPLE      = 32512
OIC_HAND        = 32513
OIC_QUES        = 32514
OIC_BANG        = 32515
OIC_NOTE        = 32516
OIC_WINLOGO     = 32517
OIC_WARNING     = OIC_BANG
OIC_ERROR       = OIC_HAND
OIC_INFORMATION = OIC_NOTE

# 调用 Windows API 中的 LoadImage 函数
LoadImage = ctypes.windll.user32.LoadImageW
LoadImage.restype = ctypes.wintypes.HANDLE
LoadImage.argtypes = [ctypes.wintypes.HINSTANCE, ctypes.wintypes.LPCWSTR, ctypes.wintypes.UINT, ctypes.c_int, ctypes.c_int, ctypes.wintypes.UINT]

# 创建窗口调用
WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.wintypes.HWND, ctypes.c_uint, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM)
DefWindowProc = ctypes.windll.user32.DefWindowProcW
DefWindowProc.restype = ctypes.c_int
DefWindowProc.argtypes = [ctypes.wintypes.HWND, ctypes.c_uint, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM]

# 窗口样式
WS_OVERLAPPED       = 0x00000000
WS_POPUP            = 0x80000000
WS_CHILD            = 0x40000000
WS_MINIMIZE         = 0x20000000
WS_VISIBLE          = 0x10000000
WS_DISABLED         = 0x08000000
WS_CLIPSIBLINGS     = 0x04000000
WS_CLIPCHILDREN     = 0x02000000
WS_MAXIMIZE         = 0x01000000
# 定义窗口样式常量
WS_CAPTION          = 0x00C00000  # 窗口有标题栏
WS_BORDER           = 0x00800000  # 窗口有边框
WS_DLGFRAME         = 0x00400000  # 窗口有对话框边框
WS_VSCROLL          = 0x00200000  # 窗口有垂直滚动条
WS_HSCROLL          = 0x00100000  # 窗口有水平滚动条
WS_SYSMENU          = 0x00080000  # 窗口有系统菜单
WS_THICKFRAME       = 0x00040000  # 窗口有大小可调边框
WS_GROUP            = 0x00020000  # 窗口是第一个控件组的一部分
WS_TABSTOP          = 0x00010000  # 窗口是可接受 Tab 键输入的控件

# 定义窗口最小化和最大化样式常量
WS_MINIMIZEBOX      = 0x00020000  # 窗口有最小化按钮
WS_MAXIMIZEBOX      = 0x00010000  # 窗口有最大化按钮

# 定义窗口重叠样式常量
WS_OVERLAPPEDWINDOW = (WS_OVERLAPPED     |
                       WS_CAPTION        |
                       WS_SYSMENU        |
                       WS_THICKFRAME     |
                       WS_MINIMIZEBOX    |
                       WS_MAXIMIZEBOX)  # 窗口具有重叠样式

# 定义系统参数常量
SM_XVIRTUALSCREEN      = 76  # 虚拟屏幕的左上角 x 坐标
SM_YVIRTUALSCREEN      = 77  # 虚拟屏幕的左上角 y 坐标
SM_CXVIRTUALSCREEN     = 78  # 虚拟屏幕的宽度
SM_CYVIRTUALSCREEN     = 79  # 虚拟屏幕的高度
SM_CMONITORS           = 80  # 系统中的监视器数量
SM_SAMEDISPLAYFORMAT   = 81  # 所有监视器是否具有相同的显示格式

# 定义窗口消息常量
WM_NULL                   = 0x0000  # 一个空消息，不做任何事情
WM_CREATE                 = 0x0001  # 窗口被创建
WM_DESTROY                = 0x0002  # 窗口被销毁
WM_MOVE                   = 0x0003  # 窗口移动
WM_SIZE                   = 0x0005  # 窗口大小改变
# 其他窗口消息常量...
# 定义 Windows 消息常量，用于消息处理和窗口事件
WM_GETMINMAXINFO          = 0x0024  # 当窗口大小变化时发送消息，用于获取窗口的最大和最小尺寸
WM_PAINTICON              = 0x0026  # 绘制窗口图标时发送消息
WM_ICONERASEBKGND         = 0x0027  # 擦除窗口图标的背景时发送消息
WM_NEXTDLGCTL             = 0x0028  # 设置下一个对话框控件时发送消息
WM_SPOOLERSTATUS          = 0x002A  # 打印机状态改变时发送消息
WM_DRAWITEM               = 0x002B  # 绘制列表框、组合框或按钮控件的子项时发送消息
WM_MEASUREITEM            = 0x002C  # 测量列表框、组合框或按钮控件的子项时发送消息
WM_DELETEITEM             = 0x002D  # 删除列表框、组合框或按钮控件的子项时发送消息
WM_VKEYTOITEM             = 0x002E  # 键盘输入转换为列表框或组合框控件的索引时发送消息
WM_CHARTOITEM             = 0x002F  # 字符输入转换为列表框或组合框控件的索引时发送消息
WM_SETFONT                = 0x0030  # 设置控件的字体时发送消息
WM_GETFONT                = 0x0031  # 获取控件的字体时发送消息
WM_SETHOTKEY              = 0x0032  # 设置热键时发送消息
WM_GETHOTKEY              = 0x0033  # 获取热键时发送消息
WM_QUERYDRAGICON          = 0x0037  # 拖动图标时发送消息
WM_COMPAREITEM            = 0x0039  # 比较列表框或组合框控件的子项时发送消息
WM_GETOBJECT              = 0x003D  # 获取对象的相关信息时发送消息
WM_COMPACTING             = 0x0041  # 系统内存紧张时发送消息
WM_COMMNOTIFY             = 0x0044  # 通信错误时发送消息
WM_WINDOWPOSCHANGING      = 0x0046  # 窗口位置即将改变时发送消息
WM_WINDOWPOSCHANGED       = 0x0047  # 窗口位置已经改变时发送消息
WM_POWER                  = 0x0048  # 电源状态改变时发送消息
WM_COPYDATA               = 0x004A  # 在进程间传递数据时发送消息
WM_CANCELJOURNAL          = 0x004B  # 取消日志记录时发送消息
WM_NOTIFY                 = 0x004E  # 控件发送通知消息时发送消息
WM_INPUTLANGCHANGEREQUEST = 0x0050  # 输入语言即将改变时发送消息
WM_INPUTLANGCHANGE        = 0x0051  # 输入语言已经改变时发送消息
WM_TCARD                  = 0x0052  # 卡片控件发送通知消息时发送消息
WM_HELP                   = 0x0053  # 请求帮助时发送消息
WM_USERCHANGED            = 0x0054  # 用户更改设置时发送消息
WM_NOTIFYFORMAT           = 0x0055  # 控件发送通知格式消息时发送消息
WM_CONTEXTMENU            = 0x007B  # 请求上下文菜单时发送消息
WM_STYLECHANGING          = 0x007C  # 窗口样式即将改变时发送消息
WM_STYLECHANGED           = 0x007D  # 窗口样式已经改变时发送消息
WM_DISPLAYCHANGE          = 0x007E  # 显示器分辨率或色彩属性改变时发送消息
WM_GETICON                = 0x007F  # 获取窗口图标时发送消息
WM_SETICON                = 0x0080  # 设置窗口图标时发送消息
WM_NCCREATE               = 0x0081  # 窗口即将被创建时发送消息
WM_NCDESTROY              = 0x0082  # 窗口即将被销毁时发送消息
WM_NCCALCSIZE             = 0x0083  # 计算窗口客户区的大小和位置时发送消息
WM_NCHITTEST              = 0x0084  # 确定鼠标位置属于哪个窗口部分时发送消息
WM_NCPAINT                = 0x0085  # 绘制窗口的非客户区时发送消息
WM_NCACTIVATE             = 0x0086  # 窗口非客户区激活状态改变时发送消息
WM_GETDLGCODE             = 0x0087  # 获取对话框控件的输入消息时发送消息
WM_SYNCPAINT              = 0x0088  # 同步绘制窗口时发送消息
WM_NCMOUSEMOVE            = 0x00A0  # 非客户区鼠标移动时发送消息
WM_NCLBUTTONDOWN          = 0x00A1  # 非客户区鼠标左键按下时发送消息
WM_NCLBUTTONUP            = 0x00A2  # 非客户区鼠标左键释放时发送消息
WM_NCLBUTTONDBLCLK        = 0x00A3  # 非客户区鼠标左键双击时发送消息
WM_NCRBUTTONDOWN          = 0x00A4  # 非客户区鼠标右键按下时发送消息
WM_NCRBUTTONUP            = 0x00A5  # 非客户区鼠标右键释放时发送消息
WM_NCRBUTTONDBLCLK        = 0x00A6  # 非客户区鼠标右键双击时发送消息
WM_NCMBUTTONDOWN          = 0x00A7  # 非客户区鼠标中键按下时发送消息
WM_NCMBUTTONUP            = 0x00A8  # 非客户区鼠标中键释放时发送消息
WM_NCMBUTTONDBLCLK        = 0x00A9  # 非客户区鼠标中键双击时发送消息
WM_KEYDOWN                = 0x0100  # 键盘按键按下时发送消息
WM_KEYUP                  = 0x0101  # 键盘按键释放时发送消息
# 定义 Windows 消息常量
WM_CHAR                   = 0x0102  # 发送一个字符消息
WM_DEADCHAR               = 0x0103  # 发送一个死字符消息
WM_SYSKEYDOWN             = 0x0104  # 发送一个系统按键按下消息
WM_SYSKEYUP               = 0x0105  # 发送一个系统按键释放消息
WM_SYSCHAR                = 0x0106  # 发送一个系统字符消息
WM_SYSDEADCHAR            = 0x0107  # 发送一个系统死字符消息
WM_KEYLAST                = 0x0108  # 保留
WM_IME_STARTCOMPOSITION   = 0x010D  # IME 开始组合消息
WM_IME_ENDCOMPOSITION     = 0x010E  # IME 结束组合消息
WM_IME_COMPOSITION        = 0x010F  # IME 组合消息
WM_IME_KEYLAST            = 0x010F  # 保留
WM_INITDIALOG             = 0x0110  # 初始化对话框消息
WM_COMMAND                = 0x0111  # 发送一个命令消息
WM_SYSCOMMAND             = 0x0112  # 发送一个系统命令消息
WM_TIMER                  = 0x0113  # 定时器消息
WM_HSCROLL                = 0x0114  # 水平滚动条消息
WM_VSCROLL                = 0x0115  # 垂直滚动条消息
WM_INITMENU               = 0x0116  # 初始化菜单消息
WM_INITMENUPOPUP          = 0x0117  # 初始化弹出菜单消息
WM_MENUSELECT             = 0x011F  # 选择菜单消息
WM_MENUCHAR               = 0x0120  # 菜单字符消息
WM_ENTERIDLE              = 0x0121  # 进入空闲状态消息
WM_MENURBUTTONUP          = 0x0122  # 右键弹出菜单消息
WM_MENUDRAG               = 0x0123  # 拖拽菜单消息
WM_MENUGETOBJECT          = 0x0124  # 获取菜单对象消息
WM_UNINITMENUPOPUP        = 0x0125  # 取消初始化弹出菜单消息
WM_MENUCOMMAND            = 0x0126  # 菜单命令消息
WM_CTLCOLORMSGBOX         = 0x0132  # 控件颜色消息 - 消息框
WM_CTLCOLOREDIT           = 0x0133  # 控件颜色消息 - 编辑框
WM_CTLCOLORLISTBOX        = 0x0134  # 控件颜色消息 - 列表框
WM_CTLCOLORBTN            = 0x0135  # 控件颜色消息 - 按钮
WM_CTLCOLORDLG            = 0x0136  # 控件颜色消息 - 对话框
WM_CTLCOLORSCROLLBAR      = 0x0137  # 控件颜色消息 - 滚动条
WM_CTLCOLORSTATIC         = 0x0138  # 控件颜色消息 - 静态控件
WM_MOUSEMOVE              = 0x0200  # 鼠标移动消息
WM_LBUTTONDOWN            = 0x0201  # 鼠标左键按下消息
WM_LBUTTONUP              = 0x0202  # 鼠标左键释放消息
WM_LBUTTONDBLCLK          = 0x0203  # 鼠标左键双击消息
WM_RBUTTONDOWN            = 0x0204  # 鼠标右键按下消息
WM_RBUTTONUP              = 0x0205  # 鼠标右键释放消息
WM_RBUTTONDBLCLK          = 0x0206  # 鼠标右键双击消息
WM_MBUTTONDOWN            = 0x0207  # 鼠标中键按下消息
WM_MBUTTONUP              = 0x0208  # 鼠标中键释放消息
WM_MBUTTONDBLCLK          = 0x0209  # 鼠标中键双击消息
WM_MOUSEWHEEL             = 0x020A  # 鼠标滚轮滚动消息
WM_PARENTNOTIFY           = 0x0210  # 父窗口通知消息
WM_ENTERMENULOOP          = 0x0211  # 进入菜单循环消息
WM_EXITMENULOOP           = 0x0212  # 退出菜单循环消息
WM_NEXTMENU               = 0x0213  # 下一个菜单消息
WM_SIZING                 = 0x0214  # 调整大小消息
WM_CAPTURECHANGED         = 0x0215  # 捕获改变消息
WM_MOVING                 = 0x0216  # 移动消息
WM_DEVICECHANGE           = 0x0219  # 设备改变消息
WM_MDICREATE              = 0x0220  # MDI 创建消息
WM_MDIDESTROY             = 0x0221  # MDI 销毁消息
WM_MDIACTIVATE            = 0x0222  # MDI 激活消息
WM_MDIRESTORE             = 0x0223  # MDI 恢复消息
# 定义 Windows 消息常量
WM_MDINEXT                = 0x0224
WM_MDIMAXIMIZE            = 0x0225
WM_MDITILE                = 0x0226
WM_MDICASCADE             = 0x0227
WM_MDIICONARRANGE         = 0x0228
WM_MDIGETACTIVE           = 0x0229
WM_MDISETMENU             = 0x0230
WM_ENTERSIZEMOVE          = 0x0231
WM_EXITSIZEMOVE           = 0x0232
WM_DROPFILES              = 0x0233
WM_MDIREFRESHMENU         = 0x0234
WM_IME_SETCONTEXT         = 0x0281
WM_IME_NOTIFY             = 0x0282
WM_IME_CONTROL            = 0x0283
WM_IME_COMPOSITIONFULL    = 0x0284
WM_IME_SELECT             = 0x0285
WM_IME_CHAR               = 0x0286
WM_IME_REQUEST            = 0x0288
WM_IME_KEYDOWN            = 0x0290
WM_IME_KEYUP              = 0x0291
WM_MOUSEHOVER             = 0x02A1
WM_MOUSELEAVE             = 0x02A3
WM_CUT                    = 0x0300
WM_COPY                   = 0x0301
WM_PASTE                  = 0x0302
WM_CLEAR                  = 0x0303
WM_UNDO                   = 0x0304
WM_RENDERFORMAT           = 0x0305
WM_RENDERALLFORMATS       = 0x0306
WM_DESTROYCLIPBOARD       = 0x0307
WM_DRAWCLIPBOARD          = 0x0308
WM_PAINTCLIPBOARD         = 0x0309
WM_VSCROLLCLIPBOARD       = 0x030A
WM_SIZECLIPBOARD          = 0x030B
WM_ASKCBFORMATNAME        = 0x030C
WM_CHANGECBCHAIN          = 0x030D
WM_HSCROLLCLIPBOARD       = 0x030E
WM_QUERYNEWPALETTE        = 0x030F
WM_PALETTEISCHANGING      = 0x0310
WM_PALETTECHANGED         = 0x0311
WM_HOTKEY                 = 0x0312
WM_PRINT                  = 0x0317
WM_PRINTCLIENT            = 0x0318
WM_HANDHELDFIRST          = 0x0358
WM_HANDHELDLAST           = 0x035F
WM_AFXFIRST               = 0x0360
WM_AFXLAST                = 0x037F
WM_PENWINFIRST            = 0x0380
WM_PENWINLAST             = 0x038F
WM_APP                    = 0x8000
WM_USER                   = 0x0400
WM_REFLECT                = WM_USER + 0x1c00

class WNDCLASSEX(ctypes.Structure):
    # 初始化方法，接受任意数量的位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super(WNDCLASSEX, self).__init__(*args, **kwargs)
        # 设置 cbSize 属性为自身所占内存大小
        self.cbSize = ctypes.sizeof(self)
    # 定义结构体的字段
    _fields_ = [("cbSize", ctypes.c_uint),
                ("style", ctypes.c_uint),
                ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", ctypes.wintypes.HANDLE),
                ("hIcon", ctypes.wintypes.HANDLE),
                ("hCursor", ctypes.wintypes.HANDLE),
                ("hBrush", ctypes.wintypes.HANDLE),
                ("lpszMenuName", ctypes.wintypes.LPCWSTR),
                ("lpszClassName", ctypes.wintypes.LPCWSTR),
                ("hIconSm", ctypes.wintypes.HANDLE)]
# 定义 ShowWindow 函数，用于显示窗口
ShowWindow = ctypes.windll.user32.ShowWindow
# 设置 ShowWindow 函数的参数类型
ShowWindow.argtypes = [ctypes.wintypes.HWND, ctypes.c_int]

# 定义 GenerateDummyWindow 函数，用于生成虚拟窗口
def GenerateDummyWindow(callback, uid):
    # 创建 WNDCLASSEX 结构体对象
    newclass = WNDCLASSEX()
    # 设置 WNDCLASSEX 对象的回调函数
    newclass.lpfnWndProc = callback
    # 设置 WNDCLASSEX 对象的窗口类名
    newclass.lpszClassName = uid.replace("-", "")
    # 注册窗口类
    ATOM = ctypes.windll.user32.RegisterClassExW(ctypes.byref(newclass))
    # 创建窗口
    hwnd = ctypes.windll.user32.CreateWindowExW(0, newclass.lpszClassName, None, WS_POPUP, 0, 0, 0, 0, 0, 0, 0, 0)
    return hwnd

# 定义 TIMERCALLBACK 回调函数类型
TIMERCALLBACK = ctypes.WINFUNCTYPE(None, ctypes.wintypes.HWND, ctypes.wintypes.UINT, ctypes.POINTER(ctypes.wintypes.UINT), ctypes.wintypes.DWORD)

# 定义 SetTimer 函数，用于设置定时器
SetTimer = ctypes.windll.user32.SetTimer
# 设置 SetTimer 函数的返回类型
SetTimer.restype = ctypes.POINTER(ctypes.wintypes.UINT)
# 设置 SetTimer 函数的参数类型
SetTimer.argtypes = [ctypes.wintypes.HWND, ctypes.POINTER(ctypes.wintypes.UINT), ctypes.wintypes.UINT, TIMERCALLBACK]

# 定义 KillTimer 函数，用于销毁定时器
KillTimer = ctypes.windll.user32.KillTimer
# 设置 KillTimer 函数的返回类型
KillTimer.restype = ctypes.wintypes.BOOL
# 设置 KillTimer 函数的参数类型
KillTimer.argtypes = [ctypes.wintypes.HWND, ctypes.POINTER(ctypes.wintypes.UINT)]

# 定义 MSG 结构体
class MSG(ctypes.Structure):
    _fields_ = [ ('HWND', ctypes.wintypes.HWND),
                 ('message', ctypes.wintypes.UINT),
                 ('wParam', ctypes.wintypes.WPARAM),
                 ('lParam', ctypes.wintypes.LPARAM),
                 ('time', ctypes.wintypes.DWORD),
                 ('pt', POINT)]

# 定义 GetMessage 函数，用于获取消息
GetMessage = ctypes.windll.user32.GetMessageW
# 设置 GetMessage 函数的返回类型
GetMessage.restype = ctypes.wintypes.BOOL
# 设置 GetMessage 函数的参数类型
GetMessage.argtypes = [ctypes.POINTER(MSG), ctypes.wintypes.HWND, ctypes.wintypes.UINT, ctypes.wintypes.UINT]

# 定义 TranslateMessage 函数，用于翻译消息
TranslateMessage = ctypes.windll.user32.TranslateMessage
# 设置 TranslateMessage 函数的返回类型
TranslateMessage.restype = ctypes.wintypes.ULONG
# 设置 TranslateMessage 函数的参数类型
TranslateMessage.argtypes = [ctypes.POINTER(MSG)]

# 定义 DispatchMessage 函数，用于分发消息
DispatchMessage = ctypes.windll.user32.DispatchMessageW
# 设置 DispatchMessage 函数的返回类型为 ULONG
DispatchMessage.restype = ctypes.wintypes.ULONG
# 设置 DispatchMessage 函数的参数类型为指向 MSG 结构体的指针
DispatchMessage.argtypes = [ctypes.POINTER(MSG)]

# 定义 LoadIcon 函数，用于加载图标文件
def LoadIcon(iconfilename, small=False):
        # 调用 LoadImage 函数加载图标
        return LoadImage(0,
                         str(iconfilename),
                         IMAGE_ICON,
                         16 if small else 0,
                         16 if small else 0,
                         LR_LOADFROMFILE)

# 定义 NotificationIcon 类
class NotificationIcon(object):
    # 初始化方法
    def __init__(self, iconfilename, tooltip=None):
        # 检查图标文件是否存在
        assert os.path.isfile(str(iconfilename)), "{} doesn't exist".format(iconfilename)
        # 保存图标文件名
        self._iconfile = str(iconfilename)
        # 加载图标
        self._hicon = LoadIcon(self._iconfile, True)
        # 检查图标是否成功加载
        assert self._hicon, "Failed to load {}".format(iconfilename)
        # 初始化变量
        self._die = False
        self._timerid = None
        self._uid = uuid.uuid4()
        # 设置提示信息
        self._tooltip = str(tooltip) if tooltip else ''
        # 初始化信息气泡
        self._info_bubble = None
        # 初始化项目列表
        self.items = []

    # 显示信息气泡
    def _bubble(self, iconinfo):
        if self._info_bubble:
            # 保存信息气泡
            info_bubble = self._info_bubble
            self._info_bubble = None
            # 设置图标信息
            message = str(self._info_bubble)
            iconinfo.uFlags |= NIF_INFO
            iconinfo.szInfo = message
            iconinfo.szInfoTitle = message
            iconinfo.dwInfoFlags = NIIF_INFO
            iconinfo.union.uTimeout = 10000
            # 修改通知区域图标
            Shell_NotifyIcon(NIM_MODIFY, ctypes.pointer(iconinfo))
    # 定义一个私有方法 _run，用于执行系统托盘图标的相关操作
    def _run(self):
        # 注册一个 Windows 消息，用于在任务栏重建时重新创建系统托盘图标
        self.WM_TASKBARCREATED = ctypes.windll.user32.RegisterWindowMessageW('TaskbarCreated')

        # 定义一个窗口过程回调函数，并创建一个虚拟窗口
        self._windowproc = WNDPROC(self._callback)
        self._hwnd = GenerateDummyWindow(self._windowproc, str(self._uid))

        # 定义系统托盘图标的信息
        iconinfo = NOTIFYICONDATA()
        iconinfo.hWnd = self._hwnd
        iconinfo.uID = 100
        iconinfo.uFlags = NIF_ICON | NIF_SHOWTIP | NIF_MESSAGE | (NIF_TIP if self._tooltip else 0)
        iconinfo.uCallbackMessage = WM_MENUCOMMAND
        iconinfo.hIcon = self._hicon
        iconinfo.szTip = self._tooltip

        # 将系统托盘图标添加到任务栏
        Shell_NotifyIcon(NIM_ADD, ctypes.pointer(iconinfo))

        # 保存系统托盘图标的信息
        self.iconinfo = iconinfo

        # 发送一个空消息给虚拟窗口
        PostMessage(self._hwnd, WM_NULL, 0, 0)

        # 定义消息和时间变量
        message = MSG()
        last_time = -1
        ret = None
        # 循环处理消息，直到退出
        while not self._die:
            try:
                ret = GetMessage(ctypes.pointer(message), 0, 0, 0)
                TranslateMessage(ctypes.pointer(message))
                DispatchMessage(ctypes.pointer(message))
            except Exception as err:
                # 捕获异常并打印错误信息
                # print "NotificationIcon error", err, message
                message = MSG()
            # 线程休眠一段时间
            time.sleep(0.125)
        # 打印线程停止的消息，并移除系统托盘图标
        print("Icon thread stopped, removing icon (hicon: %s, hwnd: %s)..." % (self._hicon, self._hwnd))

        # 从任务栏移除系统托盘图标，并销毁虚拟窗口和图标
        Shell_NotifyIcon(NIM_DELETE, ctypes.cast(ctypes.pointer(iconinfo), ctypes.POINTER(NOTIFYICONDATA)))
        ctypes.windll.user32.DestroyWindow(self._hwnd)
        ctypes.windll.user32.DestroyIcon.argtypes = [ctypes.wintypes.HICON]
        ctypes.windll.user32.DestroyIcon(self._hicon)

    # 定义一个点击系统托盘图标时的方法
    def clicked(self):
        self._menu()
    # 定义一个回调函数，处理窗口消息
    def _callback(self, hWnd, msg, wParam, lParam):
        # 检查主线程是否仍然存活
        if msg == WM_TIMER:
            if not any(thread.getName() == 'MainThread' and thread.isAlive()
                       for thread in threading.enumerate()):
                self._die = True
        # 如果消息是菜单命令，并且是鼠标左键释放
        elif msg == WM_MENUCOMMAND and lParam == WM_LBUTTONUP:
            self.clicked()
        # 如果消息是菜单命令，并且是鼠标右键释放
        elif msg == WM_MENUCOMMAND and lParam == WM_RBUTTONUP:
            self._menu()
        # 如果消息是任务栏创建消息
        elif msg == self.WM_TASKBARCREATED: # Explorer 重新启动，重新添加图标
            Shell_NotifyIcon(NIM_ADD, ctypes.pointer(self.iconinfo))
        else:
            # 其他情况下调用默认的窗口过程
            return DefWindowProc(hWnd, msg, wParam, lParam)
        return 1


    # 关闭窗口
    def die(self):
        self._die = True
        PostMessage(self._hwnd, WM_NULL, 0, 0)
        time.sleep(0.2)
        try:
            # 移除系统托盘图标
            Shell_NotifyIcon(NIM_DELETE, self.iconinfo)
        except Exception as err:
            print("Icon remove error", err)
        # 销毁窗口和图标
        ctypes.windll.user32.DestroyWindow(self._hwnd)
        ctypes.windll.user32.DestroyIcon(self._hicon)


    # 处理消息队列中的消息
    def pump(self):
        try:
            while not self._pumpqueue.empty():
                callable = self._pumpqueue.get(False)
                callable()
        except queue_Empty:
            pass


    # 显示通知气泡
    def announce(self, text):
        self._info_bubble = text
# 隐藏控制台窗口
def hideConsole():
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

# 显示控制台窗口
def showConsole():
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 1)

# 检查是否存在控制台窗口
def hasConsole():
    return ctypes.windll.kernel32.GetConsoleWindow() != 0

# 如果作为主程序执行
if __name__ == "__main__":
    import time

    # 打招呼函数
    def greet():
        # 隐藏控制台窗口
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
        # 打印"Hello"
        print("Hello")

    # 退出函数
    def quit():
        ni._die = True

    # 通知函数
    def announce():
        # 显示控制台窗口
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 1)
        # 发送通知消息
        ni.announce("Hello there")

    # 点击函数
    def clicked():
        # 发送通知消息
        ni.announce("Hello")

    # 动态标题函数
    def dynamicTitle():
        return "!The time is: %s" % time.time()

    # 创建通知图标对象
    ni = NotificationIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../trayicon.ico'), "ZeroNet 0.2.9")
    # 设置通知图标的项目
    ni.items = [
        (dynamicTitle, False),
        ('Hello', greet),
        ('Title', False),
        ('!Default', greet),
        ('+Popup bubble', announce),
        'Nothing',
        '--',
        ('Quit', quit)
    ]
    # 设置通知图标的点击事件处理函数
    ni.clicked = clicked
    import atexit

    # 在程序退出时打印消息
    @atexit.register
    def goodbye():
        print("You are now leaving the Python sector.")

    # 运行通知图标程序
    ni._run()
```