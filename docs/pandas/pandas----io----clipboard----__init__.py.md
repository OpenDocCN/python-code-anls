# `D:\src\scipysrc\pandas\pandas\io\clipboard\__init__.py`

```
"""
Pyperclip

A cross-platform clipboard module for Python,
with copy & paste functions for plain text.
By Al Sweigart al@inventwithpython.com
Licence at LICENSES/PYPERCLIP_LICENSE

Usage:
  import pyperclip
  pyperclip.copy('The text to be copied to the clipboard.')
  spam = pyperclip.paste()

  if not pyperclip.is_available():
    print("Copy functionality unavailable!")

On Windows, no additional modules are needed.
On Mac, the pyobjc module is used, falling back to the pbcopy and pbpaste cli
    commands. (These commands should come with OS X.).
On Linux, install xclip, xsel, or wl-clipboard (for "wayland" sessions) via
package manager.
For example, in Debian:
    sudo apt-get install xclip
    sudo apt-get install xsel
    sudo apt-get install wl-clipboard

Otherwise on Linux, you will need the PyQt5 modules installed.

This module does not work with PyGObject yet.

Cygwin is currently not supported.

Security Note: This module runs programs with these names:
    - pbcopy
    - pbpaste
    - xclip
    - xsel
    - wl-copy/wl-paste
    - klipper
    - qdbus
A malicious user could rename or add programs with these names, tricking
Pyperclip into running them with whatever permissions the Python process has.

"""

__version__ = "1.8.2"


import contextlib
import ctypes
from ctypes import (
    c_size_t,
    c_wchar,
    c_wchar_p,
    get_errno,
    sizeof,
)
import os
import platform
from shutil import which as _executable_exists
import subprocess
import time
import warnings

from pandas.errors import (
    PyperclipException,
    PyperclipWindowsException,
)
from pandas.util._exceptions import find_stack_level

# `import PyQt4` sys.exit()s if DISPLAY is not in the environment.
# Thus, we need to detect the presence of $DISPLAY manually
# and not load PyQt4 if it is absent.
HAS_DISPLAY = os.getenv("DISPLAY")

EXCEPT_MSG = """
    Pyperclip could not find a copy/paste mechanism for your system.
    For more information, please visit
    https://pyperclip.readthedocs.io/en/latest/index.html#not-implemented-error
    """

ENCODING = "utf-8"


class PyperclipTimeoutException(PyperclipException):
    pass


def _stringifyText(text) -> str:
    acceptedTypes = (str, int, float, bool)
    if not isinstance(text, acceptedTypes):
        raise PyperclipException(
            f"only str, int, float, and bool values "
            f"can be copied to the clipboard, not {type(text).__name__}"
        )
    return str(text)


def init_osx_pbcopy_clipboard():
    # 在 macOS 上使用 pbcopy 命令复制文本到剪贴板
    def copy_osx_pbcopy(text):
        text = _stringifyText(text)  # 将非字符串值转换为字符串
        with subprocess.Popen(
            ["pbcopy", "w"], stdin=subprocess.PIPE, close_fds=True
        ) as p:
            p.communicate(input=text.encode(ENCODING))

    # 在 macOS 上使用 pbpaste 命令从剪贴板粘贴文本
    def paste_osx_pbcopy():
        with subprocess.Popen(
            ["pbpaste", "r"], stdout=subprocess.PIPE, close_fds=True
        ) as p:
            stdout = p.communicate()[0]
        return stdout.decode(ENCODING)
    # 返回两个函数对象，用于在 macOS 上实现剪贴板复制和粘贴功能
    return copy_osx_pbcopy, paste_osx_pbcopy
# 定义一个函数，初始化在 macOS 上使用 PyObjC 库操作剪贴板的功能
def init_osx_pyobjc_clipboard():
    # 定义一个函数，将字符串复制到剪贴板
    def copy_osx_pyobjc(text):
        """Copy string argument to clipboard"""
        # 将输入的文本转换为字符串（如果不是字符串类型）
        text = _stringifyText(text)  # Converts non-str values to str.
        # 创建一个新的 NSString 对象
        newStr = Foundation.NSString.stringWithString_(text).nsstring()
        # 将 NSString 对象转换为 UTF-8 编码的 NSData 对象
        newData = newStr.dataUsingEncoding_(Foundation.NSUTF8StringEncoding)
        # 获取系统剪贴板对象
        board = AppKit.NSPasteboard.generalPasteboard()
        # 声明剪贴板的数据类型为 NSStringPboardType
        board.declareTypes_owner_([AppKit.NSStringPboardType], None)
        # 将数据设置到剪贴板上
        board.setData_forType_(newData, AppKit.NSStringPboardType)

    # 定义一个函数，从剪贴板获取内容并返回
    def paste_osx_pyobjc():
        """Returns contents of clipboard"""
        # 获取系统剪贴板对象
        board = AppKit.NSPasteboard.generalPasteboard()
        # 获取剪贴板中 NSStringPboardType 类型的内容
        content = board.stringForType_(AppKit.NSStringPboardType)
        return content

    # 返回两个函数引用：复制和粘贴函数
    return copy_osx_pyobjc, paste_osx_pyobjc


# 定义一个函数，初始化使用 Qt 库操作剪贴板的功能
def init_qt_clipboard():
    global QApplication
    # 确保 $DISPLAY 环境变量存在

    # 尝试从 qtpy 中导入 QApplication
    try:
        from qtpy.QtWidgets import QApplication
    # 如果导入失败，则尝试从 PyQt5 中导入 QApplication
    except ImportError:
        try:
            from PyQt5.QtWidgets import QApplication
        # 如果再次导入失败，则尝试从 PyQt4 中导入 QApplication
        except ImportError:
            from PyQt4.QtGui import QApplication

    # 获取当前应用程序实例，如果不存在则创建一个空的 QApplication 实例
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    # 定义一个函数，将字符串复制到 Qt 库的剪贴板
    def copy_qt(text):
        text = _stringifyText(text)  # Converts non-str values to str.
        cb = app.clipboard()
        cb.setText(text)

    # 定义一个函数，从 Qt 库的剪贴板获取文本内容并返回
    def paste_qt() -> str:
        cb = app.clipboard()
        return str(cb.text())

    # 返回两个函数引用：复制和粘贴函数
    return copy_qt, paste_qt


# 定义一个函数，初始化使用 xclip 命令行工具操作剪贴板的功能
def init_xclip_clipboard():
    DEFAULT_SELECTION = "c"
    PRIMARY_SELECTION = "p"

    # 定义一个函数，使用 xclip 命令将文本复制到剪贴板
    def copy_xclip(text, primary=False):
        text = _stringifyText(text)  # Converts non-str values to str.
        selection = DEFAULT_SELECTION
        if primary:
            selection = PRIMARY_SELECTION
        # 使用 subprocess 执行 xclip 命令，并将文本输入到剪贴板
        with subprocess.Popen(
            ["xclip", "-selection", selection], stdin=subprocess.PIPE, close_fds=True
        ) as p:
            p.communicate(input=text.encode(ENCODING))

    # 定义一个函数，使用 xclip 命令从剪贴板获取文本内容并返回
    def paste_xclip(primary=False):
        selection = DEFAULT_SELECTION
        if primary:
            selection = PRIMARY_SELECTION
        # 使用 subprocess 执行 xclip 命令，并从标准输出获取剪贴板内容
        with subprocess.Popen(
            ["xclip", "-selection", selection, "-o"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        ) as p:
            stdout = p.communicate()[0]
        # 当剪贴板为空时，忽略标准错误输出，返回剪贴板内容（解码为字符串）
        return stdout.decode(ENCODING)

    # 返回两个函数引用：复制和粘贴函数
    return copy_xclip, paste_xclip


# 定义一个函数，初始化使用 xsel 命令行工具操作剪贴板的功能
def init_xsel_clipboard():
    DEFAULT_SELECTION = "-b"
    PRIMARY_SELECTION = "-p"
    # 定义函数 copy_xsel，用于将文本复制到剪贴板
    def copy_xsel(text, primary=False):
        # 调用 _stringifyText 函数，将非字符串类型的值转换为字符串
        text = _stringifyText(text)  # Converts non-str values to str.
        
        # 设置默认的选择标志为默认选择
        selection_flag = DEFAULT_SELECTION
        
        # 如果 primary 参数为 True，则将选择标志设置为主选择
        if primary:
            selection_flag = PRIMARY_SELECTION
        
        # 使用 subprocess.Popen 打开一个进程，调用 xsel 命令，将文本输入到剪贴板
        with subprocess.Popen(
            ["xsel", selection_flag, "-i"], stdin=subprocess.PIPE, close_fds=True
        ) as p:
            # 将文本编码为指定编码格式并输入到 subprocess 的标准输入
            p.communicate(input=text.encode(ENCODING))
    
    # 定义函数 paste_xsel，用于从剪贴板粘贴文本
    def paste_xsel(primary=False):
        # 设置默认的选择标志为默认选择
        selection_flag = DEFAULT_SELECTION
        
        # 如果 primary 参数为 True，则将选择标志设置为主选择
        if primary:
            selection_flag = PRIMARY_SELECTION
        
        # 使用 subprocess.Popen 打开一个进程，调用 xsel 命令，从剪贴板读取文本
        with subprocess.Popen(
            ["xsel", selection_flag, "-o"], stdout=subprocess.PIPE, close_fds=True
        ) as p:
            # 读取 subprocess 的标准输出，即从剪贴板得到的文本内容
            stdout = p.communicate()[0]
        
        # 将 subprocess 的标准输出解码为指定编码格式的文本并返回
        return stdout.decode(ENCODING)
    
    # 返回 copy_xsel 函数和 paste_xsel 函数，使其可以在外部使用
    return copy_xsel, paste_xsel
# 初始化 Wayland 剪贴板操作函数
def init_wl_clipboard():
    PRIMARY_SELECTION = "-p"  # 定义主剪贴板选项

    # 复制文本到 Wayland 剪贴板
    def copy_wl(text, primary=False):
        text = _stringifyText(text)  # 将非字符串值转换为字符串
        args = ["wl-copy"]
        if primary:
            args.append(PRIMARY_SELECTION)  # 如果指定主剪贴板，则添加主剪贴板选项
        if not text:
            args.append("--clear")  # 如果文本为空，则添加清除选项
            subprocess.check_call(args, close_fds=True)
        else:
            p = subprocess.Popen(args, stdin=subprocess.PIPE, close_fds=True)
            p.communicate(input=text.encode(ENCODING))  # 将文本编码并传递给子进程

    # 从 Wayland 剪贴板粘贴文本
    def paste_wl(primary=False):
        args = ["wl-paste", "-n"]
        if primary:
            args.append(PRIMARY_SELECTION)  # 如果指定主剪贴板，则添加主剪贴板选项
        p = subprocess.Popen(args, stdout=subprocess.PIPE, close_fds=True)
        stdout, _stderr = p.communicate()
        return stdout.decode(ENCODING)  # 解码并返回从剪贴板获取的文本

    return copy_wl, paste_wl


# 初始化 KDE Klipper 剪贴板操作函数
def init_klipper_clipboard():
    # 复制文本到 Klipper 剪贴板
    def copy_klipper(text):
        text = _stringifyText(text)  # 将非字符串值转换为字符串
        with subprocess.Popen(
            [
                "qdbus",
                "org.kde.klipper",
                "/klipper",
                "setClipboardContents",
                text.encode(ENCODING),
            ],
            stdin=subprocess.PIPE,
            close_fds=True,
        ) as p:
            p.communicate(input=None)  # 将文本内容写入 Klipper 剪贴板

    # 从 Klipper 剪贴板粘贴文本
    def paste_klipper():
        with subprocess.Popen(
            ["qdbus", "org.kde.klipper", "/klipper", "getClipboardContents"],
            stdout=subprocess.PIPE,
            close_fds=True,
        ) as p:
            stdout = p.communicate()[0]

        # 解决 https://bugs.kde.org/show_bug.cgi?id=342874 的问题
        # TODO: https://github.com/asweigart/pyperclip/issues/43
        clipboardContents = stdout.decode(ENCODING)
        assert len(clipboardContents) > 0  # 确保获取的剪贴板内容长度大于零
        if clipboardContents.endswith("\n"):
            clipboardContents = clipboardContents[:-1]  # 如果末尾是换行符，则去除之
        return clipboardContents  # 返回从 Klipper 剪贴板获取的内容

    return copy_klipper, paste_klipper


# 初始化 /dev/clipboard 设备剪贴板操作函数
def init_dev_clipboard_clipboard():
    # 复制文本到 /dev/clipboard 设备剪贴板
    def copy_dev_clipboard(text):
        text = _stringifyText(text)  # 将非字符串值转换为字符串
        if text == "":
            warnings.warn(
                "Pyperclip cannot copy a blank string to the clipboard on Cygwin. "
                "This is effectively a no-op.",
                stacklevel=find_stack_level(),
            )
        if "\r" in text:
            warnings.warn(
                "Pyperclip cannot handle \\r characters on Cygwin.",
                stacklevel=find_stack_level(),
            )

        with open("/dev/clipboard", "w", encoding="utf-8") as fd:
            fd.write(text)  # 将文本写入 /dev/clipboard 设备剪贴板

    # 从 /dev/clipboard 设备剪贴板粘贴文本
    def paste_dev_clipboard() -> str:
        with open("/dev/clipboard", encoding="utf-8") as fd:
            content = fd.read()
        return content  # 返回从 /dev/clipboard 设备剪贴板获取的内容

    return copy_dev_clipboard, paste_dev_clipboard
def init_no_clipboard():
    # 定义一个无剪贴板可用的类
    class ClipboardUnavailable:
        def __call__(self, *args, **kwargs):
            # 调用此类实例时抛出 PyperclipException 异常
            raise PyperclipException(EXCEPT_MSG)

        def __bool__(self) -> bool:
            # 始终返回 False，表示剪贴板不可用
            return False

    # 返回两个 ClipboardUnavailable 实例作为无剪贴板情况下的剪贴板对象
    return ClipboardUnavailable(), ClipboardUnavailable()


# Windows相关剪贴板函数：
class CheckedCall:
    def __init__(self, f) -> None:
        super().__setattr__("f", f)

    def __call__(self, *args):
        # 调用被包装的函数 f，并检查返回值和错误码
        ret = self.f(*args)
        if not ret and get_errno():
            # 如果返回值为假且存在错误码，则抛出 PyperclipWindowsException 异常
            raise PyperclipWindowsException("Error calling " + self.f.__name__)
        return ret

    def __setattr__(self, key, value):
        # 设置属性时委托给被包装的函数 f 的相应属性设置
        setattr(self.f, key, value)


def init_windows_clipboard():
    # 导入所需的 Windows API 类型和函数
    global HGLOBAL, LPVOID, DWORD, LPCSTR, INT
    global HWND, HINSTANCE, HMENU, BOOL, UINT, HANDLE
    from ctypes.wintypes import (
        BOOL,
        DWORD,
        HANDLE,
        HGLOBAL,
        HINSTANCE,
        HMENU,
        HWND,
        INT,
        LPCSTR,
        LPVOID,
        UINT,
    )

    # 获取相关 DLL 和库函数
    windll = ctypes.windll
    msvcrt = ctypes.CDLL("msvcrt")

    # 安全包装需要检查的 Windows API 调用
    safeCreateWindowExA = CheckedCall(windll.user32.CreateWindowExA)
    safeCreateWindowExA.argtypes = [
        DWORD,
        LPCSTR,
        LPCSTR,
        DWORD,
        INT,
        INT,
        INT,
        INT,
        HWND,
        HMENU,
        HINSTANCE,
        LPVOID,
    ]
    safeCreateWindowExA.restype = HWND

    safeDestroyWindow = CheckedCall(windll.user32.DestroyWindow)
    safeDestroyWindow.argtypes = [HWND]
    safeDestroyWindow.restype = BOOL

    OpenClipboard = windll.user32.OpenClipboard
    OpenClipboard.argtypes = [HWND]
    OpenClipboard.restype = BOOL

    safeCloseClipboard = CheckedCall(windll.user32.CloseClipboard)
    safeCloseClipboard.argtypes = []
    safeCloseClipboard.restype = BOOL

    safeEmptyClipboard = CheckedCall(windll.user32.EmptyClipboard)
    safeEmptyClipboard.argtypes = []
    safeEmptyClipboard.restype = BOOL

    safeGetClipboardData = CheckedCall(windll.user32.GetClipboardData)
    safeGetClipboardData.argtypes = [UINT]
    safeGetClipboardData.restype = HANDLE

    safeSetClipboardData = CheckedCall(windll.user32.SetClipboardData)
    safeSetClipboardData.argtypes = [UINT, HANDLE]
    safeSetClipboardData.restype = HANDLE

    safeGlobalAlloc = CheckedCall(windll.kernel32.GlobalAlloc)
    safeGlobalAlloc.argtypes = [UINT, c_size_t]
    safeGlobalAlloc.restype = HGLOBAL

    safeGlobalLock = CheckedCall(windll.kernel32.GlobalLock)
    safeGlobalLock.argtypes = [HGLOBAL]
    safeGlobalLock.restype = LPVOID

    safeGlobalUnlock = CheckedCall(windll.kernel32.GlobalUnlock)
    safeGlobalUnlock.argtypes = [HGLOBAL]
    safeGlobalUnlock.restype = BOOL

    wcslen = CheckedCall(msvcrt.wcslen)
    wcslen.argtypes = [c_wchar_p]
    wcslen.restype = UINT

    # 定义一些常量
    GMEM_MOVEABLE = 0x0002
    CF_UNICODETEXT = 13

    @contextlib.contextmanager
    def window():
        """
        Context that provides a valid Windows hwnd.
        """
        # 创建一个窗口句柄 hwnd，用 "STATIC" 作为预定义的 lpClass 就足够了
        hwnd = safeCreateWindowExA(
            0, b"STATIC", None, 0, 0, 0, 0, 0, None, None, None, None
        )
        try:
            yield hwnd  # 返回 hwnd，在退出此函数前可作为上下文管理器使用
        finally:
            safeDestroyWindow(hwnd)  # 确保在退出前销毁窗口句柄 hwnd

    @contextlib.contextmanager
    def clipboard(hwnd):
        """
        Context manager that opens the clipboard and prevents
        other applications from modifying the clipboard content.
        """
        # 尝试获取剪贴板句柄，最多尝试 500 毫秒
        t = time.time() + 0.5
        success = False
        while time.time() < t:
            success = OpenClipboard(hwnd)
            if success:
                break
            time.sleep(0.01)
        if not success:
            raise PyperclipWindowsException("Error calling OpenClipboard")

        try:
            yield  # 打开剪贴板后，允许执行剪贴板操作
        finally:
            safeCloseClipboard()  # 确保在退出前关闭剪贴板

    def copy_windows(text):
        # This function is heavily based on
        # http://msdn.com/ms649016#_win32_Copying_Information_to_the_Clipboard

        text = _stringifyText(text)  # 将非字符串值转换为字符串

        with window() as hwnd:
            # 需要一个有效的窗口句柄才能复制内容
            with clipboard(hwnd):
                safeEmptyClipboard()  # 清空剪贴板

                if text:
                    # 分配一个带有 GMEM_MOVEABLE 标志的内存对象
                    count = wcslen(text) + 1
                    handle = safeGlobalAlloc(GMEM_MOVEABLE, count * sizeof(c_wchar))
                    locked_handle = safeGlobalLock(handle)

                    ctypes.memmove(
                        c_wchar_p(locked_handle),
                        c_wchar_p(text),
                        count * sizeof(c_wchar),
                    )

                    safeGlobalUnlock(handle)  # 解锁内存对象
                    safeSetClipboardData(CF_UNICODETEXT, handle)  # 将数据设置到剪贴板中
    # 定义一个名为 paste_windows 的函数，用于粘贴文本到 Windows 剪贴板
    def paste_windows():
        # 使用 clipboard(None) 上下文管理器，确保在操作完成后剪贴板自动释放
        with clipboard(None):
            # 调用 safeGetClipboardData(CF_UNICODETEXT) 获取剪贴板数据的句柄
            handle = safeGetClipboardData(CF_UNICODETEXT)
            # 如果未能获取到句柄（可能是因为剪贴板为空）
            if not handle:
                # 返回空字符串，表示剪贴板中无可用文本
                return ""
            # 将句柄转换为 c_wchar_p 类型，然后获取其值（即剪贴板中的文本内容），并返回
            return c_wchar_p(handle).value

    # 返回 paste_windows 函数本身
    return copy_windows, paste_windows
# 定义一个函数 init_wsl_clipboard()，返回两个函数：copy_wsl 和 paste_wsl
def init_wsl_clipboard():
    # 定义 copy_wsl 函数，将输入的文本转换为字符串，并通过 subprocess 调用 clip.exe 将其复制到剪贴板
    def copy_wsl(text):
        text = _stringifyText(text)  # Converts non-str values to str.
        with subprocess.Popen(["clip.exe"], stdin=subprocess.PIPE, close_fds=True) as p:
            p.communicate(input=text.encode(ENCODING))

    # 定义 paste_wsl 函数，通过 subprocess 调用 powershell.exe 执行 Get-Clipboard 命令获取剪贴板内容，
    # 并返回结果（去除末尾的 "\r\n" 后解码为指定编码的字符串）
    def paste_wsl():
        with subprocess.Popen(
            ["powershell.exe", "-command", "Get-Clipboard"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        ) as p:
            stdout = p.communicate()[0]
        # WSL 在内容末尾追加了 "\r\n"，返回结果时去除这部分并解码
        return stdout[:-2].decode(ENCODING)

    # 返回 copy_wsl 和 paste_wsl 两个函数
    return copy_wsl, paste_wsl


# 在 determine_clipboard() 函数中自动检测剪贴板机制并导入相关模块：
def determine_clipboard():
    """
    确定操作系统/平台，并相应设置 copy() 和 paste() 函数。
    """
    global Foundation, AppKit, qtpy, PyQt4, PyQt5

    # 对 CYGWIN 平台的设置：
    if (
        "cygwin" in platform.system().lower()
    ):  # Cygwin 返回多种可能的 platform.system() 值，如 'CYGWIN_NT-6.1'
        # FIXME(pyperclip#55): pyperclip 目前不支持 Cygwin，参见 https://github.com/asweigart/pyperclip/issues/55
        if os.path.exists("/dev/clipboard"):
            warnings.warn(
                "Pyperclip's support for Cygwin is not perfect, "
                "see https://github.com/asweigart/pyperclip/issues/55",
                stacklevel=find_stack_level(),
            )
            return init_dev_clipboard_clipboard()

    # 对 WINDOWS 平台的设置：
    elif os.name == "nt" or platform.system() == "Windows":
        return init_windows_clipboard()

    # 对 Linux 平台的设置：
    if platform.system() == "Linux":
        if _executable_exists("wslconfig.exe"):
            return init_wsl_clipboard()

    # 对 macOS 平台的设置：
    if os.name == "mac" or platform.system() == "Darwin":
        try:
            import AppKit
            import Foundation  # 检查是否安装了 pyobjc
        except ImportError:
            return init_osx_pbcopy_clipboard()
        else:
            return init_osx_pyobjc_clipboard()

    # 对 LINUX 平台的设置（未完整列出，此处可能包含进一步的设置逻辑）：
    # 如果有显示设备可用
    if HAS_DISPLAY:
        # 如果环境变量中存在 WAYLAND_DISPLAY，并且 wl-copy 可执行文件存在
        if os.environ.get("WAYLAND_DISPLAY") and _executable_exists("wl-copy"):
            # 初始化 Wayland 剪贴板
            return init_wl_clipboard()
        # 如果 xsel 可执行文件存在
        if _executable_exists("xsel"):
            # 初始化 xsel 剪贴板
            return init_xsel_clipboard()
        # 如果 xclip 可执行文件存在
        if _executable_exists("xclip"):
            # 初始化 xclip 剪贴板
            return init_xclip_clipboard()
        # 如果 klipper 和 qdbus 可执行文件都存在
        if _executable_exists("klipper") and _executable_exists("qdbus"):
            # 初始化 klipper 剪贴板
            return init_klipper_clipboard()

        try:
            # 尝试导入 qtpy 库，qtpy 是一个抽象层，用于在 PyQt 或 PySide 之间进行统一的 API 调用
            import qtpy
        except ImportError:
            # 如果 qtpy 未安装，尝试导入 PyQt5
            try:
                import PyQt5
            except ImportError:
                # 如果 PyQt5 未安装，尝试导入 PyQt4
                try:
                    import PyQt4
                except ImportError:
                    pass  # 对于所有非 ImportError 异常，我们希望尽快失败
                else:
                    # 如果成功导入 PyQt4，初始化 Qt 剪贴板
                    return init_qt_clipboard()
            else:
                # 如果成功导入 PyQt5，初始化 Qt 剪贴板
                return init_qt_clipboard()
        else:
            # 如果成功导入 qtpy，初始化 Qt 剪贴板
            return init_qt_clipboard()

    # 如果没有显示设备可用，初始化无剪贴板支持
    return init_no_clipboard()
# 明确设置剪贴板机制。剪贴板机制是指复制（copy）和粘贴（paste）功能与操作系统交互的方式。
# clipboard 参数必须是以下之一：
#   - pbcopy
#   - pyobjc（macOS 上的默认值）
#   - qt
#   - xclip
#   - xsel
#   - klipper
#   - windows（Windows 上的默认值）
#   - no（当找不到剪贴板机制时使用的值）
def set_clipboard(clipboard):
    global copy, paste

    # 定义不同剪贴板机制对应的初始化函数
    clipboard_types = {
        "pbcopy": init_osx_pbcopy_clipboard,
        "pyobjc": init_osx_pyobjc_clipboard,
        "qt": init_qt_clipboard,  # TODO - 将此分解为 'qtpy'、'pyqt4' 和 'pyqt5'
        "xclip": init_xclip_clipboard,
        "xsel": init_xsel_clipboard,
        "wl-clipboard": init_wl_clipboard,
        "klipper": init_klipper_clipboard,
        "windows": init_windows_clipboard,
        "no": init_no_clipboard,
    }

    # 如果指定的 clipboard 不在定义的剪贴板类型中，则引发 ValueError
    if clipboard not in clipboard_types:
        allowed_clipboard_types = [repr(_) for _ in clipboard_types]
        raise ValueError(
            f"Argument must be one of {', '.join(allowed_clipboard_types)}"
        )

    # 设置 pyperclip 的 copy() 和 paste() 函数为所选 clipboard 类型对应的函数
    copy, paste = clipboard_types[clipboard]()


def lazy_load_stub_copy(text):
    """
    copy() 的存根函数，在调用时加载真实的 copy() 函数，以便后续调用使用真实的 copy() 函数。
    
    这允许用户在导入 pyperclip 时不会自动运行 determine_clipboard()，后者会自动选择剪贴板机制。
    如果选择了内存密集型的 PyQt4 模块，但用户将立即调用 set_clipboard() 来使用另一种剪贴板机制，这可能会成为问题。
    
    这个存根函数实现的惰性加载，给用户一个机会来调用 set_clipboard() 选择另一种剪贴板机制。
    或者，如果用户在调用 set_clipboard() 之前简单地调用 copy() 或 paste()，则会回退到 determine_clipboard() 自动选择的剪贴板机制。
    """
    global copy, paste
    # 调用 determine_clipboard() 加载真实的 copy() 和 paste() 函数
    copy, paste = determine_clipboard()
    return copy(text)


def lazy_load_stub_paste():
    """
    paste() 的存根函数，在调用时加载真实的 paste() 函数，以便后续调用使用真实的 paste() 函数。
    
    这允许用户在导入 pyperclip 时不会自动运行 determine_clipboard()，后者会自动选择剪贴板机制。
    如果选择了内存密集型的 PyQt4 模块，但用户将立即调用 set_clipboard() 来使用另一种剪贴板机制，这可能会成为问题。
    
    这个存根函数实现的惰性加载，给用户一个机会来调用 set_clipboard() 选择另一种剪贴板机制。
    或者，如果用户
    # 在全局作用域中声明全局变量 copy 和 paste
    global copy, paste
    # 调用 determine_clipboard() 函数获取剪贴板操作函数的引用，并分别赋值给 copy 和 paste
    copy, paste = determine_clipboard()
    # 调用 paste() 函数，返回其结果
    return paste()
# 返回一个布尔值，指示是否已经完成剪贴板的懒加载，并且复制和粘贴函数现在是可用的
def is_available() -> bool:
    return copy != lazy_load_stub_copy and paste != lazy_load_stub_paste


# 初始情况下，将 copy() 和 paste() 设置为懒加载包装器，它们将在首次使用时将 `copy` 和 `paste` 设置为真实函数，
# 除非先调用 set_clipboard() 或 determine_clipboard()。
copy, paste = lazy_load_stub_copy, lazy_load_stub_paste


def waitForPaste(timeout=None):
    """此函数调用会阻塞，直到剪贴板上存在非空文本字符串。它返回这个文本。

    如果设置了超时时间（以秒为单位），并且在剪贴板上未放置非空文本而已经过去了指定的时间，此函数将引发 PyperclipTimeoutException 异常。"""
    startTime = time.time()
    while True:
        clipboardText = paste()
        if clipboardText != "":
            return clipboardText
        time.sleep(0.01)

        if timeout is not None and time.time() > startTime + timeout:
            raise PyperclipTimeoutException(
                "waitForPaste() timed out after " + str(timeout) + " seconds."
            )


def waitForNewPaste(timeout=None):
    """此函数调用会阻塞，直到剪贴板上存在与首次调用此函数时不同的新文本字符串。它返回这个文本。

    如果设置了超时时间（以秒为单位），并且在剪贴板上未放置非空文本而已经过去了指定的时间，此函数将引发 PyperclipTimeoutException 异常。"""
    startTime = time.time()
    originalText = paste()
    while True:
        currentText = paste()
        if currentText != originalText:
            return currentText
        time.sleep(0.01)

        if timeout is not None and time.time() > startTime + timeout:
            raise PyperclipTimeoutException(
                "waitForNewPaste() timed out after " + str(timeout) + " seconds."
            )


# 将以下变量添加到 __all__ 列表中，使它们在 `from module import *` 语句中可用
__all__ = [
    "copy",
    "paste",
    "waitForPaste",
    "waitForNewPaste",
    "set_clipboard",
    "determine_clipboard",
]

# pandas 的别名
clipboard_get = paste
clipboard_set = copy
```