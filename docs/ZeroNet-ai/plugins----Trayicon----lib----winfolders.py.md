# `ZeroNet\plugins\Trayicon\lib\winfolders.py`

```
# 导入 ctypes 库，并使用 _syntax 来遮盖它们，以避免自动补全 IDE 的干扰
import ctypes as _ctypes
# 从 ctypes 中导入 create_unicode_buffer 函数，并使用 _cub 来遮盖它
from ctypes import create_unicode_buffer as _cub
# 从 ctypes.wintypes 中导入 HWND、HANDLE、DWORD、LPCWSTR、MAX_PATH，并使用 _HWND、_HANDLE、_DWORD、_LPCWSTR、_MAX_PATH 来遮盖它们
from ctypes.wintypes import HWND as _HWND, HANDLE as _HANDLE, DWORD as _DWORD, LPCWSTR as _LPCWSTR, MAX_PATH as _MAX_PATH
# 从 shell32 中导入 SHGetFolderPath 函数，并使用 _SHGetFolderPath 来遮盖它
_SHGetFolderPath = _ctypes.windll.shell32.SHGetFolderPathW

# 公共特殊文件夹常量
DESKTOP=                             0
PROGRAMS=                            2
MYDOCUMENTS=                         5
FAVORITES=                           6
STARTUP=                             7
RECENT=                              8
SENDTO=                              9
STARTMENU=                          11
MYMUSIC=                            13
MYVIDEOS=                           14
NETHOOD=                            19
FONTS=                              20
TEMPLATES=                          21
ALLUSERSSTARTMENU=                  22
ALLUSERSPROGRAMS=                   23
ALLUSERSSTARTUP=                    24
ALLUSERSDESKTOP=                    25
APPLICATIONDATA=                    26
PRINTHOOD=                          27
LOCALSETTINGSAPPLICATIONDATA=       28
ALLUSERSFAVORITES=                  31
LOCALSETTINGSTEMPORARYINTERNETFILES=32
COOKIES=                            33
LOCALSETTINGSHISTORY=               34
ALLUSERSAPPLICATIONDATA=            35

# 定义函数 get，接收一个整数参数 intFolder
def get(intFolder):
    # 设置 _SHGetFolderPath 函数的参数类型
    _SHGetFolderPath.argtypes = [_HWND, _ctypes.c_int, _HANDLE, _DWORD, _LPCWSTR]
    # 创建一个最大路径长度的 Unicode 缓冲区
    auPathBuffer = _cub(_MAX_PATH)
    # 调用 _SHGetFolderPath 函数，将结果存储在 auPathBuffer 中
    exit_code = _SHGetFolderPath(0, intFolder, 0, 0, auPathBuffer)
    # 返回 auPathBuffer 的值
    return auPathBuffer.value

# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 导入 os 库
    import os
    # 打印 STARTUP 文件夹的路径
    print(get(STARTUP))
    # 打开 STARTUP 文件夹下的 zeronet.cmd 文件，并写入内容
    open(get(STARTUP) + "\\zeronet.cmd", "w").write("cd /D %s\r\nzeronet.py" % os.getcwd())
```