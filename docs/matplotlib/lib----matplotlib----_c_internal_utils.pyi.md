# `D:\src\scipysrc\matplotlib\lib\matplotlib\_c_internal_utils.pyi`

```py
# 检查当前显示器是否有效
def display_is_valid() -> bool:
    ...

# 获取当前 Win32 前台窗口的句柄
def Win32_GetForegroundWindow() -> int | None:
    ...

# 设置当前 Win32 前台窗口的句柄
def Win32_SetForegroundWindow(hwnd: int) -> None:
    ...

# 设置当前进程的 DPI 感知等级为最大
def Win32_SetProcessDpiAwareness_max() -> None:
    ...

# 设置当前进程的显式应用程序用户模型 ID
def Win32_SetCurrentProcessExplicitAppUserModelID(appid: str) -> None:
    ...

# 获取当前进程的显式应用程序用户模型 ID
def Win32_GetCurrentProcessExplicitAppUserModelID() -> str | None:
    ...
```