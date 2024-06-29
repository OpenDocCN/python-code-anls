# `D:\src\scipysrc\matplotlib\src\_c_internal_utils.cpp`

```
/* Python.h 必须在任何系统头文件之前包含，
   以确保可见性宏被正确设置。 */
#include <Python.h>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
// 用于最新的 HiDPI API 支持的 Windows 10。
#define WINVER 0x0A00
#define _WIN32_WINNT 0x0A00
#endif

#include <pybind11/pybind11.h>

#ifdef __linux__
#include <dlfcn.h>
#endif

#ifdef _WIN32
#include <Objbase.h>
#include <Shobjidl.h>
#include <Windows.h>
// 非 Windows 系统下不使用的宏。
#define UNUSED_ON_NON_WINDOWS(x) x
#else
#define UNUSED_ON_NON_WINDOWS Py_UNUSED
#endif

namespace py = pybind11;
using namespace pybind11::literals;

// 检查当前环境中是否可以有效显示图形界面。
static bool
mpl_display_is_valid(void)
{
#ifdef __linux__
    void* libX11;
    // getenv 的检查是多余的，但可以提高性能，因为比 dlopen() 要快得多。
    if (getenv("DISPLAY")
        && (libX11 = dlopen("libX11.so.6", RTLD_LAZY))) {
        typedef struct Display* (*XOpenDisplay_t)(char const*);
        typedef int (*XCloseDisplay_t)(struct Display*);
        struct Display* display = NULL;
        // 尝试从动态链接库中加载并调用 XOpenDisplay 和 XCloseDisplay 函数。
        XOpenDisplay_t XOpenDisplay = (XOpenDisplay_t)dlsym(libX11, "XOpenDisplay");
        XCloseDisplay_t XCloseDisplay = (XCloseDisplay_t)dlsym(libX11, "XCloseDisplay");
        if (XOpenDisplay && XCloseDisplay
                && (display = XOpenDisplay(NULL))) {
            XCloseDisplay(display);
        }
        // 关闭动态链接库 libX11。
        if (dlclose(libX11)) {
            throw std::runtime_error(dlerror());
        }
        // 如果成功打开了显示器，返回 true。
        if (display) {
            return true;
        }
    }
    void* libwayland_client;
    // 检查环境变量 WAYLAND_DISPLAY 是否设置，并尝试加载 libwayland-client.so.0。
    if (getenv("WAYLAND_DISPLAY")
        && (libwayland_client = dlopen("libwayland-client.so.0", RTLD_LAZY))) {
        typedef struct wl_display* (*wl_display_connect_t)(char const*);
        typedef void (*wl_display_disconnect_t)(struct wl_display*);
        struct wl_display* display = NULL;
        // 尝试从 libwayland-client.so.0 中加载并调用 wl_display_connect 和 wl_display_disconnect 函数。
        wl_display_connect_t wl_display_connect =
            (wl_display_connect_t)dlsym(libwayland_client, "wl_display_connect");
        wl_display_disconnect_t wl_display_disconnect =
            (wl_display_disconnect_t)dlsym(libwayland_client, "wl_display_disconnect");
        if (wl_display_connect && wl_display_disconnect
                && (display = wl_display_connect(NULL))) {
            wl_display_disconnect(display);
        }
        // 关闭动态链接库 libwayland-client.so.0。
        if (dlclose(libwayland_client)) {
            throw std::runtime_error(dlerror());
        }
        // 如果成功连接了显示器，返回 true。
        if (display) {
            return true;
        }
    }
    // 默认情况下返回 false。
    return false;
#else
    // 在非 Linux 系统下，始终返回 true。
    return true;
#endif
}

// 获取当前进程的应用程序用户模型 ID。
static py::object
mpl_GetCurrentProcessExplicitAppUserModelID(void)
{
#ifdef _WIN32
    wchar_t* appid = NULL;
    // 获取当前进程的应用程序用户模型 ID，并检查返回状态。
    HRESULT hr = GetCurrentProcessExplicitAppUserModelID(&appid);
    if (FAILED(hr)) {
        PyErr_SetFromWindowsErr(hr);
        throw py::error_already_set();
    }
    // 将 wchar_t* 转换为 Python 字符串对象。
    auto py_appid = py::cast(appid);
    // 释放从 GetCurrentProcessExplicitAppUserModelID 中分配的内存。
    CoTaskMemFree(appid);
    return py_appid;
#else
    // 在非 Windows 系统下，返回 None。
    return py::none();
#endif
}

// 设置当前进程的应用程序用户模型 ID。
static void
mpl_SetCurrentProcessExplicitAppUserModelID(const wchar_t* UNUSED_ON_NON_WINDOWS(appid))
{
#ifdef _WIN32
    // Windows 下设置当前进程的应用程序用户模型 ID。
    // 这里可以添加相应的 Windows 特定实现代码。
#endif
}
    # 调用 Windows API 设置当前进程的显式应用用户模型ID
    HRESULT hr = SetCurrentProcessExplicitAppUserModelID(appid);
    # 检查操作结果，如果失败，则根据错误码设置 Python 异常
    if (FAILED(hr)) {
        PyErr_SetFromWindowsErr(hr);
        # 抛出 Python 异常给上层调用者
        throw py::error_already_set();
    }
#endif
}

// 返回当前前台窗口的句柄对象，仅在 Windows 平台下有效
static py::object
mpl_GetForegroundWindow(void)
{
#ifdef _WIN32
  // 获取当前前台窗口的句柄
  if (HWND hwnd = GetForegroundWindow()) {
    // 将句柄封装成 Python Capsule 对象并返回
    return py::capsule(hwnd, "HWND");
  } else {
    // 如果未获取到前台窗口句柄，返回 Python 的 None 对象
    return py::none();
  }
#else
  // 在非 Windows 平台下，返回 Python 的 None 对象
  return py::none();
#endif
}

// 设置指定窗口为前台窗口，仅在 Windows 平台下有效
static void
mpl_SetForegroundWindow(py::capsule UNUSED_ON_NON_WINDOWS(handle_p))
{
#ifdef _WIN32
    // 检查传入的 handle_p 是否为 HWND 类型的 Capsule
    if (handle_p.name() != "HWND") {
        // 若不是，则抛出运行时错误
        throw std::runtime_error("Handle must be a value returned from Win32_GetForegroundWindow");
    }
    // 将 Capsule 中的指针转换为 HWND 类型
    HWND handle = static_cast<HWND>(handle_p.get_pointer());
    // 尝试设置指定窗口为前台窗口，若失败则抛出运行时错误
    if (!SetForegroundWindow(handle)) {
        throw std::runtime_error("Error setting window");
    }
#endif
}

// 设置进程 DPI 感知性的最大值，仅在 Windows 平台下有效
static void
mpl_SetProcessDpiAwareness_max(void)
{
#ifdef _WIN32
#ifdef _DPI_AWARENESS_CONTEXTS_
    // 这些函数和选项是在较新的 Windows 10 更新中添加的，因此需要动态加载
    typedef BOOL (WINAPI *IsValidDpiAwarenessContext_t)(DPI_AWARENESS_CONTEXT);
    typedef BOOL (WINAPI *SetProcessDpiAwarenessContext_t)(DPI_AWARENESS_CONTEXT);

    // 加载 user32.dll 动态链接库
    HMODULE user32 = LoadLibrary("user32.dll");
    // 获取 IsValidDpiAwarenessContext 函数指针
    IsValidDpiAwarenessContext_t IsValidDpiAwarenessContextPtr =
        (IsValidDpiAwarenessContext_t)GetProcAddress(
            user32, "IsValidDpiAwarenessContext");
    // 获取 SetProcessDpiAwarenessContext 函数指针
    SetProcessDpiAwarenessContext_t SetProcessDpiAwarenessContextPtr =
        (SetProcessDpiAwarenessContext_t)GetProcAddress(
            user32, "SetProcessDpiAwarenessContext");
    // 定义 DPI 感知性的上下文数组
    DPI_AWARENESS_CONTEXT ctxs[3] = {
        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,  // Win10 Creators Update
        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE,     // Win10
        DPI_AWARENESS_CONTEXT_SYSTEM_AWARE};         // Win10
    // 如果函数指针有效，则尝试设置进程的 DPI 感知性
    if (IsValidDpiAwarenessContextPtr != NULL
            && SetProcessDpiAwarenessContextPtr != NULL) {
        for (int i = 0; i < sizeof(ctxs) / sizeof(DPI_AWARENESS_CONTEXT); ++i) {
            if (IsValidDpiAwarenessContextPtr(ctxs[i])) {
                SetProcessDpiAwarenessContextPtr(ctxs[i]);
                break;
            }
        }
    } else {
        // 若无法动态加载，回退至 SetProcessDPIAware 函数（Vista 后的 Windows 版本支持）
        SetProcessDPIAware();
    }
    // 释放 user32.dll 动态链接库
    FreeLibrary(user32);
#else
    // 若不支持 _DPI_AWARENESS_CONTEXTS_，则回退至 SetProcessDPIAware 函数（Vista 后的 Windows 版本支持）
    SetProcessDPIAware();
#endif
#endif
}

// 定义模块的入口点，为 Python 提供接口函数
PYBIND11_MODULE(_c_internal_utils, m)
{
    // 绑定函数 display_is_valid 到 Python 中的 mpl_display_is_valid 函数
    m.def(
        "display_is_valid", &mpl_display_is_valid,
        R"""(        --
        检查当前 X11 或 Wayland 显示是否有效。

        在 Linux 中，如果 $DISPLAY 设置并且 XOpenDisplay(NULL) 成功，或者 $WAYLAND_DISPLAY 设置并且 wl_display_connect(NULL) 成功，则返回 True。

        在其他平台上，始终返回 True。)""");
    // 绑定函数 Win32_GetCurrentProcessExplicitAppUserModelID 到 Python 中的 mpl_GetCurrentProcessExplicitAppUserModelID 函数
    m.def(
        "Win32_GetCurrentProcessExplicitAppUserModelID",
        &mpl_GetCurrentProcessExplicitAppUserModelID,
        R"""(        --
        Windows GetCurrentProcessExplicitAppUserModelID 的包装器。

        在非 Windows 平台上，始终返回 None。)""");
}
    m.def(
        "Win32_SetCurrentProcessExplicitAppUserModelID",
        &mpl_SetCurrentProcessExplicitAppUserModelID,
        "appid"_a, py::pos_only(),
        R"""(
        Wrapper for Windows's SetCurrentProcessExplicitAppUserModelID.

        On non-Windows platforms, does nothing.
        )"""
    );
    m.def(
        "Win32_GetForegroundWindow", &mpl_GetForegroundWindow,
        R"""(
        Wrapper for Windows' GetForegroundWindow.

        On non-Windows platforms, always returns None.
        )"""
    );
    m.def(
        "Win32_SetForegroundWindow", &mpl_SetForegroundWindow,
        "hwnd"_a,
        R"""(
        Wrapper for Windows' SetForegroundWindow.

        On non-Windows platforms, does nothing.
        )"""
    );
    m.def(
        "Win32_SetProcessDpiAwareness_max", &mpl_SetProcessDpiAwareness_max,
        R"""(
        Set Windows' process DPI awareness to best option available.

        On non-Windows platforms, does nothing.
        )"""
    );


这些注释解释了每个函数在代码中的作用和行为。每个注释都描述了函数的功能及其在非Windows平台上的行为。
}



# 这行代码是一个单独的右花括号 '}'，用于结束一个代码块或函数的定义
```