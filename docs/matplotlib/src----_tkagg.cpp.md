# `D:\src\scipysrc\matplotlib\src\_tkagg.cpp`

```py
// -*- mode: c++; c-basic-offset: 4 -*-

// Where is PIL?
//
// Many years ago, Matplotlib used to include code from PIL (the Python Imaging
// Library).  Since then, the code has changed a lot - the organizing principle
// and methods of operation are now quite different.  Because our review of
// the codebase showed that all the code that came from PIL was removed or
// rewritten, we have removed the PIL licensing information.  If you want PIL,
// you can get it at https://python-pillow.org/

#include <Python.h>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
// Windows 8.1
#define WINVER 0x0603
#define _WIN32_WINNT 0x0603
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace pybind11::literals;

#ifdef _WIN32
#define WIN32_DLL
#endif
#ifdef __CYGWIN__
/*
 * Unfortunately cygwin's libdl inherits restrictions from the underlying
 * Windows OS, at least currently. Therefore, a symbol may be loaded from a
 * module by dlsym() only if it is really located in the given module,
 * dependencies are not included. So we have to use native WinAPI on Cygwin
 * also.
 */
#define WIN32_DLL
static inline PyObject *PyErr_SetFromWindowsErr(int ierr) {
    PyErr_SetString(PyExc_OSError, "Call to EnumProcessModules failed");
    return NULL;
}
#endif

#ifdef WIN32_DLL
#include <vector>

#include <windows.h>
#include <commctrl.h>
#define PSAPI_VERSION 1
#include <psapi.h>  // Must be linked with 'psapi' library
#define dlsym GetProcAddress
#define UNUSED_ON_NON_WINDOWS(x) x
// Check for old headers that do not defined HiDPI functions and constants.
#if defined(__MINGW64_VERSION_MAJOR)
static_assert(__MINGW64_VERSION_MAJOR >= 6,
              "mingw-w64-x86_64-headers >= 6 are required when compiling with MinGW");
#endif
#else
#include <dlfcn.h>
#define UNUSED_ON_NON_WINDOWS Py_UNUSED
#endif

// Include our own excerpts from the Tcl / Tk headers
#include "_tkmini.h"

// Template function to convert Python object to void pointer of type T
template <class T>
static T
convert_voidptr(const py::object &obj)
{
    auto result = static_cast<T>(PyLong_AsVoidPtr(obj.ptr()));
    if (PyErr_Occurred()) {
        throw py::error_already_set();
    }
    return result;
}

// Global variables for Tk functions. We load these symbols from the tkinter
// extension module or loaded Tk libraries at run-time.
static Tk_FindPhoto_t TK_FIND_PHOTO;
static Tk_PhotoPutBlock_t TK_PHOTO_PUT_BLOCK;
// Global variables for Tcl functions. We load these symbols from the tkinter
// extension module or loaded Tcl libraries at run-time.
static Tcl_SetVar_t TCL_SETVAR;

// Function to perform Tk image blitting
static void
mpl_tk_blit(py::object interp_obj, const char *photo_name,
            py::array_t<unsigned char> data, int comp_rule,
            std::tuple<int, int, int, int> offset, std::tuple<int, int, int, int> bbox)
{
    auto interp = convert_voidptr<Tcl_Interp *>(interp_obj);

    Tk_PhotoHandle photo;
    //`
    # 检查 photo 是否为有效的 Tk_PhotoHandle 对象，如果无效则抛出异常
    if (!(photo = TK_FIND_PHOTO(interp, photo_name))) {
        throw py::value_error("Failed to extract Tk_PhotoHandle");
    }

    # 获取 data 的三维矩阵数据，检查其维度和可写标志
    auto data_ptr = data.mutable_unchecked<3>();  
    # 检查 data 的第三维度是否为 4，确保数据是 RGBA 格式
    if (data.shape(2) != 4) {
        throw py::value_error(
            "Data pointer must be RGBA; last dimension is {}, not 4"_s.format(
                data.shape(2)));
    }
    # 检查 data 的高度是否超出 Tk_PhotoPutBlock 参数类型的限制
    if (data.shape(0) > INT_MAX) {  
        throw std::range_error(
            "Height ({}) exceeds maximum allowable size ({})"_s.format(
                data.shape(0), INT_MAX));
    }
    # 检查 data 的宽度是否超出 Tk_PhotoImageBlock.pitch 字段的限制
    if (data.shape(1) > INT_MAX / 4) {  
        throw std::range_error(
            "Width ({}) exceeds maximum allowable size ({})"_s.format(
                data.shape(1), INT_MAX / 4));
    }
    # 将 data 的高度和宽度转换为整数类型
    const auto height = static_cast<int>(data.shape(0));
    const auto width = static_cast<int>(data.shape(1));
    int x1, x2, y1, y2;
    # 解包 bbox 中的坐标值
    std::tie(x1, x2, y1, y2) = bbox;
    # 检查坐标是否越界
    if (0 > y1 || y1 > y2 || y2 > height || 0 > x1 || x1 > x2 || x2 > width) {
        throw py::value_error("Attempting to draw out of bounds");
    }
    # 检查 comp_rule 是否为有效的合成规则
    if (comp_rule != TK_PHOTO_COMPOSITE_OVERLAY && comp_rule != TK_PHOTO_COMPOSITE_SET) {
        throw py::value_error("Invalid comp_rule argument");
    }

    int put_retval;
    Tk_PhotoImageBlock block;
    # 设置 block 的像素数据指针，宽度、高度、行距和像素大小
    block.pixelPtr = data_ptr.mutable_data(height - y2, x1, 0);
    block.width = x2 - x1;
    block.height = y2 - y1;
    block.pitch = 4 * width;
    block.pixelSize = 4;
    # 设置 block 的偏移量
    std::tie(block.offset[0], block.offset[1], block.offset[2], block.offset[3]) = offset;
    {
        # 释放 GIL，调用 Tk_PhotoPutBlock 函数进行图片块的放置
        py::gil_scoped_release release;
        put_retval = TK_PHOTO_PUT_BLOCK(
            interp, photo, &block, x1, height - y2, x2 - x1, y2 - y1, comp_rule);
    }
    # 检查 TK_PHOTO_PUT_BLOCK 函数调用是否失败
    if (put_retval == TCL_ERROR) {
        throw std::bad_alloc();
    }
// 如果定义了 WIN32_DLL 宏，则定义一个回调函数 DpiSubclassProc，处理窗口消息
LRESULT CALLBACK
DpiSubclassProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam,
                UINT_PTR uIdSubclass, DWORD_PTR dwRefData)
{
    // 根据消息类型进行不同的处理
    switch (uMsg) {
    case WM_DPICHANGED:
        // 当接收到 WM_DPICHANGED 消息时，执行以下操作：
        // 这是一个被子类化的窗口过程函数，在 Tcl/Tk 事件循环中执行。
        // 不幸的是，Tkinter 在 Tcl 线程中有第二个锁，虽然不公开，但在窗口过程中被占用。
        // 因此，虽然我们可以获取 GIL 来调用 Python 代码，但不能从 Python 中调用任何 Tk 代码。
        // 因此，在这里只能使用纯 C 调用 Tcl。
        {
            // 变量名 var_name 必须与 lib/matplotlib/backends/_backend_tk.py:FigureManagerTk 中的名字相匹配。
            std::string var_name("window_dpi");
            var_name += std::to_string((unsigned long long)hwnd);

            // wParam 中的 X 是高位字，Y 是低位字，但它们始终相等。
            std::string dpi = std::to_string(LOWORD(wParam));

            // 将 Tcl_Interp 指针转换为正确类型
            Tcl_Interp* interp = (Tcl_Interp*)dwRefData;
            // 设置 Tcl 变量，var_name 为变量名，dpi 为值，0 表示不使用 TCL_GLOBAL_ONLY
            TCL_SETVAR(interp, var_name.c_str(), dpi.c_str(), 0);
        }
        return 0;
    case WM_NCDESTROY:
        // 当接收到 WM_NCDESTROY 消息时，移除窗口子类化
        RemoveWindowSubclass(hwnd, DpiSubclassProc, uIdSubclass);
        break;
    }

    // 默认情况下，调用系统提供的窗口子类化过程
    return DefSubclassProc(hwnd, uMsg, wParam, lParam);
}
#endif
    # 如果 per_monitor 为真，则需要对每个监视器进行处理
    if (per_monitor) {
        // Per monitor aware means we need to handle WM_DPICHANGED by wrapping
        // the Window Procedure, and the Python side needs to trace the Tk
        // window_dpi variable stored on interp.
        // 设置窗口子类化，使用 DpiSubclassProc 函数处理 WM_DPICHANGED 事件，
        // 并且需要在 Python 侧跟踪存储在 interp 上的 Tk window_dpi 变量。
        SetWindowSubclass(frame_handle, DpiSubclassProc, 0, (DWORD_PTR)interp);
    }
    // 释放 user32.dll 库
    FreeLibrary(user32);
    // 返回 per_monitor 的 Python 包装对象
    return py::cast(per_monitor);
#endif
#endif

    return py::none();
}

// Functions to fill global Tcl/Tk function pointers by dynamic loading.

template <class T>
bool load_tcl_tk(T lib)
{
    // 尝试通过动态加载填充 Tcl/Tk 全局变量的函数指针。返回是否所有函数指针都已填充。
    if (auto ptr = dlsym(lib, "Tcl_SetVar")) {
        TCL_SETVAR = (Tcl_SetVar_t)ptr;  // 设置 TCL_SETVAR 变量为 Tcl_SetVar 函数的地址
    }
    if (auto ptr = dlsym(lib, "Tk_FindPhoto")) {
        TK_FIND_PHOTO = (Tk_FindPhoto_t)ptr;  // 设置 TK_FIND_PHOTO 变量为 Tk_FindPhoto 函数的地址
    }
    if (auto ptr = dlsym(lib, "Tk_PhotoPutBlock")) {
        TK_PHOTO_PUT_BLOCK = (Tk_PhotoPutBlock_t)ptr;  // 设置 TK_PHOTO_PUT_BLOCK 变量为 Tk_PhotoPutBlock 函数的地址
    }
    return TCL_SETVAR && TK_FIND_PHOTO && TK_PHOTO_PUT_BLOCK;  // 返回是否所有的函数指针都已填充
}

#ifdef WIN32_DLL

/* On Windows, we can't load the tkinter module to get the Tcl/Tk symbols,
 * because Windows does not load symbols into the library name-space of
 * importing modules. So, knowing that tkinter has already been imported by
 * Python, we scan all modules in the running process for the Tcl/Tk function
 * names.
 */

static void
load_tkinter_funcs()
{
    HANDLE process = GetCurrentProcess();  // 获取当前进程的伪句柄，不需要关闭
    DWORD size;
    if (!EnumProcessModules(process, NULL, 0, &size)) {  // 枚举当前进程的模块，获取模块数量
        PyErr_SetFromWindowsErr(0);  // 设置 Python 异常，表示 Windows 错误
        throw py::error_already_set();  // 抛出 Python 异常
    }
    auto count = size / sizeof(HMODULE);  // 计算模块数量
    auto modules = std::vector<HMODULE>(count);  // 创建存储模块句柄的向量
    if (!EnumProcessModules(process, modules.data(), size, &size)) {  // 再次枚举模块并获取模块句柄
        PyErr_SetFromWindowsErr(0);  // 设置 Python 异常，表示 Windows 错误
        throw py::error_already_set();  // 抛出 Python 异常
    }
    for (auto mod: modules) {
        if (load_tcl_tk(mod)) {  // 尝试从每个模块中加载 Tcl/Tk 函数指针
            return;  // 如果加载成功，则返回
        }
    }
}

#else  // not Windows

/*
 * On Unix, we can get the Tk symbols from the tkinter module, because tkinter
 * uses these symbols, and the symbols are therefore visible in the tkinter
 * dynamic library (module).
 */

static void
load_tkinter_funcs()
{
    // 从 tkinter 编译的模块中加载 tkinter 的全局函数。

    // 首先尝试从主程序命名空间加载
    auto main_program = dlopen(NULL, RTLD_LAZY);  // 打开主程序句柄，延迟加载
    auto success = load_tcl_tk(main_program);  // 尝试从主程序句柄加载 Tcl/Tk 函数
    // 主程序总是存在，所以不需要保持引用打开
    if (dlclose(main_program)) {  // 关闭主程序句柄
        throw std::runtime_error(dlerror());  // 如果关闭失败，抛出运行时错误
    }
    if (success) {
        return;  // 如果加载成功，则返回
    }

    py::object module;
    // 首先处理 PyPy，因为在 CPython 上会正确失败。
    try {
        module = py::module_::import("_tkinter.tklib_cffi");  // 尝试导入 PyPy 下的 _tkinter.tklib_cffi 模块
    } catch (py::error_already_set &e) {
        module = py::module_::import("_tkinter");  // 尝试导入 CPython 下的 _tkinter 模块
    }
    auto py_path = module.attr("__file__");  // 获取模块文件路径
    auto py_path_b = py::reinterpret_steal<py::bytes>(
        PyUnicode_EncodeFSDefault(py_path.ptr()));  // 将 Python 字符串路径编码为字节流
    std::string path = py_path_b;  // 转换为 C++ 标准字符串
    auto tkinter_lib = dlopen(path.c_str(), RTLD_LAZY);  // 打开 tkinter 动态库
    if (!tkinter_lib) {
        throw std::runtime_error(dlerror());  // 如果打开失败，抛出运行时错误
    }
    load_tcl_tk(tkinter_lib);  // 尝试从 tkinter 动态库加载 Tcl/Tk 函数
    // 不需要保持引用打开，因为 tkinter 已经被导入
}
    #`
    # 如果关闭 tkinter 库的动态链接库失败，抛出运行时错误
    if (dlclose(tkinter_lib)) {
        # 抛出包含 dlerror() 返回的错误信息的运行时错误
        throw std::runtime_error(dlerror());
    }
{
    try {
        // 尝试加载 tkinter 函数
        load_tkinter_funcs();
    } catch (py::error_already_set& e) {
        // 捕获异常并重新抛出为 ImportError，以便与后端自动回退交互
        py::raise_from(e, PyExc_ImportError, "failed to load tkinter functions");
        throw py::error_already_set();
    }

    // 如果 TCL_SETVAR 未定义，则抛出导入错误
    if (!TCL_SETVAR) {
        throw py::import_error("Failed to load Tcl_SetVar");
    } else if (!TK_FIND_PHOTO) {
        // 如果 TK_FIND_PHOTO 未定义，则抛出导入错误
        throw py::import_error("Failed to load Tk_FindPhoto");
    } else if (!TK_PHOTO_PUT_BLOCK) {
        // 如果 TK_PHOTO_PUT_BLOCK 未定义，则抛出导入错误
        throw py::import_error("Failed to load Tk_PhotoPutBlock");
    }

    // 定义 Python 函数 blit，用于在 tkinter 中绘制图像
    m.def("blit", &mpl_tk_blit,
          "interp"_a, "photo_name"_a, "data"_a, "comp_rule"_a, "offset"_a, "bbox"_a);
    
    // 定义 Python 函数 enable_dpi_awareness，用于启用 DPI 感知
    m.def("enable_dpi_awareness", &mpl_tk_enable_dpi_awareness,
          "frame_handle"_a, "interp"_a);

    // 设置 Python 模块属性 TK_PHOTO_COMPOSITE_OVERLAY
    m.attr("TK_PHOTO_COMPOSITE_OVERLAY") = TK_PHOTO_COMPOSITE_OVERLAY;
    
    // 设置 Python 模块属性 TK_PHOTO_COMPOSITE_SET
    m.attr("TK_PHOTO_COMPOSITE_SET") = TK_PHOTO_COMPOSITE_SET;
}
```