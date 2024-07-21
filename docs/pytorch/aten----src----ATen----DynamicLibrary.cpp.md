# `.\pytorch\aten\src\ATen\DynamicLibrary.cpp`

```py
// 引入 C10 库中的异常处理和 Unicode 支持
#include <c10/util/Exception.h>
#include <c10/util/Unicode.h>
// 引入 ATen 库中的动态库加载和工具函数
#include <ATen/DynamicLibrary.h>
#include <ATen/Utils.h>

// 根据操作系统选择性地引入不同的头文件
#ifndef _WIN32
#include <dlfcn.h>  // POSIX 动态库加载相关头文件
#include <libgen.h> // POSIX 文件路径操作相关头文件
#else
#include <c10/util/win32-headers.h> // Windows 平台特定头文件
#endif

// at 命名空间
namespace at {

// 非移动平台下，针对 Unix 系统的实现
#ifndef C10_MOBILE
#ifndef _WIN32

// Unix 平台下的动态库加载器实现

// 辅助函数，用于检查动态库操作返回的指针是否有效
static void* checkDL(void* x) {
  // 如果指针为空，抛出 DynamicLibraryError 异常并输出详细错误信息
  if (!x) {
    TORCH_CHECK_WITH(DynamicLibraryError, false, "Error in dlopen or dlsym: ", dlerror());
  }
  return x;
}

// DynamicLibrary 类的构造函数实现，负责打开指定的动态链接库
DynamicLibrary::DynamicLibrary(const char* name, const char* alt_name, bool leak_handle_): leak_handle(leak_handle_), handle(dlopen(name, RTLD_LOCAL | RTLD_NOW)) {
  // 如果动态库打开失败
  if (!handle) {
    // 如果提供了备用名称，则尝试再次打开动态链接库
    if (alt_name) {
      handle = dlopen(alt_name, RTLD_LOCAL | RTLD_NOW);
      // 如果再次失败，则抛出详细的 DynamicLibraryError 异常
      if (!handle) {
        TORCH_CHECK_WITH(DynamicLibraryError, false, "Error in dlopen for library ", name, "and ", alt_name);
      }
    } else {
      // 否则直接抛出详细的 DynamicLibraryError 异常
      TORCH_CHECK_WITH(DynamicLibraryError, false, "Error in dlopen: ", dlerror());
    }
  }
}

// DynamicLibrary 类的 sym 方法实现，用于获取动态库中指定符号的地址
void* DynamicLibrary::sym(const char* name) {
  // 断言动态库句柄有效
  AT_ASSERT(handle);
  // 调用辅助函数 checkDL 检查符号是否成功获取，并返回其地址
  return checkDL(dlsym(handle, name));
}

// DynamicLibrary 类的析构函数实现，负责关闭已打开的动态链接库
DynamicLibrary::~DynamicLibrary() {
  // 如果动态库句柄为空或允许泄漏，则直接返回
  if (!handle || leak_handle) {
    return;
  }
  // 否则关闭动态链接库
  dlclose(handle);
}

#else

// Windows 平台下的动态库加载器实现

// DynamicLibrary 类的构造函数实现，负责打开指定的动态链接库
DynamicLibrary::DynamicLibrary(const char* name, const char* alt_name, bool leak_handle_): leak_handle(leak_handle_) {
  // NOLINTNEXTLINE(hicpp-signed-bitwise)
  HMODULE theModule;
  bool reload = true;
  // 将 UTF-8 编码的动态库名称转换为 UTF-16 编码
  auto wname = c10::u8u16(name);
  // 检查是否支持 LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
  if (GetProcAddress(GetModuleHandleW(L"KERNEL32.DLL"), "AddDllDirectory") != NULL) {
    // 尝试通过指定的搜索路径加载动态链接库
    theModule = LoadLibraryExW(
        wname.c_str(),
        NULL,
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    // 如果加载成功或者指定的动态链接库未找到，则停止重试
    if (theModule != NULL || (GetLastError() != ERROR_MOD_NOT_FOUND)) {
      reload = false;
    }
  }

  // 如果需要重新加载，则使用标准方式加载动态链接库
  if (reload) {
    theModule = LoadLibraryW(wname.c_str());
  }

  // 如果成功加载了动态链接库，则将句柄赋给成员变量 handle
  if (theModule) {
    handle = theModule;
  } else {
    // 否则输出详细的错误信息，并抛出 DynamicLibraryError 异常
    char buf[256];
    DWORD dw = GetLastError();
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                  NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                  buf, (sizeof(buf) / sizeof(char)), NULL);
    TORCH_CHECK_WITH(DynamicLibraryError, false, "error in LoadLibrary for ", name, ". WinError ", dw, ": ", buf);
  }
}

// DynamicLibrary 类的 sym 方法实现，用于获取动态库中指定符号的地址
void* DynamicLibrary::sym(const char* name) {
  // 断言动态库句柄有效
  AT_ASSERT(handle);
  // 调用 Windows 平台的 GetProcAddress 函数获取指定符号的地址
  FARPROC procAddress = GetProcAddress((HMODULE)handle, name);
  // 如果获取失败，则抛出详细的 DynamicLibraryError 异常
  if (!procAddress) {
    TORCH_CHECK_WITH(DynamicLibraryError, false, "error in GetProcAddress");
  }
  // 返回获取到的符号地址
  return (void*)procAddress;
}

// DynamicLibrary 类的析构函数实现，负责关闭已打开的动态链接库
DynamicLibrary::~DynamicLibrary() {
  // 如果动态库句柄为空或允许泄漏，则直接返回
  if (!handle || leak_handle) {
    return;
  }
  // 否则释放已加载的动态链接库
  FreeLibrary((HMODULE)handle);
}

#endif
#endif

} // namespace at
```