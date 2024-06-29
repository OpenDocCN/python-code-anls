# `.\numpy\numpy\_core\include\numpy\npy_os.h`

```py
#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_OS_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_OS_H_

#if defined(linux) || defined(__linux) || defined(__linux__)
    // 如果定义了 linux、__linux 或者 __linux__，则定义 NPY_OS_LINUX
    #define NPY_OS_LINUX
#elif defined(__FreeBSD__) || defined(__NetBSD__) || \
            defined(__OpenBSD__) || defined(__DragonFly__)
    // 如果定义了 __FreeBSD__、__NetBSD__、__OpenBSD__ 或者 __DragonFly__，则定义 NPY_OS_BSD
    #define NPY_OS_BSD
    // 根据具体平台的定义进一步细化为 NPY_OS_FREEBSD、NPY_OS_NETBSD、NPY_OS_OPENBSD 或 NPY_OS_DRAGONFLY
    #ifdef __FreeBSD__
        #define NPY_OS_FREEBSD
    #elif defined(__NetBSD__)
        #define NPY_OS_NETBSD
    #elif defined(__OpenBSD__)
        #define NPY_OS_OPENBSD
    #elif defined(__DragonFly__)
        #define NPY_OS_DRAGONFLY
    #endif
#elif defined(sun) || defined(__sun)
    // 如果定义了 sun 或 __sun，则定义 NPY_OS_SOLARIS
    #define NPY_OS_SOLARIS
#elif defined(__CYGWIN__)
    // 如果定义了 __CYGWIN__，则定义 NPY_OS_CYGWIN
    #define NPY_OS_CYGWIN
/* We are on Windows.*/
#elif defined(_WIN32)
  /* We are using MinGW (64-bit or 32-bit)*/
  // 如果定义了 _WIN32
  #if defined(__MINGW32__) || defined(__MINGW64__)
    // 如果同时定义了 __MINGW32__ 或者 __MINGW64__，则定义 NPY_OS_MINGW
    #define NPY_OS_MINGW
  /* Otherwise, if _WIN64 is defined, we are targeting 64-bit Windows*/
  // 否则，如果定义了 _WIN64，则目标是 64 位 Windows
  #elif defined(_WIN64)
    // 定义 NPY_OS_WIN64
    #define NPY_OS_WIN64
  /* Otherwise assume we are targeting 32-bit Windows*/
  // 否则假设目标是 32 位 Windows
  #else
    // 定义 NPY_OS_WIN32
    #define NPY_OS_WIN32
  #endif
#elif defined(__APPLE__)
    // 如果定义了 __APPLE__，则定义 NPY_OS_DARWIN
    #define NPY_OS_DARWIN
#elif defined(__HAIKU__)
    // 如果定义了 __HAIKU__，则定义 NPY_OS_HAIKU
    #define NPY_OS_HAIKU
#else
    // 如果未定义任何已知平台，则定义 NPY_OS_UNKNOWN
    #define NPY_OS_UNKNOWN
#endif

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_OS_H_ */
```