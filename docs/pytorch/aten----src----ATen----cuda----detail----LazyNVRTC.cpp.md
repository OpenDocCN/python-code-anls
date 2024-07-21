# `.\pytorch\aten\src\ATen\cuda\detail\LazyNVRTC.cpp`

```
#include <ATen/cuda/detail/LazyNVRTC.h>

#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/DynamicLibrary.h>
#include <stdexcept>

namespace at {
namespace cuda {
namespace detail {
namespace _stubs {

// 获取 CUDA 库的动态链接库对象
at::DynamicLibrary& getCUDALibrary() {
#if defined(_WIN32)
  // 在 Windows 平台上静态创建名为 "nvcuda.dll" 的 DynamicLibrary 对象
  static at::DynamicLibrary lib("nvcuda.dll");
#else
  // 在非 Windows 平台上静态创建名为 "libcuda.so.1" 的 DynamicLibrary 对象
  static at::DynamicLibrary lib("libcuda.so.1");
#endif
  return lib;
}

// 获取 NVRTC 库的版本号字符串
static std::string getLibVersion() {
  // [NVRTC 版本信息]
  // 引用自 https://docs.nvidia.com/cuda/nvrtc/index.html 第 8.1 节 NVRTC 库版本信息
  //
  // 在以下中，MAJOR 和 MINOR 表示 CUDA Toolkit 的主要和次要版本号。
  // 例如对于 CUDA 11.2，MAJOR 是 "11"，MINOR 是 "2"。
  //
  // Linux:
  //   - 在 CUDA 11.3 及之前的 CUDA 工具包中，soname 设置为 "MAJOR.MINOR"。
  //   - 在 CUDA 11.x 工具包中，soname 字段设置为 "11.2"。
  //   - 对于主要版本号大于 11 的 CUDA 工具包（例如 CUDA 12.x），soname 字段设置为 "MAJOR"。
  //
  // Windows:
  //   - 在 CUDA 11.3 及之前的 CUDA 工具包中，DLL 名称的形式为 "nvrtc64_XY_0.dll"，其中 X = MAJOR，Y = MINOR。
  //   - 在 CUDA 11.x 工具包中，DLL 名称为 "nvrtc64_112_0.dll"。
  //   - 对于主要版本号大于 11 的 CUDA 工具包，DLL 名称的形式为 "nvrtc64_X0_0.dll"，其中 X = MAJOR。
  //
  // 考虑一个主要版本号大于 11 的 CUDA 工具包。此 CUDA 工具包中的 NVRTC 库将与同一 CUDA 工具包的
  // 先前次要版本的 NVRTC 库具有相同的 soname（Linux）或 DLL 名称（Windows）。类似地，CUDA 11.3 及以后的
  // 11.x 发布版本的 NVRTC 库将具有与 CUDA 11.2 中的 NVRTC 库相同的 soname（Linux）或 DLL 名称（Windows）。
  constexpr auto major = CUDA_VERSION / 1000;
  constexpr auto minor = ( CUDA_VERSION / 10 ) % 10;
#if defined(_WIN32)
  if (major < 11 || (major == 11 && minor < 3)) {
    return std::to_string(major) + std::to_string(minor);
  } else if (major == 11) {
    return "112";
  } else {
    return std::to_string(major) + "0";
  }
#else
  if (major < 11 || (major == 11 && minor < 3)) {
    return std::to_string(major) + "." + std::to_string(minor);
  } else if (major == 11) {
    return "11.2";
  } else {
    return std::to_string(major);
  }
#endif
}

// 获取 NVRTC 库的名称字符串
static std::string getLibName() {
#if defined(_WIN32)
  // 返回形如 "nvrtc64_XY_0.dll" 的 DLL 名称，其中 XY 是通过 getLibVersion() 获取的版本号
  return std::string("nvrtc64_") + getLibVersion() + "_0.dll";
#else
  // 返回形如 "libnvrtc.so.XY" 的动态库名称，其中 XY 是通过 getLibVersion() 获取的版本号
  return std::string("libnvrtc.so.") + getLibVersion();
#endif
}

// 获取备用的 NVRTC 库名称字符串
static std::string getAltLibName() {
#if !defined(_WIN32) && defined(NVRTC_SHORTHASH)
  // 在非 Windows 平台上，如果定义了 NVRTC_SHORTHASH，则返回形如 "libnvrtc-SHORTHASH.so.XY" 的备用库名称
  return std::string("libnvrtc-") + C10_STRINGIZE(NVRTC_SHORTHASH) + ".so." + getLibVersion();
#else
  // 否则返回空字符串
  return {};
#endif
}

// 获取 NVRTC 库的动态链接库对象
at::DynamicLibrary& getNVRTCLibrary() {
  // 静态创建 NVRTC 库的 DynamicLibrary 对象，使用静态获取的库名称和备用库名称
  static std::string libname = getLibName();
  static std::string alt_libname = getAltLibName();
  static at::DynamicLibrary lib(libname.c_str(), alt_libname.empty() ? nullptr : alt_libname.c_str());
  return lib;
}
# 定义一个宏函数，用于生成带有一个参数的 CUDA 库函数的存根
#define CUDA_STUB1(NAME, A1) _STUB_1(CUDA, NAME, CUresult CUDAAPI, A1)

# 定义一个宏函数，用于生成带有两个参数的 CUDA 库函数的存根
#define CUDA_STUB2(NAME, A1, A2) _STUB_2(CUDA, NAME, CUresult CUDAAPI, A1, A2)

# 定义一个宏函数，用于生成带有三个参数的 CUDA 库函数的存根
#define CUDA_STUB3(NAME, A1, A2, A3) _STUB_3(CUDA, NAME, CUresult CUDAAPI, A1, A2, A3)

# 定义一个宏函数，用于生成带有四个参数的 CUDA 库函数的存根
#define CUDA_STUB4(NAME, A1, A2, A3, A4) _STUB_4(CUDA, NAME, CUresult CUDAAPI, A1, A2, A3, A4)

# 定义一个宏函数，用于生成带有一个参数的 NVRTC 库函数的存根
#define NVRTC_STUB1(NAME, A1) _STUB_1(NVRTC, NAME, nvrtcResult, A1)

# 定义一个宏函数，用于生成带有两个参数的 NVRTC 库函数的存根
#define NVRTC_STUB2(NAME, A1, A2) _STUB_2(NVRTC, NAME, nvrtcResult, A1, A2)
// 定义宏 NVRTC_STUB3，用于生成名为 NVRTC_STUB3(NAME, A1, A2, A3) 的宏，该宏调用 _STUB_3 宏，用于处理 NVRTC 库中带有三个参数的函数
#define NVRTC_STUB3(NAME, A1, A2, A3) _STUB_3(NVRTC, NAME, nvrtcResult, A1, A2, A3)

// 调用 NVRTC_STUB2 宏生成 nvrtcVersion 函数的声明
NVRTC_STUB2(nvrtcVersion, int*, int*);
// 调用 NVRTC_STUB2 宏生成 nvrtcAddNameExpression 函数的声明
NVRTC_STUB2(nvrtcAddNameExpression, nvrtcProgram, const char * const);

// 定义 nvrtcCreateProgram 函数，用于创建 NVRTC 程序
nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog,
                               const char *src,
                               const char *name,
                               int numHeaders,
                               const char * const *headers,
                               const char * const *includeNames) {
  // 通过 getNVRTCLibrary().sym(__func__) 获取当前函数名对应的符号指针
  auto fn = reinterpret_cast<decltype(&nvrtcCreateProgram)>(getNVRTCLibrary().sym(__func__));
  // 如果获取失败，则抛出运行时异常
  if (!fn)
    throw std::runtime_error("Can't get nvrtcCreateProgram");
  // 将获取到的函数指针赋值给 lazyNVRTC.nvrtcCreateProgram
  lazyNVRTC.nvrtcCreateProgram = fn;
  // 调用实际的 nvrtcCreateProgram 函数，并返回其结果
  return fn(prog, src, name, numHeaders, headers, includeNames);
}

// 调用 NVRTC_STUB1 宏生成 nvrtcDestroyProgram 函数的声明
NVRTC_STUB1(nvrtcDestroyProgram, nvrtcProgram *);
// 调用 NVRTC_STUB2 宏生成 nvrtcGetPTXSize 函数的声明
NVRTC_STUB2(nvrtcGetPTXSize, nvrtcProgram, size_t *);
// 调用 NVRTC_STUB2 宏生成 nvrtcGetPTX 函数的声明
NVRTC_STUB2(nvrtcGetPTX, nvrtcProgram, char *);

// 如果 CUDA 版本大于等于 11010，则调用 NVRTC_STUB2 宏生成 nvrtcGetCUBINSize 和 nvrtcGetCUBIN 函数的声明
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11010
NVRTC_STUB2(nvrtcGetCUBINSize, nvrtcProgram, size_t *);
NVRTC_STUB2(nvrtcGetCUBIN, nvrtcProgram, char *);
#endif

// 调用 NVRTC_STUB3 宏生成 nvrtcCompileProgram 函数的声明
NVRTC_STUB3(nvrtcCompileProgram, nvrtcProgram, int, const char * const *);
// 调用 _STUB_1 宏生成 NVRTC 库中 nvrtcGetErrorString 函数的声明
_STUB_1(NVRTC, nvrtcGetErrorString, const char *, nvrtcResult);
// 调用 NVRTC_STUB2 宏生成 nvrtcGetProgramLogSize 函数的声明
NVRTC_STUB2(nvrtcGetProgramLogSize,nvrtcProgram, size_t*);
// 调用 NVRTC_STUB2 宏生成 nvrtcGetProgramLog 函数的声明
NVRTC_STUB2(nvrtcGetProgramLog, nvrtcProgram, char *);
// 调用 NVRTC_STUB3 宏生成 nvrtcGetLoweredName 函数的声明
NVRTC_STUB3(nvrtcGetLoweredName, nvrtcProgram, const char *, const char **);

// 调用 CUDA_STUB2 宏生成 cuModuleLoadData 函数的声明
CUDA_STUB2(cuModuleLoadData, CUmodule *, const void *);
// 调用 CUDA_STUB3 宏生成 cuModuleGetFunction 函数的声明
CUDA_STUB3(cuModuleGetFunction, CUfunction *, CUmodule, const char *);
// 调用 CUDA_STUB4 宏生成 cuOccupancyMaxActiveBlocksPerMultiprocessor 函数的声明
CUDA_STUB4(cuOccupancyMaxActiveBlocksPerMultiprocessor, int *, CUfunction, int, size_t);
// 调用 CUDA_STUB2 宏生成 cuGetErrorString 函数的声明
CUDA_STUB2(cuGetErrorString, CUresult, const char **);
// 调用 CUDA_STUB1 宏生成 cuCtxGetCurrent 函数的声明
CUDA_STUB1(cuCtxGetCurrent, CUcontext *);
// 调用 CUDA_STUB1 宏生成 cuCtxSetCurrent 函数的声明
CUDA_STUB1(cuCtxSetCurrent, CUcontext);
// 调用 CUDA_STUB1 宏生成 cuModuleUnload 函数的声明
CUDA_STUB1(cuModuleUnload, CUmodule);
// 调用 CUDA_STUB3 宏生成 cuDevicePrimaryCtxGetState 函数的声明
CUDA_STUB3(cuDevicePrimaryCtxGetState, CUdevice, unsigned int *, int *);
// 调用 CUDA_STUB2 宏生成 cuDevicePrimaryCtxRetain 函数的声明
CUDA_STUB2(cuDevicePrimaryCtxRetain, CUcontext *, CUdevice);
// 调用 CUDA_STUB4 宏生成 cuLinkCreate 函数的声明
CUDA_STUB4(cuLinkCreate, unsigned int, CUjit_option *, void **, CUlinkState *);
// 调用 CUDA_STUB3 宏生成 cuLinkComplete 函数的声明
CUDA_STUB3(cuLinkComplete, CUlinkState, void **, size_t *);
// 调用 CUDA_STUB3 宏生成 cuFuncSetAttribute 函数的声明
CUDA_STUB3(cuFuncSetAttribute, CUfunction, CUfunction_attribute, int);
// 调用 CUDA_STUB3 宏生成 cuFuncGetAttribute 函数的声明
CUDA_STUB3(cuFuncGetAttribute, int*, CUfunction_attribute, CUfunction);

// 如果 CUDA 版本大于等于 12000，则定义 cuTensorMapEncodeTiled 函数，用于编码张量映射
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
CUresult CUDAAPI
cuTensorMapEncodeTiled(
    CUtensorMap* tensorMap,
    CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank,
    void* globalAddress,
    const cuuint64_t* globalDim,
    const cuuint64_t* globalStrides,
    const cuuint32_t* boxDim,
    const cuuint32_t* elementStrides,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
  // 通过 getCUDALibrary().sym(__func__) 获取当前函数名对应的符号指针
  auto fn = reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(
      getCUDALibrary().sym(__func__));
  // 如果获取失败，则返回错误码
  if (!fn)
    // 如果获取失败，则抛出运行时异常
    throw std::runtime_error("Can't get cuTensorMapEncodeTiled");
  // 返回调用实际函数指针 fn 所指向的函数
  return fn(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim,
            globalStrides, boxDim, elementStrides, interleave, swizzle,
            l2Promotion, oobFill);
}
#endif
    // 抛出运行时异常，指示无法获取 cuTensorMapEncodeTiled
    throw std::runtime_error("Can't get cuTensorMapEncodeTiled");
  // 将 lazyNVRTC.cuTensorMapEncodeTiled 设置为 fn，表示成功获取 cuTensorMapEncodeTiled 函数
  lazyNVRTC.cuTensorMapEncodeTiled = fn;
  // 调用 cuTensorMapEncodeTiled 函数，传入多个参数进行张量映射编码处理，并返回结果
  return fn(
      tensorMap,
      tensorDataType,
      tensorRank,
      globalAddress,
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      interleave,
      swizzle,
      l2Promotion,
      oobFill);
// 结束 namespace _stubs 的定义

// 定义了一个函数 cuLaunchKernel，用于启动 CUDA 核函数
CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra) {
  // 获取 cuLaunchKernel 函数指针并赋给 fn
  auto fn = reinterpret_cast<decltype(&cuLaunchKernel)>(getCUDALibrary().sym(__func__));
  // 如果获取失败，抛出运行时错误
  if (!fn)
    throw std::runtime_error("Can't get cuLaunchKernel");
  // 将获取到的函数指针保存到 lazyNVRTC.cuLaunchKernel
  lazyNVRTC.cuLaunchKernel = fn;
  // 调用获取到的函数指针 fn 来执行 cuLaunchKernel 函数
  return fn(f,
            gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, hStream, kernelParams, extra);
}

// 定义了一个函数 cuLaunchCooperativeKernel，用于启动协作 CUDA 核函数
CUresult CUDAAPI cuLaunchCooperativeKernel(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams) {
  // 获取 cuLaunchCooperativeKernel 函数指针并赋给 fn
  auto fn = reinterpret_cast<decltype(&cuLaunchCooperativeKernel)>(
      getCUDALibrary().sym(__func__));
  // 如果获取失败，抛出运行时错误
  if (!fn)
    throw std::runtime_error("Can't get cuLaunchCooperativeKernel");
  // 将获取到的函数指针保存到 lazyNVRTC.cuLaunchCooperativeKernel
  lazyNVRTC.cuLaunchCooperativeKernel = fn;
  // 调用获取到的函数指针 fn 来执行 cuLaunchCooperativeKernel 函数
  return fn(
      f,
      gridDimX,
      gridDimY,
      gridDimZ,
      blockDimX,
      blockDimY,
      blockDimZ,
      sharedMemBytes,
      hStream,
      kernelParams);
}

// 定义了一个函数 cuModuleLoadDataEx，用于加载 CUDA 模块数据
CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module,
                                    const void *image,
                                    unsigned int numOptions,
                                    CUjit_option *options,
                                    void **optionValues) {
  // 获取 cuModuleLoadDataEx 函数指针并赋给 fn
  auto fn = reinterpret_cast<decltype(&cuModuleLoadDataEx)>(getCUDALibrary().sym(__func__));
  // 如果获取失败，抛出运行时错误
  if (!fn)
    throw std::runtime_error("Can't get cuModuleLoadDataEx");
  // 将获取到的函数指针保存到 lazyNVRTC.cuModuleLoadDataEx
  lazyNVRTC.cuModuleLoadDataEx = fn;
  // 调用获取到的函数指针 fn 来执行 cuModuleLoadDataEx 函数
  return fn(module, image, numOptions, options, optionValues);
}

// 定义了一个函数 cuLinkAddData，用于向 CUDA 链接状态中添加数据
CUresult CUDAAPI
cuLinkAddData(CUlinkState state,
              CUjitInputType type,
              void *data,
              size_t size,
              const char *name,
              unsigned int numOptions,
              CUjit_option *options,
              void **optionValues) {
  // 获取 cuLinkAddData 函数指针并赋给 fn
  auto fn = reinterpret_cast<decltype(&cuLinkAddData)>(getCUDALibrary().sym(__func__));
  // 如果获取失败，抛出运行时错误
  if (!fn)
    throw std::runtime_error("Can't get cuLinkAddData");
  // 将获取到的函数指针保存到 lazyNVRTC.cuLinkAddData
  lazyNVRTC.cuLinkAddData = fn;
  // 调用获取到的函数指针 fn 来执行 cuLinkAddData 函数
  return fn(state, type, data, size, name, numOptions, options, optionValues);
}

// namespace _stubs 的结束标记

// 定义了 lazyNVRTC 结构体，并初始化其中的成员为 _stubs 命名空间中定义的函数指针
NVRTC lazyNVRTC = {
#define _REFERENCE_MEMBER(name) _stubs::name,
  AT_FORALL_NVRTC(_REFERENCE_MEMBER)
#undef _REFERENCE_MEMBER
};
} // 结束 at 命名空间
} // 结束 cuda 命名空间
} // 结束 detail 命名空间
```