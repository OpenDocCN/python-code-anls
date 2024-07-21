# `.\pytorch\c10\cuda\driver_api.cpp`

```
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>

// 命名空间 c10::cuda 中的实现
namespace c10::cuda {

// 匿名命名空间，用于封装内部实现细节
namespace {

// 创建 DriverAPI 对象的函数
DriverAPI create_driver_api() {
  // 尝试以懒加载和不加载方式打开 libcuda.so.1 动态链接库
  void* handle_0 = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
  // 检查动态链接库是否成功打开，否则输出错误信息并终止程序
  TORCH_CHECK(handle_0, "Can't open libcuda.so.1: ", dlerror());

  // 获取 libnvidia-ml.so.1 动态链接库的句柄
  void* handle_1 = DriverAPI::get_nvml_handle();
  // 创建 DriverAPI 对象
  DriverAPI r{};

  // 宏定义，用于查找 libcuda.so.1 中指定 CUDA 驱动 API 函数的地址并赋值给 DriverAPI 对象的成员函数指针
#define LOOKUP_LIBCUDA_ENTRY(name)                       \
  r.name##_ = ((decltype(&name))dlsym(handle_0, #name)); \
  TORCH_INTERNAL_ASSERT(r.name##_, "Can't find ", #name, ": ", dlerror())

  // 调用宏 LOOKUP_LIBCUDA_ENTRY 完成 CUDA 驱动 API 函数的查找与赋值
  C10_LIBCUDA_DRIVER_API(LOOKUP_LIBCUDA_ENTRY)
#undef LOOKUP_LIBCUDA_ENTRY

  // 如果存在 handle_1，即 libnvidia-ml.so.1 被成功加载
  if (handle_1) {
    // 宏定义，用于查找 libnvidia-ml.so.1 中指定 NVML 驱动 API 函数的地址并赋值给 DriverAPI 对象的成员函数指针
#define LOOKUP_NVML_ENTRY(name)                          \
  r.name##_ = ((decltype(&name))dlsym(handle_1, #name)); \
  TORCH_INTERNAL_ASSERT(r.name##_, "Can't find ", #name, ": ", dlerror())

    // 调用宏 LOOKUP_NVML_ENTRY 完成 NVML 驱动 API 函数的查找与赋值
    C10_NVML_DRIVER_API(LOOKUP_NVML_ENTRY)
#undef LOOKUP_NVML_ENTRY
  }

  // 返回创建的 DriverAPI 对象
  return r;
}

} // namespace

// 获取 libnvidia-ml.so.1 动态链接库的句柄，静态函数
void* DriverAPI::get_nvml_handle() {
  static void* nvml_hanle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
  return nvml_hanle;
}

// 获取 DriverAPI 的静态函数，返回 DriverAPI 单例对象指针
C10_EXPORT DriverAPI* DriverAPI::get() {
  // 静态 DriverAPI 单例对象，初始化时调用 create_driver_api() 创建
  static DriverAPI singleton = create_driver_api();
  return &singleton;
}

} // namespace c10::cuda
#endif
```