# `.\pytorch\c10\core\CopyBytes.cpp`

```
// 包含 C10 库的头文件 CopyBytes.h 和 Logging.h
#include <c10/core/CopyBytes.h>
#include <c10/util/Logging.h>

// 命名空间 c10 中定义以下内容
namespace c10 {

// 数组 g_copy_bytes 的第一个维度表示是否异步（0 表示同步，1 表示异步），其它维度大小由 COMPILE_TIME_MAX_DEVICE_TYPES 决定
// NOLINTNEXTLINE 是一个 lint 指令，用于禁止特定的 lint 警告，此处忽略有关避免使用 C 风格数组和避免非常量全局变量的指导方针
static CopyBytesFunction g_copy_bytes[2][COMPILE_TIME_MAX_DEVICE_TYPES][COMPILE_TIME_MAX_DEVICE_TYPES];

// _CopyBytesFunctionRegisterer 类的构造函数定义
_CopyBytesFunctionRegisterer::_CopyBytesFunctionRegisterer(
    DeviceType fromType,
    DeviceType toType,
    CopyBytesFunction func_sync,
    CopyBytesFunction func_async) {
  auto from = static_cast<int>(fromType);
  auto to = static_cast<int>(toType);
  if (!func_async) {
    // 如果没有提供异步函数，则默认使用同步函数
    func_async = func_sync;
  }
  // 检查是否已经注册了指定设备类型对应的复制函数，防止重复注册
  CHECK(
      g_copy_bytes[0][from][to] == nullptr &&
      g_copy_bytes[1][from][to] == nullptr)
      << "Duplicate registration for device type pair "
      << c10::DeviceTypeName(fromType) << ", " << c10::DeviceTypeName(toType);
  // 将同步和异步复制函数注册到 g_copy_bytes 数组中
  g_copy_bytes[0][from][to] = func_sync;
  g_copy_bytes[1][from][to] = func_async;
}

// CopyBytes 函数的定义
void CopyBytes(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device,
    bool async) {
  // 根据 async 的值选择同步或异步的复制函数指针
  auto ptr = g_copy_bytes[async ? 1 : 0][static_cast<int>(src_device.type())]
                         [static_cast<int>(dst_device.type())];
  // 断言找到了合适的复制函数指针，如果没有找到，则报错
  CAFFE_ENFORCE(
      ptr,
      "No function found for copying from ",
      c10::DeviceTypeName(src_device.type()),
      " to ",
      c10::DeviceTypeName(dst_device.type()));
  // 调用选择的复制函数指针进行数据复制
  ptr(nbytes, src, src_device, dst, dst_device);
}

} // namespace c10


这些注释解释了 C++ 代码中各个函数和变量的作用和功能，确保了每行代码的功能和上下文都得到了说明。
```