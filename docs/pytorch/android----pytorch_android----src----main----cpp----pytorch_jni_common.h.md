# `.\pytorch\android\pytorch_android\src\main\cpp\pytorch_jni_common.h`

```py
#pragma once
// 使用预处理器指令#pragma once确保头文件只被包含一次

#include <c10/util/FunctionRef.h>
// 包含c10库中的FunctionRef头文件

#include <fbjni/fbjni.h>
// 包含fbjni库的头文件

#include <torch/csrc/api/include/torch/types.h>
// 包含PyTorch的类型定义头文件

#include "caffe2/serialize/read_adapter_interface.h"
// 包含caffe2库中的读适配器接口头文件

#include "cmake_macros.h"
// 包含cmake_macros.h中定义的宏

#ifdef __ANDROID__
#include <android/log.h>
// 在Android平台下，包含Android日志头文件

#define ALOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "pytorch-jni", __VA_ARGS__)
// 定义ALOGI宏，用于在Android平台上打印信息级别的日志

#define ALOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, "pytorch-jni", __VA_ARGS__)
// 定义ALOGE宏，用于在Android平台上打印错误级别的日志
#endif

#if defined(TRACE_ENABLED) && defined(__ANDROID__)
#include <android/trace.h>
#include <dlfcn.h>
// 如果TRACE_ENABLED和__ANDROID__被定义，则包含Android跟踪和动态链接库头文件
#endif

namespace pytorch_jni {

constexpr static int kDeviceCPU = 1;
// 定义常量kDeviceCPU为1，表示CPU设备类型

constexpr static int kDeviceVulkan = 2;
// 定义常量kDeviceVulkan为2，表示Vulkan设备类型

c10::DeviceType deviceJniCodeToDeviceType(jint deviceJniCode);
// 声明函数deviceJniCodeToDeviceType，用于将JNI代码映射为设备类型

class Trace {
 public:
#if defined(TRACE_ENABLED) && defined(__ANDROID__)
  typedef void* (*fp_ATrace_beginSection)(const char* sectionName);
  typedef void* (*fp_ATrace_endSection)(void);
  // 如果TRACE_ENABLED和__ANDROID__被定义，则声明函数指针类型fp_ATrace_beginSection和fp_ATrace_endSection
  static fp_ATrace_beginSection ATrace_beginSection;
  static fp_ATrace_endSection ATrace_endSection;
  // 声明静态成员变量ATrace_beginSection和ATrace_endSection，用于跟踪开始和结束操作
#endif

  static void ensureInit() {
    if (!Trace::is_initialized_) {
      init();
      Trace::is_initialized_ = true;
    }
  }
  // 静态函数ensureInit，确保跟踪功能初始化完成

  static void beginSection(const char* name) {
    Trace::ensureInit();
#if defined(TRACE_ENABLED) && defined(__ANDROID__)
    ATrace_beginSection(name);
#endif
  }
  // 开始一个跟踪段落，如果支持跟踪功能和在Android平台下

  static void endSection() {
#if defined(TRACE_ENABLED) && defined(__ANDROID__)
    ATrace_endSection();
#endif
  }
  // 结束当前的跟踪段落，如果支持跟踪功能和在Android平台下

  Trace(const char* name) {
    ensureInit();
    beginSection(name);
  }
  // 构造函数Trace，开始一个跟踪段落

  ~Trace() {
    endSection();
  }
  // 析构函数Trace，结束当前的跟踪段落

 private:
  static void init();
  // 声明静态私有函数init，用于初始化跟踪功能

  static bool is_initialized_;
  // 声明静态私有变量is_initialized_，标记跟踪功能是否已初始化
};

class MemoryReadAdapter final : public caffe2::serialize::ReadAdapterInterface {
 public:
  explicit MemoryReadAdapter(const void* data, off_t size)
      : data_(data), size_(size){};
  // 显式构造函数MemoryReadAdapter，用给定数据和大小初始化

  size_t size() const override {
    return size_;
  }
  // 实现ReadAdapterInterface接口的size函数，返回数据大小

  size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override {
    memcpy(buf, (int8_t*)(data_) + pos, n);
    return n;
  }
  // 实现ReadAdapterInterface接口的read函数，从指定位置读取数据到缓冲区

  ~MemoryReadAdapter() {}
  // 析构函数MemoryReadAdapter，无特殊操作

 private:
  const void* data_;
  off_t size_;
  // 私有成员变量data_和size_，分别存储数据指针和数据大小
};
# 定义 JIValue 类，继承自 facebook::jni::JavaClass<JIValue>
class JIValue : public facebook::jni::JavaClass<JIValue> {
  # 定义 DictCallback 类型，用于处理 c10::Dict<c10::IValue, c10::IValue> 的回调函数
  using DictCallback = c10::function_ref<facebook::jni::local_ref<JIValue>(
      c10::Dict<c10::IValue, c10::IValue>)>;

 public:
  # Java 类描述符，用于 JNI 交互
  constexpr static const char* kJavaDescriptor = "Lorg/pytorch/IValue;";

  # 各种数据类型的类型码常量
  constexpr static int kTypeCodeNull = 1;
  constexpr static int kTypeCodeTensor = 2;
  constexpr static int kTypeCodeBool = 3;
  constexpr static int kTypeCodeLong = 4;
  constexpr static int kTypeCodeDouble = 5;
  constexpr static int kTypeCodeString = 6;

  constexpr static int kTypeCodeTuple = 7;
  constexpr static int kTypeCodeBoolList = 8;
  constexpr static int kTypeCodeLongList = 9;
  constexpr static int kTypeCodeDoubleList = 10;
  constexpr static int kTypeCodeTensorList = 11;
  constexpr static int kTypeCodeList = 12;

  constexpr static int kTypeCodeDictStringKey = 13;
  constexpr static int kTypeCodeDictLongKey = 14;

  # 创建 JIValue 对象的静态方法，从 at::IValue 转换为 JIValue
  static facebook::jni::local_ref<JIValue> newJIValueFromAtIValue(
      const at::IValue& ivalue,
      DictCallback stringDictCallback = newJIValueFromStringDict,
      DictCallback intDictCallback = newJIValueFromIntDict);

  # 将 JIValue 转换为 at::IValue
  static at::IValue JIValueToAtIValue(
      facebook::jni::alias_ref<JIValue> jivalue);

 private:
  # 从 c10::Dict<c10::IValue, c10::IValue> 中创建 JIValue 的静态方法，键为字符串类型
  static facebook::jni::local_ref<JIValue> newJIValueFromStringDict(
      c10::Dict<c10::IValue, c10::IValue>);
  
  # 从 c10::Dict<c10::IValue, c10::IValue> 中创建 JIValue 的静态方法，键为整数类型
  static facebook::jni::local_ref<JIValue> newJIValueFromIntDict(
      c10::Dict<c10::IValue, c10::IValue>);
};

# 注册 JNI 方法的函数声明
void common_registerNatives();
} // namespace pytorch_jni
```