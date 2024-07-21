# `.\pytorch\android\pytorch_android\src\main\cpp\pytorch_jni_lite.cpp`

```py
// C++头文件包含
#include <cassert>
#include <iostream>
#include <memory>
#include <string>

// JNI和Facebook JNI库头文件包含
#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

// Torch相关头文件包含
#include <c10/util/irange.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/script.h>

// Caffe2相关头文件包含
#include "caffe2/serialize/read_adapter_interface.h"

// 自定义的PyTorch JNI通用头文件
#include "pytorch_jni_common.h"

// Android平台相关头文件包含
#ifdef __ANDROID__
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#endif

// 命名空间定义
namespace pytorch_jni {

// 匿名命名空间，用于定义局部结构和变量
namespace {

// 结构体：LiteJITCallGuard
struct LiteJITCallGuard {
  // 构造函数注释
  // 对于默认移动构建中未包含VariableType分派，我们需要全局设置此保护以避免分派错误（仅适用于动态分派）。
  // 随着Variable类和Tensor类的统一，不再需要为每个操作切换NonVariableTypeMode。
  // TODO: 理想情况下，此文件中的AutoNonVariableTypeMode应该更改为InferenceMode，但由于Oculus上的typeahead应用（D27943428）而被阻塞。
  // 要解除阻塞，我们需要找出哪个操作在InferenceMode外部对推断张量进行原位更新，并正确保护它。
  torch::AutoNonVariableTypeMode non_var_guard;
};

} // namespace

// 类定义：PytorchJni，继承自HybridClass
class PytorchJni : public facebook::jni::HybridClass<PytorchJni> {
 private:
  friend HybridBase;
  torch::jit::mobile::Module module_;  // Torch移动模块对象
  c10::DeviceType deviceType_;          // Torch设备类型

 public:
  constexpr static auto kJavaDescriptor = "Lorg/pytorch/LiteNativePeer;";

  // 静态方法：initHybrid，初始化Hybrid实例
  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> modelPath,
      facebook::jni::alias_ref<
          facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>>
          extraFiles,
      jint device) {
    return makeCxxInstance(modelPath, extraFiles, device);
  }

  // Android平台下的静态方法：initHybridAndroidAsset，初始化Hybrid实例（使用Asset作为输入）
#ifdef __ANDROID__
  static facebook::jni::local_ref<jhybriddata> initHybridAndroidAsset(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> assetName,
      facebook::jni::alias_ref<jobject> assetManager,
      jint device) {
    return makeCxxInstance(assetName, assetManager, device);
  }
#endif

  // 构造函数：PytorchJni
  PytorchJni(
      facebook::jni::alias_ref<jstring> modelPath,
      facebook::jni::alias_ref<
          facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>>
          extraFiles,
      jint device) {
    LiteJITCallGuard guard;  // 创建LiteJITCallGuard对象，用于保护
    std::unordered_map<std::string, std::string> extra_files;  // 初始化额外文件的无序映射
    const auto has_extra = extraFiles && extraFiles->size() > 0;  // 检查是否有额外的文件

    // 如果有额外文件，则将其添加到extra_files映射中
    if (has_extra) {
      for (const auto& e : *extraFiles) {
        extra_files[e.first->toStdString()] = "";
      }
    }

    deviceType_ = deviceJniCodeToDeviceType(device);  // 将JNI设备代码转换为Torch设备类型
    // 使用_load_for_mobile加载模型到module_中，可以传递额外文件映射
    module_ = torch::jit::_load_for_mobile(
        std::move(modelPath->toStdString()), c10::nullopt, extra_files);


这里只是代码的一部分，根据要求逐行添加了注释。
    // 调用 TorchScript 提供的函数，加载模型及其附加资源
    torch::jit::_load_extra_only_for_mobile(
        std::move(modelPath->toStdString()), c10::nullopt, extra_files);
    // 如果有额外的文件需要加载
    if (has_extra) {
      // 获取 Java 静态方法 putMethod，用于将键值对存入 Java 的 Map 对象
      static auto putMethod =
          facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>::
              javaClassStatic()
                  ->template getMethod<facebook::jni::alias_ref<jobject>(
                      facebook::jni::alias_ref<jobject>,
                      facebook::jni::alias_ref<jobject>)>("put");
      // 遍历额外文件列表，将文件路径和内容放入 Java Map 对象 extraFiles
      for (const auto& ef : extra_files) {
        putMethod(
            extraFiles,
            facebook::jni::make_jstring(ef.first),  // 将 C++ 字符串转换为 Java 字符串
            facebook::jni::make_jstring(ef.second));  // 将 C++ 字符串转换为 Java 字符串
      }
    }
#ifdef __ANDROID__
  // 如果是在 Android 平台编译，则定义 PytorchJni 构造函数
  PytorchJni(
      // 构造函数参数：assetName 是 Android 资产的名称，assetManager 是资产管理器对象的引用，device 是设备类型
      facebook::jni::alias_ref<jstring> assetName,
      facebook::jni::alias_ref<jobject> assetManager,
      jint device) {
    // 获取当前 JNI 环境
    JNIEnv* env = facebook::jni::Environment::current();
    // 从 Java 的 assetManager 中获取 AAssetManager 对象
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager.get());
    // 如果获取失败，则抛出 IllegalArgumentException 异常
    if (!mgr) {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "Unable to get asset manager");
    }
    // 打开指定名称的资产文件
    AAsset* asset = AAssetManager_open(
        mgr, assetName->toStdString().c_str(), AASSET_MODE_BUFFER);
    // 如果打开失败，则抛出 IllegalArgumentException 异常
    if (!asset) {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "Failed to open asset '%s'",
          assetName->toStdString().c_str());
    }
    // 获取资产文件的缓冲区指针
    auto assetBuffer = AAsset_getBuffer(asset);
    // 如果获取失败，则抛出 IllegalArgumentException 异常
    if (!assetBuffer) {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "Could not get buffer for asset '%s'",
          assetName->toStdString().c_str());
    }
    // 使用 LiteJITCallGuard 保护，加载并初始化 Torch 模型
    LiteJITCallGuard guard;
    module_ =
        torch::jit::_load_for_mobile(std::make_unique<MemoryReadAdapter>(
            assetBuffer, AAsset_getLength(asset)));
    // 关闭资产文件
    AAsset_close(asset);
    // 将设备类型转换为 Torch 可识别的设备类型
    deviceType_ = deviceJniCodeToDeviceType(device);
  }
#endif

  // 注册 JNI 方法
  static void registerNatives() {
    registerHybrid({
        // 注册 "initHybrid" 方法
        makeNativeMethod("initHybrid", PytorchJni::initHybrid),
#ifdef __ANDROID__
        // 如果是在 Android 平台，注册 "initHybridAndroidAsset" 方法
        makeNativeMethod(
            "initHybridAndroidAsset", PytorchJni::initHybridAndroidAsset),
#endif
        // 注册 "forward" 方法
        makeNativeMethod("forward", PytorchJni::forward),
        // 注册 "runMethod" 方法
        makeNativeMethod("runMethod", PytorchJni::runMethod),
    });
  }

  // 执行 Torch 模型的前向推理，并返回结果
  facebook::jni::local_ref<JIValue> forward(
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JIValue::javaobject>::javaobject>
          jinputs) {
    // 创建 IValue 输入向量
    std::vector<at::IValue> inputs{};
    size_t n = jinputs->size();
    inputs.reserve(n);
    // 将 JNI 输入数组转换为 IValue 对象
    for (const auto i : c10::irange(n)) {
      at::IValue atIValue = JIValue::JIValueToAtIValue(jinputs->getElement(i));
      inputs.push_back(std::move(atIValue));
    }

    // 执行前向推理并获取输出
    auto output = [&]() {
      LiteJITCallGuard guard;
      return module_.forward(inputs);
    }();
    // 将 Torch 的输出转换为 JNI 的 JIValue 对象并返回
    return JIValue::newJIValueFromAtIValue(output);
  }

  // 调用 Torch 模型的指定方法，并返回结果
  facebook::jni::local_ref<JIValue> runMethod(
      facebook::jni::alias_ref<facebook::jni::JString::javaobject> jmethodName,
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JIValue::javaobject>::javaobject>
          jinputs) {
    // 获取 JNI 字符串表示的方法名
    std::string methodName = jmethodName->toStdString();

    // 创建 IValue 输入向量
    std::vector<at::IValue> inputs{};
    size_t n = jinputs->size();
    inputs.reserve(n);
    // 将 JNI 输入数组转换为 IValue 对象
    for (const auto i : c10::irange(n)) {
      at::IValue atIValue = JIValue::JIValueToAtIValue(jinputs->getElement(i));
      inputs.push_back(std::move(atIValue));
    }
    // 如果找到指定方法名的方法对象
    if (auto method = module_.find_method(methodName)) {
      // 使用 Lambda 表达式调用方法，捕获返回值并返回其结果
      auto output = [&]() {
        // 创建 LiteJITCallGuard 对象，确保 JIT 调用期间的资源安全释放
        LiteJITCallGuard guard;
        // 调用找到的方法，并传入输入参数 inputs
        return module_.get_method(methodName)(inputs);
      }();
      // 将调用结果封装为 JIValue 对象并返回
      return JIValue::newJIValueFromAtIValue(output);
    }

    // 如果未找到指定方法名的方法对象，则抛出 Java 异常
    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Undefined method %s",
        methodName.c_str());
};

} // namespace pytorch_jni



// 结束 pytorch_jni 命名空间的定义

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  // JNI_OnLoad 是一个 JNI 的标准函数，当 JNI 库被加载时由 JVM 调用，初始化 JNI 环境
  return facebook::jni::initialize(vm, [] {
    // 使用 lambda 表达式初始化 JNI 环境
    pytorch_jni::common_registerNatives();
    // 调用 pytorch_jni 命名空间下的 common_registerNatives 函数注册 JNI 方法
    pytorch_jni::PytorchJni::registerNatives();
    // 调用 pytorch_jni 命名空间下的 PytorchJni 类的 registerNatives 静态函数注册 JNI 方法
  });
}



// 注册 JNI 方法，并返回初始化状态
```