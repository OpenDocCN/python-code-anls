# `.\pytorch\android\test_app\app\src\main\cpp\pytorch_testapp_jni.cpp`

```
#include <android/log.h>
#include <pthread.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <vector>

// 定义 ALOGI 宏，用于打印信息级别日志到 Android 日志系统
#define ALOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "PyTorchTestAppJni", __VA_ARGS__)
// 定义 ALOGE 宏，用于打印错误级别日志到 Android 日志系统
#define ALOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, "PyTorchTestAppJni", __VA_ARGS__)

#include "jni.h"

// 引入 PyTorch 的头文件
#include <torch/script.h>

namespace pytorch_testapp_jni {
namespace {

// 用于打印日志信息的模板函数
template <typename T>
void log(const char* m, T t) {
  std::ostringstream os;
  os << t << std::endl;
  // 调用 ALOGI 宏打印信息级别日志
  ALOGI("%s %s", m, os.str().c_str());
}

// 定义一个结构体，设置推断模式和禁用图优化器的守卫
struct JITCallGuard {
  c10::InferenceMode guard;  // 设置推断模式守卫
  torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};  // 禁用图优化器守卫
};

} // namespace

// 加载并前向推断模型的函数
static void loadAndForwardModel(JNIEnv* env, jclass, jstring jModelPath) {
  // 获取模型路径的 C 字符串
  const char* modelPath = env->GetStringUTFChars(jModelPath, 0);
  assert(modelPath);  // 断言确保模型路径非空

  // 创建 JITCallGuard 对象，设置推断模式守卫和禁用图优化器守卫
  JITCallGuard guard;

  // 加载 TorchScript 模型
  torch::jit::Module module = torch::jit::load(modelPath);
  module.eval();  // 将模型设置为评估模式

  // 创建一个随机张量作为输入
  torch::Tensor t = torch::randn({1, 3, 224, 224});
  // 调用 log 函数打印输入张量的信息
  log("input tensor:", t);

  // 执行模型的前向推断
  c10::IValue t_out = module.forward({t});
  // 调用 log 函数打印输出张量的信息
  log("output tensor:", t_out);

  // 释放模型路径的 C 字符串
  env->ReleaseStringUTFChars(jModelPath, modelPath);
}

} // namespace pytorch_testapp_jni

// JNI_OnLoad 函数，加载 JNI 库时调用
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env;
  // 获取 JNI 环境
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  // 查找 Java 类 org.pytorch.testapp.LibtorchNativeClient$NativePeer
  jclass c =
      env->FindClass("org/pytorch/testapp/LibtorchNativeClient$NativePeer");
  if (c == nullptr) {
    return JNI_ERR;
  }

  // 定义 JNI 方法数组，将 loadAndForwardModel 方法注册到 c 类中
  static const JNINativeMethod methods[] = {
      {"loadAndForwardModel",
       "(Ljava/lang/String;)V",
       (void*)pytorch_testapp_jni::loadAndForwardModel},
  };

  // 注册 JNI 方法到 c 类中
  int rc = env->RegisterNatives(
      c, methods, sizeof(methods) / sizeof(JNINativeMethod));

  // 检查注册结果
  if (rc != JNI_OK) {
    return rc;
  }

  return JNI_VERSION_1_6;  // 返回 JNI 版本号
}
```