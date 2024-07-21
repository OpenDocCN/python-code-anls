# `.\pytorch\torch\csrc\jit\mobile\model_tracer\BuildFeatureTracer.cpp`

```py
#include <torch/csrc/jit/mobile/model_tracer/BuildFeatureTracer.h>
#include <mutex>

namespace torch {
namespace jit {
namespace mobile {

// 构造函数，初始化 BuildFeatureTracer 对象
BuildFeatureTracer::BuildFeatureTracer() {
  // 定义记录函数的回调函数，捕获传入的函数对象 fn
  auto recorder_cb =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    // 获取函数名称
    std::string name = fn.name();
    // 获取全局的 BuildFeatures 对象，并加锁，确保线程安全
    getBuildFeatures().withLock(
        [&](BuildFeatureTracer::build_feature_type& build_features) {
          // 将函数名称插入到 build_features 集合中
          build_features.insert(name);
        });
    // 返回空指针，表示不需要额外的观察上下文
    return nullptr;
  };

  // 向全局回调中注册记录函数的回调函数，限定作用域为 BUILD_FEATURE
  handle_ =
      at::addGlobalCallback(at::RecordFunctionCallback(recorder_cb)
                                .scopes({at::RecordScope::BUILD_FEATURE}));
}

// 返回 BuildFeatures 对象的引用，采用 c10::Synchronized 保证线程安全
c10::Synchronized<BuildFeatureTracer::build_feature_type>& BuildFeatureTracer::
    getBuildFeatures() {
  // 静态局部变量，确保在函数调用间保持其内容不变
  static c10::Synchronized<build_feature_type> build_features;
  return build_features;
}

} // namespace mobile
} // namespace jit
} // namespace torch
```