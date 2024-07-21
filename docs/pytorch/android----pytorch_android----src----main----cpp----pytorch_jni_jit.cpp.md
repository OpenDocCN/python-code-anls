# `.\pytorch\android\pytorch_android\src\main\cpp\pytorch_jni_jit.cpp`

```py
namespace pytorch_jni {

namespace {

// 定义一个结构体 JITCallGuard，用于管理 JIT 调用的上下文
struct JITCallGuard {
  // 进入推理模式的上下文保护
  c10::InferenceMode guard;
  // 禁用图优化器，确保自定义移动构建中未使用的操作列表不会改变
  torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
};

} // namespace

// PytorchJni 类，继承自 HybridClass，用于与 Java 互操作
class PytorchJni : public facebook::jni::HybridClass<PytorchJni> {
 private:
  friend HybridBase;
  torch::jit::Module module_;  // PyTorch 模型的模块
  c10::DeviceType deviceType_; // 设备类型

 public:
  // Java 层描述符
  constexpr static auto kJavaDescriptor = "Lorg/pytorch/NativePeer;";

  // 初始化实例的静态方法，从文件路径创建模型
  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> modelPath,
      facebook::jni::alias_ref<
          facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>>
          extraFiles,
      jint device) {
    return makeCxxInstance(modelPath, extraFiles, device);
  }

#ifdef __ANDROID__
  // 初始化实例的静态方法，从 Android 资产管理器创建模型
  static facebook::jni::local_ref<jhybriddata> initHybridAndroidAsset(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> assetName,
      facebook::jni::alias_ref<jobject> assetManager,
      jint device) {
    return makeCxxInstance(assetName, assetManager, device);
  }
#endif

#ifdef TRACE_ENABLED
  // 进入函数跟踪的回调函数，开始跟踪指定函数
  static std::unique_ptr<at::ObserverContext> onFunctionEnter(
      const at::RecordFunction& fn) {
    Trace::beginSection(fn.name().str());
    return nullptr;
  }

  // 离开函数跟踪的回调函数，结束跟踪当前函数
  static void onFunctionExit(const at::RecordFunction&, at::ObserverContext*) {
    Trace::endSection();
  }
#endif

  // 预加载模块时的一次性设置，包括设置量化引擎和打印处理程序
  static void preModuleLoadSetupOnce() {
    auto qengines = at::globalContext().supportedQEngines();
    // 如果支持 QNNPACK，则设置为当前量化引擎
    if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) !=
        qengines.end()) {
      at::globalContext().setQEngine(at::QEngine::QNNPACK);
    }

#ifdef __ANDROID__
    // 设置 PyTorch 的打印处理程序，用于 Android 平台
    torch::jit::setPrintHandler([](const std::string& s) {
      __android_log_print(ANDROID_LOG_DEBUG, "pytorch-print", "%s", s.c_str());
    });
#endif

#ifdef TRACE_ENABLED
    // 添加全局回调，用于记录函数调用的开始和结束
    at::addGlobalCallback(
        at::RecordFunctionCallback(&onFunctionEnter, &onFunctionExit)
            .scopes({RecordScope::FUNCTION, RecordScope::USER_SCOPE}));
#endif
  }

  // 每次加载模块前的设置，包括一次性设置的调用
  void preModuleLoadSetup() {
    static const int once = []() {
      preModuleLoadSetupOnce();
      return 0;
    }();


这段代码定义了一个 C++ 类 `PytorchJni`，用于在 C++ 和 Java 之间进行交互，封装了 PyTorch 模型的加载和配置过程，包括设置推理模式、禁用图优化器、设置量化引擎以及设置打印处理程序等功能。
    // 忽略参数 `once`，不使用它
    ((void)once);
  }

  // 构造函数，接受模型路径、额外文件和设备类型作为参数
  PytorchJni(
      facebook::jni::alias_ref<jstring> modelPath,                           // 模型文件路径的 JNI 引用
      facebook::jni::alias_ref<
          facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>>
          extraFiles,                                                        // 额外文件的 JNI Map 引用
      jint device) {                                                         // 设备类型的整数值

    // 在加载模型之前进行预加载设置
    preModuleLoadSetup();

    // 创建 JIT 调用保护对象
    JITCallGuard guard;

    // 初始化额外文件的无序映射
    std::unordered_map<std::string, std::string> extra_files;

    // 检查是否存在额外文件，并且额外文件数量大于 0
    const auto has_extra = extraFiles && extraFiles->size() > 0;

    // 如果有额外文件，则将其转换为标准 C++ 字符串存储到 `extra_files` 中
    if (has_extra) {
      for (const auto& e : *extraFiles) {
        extra_files[e.first->toStdString()] = "";
      }
    }

    // 将设备类型从 JNI 代码转换为 Torch 设备类型
    deviceType_ = deviceJniCodeToDeviceType(device);

    // 加载模型文件到 `module_` 中，并传入额外文件的映射
    module_ = torch::jit::load(
        std::move(modelPath->toStdString()), c10::nullopt, extra_files);

    // 如果存在额外文件，则将其放入 Java Map `extraFiles` 中
    if (has_extra) {
      static auto putMethod =
          facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>::
              javaClassStatic()
                  ->template getMethod<facebook::jni::alias_ref<jobject>(
                      facebook::jni::alias_ref<jobject>,
                      facebook::jni::alias_ref<jobject>)>("put");
      for (const auto& ef : extra_files) {
        putMethod(
            extraFiles,
            facebook::jni::make_jstring(ef.first),
            facebook::jni::make_jstring(ef.second));
      }
    }

    // 将模型设置为评估模式
    module_.eval();
  }
#ifdef __ANDROID__
  // 定义 PytorchJni 构造函数，用于在 Android 平台上加载 PyTorch 模型
  PytorchJni(
      facebook::jni::alias_ref<jstring> assetName,
      facebook::jni::alias_ref<jobject> assetManager,
      jint device) {
    // 在加载模块之前执行必要的预处理设置
    preModuleLoadSetup();
    // 获取当前 JNI 环境
    JNIEnv* env = facebook::jni::Environment::current();
    // 从 Java 的 AssetManager 中获取 AAssetManager 对象
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager.get());
    // 如果获取失败，则抛出 Java 异常
    if (!mgr) {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "Unable to get asset manager");
    }
    // 根据文件名从 AAssetManager 中打开指定的资产文件
    AAsset* asset = AAssetManager_open(
        mgr, assetName->toStdString().c_str(), AASSET_MODE_BUFFER);
    // 如果打开失败，则抛出 Java 异常
    if (!asset) {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "Failed to open asset '%s'",
          assetName->toStdString().c_str());
    }
    // 获取资产文件的数据缓冲区
    auto assetBuffer = AAsset_getBuffer(asset);
    // 如果获取缓冲区失败，则抛出 Java 异常
    if (!assetBuffer) {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "Could not get buffer for asset '%s'",
          assetName->toStdString().c_str());
    }
    // 使用 JITCallGuard 保护加载模型的过程
    JITCallGuard guard;
    // 使用 MemoryReadAdapter 将资产文件的数据缓冲区加载为 PyTorch 模型
    module_ = torch::jit::load(std::make_unique<MemoryReadAdapter>(
        assetBuffer, AAsset_getLength(asset)));
    // 关闭 AAsset 资产文件
    AAsset_close(asset);
    // 将加载的模型设为评估模式
    module_.eval();
    // 将设备类型转换为对应的枚举值
    deviceType_ = deviceJniCodeToDeviceType(device);
  }
#endif

  // 静态函数，用于注册 JNI 的本地方法
  static void registerNatives() {
    registerHybrid({
        // 注册 initHybrid 方法
        makeNativeMethod("initHybrid", PytorchJni::initHybrid),
#ifdef __ANDROID__
        // 如果在 Android 平台，注册 initHybridAndroidAsset 方法
        makeNativeMethod(
            "initHybridAndroidAsset", PytorchJni::initHybridAndroidAsset),
#endif
        // 注册 forward 方法
        makeNativeMethod("forward", PytorchJni::forward),
        // 注册 runMethod 方法
        makeNativeMethod("runMethod", PytorchJni::runMethod),
    });
  }

  // JNI 方法，用于执行模型的前向推断
  facebook::jni::local_ref<JIValue> forward(
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JIValue::javaobject>::javaobject>
          jinputs) {
    // 跟踪 JNI 方法的执行
    Trace _s{"jni::Module::forward"};
    // 创建用于存储输入数据的 IValue 向量
    std::vector<at::IValue> inputs{};
    // 获取输入数组的大小
    size_t n = jinputs->size();
    // 预留空间以存储输入数据
    inputs.reserve(n);
    // 遍历 JNI 输入数组，并将其转换为对应的 IValue 存入 inputs 中
    for (const auto i : c10::irange(n)) {
      at::IValue atIValue = JIValue::JIValueToAtIValue(jinputs->getElement(i));
      inputs.push_back(std::move(atIValue));
    }
    // 执行前向推断，并使用 JITCallGuard 保护该过程
    auto output = [&]() {
      JITCallGuard guard;
      return module_.forward(std::move(inputs));
    }();
    // 将推断结果转换为 JNI 可用的 JIValue 类型并返回
    return JIValue::newJIValueFromAtIValue(output);
  }

  // JNI 方法，用于执行模型的指定方法
  facebook::jni::local_ref<JIValue> runMethod(
      facebook::jni::alias_ref<facebook::jni::JString::javaobject> jmethodName,
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JIValue::javaobject>::javaobject>
          jinputs) {
    // 将 JNI 字符串转换为标准 C++ 字符串
    std::string methodName = jmethodName->toStdString();

    // 创建用于存储输入数据的 IValue 向量
    std::vector<at::IValue> inputs{};
    // 获取输入数组的大小
    size_t n = jinputs->size();
    // 预留空间以存储输入数据
    inputs.reserve(n);
    // 遍历 JNI 输入数组，并将其转换为对应的 IValue 存入 inputs 中
    for (const auto i : c10::irange(n)) {
      at::IValue atIValue = JIValue::JIValueToAtIValue(jinputs->getElement(i));
      inputs.push_back(std::move(atIValue));
    }
    // 如果找到指定方法，则执行以下代码块
    if (auto method = module_.find_method(methodName)) {
      // 定义一个 lambda 表达式 output，用于调用找到的方法并获取输出结果
      auto output = [&]() {
        // 进入 JIT 调用保护区域
        JITCallGuard guard;
        // 调用找到的方法，并传入 inputs，获取输出结果
        return (*method)(std::move(inputs));
      }();
      // 根据输出结果创建一个新的 JIValue 对象，并返回
      return JIValue::newJIValueFromAtIValue(output);
    }

    // 如果未找到指定方法，抛出 Java 异常 IllegalArgumentException，并提供错误信息
    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Undefined method %s",
        methodName.c_str());
  }
};

} // namespace pytorch_jni

// JNI_OnLoad 是一个特殊的 JNI 函数，当 Java 虚拟机加载本地库时自动调用
// 它返回一个整数作为加载结果，通常用于初始化本地库或者注册本地方法
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  // 使用 facebook::jni::initialize 函数初始化 JNI 环境，并传入一个 Lambda 表达式
  // Lambda 表达式用于在初始化时注册本地方法
  return facebook::jni::initialize(vm, [] {
    // 调用 pytorch_jni 命名空间下的 common_registerNatives 函数，注册通用的本地方法
    pytorch_jni::common_registerNatives();
    // 调用 pytorch_jni 命名空间下的 PytorchJni 类的 registerNatives 函数，注册 PyTorch 相关的本地方法
    pytorch_jni::PytorchJni::registerNatives();
  });
}
```