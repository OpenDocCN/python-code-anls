# `.\pytorch\aten\src\ATen\nnapi\nnapi_wrapper.cpp`

```
// 声明一个整数变量 loaded 并初始化为 0
static int loaded = 0;
// 声明 nnapi_wrapper 结构体 nnapi_
static struct nnapi_wrapper nnapi_;
// 声明 nnapi_wrapper 结构体 check_nnapi_
static struct nnapi_wrapper check_nnapi_;

// 检查函数 _getDeviceCount，确保其在 nnapi_ 结构体中存在
static int check__getDeviceCount(uint32_t* numDevices) {
  // 断言 nnapi_ 结构体中存在 _getDeviceCount 函数指针
  CAFFE_ENFORCE(nnapi_._getDeviceCount);
  // 调用 nnapi_ 结构体中的 _getDeviceCount 函数，获取设备数量
  int ret = nnapi_._getDeviceCount(numDevices);
  // TODO: 可能在此处添加更好的日志记录。
  // 断言函数执行结果为 ANEURALNETWORKS_NO_ERROR，否则报告错误信息
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "_getDeviceCount", "failed with error ", ret
  );
  return ret;
}

// 检查函数 _getDevice，确保其在 nnapi_ 结构体中存在
static int check__getDevice(uint32_t devIndex, ANeuralNetworksDevice** device) {
  // 断言 nnapi_ 结构体中存在 _getDevice 函数指针
  CAFFE_ENFORCE(nnapi_._getDevice);
  // 调用 nnapi_ 结构体中的 _getDevice 函数，获取特定设备的句柄
  int ret = nnapi_._getDevice(devIndex, device);
  // TODO: 可能在此处添加更好的日志记录。
  // 断言函数执行结果为 ANEURALNETWORKS_NO_ERROR，否则报告错误信息
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "_getDevice", "failed with error ", ret
  );
  return ret;
}

// 检查函数 Device_getName，确保其在 nnapi_ 结构体中存在
static int check_Device_getName(const ANeuralNetworksDevice* device, const char** name) {
  // 断言 nnapi_ 结构体中存在 Device_getName 函数指针
  CAFFE_ENFORCE(nnapi_.Device_getName);
  // 调用 nnapi_ 结构体中的 Device_getName 函数，获取设备的名称
  int ret = nnapi_.Device_getName(device, name);
  // TODO: 可能在此处添加更好的日志记录。
  // 断言函数执行结果为 ANEURALNETWORKS_NO_ERROR，否则报告错误信息
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Device_getName", "failed with error ", ret
  );
  return ret;
}

// 检查函数 Device_getVersion，确保其在 nnapi_ 结构体中存在
static int check_Device_getVersion(const ANeuralNetworksDevice* device, const char** version) {
  // 断言 nnapi_ 结构体中存在 Device_getVersion 函数指针
  CAFFE_ENFORCE(nnapi_.Device_getVersion);
  // 调用 nnapi_ 结构体中的 Device_getVersion 函数，获取设备的版本信息
  int ret = nnapi_.Device_getVersion(device, version);
  // TODO: 可能在此处添加更好的日志记录。
  // 断言函数执行结果为 ANEURALNETWORKS_NO_ERROR，否则报告错误信息
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Device_getVersion", "failed with error ", ret
  );
  return ret;
}

// 检查函数 Device_getFeatureLevel，确保其在 nnapi_ 结构体中存在
static int check_Device_getFeatureLevel(const ANeuralNetworksDevice* device, int64_t* featureLevel) {
  // 断言 nnapi_ 结构体中存在 Device_getFeatureLevel 函数指针
  CAFFE_ENFORCE(nnapi_.Device_getFeatureLevel);
  // 调用 nnapi_ 结构体中的 Device_getFeatureLevel 函数，获取设备的特性级别
  int ret = nnapi_.Device_getFeatureLevel(device, featureLevel);
  // TODO: 可能在此处添加更好的日志记录。
  // 断言函数执行结果为 ANEURALNETWORKS_NO_ERROR，否则报告错误信息
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Device_getFeatureLevel", "failed with error ", ret
  );
  return ret;
}
// 检查指定模型在给定设备上支持的操作列表，并返回操作支持情况
static int check_Model_getSupportedOperationsForDevices( const ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices, uint32_t numDevices, bool* supportedOps) {
  // 确保 nnapi_.Model_getSupportedOperationsForDevices 函数存在
  CAFFE_ENFORCE(nnapi_.Model_getSupportedOperationsForDevices);
  // 调用 nnapi_.Model_getSupportedOperationsForDevices 函数来获取支持情况
  int ret = nnapi_.Model_getSupportedOperationsForDevices(model, devices, numDevices, supportedOps);
  // TODO: Maybe add better logging here. 可能需要在这里添加更好的日志记录
  // 确保操作执行成功，否则抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Model_getSupportedOperationsForDevices", "failed with error ", ret
  );
  return ret;
}

// 在给定设备上为指定模型创建编译对象
static int check_Compilation_createForDevices(ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices, uint32_t numDevices, ANeuralNetworksCompilation** compilation) {
  // 确保 nnapi_.Compilation_createForDevices 函数存在
  CAFFE_ENFORCE(nnapi_.Compilation_createForDevices);
  // 调用 nnapi_.Compilation_createForDevices 函数来创建编译对象
  int ret = nnapi_.Compilation_createForDevices(model, devices, numDevices, compilation);
  // TODO: Maybe add better logging here. 可能需要在这里添加更好的日志记录
  // 确保操作执行成功，否则抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Compilation_createForDevices", "failed with error ", ret
  );
  return ret;
}

// 执行指定的神经网络计算任务
static int check_Execution_compute(ANeuralNetworksExecution* execution) {
  // 确保 nnapi_.Execution_compute 函数存在
  CAFFE_ENFORCE(nnapi_.Execution_compute);
  // 调用 nnapi_.Execution_compute 函数来执行计算任务
  int ret = nnapi_.Execution_compute(execution);
  // TODO: Maybe add better logging here. 可能需要在这里添加更好的日志记录
  // 确保操作执行成功，否则抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Execution_compute", "failed with error ", ret
  );
  return ret;
}

// 根据文件描述符创建神经网络内存对象
static int check_Memory_createFromFd(size_t size, int protect, int fd, size_t offset, ANeuralNetworksMemory** memory) {
  // 确保 nnapi_.Memory_createFromFd 函数存在
  CAFFE_ENFORCE(nnapi_.Memory_createFromFd);
  // 调用 nnapi_.Memory_createFromFd 函数来创建内存对象
  int ret = nnapi_.Memory_createFromFd(size, protect, fd, offset, memory);
  // TODO: Maybe add better logging here. 可能需要在这里添加更好的日志记录
  // 确保操作执行成功，否则抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Memory_createFromFd", "failed with error ", ret
  );
  return ret;
}

// 释放指定的神经网络内存对象
static void check_Memory_free(ANeuralNetworksMemory* memory) {
  // 确保 nnapi_.Memory_free 函数存在
  CAFFE_ENFORCE(nnapi_.Memory_free);
  // 调用 nnapi_.Memory_free 函数来释放内存对象
  nnapi_.Memory_free(memory);
}

// 创建空的神经网络模型对象
static int check_Model_create(ANeuralNetworksModel** model) {
  // 确保 nnapi_.Model_create 函数存在
  CAFFE_ENFORCE(nnapi_.Model_create);
  // 调用 nnapi_.Model_create 函数来创建模型对象
  int ret = nnapi_.Model_create(model);
  // TODO: Maybe add better logging here. 可能需要在这里添加更好的日志记录
  // 确保操作执行成功，否则抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Model_create", "failed with error ", ret
  );
  return ret;
}

// 释放指定的神经网络模型对象
static void check_Model_free(ANeuralNetworksModel* model) {
  // 确保 nnapi_.Model_free 函数存在
  CAFFE_ENFORCE(nnapi_.Model_free);
  // 调用 nnapi_.Model_free 函数来释放模型对象
  nnapi_.Model_free(model);
}

// 完成指定的神经网络模型构建过程
static int check_Model_finish(ANeuralNetworksModel* model) {
  // 确保 nnapi_.Model_finish 函数存在
  CAFFE_ENFORCE(nnapi_.Model_finish);
  // 调用 nnapi_.Model_finish 函数来完成模型构建
  int ret = nnapi_.Model_finish(model);
  // TODO: Maybe add better logging here. 可能需要在这里添加更好的日志记录
  // 确保操作执行成功，否则抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Model_finish", "failed with error ", ret
  );
  return ret;
}

// 向指定神经网络模型添加操作数
static int check_Model_addOperand(ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type) {
  // 确保 nnapi_.Model_addOperand 函数存在
  CAFFE_ENFORCE(nnapi_.Model_addOperand);
  // 调用 nnapi_.Model_addOperand 函数来向模型添加操作数
  int ret = nnapi_.Model_addOperand(model, type);
  // TODO: Maybe add better logging here. 可能需要在这里添加更好的日志记录
  // 确保操作执行成功，否则抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Model_addOperand", "failed with error ", ret
  );
  return ret;
}
static int check_Model_setOperandValue(ANeuralNetworksModel* model, int32_t index, const void* buffer, size_t length) {
  // 检查是否存在 Model_setOperandValue 函数指针
  CAFFE_ENFORCE(nnapi_.Model_setOperandValue);
  // 调用 Model_setOperandValue 函数，并获取返回值
  int ret = nnapi_.Model_setOperandValue(model, index, buffer, length);
  // TODO: 可能在这里添加更好的日志记录。
  // 检查返回值是否为 NO_ERROR，如果不是，抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Model_setOperandValue", "failed with error ", ret
  );
  // 返回操作的结果值
  return ret;
}

static int check_Model_setOperandValueFromMemory(ANeuralNetworksModel* model, int32_t index, const ANeuralNetworksMemory* memory, size_t offset, size_t length) {
  // 检查是否存在 Model_setOperandValueFromMemory 函数指针
  CAFFE_ENFORCE(nnapi_.Model_setOperandValueFromMemory);
  // 调用 Model_setOperandValueFromMemory 函数，并获取返回值
  int ret = nnapi_.Model_setOperandValueFromMemory(model, index, memory, offset, length);
  // TODO: 可能在这里添加更好的日志记录。
  // 检查返回值是否为 NO_ERROR，如果不是，抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Model_setOperandValueFromMemory", "failed with error ", ret
  );
  // 返回操作的结果值
  return ret;
}

static int check_Model_addOperation(ANeuralNetworksModel* model, ANeuralNetworksOperationType type, uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs) {
  // 检查是否存在 Model_addOperation 函数指针
  CAFFE_ENFORCE(nnapi_.Model_addOperation);
  // 调用 Model_addOperation 函数，并获取返回值
  int ret = nnapi_.Model_addOperation(model, type, inputCount, inputs, outputCount, outputs);
  // TODO: 可能在这里添加更好的日志记录。
  // 检查返回值是否为 NO_ERROR，如果不是，抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Model_addOperation", "failed with error ", ret
  );
  // 返回操作的结果值
  return ret;
}

static int check_Model_identifyInputsAndOutputs(ANeuralNetworksModel* model, uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs) {
  // 检查是否存在 Model_identifyInputsAndOutputs 函数指针
  CAFFE_ENFORCE(nnapi_.Model_identifyInputsAndOutputs);
  // 调用 Model_identifyInputsAndOutputs 函数，并获取返回值
  int ret = nnapi_.Model_identifyInputsAndOutputs(model, inputCount, inputs, outputCount, outputs);
  // TODO: 可能在这里添加更好的日志记录。
  // 检查返回值是否为 NO_ERROR，如果不是，抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Model_identifyInputsAndOutputs", "failed with error ", ret
  );
  // 返回操作的结果值
  return ret;
}

static int check_Model_relaxComputationFloat32toFloat16(ANeuralNetworksModel* model, bool allow) {
  // 检查是否存在 Model_relaxComputationFloat32toFloat16 函数指针
  CAFFE_ENFORCE(nnapi_.Model_relaxComputationFloat32toFloat16);
  // 调用 Model_relaxComputationFloat32toFloat16 函数，并获取返回值
  int ret = nnapi_.Model_relaxComputationFloat32toFloat16(model, allow);
  // TODO: 可能在这里添加更好的日志记录。
  // 检查返回值是否为 NO_ERROR，如果不是，抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Model_relaxComputationFloat32toFloat16", "failed with error ", ret
  );
  // 返回操作的结果值
  return ret;
}

static int check_Compilation_create(ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation) {
  // 检查是否存在 Compilation_create 函数指针
  CAFFE_ENFORCE(nnapi_.Compilation_create);
  // 调用 Compilation_create 函数，并获取返回值
  int ret = nnapi_.Compilation_create(model, compilation);
  // TODO: 可能在这里添加更好的日志记录。
  // 检查返回值是否为 NO_ERROR，如果不是，抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Compilation_create", "failed with error ", ret
  );
  // 返回操作的结果值
  return ret;
}

static void check_Compilation_free(ANeuralNetworksCompilation* compilation) {
  // 检查是否存在 Compilation_free 函数指针
  CAFFE_ENFORCE(nnapi_.Compilation_free);
  // 调用 Compilation_free 函数释放 Compilation 对象
  nnapi_.Compilation_free(compilation);
}
static int check_Compilation_setPreference(ANeuralNetworksCompilation* compilation, int32_t preference) {
    // 检查是否 nnapi_.Compilation_setPreference 函数已经定义
    CAFFE_ENFORCE(nnapi_.Compilation_setPreference);
    // 调用 nnapi_.Compilation_setPreference 函数，设置编译优先级
    int ret = nnapi_.Compilation_setPreference(compilation, preference);
    // TODO: 可以在此处添加更好的日志记录功能。
    // 检查返回值，如果不是 ANEURALNETWORKS_NO_ERROR 则抛出异常
    CAFFE_ENFORCE(
        ret == ANEURALNETWORKS_NO_ERROR,
        "Compilation_setPreference", "failed with error ", ret
    );
    return ret;
}

static int check_Compilation_finish(ANeuralNetworksCompilation* compilation) {
    // 检查是否 nnapi_.Compilation_finish 函数已经定义
    CAFFE_ENFORCE(nnapi_.Compilation_finish);
    // 调用 nnapi_.Compilation_finish 函数，完成编译
    int ret = nnapi_.Compilation_finish(compilation);
    // TODO: 可以在此处添加更好的日志记录功能。
    // 检查返回值，如果不是 ANEURALNETWORKS_NO_ERROR 则抛出异常
    CAFFE_ENFORCE(
        ret == ANEURALNETWORKS_NO_ERROR,
        "Compilation_finish", "failed with error ", ret
    );
    return ret;
}

static int check_Execution_create(ANeuralNetworksCompilation* compilation, ANeuralNetworksExecution** execution) {
    // 检查是否 nnapi_.Execution_create 函数已经定义
    CAFFE_ENFORCE(nnapi_.Execution_create);
    // 调用 nnapi_.Execution_create 函数，创建执行对象
    int ret = nnapi_.Execution_create(compilation, execution);
    // TODO: 可以在此处添加更好的日志记录功能。
    // 检查返回值，如果不是 ANEURALNETWORKS_NO_ERROR 则抛出异常
    CAFFE_ENFORCE(
        ret == ANEURALNETWORKS_NO_ERROR,
        "Execution_create", "failed with error ", ret
    );
    return ret;
}

static void check_Execution_free(ANeuralNetworksExecution* execution) {
    // 检查是否 nnapi_.Execution_free 函数已经定义
    CAFFE_ENFORCE(nnapi_.Execution_free);
    // 调用 nnapi_.Execution_free 函数，释放执行对象
    nnapi_.Execution_free(execution);
}

static int check_Execution_setInput(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, const void* buffer, size_t length) {
    // 检查是否 nnapi_.Execution_setInput 函数已经定义
    CAFFE_ENFORCE(nnapi_.Execution_setInput);
    // 调用 nnapi_.Execution_setInput 函数，设置输入数据
    int ret = nnapi_.Execution_setInput(execution, index, type, buffer, length);
    // TODO: 可以在此处添加更好的日志记录功能。
    // 检查返回值，如果不是 ANEURALNETWORKS_NO_ERROR 则抛出异常
    CAFFE_ENFORCE(
        ret == ANEURALNETWORKS_NO_ERROR,
        "Execution_setInput", "failed with error ", ret
    );
    return ret;
}

static int check_Execution_setInputFromMemory(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory, size_t offset, size_t length) {
    // 检查是否 nnapi_.Execution_setInputFromMemory 函数已经定义
    CAFFE_ENFORCE(nnapi_.Execution_setInputFromMemory);
    // 调用 nnapi_.Execution_setInputFromMemory 函数，从内存设置输入数据
    int ret = nnapi_.Execution_setInputFromMemory(execution, index, type, memory, offset, length);
    // TODO: 可以在此处添加更好的日志记录功能。
    // 检查返回值，如果不是 ANEURALNETWORKS_NO_ERROR 则抛出异常
    CAFFE_ENFORCE(
        ret == ANEURALNETWORKS_NO_ERROR,
        "Execution_setInputFromMemory", "failed with error ", ret
    );
    return ret;
}

static int check_Execution_setOutput(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, void* buffer, size_t length) {
    // 检查是否 nnapi_.Execution_setOutput 函数已经定义
    CAFFE_ENFORCE(nnapi_.Execution_setOutput);
    // 调用 nnapi_.Execution_setOutput 函数，设置输出数据
    int ret = nnapi_.Execution_setOutput(execution, index, type, buffer, length);
    // TODO: 可以在此处添加更好的日志记录功能。
    // 检查返回值，如果不是 ANEURALNETWORKS_NO_ERROR 则抛出异常
    CAFFE_ENFORCE(
        ret == ANEURALNETWORKS_NO_ERROR,
        "Execution_setOutput", "failed with error ", ret
    );
    return ret;
}
static int check_Execution_setOutputFromMemory(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory, size_t offset, size_t length) {
  // 确保 nnapi_.Execution_setOutputFromMemory 函数存在
  CAFFE_ENFORCE(nnapi_.Execution_setOutputFromMemory);
  // 调用 nnapi_.Execution_setOutputFromMemory 函数，并返回结果
  int ret = nnapi_.Execution_setOutputFromMemory(execution, index, type, memory, offset, length);
  // TODO: 可能在此处添加更好的日志记录
  // 确保函数执行成功，否则抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Execution_setOutputFromMemory", "failed with error ", ret
  );
  return ret;
}

static int check_Execution_startCompute(ANeuralNetworksExecution* execution, ANeuralNetworksEvent** event) {
  // 确保 nnapi_.Execution_startCompute 函数存在
  CAFFE_ENFORCE(nnapi_.Execution_startCompute);
  // 调用 nnapi_.Execution_startCompute 函数，并返回结果
  int ret = nnapi_.Execution_startCompute(execution, event);
  // TODO: 可能在此处添加更好的日志记录
  // 确保函数执行成功，否则抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Execution_startCompute", "failed with error ", ret
  );
  return ret;
}

static int check_Event_wait(ANeuralNetworksEvent* event) {
  // 确保 nnapi_.Event_wait 函数存在
  CAFFE_ENFORCE(nnapi_.Event_wait);
  // 调用 nnapi_.Event_wait 函数，并返回结果
  int ret = nnapi_.Event_wait(event);
  // TODO: 可能在此处添加更好的日志记录
  // 确保函数执行成功，否则抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Event_wait", "failed with error ", ret
  );
  return ret;
}

static void check_Event_free(ANeuralNetworksEvent* event) {
  // 确保 nnapi_.Event_free 函数存在
  CAFFE_ENFORCE(nnapi_.Event_free);
  // 调用 nnapi_.Event_free 函数释放事件对象
  nnapi_.Event_free(event);
}

static int check_Execution_getOutputOperandRank(ANeuralNetworksExecution* execution, int32_t index, uint32_t* rank) {
  // 确保 nnapi_.Execution_getOutputOperandRank 函数存在
  CAFFE_ENFORCE(nnapi_.Execution_getOutputOperandRank);
  // 调用 nnapi_.Execution_getOutputOperandRank 函数，并返回结果
  int ret = nnapi_.Execution_getOutputOperandRank(execution, index, rank);
  // TODO: 可能在此处添加更好的日志记录
  // 确保函数执行成功，否则抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Execution_getOutputOperandRank", "failed with error ", ret
  );
  return ret;
}

static int check_Execution_getOutputOperandDimensions(ANeuralNetworksExecution* execution, int32_t index, uint32_t* dimensions) {
  // 确保 nnapi_.Execution_getOutputOperandDimensions 函数存在
  CAFFE_ENFORCE(nnapi_.Execution_getOutputOperandDimensions);
  // 调用 nnapi_.Execution_getOutputOperandDimensions 函数，并返回结果
  int ret = nnapi_.Execution_getOutputOperandDimensions(execution, index, dimensions);
  // TODO: 可能在此处添加更好的日志记录
  // 确保函数执行成功，否则抛出异常
  CAFFE_ENFORCE(
    ret == ANEURALNETWORKS_NO_ERROR,
    "Execution_getOutputOperandDimensions", "failed with error ", ret
  );
  return ret;
}

void nnapi_wrapper_load(struct nnapi_wrapper** nnapi, struct nnapi_wrapper** check_nnapi) {
#ifdef _WIN32
  // 在 Windows 平台上不支持运行 NNAPI 模型，抛出错误
  TORCH_CHECK(false, "Running NNAPI models is not supported on Windows.");
#else
  if (!loaded) {
    // 清除错误标志
    dlerror();
    // 动态加载 libneuralnetworks.so 库
    void* handle = dlopen("libneuralnetworks.so", RTLD_LAZY | RTLD_LOCAL);
    // 确保库加载成功，否则抛出异常
    CAFFE_ENFORCE(handle, "Failed to load libneuralnetworks.so ", dlerror());
    // 使用 dlsym 获取并设置 nnapi_ 结构体中相关函数的指针
    *(void**)&nnapi_._getDeviceCount = dlsym(handle, "ANeuralNetworks_getDeviceCount");
    check_nnapi_._getDeviceCount = check__getDeviceCount;
    *(void**)&nnapi_._getDevice = dlsym(handle, "ANeuralNetworks_getDevice");
    check_nnapi_._getDevice = check__getDevice;
    *(void**)&nnapi_.Device_getName = dlsym(handle, "ANeuralNetworksDevice_getName");
    # 设置指向 `ANeuralNetworksDevice_getName` 函数的指针，并将其保存到 `check_nnapi_.Device_getName` 中
    check_nnapi_.Device_getName = check_Device_getName;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksDevice_getVersion` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Device_getVersion` 中
    *(void**)&nnapi_.Device_getVersion = dlsym(handle, "ANeuralNetworksDevice_getVersion");
    
    # 设置指向 `ANeuralNetworksDevice_getVersion` 函数的指针，并将其保存到 `check_nnapi_.Device_getVersion` 中
    check_nnapi_.Device_getVersion = check_Device_getVersion;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksDevice_getFeatureLevel` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Device_getFeatureLevel` 中
    *(void**)&nnapi_.Device_getFeatureLevel = dlsym(handle, "ANeuralNetworksDevice_getFeatureLevel");
    
    # 设置指向 `ANeuralNetworksDevice_getFeatureLevel` 函数的指针，并将其保存到 `check_nnapi_.Device_getFeatureLevel` 中
    check_nnapi_.Device_getFeatureLevel = check_Device_getFeatureLevel;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksModel_getSupportedOperationsForDevices` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Model_getSupportedOperationsForDevices` 中
    *(void**)&nnapi_.Model_getSupportedOperationsForDevices = dlsym(handle, "ANeuralNetworksModel_getSupportedOperationsForDevices");
    
    # 设置指向 `ANeuralNetworksModel_getSupportedOperationsForDevices` 函数的指针，并将其保存到 `check_nnapi_.Model_getSupportedOperationsForDevices` 中
    check_nnapi_.Model_getSupportedOperationsForDevices = check_Model_getSupportedOperationsForDevices;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksCompilation_createForDevices` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Compilation_createForDevices` 中
    *(void**)&nnapi_.Compilation_createForDevices = dlsym(handle, "ANeuralNetworksCompilation_createForDevices");
    
    # 设置指向 `ANeuralNetworksCompilation_createForDevices` 函数的指针，并将其保存到 `check_nnapi_.Compilation_createForDevices` 中
    check_nnapi_.Compilation_createForDevices = check_Compilation_createForDevices;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksExecution_compute` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Execution_compute` 中
    *(void**)&nnapi_.Execution_compute = dlsym(handle, "ANeuralNetworksExecution_compute");
    
    # 设置指向 `ANeuralNetworksExecution_compute` 函数的指针，并将其保存到 `check_nnapi_.Execution_compute` 中
    check_nnapi_.Execution_compute = check_Execution_compute;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksMemory_createFromFd` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Memory_createFromFd` 中
    *(void**)&nnapi_.Memory_createFromFd = dlsym(handle, "ANeuralNetworksMemory_createFromFd");
    
    # 设置指向 `ANeuralNetworksMemory_createFromFd` 函数的指针，并将其保存到 `check_nnapi_.Memory_createFromFd` 中
    check_nnapi_.Memory_createFromFd = check_Memory_createFromFd;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksMemory_free` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Memory_free` 中
    *(void**)&nnapi_.Memory_free = dlsym(handle, "ANeuralNetworksMemory_free");
    
    # 设置指向 `ANeuralNetworksMemory_free` 函数的指针，并将其保存到 `check_nnapi_.Memory_free` 中
    check_nnapi_.Memory_free = check_Memory_free;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksModel_create` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Model_create` 中
    *(void**)&nnapi_.Model_create = dlsym(handle, "ANeuralNetworksModel_create");
    
    # 设置指向 `ANeuralNetworksModel_create` 函数的指针，并将其保存到 `check_nnapi_.Model_create` 中
    check_nnapi_.Model_create = check_Model_create;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksModel_free` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Model_free` 中
    *(void**)&nnapi_.Model_free = dlsym(handle, "ANeuralNetworksModel_free");
    
    # 设置指向 `ANeuralNetworksModel_free` 函数的指针，并将其保存到 `check_nnapi_.Model_free` 中
    check_nnapi_.Model_free = check_Model_free;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksModel_finish` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Model_finish` 中
    *(void**)&nnapi_.Model_finish = dlsym(handle, "ANeuralNetworksModel_finish");
    
    # 设置指向 `ANeuralNetworksModel_finish` 函数的指针，并将其保存到 `check_nnapi_.Model_finish` 中
    check_nnapi_.Model_finish = check_Model_finish;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksModel_addOperand` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Model_addOperand` 中
    *(void**)&nnapi_.Model_addOperand = dlsym(handle, "ANeuralNetworksModel_addOperand");
    
    # 设置指向 `ANeuralNetworksModel_addOperand` 函数的指针，并将其保存到 `check_nnapi_.Model_addOperand` 中
    check_nnapi_.Model_addOperand = check_Model_addOperand;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksModel_setOperandValue` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Model_setOperandValue` 中
    *(void**)&nnapi_.Model_setOperandValue = dlsym(handle, "ANeuralNetworksModel_setOperandValue");
    
    # 设置指向 `ANeuralNetworksModel_setOperandValue` 函数的指针，并将其保存到 `check_nnapi_.Model_setOperandValue` 中
    check_nnapi_.Model_setOperandValue = check_Model_setOperandValue;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksModel_setOperandValueFromMemory` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Model_setOperandValueFromMemory` 中
    *(void**)&nnapi_.Model_setOperandValueFromMemory = dlsym(handle, "ANeuralNetworksModel_setOperandValueFromMemory");
    
    # 设置指向 `ANeuralNetworksModel_setOperandValueFromMemory` 函数的指针，并将其保存到 `check_nnapi_.Model_setOperandValueFromMemory` 中
    check_nnapi_.Model_setOperandValueFromMemory = check_Model_setOperandValueFromMemory;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksModel_addOperation` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Model_addOperation` 中
    *(void**)&nnapi_.Model_addOperation = dlsym(handle, "ANeuralNetworksModel_addOperation");
    
    # 设置指向 `ANeuralNetworksModel_addOperation` 函数的指针，并将其保存到 `check_nnapi_.Model_addOperation` 中
    check_nnapi_.Model_addOperation = check_Model_addOperation;
    
    # 通过 `dlsym` 函数获取 `ANeuralNetworksModel_identifyInputsAndOutputs` 函数的地址，将其转换为函数指针，并保存到 `nnapi_.Model_identifyInputsAndOutputs` 中
    *(void**)&nnapi_.Model_identifyInputsAndOutputs = dlsym(handle, "ANeuralNetworksModel_identifyInputsAndOutputs");
    
    # 设置指向 `ANeuralNetworksModel_identifyInputsAndOutputs` 函数的指针，并将其保存到 `check_nnapi_.Model_identifyInputsAndOutputs` 中
    check_nnapi_.Model_identifyInputsAndOutputs = check_Model_identifyInputsAndOutputs;
    
    # 通过
    *(void**)&nnapi_.Compilation_free = dlsym(handle, "ANeuralNetworksCompilation_free");
    # 将动态链接库中的函数符号 ANeuralNetworksCompilation_free 赋值给 nnapi_.Compilation_free

    check_nnapi_.Compilation_free = check_Compilation_free;
    # 将自定义的检查函数 check_Compilation_free 赋值给 check_nnapi_.Compilation_free

    *(void**)&nnapi_.Compilation_setPreference = dlsym(handle, "ANeuralNetworksCompilation_setPreference");
    # 将动态链接库中的函数符号 ANeuralNetworksCompilation_setPreference 赋值给 nnapi_.Compilation_setPreference

    check_nnapi_.Compilation_setPreference = check_Compilation_setPreference;
    # 将自定义的检查函数 check_Compilation_setPreference 赋值给 check_nnapi_.Compilation_setPreference

    *(void**)&nnapi_.Compilation_finish = dlsym(handle, "ANeuralNetworksCompilation_finish");
    # 将动态链接库中的函数符号 ANeuralNetworksCompilation_finish 赋值给 nnapi_.Compilation_finish

    check_nnapi_.Compilation_finish = check_Compilation_finish;
    # 将自定义的检查函数 check_Compilation_finish 赋值给 check_nnapi_.Compilation_finish

    *(void**)&nnapi_.Execution_create = dlsym(handle, "ANeuralNetworksExecution_create");
    # 将动态链接库中的函数符号 ANeuralNetworksExecution_create 赋值给 nnapi_.Execution_create

    check_nnapi_.Execution_create = check_Execution_create;
    # 将自定义的检查函数 check_Execution_create 赋值给 check_nnapi_.Execution_create

    *(void**)&nnapi_.Execution_free = dlsym(handle, "ANeuralNetworksExecution_free");
    # 将动态链接库中的函数符号 ANeuralNetworksExecution_free 赋值给 nnapi_.Execution_free

    check_nnapi_.Execution_free = check_Execution_free;
    # 将自定义的检查函数 check_Execution_free 赋值给 check_nnapi_.Execution_free

    *(void**)&nnapi_.Execution_setInput = dlsym(handle, "ANeuralNetworksExecution_setInput");
    # 将动态链接库中的函数符号 ANeuralNetworksExecution_setInput 赋值给 nnapi_.Execution_setInput

    check_nnapi_.Execution_setInput = check_Execution_setInput;
    # 将自定义的检查函数 check_Execution_setInput 赋值给 check_nnapi_.Execution_setInput

    *(void**)&nnapi_.Execution_setInputFromMemory = dlsym(handle, "ANeuralNetworksExecution_setInputFromMemory");
    # 将动态链接库中的函数符号 ANeuralNetworksExecution_setInputFromMemory 赋值给 nnapi_.Execution_setInputFromMemory

    check_nnapi_.Execution_setInputFromMemory = check_Execution_setInputFromMemory;
    # 将自定义的检查函数 check_Execution_setInputFromMemory 赋值给 check_nnapi_.Execution_setInputFromMemory

    *(void**)&nnapi_.Execution_setOutput = dlsym(handle, "ANeuralNetworksExecution_setOutput");
    # 将动态链接库中的函数符号 ANeuralNetworksExecution_setOutput 赋值给 nnapi_.Execution_setOutput

    check_nnapi_.Execution_setOutput = check_Execution_setOutput;
    # 将自定义的检查函数 check_Execution_setOutput 赋值给 check_nnapi_.Execution_setOutput

    *(void**)&nnapi_.Execution_setOutputFromMemory = dlsym(handle, "ANeuralNetworksExecution_setOutputFromMemory");
    # 将动态链接库中的函数符号 ANeuralNetworksExecution_setOutputFromMemory 赋值给 nnapi_.Execution_setOutputFromMemory

    check_nnapi_.Execution_setOutputFromMemory = check_Execution_setOutputFromMemory;
    # 将自定义的检查函数 check_Execution_setOutputFromMemory 赋值给 check_nnapi_.Execution_setOutputFromMemory

    *(void**)&nnapi_.Execution_startCompute = dlsym(handle, "ANeuralNetworksExecution_startCompute");
    # 将动态链接库中的函数符号 ANeuralNetworksExecution_startCompute 赋值给 nnapi_.Execution_startCompute

    check_nnapi_.Execution_startCompute = check_Execution_startCompute;
    # 将自定义的检查函数 check_Execution_startCompute 赋值给 check_nnapi_.Execution_startCompute

    *(void**)&nnapi_.Event_wait = dlsym(handle, "ANeuralNetworksEvent_wait");
    # 将动态链接库中的函数符号 ANeuralNetworksEvent_wait 赋值给 nnapi_.Event_wait

    check_nnapi_.Event_wait = check_Event_wait;
    # 将自定义的检查函数 check_Event_wait 赋值给 check_nnapi_.Event_wait

    *(void**)&nnapi_.Event_free = dlsym(handle, "ANeuralNetworksEvent_free");
    # 将动态链接库中的函数符号 ANeuralNetworksEvent_free 赋值给 nnapi_.Event_free

    check_nnapi_.Event_free = check_Event_free;
    # 将自定义的检查函数 check_Event_free 赋值给 check_nnapi_.Event_free

    *(void**)&nnapi_.Execution_getOutputOperandRank = dlsym(handle, "ANeuralNetworksExecution_getOutputOperandRank");
    # 将动态链接库中的函数符号 ANeuralNetworksExecution_getOutputOperandRank 赋值给 nnapi_.Execution_getOutputOperandRank

    check_nnapi_.Execution_getOutputOperandRank = check_Execution_getOutputOperandRank;
    # 将自定义的检查函数 check_Execution_getOutputOperandRank 赋值给 check_nnapi_.Execution_getOutputOperandRank

    *(void**)&nnapi_.Execution_getOutputOperandDimensions = dlsym(handle, "ANeuralNetworksExecution_getOutputOperandDimensions");
    # 将动态链接库中的函数符号 ANeuralNetworksExecution_getOutputOperandDimensions 赋值给 nnapi_.Execution_getOutputOperandDimensions

    check_nnapi_.Execution_getOutputOperandDimensions = check_Execution_getOutputOperandDimensions;
    # 将自定义的检查函数 check_Execution_getOutputOperandDimensions 赋值给 check_nnapi_.Execution_getOutputOperandDimensions

    loaded = 1;
  }
  *nnapi = &nnapi_;
  # 将 nnapi_ 的地址赋给 nnapi

  *check_nnapi = &check_nnapi_;
  # 将 check_nnapi_ 的地址赋给 check_nnapi
#endif

这段代码是一个预处理器指令，通常在C或C++代码中用于条件编译。它的作用是结束一个条件编译区块。在代码中的具体作用取决于前面的条件编译指令，`#ifdef` 或者 `#ifndef`，以及它们指定的条件。
```