# `.\pytorch\aten\src\ATen\nnapi\nnapi_register.cpp`

```
// 包含 ATen 库中的 NNAPI 绑定头文件
#include <ATen/nnapi/nnapi_bind.h>

// 如果编译目标是 iOS，定义 IS_IOS_NNAPI_BIND 宏
#ifdef __APPLE__
  // 包含目标操作系统宏定义头文件
  #include <TargetConditionals.h>
  // 如果是 iOS 平台，定义 IS_IOS_NNAPI_BIND 宏
  #if TARGET_OS_IPHONE
    #define IS_IOS_NNAPI_BIND
  #endif
#endif

// 如果未定义 IS_IOS_NNAPI_BIND 宏，则注册 _nnapi Torch 库
#ifndef IS_IOS_NNAPI_BIND
TORCH_LIBRARY(_nnapi, m) {
  // 声明 torch::nnapi::bind::NnapiCompilation 类型为 Torch 库中的 Compilation 类
  m.class_<torch::nnapi::bind::NnapiCompilation>("Compilation")
    // 使用默认构造函数进行初始化
    .def(torch::jit::init<>())
    // 绑定 init 方法到 NnapiCompilation 类的 init 方法
    .def("init", &torch::nnapi::bind::NnapiCompilation::init)
    // 绑定 init2 方法到 NnapiCompilation 类的 init2 方法
    .def("init2", &torch::nnapi::bind::NnapiCompilation::init2)
    // 绑定 run 方法到 NnapiCompilation 类的 run 方法
    .def("run", &torch::nnapi::bind::NnapiCompilation::run)
    ;
}
// 如果定义了 IS_IOS_NNAPI_BIND 宏，则取消 IS_IOS_NNAPI_BIND 宏定义
#else
  #undef IS_IOS_NNAPI_BIND
#endif
```