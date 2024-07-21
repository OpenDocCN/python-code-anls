# `.\pytorch\caffe2\utils\threadpool\ThreadPoolCommon.h`

```py
#ifndef CAFFE2_UTILS_THREADPOOL_COMMON_H_
#define CAFFE2_UTILS_THREADPOOL_COMMON_H_

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

// caffe2 依赖于 NNPACK，而 NNPACK 又依赖于这个线程池，所以无法在这里引用 core/common.h

// 从 core/common.h 中的 C10_MOBILE 定义中复制而来
// 在构建 iOS 或 Android 设备时定义为启用状态
#if defined(__ANDROID__)
#define C10_ANDROID 1
#elif (defined(__APPLE__) &&                                            \
       (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define C10_IOS 1
#endif // ANDROID / IOS

#endif  // CAFFE2_UTILS_THREADPOOL_COMMON_H_
```