# `.\pytorch\caffe2\utils\proto_wrap.h`

```py
#ifndef CAFFE2_UTILS_PROTO_WRAP_H_
#define CAFFE2_UTILS_PROTO_WRAP_H_

// 包含日志记录工具的头文件，用于日志输出
#include <c10/util/Logging.h>

// 进入 caffe2 命名空间
namespace caffe2 {

// 定义 TORCH_API 关键字修饰的函数 ShutdownProtobufLibrary
// 用于关闭 Protobuf 库，在 ASAN 测试和 valgrind 测试中避免 Protobuf
// 看起来像是内存泄漏的情况。
TORCH_API void ShutdownProtobufLibrary();

} // namespace caffe2

// 结束 ifdef 预处理指令，保证头文件内容不重复引入
#endif // CAFFE2_UTILS_PROTO_WRAP_H_
```