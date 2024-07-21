# `.\pytorch\aten\src\ATen\native\quantized\cpu\init_qnnpack.cpp`

```py
#ifdef USE_PYTORCH_QNNPACK
// 如果定义了 USE_PYTORCH_QNNPACK 宏，则编译以下代码块

#include <ATen/native/quantized/cpu/init_qnnpack.h>
// 包含 QNNPACK 的初始化头文件

#include <c10/util/Exception.h>
// 包含 C10 库的异常处理工具

#include <pytorch_qnnpack.h>
// 包含 PyTorch QNNPACK 的头文件

#include <c10/util/CallOnce.h>
// 包含 C10 库的一次执行工具

namespace at {
namespace native {

void initQNNPACK() {
  // 定义静态的一次执行标志和 QNNPACK 状态
  static c10::once_flag once;
  static enum pytorch_qnnp_status qnnpackStatus =
      pytorch_qnnp_status_uninitialized;

  // 使用 C10 提供的 call_once 函数确保以下代码只执行一次，初始化 QNNPACK
  c10::call_once(once, []() { qnnpackStatus = pytorch_qnnp_initialize(); });

  // 检查 QNNPACK 的初始化状态，如果失败则抛出异常
  TORCH_CHECK(
      qnnpackStatus == pytorch_qnnp_status_success,
      "failed to initialize QNNPACK");
}

} // namespace native
} // namespace at

#endif
// 结束条件编译指令，结束 USE_PYTORCH_QNNPACK 宏的条件编译
```