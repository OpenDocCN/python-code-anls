# `.\pytorch\torch\csrc\jit\tensorexpr\external_functions_core.cpp`

```
#include <torch/csrc/jit/tensorexpr/external_functions_core.h>
// 包含外部函数核心头文件

namespace torch::jit::tensorexpr {

#ifdef C10_MOBILE
extern "C" {
#endif

using ParallelCallee = void (*)(int64_t, int8_t*);
// 定义并声明一个函数指针类型 ParallelCallee，指向接受 int64_t 和 int8_t* 参数并返回 void 的函数

void DispatchParallel(
    int8_t* func,
    int64_t start,
    int64_t stop,
    int8_t* packed_data) noexcept {
  // 定义名为 DispatchParallel 的函数，接受一个函数指针 func、起始索引 start、结束索引 stop 和打包数据 packed_data，不会抛出异常

  // TODO: preserve the func type.
  // TODO: 保留 func 的类型。

  try {
    // 尝试执行以下代码块

    // 将 func 转换为 ParallelCallee 类型的函数指针
    ParallelCallee callee = reinterpret_cast<ParallelCallee>(func);

    // 使用 at::parallel_for 并行执行循环，将任务分割成若干片段，在每个片段上调用 callee 函数
    at::parallel_for(start, stop, 1, [&](int64_t f_begin, int64_t f_end) {
      for (int64_t index = f_begin; index < f_end; index++) {
        callee(index, packed_data);  // 调用 callee 函数，传递当前索引 index 和 packed_data
      }
    });

  } catch (...) {
    // 捕获所有异常，不做任何处理
  }
}

void nnc_aten_free(int64_t bufs_num, void** ptrs) noexcept {
  // 定义名为 nnc_aten_free 的函数，接受缓冲区数 bufs_num 和指向指针数组的 ptrs，不会抛出异常

  // 使用范围遍历循环，对每个缓冲区执行释放操作
  for (const auto i : c10::irange(bufs_num)) {
    c10::raw::intrusive_ptr::decref((c10::TensorImpl*)ptrs[i]);
    // 调用 c10::raw::intrusive_ptr::decref 函数，将 ptrs[i] 强制转换为 c10::TensorImpl* 类型，递减引用计数
  }
}

#ifdef C10_MOBILE
} // extern "C"
#endif

} // namespace torch::jit::tensorexpr
// 结束 torch::jit::tensorexpr 命名空间
```