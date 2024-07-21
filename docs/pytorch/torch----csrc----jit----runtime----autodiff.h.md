# `.\pytorch\torch\csrc\jit\runtime\autodiff.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/Export.h>
// 引入 Torch 导出相关的头文件

#include <torch/csrc/jit/ir/ir.h>
// 引入 Torch 的 JIT IR 相关的头文件

#include <memory>
// 引入 C++ 标准库中的内存管理相关的头文件

#include <vector>
// 引入 C++ 标准库中的向量容器相关的头文件

namespace torch::jit {
// 进入 Torch 的 JIT 命名空间

using value_list = std::vector<Value*>;
// 定义一个类型别名 value_list，表示一个 Value 指针的向量

// clang-format off
// 禁用 clang 格式化，不对接下来的代码进行自动格式化

// Example showcasing how Gradient is constructed:
// 演示 Gradient 结构的构建方式：

// Let's assume we have a function f, `m` and `n` do not require grad
// (`n` can depend only on `m`):
// 假设我们有一个函数 f，其中 `m` 和 `n` 不需要梯度
// (`n` 可能仅依赖于 `m`)：

//   y, n = f(x, m)
// 然后，假设反向函数 f' 需要使用 `x`、`t` 和 `y` 的值。

// `t` 是在 f 主体中产生的中间值，假设它也需要梯度。

// In this case differentiate(f) will return this:
// 在这种情况下，differentiate(f) 将返回以下内容：

//   y, n, t = f(x, m)        // `t` is appended to the output list
//   dx = f'(dy, dt, x, t, y) // No `dm` or `dn` because they do not require gradient
//                            // All needed values from f are prepended to the input list

//   f_real_outputs = 2       // Only first two outputs were present in f originally
//   df_input_vjps = {0, 2}   // i.e. connect grad_fn of y and t variables produced by f,
//                    y  t    // with y's output_nr = 0 and t's output_nr = 1

//   df_input_captures = {I0, O2, O0} // Order matches the prefix of inputs to df
//                        x   t   y

//   df_output_vjps = {0}     // i.e. connect next_edge[0] of grad_fn to x's (grad_fn, output_nr).

// Terminology: vjp = vector-jacobian product
// 术语说明：vjp = 向量-雅可比积

// clang-format on

struct Gradient {
// 定义结构体 Gradient

  explicit operator bool() const {
  // 明确声明 bool 类型的类型转换函数，const 表示这个函数不会修改结构体中的成员
    // 返回指针 df 是否不为 nullptr
    return df != nullptr;
  }

  // 定义共享指针 f 和 df，分别指向 Graph 类的对象
  std::shared_ptr<Graph> f;
  std::shared_ptr<Graph> df;

  // 描述如何从 f 的图返回的数据构造输出的方式。
  // 这是必要的，因为一些末尾的输出是仅用于 df 的中间结果（应该被忽略）。
  // 初始化为安全起见
  size_t f_real_outputs = 0;

  // df 的输入分为两部分：vjps（梯度输出）和 captures（捕获值）。
  // VJPs 是用于每个输入捕获的梯度计算的“种子”。
  // Captures 是运行 f 时需要保存的值。
  // 我们特别处理输入，因为这样可以避免将额外的 vjps 添加为 df 的输入。
  
  // df_input_vjps 存储了 f 输出的偏移量。
  std::vector<size_t> df_input_vjps;

  // df_input_captured_inputs 存储了 f 输入的偏移量，这些输入被捕获。
  std::vector<size_t> df_input_captured_inputs;

  // df_input_captured_outputs 存储了 f 输出的偏移量，这些输出被捕获。
  std::vector<size_t> df_input_captured_outputs;

  // df 将会为 f 的一部分需要梯度的输入生成 vjps。
  // df_output_vjps 中的每个元素 idx 意味着 df 的第 idx 个输出为 f 的第 inp_idx 个输入生成 vjp。
  std::vector<size_t> df_output_vjps;

  // 描述如何使用梯度来实现可微分的自动求导函数：
  // 当运行 f 时：
  //   - 展开输入变量
  //   - 运行 f 的图
  //   - 创建 grad_fn
  //   - 将输出包装为 Variables （假设有一个 tensor_outputs 数组）：
  //       outputs = map(Variable, tensor_output)
  //       for i, offset in enumerate(df_input_vjps):
  //         outputs[offset].set_grad_fn(grad_fn, output_nr=i)
  //   - 使用 df_output_vjps 连接 grad_fn 的 next_edges：
  //       for idx in df_output_vjps:
  //         grad_fn.add_next_edge(inputs[idx].gradient_edge())
  //   - 保存 df 所需的 captures（需要注意使用 SavedVariables 来处理实际返回的输入和输出）
  //   - 返回 outputs[:f_real_outputs]
  //
  // 当运行 df 时：
  //   - 连接接收到的 vjps 和捕获的变量
  //   - 解释 df
  //   - 将 df 的输出包装为不需要梯度的 Variables
};
// TORCH_API 是一个宏，用于声明函数的导出或可见性，指示这些函数是 Torch 库的一部分

// 对给定的图形进行符号微分，返回梯度对象
TORCH_API Gradient differentiate(std::shared_ptr<Graph>& graph);

// 判断指定节点是否可以进行符号微分
TORCH_API bool isDifferentiable(const Node* n);

// 判断给定图形是否可以进行符号微分
TORCH_API bool isDifferentiable(Graph& g);

// 判断给定值是否为零
TORCH_API bool isZero(Value* v);

// torch::jit 是 Torch 库中的命名空间
} // namespace torch::jit
```