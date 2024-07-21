# `.\pytorch\test\custom_operator\op.cpp`

```py
#include <c10/util/irange.h>
#include <torch/script.h>

#include "op.h"

#include <cstddef>
#include <string>

// 定义一个自定义操作函数，接受一个张量、一个标量和一个重复次数作为参数，并返回一个张量列表
torch::List<torch::Tensor> custom_op(
    torch::Tensor tensor,  // 输入张量
    double scalar,          // 标量
    int64_t repeat) {       // 重复次数
  torch::List<torch::Tensor> output;  // 创建一个张量列表对象
  output.reserve(repeat);             // 预留足够的空间以容纳重复次数个张量
  for (const auto i : c10::irange(repeat)) {  // 循环重复次数次
    (void)i; // 抑制未使用变量警告
    output.push_back(tensor * scalar);  // 将标量乘以输入张量并添加到输出列表中
  }
  return output;  // 返回填充好的张量列表
}

// 定义一个自定义操作函数，比较两个字符串的大小并返回结果
int64_t custom_op2(std::string s1,  // 第一个字符串
                   std::string s2) {  // 第二个字符串
  return s1.compare(s2);  // 返回字符串比较的结果
}

// 自定义自动求导函数的结构体，继承自 torch::autograd::Function
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  // 前向传播函数
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,  // 自动求导上下文
      torch::Tensor var1,  // 输入张量1
      int64_t mul,          // 整数倍数
      torch::Tensor var2,   // 输入张量2
      std::optional<torch::Tensor> var3) {   // 可选输入张量3
    ctx->saved_data["mul"] = mul;  // 在上下文中保存倍数
    ctx->saved_data["var3_has_value"] = var3.has_value();  // 保存 var3 是否有值的状态
    ctx->save_for_backward({var1, var2});  // 在上下文中保存 var1 和 var2
    if (var3) {  // 如果 var3 有值
      return var1 + mul * var2 + var1 * var2 + var3.value();  // 返回计算结果
    }
    return var1 + mul * var2 + var1 * var2;  // 返回计算结果（不包括 var3）
  }

  // 反向传播函数
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_output) {
    int mul = ctx->saved_data["mul"].toInt();  // 从上下文中获取倍数
    bool var3_has_value = ctx->saved_data["var3_has_value"].toBool();  // 获取 var3 是否有值的状态
    auto saved = ctx->get_saved_variables();  // 获取保存的变量
    auto var1 = saved[0];  // 从保存的变量中获取 var1
    auto var2 = saved[1];  // 从保存的变量中获取 var2
    auto var3_grad = var3_has_value ? grad_output[0] : torch::Tensor();  // 根据 var3 是否有值选择性地获取梯度
    torch::autograd::variable_list output = {  // 定义输出梯度列表
        grad_output[0] + grad_output[0] * var2,  // 计算梯度
        torch::Tensor(),  // 空张量
        grad_output[0] * mul + grad_output[0] * var1,  // 计算梯度
        var3_grad};  // 添加 var3 的梯度
    return output;  // 返回输出梯度列表
  }
};

// 定义一个使用自定义自动求导函数的操作，接受一个张量、一个整数倍数、一个张量和一个可选张量作为参数，并返回一个张量
torch::Tensor custom_op_with_autograd(
    torch::Tensor var1,  // 输入张量1
    int64_t mul,          // 整数倍数
    torch::Tensor var2,   // 输入张量2
    std::optional<torch::Tensor> var3) {  // 可选输入张量3
  return CustomOpAutogradFunction::apply(var1, mul, var2, var3);  // 调用自定义自动求导函数的前向传播
}

// 定义一个自定义非零函数，接受一个张量作为参数，并返回非零元素的索引
torch::Tensor custom_nonzero(torch::Tensor x) {  // 输入张量
  return x.nonzero();  // 返回非零元素的索引
}

// 定义一个自定义正弦函数，接受一个张量作为参数，并返回张量的正弦值
torch::Tensor custom_sin(torch::Tensor x) {  // 输入张量
  return x.sin();  // 返回张量的正弦值
}

// 定义一个自定义库的片段，为 custom 库注册实现和抽象
TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.impl_abstract_pystub("my_custom_ops2");  // 注册 my_custom_ops2 的抽象实现
    m.def("op", custom_op);  // 定义 op 操作，使用 custom_op 函数
    m.def("op2", custom_op2);  // 定义 op2 操作，使用 custom_op2 函数
    m.def("op_with_defaults(Tensor tensor, float scalar = 1, int repeat = 1) -> Tensor[]", custom_op);  // 定义带默认参数的 op_with_defaults 操作，使用 custom_op 函数
    m.def("op_with_autograd(Tensor var1, int mul, Tensor var2, Tensor? var3=None) -> Tensor", custom_op_with_autograd);  // 定义带自动求导的 op_with_autograd 操作，使用 custom_op_with_autograd 函数
    m.def("sin(Tensor x) -> Tensor");  // 定义 sin 操作，使用 custom_sin 函数
    m.def("cos(Tensor x) -> Tensor");  // 定义 cos 操作
}

// 定义一个自定义库的片段，为 custom 库注册实现和抽象
TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.impl_abstract_pystub("my_custom_ops");  // 注册 my_custom_ops 的抽象实现
    m.def("nonzero(Tensor x) -> Tensor");  // 定义 nonzero 操作，使用 custom_nonzero 函数
}

// 定义一个自定义库的片段，为 custom 库注册实现和抽象
TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.impl_abstract_pystub("nonexistent");  // 注册 nonexistent 的抽象实现
    m.def("asin(Tensor x) -> Tensor");  // 定义 asin 操作
}

// 定义一个自定义库的片段，为 custom 库注册实现
TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.def("tan(Tensor x) -> Tensor");  // 定义 tan 操作
}

// 定义一个自定义库的实现，为 custom 库注册实现
TORCH_LIBRARY_IMPL(custom, CPU, m) {
  m.impl("nonzero", &custom_nonzero);  // 实现 nonzero 操作，使用 custom_nonzero 函数
  m.impl("sin", &custom_sin);  // 实现 sin 操作，使用 custom_sin 函数
  m.impl("asin", &at::asin);  // 实现 asin 操作，使用 at::asin 函数
}
```