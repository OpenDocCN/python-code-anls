# `.\pytorch\torch\csrc\lazy\ts_backend\ops\to_copy.h`

```py
#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

// This IR was copied from code-generated output, but the entire _to_copy
// operator cannot be trivially code genereated since it is only desirable to
// capture IR for certain permutaions of _to_copy (e.g. dtype), and for the
// others it is difficult to even invoke the aten/eager fallback necessitating
// directly implementing the right to(device) behavior

// 表示 _to_copy 操作的 IR，从代码生成的输出中复制而来，但整个 _to_copy 操作
// 不能被轻易地代码生成，因为只有对某些 _to_copy 的排列（例如 dtype）才希望捕获 IR，
// 对于其他的情况，甚至调用 aten/eager 回退都很困难，需要直接实现正确的 to(device) 行为
class ToCopy : public torch::lazy::TsNode {
 public:
  // 静态方法，返回类的操作类型 OpKind
  static OpKind ClassOpKind() {
    return OpKind(at::aten::_to_copy);
  }

  // 构造函数，初始化 ToCopy 类
  ToCopy(
      const torch::lazy::Value& self,                          // 输入值 self
      const std::optional<at::ScalarType>& dtype,              // 可选的数据类型 dtype
      const std::optional<at::Layout>& layout,                 // 可选的布局 layout
      const std::optional<at::Device>& device,                 // 可选的设备 device
      const std::optional<bool>& pin_memory,                   // 可选的 pin_memory
      const bool& non_blocking,                                // 非阻塞标志
      const std::optional<at::MemoryFormat>& memory_format,    // 可选的内存格式 memory_format
      std::vector<torch::lazy::Shape>&& shapes)                // 移动语义的形状向量 shapes
      : torch::lazy::TsNode(
            ClassOpKind(),                                    // 调用基类构造函数，传入操作类型
            {self},                                           // 输入值 self
            std::move(shapes),                                // 移动语义的形状向量 shapes
            /* num_outputs */ 1,                              // 输出数量为1
            torch::lazy::MHash(                               // 调用 MHash 函数计算哈希值
                dtype,                                        // 数据类型 dtype
                layout,                                       // 布局 layout
                device,                                       // 设备 device
                pin_memory,                                   // pin_memory
                non_blocking,                                 // 非阻塞标志
                memory_format)),                              // 内存格式 memory_format

        dtype(dtype),                                         // 初始化成员变量 dtype
        layout(layout),                                       // 初始化成员变量 layout
        device(device),                                       // 初始化成员变量 device
        pin_memory(pin_memory),                               // 初始化成员变量 pin_memory
        non_blocking(non_blocking),                           // 初始化成员变量 non_blocking
        memory_format(memory_format) {}                       // 初始化成员变量 memory_format

  // 判断当前 ToCopy 对象是否可以重用的方法
  bool CanBeReused(
      const torch::lazy::Value& self,                         // 输入值 self
      const std::optional<at::ScalarType>& dtype,             // 可选的数据类型 dtype
      const std::optional<at::Layout>& layout,                // 可选的布局 layout
      const std::optional<at::Device>& device,                // 可选的设备 device
      const std::optional<bool>& pin_memory,                  // 可选的 pin_memory
      const bool& non_blocking,                               // 非阻塞标志
      const std::optional<at::MemoryFormat>& memory_format) const {  // 可选的内存格式 memory_format
    size_t i = 0;
    // 返回是否可以重用的布尔值
    return (
        operand(i++) == self && this->dtype == dtype &&
        this->layout == layout && this->device == device &&
        this->pin_memory == pin_memory && this->non_blocking == non_blocking &&
        this->memory_format == memory_format);
  }

  // 重写 ToString 方法，返回 ToCopy 对象的字符串表示
  std::string ToString() const override {
    std::stringstream ss;
    ss << torch::lazy::TsNode::ToString();                    // 调用基类的 ToString 方法
    if (dtype.has_value()) {                                  // 如果 dtype 有值
      ss << ", dtype=" << dtype.value();                      // 输出 dtype 的值
    } else {
      ss << ", dtype=null";                                   // 否则输出 dtype 为空
    }
    if (layout.has_value()) {                                 // 如果 layout 有值
      ss << ", layout=" << layout.value();                    // 输出 layout 的值
    } else {
      ss << ", layout=null";                                  // 否则输出 layout 为空
    }
    if (device.has_value()) {                                 // 如果 device 有值
      ss << ", device=" << device.value();                    // 输出 device 的值
    } else {
      ss << ", device=null";                                  // 否则输出 device 为空
    }
    if (pin_memory.has_value()) {                             // 如果 pin_memory 有值
      ss << ", pin_memory=" << pin_memory.value();            // 输出 pin_memory 的值
    } else {
      ss << ", pin_memory=null";                              // 否则输出 pin_memory 为空
    }
    ss << ", non_blocking=" << non_blocking;                  // 输出 non_blocking 的值
    if (memory_format.has_value()) {                          // 如果 memory_format 有值
      ss << ", memory_format=" << memory_format.value();      // 输出 memory_format 的值
    } else {
      ss << ", memory_format=null";                           // 否则输出 memory_format 为空
    }
    return ss.str();
  }

  // 实现 Lower 方法，将 TorchScript 函数降低为指定上下文的操作向量
  torch::lazy::TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      torch::lazy::TSLoweringContext* loctx) const override {
    // 准备参数列表和关键字参数列表
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve(1); // 预留一个参数的空间
    kwarguments.reserve(6); // 预留六个关键字参数的空间
    size_t i = 0;
    // 将操作数的输出操作添加到参数列表中
    arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
    // 添加关键字参数
    kwarguments.emplace_back("dtype", dtype);
    kwarguments.emplace_back("layout", layout);
    kwarguments.emplace_back("device", device);
    kwarguments.emplace_back("pin_memory", pin_memory);
    kwarguments.emplace_back("non_blocking", non_blocking);
    kwarguments.emplace_back("memory_format", memory_format);
    // 调用 TorchScript 内置函数降低操作
    torch::lazy::TSOpVector _to_copy_out =
        torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
    // 检查返回的操作向量的大小是否为1
    TORCH_CHECK_EQ(_to_copy_out.size(), 1);

    // 返回操作向量
    return _to_copy_out;
  }

  // 可选的张量数据类型
  std::optional<at::ScalarType> dtype;
  // 可选的张量布局
  std::optional<at::Layout> layout;
  // 可选的张量设备
  std::optional<at::Device> device;
  // 可选的内存锁定标志
  std::optional<bool> pin_memory;
  // 非阻塞标志
  bool non_blocking;
  // 可选的内存格式
  std::optional<at::MemoryFormat> memory_format;
};

// 结束 lazy 命名空间
} // namespace lazy

// 结束 torch 命名空间
} // namespace torch
```