# `.\pytorch\torch\csrc\jit\python\python_arg_flatten.h`

```py
#pragma once

#include <c10/util/hash.h>  // 引入 c10 库中的哈希函数
#include <c10/util/irange.h>  // 引入 c10 库中的范围函数
#include <torch/csrc/autograd/variable.h>  // 引入 Torch 自动求导变量定义
#include <torch/csrc/jit/python/pybind.h>  // 引入 Torch Python 绑定

#include <ATen/ATen.h>  // 引入 ATen 库
#include <functional>  // 引入函数对象标准库
#include <tuple>  // 引入元组标准库
#include <vector>  // 引入向量标准库

namespace torch::jit::python {

// 描述输入输出的结构体
struct IODescriptor {
  // 变量的元数据结构
  struct VariableMetadata {
    // 构造函数，根据自动求导变量初始化元数据
    VariableMetadata(const autograd::Variable& var)
        : sizes(var.sizes().vec()),  // 记录变量尺寸
          type(var.scalar_type()),  // 记录变量数据类型
          device(var.device()),  // 记录变量设备
          requires_grad(var.requires_grad()) {}  // 记录变量是否需要梯度

    // 比较运算符重载，判断两个变量元数据是否相等
    bool operator==(const VariableMetadata& o) const {
      return std::tie(device, requires_grad, type, sizes) ==
          std::tie(o.device, o.requires_grad, o.type, o.sizes);
    }

    // 计算变量元数据的哈希值
    static size_t hash(const VariableMetadata& m) {
      return c10::get_hash(m.sizes, m.device, m.requires_grad, m.type);
    }

    std::vector<int64_t> sizes;  // 变量尺寸
    at::ScalarType type;  // 变量数据类型
    at::Device device;  // 变量设备
    bool requires_grad;  // 是否需要梯度
  };

  // 比较运算符重载，判断两个描述符是否相等
  bool operator==(const IODescriptor& o) const {
    return std::tie(structure, metadata, grad_enabled) ==
        std::tie(o.structure, o.metadata, o.grad_enabled);
  }

  // 计算描述符的哈希值
  static size_t hash(const IODescriptor& o) {
    return c10::get_hash(o.structure, o.metadata, o.grad_enabled);
  }

  // 扩展元数据列表
  void extend(const autograd::variable_list& list) {
    metadata.reserve(metadata.size() + list.size());
    for (auto& var : list)
      metadata.emplace_back(var);
  }

  // 描述参数结构的字符串，变量用不同的字符表示，取决于其标志位，
  // 元组和列表的开始和结束用对应种类的括号表示，括号应该成对出现。
  // 示例 desc: (vv[v(v)v])
  // 注意：如果调用了 extend()，则 metadata.size() 可能与 structure 中 'v' 的数量不同。
  std::string structure;  // 参数结构描述字符串
  std::vector<std::string> strings;  // 字符串向量
  std::vector<VariableMetadata> metadata;  // 变量元数据向量
  bool grad_enabled = false;  // 梯度是否启用
};

// 变量元数据的输出流运算符重载
static inline std::ostream& operator<<(
    std::ostream& out,
    const IODescriptor::VariableMetadata& meta) {
  at::Device meta_device = meta.device;
  auto& t = at::getDeprecatedTypeProperties(
      meta_device.is_cpu() ? at::Backend::CPU : at::Backend::CUDA, meta.type);
  out << t << "(requires_grad=" << meta.requires_grad;
  if (meta_device.is_cuda()) {
    out << ", device=" << meta_device.index();
  }
  out << ") {";
  for (const auto i : c10::irange(meta.sizes.size())) {
    if (i > 0)
      out << ", ";
    out << meta.sizes[i];
  }
  out << "}";
  return out;
}

// 描述符的输出流运算符重载
static inline std::ostream& operator<<(
    std::ostream& out,
    const IODescriptor& desc) {
  out << desc.structure << "\n";  // 输出描述符结构
  out << "  with grad_enabled=" << desc.grad_enabled << "\n";  // 输出是否启用梯度
  for (const auto i : c10::irange(desc.metadata.size())) {
    out << "  with v" << i << " having type " << desc.metadata[i] << "\n";  // 输出每个变量的类型和元数据
  }
  return out;
}

}  // namespace torch::jit::python
struct ParsedArgs {
  // Flat vector of Variables found in arguments
  // 存储在参数中找到的变量的扁平向量
  autograd::variable_list vars;
  // Metadata describing nesting of objects received from Python and
  // metadata of vars and whether grad is enabled.
  // 描述从 Python 接收的对象嵌套结构的元数据，以及变量的元数据和梯度是否已启用
  IODescriptor desc;

  void extend(const autograd::variable_list& list) {
    if (list.empty())
      return;
    // 扩展当前变量列表
    vars.reserve(vars.size() + list.size());
    // 将新列表中的变量逐个添加到 vars 中
    for (auto& var : list)
      vars.emplace_back(var);
    // 扩展描述符以反映新变量的结构
    desc.extend(list);
  }
};

// 将 Python 对象展平为 ParsedArgs 结构
ParsedArgs flatten(py::handle obj);

// 将变量列表 unflatten 为 Python 对象
PyObject* unflatten(
    at::ArrayRef<autograd::Variable> vars,
    const IODescriptor& structure);

// 命名空间结束声明
} // namespace torch::jit::python
```