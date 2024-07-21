# `.\pytorch\test\cpp\aoti_inference\aoti_custom_class.h`

```py
#pragma once
// 预处理指令，确保头文件只包含一次

#include <memory>
// 包含标准库的内存管理组件

#include <torch/torch.h>
// 包含 Torch C++ 库的头文件

namespace torch::inductor {
// 声明 torch::inductor 命名空间

class AOTIModelContainerRunner;
// 声明 AOTIModelContainerRunner 类
} // namespace torch::inductor

namespace torch::aot_inductor {
// 声明 torch::aot_inductor 命名空间

class MyAOTIClass : public torch::CustomClassHolder {
// 定义 MyAOTIClass 类，继承自 torch::CustomClassHolder
 public:
  explicit MyAOTIClass(
      const std::string& model_path,
      const std::string& device = "cuda");
  // 构造函数，接受模型路径和设备类型作为参数

  ~MyAOTIClass() {}
  // 析构函数，无需执行任何操作

  MyAOTIClass(const MyAOTIClass&) = delete;
  // 禁用拷贝构造函数

  MyAOTIClass& operator=(const MyAOTIClass&) = delete;
  // 禁用拷贝赋值运算符

  MyAOTIClass& operator=(MyAOTIClass&&) = delete;
  // 禁用移动赋值运算符

  const std::string& lib_path() const {
    return lib_path_;
  }
  // 返回模型库路径的常量引用的成员函数

  const std::string& device() const {
    return device_;
  }
  // 返回设备类型的常量引用的成员函数

  std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs);
  // 前向推理函数，接受输入 Tensor 列表并返回输出 Tensor 列表

 private:
  const std::string lib_path_;
  // 常量成员变量，存储模型库路径

  const std::string device_;
  // 常量成员变量，存储设备类型

  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner_;
  // 指向 AOTIModelContainerRunner 对象的独占指针
};

} // namespace torch::aot_inductor
// 结束 torch::aot_inductor 命名空间
```