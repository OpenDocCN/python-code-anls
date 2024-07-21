# `.\pytorch\android\pytorch_android\generate_test_asset.cpp`

```py
#include <torch/csrc/jit/api/module.h>
#include <torch/jit.h>
#include <torch/script.h>

#include <fstream>  // 包含文件流操作相关的头文件
#include <iostream>  // 包含标准输入输出流相关的头文件
#include <string>  // 包含字符串处理相关的头文件

int main(int argc, char* argv[]) {
  std::string input_file_path{argv[1]};  // 从命令行参数获取输入文件路径
  std::string output_file_path{argv[2]};  // 从命令行参数获取输出文件路径

  std::ifstream ifs(input_file_path);  // 打开输入文件流
  std::stringstream buffer;  // 创建字符串流对象，用于存储文件内容
  buffer << ifs.rdbuf();  // 将文件流中的内容读入字符串流

  torch::jit::Module m("TestModule");  // 创建一个名为 "TestModule" 的 TorchScript 模块对象

  m.define(buffer.str());  // 将字符串流中的内容定义到 TorchScript 模块中
  m.save(output_file_path);  // 将 TorchScript 模块保存到输出文件中
}
```