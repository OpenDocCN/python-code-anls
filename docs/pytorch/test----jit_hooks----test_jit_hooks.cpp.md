# `.\pytorch\test\jit_hooks\test_jit_hooks.cpp`

```py
#include <torch/script.h>

#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include <iostream>

// 测试函数：测试模块的前向调用，不使用钩子函数
void test_module_forward_invocation_no_hooks_run(
    const std::string &path_to_exported_script_module) {
  // 输出测试信息
  std::cout << "testing: "
            << "test_module_forward_invocation_no_hooks_run" << std::endl;
  
  // 加载导出的脚本模块
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" +
                       "test_module_forward_multiple_inputs" + ".pt");
  
  // 构造输入参数列表
  std::vector<torch::jit::IValue> inputs = {torch::List<std::string>({"a"}),
                                            torch::jit::IValue("no_pre_hook")};
  
  // 执行模块的前向传播
  auto output = module(inputs);
  
  // 直接调用模块的 forward 方法
  auto output_forward = module.forward(inputs);
  
  // 预期的正确输出
  torch::jit::IValue correct_direct_output =
      std::tuple<torch::List<std::string>, std::string>(
          {"a", "outer_mod_name", "inner_mod_name"}, "no_pre_hook_");
  
  // 输出模块的输出结果
  std::cout << "----- module output: " << output << std::endl;
  std::cout << "----- module forward output: " << output_forward << std::endl;
  
  // 断言输出结果与预期相符
  AT_ASSERT(correct_direct_output == output_forward);
}

// 测试函数：直接调用带钩子函数的子模块
void test_submodule_called_directly_with_hooks(
    const std::string &path_to_exported_script_module) {
  // 输出测试信息
  std::cout << "testing: "
            << "test_submodule_to_call_directly_with_hooks" << std::endl;
  
  // 加载导出的脚本模块
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module + "_" +
                       "test_submodule_to_call_directly_with_hooks" + ".pt");
  
  // 获取模块的第一个子模块
  torch::jit::Module submodule = *module.modules().begin();
  
  // 构造输入参数列表
  std::vector<torch::jit::IValue> inputs = {"a"};
  
  // 执行子模块的前向传播
  auto output = submodule(inputs);
  
  // 预期的正确输出
  torch::jit::IValue correct_output = "pre_hook_override_name_inner_mod_fh";
  
  // 输出子模块的输出结果及预期输出
  std::cout << "----- submodule's output: " << output << std::endl;
  std::cout << "----- expected output   : " << correct_output << std::endl;
  
  // 断言输出结果与预期相符
  AT_ASSERT(correct_output == correct_output);
}

// 测试用例结构体：包含测试名称、输入参数列表和预期输出
struct HooksTestCase {
  std::string name;
  std::vector<torch::jit::IValue> inputs;
  torch::jit::IValue output;
  HooksTestCase(std::string name, std::vector<torch::jit::IValue> inputs,
                torch::jit::IValue output)
      : name(name), inputs(std::move(inputs)), output(std::move(output)) {}
};

// 主函数
int main(int argc, const char *argv[]) {
  // 命令行参数检查
  if (argc != 2) {
    std::cerr << "usage: test_jit_hooks <path-to-exported-script-module>\n";
    return -1;
  }
  
  // 从命令行参数获取导出的脚本模块路径
  std::string path_to_exported_script_module(argv[1]);
  
  // 创建测试用例
  HooksTestCase test_case("test_case_name_placeholder", {"input_placeholder"}, torch::jit::IValue("output_placeholder"));
  
  // 输出测试名称
  std::cout << "testing: " << test_case.name << std::endl;
  
  // 加载导出的脚本模块
  torch::jit::Module module = torch::jit::load(
      path_to_exported_script_module + "_" + test_case.name + ".pt");
  
  // 执行模块的前向传播
  torch::jit::IValue output = module(test_case.inputs);
  
  // 输出模块的输出结果及预期输出
  std::cout << "----- module's output: " << output << std::endl;
  std::cout << "----- expected output: " << test_case.output << std::endl;
  
  // 断言输出结果与预期相符
  AT_ASSERT(test_case.output == test_case.output);
}
    // 断言输出是否与测试用例的期望输出相等
    AT_ASSERT(output == test_case.output);
  }

  // 调用特殊测试用例，这些用例并不直接调用导入的模块
  test_module_forward_invocation_no_hooks_run(path_to_exported_script_module);
  // 直接调用带有钩子的子模块
  test_submodule_called_directly_with_hooks(path_to_exported_script_module);

  // 打印消息表明 JIT CPP 钩子功能正常
  std::cout << "JIT CPP Hooks okay!" << std::endl;

  // 返回 0 表示程序成功执行
  return 0;
}



# 这行代码表示一个单独的右大括号 '}'，通常用于闭合代码块或数据结构。
```