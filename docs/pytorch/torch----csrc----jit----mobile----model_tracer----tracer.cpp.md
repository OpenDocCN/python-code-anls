# `.\pytorch\torch\csrc\jit\mobile\model_tracer\tracer.cpp`

```
/**
 * This program is designed to trace operations and operators within Torch Mobile models.
 * It accepts multiple model files (.ptl) with bundled inputs and executes them using
 * a lite interpreter. Traced operators, both root and called, are recorded and saved
 * into a YAML file specified via command line arguments.
 *
 * Root operators may include primary and other operators not invoked using the dispatcher,
 * which may not appear in the Traced Operator list.
 */

#include <iostream>
#include <sstream>
#include <string>

#include <ATen/core/dispatch/ObservedOperators.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.h>
#include <torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h>
#include <torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.h>
#include <torch/csrc/jit/mobile/model_tracer/TensorUtils.h>
#include <torch/csrc/jit/mobile/model_tracer/TracerRunner.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/parse_operators.h>
#include <torch/script.h>

typedef std::map<std::string, std::set<std::string>> kt_type;

// Define command line flags for input and output paths
C10_DEFINE_string(
    model_input_path,
    "",
    "A comma separated list of path(s) to the input model file(s) (.ptl).");

C10_DEFINE_string(
    build_yaml_path,
    "",
    "The path of the output YAML file containing traced operator information.");

// Macro to ensure required string argument is provided
#define REQUIRE_STRING_ARG(name)                            \
  if (FLAGS_##name.empty()) {                               \
    std::cerr << "You must specify the flag --" #name "\n"; \
    return 1;                                               \
  }

// Macro to ensure required integer argument is provided
#define REQUIRE_INT_ARG(name)                               \
  if (FLAGS_##name == -1) {                                 \
    std::cerr << "You must specify the flag --" #name "\n"; \
    return 1;                                               \
  }

// Function to print YAML representation of a single operator
void printOpYAML(
    std::ostream& out,
    int indent,
    const std::string& op_name,
    bool is_used_for_training,
    bool is_root_operator,
    bool include_all_overloads) {
  out << std::string(indent, ' ') << op_name << ":" << std::endl;
  out << std::string(indent + 2, ' ')
      << "is_used_for_training: " << (is_used_for_training ? "true" : "false")
      << std::endl;
  out << std::string(indent + 2, ' ')
      << "is_root_operator: " << (is_root_operator ? "true" : "false")
      << std::endl;
  out << std::string(indent + 2, ' ')
      << "include_all_overloads: " << (include_all_overloads ? "true" : "false")
      << std::endl;
}

// Function to print YAML representation of multiple operators
void printOpsYAML(
    std::ostream& out,
    const std::set<std::string>& operator_list,
    bool is_used_for_training,
    bool is_root_operator,
    bool include_all_overloads) {
  for (auto& it : operator_list) {
    printOpYAML(out, 2, it, false, is_root_operator, false);
  }
}
  // 打印 DType 的 YAML 表示到输出流中，带有指定的缩进和内核标签名
void printDTypeYAML(
    std::ostream& out,                          // 输出流对象
    int indent,                                 // 缩进级别
    const std::string& kernel_tag_name,         // 内核标签名
    const std::set<std::string> dtypes) {       // 包含数据类型的集合
  std::string indent_str = std::string(indent, ' ');  // 创建指定数量空格的缩进字符串
  out << indent_str << kernel_tag_name << ":" << std::endl;  // 输出内核标签名到输出流
  for (auto& dtype : dtypes) {                  // 遍历数据类型集合
    out << indent_str << "- " << dtype << std::endl;  // 输出每个数据类型带有前导符号 '-' 到输出流
  }
}

// 打印多个内核 DType 的 YAML 表示到输出流中
void printDTypesYAML(
    std::ostream& out,                                      // 输出流对象
    const torch::jit::mobile::KernelDTypeTracer::kernel_tags_type& kernel_tags) {  // 内核标签名到数据类型集合的映射
  for (auto& it : kernel_tags) {                             // 遍历内核标签名到数据类型集合的映射
    printDTypeYAML(out, 2, it.first, it.second);             // 调用打印单个 DType 的 YAML 表示的函数
  }
}

// 打印加载/使用的 TorchBind 自定义类到输出流中的 YAML 表示
void printCustomClassesYAML(
    std::ostream& out,                                      // 输出流对象
    const torch::jit::mobile::CustomClassTracer::custom_classes_type& loaded_classes) {  // 加载的自定义类名称集合
  for (auto& class_name : loaded_classes) {                  // 遍历加载的自定义类名称集合
    out << "- " << class_name << std::endl;                  // 输出每个类名带有前导符号 '-' 到输出流
  }
}

/**
 * 运行多个 PyTorch Lite 解释器模型，并额外写出根和调用的运算符、内核 DType 和加载/使用的 TorchBind 自定义类的列表。
 */
int main(int argc, char* argv[]) {
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {           // 解析命令行参数
    std::cerr << "Failed to parse command line flags!" << std::endl;  // 输出错误信息到标准错误流
    return 1;                                                // 返回错误码
  }

  REQUIRE_STRING_ARG(model_input_path);                      // 确保存在模型输入路径参数
  REQUIRE_STRING_ARG(build_yaml_path);                       // 确保存在生成的 YAML 路径参数

  std::istringstream sin(FLAGS_model_input_path);            // 使用模型输入路径初始化字符串流对象
  std::ofstream yaml_out(FLAGS_build_yaml_path);             // 使用生成的 YAML 路径初始化输出文件流对象

  std::cout << "Output: " << FLAGS_build_yaml_path << std::endl;  // 输出生成的 YAML 路径到标准输出流
  torch::jit::mobile::TracerResult tracer_result;            // 定义用于跟踪结果的对象
  std::vector<std::string> model_input_paths;                // 存储模型输入路径的向量

  for (std::string model_input_path;                          // 遍历模型输入路径
       std::getline(sin, model_input_path, ',')) {            // 使用逗号分隔符分割
    std::cout << "Processing: " << model_input_path << std::endl;  // 输出正在处理的模型输入路径到标准输出流
    model_input_paths.push_back(model_input_path);            // 将模型输入路径添加到向量中
  }

  try {
    tracer_result = torch::jit::mobile::trace_run(model_input_paths);  // 执行模型跟踪运行
  } catch (std::exception& ex) {
    std::cerr
        << "ModelTracer has not been able to load the module for the following reasons:\n"  // 输出模型加载失败的错误信息
        << ex.what()
        << "\nPlease consider opening an issue at https://github.com/pytorch/pytorch/issues "
        << "with the detailed error message." << std::endl;

    throw ex;                                                // 抛出异常
  }

  if (tracer_result.traced_operators.size() <=                // 如果跟踪的运算符数量小于等于默认值
      torch::jit::mobile::always_included_traced_ops.size()) {  // 默认跟踪运算符的大小
    std::cerr
        << c10::str(
               "Error traced_operators size: ",
               tracer_result.traced_operators.size(),
               ". Expected the traced operator list to be bigger then the default size ",
               torch::jit::mobile::always_included_traced_ops.size(),
               ". Please report a bug in PyTorch.")
        << std::endl;                                        // 输出错误信息到标准错误流
  }

  // 如果跟踪的运算符同时存在于跟踪的运算符集合和根运算符集合中，则仅保留在根运算符集合中
  for (const auto& root_op : tracer_result.root_ops) {       // 遍历根运算符集合
    if (tracer_result.traced_operators.find(root_op) !=      // 如果根运算符存在于跟踪的运算符集合中
        tracer_result.traced_operators.end()) {
      tracer_result.traced_operators.erase(root_op);         // 从跟踪的运算符集合中移除根运算符
  }
}

yaml_out << "include_all_non_op_selectives: false" << std::endl;
// 将字符串写入 YAML 输出流，设定包含所有非操作选择项为假

yaml_out << "build_features: []" << std::endl;
// 将空列表写入 YAML 输出流，表示构建特性为空

yaml_out << "operators:" << std::endl;
// 将字符串写入 YAML 输出流，表示接下来将写入操作符信息

printOpsYAML(
    yaml_out,
    tracer_result.root_ops,
    false /* is_used_for_training */,
    true /* is_root_operator */,
    false /* include_all_overloads */
);
// 调用函数 printOpsYAML 将根操作符信息写入 YAML 输出流，
// 标记不用于训练、是根操作符、不包含所有重载版本

printOpsYAML(
    yaml_out,
    tracer_result.traced_operators,
    false /* is_used_for_training */,
    false /* is_root_operator */,
    false /* include_all_overloads */
);
// 调用函数 printOpsYAML 将跟踪的操作符信息写入 YAML 输出流，
// 标记不用于训练、不是根操作符、不包含所有重载版本

yaml_out << "kernel_metadata:";
// 将字符串写入 YAML 输出流，表示接下来将写入内核元数据信息

if (tracer_result.called_kernel_tags.empty()) {
  yaml_out << " []";
}
// 如果调用的内核标签为空，将空列表写入 YAML 输出流

yaml_out << std::endl;
// 写入换行符到 YAML 输出流

printDTypesYAML(yaml_out, tracer_result.called_kernel_tags);
// 调用函数 printDTypesYAML 将调用的内核标签信息写入 YAML 输出流

yaml_out << "custom_classes:";
// 将字符串写入 YAML 输出流，表示接下来将写入自定义类信息

if (tracer_result.loaded_classes.empty()) {
  yaml_out << " []";
}
// 如果加载的类为空，将空列表写入 YAML 输出流

yaml_out << std::endl;
// 写入换行符到 YAML 输出流

printCustomClassesYAML(yaml_out, tracer_result.loaded_classes);
// 调用函数 printCustomClassesYAML 将加载的自定义类信息写入 YAML 输出流

return 0;
// 返回 0，表示程序正常结束
}


注释：


# 这行代码关闭了一个代码块。
```