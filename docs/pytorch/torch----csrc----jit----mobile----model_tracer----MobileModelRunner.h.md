# `.\pytorch\torch\csrc\jit\mobile\model_tracer\MobileModelRunner.h`

```py
// 预处理指令，确保头文件仅被包含一次
#pragma once

// 包含互斥量和字符串流头文件
#include <mutex>
#include <sstream>

// 包含 Torch 的自动微分模块、移动端模型导入、移动端模型模块、序列化导出以及 Torch 脚本头文件
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/script.h>

// 命名空间开始：torch -> jit -> mobile
namespace torch {
namespace jit {
namespace mobile {

// 移动端模型运行器类定义开始
class MobileModelRunner {
  std::shared_ptr<torch::jit::mobile::Module> module_;  // 私有成员变量：共享指针指向移动端模型模块

 public:
  // 显式构造函数：根据文件路径加载移动端模型
  explicit MobileModelRunner(std::string const& file_path) {
    module_ = std::make_shared<torch::jit::mobile::Module>(
        torch::jit::_load_for_mobile(file_path));  // 加载移动端模型到模块指针
  }

  // 构造函数：根据文件路径和模块加载选项加载移动端模型
  MobileModelRunner(
      std::string const& file_path,
      uint64_t module_load_options) {
    std::unordered_map<std::string, std::string> extra_files;  // 定义空的附加文件映射
    module_ = std::make_shared<torch::jit::mobile::Module>(
        torch::jit::_load_for_mobile(
            file_path,                                 // 文件路径
            at::Device(at::DeviceType::CPU, 0),        // 设备类型和索引
            extra_files,                               // 附加文件映射
            module_load_options));                     // 模块加载选项
  }

  // 构造函数：根据字符串流加载移动端模型
  MobileModelRunner(std::stringstream oss) {
    module_ = std::make_shared<torch::jit::mobile::Module>(
        torch::jit::_load_for_mobile(oss, at::Device(at::DeviceType::CPU, 0)));
  }

  /**
   * Returns true if the list of operators passed in has a Metal GPU operator,
   * and false otherwise.
   *
   */
  static bool set_has_metal_gpu_operators(std::set<std::string> const& op_list);

  /**
   * Fetches the set of root operators in the file "extra/mobile_info.json"
   * within the .ptl archive at location file_path.
   *
   * An exception is thrown if:
   *
   * 1. The file at file_path does not exist, or
   * 2. The contents of extra/mobile_info.json is not a JSON, or
   * 3. The file extra/mobile_info.json does not exist, or
   * 4. The JSON is malformed in some way and the operator list can not be
   * extracted correctly.
   *
   */
  static std::set<std::string> get_operators_from_mobile_info_json(
      std::string const& file_path);

  static std::vector<std::vector<at::IValue>> ivalue_to_bundled_inputs(
      const c10::IValue& bundled_inputs);

  static std::unordered_map<std::string, std::string>
  ivalue_to_bundled_inputs_map(const c10::IValue& bundled_inputs);

  /**
   * Fetches all the bundled inputs of the loaded mobile model.
   *
   * A bundled input itself is of type std::vector<at::IValue> and the
   * elements of this vector<> are the arguments that the "forward"
   * method of the model accepts. i.e. each of the at::IValue is a
   * single argument to the model's "forward" method.
   *
   * The outer vector holds a bundled input. For models with bundled
   * inputs, the outer most vector will have size > 0.
   */
  std::vector<std::vector<at::IValue>> get_all_bundled_inputs();

  /**
   * Fetches all the bundled inputs for all functions of the loaded mobile
   * model.
   *
   * The mapping is from 'function_names' eg 'forward' to bundled inputs for
   * that function
   *
   * A bundled input itself is of type std::vector<at::IValue> and the
   * elements of this vector<> are the arguments that the corresponding
   * method of the model accepts. i.e. each of the at::IValue in the entry
   * for forward is a single argument to the model's "forward" method.
   *
   * The outer vector of each value holds a bundled input. For models with
   * bundled inputs, the outer most vector will have size > 0.
   */
  std::unordered_map<std::string, std::vector<std::vector<at::IValue>>>
  get_many_functions_bundled_inputs();

  /**
   * Returns true if a model possesses get_bundled_inputs_functions_and_info()
   */
  bool has_new_style_bundled_inputs() const {
  /**
   * 检查模块是否具有方法"get_bundled_inputs_functions_and_info"，并返回是否不为空值。
   */
  return module_->find_method("get_bundled_inputs_functions_and_info") !=
      c10::nullopt;
}

/**
 * 对于每个捆绑输入中的张量，调用用户提供的函数'func'。
 */
void for_each_tensor_in_bundled_inputs(
    std::function<void(const ::at::Tensor&)> const& func);

/**
 * 获取直接被此模型字节码调用的根操作符。
 */
std::set<std::string> get_root_operators() {
  return torch::jit::mobile::_export_operator_list(*module_);
}

/**
 * 使用模型的"forward"方法运行模型对所有提供的输入。返回一个std::vector<at::IValue>，
 * 其中返回向量的每个元素都是调用forward()后的返回值之一。
 */
std::vector<at::IValue> run_with_inputs(
    std::vector<std::vector<at::IValue>> const& bundled_inputs);

/**
 * 使用模型的指定函数名在所有提供的输入上运行模型。返回一个std::vector<at::IValue>，
 * 其中返回向量的每个元素都是在该模型上调用名称为"function_name"的方法后的返回值之一。
 */
std::vector<at::IValue> run_with_inputs(
    const std::string& function_name,
    std::vector<std::vector<at::IValue>> const& bundled_inputs) const;

/**
 * 尝试运行传入列表中的所有函数（如果存在）。所有函数都不应该需要参数。
 */
void run_argless_functions(const std::vector<std::string>& functions);
};

// 结束 mobile 命名空间
} // namespace mobile
// 结束 jit 命名空间
} // namespace jit
// 结束 torch 命名空间
} // namespace torch
```