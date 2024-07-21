# `.\pytorch\torch\csrc\jit\mobile\model_tracer\MobileModelRunner.cpp`

```py
// 包含 Torch 的头文件：用于移动端模型追踪器的功能
#include <torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h>
#include <torch/csrc/jit/mobile/model_tracer/TensorUtils.h>

// Torch 命名空间
namespace torch {
namespace jit {
namespace mobile {

// 将输入转换为捆绑输入的函数定义
std::vector<std::vector<at::IValue>> MobileModelRunner::
    ivalue_to_bundled_inputs(const c10::IValue& bundled_inputs) {
  // 断言确保捆绑输入是一个列表
  CAFFE_ENFORCE(
      bundled_inputs.isList(),
      "Expected get_all_bundled_inputs to ",
      "return a list but got a ",
      bundled_inputs.tagKind(),
      " instead");

  // 将捆绑输入转换为列表
  c10::List<at::IValue> all_inputs = bundled_inputs.toList();
  // 断言确保至少有一个捆绑输入
  CAFFE_ENFORCE(
      !all_inputs.empty(),
      "Expected at least 1 bundled input, ",
      "but found none. Please use ",
      "torch.utils.bundled_inputs.augment_model_with_bundled_inputs to add.");

  // 准备返回的结果向量
  std::vector<std::vector<at::IValue>> ret;
  // 遍历所有捆绑输入
  for (at::IValue input : all_inputs) {
    // 断言每个输入是一个元组
    CAFFE_ENFORCE(
        input.isTuple(),
        "Expected list element to be a tuple ",
        "but got a ",
        input.tagKind(),
        " instead");
    // 将元组中的元素转换为向量并加入返回结果中
    ret.push_back(input.toTupleRef().elements());
  }

  // 返回结果
  return ret;
}

// 将输入转换为捆绑输入字典的函数定义
std::unordered_map<std::string, std::string> MobileModelRunner::
    ivalue_to_bundled_inputs_map(const c10::IValue& bundled_inputs) {
  // 断言确保捆绑输入是一个通用字典
  CAFFE_ENFORCE(
      bundled_inputs.isGenericDict(),
      "Expected get_bundled_inputs_functions_and_info to ",
      "return a dict but got a ",
      bundled_inputs.tagKind(),
      " instead");

  // 将捆绑输入转换为通用字典
  c10::Dict<at::IValue, at::IValue> all_inputs = bundled_inputs.toGenericDict();
  // 断言确保至少有一个函数带有捆绑输入
  CAFFE_ENFORCE(
      !all_inputs.empty(),
      "Expected at least 1 function with bundled inputs, ",
      "but found none. Please use ",
      "torch.utils.bundled_inputs.augment_model_with_bundled_inputs to add.");

  // 准备返回的结果字典
  std::unordered_map<std::string, std::string> ret;
  // 遍历所有捆绑输入字典的条目
  for (auto& input : all_inputs) {
    // 获取函数名和嵌套字典
    at::IValue function_name = input.key();
    at::IValue nested_dict = input.value();
    // 断言函数名是一个字符串
    CAFFE_ENFORCE(
        function_name.isString(),
        "Expected function with inputs to be a string ",
        "but got a ",
        function_name.tagKind(),
        " instead");
    // 断言嵌套字典是一个通用字典
    CAFFE_ENFORCE(
        nested_dict.isGenericDict(),
        "Expected function name to map to dictionary ",
        "but got a ",
        nested_dict.tagKind(),
        " instead");

    // 现在得到了嵌套字典，需要将其转换为标准类型
    c10::Dict<at::IValue, at::IValue> function_and_info_ival_dict =
        nested_dict.toGenericDict();
    // 准备存储函数名及其信息的字典
    std::unordered_map<std::string, std::vector<std::string>>
        function_and_info_dict;
    // 遍历 function_and_info_ival_dict 中的每个条目
    for (auto& entry : function_and_info_ival_dict) {
      // 提取键值对中的键和值
      at::IValue key = entry.key();
      at::IValue value = entry.value();
      // 确保键是字符串类型，否则抛出异常
      CAFFE_ENFORCE(
          key.isString(),
          "Expected extra information key to be a string ",
          "but got a ",
          value.tagKind(),
          " instead");
      // 确保值是列表类型，否则抛出异常
      CAFFE_ENFORCE(
          value.isList(),
          "Expected extra information values to be a list ",
          "but got a ",
          value.tagKind(),
          " instead");

      // 将值转换为 c10::List<at::IValue> 类型以便访问列表元素
      std::vector<std::string> data_list;
      c10::List<at::IValue> ival_data = value.toList();
      // 遍历列表中的每个元素，确保每个元素是字符串类型，然后加入到 data_list 中
      for (at::IValue data : ival_data) {
        CAFFE_ENFORCE(
            data.isString(),
            "Expected list element of nested dict entries to be a string ",
            "but got a ",
            data.tagKind(),
            " instead");
        data_list.push_back(data.toStringRef());
      }

      // 将键值对存入 function_and_info_dict，键转换为 std::string 类型
      function_and_info_dict[key.toStringRef()] = data_list;
    }

    // 从 function_and_info_dict 中获取 "get_inputs_function_name" 键对应的第一个值作为输入函数名称
    std::string input_function =
        function_and_info_dict["get_inputs_function_name"][0];
    // 将结果存入返回值映射中，键转换为 std::string 类型
    ret[function_name.toStringRef()] = input_function;
  }

  // 返回最终的结果映射 ret
  return ret;
}

std::vector<std::vector<at::IValue>> MobileModelRunner::
    get_all_bundled_inputs() {
  // 查找模块中是否存在名为 "get_all_bundled_inputs" 的方法
  auto has_bundled_input = module_->find_method("get_all_bundled_inputs");
  // 断言确保方法存在，否则输出错误信息并提示如何修复
  CAFFE_ENFORCE(
      has_bundled_input,
      "Model does not have bundled inputs. ",
      "Use torch.utils.bundled_inputs.augment_model_with_bundled_inputs to add.");

  // 运行模块的 "get_all_bundled_inputs" 方法，获取其返回值
  c10::IValue bundled_inputs = module_->run_method("get_all_bundled_inputs");
  // 将返回的 bundled_inputs 转换为标准输入格式并返回
  return ivalue_to_bundled_inputs(bundled_inputs);
}

std::unordered_map<std::string, std::vector<std::vector<at::IValue>>>
MobileModelRunner::get_many_functions_bundled_inputs() {
  // 查找模块中是否存在名为 "get_bundled_inputs_functions_and_info" 的方法
  auto has_bundled_input =
      module_->find_method("get_bundled_inputs_functions_and_info");
  // 断言确保方法存在，否则输出错误信息并提示如何修复
  CAFFE_ENFORCE(
      has_bundled_input,
      "Model does not have bundled inputs. ",
      "Use torch.utils.bundled_inputs.augment_many_model_functions_with_bundled_inputs to add.");

  // 运行模块的 "get_bundled_inputs_functions_and_info" 方法，获取其返回值
  auto ival_bundled_inputs_mapping =
      module_->run_method("get_bundled_inputs_functions_and_info");
  // 将返回的映射关系转换为标准输入格式并返回
  auto bundled_inputs_mapping =
      ivalue_to_bundled_inputs_map(ival_bundled_inputs_mapping);

  // 准备用于返回的 unordered_map
  std::unordered_map<std::string, std::vector<std::vector<at::IValue>>> ret;

  // 遍历映射关系，调用相应的方法并获取输入，存入返回的 unordered_map 中
  for (auto& entry : bundled_inputs_mapping) {
    std::string function_name = entry.first;
    std::string function_to_call = entry.second;

    // 检查模块中是否存在指定的函数名
    auto has_func_to_call = module_->find_method(function_to_call);
    // 断言确保函数存在，否则输出错误信息并提示如何修复
    CAFFE_ENFORCE(
        has_func_to_call,
        "Model does not have ",
        function_to_call,
        "Use torch.utils.bundled_inputs.augment_many_model_functions_with_bundled_inputs to add.");

    // 运行指定的函数，并获取其返回值
    c10::IValue bundled_inputs = module_->run_method(function_to_call);
    // 将返回的输入格式化后存入返回的 unordered_map 中
    ret[function_name] = ivalue_to_bundled_inputs(bundled_inputs);
  }
  return ret;
}

std::vector<at::IValue> MobileModelRunner::run_with_inputs(
    std::vector<std::vector<at::IValue>> const& bundled_inputs) {
  // 准备用于返回的 vector
  std::vector<at::IValue> ret;
  ret.reserve(bundled_inputs.size());

  // 遍历输入的 bundled_inputs，每个都调用模块的 forward 方法，并将结果存入返回的 vector 中
  for (std::vector<at::IValue> const& input : bundled_inputs) {
    ret.emplace_back(module_->forward(input));
  }
  return ret;
}

std::vector<at::IValue> MobileModelRunner::run_with_inputs(
    const std::string& function_name,
    std::vector<std::vector<at::IValue>> const& bundled_inputs) const {
  // 准备用于返回的 vector
  std::vector<at::IValue> ret;
  ret.reserve(bundled_inputs.size());

  // 检查模块中是否存在指定的函数名
  auto has_bundled_input = module_->find_method(function_name);
  // 断言确保函数存在，否则输出错误信息并提示如何修复
  CAFFE_ENFORCE(
      has_bundled_input,
      "Model does not have the method named ",
      function_name,
      "Please ensure that it was exported correctly");

  // 遍历输入的 bundled_inputs，每个都调用指定函数，并将结果存入返回的 vector 中
  for (std::vector<at::IValue> const& input : bundled_inputs) {
    auto func = module_->get_method(function_name);
    ret.emplace_back(func(input));
  }
  return ret;
}

void MobileModelRunner::run_argless_functions(
    const std::vector<std::string>& functions) {
  // 遍历输入的函数名列表，如果模块中存在对应函数，则运行该函数
  for (auto& function_name : functions) {
    if (module_->find_method(function_name)) {
      module_->run_method(function_name);
    }
  }
}

bool MobileModelRunner::set_has_metal_gpu_operators(
    // 对给定的操作列表进行迭代，每次迭代中将当前操作存储在变量 op 中
    for (std::string const& op : op_list) {
        // 检查当前操作是否以 "metal::" 开头，或者以 "metal_prepack::" 开头，
        // 或者以 "metal_prepack_unet::" 开头
        if (op.find("metal::") == 0 || op.find("metal_prepack::") == 0 ||
            op.find("metal_prepack_unet::") == 0) {
            // 如果是以上述任一条件开头，则返回 true
            return true;
        }
    }
    // 如果未找到任何匹配条件的操作，则返回 false
    return false;
}

void MobileModelRunner::for_each_tensor_in_bundled_inputs(
    std::function<void(const ::at::Tensor&)> const& func) {
  // 如果使用新样式的捆绑输入
  if (has_new_style_bundled_inputs()) {
    // 获取捆绑输入，并访问其中存储的参数级别的 IValue
    auto bundled_inputs_mapping = this->get_many_functions_bundled_inputs();

    // 循环遍历函数
    for (auto& entry : bundled_inputs_mapping) {
      std::vector<std::vector<at::IValue>> bundled_inputs = entry.second;
      // 遍历输入
      for (const std::vector<at::IValue>& input : bundled_inputs) {
        // 遍历输入中的值
        for (const at::IValue& iv : input) {
          // 对每个 IValue 中的张量执行给定的函数
          for_each_tensor_in_ivalue(iv, func);
        }
      }
    }
  } else {
    // 否则，从模块中获取所有捆绑输入的 IValue
    c10::IValue iv = module_->run_method("get_all_bundled_inputs");
    // 对每个 IValue 中的张量执行给定的函数
    for_each_tensor_in_ivalue(iv, func);
  }
}
} // namespace mobile
} // namespace jit
} // namespace torch
```