# `.\pytorch\test\cpp_api_parity\module_impl_check.py`

```py
// 定义函数，测试 Python `torch.nn` 模块和对应的 C++ `torch::nn` 模块之间的实现一致性。
// 具体步骤如下：
//
// 1. 从 common_nn.py 获取测试参数字典，在使用这些参数创建的 Python 模块上运行前向和反向传播。
//
// 2. 序列化 Python 模块的参数/缓冲区和其前向输入参数，然后在 C++ 中反序列化并加载到 C++ 模块中。
//
// 3. 在 C++ 模块上运行相同的前向和反向传播，并序列化 C++ 模块的前向输出和反向梯度。
//
// 4. 比较 Python/C++ 模块的前向输出和反向梯度。如果它们相同，则表示 Python/C++ 模块实现一致。

import os              // 导入操作系统模块
import pprint          // 导入用于美化打印的模块
import tempfile        // 导入临时文件处理模块
import types           // 导入类型模块
from string import Template  // 导入字符串模板类

import torch           // 导入 PyTorch 模块

from cpp_api_parity.sample_module import SAMPLE_MODULE_CPP_SOURCE  // 导入 C++ 示例模块
from cpp_api_parity.utils import (                                  // 导入多个实用函数
    add_test,
    compile_cpp_code_inline,
    compute_arg_dict,
    compute_cpp_args_construction_stmts_and_forward_arg_symbols,
    compute_temp_file_path,
    decorate_test_fn,
    generate_error_msg,
    is_torch_nn_functional_test,
    move_python_tensors_to_device,
    serialize_arg_dict_as_script_module,
    set_python_tensors_requires_grad,
    TORCH_NN_COMMON_TEST_HARNESS,   // 导入 PyTorch `torch.nn` 公共测试的标志符
    TorchNNModuleTestParams,        // 导入 PyTorch `torch.nn` 模块的测试参数类
    try_remove_folder,              // 导入尝试移除文件夹的函数
)

// 预期的字符串替换：
//
// ${module_variant_name}  （例如 `Linear_no_bias_cpu`）
// ${module_qualified_name}  （例如 `torch::nn::Linear`）
// ${cpp_args_construction_stmts}
// ${cpp_constructor_args}
// ${device}
// ${cpp_forward_args_symbols}
TORCH_NN_MODULE_TEST_FORWARD_BACKWARD = Template(
    """
void ${module_variant_name}_test_forward_backward(
    const std::string& arg_dict_file_path,
    const std::string& module_file_path,
    const std::string& forward_output_file_path,
    const std::string& backward_grad_dict_file_path) {
  pybind11::gil_scoped_release no_gil;  // 释放全局解释器锁

  // 声明参数
  auto arg_dict = load_dict_from_file(arg_dict_file_path);  // 从文件加载参数字典
  ${cpp_args_construction_stmts};  // 构造 C++ 参数语句

  // 构建模块并从 Python 模块加载参数/缓冲区
  ${module_qualified_name} module${cpp_constructor_args};  // 构造 C++ 模块
  module->to(std::string("${device}"));  // 将模块移到指定设备
  torch::load(module, module_file_path);  // 加载模块

  // 某些模块（如 `RReLU`）在其前向传播中创建随机张量。
  // 为了确保在 Python/C++ 中创建的随机张量相同，需要手动设置随机数种子。
  torch::manual_seed(0);  // 手动设置随机数种子为 0

  // 前向传播
  auto cpp_output = module(${cpp_forward_args_symbols});  // 执行模块的前向传播

  // 将输出保存到文件以便之后在 Python 中进行比较
  write_ivalue_to_file(torch::IValue(cpp_output), forward_output_file_path);  // 将输出写入文件

  // 反向传播
  if (cpp_output.is_complex()) {  // 如果输出为复数类型
    cpp_output.sum().abs().backward();  // 对其进行求和、绝对值并执行反向传播
  } else {
    cpp_output.sum().backward();
  }

  // Put all gradients into a c10::Dict, save it into a file to be compared in Python later
  // 创建一个 c10::Dict 对象 grad_dict，用于存储所有参数的梯度，并将其保存到文件中，以便稍后在 Python 中进行比较
  c10::Dict<std::string, torch::Tensor> grad_dict;
  
  // 遍历模型中的每个参数，获取其梯度并存入 grad_dict
  for (const auto& param : module->named_parameters()) {
    // 获取参数的梯度
    torch::Tensor grad = param.value().grad();
    
    // 如果梯度是稀疏张量，将其索引和值分别存入 grad_dict
    if (grad.is_sparse()) {
      grad_dict.insert(param.key() + "_grad_indices", grad.coalesce().indices());
      grad_dict.insert(param.key() + "_grad_values", grad.coalesce().values());
    } else {
      // 否则直接将梯度张量存入 grad_dict
      grad_dict.insert(param.key() + "_grad", grad);
    }
  }

  // 将 grad_dict 转换为 torch::IValue，并将其写入到指定路径的文件中
  write_ivalue_to_file(torch::IValue(grad_dict), backward_grad_dict_file_path);
    )
    # 运行 Python 模块的前向和反向传播
    script_module, python_output, python_grad_dict = run_python_forward_backward(
        unit_test_class, test_params
    )

    # 保存 Python 模块和参数，以便从 C++ 函数中使用
    module_file_path = compute_temp_file_path(
        cpp_tmp_folder, module_variant_name, "module"
    )
    arg_dict_file_path = compute_temp_file_path(
        cpp_tmp_folder, module_variant_name, "arg_dict"
    )
    script_module.save(module_file_path)  # 将 Python 模块保存到临时文件
    serialize_arg_dict_as_script_module(test_params.arg_dict).save(arg_dict_file_path)  # 将参数字典序列化并保存到临时文件

    # 组合 C++ 测试函数的名称
    cpp_test_name = f"{test_params.module_variant_name}_test_forward_backward"
    # 获取并调用对应的 C++ 测试函数
    cpp_test_fn = getattr(unit_test_class.module_impl_check_cpp_module, cpp_test_name)
    def run_cpp_test_fn_and_check_output():
        # 计算前向输出文件的临时文件路径
        forward_output_file_path = compute_temp_file_path(
            cpp_tmp_folder, module_variant_name, "forward_output"
        )
        # 计算后向梯度字典文件的临时文件路径
        backward_grad_dict_file_path = compute_temp_file_path(
            cpp_tmp_folder, module_variant_name, "backward_grad_dict"
        )

        # 调用 C++ 测试函数，生成前向输出和后向梯度字典文件
        cpp_test_fn(
            arg_dict_file_path,
            module_file_path,
            forward_output_file_path,
            backward_grad_dict_file_path,
        )
        # 加载前向输出的数据
        cpp_output = torch.load(forward_output_file_path)
        # 加载后向梯度字典的数据
        cpp_grad_dict = torch.load(backward_grad_dict_file_path)

        # 检查前向输出是否相等
        unit_test_class.assertEqual(
            python_output,
            cpp_output,
            msg=generate_error_msg("forward output", cpp_output, python_output),
        )

        # 检查在反向传播后模块参数梯度是否相等
        unit_test_class.assertEqual(
            len(python_grad_dict),
            len(cpp_grad_dict),
            msg=generate_error_msg(
                "# of parameters", len(cpp_grad_dict), len(python_grad_dict)
            ),
        )
        # 遍历 Python 中的梯度字典的键
        for key in python_grad_dict:
            param_name = None
            # 根据后缀确定参数名称
            for suffix in ["_grad", "_grad_indices", "_grad_values"]:
                if key.endswith(suffix):
                    param_name = key[: -len(suffix)]
                    break
            assert param_name is not None
            # 根据后缀确定稀疏性
            sparsity_str = (
                "sparse" if key.endswith(("_grad_indices", "_grad_values")) else "dense"
            )

            # 断言 C++ 梯度字典中是否存在相应的键
            unit_test_class.assertTrue(
                key in cpp_grad_dict,
                msg=generate_error_msg(
                    f'"Does module have a parameter named `{param_name}` with {sparsity_str} gradient?"',
                    False,
                    True,
                ),
            )
            # 检查两个梯度是否相等
            unit_test_class.assertEqual(
                python_grad_dict[key],
                cpp_grad_dict[key],
                msg=generate_error_msg(
                    f"`{param_name}`'s {sparsity_str} gradient (`{key}`)",
                    cpp_grad_dict[key],
                    python_grad_dict[key],
                ),
            )

    # 运行 C++ 测试函数并检查输出
    run_cpp_test_fn_and_check_output()

    # 删除存储 C++ 输出的临时文件夹
    try_remove_folder(cpp_tmp_folder)
# 计算模块名称的函数，根据给定的测试参数字典确定模块名称
def compute_module_name(test_params_dict):
    # 从测试参数字典中获取完整名称
    fullname = test_params_dict.get("fullname", None)
    if fullname:
        # 如果完整名称存在，则按下划线分割并取第一部分作为模块名称
        module_name = fullname.split("_")[0]
    else:
        # 否则，尝试从测试参数字典中直接获取模块名称
        module_name = test_params_dict.get("module_name")
    return module_name


# 处理模块的测试参数函数，根据给定的测试参数字典、设备和测试实例类
def process_test_params_for_module(test_params_dict, device, test_instance_class):
    # 计算模块名称
    module_name = compute_module_name(test_params_dict)
    
    # 在测试参数字典中添加或更新 "constructor" 键，用于实例化模块的构造函数
    test_params_dict["constructor"] = test_params_dict.get(
        "constructor", getattr(torch.nn, module_name)
    )
    
    # 使用给定的测试实例类和测试参数字典创建测试实例
    test_instance = test_instance_class(**test_params_dict)
    
    # 断言测试实例的名称以 "test_" 开头
    assert test_instance.get_name().startswith("test_")
    
    # 根据测试实例的名称生成模块变体名称，例如 `BCELoss_weights_cuda`
    module_variant_name = test_instance.get_name()[5:] + (
        ("_" + device) if device != "cpu" else ""
    )

    # 如果测试参数字典中包含 "constructor_args"，则确保同时包含 "cpp_constructor_args"
    if "constructor_args" in test_params_dict:
        assert "cpp_constructor_args" in test_params_dict, (
            "如果测试参数字典中包含 `constructor_args`，则需要在其中包含 `cpp_constructor_args`，"
            "用于启用 C++ API 的一致性测试。请参见以下测试参数字典的详细内容：\n"
            f"{pprint.pformat(test_params_dict)}。如果您希望添加 C++ API 的一致性测试，请添加 "
            "`test_cpp_api_parity=False` 到测试参数字典，并提交此问题。"
        )

    # 返回 TorchNNModuleTestParams 实例，用于封装处理后的测试参数
    return TorchNNModuleTestParams(
        module_name=module_name,
        module_variant_name=module_variant_name,
        test_instance=test_instance,
        cpp_constructor_args=test_params_dict.get("cpp_constructor_args", ""),
        arg_dict=compute_arg_dict(test_params_dict, test_instance),
        has_parity=test_params_dict.get("has_parity", True),
        device=device,
        cpp_tmp_folder=tempfile.mkdtemp(),
    )


# 将测试写入测试类的函数，根据给定的单元测试类、测试参数字典、测试实例类、一致性表和设备列表
def write_test_to_test_class(
    unit_test_class, test_params_dict, test_instance_class, parity_table, devices
):
    # 断言测试参数字典不是 Torch 的 NN 功能性测试
    assert not is_torch_nn_functional_test(test_params_dict)

    # 计算模块名称
    module_name = compute_module_name(test_params_dict)

    # 断言 torch.nn 模块中存在模块名称对应的属性
    assert hasattr(torch.nn, module_name), (
        f"`torch.nn` 模块中不存在 `{module_name}` 模块。如果您正在添加新的测试，请确保在模块测试字典中使用 "
        "格式 `ModuleName_desc` 设置 `fullname` 或使用格式 `ModuleName` 设置 `module_name`：\n"
        f"{pprint.pformat(test_params_dict)}"
    )

    # 构建完整的模块名称，用于 C++ API 一致性测试
    module_full_name = "torch::nn::" + module_name

    # 断言完整的模块名称在一致性表的 torch::nn 部分中存在
    assert module_full_name in parity_table["torch::nn"], (
        f"请将 `{module_full_name}` 条目添加到 `test/cpp_api_parity/parity-tracker.md` 的 `torch::nn` 部分。"
        f"（在处理以下测试参数字典时发现：\n{pprint.pformat(test_params_dict)}。）"
    )
    # 对每个设备进行迭代，设备来自于 devices 列表
    for device in devices:
        # 根据设备处理测试参数，生成特定设备的测试参数
        test_params = process_test_params_for_module(
            test_params_dict=test_params_dict,
            device=device,
            test_instance_class=test_instance_class,
        )
        # 尝试删除指定路径的文件夹
        try_remove_folder(test_params.cpp_tmp_folder)
        # 构建单元测试名称，格式为 "test_torch_nn_" 后接模块变体名称
        unit_test_name = f"test_torch_nn_{test_params.module_variant_name}"
        # 将当前单元测试名称与对应的测试参数映射存入 unit_test_class 的属性中
        unit_test_class.module_test_params_map[unit_test_name] = test_params

        # 定义一个测试函数 test_fn，用于执行前向和后向测试
        def test_fn(self):
            test_forward_backward(
                unit_test_class=self,
                test_params=unit_test_class.module_test_params_map[
                    self._testMethodName
                ],
            )

        # 对测试函数进行装饰，设置测试是否使用 CUDA，是否具有实现的一致性，并指定设备
        test_fn = decorate_test_fn(
            test_fn=test_fn,
            test_cuda=test_params_dict.get("test_cuda", True),
            has_impl_parity=parity_table["torch::nn"][module_full_name][0]
            and test_params_dict.get("has_parity", True),
            device=device,
        )

        # 将装饰后的测试函数添加到单元测试类 unit_test_class 中
        add_test(unit_test_class, unit_test_name, test_fn)
# 根据测试参数和模板生成测试用的 C++ 源代码
def generate_test_cpp_sources(test_params, template):
    # 获取测试参数中的设备信息
    device = test_params.device

    # 获取 C++ 构造函数参数，如果为空则设为""
    cpp_constructor_args = test_params.cpp_constructor_args
    if cpp_constructor_args != "":
        cpp_constructor_args = f"({cpp_constructor_args})"

    # 计算 C++ 参数构造语句和前向参数符号
    (
        cpp_args_construction_stmts,
        cpp_forward_args_symbols,
    ) = compute_cpp_args_construction_stmts_and_forward_arg_symbols(test_params)

    # 使用模板替换生成 C++ 测试源代码
    test_cpp_sources = template.substitute(
        module_variant_name=test_params.module_variant_name,
        module_qualified_name=f"torch::nn::{test_params.module_name}",
        cpp_args_construction_stmts=";\n  ".join(cpp_args_construction_stmts),
        cpp_constructor_args=cpp_constructor_args,
        cpp_forward_args_symbols=", ".join(cpp_forward_args_symbols),
        device=device,
    )
    return test_cpp_sources


# 一次性编译所有的 C++ 测试，而不是每个测试都编译一次
def build_cpp_tests(unit_test_class, print_cpp_source=False):
    # 断言模块测试参数映射不为空
    assert len(unit_test_class.module_test_params_map) > 0

    # 初始化 C++ 源代码，包括公共测试工具和示例模块的源代码
    cpp_sources = TORCH_NN_COMMON_TEST_HARNESS + SAMPLE_MODULE_CPP_SOURCE
    functions = []

    # 遍历所有的模块测试参数，生成对应的 C++ 测试源代码并添加到总源代码中
    for test_params in unit_test_class.module_test_params_map.values():
        cpp_sources += generate_test_cpp_sources(
            test_params=test_params, template=TORCH_NN_MODULE_TEST_FORWARD_BACKWARD
        )
        # 添加测试函数名到函数列表中
        functions.append(f"{test_params.module_variant_name}_test_forward_backward")

    # 如果需要打印 C++ 源代码，则输出
    if print_cpp_source:
        print(cpp_sources)

    # 编译内联的 C++ 代码模块
    cpp_module = compile_cpp_code_inline(
        name="module_impl_check", cpp_sources=cpp_sources, functions=functions
    )

    # 将编译后的 C++ 模块赋值给单元测试类的相应属性
    unit_test_class.module_impl_check_cpp_module = cpp_module
```