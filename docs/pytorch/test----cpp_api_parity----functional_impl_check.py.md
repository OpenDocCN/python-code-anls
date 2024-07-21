# `.\pytorch\test\cpp_api_parity\functional_impl_check.py`

```
# 本测试的目的是检查 Python 的 `torch.nn.functional` 函数与对应的 C++ `torch::nn::functional`
# 函数之间的实现是否一致。具体而言，这个测试执行以下操作：
#
# 1. 从 common_nn.py 获取测试参数字典，对使用测试参数创建的 Python functional 执行前向传播。
#
# 2. 序列化 Python functional 前向传播的输入参数，将其在 C++ 中反序列化，并作为 C++ functional 前向传播的输入。
#
# 3. 在 C++ functional 上执行前向传播，并将 C++ functional 的前向输出序列化。
#
# 4. 比较 Python/C++ functional 的前向输出。如果它们相同，则 Python/C++ 模块实现是一致的。

import os
import pprint
import re
import tempfile
from string import Template

import torch

# 导入所需的 C++ API 相关模块和函数
from cpp_api_parity.sample_functional import SAMPLE_FUNCTIONAL_CPP_SOURCE
from cpp_api_parity.utils import (
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
    TORCH_NN_COMMON_TEST_HARNESS,
    TorchNNFunctionalTestParams,
    try_remove_folder,
)

# 预期的字符串替换模板：
#
# ${functional_variant_name}  (例如 `BCELoss_no_reduce`)
# ${cpp_args_construction_stmts}
# ${cpp_function_call}
TORCH_NN_FUNCTIONAL_TEST_FORWARD = Template(
    """
void ${functional_variant_name}_test_forward(
    const std::string& arg_dict_file_path,
    const std::string& forward_output_file_path) {
  pybind11::gil_scoped_release no_gil;

  namespace F = torch::nn::functional;

  // 声明参数
  auto arg_dict = load_dict_from_file(arg_dict_file_path);
  ${cpp_args_construction_stmts};

  // 一些 functionals（比如 `F::rrelu`）在其调用路径中创建随机张量。
  // 为了确保在 Python 和 C++ 中创建的随机张量是相同的，需要手动设置 RNG 种子。
  torch::manual_seed(0);

  // 运行带有参数的函数
  auto cpp_output = ${cpp_function_call};

  // 将输出保存到文件，以便稍后在 Python 中进行比较
  write_ivalue_to_file(torch::IValue(cpp_output), forward_output_file_path);
}
"""
)

def run_forward(unit_test_class, test_params):
    device = test_params.device

    # 设置需要梯度的 Python 张量并将其移到指定设备上
    inputs = set_python_tensors_requires_grad(
        move_python_tensors_to_device(
            [arg_value for _, arg_value in test_params.arg_dict["input"]], device
        )
    )
    # 将目标张量移到指定设备上
    inputs += move_python_tensors_to_device(
        [arg_value for _, arg_value in test_params.arg_dict["target"]], device
    )
    # 将额外的参数张量移到指定设备上
    inputs += move_python_tensors_to_device(
        [arg_value for _, arg_value in test_params.arg_dict["extra_args"]], device
    )

    # 一些 functionals（例如 `F.rrelu`）在其调用路径中创建随机张量。
    # 设置随机数生成器的种子为0，以确保在 Python 和 C++ 中创建的随机张量是相同的。
    torch.manual_seed(0)
    # 调用测试实例的构造函数，并使用给定的输入参数进行调用，得到 Python 端的输出结果。
    python_output = test_params.test_instance.constructor()(*inputs)
    
    # 返回 Python 端的输出结果作为函数的返回值。
    return python_output
# 测试向前传播函数的功能
def test_forward(unit_test_class, test_params):
    # 从测试参数中获取功能变体的名称
    functional_variant_name = test_params.functional_variant_name
    # 获取用于存储 C++ 临时文件的文件夹路径
    cpp_tmp_folder = test_params.cpp_tmp_folder
    
    # 如果临时文件夹已存在，则尝试删除
    try_remove_folder(cpp_tmp_folder)
    # 创建新的临时文件夹
    os.mkdir(cpp_tmp_folder)
    
    # 在 Python 函数实现上运行向前传播
    python_output = run_forward(unit_test_class, test_params)
    
    # 将 Python 参数序列化为脚本模块，并保存到文件中，以供 C++ 函数使用
    arg_dict_file_path = compute_temp_file_path(
        cpp_tmp_folder, functional_variant_name, "arg_dict"
    )
    serialize_arg_dict_as_script_module(test_params.arg_dict).save(arg_dict_file_path)
    
    # 构建 C++ 测试函数名
    cpp_test_name = f"{test_params.functional_variant_name}_test_forward"
    # 获取对应的 C++ 测试函数
    cpp_test_fn = getattr(
        unit_test_class.functional_impl_check_cpp_module, cpp_test_name
    )
    
    # 定义函数：运行 C++ 测试函数并检查输出
    def run_cpp_test_fn_and_check_output():
        # 计算存储 C++ 输出的临时文件路径
        forward_output_file_path = compute_temp_file_path(
            cpp_tmp_folder, functional_variant_name, "forward_output"
        )
        
        # 调用 C++ 测试函数进行向前传播，并将输出保存到文件
        cpp_test_fn(arg_dict_file_path, forward_output_file_path)
        # 从文件加载 C++ 输出
        cpp_output = torch.load(forward_output_file_path)
        
        # 检查 Python 和 C++ 的向前传播输出是否相等
        unit_test_class.assertEqual(
            python_output,
            cpp_output,
            msg=generate_error_msg("forward output", cpp_output, python_output),
        )
    
    # 运行 C++ 测试函数并检查输出
    run_cpp_test_fn_and_check_output()
    
    # 删除存储 C++ 输出的临时文件夹
    try_remove_folder(cpp_tmp_folder)


# 根据测试参数字典计算功能名称
def compute_functional_name(test_params_dict):
    # 将驼峰命名转换为蛇形命名的辅助函数
    def camel_case_to_snake_case(camel_case_str):
        return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_str).lower()

    if "cpp_options_args" in test_params_dict:
        # 预期 `cpp_options_args` 的格式为 `F::FunctionalFuncOptions(...)`
        # 示例输出为 `binary_cross_entropy`
        return camel_case_to_snake_case(
            test_params_dict["cpp_options_args"]
            .split("(")[0]
            .replace("F::", "")
            .replace("FuncOptions", "")
        )
    elif "cpp_function_call" in test_params_dict:
        # 预期 `cpp_function_call` 的格式为 `F::functional_name(...)`
        # 示例输出为 `binary_cross_entropy`
        return test_params_dict["cpp_function_call"].split("(")[0].replace("F::", "")
    else:
        # 如果测试参数字典中既没有 `cpp_options_args` 也没有 `cpp_function_call`，则抛出运行时错误
        raise RuntimeError(
            f"`cpp_options_args` or `cpp_function_call` entry must be present in test params dict:\n{pprint.pformat(test_params_dict)}"  # noqa: B950
        )


# 根据测试参数字典、参数字典和功能名称计算 C++ 函数调用
def compute_cpp_function_call(test_params_dict, arg_dict, functional_name):
    if "cpp_function_call" in test_params_dict:
        return test_params_dict["cpp_function_call"]
    elif "cpp_options_args" in test_params_dict:
        # 从 `test_params_dict` 中检查是否存在键 `cpp_options_args`
        cpp_forward_args_symbols = [
            arg_name
            for arg_name, _ in arg_dict["input"]
            + arg_dict["target"]
            + arg_dict["extra_args"]
        ]
        # 构建用于 C++ 函数调用的参数符号列表，包括 `input`、`target` 和 `extra_args` 中的参数名
        return "F::{}({}, {})".format(
            functional_name,
            ", ".join(cpp_forward_args_symbols),
            test_params_dict["cpp_options_args"],
        )
    else:
        # 如果 `cpp_options_args` 在 `test_params_dict` 中不存在，则抛出运行时错误
        raise RuntimeError(
            f"`cpp_options_args` or `cpp_function_call` entry must be present in test params dict:\n{pprint.pformat(test_params_dict)}"  # noqa: B950
        )
# 处理功能测试参数以用于功能测试
def write_test_to_test_class(
    # 将功能测试写入测试类中，传入的参数包括测试参数字典、测试实例类、平衡表和设备列表
    unit_test_class, test_params_dict, test_instance_class, parity_table, devices
):
    # 断言传入的测试参数字典确实是 Torch NN 功能测试
    assert is_torch_nn_functional_test(test_params_dict)

    # 断言在测试参数字典中要么有 "cpp_options_args"，要么有 "cpp_function_call"
    assert (
        "cpp_options_args" in test_params_dict
        or "cpp_function_call" in test_params_dict
    ), (
        "为了启用 C++ API 平衡测试，测试参数字典中必须包含 `cpp_options_args` 或 `cpp_function_call`。\n"
        f"当前测试参数字典内容为:\n{pprint.pformat(test_params_dict)}。 \n"
        "如果想要添加 C++ API 平衡测试，请参阅:\n"
        "注意 [如何检查 Python 和 C++ 前端之间的 NN 模块 / 功能 API 平衡]。 \n"
        "如果不需要，请在测试参数字典中添加 `test_cpp_api_parity=False` 并提交此问题。"
    )

    # 断言不应同时在测试参数字典中包含 "cpp_options_args" 和 "cpp_function_call"
    assert not (
        "cpp_options_args" in test_params_dict
        and "cpp_function_call" in test_params_dict
    ), (
        "测试参数字典中应该只有 `cpp_options_args` 或 `cpp_function_call` 之一，"
        f"不应同时包含两者:\n{pprint.pformat(test_params_dict)}"
    )

    # 计算功能名称
    functional_name = compute_functional_name(test_params_dict)

    # 断言 torch.nn.functional 中存在指定的功能函数名
    assert hasattr(
        torch.nn.functional, functional_name
    ), f"`torch.nn.functional` 中不存在函数 `{functional_name}`。 (发现于处理\n{pprint.pformat(test_params_dict)}。)"  # noqa: B950

    # 构造完整的功能函数名，格式为 "F::" + functional_name
    functional_full_name = "F::" + functional_name

    # 断言功能函数的完整名称在平衡表的 "torch::nn::functional" 部分中存在
    assert functional_full_name in parity_table["torch::nn::functional"], (
        f"请在 `test/cpp_api_parity/parity-tracker.md` 的 `torch::nn::functional` 部分中添加 `{functional_full_name}` 条目。 "
        f"(发现于处理\n{pprint.pformat(test_params_dict)}。)"
    )
    # 遍历设备列表，对每个设备执行以下操作
    for device in devices:
        # 根据设备处理测试参数，生成适用于功能测试的测试参数
        test_params = process_test_params_for_functional(
            test_params_dict=test_params_dict,
            device=device,
            test_instance_class=test_instance_class,
        )
        
        # 尝试删除指定的临时文件夹（如果存在）
        try_remove_folder(test_params.cpp_tmp_folder)
        
        # 构建单元测试名称，格式为特定的测试函数命名规则
        unit_test_name = (
            f"test_torch_nn_functional_{test_params.functional_variant_name}"
        )
        
        # 将当前功能测试的测试参数映射到单元测试类的映射表中
        unit_test_class.functional_test_params_map[unit_test_name] = test_params

        # 定义测试函数
        def test_fn(self):
            # 调用测试前向函数，传入单元测试类和当前测试方法的测试参数
            test_forward(
                unit_test_class=self,
                test_params=unit_test_class.functional_test_params_map[
                    self._testMethodName
                ],
            )

        # 使用装饰器装饰测试函数，设置是否使用 CUDA 进行测试和实现是否具有一致性
        test_fn = decorate_test_fn(
            test_fn=test_fn,
            test_cuda=test_params_dict.get("test_cuda", True),
            has_impl_parity=parity_table["torch::nn::functional"][functional_full_name][
                0
            ]
            and test_params_dict.get("has_parity", True),
            device=device,
        )

        # 将装饰后的测试函数添加到单元测试类中
        add_test(unit_test_class, unit_test_name, test_fn)
# 使用给定的测试参数和模板生成测试用的 C++ 源代码
def generate_test_cpp_sources(test_params, template):
    # 调用函数计算 C++ 参数构造语句和前向参数符号
    (
        cpp_args_construction_stmts,
        _,  # 解构返回结果，只取第一个值
    ) = compute_cpp_args_construction_stmts_and_forward_arg_symbols(test_params)

    # 使用模板替换功能变体名称、C++ 参数构造语句和 C++ 函数调用，并生成测试用的 C++ 源代码
    test_cpp_sources = template.substitute(
        functional_variant_name=test_params.functional_variant_name,
        cpp_args_construction_stmts=";\n  ".join(cpp_args_construction_stmts),  # 将参数构造语句连接成字符串
        cpp_function_call=test_params.cpp_function_call,
    )
    return test_cpp_sources


# 构建所有的 C++ 测试代码，而不是每个测试单独构建
def build_cpp_tests(unit_test_class, print_cpp_source=False):
    # 断言功能测试参数映射不为空
    assert len(unit_test_class.functional_test_params_map) > 0

    # 初始化 C++ 源代码为通用测试框架和样本功能 C++ 源代码
    cpp_sources = TORCH_NN_COMMON_TEST_HARNESS + SAMPLE_FUNCTIONAL_CPP_SOURCE
    functions = []

    # 遍历功能测试参数映射中的每一个测试参数
    for test_params in unit_test_class.functional_test_params_map.values():
        # 生成该测试参数对应的测试用 C++ 源代码，并添加到总的 C++ 源代码中
        cpp_sources += generate_test_cpp_sources(
            test_params=test_params, template=TORCH_NN_FUNCTIONAL_TEST_FORWARD
        )
        # 添加该测试函数的名称到函数列表中
        functions.append(f"{test_params.functional_variant_name}_test_forward")

    # 如果指定打印 C++ 源代码，则输出
    if print_cpp_source:
        print(cpp_sources)

    # 编译内联的 C++ 代码模块，命名为 functional_impl_check，包括指定的函数列表
    cpp_module = compile_cpp_code_inline(
        name="functional_impl_check", cpp_sources=cpp_sources, functions=functions
    )

    # 将编译后的 C++ 模块赋值给测试类的功能实现检查 C++ 模块属性
    unit_test_class.functional_impl_check_cpp_module = cpp_module
```