# `.\pytorch\test\cpp_api_parity\utils.py`

```
# 导入标准库 os，用于操作操作系统相关功能
import os
# 导入 shutil 库，提供高级文件操作功能
import shutil
# 导入 unittest 模块，用于编写和运行单元测试
import unittest
# 导入 warnings 模块，用于管理警告信息的显示
import warnings
# 导入 collections 模块中的 namedtuple，用于创建命名元组
from collections import namedtuple

# 导入 torch 库，用于深度学习相关功能
import torch
# 导入 torch.testing._internal.common_nn 模块，用于通用神经网络测试功能
import torch.testing._internal.common_nn as common_nn
# 导入 torch.utils.cpp_extension 模块，用于处理 C++ 扩展
import torch.utils.cpp_extension
# 从 torch.testing._internal.common_cuda 导入 TEST_CUDA 常量，用于 CUDA 相关测试

# 下面是一个命名元组，用于存储 Torch 神经网络模块测试的参数
# 这些参数包括模块名、变体名、测试实例、C++ 构造函数参数、参数字典、是否期望 Python/C++ 一致性、设备名和临时文件夹路径
TorchNNModuleTestParams = namedtuple(
    "TorchNNModuleTestParams",
    [
        # 神经网络模块名（例如 "BCELoss"）
        "module_name",
        # 此模块配置的唯一标识符（例如 "BCELoss_weights_cuda"）
        "module_variant_name",
        # NN 测试类的实例（例如 `CriterionTest`），用于存储运行 Python 测试所需的信息
        "test_instance",
        # 传递给 C++ 模块构造函数的参数，必须严格等同于 Python 模块构造函数的参数
        # （例如 `torch::nn::BCELossOptions().weight(torch::rand(10))`，
        # 这与将 `torch.rand(10)` 传递给 `torch.nn.BCELoss` 构造函数在 Python 中完全等效）
        "cpp_constructor_args",
        # 用于 NN 模块的前向传播的所有参数
        # 详细信息请参阅 `compute_arg_dict` 函数，了解如何构建此字典
        # （例如
        # ```
        # arg_dict = {
        #     'input': [python_input_tensor],
        #     'target': [python_target_tensor],
        #     'extra_args': [],
        #     'other': [],
        # }
        # ```
        # ）
        "arg_dict",
        # 是否期望此 NN 模块测试通过 Python/C++ 一致性测试
        # （例如 `True`）
        "has_parity",
        # 设备（例如 "cuda"）
        "device",
        # 用于存储 C++ 输出的临时文件夹路径（稍后与 Python 输出进行比较）
        "cpp_tmp_folder",
    ],
)

# 下面是另一个命名元组，用于存储 Torch 神经网络功能测试的参数
# 此命名元组类似于上面的模块测试参数，但是用于函数接口的测试
TorchNNFunctionalTestParams = namedtuple(
    "TorchNNFunctionalTestParams",
    [
        # 功能函数名（例如 "BCELoss"）
        "functional_name",
        # 此功能配置的唯一标识符（例如 "BCELoss_weights_cuda"）
        "functional_variant_name",
        # 功能测试类的实例，用于存储运行 Python 测试所需的信息
        "test_instance",
        # 传递给 C++ 函数构造函数的参数，必须严格等同于 Python 函数构造函数的参数
        # （例如 `torch::nn::functional::BCELossOptions().weight(torch::rand(10))`，
        # 这与将 `torch.rand(10)` 传递给 `torch.nn.functional.BCELoss` 函数构造函数在 Python 中完全等效）
        "cpp_constructor_args",
        # 用于函数的所有参数
        # 详细信息请参阅 `compute_arg_dict` 函数，了解如何构建此字典
        # （例如同上述 TorchNNModuleTestParams 的 arg_dict 描述）
        "arg_dict",
        # 是否期望此功能函数测试通过 Python/C++ 一致性测试
        # （例如 `True`）
        "has_parity",
        # 设备（例如 "cuda"）
        "device",
        # 用于存储 C++ 输出的临时文件夹路径（稍后与 Python 输出进行比较）
        "cpp_tmp_folder",
    ],
)
    [
        # NN functional name (e.g. "binary_cross_entropy")
        "functional_name",
        # Unique identifier for this functional config (e.g. "BCELoss_no_reduce_cuda")
        "functional_variant_name",
        # An instance of an NN test class (e.g. `NewModuleTest`) which stores
        # necessary information (e.g. input / target / extra_args) for running the Python test
        "test_instance",
        # The C++ function call that is strictly equivalent to the Python function call
        # (e.g. "F::binary_cross_entropy(
        #            i, t.to(i.options()),F::BinaryCrossEntropyFuncOptions().reduction(torch::kNone))",
        # which is strictly equivalent to `F.binary_cross_entropy(i, t.type_as(i), reduction='none')` in Python)
        "cpp_function_call",
        # All arguments used in NN functional's function call.
        # Please see `compute_arg_dict` function for details on how we construct this dict.
        # (e.g.
        # ```
        # arg_dict = {
        #     'input': [python_input_tensor],
        #     'target': [python_target_tensor],
        #     'extra_args': [],
        #     'other': [],
        # }
        # ```
        # )
        "arg_dict",
        # Whether we expect this NN functional test to pass the Python/C++ parity test
        # (e.g. `True`)
        "has_parity",
        # Device (e.g. "cuda")
        "device",
        # Temporary folder to store C++ outputs (to be compared with Python outputs later)
        "cpp_tmp_folder",
    ],
// 使用 namedtuple 定义一个 CppArg 结构，包含字段 name 和 value
CppArg = namedtuple("CppArg", ["name", "value"])

// 定义了一个包含 C++ 代码的字符串，用于加载和执行 Torch 脚本
TORCH_NN_COMMON_TEST_HARNESS = """
#include <torch/script.h>

// 将 torch::IValue 对象写入到文件中
void write_ivalue_to_file(const torch::IValue& ivalue, const std::string& file_path) {
    // 使用 torch::jit::pickle_save 将 ivalue 序列化为字节流
    auto bytes = torch::jit::pickle_save(ivalue);
    // 打开文件流，写入字节流数据到指定文件
    std::ofstream fout(file_path, std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();
}

// 从文件中加载字典数据，包含字符串键和 Torch 张量值
c10::Dict<std::string, torch::Tensor> load_dict_from_file(const std::string& file_path) {
    // 创建空的 c10::Dict 对象，用于存储加载的数据
    c10::Dict<std::string, torch::Tensor> arg_dict;
    // 使用 torch::jit::load 加载 Torch 模型或脚本
    auto arg_dict_module = torch::jit::load(file_path);
    // 遍历加载的模型中的 named_buffers，将其插入到 arg_dict 中
    for (const auto& p : arg_dict_module.named_buffers(/*recurse=*/false)) {
        arg_dict.insert(p.name, p.value);
    }
    return arg_dict;
}

// 生成具有不相等值的随机张量，以确保对像 MaxPooling 这样的模块的测试不会因重复值而失败
torch::Tensor _rand_tensor_non_equal(torch::IntArrayRef size) {
    // 计算张量的总元素数
    int64_t total = 1;
    for (int64_t elem : size) {
        total *= elem;
    }
    // 使用 torch::randperm 生成随机排列的整数，视图为指定尺寸的张量，并转换为双精度
    return torch::randperm(total).view(size).to(torch::kDouble);
}
"""

// 编译内联的 C++ 代码，并返回编译后的模块对象
def compile_cpp_code_inline(name, cpp_sources, functions):
    cpp_module = torch.utils.cpp_extension.load_inline(
        name=name,
        cpp_sources=cpp_sources,
        extra_cflags=[
            "-g"
        ],  // 默认启用调试符号，用于调试测试失败
        functions=functions,
        verbose=False,
    )
    return cpp_module

// 根据给定的临时文件夹路径、变体名称和文件后缀，生成临时文件路径
def compute_temp_file_path(cpp_tmp_folder, variant_name, file_suffix):
    return os.path.join(cpp_tmp_folder, f"{variant_name}_{file_suffix}.pt")

// 检查测试参数字典是否包含 "wrap_functional"，以判断是否是 Torch 函数式测试
def is_torch_nn_functional_test(test_params_dict):
    return "wrap_functional" in str(test_params_dict.get("constructor", ""))

// 将 Python 输入转换为列表，如果是 Torch 张量则转换为单元素列表
def convert_to_list(python_input):
    if isinstance(python_input, torch.Tensor):
        return [python_input]
    else:
        return list(python_input)

// 设置 Python 张量为需要梯度，除非是 torch.long 类型的张量
def set_python_tensors_requires_grad(python_tensors):
    return [
        tensor.requires_grad_(True) if tensor.dtype != torch.long else tensor
        for tensor in python_tensors
    ]

// 将 Python 张量移动到指定设备上
def move_python_tensors_to_device(python_tensors, device):
    return [tensor.to(device) for tensor in python_tensors]

// 检查单元测试类是否具有指定名称的测试方法
def has_test(unit_test_class, test_name):
    return hasattr(unit_test_class, test_name)

// 动态地为单元测试类添加测试方法
def add_test(unit_test_class, test_name, test_fn):
    if has_test(unit_test_class, test_name):
        raise RuntimeError("Found two tests with the same name: " + test_name)
    setattr(unit_test_class, test_name, test_fn)

// 将 C++ 张量声明设置为需要梯度，与给定的 Python 张量对应
def set_cpp_tensors_requires_grad(cpp_tensor_stmts, python_tensors):
    assert len(cpp_tensor_stmts) == len(python_tensors)
    return [
        f"{tensor_stmt}.requires_grad_(true)"
        if tensor.dtype != torch.long
        else tensor_stmt
        for tensor_stmt, (_, tensor) in zip(cpp_tensor_stmts, python_tensors)
    ]

// 将 C++ 张量移动到指定设备上
def move_cpp_tensors_to_device(cpp_tensor_stmts, device):
    # 返回一个列表，其中每个元素都是一个字符串，格式为 "{tensor_stmt}.to("{device}")"
    return [f'{tensor_stmt}.to("{device}")' for tensor_stmt in cpp_tensor_stmts]
def is_criterion_test(test_instance):
    # 检查传入的测试实例是否是 common_nn.CriterionTest 类的实例
    return isinstance(test_instance, common_nn.CriterionTest)


# This function computes the following:
# - What variable declaration statements should show up in the C++ parity test function
# - What arguments should be passed into the C++ module/functional's forward function
#
# For example, for the "L1Loss" test, the return values from this function are:
# ```
# // Note that `arg_dict` stores all tensor values we transfer from Python to C++
# cpp_args_construction_stmts = [
#   "auto i0 = arg_dict.at("i0").to("cpu").requires_grad_(true)",
#   "auto t0 = arg_dict.at("t0").to("cpu")",
# ],
# cpp_forward_args_symbols = [
#   "i0",
#   "t0",
# ]
# ```
def compute_cpp_args_construction_stmts_and_forward_arg_symbols(test_params):
    # 获取测试参数的设备信息
    device = test_params.device
    # 初始化用于存储 C++ forward 函数参数名的列表
    cpp_forward_args_symbols = []

    # 内部函数，用于生成 C++ forward 函数的参数声明语句
    def add_cpp_forward_args(args):
        args_stmts = []
        for arg_name, _ in args:
            args_stmts.append(f'auto {arg_name} = arg_dict.at("{arg_name}")')
            cpp_forward_args_symbols.append(arg_name)
        return args_stmts

    # 处理输入参数：设置为需要梯度并移动到指定设备
    cpp_forward_input_args_stmts = set_cpp_tensors_requires_grad(
        move_cpp_tensors_to_device(
            add_cpp_forward_args(test_params.arg_dict["input"]), device
        ),
        test_params.arg_dict["input"],
    )
    
    # 处理目标参数：移动到指定设备
    cpp_forward_target_args_stmts = move_cpp_tensors_to_device(
        add_cpp_forward_args(test_params.arg_dict["target"]), device
    )
    
    # 处理额外参数：移动到指定设备
    cpp_forward_extra_args_stmts = move_cpp_tensors_to_device(
        add_cpp_forward_args(test_params.arg_dict["extra_args"]), device
    )

    # 构建其他参数的声明语句列表
    cpp_other_args_stmts = []
    for arg_name, _ in test_params.arg_dict["other"]:
        cpp_other_args_stmts.append(f'auto {arg_name} = arg_dict.at("{arg_name}")')
    cpp_other_args_stmts = move_cpp_tensors_to_device(cpp_other_args_stmts, device)

    # 组装所有参数声明语句
    cpp_args_construction_stmts = (
        cpp_forward_input_args_stmts
        + cpp_forward_target_args_stmts
        + cpp_forward_extra_args_stmts
        + cpp_other_args_stmts
    )

    return cpp_args_construction_stmts, cpp_forward_args_symbols


def serialize_arg_dict_as_script_module(arg_dict):
    # 将参数字典扁平化为一个字典
    arg_dict_flat = dict(
        arg_dict["input"]
        + arg_dict["target"]
        + arg_dict["extra_args"]
        + arg_dict["other"]
    )
    # 创建一个空的 PyTorch 模块
    arg_dict_module = torch.nn.Module()
    # 将扁平化后的参数字典中的每个 tensor 注册为模块的 buffer
    for arg_name, arg_value in arg_dict_flat.items():
        assert isinstance(arg_value, torch.Tensor)
        arg_dict_module.register_buffer(arg_name, arg_value)

    return torch.jit.script(arg_dict_module)


# NOTE: any argument symbol used in `cpp_constructor_args` / `cpp_options_args` / `cpp_function_call`
# must have a mapping in `cpp_var_map`.
#
# The mapping can take one of the following formats:
#
# 1. `argument_name` -> Python value
# 2. `argument_name` -> '_get_input()' (which means `argument_name` in C++ will be bound to `test_instance._get_input()`)
#
# For example:
# ```
# 定义一个函数，用于生成测试参数字典
def compute_arg_dict(test_params_dict, test_instance):
    # 初始化参数字典，包含四个空列表
    arg_dict = {
        "input": [],
        "target": [],
        "extra_args": [],
        "other": [],
    }

    # 定义内部函数，将不同类型的参数放入参数字典中
    def put_args_into_arg_dict(arg_type, arg_type_prefix, args):
        for i, arg in enumerate(args):
            arg_dict[arg_type].append(CppArg(name=arg_type_prefix + str(i), value=arg))

    # 将测试实例的输入数据转换为列表并放入参数字典中
    put_args_into_arg_dict("input", "i", convert_to_list(test_instance._get_input()))
    
    # 如果是损失函数测试，则将目标数据转换为列表并放入参数字典中
    if is_criterion_test(test_instance):
        put_args_into_arg_dict("target", "t", convert_to_list(test_instance._get_target()))
    
    # 如果测试实例有额外参数，则将其转换为列表并放入参数字典中
    if test_instance.extra_args:
        put_args_into_arg_dict("extra_args", "e", convert_to_list(test_instance.extra_args))

    # 获取测试参数字典中的 cpp_var_map，处理其中的各种数据类型
    cpp_var_map = test_params_dict.get("cpp_var_map", {})
    for arg_name, arg_value in cpp_var_map.items():
        if isinstance(arg_value, str):
            # 如果值为 "_get_input()"，则将测试实例的输入数据放入参数字典中
            if arg_value == "_get_input()":
                arg_dict["other"].append(
                    CppArg(name=arg_name, value=test_instance._get_input())
                )
            else:
                # 否则抛出运行时异常，因为不支持该字符串值
                raise RuntimeError(
                    f"`{arg_name}` has unsupported string value: {arg_value}"
                )
        elif isinstance(arg_value, torch.Tensor):
            # 如果值为 torch.Tensor，则将其放入参数字典中
            arg_dict["other"].append(CppArg(name=arg_name, value=arg_value))
        else:
            # 否则抛出运行时异常，因为不支持该值类型
            raise RuntimeError(f"`{arg_name}` has unsupported value: {arg_value}")

    # 返回生成的参数字典
    return arg_dict


# 定义一个装饰器函数，根据条件装饰测试函数
def decorate_test_fn(test_fn, test_cuda, has_impl_parity, device):
    # 如果设备为 "cuda"，则根据条件装饰测试函数
    if device == "cuda":
        test_fn = unittest.skipIf(not TEST_CUDA, "CUDA unavailable")(test_fn)
        test_fn = unittest.skipIf(not test_cuda, "Excluded from CUDA tests")(test_fn)

    # 如果没有实现一致性，将测试函数标记为预期失败
    if not has_impl_parity:
        test_fn = unittest.expectedFailure(test_fn)

    # 返回装饰后的测试函数
    return test_fn
# 定义一个长字符串，用于存储 C++ API 与 Python API 不一致时的错误信息和建议
MESSAGE_HOW_TO_FIX_CPP_PARITY_TEST_FAILURE = """
What should I do when C++ API parity test is failing?

- If you are changing the implementation of an existing `torch.nn` module / `torch.nn.functional` function:
Answer: Ideally you should also change the C++ API implementation for that module / function
(you can start by searching for the module / function name in `torch/csrc/api/` folder).

- If you are adding a new test for an existing `torch.nn` module / `torch.nn.functional` function:
Answer: Ideally you should fix the C++ API implementation for that module / function
to exactly match the Python API implementation (you can start by searching for the module /
function name in `torch/csrc/api/` folder).

- If you are adding a test for a *new* `torch.nn` module / `torch.nn.functional` function:
Answer: Ideally you should add the corresponding C++ API implementation for that module / function,
and it should exactly match the Python API implementation. (We have done a large effort on this
which is tracked at https://github.com/pytorch/pytorch/issues/25883.)

However, if any of the above is proven to be too complicated, you can just add
`test_cpp_api_parity=False` to any failing test in `torch/testing/_internal/common_nn.py`,
and the C++ API parity test will be skipped accordingly. Note that you should
also file an issue when you do this.

For more details on how to add a C++ API parity test, please see:
NOTE [How to check NN module / functional API parity between Python and C++ frontends]
"""

# 定义函数 generate_error_msg，生成包含错误信息和建议的字符串
def generate_error_msg(name, cpp_value, python_value):
    return (
        f"Parity test failed: {name} in C++ has value: {cpp_value}, "
        f"which does not match the corresponding value in Python: {python_value}.\n{MESSAGE_HOW_TO_FIX_CPP_PARITY_TEST_FAILURE}"
    )

# 定义函数 try_remove_folder，尝试删除指定路径的文件夹，如果失败则作为警告显示错误消息
def try_remove_folder(folder_path):
    if os.path.exists(folder_path):
        # 不阻塞进程，但会显示错误消息作为警告
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            warnings.warn(
                f"Non-blocking folder removal fails with the following error:\n{str(e)}"
            )
```