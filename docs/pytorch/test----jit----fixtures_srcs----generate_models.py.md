# `.\pytorch\test\jit\fixtures_srcs\generate_models.py`

```py
# 导入必要的库和模块
import io  # 提供用于处理字节流的工具
import logging  # 提供日志记录功能
import sys  # 提供对系统特定参数和功能的访问
import zipfile  # 提供操作 ZIP 文件的功能
from pathlib import Path  # 提供操作路径的功能
from typing import Set  # 提供类型提示的支持

import torch  # 导入 PyTorch 深度学习库

# 导入用于升级器的测试时的工具模块
from test.jit.fixtures_srcs.fixtures_src import *  # noqa: F403

# 导入用于移动端模型导出和加载的函数
from torch.jit.mobile import _export_operator_list, _load_for_lite_interpreter

# 配置日志记录器，输出到标准输出流，并设置日志级别为 INFO
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# 创建名为 __name__ 的 logger 对象，用于记录 DEBUG 级别的日志信息
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

"""
本文件用于生成用于测试操作符更改的模型。请参考
https://github.com/pytorch/rfcs/blob/master/RFC-0017-PyTorch-Operator-Versioning.md 获取更多详细信息。

需要建立一个系统化的工作流程来更改操作符，以确保操作符变更的向后兼容性（BC）/向前兼容性（FC）。对于破坏 BC 的操作符变更，需要一个升级器。以下是正确落地破坏 BC 的操作符变更的步骤：

1. 在 caffe2/torch/csrc/jit/operator_upgraders/upgraders_entry.cpp 文件中编写一个升级器。软性要求的命名格式为 <operator_name>_<operator_overload>_<start>_<end>。例如，以下示例意味着在版本 0 到 3 的 div.Tensor 需要被此升级器替换。

/*
div_Tensor_0_3 is added for a change of operator div in pr xxxxxxx.
Create date: 12/02/2021
Expire date: 06/02/2022
*/
     {"div_Tensor_0_3", R"SCRIPT(
def div_Tensor_0_3(self: Tensor, other: Tensor) -> Tensor:
  if (self.is_floating_point() or other.is_floating_point()):
    return self.true_divide(other)
  return self.divide(other, rounding_mode='trunc')
)SCRIPT"},

2. 在 caffe2/torch/csrc/jit/operator_upgraders/version_map.h 中添加如下更改。确保条目根据版本增量号进行排序。
    {"div.Tensor",
      {{4,
        "div_Tensor_0_3",
        "aten::div.Tensor(Tensor self, Tensor other) -> Tensor"}}},

3. 重新构建 PyTorch 后，运行以下命令，将自动生成对 fbcode/caffe2/torch/csrc/jit/mobile/upgrader_mobile.cpp 的更改。

python pytorch/torchgen/operator_versions/gen_mobile_upgraders.py

4. 生成用于覆盖升级器的测试。

4.1 切换到操作符更改之前的提交，并在 `test/jit/fixtures_srcs/fixtures_src.py` 中添加一个模块。切换到旧模型的提交是因为需要旧操作符以确保升级器按预期工作。在 `test/jit/fixtures_srcs/generate_models.py` 中，添加该模块及其对应的更改操作符，如下所示。

ALL_MODULES = {
    TestVersionedDivTensorExampleV7(): "aten::div.Tensor",
}

此模块应包含更改的操作符。如果模型中未包含操作符，则在步骤 4.2 中的模型导出过程将失败。

4.2 运行以下命令将模型导出到 `test/jit/fixtures` 中。

python /Users/chenlai/pytorch/test/jit/fixtures_src/generate_models.py
"""
"""
A map of test modules and their corresponding changed operators.
key: test module instance
value: changed operator string
"""
ALL_MODULES = {
    TestVersionedDivTensorExampleV7(): "aten::div.Tensor",
    TestVersionedLinspaceV7(): "aten::linspace",
    TestVersionedLinspaceOutV7(): "aten::linspace.out",
    TestVersionedLogspaceV8(): "aten::logspace",
    TestVersionedLogspaceOutV8(): "aten::logspace.out",
    TestVersionedGeluV9(): "aten::gelu",
    TestVersionedGeluOutV9(): "aten::gelu.out",
    TestVersionedRandomV10(): "aten::random_.from",
    TestVersionedRandomFuncV10(): "aten::random.from",
    TestVersionedRandomOutV10(): "aten::random.from_out",
}

"""
获取`test/jit/fixtures`的路径，这里存储了所有包含操作符变更的测试模型。
"""
def get_fixtures_path() -> Path:
    pytorch_dir = Path(__file__).resolve().parents[3]
    fixtures_path = pytorch_dir / "test" / "jit" / "fixtures"
    return fixtures_path


"""
获取`test/jit/fixtures`中所有模型的名称。
"""
def get_all_models(model_directory_path: Path) -> Set[str]:
    files_in_fixtures = model_directory_path.glob("**/*")
    all_models_from_fixtures = [
        fixture.stem for fixture in files_in_fixtures if fixture.is_file()
    ]
    return set(all_models_from_fixtures)


"""
检查给定模型是否已经存在于`test/jit/fixtures`中。
"""
def model_exist(model_file_name: str, all_models: Set[str]) -> bool:
    return model_file_name in all_models


"""
获取给定模块的操作符列表。
"""
def get_operator_list(script_module: torch) -> Set[str]:
    buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
    buffer.seek(0)
    mobile_module = _load_for_lite_interpreter(buffer)
    operator_list = _export_operator_list(mobile_module)
    return operator_list


"""
获取输出模型的操作符版本号，给定一个模块。
"""
def get_output_model_version(script_module: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.jit.save(script_module, buffer)
    buffer.seek(0)
    zipped_model = zipfile.ZipFile(buffer)
    try:
        version = int(zipped_model.read("archive/version").decode("utf-8"))
        return version
    except KeyError:
        version = int(zipped_model.read("archive/.data/version").decode("utf-8"))
        return version


"""
遍历所有测试模块。如果对应的模型在`test/jit/fixtures`中不存在，则生成一个。以下情况不会导出模型：

1. 测试模块未覆盖变更的操作符。例如，test_versioned_div_tensor_example_v4应该测试操作符aten::div.Tensor。如果模型不包含该操作符，则会失败。
"""
# 导入必要的模块
The error message includes the actual operator list from the model.

# 检查给定路径下的所有模型，并获取它们的列表
def generate_models(model_directory_path: Path):
    all_models = get_all_models(model_directory_path)

# 此处代码暂缺，无法提供更多注释
    # 遍历 ALL_MODULES 字典，其中包含模块对象和预期操作符
    for a_module, expect_operator in ALL_MODULES.items():
        # 获取当前模块对象的类名，例如：TestVersionedDivTensorExampleV7
        torch_module_name = type(a_module).__name__

        # 检查当前模块对象是否为 torch.nn.Module 的实例，如果不是则记录错误日志
        if not isinstance(a_module, torch.nn.Module):
            logger.error(
                "The module %s "
                "is not a torch.nn.module instance. "
                "Please ensure it's a subclass of torch.nn.module in fixtures_src.py "
                "and it's registered as an instance in ALL_MODULES in generated_models.py",
                torch_module_name,
            )

        # 根据类名生成对应的模型名称，如 test_versioned_div_tensor_example_v4
        model_name = "".join(
            [
                "_" + char.lower() if char.isupper() else char
                for char in torch_module_name
            ]
        ).lstrip("_")

        # 某些模型可能已经不再需要编译，因此跳过已经存在对应模型文件的模型
        logger.info("Processing %s", torch_module_name)
        if model_exist(model_name, all_models):
            logger.info("Model %s already exists, skipping", model_name)
            continue

        # 将当前模块对象转换为 Torch 脚本模块
        script_module = torch.jit.script(a_module)
        
        # 获取当前脚本模块的实际模型版本
        actual_model_version = get_output_model_version(script_module)

        # 获取当前 Torch 运算符的最大版本号
        current_operator_version = torch._C._get_max_operator_version()
        
        # 检查实际模型版本是否大于或等于当前运算符版本加一，如果是则记录错误日志并跳过
        if actual_model_version >= current_operator_version + 1:
            logger.error(
                "Actual model version %s "
                "is equal or larger than %s + 1. "
                "Please run the script before the commit to change operator.",
                actual_model_version,
                current_operator_version,
            )
            continue

        # 获取当前脚本模块包含的实际运算符列表
        actual_operator_list = get_operator_list(script_module)
        
        # 检查预期运算符是否在实际运算符列表中，如果不在则记录错误日志并跳过
        if expect_operator not in actual_operator_list:
            logger.error(
                "The model includes operator: %s, "
                "however it doesn't cover the operator %s."
                "Please ensure the output model includes the tested operator.",
                actual_operator_list,
                expect_operator,
            )
            continue

        # 导出模型路径，以模型名称加上 ".ptl" 扩展名保存
        export_model_path = str(model_directory_path / (str(model_name) + ".ptl"))
        
        # 使用轻量级解释器保存当前脚本模块为文件
        script_module._save_for_lite_interpreter(export_model_path)
        logger.info(
            "Generating model %s and it's save to %s", model_name, export_model_path
        )
# 定义主函数，程序的入口点
def main() -> None:
    # 获取模型目录路径，通常是从配置或默认位置获取
    model_directory_path = get_fixtures_path()
    # 调用函数生成模型，传入模型目录路径作为参数
    generate_models(model_directory_path)

# 如果当前脚本作为主程序运行，则执行主函数
if __name__ == "__main__":
    main()
```