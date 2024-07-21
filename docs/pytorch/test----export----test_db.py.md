# `.\pytorch\test\export\test_db.py`

```py
# Owner(s): ["oncall: export"]

# 导入必要的库和模块
import copy  # 导入深拷贝功能模块
import unittest  # 导入单元测试框架模块

import torch._dynamo as torchdynamo  # 导入 torch._dynamo 模块
from torch._export.db.case import ExportCase, normalize_inputs, SupportLevel  # 导入导出案例相关的类和函数
from torch._export.db.examples import (
    filter_examples_by_support_level,  # 导入根据支持级别筛选示例的函数
    get_rewrite_cases,  # 导入获取重写案例的函数
)
from torch.export import export  # 导入导出功能模块
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入实例化参数化测试函数
    IS_WINDOWS,  # 导入是否为 Windows 系统的标志
    parametrize,  # 导入参数化装饰器
    run_tests,  # 导入运行测试函数
    TestCase,  # 导入测试用例基类
)


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class ExampleTests(TestCase):
    # TODO Maybe we should make this tests actually show up in a file?
    # 参数化测试，对于支持级别为 SUPPORTED 的案例执行测试
    @parametrize(
        "name,case",  # 参数化装饰器的参数，表示参数名为 name，参数值为 case
        filter_examples_by_support_level(SupportLevel.SUPPORTED).items(),  # 使用支持级别过滤示例
        name_fn=lambda name, case: f"case_{name}",  # 生成测试名称的函数
    )
    def test_exportdb_supported(self, name: str, case: ExportCase) -> None:
        model = case.model  # 获取案例中的模型

        inputs_export = normalize_inputs(case.example_inputs)  # 标准化示例输入
        inputs_model = copy.deepcopy(inputs_export)  # 深拷贝标准化后的示例输入
        exported_program = export(
            model,
            inputs_export.args,
            inputs_export.kwargs,
            dynamic_shapes=case.dynamic_shapes,
        )  # 导出模型程序

        exported_program.graph_module.print_readable()  # 打印可读的导出模型图形模块

        self.assertEqual(
            exported_program.module()(*inputs_export.args, **inputs_export.kwargs),
            model(*inputs_model.args, **inputs_model.kwargs),
        )  # 断言导出程序的执行结果与模型直接执行的结果相等

        if case.extra_inputs is not None:
            inputs = normalize_inputs(case.extra_inputs)  # 标准化额外输入
            self.assertEqual(
                exported_program.module()(*inputs.args, **inputs.kwargs),
                model(*inputs.args, **inputs.kwargs),
            )  # 如果有额外输入，则断言导出程序的额外执行结果与模型的额外执行结果相等

    # 参数化测试，对于支持级别为 NOT_SUPPORTED_YET 的案例执行测试
    @parametrize(
        "name,case",  # 参数化装饰器的参数，表示参数名为 name，参数值为 case
        filter_examples_by_support_level(SupportLevel.NOT_SUPPORTED_YET).items(),  # 使用支持级别过滤示例
        name_fn=lambda name, case: f"case_{name}",  # 生成测试名称的函数
    )
    def test_exportdb_not_supported(self, name: str, case: ExportCase) -> None:
        model = case.model  # 获取案例中的模型
        # 忽略 Pyre 检查
        with self.assertRaises(
            (torchdynamo.exc.Unsupported, AssertionError, RuntimeError)
        ):  # 使用断言上下文管理器，捕获异常
            inputs = normalize_inputs(case.example_inputs)  # 标准化示例输入
            exported_model = export(
                model,
                inputs.args,
                inputs.kwargs,
                dynamic_shapes=case.dynamic_shapes,
            )  # 尝试导出模型

    # 构造 NOT_SUPPORTED_YET 支持级别的重写案例列表
    exportdb_not_supported_rewrite_cases = [
        (name, rewrite_case)
        for name, case in filter_examples_by_support_level(
            SupportLevel.NOT_SUPPORTED_YET
        ).items()
        for rewrite_case in get_rewrite_cases(case)
    ]
    # 如果存在不支持导出的重写案例，则进行参数化测试
    @parametrize(
        "name,rewrite_case",  # 参数化测试的参数：名称和重写案例
        exportdb_not_supported_rewrite_cases,  # 使用不支持导出的重写案例作为参数
        name_fn=lambda name, case: f"case_{name}_{case.name}",  # 自定义测试名称的函数
    )
    # 定义测试方法：测试不支持导出的重写案例
    def test_exportdb_not_supported_rewrite(
        self, name: str, rewrite_case: ExportCase
    ) -> None:
        # 忽略 Pyre 的类型检查提示
        # 根据重写案例的示例输入，规范化输入数据
        inputs = normalize_inputs(rewrite_case.example_inputs)
        # 导出模型，获取导出结果
        exported_model = export(
            rewrite_case.model,  # 使用重写案例的模型
            inputs.args,  # 输入参数
            inputs.kwargs,  # 输入关键字参数
            dynamic_shapes=rewrite_case.dynamic_shapes,  # 动态形状参数
        )
# 调用函数 instantiate_parametrized_tests，并传入 ExampleTests 作为参数，实例化参数化测试
instantiate_parametrized_tests(ExampleTests)

# 检查当前脚本是否作为主程序运行
if __name__ == "__main__":
    # 如果是主程序，则执行 run_tests 函数，运行测试
    run_tests()
```