# `.\pytorch\tools\test\gen_operators_yaml_test.py`

```
#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# 引入 argparse 模块，用于解析命令行参数
import argparse
# 引入 json 模块，用于处理 JSON 格式的数据
import json
# 引入 unittest 模块，用于编写和运行单元测试
import unittest
# 引入 defaultdict 类，用于创建默认值为集合的字典
from collections import defaultdict
# 从 unittest.mock 模块中引入 Mock 和 patch 类，用于模拟对象和替换对象
from unittest.mock import Mock, patch

# 从 gen_operators_yaml.py 文件中导入多个函数和变量
from gen_operators_yaml import (
    fill_output,
    get_parser_options,
    make_filter_from_options,
    verify_all_specified_present,
)


# 定义一个函数，返回模拟的命令行选项对象
def _mock_options():
    options = argparse.Namespace()
    options.root_ops = "aten::add,aten::cat"
    options.training_root_ops = []
    options.output_path = "/tmp"
    options.dep_graph_yaml_path = "dummy_pytorch_op_deps.yaml"
    options.model_name = "test_model"
    options.model_versions = None
    options.model_assets = None
    options.model_backends = None
    options.models_yaml_path = None
    options.include_all_operators = False
    options.rule_name = "test_rule"
    options.not_include_all_overloads_static_root_ops = True
    options.not_include_all_overloads_closure_ops = True

    return options


# 定义一个函数，返回模拟的操作依赖图字典
def _mock_load_op_dep_graph():
    result = defaultdict(set)
    result["aten::add"] = {"aten::add", "aten::as_strided_"}
    result["aten::cat"] = {"aten::cat", "aten::as_strided_"}
    return dict(result)


# 定义一个单元测试类 GenOperatorsYAMLTest，继承自 unittest.TestCase
class GenOperatorsYAMLTest(unittest.TestCase):
    
    # 重写 setUp 方法，在每个测试方法执行前执行
    def setUp(self) -> None:
        pass

    # 定义一个测试方法 test_filter_creation，测试过滤器函数的创建
    def test_filter_creation(self) -> None:
        # 调用 make_filter_from_options 函数创建过滤器函数
        filter_func = make_filter_from_options(
            model_name="abc",
            model_versions=["100", "101"],
            model_assets=None,
            model_backends=None,
        )
        # 定义一个配置列表 config，包含多个模型配置字典
        config = [
            {
                "model": {
                    "name": "abc",
                    "version": 100,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
                "traced_operators": [],
            },
            {
                "model": {
                    "name": "abc",
                    "version": 102,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
            },
            {
                "model": {
                    "name": "abcd",
                    "version": 100,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
                "traced_operators": [],
            },
            {
                "model": {
                    "name": "abc",
                    "version": 101,
                    "asset": "asset-2",
                    "backend": "CPU",
                },
                "root_operators": [],
            },
        ]

        # 对配置列表应用过滤器函数，得到过滤后的配置列表 filtered_configs
        filtered_configs = list(filter(filter_func, config))
        # 使用断言验证过滤后的配置列表长度是否为 2
        assert (
            len(filtered_configs) == 2
        ), f"Expected 2 elements in filtered_configs, but got {len(filtered_configs)}"
    # 定义一个测试方法，用于验证成功的情况
    def test_verification_success(self) -> None:
        # 创建一个过滤函数，根据给定选项生成过滤条件
        filter_func = make_filter_from_options(
            model_name="abc",
            model_versions=["100", "101"],
            model_assets=["asset-1", "asset-2"],
            model_backends=None,
        )
        # 定义配置列表，包含两个字典，每个字典描述一个模型的配置
        config = [
            {
                "model": {
                    "name": "abc",
                    "version": 100,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
                "traced_operators": [],
            },
            {
                "model": {
                    "name": "abc",
                    "version": 101,
                    "asset": "asset-2",
                    "backend": "CPU",
                },
                "root_operators": [],
            },
        ]
        # 使用过滤函数过滤配置列表，生成过滤后的配置列表
        filtered_configs = list(filter(filter_func, config))
        # 尝试调用验证函数，检查指定的模型版本和资产是否存在于给定的配置中
        try:
            verify_all_specified_present(
                model_assets=["asset-1", "asset-2"],
                model_versions=["100", "101"],
                selected_models_yaml=filtered_configs,
                rule_name="test",
                model_name="abc",
                new_style_rule=True,
            )
        # 如果抛出异常，则用测试框架中的fail方法标记测试失败
        except Exception:
            self.fail(
                "expected verify_all_specified_present to succeed instead it raised an exception"
            )
    # 定义一个测试方法，用于验证失败情况
    def test_verification_fail(self) -> None:
        # 定义配置信息列表，包含两个字典，每个字典描述一个模型的配置及其操作符信息
        config = [
            {
                "model": {
                    "name": "abc",
                    "version": 100,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],  # 根操作符为空列表
                "traced_operators": [],  # 被跟踪的操作符为空列表
            },
            {
                "model": {
                    "name": "abc",
                    "version": 101,
                    "asset": "asset-2",
                    "backend": "CPU",
                },
                "root_operators": [],  # 根操作符为空列表
            },
        ]

        # 定义正确的资产（asset）、版本（version）和模型名称（name）
        good_assets = ["asset-1", "asset-2"]
        good_versions = ["100", "101"]
        good_name = "abc"

        # 测试不良资产（bad asset）
        filter_func_bad_asset = make_filter_from_options(
            model_name=good_name,
            model_versions=good_versions,
            model_assets=["asset-1", "asset-2", "asset-3"],  # 包含一个错误的资产名
            model_backends=None,
        )
        # 使用过滤函数过滤配置列表，得到不符合条件的配置列表
        filtered_configs_asset = list(filter(filter_func_bad_asset, config))
        # 断言运行时错误应该被触发
        with self.assertRaises(RuntimeError):
            verify_all_specified_present(
                model_assets=["asset-1", "asset-2", "asset-3"],
                model_versions=good_versions,
                selected_models_yaml=filtered_configs_asset,
                rule_name="test",
                model_name=good_name,
                new_style_rule=True,
            )

        # 测试不良版本（bad version）
        filter_func_bad_version = make_filter_from_options(
            model_name=good_name,
            model_versions=["100", "101", "102"],  # 包含一个错误的版本号
            model_assets=good_assets,
            model_backends=None,
        )
        # 使用过滤函数过滤配置列表，得到不符合条件的配置列表
        filtered_configs_version = list(filter(filter_func_bad_version, config))
        # 断言运行时错误应该被触发
        with self.assertRaises(RuntimeError):
            verify_all_specified_present(
                model_assets=good_assets,
                model_versions=["100", "101", "102"],
                selected_models_yaml=filtered_configs_version,
                rule_name="test",
                model_name=good_name,
                new_style_rule=True,
            )

        # 测试不良模型名称（bad name）
        filter_func_bad_name = make_filter_from_options(
            model_name="abcd",  # 使用一个错误的模型名称
            model_versions=good_versions,
            model_assets=good_assets,
            model_backends=None,
        )
        # 使用过滤函数过滤配置列表，得到不符合条件的配置列表
        filtered_configs_name = list(filter(filter_func_bad_name, config))
        # 断言运行时错误应该被触发
        with self.assertRaises(RuntimeError):
            verify_all_specified_present(
                model_assets=good_assets,
                model_versions=good_versions,
                selected_models_yaml=filtered_configs_name,
                rule_name="test",
                model_name="abcd",
                new_style_rule=True,
            )

    @patch("gen_operators_yaml.parse_options", return_value=_mock_options())
    @patch(
        "gen_operators_yaml.load_op_dep_graph", return_value=_mock_load_op_dep_graph()
    )
    # 使用 patch 装饰器，模拟 gen_operators_yaml.load_op_dep_graph 函数调用并返回 _mock_load_op_dep_graph() 的结果
    def test_fill_output_with_arguments_not_include_all_overloads(
        self, mock_parse_options: Mock, mock_load_op_dep_graph: Mock
    ) -> None:
        # 定义测试函数，设置 mock_parse_options 和 mock_load_op_dep_graph 为 Mock 对象
        parser = argparse.ArgumentParser(description="Generate used operators YAML")
        # 创建 argparse.ArgumentParser 对象，描述为 "Generate used operators YAML"
        options = get_parser_options(parser)
        # 调用 get_parser_options 函数，获取解析器选项

        model_dict = {
            "model_name": options.model_name,
            "asset_info": {},
            "is_new_style_rule": False,
        }
        # 创建 model_dict 字典，包含模型名、空的资产信息和 is_new_style_rule 标志

        output = {"debug_info": [json.dumps(model_dict)]}
        # 创建 output 字典，包含一个 "debug_info" 键，值为 model_dict 的 JSON 字符串形式的列表

        fill_output(output, options)
        # 调用 fill_output 函数，将结果填充到 output 中

        for op_val in output["operators"].values():
            # 遍历 output 字典中 "operators" 键对应值的所有元素
            self.assertFalse(op_val["include_all_overloads"])
            # 对每个操作值，断言其 "include_all_overloads" 键的值为 False
```