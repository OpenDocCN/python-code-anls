# `.\pytorch\test\distributed\argparse_util_test.py`

```py
#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入必要的模块
import os
import unittest
from argparse import ArgumentParser

# 导入自定义的分布式环境变量检查工具函数
from torch.distributed.argparse_util import check_env, env


class ArgParseUtilTest(unittest.TestCase):
    def setUp(self):
        # 清除所有以"PET_"开头的环境变量，以确保测试环境干净
        for e in os.environ.keys():
            if e.startswith("PET_"):
                del os.environ[e]

    # 测试对字符串类型参数的环境变量处理，当环境变量不存在时的默认值
    def test_env_string_arg_no_env(self):
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default="bar")

        # 测试默认情况下的参数值
        self.assertEqual("bar", parser.parse_args([]).foo)
        # 测试在命令行中指定参数值的情况
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    # 测试对字符串类型参数的环境变量处理，当环境变量存在时
    def test_env_string_arg_env(self):
        os.environ["PET_FOO"] = "env_baz"
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default="bar")

        # 测试环境变量存在时的参数值
        self.assertEqual("env_baz", parser.parse_args([]).foo)
        # 测试在命令行中指定参数值时，环境变量仍起作用
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    # 测试对整数类型参数的环境变量处理，当环境变量不存在时的默认值
    def test_env_int_arg_no_env(self):
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default=1, type=int)

        self.assertEqual(1, parser.parse_args([]).foo)
        self.assertEqual(2, parser.parse_args(["-f", "2"]).foo)
        self.assertEqual(2, parser.parse_args(["--foo", "2"]).foo)

    # 测试对整数类型参数的环境变量处理，当环境变量存在时
    def test_env_int_arg_env(self):
        os.environ["PET_FOO"] = "3"
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default=1, type=int)

        self.assertEqual(3, parser.parse_args([]).foo)
        self.assertEqual(2, parser.parse_args(["-f", "2"]).foo)
        self.assertEqual(2, parser.parse_args(["--foo", "2"]).foo)

    # 测试对未指定默认值的参数的环境变量处理，当环境变量不存在时
    def test_env_no_default_no_env(self):
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env)

        self.assertIsNone(parser.parse_args([]).foo)
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    # 测试对未指定默认值的参数的环境变量处理，当环境变量存在时
    def test_env_no_default_env(self):
        os.environ["PET_FOO"] = "env_baz"
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env)

        self.assertEqual("env_baz", parser.parse_args([]).foo)
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)
    # 测试环境中未设置环境变量时的情况
    def test_env_required_no_env(self):
        # 创建参数解析器
        parser = ArgumentParser()
        # 添加一个要求必须通过环境变量设置的参数选项
        parser.add_argument("-f", "--foo", action=env, required=True)

        # 测试解析器能否正确解析参数并返回预期值
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    # 测试环境中设置了环境变量时的情况
    def test_env_required_env(self):
        # 设置环境变量
        os.environ["PET_FOO"] = "env_baz"
        # 创建参数解析器
        parser = ArgumentParser()
        # 添加一个要求必须通过环境变量设置的参数选项，设置默认值为 "bar"
        parser.add_argument("-f", "--foo", action=env, default="bar", required=True)

        # 测试解析器能否正确解析参数并返回预期的环境变量值
        self.assertEqual("env_baz", parser.parse_args([]).foo)
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    # 测试环境中未设置环境变量时的情况
    def test_check_env_no_env(self):
        # 创建参数解析器
        parser = ArgumentParser()
        # 添加一个检查环境变量是否存在的参数选项
        parser.add_argument("-v", "--verbose", action=check_env)

        # 测试解析器能否正确解析参数并返回预期的布尔值
        self.assertFalse(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["-v"]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    # 测试环境中未设置环境变量时的情况，并设置默认值为 True
    def test_check_env_default_no_env(self):
        # 创建参数解析器
        parser = ArgumentParser()
        # 添加一个检查环境变量是否存在的参数选项，并设置默认值为 True
        parser.add_argument("-v", "--verbose", action=check_env, default=True)

        # 测试解析器能否正确解析参数并返回预期的布尔值
        self.assertTrue(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["-v"]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    # 测试环境中设置了环境变量为 "0" 的情况
    def test_check_env_env_zero(self):
        # 设置环境变量
        os.environ["PET_VERBOSE"] = "0"
        # 创建参数解析器
        parser = ArgumentParser()
        # 添加一个检查环境变量是否存在的参数选项

        parser.add_argument("-v", "--verbose", action=check_env)

        # 测试解析器能否正确解析参数并返回预期的布尔值
        self.assertFalse(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    # 测试环境中设置了环境变量为 "1" 的情况
    def test_check_env_env_one(self):
        # 设置环境变量
        os.environ["PET_VERBOSE"] = "1"
        # 创建参数解析器
        parser = ArgumentParser()
        # 添加一个检查环境变量是否存在的参数选项
        parser.add_argument("-v", "--verbose", action=check_env)

        # 测试解析器能否正确解析参数并返回预期的布尔值
        self.assertTrue(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    # 测试环境中设置了环境变量为 "0" 的情况，并设置默认值为 True
    def test_check_env_default_env_zero(self):
        # 设置环境变量
        os.environ["PET_VERBOSE"] = "0"
        # 创建参数解析器
        parser = ArgumentParser()
        # 添加一个检查环境变量是否存在的参数选项，并设置默认值为 True
        parser.add_argument("-v", "--verbose", action=check_env, default=True)

        # 测试解析器能否正确解析参数并返回预期的布尔值
        self.assertFalse(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    # 测试环境中设置了环境变量为 "1" 的情况，并设置默认值为 True
    def test_check_env_default_env_one(self):
        # 设置环境变量
        os.environ["PET_VERBOSE"] = "1"
        # 创建参数解析器
        parser = ArgumentParser()
        # 添加一个检查环境变量是否存在的参数选项，并设置默认值为 True
        parser.add_argument("-v", "--verbose", action=check_env, default=True)

        # 测试解析器能否正确解析参数并返回预期的布尔值
        self.assertTrue(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)
```