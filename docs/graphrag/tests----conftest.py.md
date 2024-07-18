# `.\graphrag\tests\conftest.py`

```py
# 版权声明和许可声明，指明版权所有者和许可协议
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 定义 pytest 的插件函数 pytest_addoption，用于添加命令行选项
def pytest_addoption(parser):
    # 向解析器 parser 添加一个选项 "--run_slow"
    parser.addoption(
        # 指定选项的名称为 "--run_slow"
        "--run_slow",
        # 指定选项的动作为存储为布尔值（True/False）
        action="store_true",
        # 设置默认值为 False
        default=False,
        # 提供帮助信息，指示该选项用于运行慢速测试
        help="run slow tests"
    )
```