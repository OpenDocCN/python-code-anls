# `D:\src\scipysrc\pandas\scripts\tests\conftest.py`

```
# 定义 pytest 插件的函数 pytest_addoption，用于添加命令行选项到 pytest 的参数解析器中
def pytest_addoption(parser) -> None:
    # 向参数解析器添加一个选项 --strict-data-files
    parser.addoption(
        "--strict-data-files",
        action="store_true",  # 设置选项的行为为存储为 True，即命令行中使用 --strict-data-files 时，该选项的值为 True
        help="Unused",  # 帮助信息，提示该选项当前未被使用
    )
```