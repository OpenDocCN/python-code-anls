# `.\commands\__init__.py`

```py
# 引入抽象基类（ABC）和抽象方法（abstractmethod）来定义一个基于命令行接口（CLI）的Transformers库命令的基类
from abc import ABC, abstractmethod
# 从argparse模块导入ArgumentParser类，用于解析命令行参数
from argparse import ArgumentParser

# 定义一个抽象基类BaseTransformersCLICommand，继承自ABC类
class BaseTransformersCLICommand(ABC):
    # 声明一个静态方法，用于注册子命令到给定的ArgumentParser对象中
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError()

    # 声明一个抽象方法run，表示子类需要实现的运行方法
    @abstractmethod
    def run(self):
        raise NotImplementedError()
```