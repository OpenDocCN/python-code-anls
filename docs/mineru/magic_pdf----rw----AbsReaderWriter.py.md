# `.\MinerU\magic_pdf\rw\AbsReaderWriter.py`

```
# 从 abc 模块导入 ABC 和 abstractmethod，用于创建抽象基类
from abc import ABC, abstractmethod


# 定义一个抽象基类 AbsReaderWriter，继承自 ABC
class AbsReaderWriter(ABC):
    # 定义文本模式常量
    MODE_TXT = "text"
    # 定义二进制模式常量
    MODE_BIN = "binary"
    # 声明一个抽象方法 read，接受路径和模式参数
    @abstractmethod
    def read(self, path: str, mode=MODE_TXT):
        # 抛出未实现错误，强制子类实现此方法
        raise NotImplementedError

    # 声明一个抽象方法 write，接受内容、路径和模式参数
    @abstractmethod
    def write(self, content: str, path: str, mode=MODE_TXT):
        # 抛出未实现错误，强制子类实现此方法
        raise NotImplementedError

    # 声明一个抽象方法 read_offset，接受路径、偏移量和限制参数
    @abstractmethod
    def read_offset(self, path: str, offset=0, limit=None) -> bytes:
        # 抛出未实现错误，强制子类实现此方法
        raise NotImplementedError
```