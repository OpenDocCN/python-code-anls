# `D:\src\scipysrc\pandas\pandas\_libs\sas.pyi`

```
from pandas.io.sas.sas7bdat import SAS7BDATReader
从 pandas 库中导入 SAS7BDATReader 类，用于读取 SAS 数据集

class Parser:
    def __init__(self, parser: SAS7BDATReader) -> None:
        # Parser 类的构造函数，接受一个 SAS7BDATReader 对象作为参数

    def read(self, nrows: int) -> None:
        # Parser 类的 read 方法，接受一个整数参数 nrows，用于读取数据集的指定行数

def get_subheader_index(signature: bytes) -> int:
    # get_subheader_index 函数，接受一个 bytes 类型的参数 signature，返回一个整数
```