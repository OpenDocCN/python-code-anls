# `D:\src\scipysrc\matplotlib\doc\sphinxext\mock_gui_toolkits.py`

```
import sys  # 导入系统模块sys，用于操作系统相关功能
from unittest.mock import MagicMock  # 从unittest.mock模块导入MagicMock类，用于创建虚拟对象


class MyCairoCffi(MagicMock):
    __name__ = "cairocffi"  # 设置MyCairoCffi类的特殊属性__name__为"cairocffi"


def setup(app):
    sys.modules.update(  # 更新sys.modules字典，将'cairocffi'映射为MyCairoCffi类的实例
        cairocffi=MyCairoCffi(),
    )
    # 返回一个字典，指示应用程序在并行读取和写入时是安全的
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
```