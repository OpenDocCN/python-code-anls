# `ZeroNet\plugins\AnnounceLocal\Test\conftest.py`

```
# 从src.Test.conftest模块中导入所有内容
from src.Test.conftest import *
# 从Config模块中导入config对象
from Config import config
# 将config对象的broadcast_port属性设置为0
config.broadcast_port = 0
```