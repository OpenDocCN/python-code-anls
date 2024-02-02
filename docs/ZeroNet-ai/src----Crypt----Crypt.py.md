# `ZeroNet\src\Crypt\Crypt.py`

```py
# 从Config模块中导入config对象
from Config import config
# 从util模块中导入ThreadPool类
from util import ThreadPool
# 使用config对象中的threads_crypt属性创建一个线程池对象
thread_pool_crypt = ThreadPool.ThreadPool(config.threads_crypt)
```