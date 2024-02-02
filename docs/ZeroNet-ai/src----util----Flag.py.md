# `ZeroNet\src\util\Flag.py`

```py
# 导入 defaultdict 类
from collections import defaultdict

# 定义 Flag 类
class Flag(object):
    # 初始化方法
    def __init__(self):
        # 设置有效标志的集合
        self.valid_flags = set([
            "admin",  # 只允许具有 ADMIN 权限的站点运行
            "async_run",  # 动作将使用 gevent.spawn 异步运行
            "no_multiuser"  # 如果多用户插件以开放代理模式运行，则禁用该动作
        ])
        # 创建一个默认值为集合的字典
        self.db = defaultdict(set)

    # 获取属性的方法
    def __getattr__(self, key):
        # 定义内部函数
        def func(f):
            # 如果 key 不在有效标志集合中，则抛出异常
            if key not in self.valid_flags:
                raise Exception("Invalid flag: %s (valid: %s)" % (key, self.valid_flags))
            # 将函数名和标志添加到字典中
            self.db[f.__name__].add(key)
            return f
        return func

# 创建 Flag 类的实例
flag = Flag()
```