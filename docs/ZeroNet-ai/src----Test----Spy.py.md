# `ZeroNet\src\Test\Spy.py`

```
# 导入日志模块
import logging

# 定义一个Spy类
class Spy:
    # 初始化方法，接收对象和函数名作为参数
    def __init__(self, obj, func_name):
        # 将对象和函数名保存为实例属性
        self.obj = obj
        self.__name__ = func_name
        # 获取原始函数并保存为实例属性
        self.func_original = getattr(self.obj, func_name)
        # 初始化调用列表
        self.calls = []

    # 进入上下文时调用的方法
    def __enter__(self, *args, **kwargs):
        # 记录调试信息
        logging.debug("Spy started")
        # 定义一个装饰函数，用于记录函数调用信息
        def loggedFunc(cls, *args, **kwargs):
            # 将参数保存为字典
            call = dict(enumerate(args, 1))
            call[0] = cls
            call.update(kwargs)
            # 记录调试信息
            logging.debug("Spy call: %s" % call)
            # 将调用信息添加到调用列表中
            self.calls.append(call)
            # 调用原始函数并返回结果
            return self.func_original(cls, *args, **kwargs)
        # 将装饰函数设置为对象的函数属性
        setattr(self.obj, self.__name__, loggedFunc)
        # 返回调用列表
        return self.calls

    # 退出上下文时调用的方法
    def __exit__(self, *args, **kwargs):
        # 将原始函数恢复为对象的函数属性
        setattr(self.obj, self.__name__, self.func_original)
```