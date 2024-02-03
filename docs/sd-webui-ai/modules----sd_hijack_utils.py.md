# `stable-diffusion-webui\modules\sd_hijack_utils.py`

```py
# 导入 importlib 模块，用于动态导入模块
import importlib

# 定义一个类 CondFunc
class CondFunc:
    # 定义 __new__ 方法，用于创建实例
    def __new__(cls, orig_func, sub_func, cond_func):
        # 创建一个新实例
        self = super(CondFunc, cls).__new__(cls)
        # 如果 orig_func 是字符串
        if isinstance(orig_func, str):
            # 将字符串按 '.' 分割成列表
            func_path = orig_func.split('.')
            # 从后往前遍历 func_path 列表
            for i in range(len(func_path)-1, -1, -1):
                try:
                    # 尝试导入模块
                    resolved_obj = importlib.import_module('.'.join(func_path[:i]))
                    break
                except ImportError:
                    pass
            # 遍历 func_path[i:-1] 列表
            for attr_name in func_path[i:-1]:
                # 获取属性值
                resolved_obj = getattr(resolved_obj, attr_name)
            # 获取 orig_func 对应的函数对象
            orig_func = getattr(resolved_obj, func_path[-1])
            # 将 orig_func 替换为 lambda 函数，调用 self
            setattr(resolved_obj, func_path[-1], lambda *args, **kwargs: self(*args, **kwargs))
        # 调用 __init__ 方法
        self.__init__(orig_func, sub_func, cond_func)
        # 返回一个 lambda 函数，调用 self
        return lambda *args, **kwargs: self(*args, **kwargs)
    
    # 定义 __init__ 方法，初始化实例属性
    def __init__(self, orig_func, sub_func, cond_func):
        self.__orig_func = orig_func
        self.__sub_func = sub_func
        self.__cond_func = cond_func
    
    # 定义 __call__ 方法，实现实例的调用
    def __call__(self, *args, **kwargs):
        # 如果没有条件函数或者条件函数返回 True
        if not self.__cond_func or self.__cond_func(self.__orig_func, *args, **kwargs):
            # 调用替代函数
            return self.__sub_func(self.__orig_func, *args, **kwargs)
        else:
            # 调用原始函数
            return self.__orig_func(*args, **kwargs)
```