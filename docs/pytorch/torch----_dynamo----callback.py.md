# `.\pytorch\torch\_dynamo\callback.py`

```py
# mypy: allow-untyped-defs
# 定义一个编译回调处理器类
class CompilationCallbackHandler:
    # 初始化方法，创建空的开始和结束回调列表
    def __init__(self):
        self.start_callbacks = []  # 存储编译开始时的回调函数列表
        self.end_callbacks = []    # 存储编译结束时的回调函数列表

    # 注册编译开始时的回调函数
    def register_start_callback(self, callback):
        """
        Register a callback function to be called when the compilation starts.

        Args:
        - callback (callable): The callback function to register.
        """
        self.start_callbacks.append(callback)  # 将回调函数添加到开始回调列表
        return callback  # 返回回调函数本身

    # 注册编译结束时的回调函数
    def register_end_callback(self, callback):
        """
        Register a callback function to be called when the compilation ends.

        Args:
        - callback (callable): The callback function to register.
        """
        self.end_callbacks.append(callback)  # 将回调函数添加到结束回调列表
        return callback  # 返回回调函数本身

    # 移除已注册的编译开始时的回调函数
    def remove_start_callback(self, callback):
        """
        Remove a registered start callback function.

        Args:
        - callback (callable): The callback function to remove.
        """
        self.start_callbacks.remove(callback)  # 从开始回调列表中移除指定的回调函数

    # 移除已注册的编译结束时的回调函数
    def remove_end_callback(self, callback):
        """
        Remove a registered end callback function.

        Args:
        - callback (callable): The callback function to remove.
        """
        self.end_callbacks.remove(callback)  # 从结束回调列表中移除指定的回调函数

    # 执行所有已注册的编译开始时的回调函数
    def run_start_callbacks(self):
        """
        Execute all registered start callbacks.
        """
        for callback in self.start_callbacks:
            callback()  # 依次执行开始回调列表中的每个回调函数

    # 执行所有已注册的编译结束时的回调函数
    def run_end_callbacks(self):
        """
        Execute all registered end callbacks.
        """
        for callback in self.end_callbacks:
            callback()  # 依次执行结束回调列表中的每个回调函数

    # 清空所有已注册的回调函数
    def clear(self):
        """
        Clear all registered callbacks.
        """
        self.start_callbacks.clear()  # 清空开始回调列表
        self.end_callbacks.clear()    # 清空结束回调列表


# 创建一个编译回调处理器的实例
callback_handler = CompilationCallbackHandler()

# 装饰器：注册编译开始时的回调函数
def on_compile_start(callback):
    """
    Decorator to register a callback function for the start of the compilation.
    """
    callback_handler.register_start_callback(callback)  # 调用处理器实例的注册开始回调方法
    return callback  # 返回回调函数本身

# 装饰器：注册编译结束时的回调函数
def on_compile_end(callback):
    """
    Decorator to register a callback function for the end of the compilation.
    """
    callback_handler.register_end_callback(callback)  # 调用处理器实例的注册结束回调方法
    return callback  # 返回回调函数本身
```