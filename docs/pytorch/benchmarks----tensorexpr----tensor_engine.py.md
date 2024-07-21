# `.\pytorch\benchmarks\tensorexpr\tensor_engine.py`

```
# 定义全局变量 tensor_engine，并初始化为 None
tensor_engine = None

# 定义一个装饰器 unsupported，用于标记不支持的方法
def unsupported(func):
    # 定义装饰器的内部函数 wrapper，用于包装传入的函数 func
    def wrapper(self):
        return func(self)
    
    # 设置 wrapper 函数的属性 is_supported 为 False，表示不支持
    wrapper.is_supported = False
    return wrapper

# 定义函数 is_supported，用于检查方法是否被支持
def is_supported(method):
    # 检查方法是否具有属性 is_supported
    if hasattr(method, "is_supported"):
        return method.is_supported
    # 默认情况下认为方法是被支持的
    return True

# 定义函数 set_engine_mode，用于设置 tensor_engine 的模式
def set_engine_mode(mode):
    # 声明全局变量 tensor_engine
    global tensor_engine
    
    # 根据传入的 mode 参数选择不同的引擎
    if mode == "tf":
        # 如果 mode 是 "tf"，导入 tf_engine 模块，并实例化 TensorFlowEngine 类
        from . import tf_engine
        tensor_engine = tf_engine.TensorFlowEngine()
    elif mode == "pt":
        # 如果 mode 是 "pt"，导入 pt_engine 模块，并实例化 TorchTensorEngine 类
        from . import pt_engine
        tensor_engine = pt_engine.TorchTensorEngine()
    elif mode == "topi":
        # 如果 mode 是 "topi"，导入 topi_engine 模块，并实例化 TopiEngine 类
        from . import topi_engine
        tensor_engine = topi_engine.TopiEngine()
    elif mode == "relay":
        # 如果 mode 是 "relay"，导入 relay_engine 模块，并实例化 RelayEngine 类
        from . import relay_engine
        tensor_engine = relay_engine.RelayEngine()
    elif mode == "nnc":
        # 如果 mode 是 "nnc"，导入 nnc_engine 模块，并实例化 NncEngine 类
        from . import nnc_engine
        tensor_engine = nnc_engine.NncEngine()
    else:
        # 如果 mode 不属于上述任何一种情况，抛出 ValueError 异常
        raise ValueError(f"invalid tensor engine mode: {mode}")
    
    # 设置 tensor_engine 的 mode 属性为传入的 mode
    tensor_engine.mode = mode

# 定义函数 get_engine，用于获取当前的 tensor_engine
def get_engine():
    # 如果 tensor_engine 还未被设置，则抛出 ValueError 异常
    if tensor_engine is None:
        raise ValueError("use of get_engine, before calling set_engine_mode is illegal")
    # 返回当前的 tensor_engine 对象
    return tensor_engine
```