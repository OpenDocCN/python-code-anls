# `stable-diffusion-webui\modules\sd_disable_initialization.py`

```py
# 导入所需的模块
import ldm.modules.encoders.modules
import open_clip
import torch
import transformers.utils.hub

# 从自定义模块中导入 shared 模块
from modules import shared

# 定义一个用于替换属性值的辅助类
class ReplaceHelper:
    def __init__(self):
        self.replaced = []

    # 替换对象的属性值
    def replace(self, obj, field, func):
        # 获取对象原始属性值
        original = getattr(obj, field, None)
        if original is None:
            return None

        # 将原始对象、属性名和属性值添加到替换列表中
        self.replaced.append((obj, field, original))
        # 设置对象的属性值为新值
        setattr(obj, field, func)

        return original

    # 恢复对象的属性值
    def restore(self):
        for obj, field, original in self.replaced:
            setattr(obj, field, original)

        self.replaced.clear()

# 继承自 ReplaceHelper 类，用于禁用初始化操作
class DisableInitialization(ReplaceHelper):
    """
    当此类的对象进入 `with` 块时，它会：
    - 阻止 torch 的层初始化函数工作
    - 更改 CLIP 和 OpenCLIP 不下载模型权重
    - 更改 CLIP 不发送请求检查是否有新版本的文件

    当离开块时，会将一切恢复到之前的状态。

    使用方法：
    ```
    with DisableInitialization():
        do_things()
    ```py
    """

    def __init__(self, disable_clip=True):
        super().__init__()
        self.disable_clip = disable_clip

    # 替换对象的属性值
    def replace(self, obj, field, func):
        original = getattr(obj, field, None)
        if original is None:
            return None

        self.replaced.append((obj, field, original))
        setattr(obj, field, func)

        return original

    # 退出上下文管理器时执行恢复操作
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()

# 继承自 ReplaceHelper 类，用于在元设备上初始化参数
class InitializeOnMeta(ReplaceHelper):
    """
    上下文管理器，使所有线性/卷积/多头注意力层的参数都分配在元设备上，
    这样这些参数将没有值并且不占用内存。model.to() 将被破坏，需要使用下面的 LoadStateDictOnMeta 来修复加载状态字典时的参数。

    用法:
    ```

    ```py
    """
    # 使用 sd_disable_initialization.InitializeOnMeta() 上下文管理器初始化模型
    with sd_disable_initialization.InitializeOnMeta():
        # 从配置文件中实例化模型
        sd_model = instantiate_from_config(sd_config.model)
    ```
    """

    # 进入上下文管理器时执行的操作
    def __enter__(self):
        # 如果命令行选项中禁用了模型加载 RAM 优化，则直接返回
        if shared.cmd_opts.disable_model_loading_ram_optimization:
            return

        # 定义一个函数，用于设置设备为 "meta"
        def set_device(x):
            x["device"] = "meta"
            return x

        # 替换 torch.nn.Linear 类的 __init__ 方法，将设备设置为 "meta"
        linear_init = self.replace(torch.nn.Linear, '__init__', lambda *args, **kwargs: linear_init(*args, **set_device(kwargs)))
        # 替换 torch.nn.Conv2d 类的 __init__ 方法，将设备设置为 "meta"
        conv2d_init = self.replace(torch.nn.Conv2d, '__init__', lambda *args, **kwargs: conv2d_init(*args, **set_device(kwargs)))
        # 替换 torch.nn.MultiheadAttention 类的 __init__ 方法，将设备设置为 "meta"
        mha_init = self.replace(torch.nn.MultiheadAttention, '__init__', lambda *args, **kwargs: mha_init(*args, **set_device(kwargs)))
        # 替换 torch.nn.Module 类的 to 方法，将其设为 None
        self.replace(torch.nn.Module, 'to', lambda *args, **kwargs: None)

    # 退出上下文管理器时执行的操作
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始状态
        self.restore()
# 定义一个类 LoadStateDictOnMeta，继承自 ReplaceHelper
class LoadStateDictOnMeta(ReplaceHelper):
    """
    Context manager that allows to read parameters from state_dict into a model that has some of its parameters in the meta device.
    As those parameters are read from state_dict, they will be deleted from it, so by the end state_dict will be mostly empty, to save memory.
    Meant to be used together with InitializeOnMeta above.

    Usage:
    ```py
    with sd_disable_initialization.LoadStateDictOnMeta(state_dict):
        model.load_state_dict(state_dict, strict=False)
    ```
    """

    # 初始化方法，接受 state_dict、device 和 weight_dtype_conversion 作为参数
    def __init__(self, state_dict, device, weight_dtype_conversion=None):
        super().__init__()
        # 将参数赋值给实例变量
        self.state_dict = state_dict
        self.device = device
        self.weight_dtype_conversion = weight_dtype_conversion or {}
        self.default_dtype = self.weight_dtype_conversion.get('')

    # 获取权重数据类型的方法，根据键值获取对应的权重数据类型
    def get_weight_dtype(self, key):
        key_first_term, _ = key.split('.', 1)
        return self.weight_dtype_conversion.get(key_first_term, self.default_dtype)

    # 退出上下文管理器时调用的方法，恢复状态
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()
```