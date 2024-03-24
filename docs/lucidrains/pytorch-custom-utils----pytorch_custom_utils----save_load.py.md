# `.\lucidrains\pytorch-custom-utils\pytorch_custom_utils\save_load.py`

```py
# 导入所需的模块
import pickle
from functools import wraps
from pathlib import Path
from packaging import version
import torch
from torch.nn import Module
from beartype import beartype
from beartype.typing import Optional

# 定义一个辅助函数，用于检查变量是否存在
def exists(v):
    return v is not None

# 装饰器函数，用于保存和加载模型
@beartype
def save_load(
    save_method_name = 'save',
    load_method_name = 'load',
    config_instance_var_name = '_config',
    init_and_load_classmethod_name = 'init_and_load',
    version: Optional[str] = None
):
    # 内部函数，用于实现保存和加载功能
    def _save_load(klass):
        # 断言被装饰的类是 torch.nn.Module 的子类
        assert issubclass(klass, Module), 'save_load should decorate a subclass of torch.nn.Module'

        # 保存原始的 __init__ 方法
        _orig_init = klass.__init__

        # 重写 __init__ 方法
        @wraps(_orig_init)
        def __init__(self, *args, **kwargs):
            # 序列化参数和关键字参数
            _config = pickle.dumps((args, kwargs))
            # 将序列化后的参数保存到实例变量中
            setattr(self, config_instance_var_name, _config)
            # 调用原始的 __init__ 方法
            _orig_init(self, *args, **kwargs)

        # 保存模型到文件
        def _save(self, path, overwrite = True):
            path = Path(path)
            assert overwrite or not path.exists()

            pkg = dict(
                model = self.state_dict(),
                config = getattr(self, config_instance_var_name),
                version = version,
            )

            torch.save(pkg, str(path))

        # 从文件加载模型
        def _load(self, path, strict = True):
            path = Path(path)
            assert path.exists()

            pkg = torch.load(str(path), map_location = 'cpu')

            if exists(version) and exists(pkg['version']) and version.parse(version) != version.parse(pkg['version']):
                self.print(f'loading saved model at version {pkg["version"]}, but current package version is {__version__}')

            self.load_state_dict(pkg['model'], strict = strict)

        # 从文件初始化并加载模型
        @classmethod
        def _init_and_load_from(cls, path, strict = True):
            path = Path(path)
            assert path.exists()
            pkg = torch.load(str(path), map_location = 'cpu')

            assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

            config = pickle.loads(pkg['config'])
            args, kwargs = config
            model = cls(*args, **kwargs)

            _load(model, path, strict = strict)
            return model

        # 设置装饰后的 __init__ 方法，以及保存、加载和初始化加载方法
        klass.__init__ = __init__
        setattr(klass, save_method_name, _save)
        setattr(klass, load_method_name, _load)
        setattr(klass, init_and_load_classmethod_name, _init_and_load_from)

        return klass

    return _save_load
```