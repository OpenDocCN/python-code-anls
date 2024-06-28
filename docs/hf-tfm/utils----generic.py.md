# `.\utils\generic.py`

```py
# 版权声明
# 版权声明和许可证信息

# 导入模块
import inspect  # 导入inspect模块，用于解析源文件
import tempfile  # 临时文件模块
from collections import OrderedDict, UserDict  # 导入OrderedDict和UserDict类
from collections.abc import MutableMapping  # 导入MutableMapping抽象基类
from contextlib import ExitStack, contextmanager  # 导入ExitStack和contextmanager上下文管理器
from dataclasses import fields, is_dataclass  # 导入fields和is_dataclass函数
from enum import Enum  # 导入枚举类Enum
from functools import partial  # 导入partial函数
from typing import Any, ContextManager, Iterable, List, Tuple  # 导入类型提示相关的模块和类

import numpy as np  # 导入numpy模块并重命名为np
from packaging import version  # 导入version类

from .import_utils import (  # 导入import_utils中的指定函数和类
    get_torch_version,
    is_flax_available,
    is_mlx_available,
    is_tf_available,
    is_torch_available,
    is_torch_fx_proxy,
)

# 如果Flax可用，则导入jax.numpy as jnp
if is_flax_available():
    import jax.numpy as jnp

# 自定义缓存属性装饰器
class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.

    From tensorflow_datasets

    Built-in in functools from Python 3.8.
    """
    def __get__(self, obj, objtype=None):
        # 获取属性的值，并对其进行缓存
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached


# 从distutils.util模块中引入strtobool函数，用于将表示真假的字符串转换为1或0
def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return 1
    if val in {"n", "no", "f", "false", "off", "0"}:
        return 0
    raise ValueError(f"invalid truth value {val!r}")


# 从对象的repr中推断出其所属框架的函数
def infer_framework_from_repr(x):
    """
    Tries to guess the framework of an object `x` from its repr (brittle but will help in `is_tensor` to try the
    frameworks in a smart order, without the need to import the frameworks).
    """
    representation = str(type(x))
    if representation.startswith("<class 'torch."):
        return "pt"
    elif representation.startswith("<class 'tensorflow."):
        return "tf"
    elif representation.startswith("<class 'jax"):
        return "jax"
    elif representation.startswith("<class 'numpy."):
        return "np"
    # 如果表示字符串以 "<class 'mlx." 开头，则返回字符串 "mlx"
    elif representation.startswith("<class 'mlx."):
        return "mlx"
# 返回一个按顺序排列的字典，包含了根据推断的优先框架来测试函数，优先顺序为我们从repr中能猜测到的框架首先，
# 然后是Numpy，最后是其他框架。
def _get_frameworks_and_test_func(x):
    framework_to_test = {
        "pt": is_torch_tensor,
        "tf": is_tf_tensor,
        "jax": is_jax_tensor,
        "np": is_numpy_array,
        "mlx": is_mlx_array,
    }
    preferred_framework = infer_framework_from_repr(x)
    # 首先测试推断的优先框架，然后是Numpy，最后是其他框架。
    frameworks = [] if preferred_framework is None else [preferred_framework]
    if preferred_framework != "np":
        frameworks.append("np")
    frameworks.extend([f for f in framework_to_test if f not in [preferred_framework, "np"]])
    return {f: framework_to_test[f] for f in frameworks}


# 测试是否 `x` 是 `torch.Tensor`, `tf.Tensor`, `jaxlib.xla_extension.DeviceArray`, `np.ndarray` 或 `mlx.array`，
# 按照 `infer_framework_from_repr` 定义的顺序进行测试。
def is_tensor(x):
    framework_to_test_func = _get_frameworks_and_test_func(x)
    for test_func in framework_to_test_func.values():
        if test_func(x):
            return True

    # 检查是否是跟踪器
    if is_torch_fx_proxy(x):
        return True

    if is_flax_available():
        from jax.core import Tracer

        if isinstance(x, Tracer):
            return True

    return False


# 测试是否 `x` 是一个 numpy 数组。
def is_numpy_array(x):
    return _is_numpy(x)


# 判断 `x` 是否是 torch 的 tensor。
def is_torch_tensor(x):
    return False if not is_torch_available() else _is_torch(x)


# 判断 `x` 是否是 torch 的 device。
def is_torch_device(x):
    return False if not is_torch_available() else _is_torch_device(x)


# 判断 `x` 是否是 torch 的 dtype。
def is_torch_dtype(x):
    return False if not is_torch_available() else _is_torch_dtype(x)


# 判断 `x` 是否是 tensorflow 的 tensor。
def is_tf_tensor(x):
    return False if not is_tf_available() else _is_tensorflow(x)
    # 检查 TensorFlow 模块是否具有 `is_symbolic_tensor` 属性，该属性从 TensorFlow 2.14 开始可用
    if hasattr(tf, "is_symbolic_tensor"):
        # 如果有 `is_symbolic_tensor` 方法，则调用该方法来检查 x 是否为符号张量
        return tf.is_symbolic_tensor(x)
    # 如果 TensorFlow 模块没有 `is_symbolic_tensor` 方法，则直接比较 x 的类型是否为 tf.Tensor 类型
    return type(x) == tf.Tensor
# 测试 `x` 是否为 TensorFlow 符号张量（即非即时执行模式）。即使没有安装 TensorFlow 也可以安全调用。
def is_tf_symbolic_tensor(x):
    return False if not is_tf_available() else _is_tf_symbolic_tensor(x)


# 检查 `x` 是否为 Jax 数组。
def _is_jax(x):
    import jax.numpy as jnp  # noqa: F811
    return isinstance(x, jnp.ndarray)


# 测试 `x` 是否为 Jax 张量。即使没有安装 Jax 也可以安全调用。
def is_jax_tensor(x):
    return False if not is_flax_available() else _is_jax(x)


# 检查 `x` 是否为 MLX 数组。
def _is_mlx(x):
    import mlx.core as mx
    return isinstance(x, mx.array)


# 测试 `x` 是否为 MLX 数组。即使没有安装 MLX 也可以安全调用。
def is_mlx_array(x):
    return False if not is_mlx_available() else _is_mlx(x)


# 将 TensorFlow 张量、PyTorch 张量、Numpy 数组或 Python 列表转换为 Python 列表。
def to_py_obj(obj):
    framework_to_py_obj = {
        "pt": lambda obj: obj.detach().cpu().tolist(),
        "tf": lambda obj: obj.numpy().tolist(),
        "jax": lambda obj: np.asarray(obj).tolist(),
        "np": lambda obj: obj.tolist(),
    }

    if isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]

    # 根据测试函数智能确定使用哪个框架的转换函数
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_py_obj[framework](obj)

    # tolist 也适用于 0 维的 np 数组
    if isinstance(obj, np.number):
        return obj.tolist()
    else:
        return obj


# 将 TensorFlow 张量、PyTorch 张量、Numpy 数组或 Python 列表转换为 Numpy 数组。
def to_numpy(obj):
    framework_to_numpy = {
        "pt": lambda obj: obj.detach().cpu().numpy(),
        "tf": lambda obj: obj.numpy(),
        "jax": lambda obj: np.asarray(obj),
        "np": lambda obj: obj,
    }

    if isinstance(obj, (dict, UserDict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return np.array(obj)

    # 根据测试函数智能确定使用哪个框架的转换函数
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_numpy[framework](obj)

    return obj


# 表示模型输出的基类，继承自 OrderedDict，作为数据类。具有一个 `__getitem__` 方法，允许按整数或切片（如元组）或字符串（如字典）进行索引，忽略 `None` 属性。否则表现类似于普通的 Python 字典。
class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.
    
    <Tip warning={true}>
    """
    """
    # 注册子类作为 pytree 节点
    def __init_subclass__(cls) -> None:
        """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output `ModelOutput` subclasses.
        """
        # 如果 PyTorch 可用且版本大于等于 2.2，则注册 pytree 节点
        if is_torch_available():
            if version.parse(get_torch_version()) >= version.parse("2.2"):
                _torch_pytree.register_pytree_node(
                    cls,
                    _model_output_flatten,
                    partial(_model_output_unflatten, output_type=cls),
                    serialized_type_name=f"{cls.__module__}.{cls.__name__}",
                )
            else:
                # 对于低版本的 PyTorch，使用旧的注册方式
                _torch_pytree._register_pytree_node(
                    cls,
                    _model_output_flatten,
                    partial(_model_output_unflatten, output_type=cls),
                )

    # 初始化函数，检查是否为 ModelOutput 的子类，并且必须使用 @dataclass 装饰器
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 子类必须使用 @dataclass 装饰器，这个检查在 __init__ 中进行，因为 @dataclass 装饰器
        # 在 __init_subclass__ 之后才生效
        # 如果当前类不是 ModelOutput 本身，即当前类是其子类
        is_modeloutput_subclass = self.__class__ != ModelOutput

        # 如果当前类是 ModelOutput 的子类，并且没有使用 @dataclass 装饰器，则抛出 TypeError
        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclasss."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )
    def __post_init__(self):
        """初始化后检查ModelOutput数据类。

        仅在使用@dataclass装饰器时发生。
        """
        # 获取数据类的所有字段
        class_fields = fields(self)

        # 安全性和一致性检查
        if not len(class_fields):
            # 如果没有字段，则引发值错误异常
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            # 如果有超过一个必需字段，则引发值错误异常
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        # 获取第一个字段的值
        first_field = getattr(self, class_fields[0].name)
        # 检查其它字段是否都为None
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                # 如果第一个字段是字典，则遍历字典项
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    # 尝试迭代第一个字段
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # 如果第一个字段是迭代器且是(key, value)形式的迭代器
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # 如果不是(key, value)形式的迭代器，将其设置为属性
                            self[class_fields[0].name] = first_field
                        else:
                            # 如果是混合迭代器，引发值错误异常
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    # 设置属性为(key, value)对
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                # 如果第一个字段不为空，则将其设置为属性
                self[class_fields[0].name] = first_field
        else:
            # 如果存在非None的字段，则将其设置为属性
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        """阻止对ModelOutput实例使用``__delitem__``方法。"""
        # 抛出异常，不允许使用``__delitem__``方法
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        """阻止对ModelOutput实例使用``setdefault``方法。"""
        # 抛出异常，不允许使用``setdefault``方法
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        """阻止对ModelOutput实例使用``pop``方法。"""
        # 抛出异常，不允许使用``pop``方法
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")
    def update(self, *args, **kwargs):
        # 抛出异常，阻止在该类实例上使用 `update` 方法
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            # 将内部数据转换为字典，然后返回键 `k` 对应的值
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            # 调用 `to_tuple()` 方法返回的元组，并使用 `k` 作为索引获取元组中的值
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # 避免递归错误，不调用 `self.__setitem__` 方法
            super().__setitem__(name, value)
        # 设置对象的属性 `name` 为 `value`
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # 调用父类的 `__setitem__` 方法设置键 `key` 对应的值 `value`
        super().__setitem__(key, value)
        # 避免递归错误，不调用 `self.__setattr__` 方法
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            # 如果对象不是数据类，则调用父类的 `__reduce__` 方法
            return super().__reduce__()
        # 否则，获取对象所有非 `None` 属性或键的元组，并返回
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        # 返回包含所有非 `None` 属性或键的元组
        return tuple(self[k] for k in self.keys())
# 检查是否安装了 Torch
if is_torch_available():
    # 导入 Torch 的私有模块 _pytree
    import torch.utils._pytree as _torch_pytree

    # 将模型输出展平化的函数，返回值和上下文信息
    def _model_output_flatten(output: ModelOutput) -> Tuple[List[Any], "_torch_pytree.Context"]:
        return list(output.values()), list(output.keys())

    # 将模型输出还原为原始结构的函数
    def _model_output_unflatten(
        values: Iterable[Any],
        context: "_torch_pytree.Context",
        output_type=None,
    ) -> ModelOutput:
        return output_type(**dict(zip(context, values)))

    # 如果 Torch 的版本大于等于 2.2，则注册 PyTree 节点
    if version.parse(get_torch_version()) >= version.parse("2.2"):
        _torch_pytree.register_pytree_node(
            ModelOutput,
            _model_output_flatten,
            partial(_model_output_unflatten, output_type=ModelOutput),
            serialized_type_name=f"{ModelOutput.__module__}.{ModelOutput.__name__}",
        )
    else:
        # 否则使用旧的注册方式
        _torch_pytree._register_pytree_node(
            ModelOutput,
            _model_output_flatten,
            partial(_model_output_unflatten, output_type=ModelOutput),
        )


# 定义一个显式枚举类 ExplicitEnum，继承自 str 和 Enum
class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    # 当枚举值缺失时，提供更明确的错误消息
    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


# 定义一个填充策略枚举类 PaddingStrategy，继承自 ExplicitEnum
class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


# 定义一个张量类型枚举类 TensorType，继承自 ExplicitEnum
class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"
    MLX = "mlx"


# 定义一个上下文管理器类 ContextManagers
class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    # 初始化方法，接受一个上下文管理器列表作为参数
    def __init__(self, context_managers: List[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()  # 使用 contextlib.ExitStack 创建堆栈

    # 进入上下文管理器的方法
    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    # 退出上下文管理器的方法
    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


# 定义一个函数，检查给定的模型类是否能返回损失值
def can_return_loss(model_class):
    """
    Check if a given model can return loss.

    Args:
        model_class (`type`): The class of the model.
    """
    framework = infer_framework(model_class)  # 推断模型所属的框架
    if framework == "tf":
        signature = inspect.signature(model_class.call)  # TensorFlow 模型
    elif framework == "pt":
        signature = inspect.signature(model_class.forward)  # PyTorch 模型
    else:
        signature = inspect.signature(model_class.__call__)  # Flax 模型
    # 遍历函数签名的参数列表
    for p in signature.parameters:
        # 检查当前参数是否为 "return_loss"，且其默认值为 True
        if p == "return_loss" and signature.parameters[p].default is True:
            # 如果满足条件，返回 True
            return True
    
    # 如果未找到符合条件的参数，返回 False
    return False
# 查找给定模型使用的标签参数列表
def find_labels(model_class):
    model_name = model_class.__name__  # 获取模型类的名称
    framework = infer_framework(model_class)  # 推断模型使用的框架
    if framework == "tf":
        signature = inspect.signature(model_class.call)  # 获取TensorFlow模型的调用签名
    elif framework == "pt":
        signature = inspect.signature(model_class.forward)  # 获取PyTorch模型的前向方法签名
    else:
        signature = inspect.signature(model_class.__call__)  # 获取Flax模型的调用方法签名

    if "QuestionAnswering" in model_name:  # 如果模型名称中包含"QuestionAnswering"
        return [p for p in signature.parameters if "label" in p or p in ("start_positions", "end_positions")]  # 返回标签相关的参数列表
    else:
        return [p for p in signature.parameters if "label" in p]  # 返回标签相关的参数列表


# 将嵌套字典展开为单层字典
def flatten_dict(d: MutableMapping, parent_key: str = "", delimiter: str = "."):
    def _flatten_dict(d, parent_key="", delimiter="."):
        for k, v in d.items():
            key = str(parent_key) + delimiter + str(k) if parent_key else k
            if v and isinstance(v, MutableMapping):
                yield from flatten_dict(v, key, delimiter=delimiter).items()  # 递归展开嵌套字典
            else:
                yield key, v  # 直接添加键值对到展开的字典中

    return dict(_flatten_dict(d, parent_key, delimiter))


# 提供工作目录或临时目录的上下文管理器
@contextmanager
def working_or_temp_dir(working_dir, use_temp_dir: bool = False):
    if use_temp_dir:
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir  # 使用临时目录作为上下文环境
    else:
        yield working_dir  # 使用指定的工作目录作为上下文环境


# 框架无关的数组转置函数，支持numpy、torch、tensorflow和jax的数组
def transpose(array, axes=None):
    if is_numpy_array(array):  # 如果是numpy数组
        return np.transpose(array, axes=axes)  # 使用numpy的转置函数
    elif is_torch_tensor(array):  # 如果是torch张量
        return array.T if axes is None else array.permute(*axes)  # 使用torch的转置或者按指定轴排列
    elif is_tf_tensor(array):  # 如果是tensorflow张量
        import tensorflow as tf
        return tf.transpose(array, perm=axes)  # 使用tensorflow的转置函数
    elif is_jax_tensor(array):  # 如果是jax张量
        return jnp.transpose(array, axes=axes)  # 使用jax的转置函数
    else:
        raise ValueError(f"Type not supported for transpose: {type(array)}.")  # 抛出类型不支持的异常


# 框架无关的数组重塑函数，支持numpy、torch、tensorflow和jax的数组
def reshape(array, newshape):
    if is_numpy_array(array):  # 如果是numpy数组
        return np.reshape(array, newshape)  # 使用numpy的重塑函数
    elif is_torch_tensor(array):  # 如果是torch张量
        return array.reshape(*newshape)  # 使用torch的重塑方法
    elif is_tf_tensor(array):  # 如果是tensorflow张量
        import tensorflow as tf
        return tf.reshape(array, newshape)  # 使用tensorflow的重塑函数
    elif is_jax_tensor(array):  # 如果是jax张量
        return jnp.reshape(array, newshape)  # 使用jax的重塑函数
    else:
        raise ValueError(f"Type not supported for reshape: {type(array)}.")  # 抛出类型不支持的异常


# 框架无关的数组挤压函数，支持numpy、torch、tensorflow和jax的数组
def squeeze(array, axis=None):
    if is_numpy_array(array):  # 如果是numpy数组
        return np.squeeze(array, axis=axis)  # 使用numpy的挤压函数
    # 如果输入的数组是 PyTorch 张量，则进行挤压操作，去除维度为1的轴
    elif is_torch_tensor(array):
        return array.squeeze() if axis is None else array.squeeze(dim=axis)
    # 如果输入的数组是 TensorFlow 张量，则导入 TensorFlow 库并进行挤压操作，去除指定的轴
    elif is_tf_tensor(array):
        import tensorflow as tf

        return tf.squeeze(array, axis=axis)
    # 如果输入的数组是 JAX 张量，则进行挤压操作，去除指定的轴
    elif is_jax_tensor(array):
        return jnp.squeeze(array, axis=axis)
    # 如果输入的数组类型不被支持，则抛出异常并显示错误信息
    else:
        raise ValueError(f"Type not supported for squeeze: {type(array)}.")
# 定义一个函数，用于在不同深度学习框架下扩展张量的维度
def expand_dims(array, axis):
    """
    Framework-agnostic version of `numpy.expand_dims` that will work on torch/TensorFlow/Jax tensors as well as NumPy
    arrays.
    """
    # 如果输入数组是 NumPy 数组，则使用 NumPy 的 `expand_dims` 函数
    if is_numpy_array(array):
        return np.expand_dims(array, axis)
    # 如果输入数组是 PyTorch 张量，则使用 PyTorch 的 `unsqueeze` 函数
    elif is_torch_tensor(array):
        return array.unsqueeze(dim=axis)
    # 如果输入数组是 TensorFlow 张量，则使用 TensorFlow 的 `expand_dims` 函数
    elif is_tf_tensor(array):
        import tensorflow as tf
        
        return tf.expand_dims(array, axis=axis)
    # 如果输入数组是 Jax 张量，则使用 Jax 的 `expand_dims` 函数
    elif is_jax_tensor(array):
        return jnp.expand_dims(array, axis=axis)
    else:
        # 如果输入数组类型不被支持，则抛出 ValueError 异常
        raise ValueError(f"Type not supported for expand_dims: {type(array)}.")


# 定义一个函数，用于计算不同深度学习框架下张量的大小
def tensor_size(array):
    """
    Framework-agnostic version of `numpy.size` that will work on torch/TensorFlow/Jax tensors as well as NumPy arrays.
    """
    # 如果输入数组是 NumPy 数组，则返回数组的大小
    if is_numpy_array(array):
        return np.size(array)
    # 如果输入数组是 PyTorch 张量，则返回张量的元素个数
    elif is_torch_tensor(array):
        return array.numel()
    # 如果输入数组是 TensorFlow 张量，则返回张量的大小
    elif is_tf_tensor(array):
        import tensorflow as tf
        
        return tf.size(array)
    # 如果输入数组是 Jax 张量，则返回张量的大小
    elif is_jax_tensor(array):
        return array.size
    else:
        # 如果输入数组类型不被支持，则抛出 ValueError 异常
        raise ValueError(f"Type not supported for tensor_size: {type(array)}.")


# 定义一个函数，将 repo_id 的信息添加到给定的自动映射 auto_map 中
def add_model_info_to_auto_map(auto_map, repo_id):
    """
    Adds the information of the repo_id to a given auto map.
    """
    # 遍历 auto_map 的键值对
    for key, value in auto_map.items():
        # 如果值是列表或元组，则将每个元素前添加 repo_id，避免重复添加
        if isinstance(value, (tuple, list)):
            auto_map[key] = [f"{repo_id}--{v}" if (v is not None and "--" not in v) else v for v in value]
        # 如果值不是 None 且不包含 "--"，则在值前添加 repo_id
        elif value is not None and "--" not in value:
            auto_map[key] = f"{repo_id}--{value}"

    # 返回更新后的 auto_map
    return auto_map


# 定义一个函数，推断给定模型类的深度学习框架
def infer_framework(model_class):
    """
    Infers the framework of a given model without using isinstance(), because we cannot guarantee that the relevant
    classes are imported or available.
    """
    # 遍历模型类的方法解析顺序（Method Resolution Order）
    for base_class in inspect.getmro(model_class):
        module = base_class.__module__
        name = base_class.__name__
        # 如果基类模块名以 "tensorflow" 或 "keras" 开头，或者基类名为 "TFPreTrainedModel"，则推断为 TensorFlow 框架
        if module.startswith("tensorflow") or module.startswith("keras") or name == "TFPreTrainedModel":
            return "tf"
        # 如果基类模块名以 "torch" 开头，或者基类名为 "PreTrainedModel"，则推断为 PyTorch 框架
        elif module.startswith("torch") or name == "PreTrainedModel":
            return "pt"
        # 如果基类模块名以 "flax" 或 "jax" 开头，或者基类名为 "FlaxPreTrainedModel"，则推断为 Jax/Flax 框架
        elif module.startswith("flax") or module.startswith("jax") or name == "FlaxPreTrainedModel":
            return "flax"
    else:
        # 如果无法推断出框架，则抛出 TypeError 异常
        raise TypeError(f"Could not infer framework from class {model_class}.")
```